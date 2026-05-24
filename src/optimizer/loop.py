"""
Central optimization engine.

Implements a Greedy Accept/Reject loop augmented with:
  - Schema-aware F1 scoring via Scorer (independent of this module)
  - Resumability: warm-starts from SQLite if a prior run exists
  - Stall detection: escalates mutation aggressiveness after N iterations
  - Rejection memory: Mutator sees previously tried-and-failed prompts
  - Budget enforcement: stops on max_iterations OR max_cost_dollars (if > 0)
  - Daily-limit detection: graceful shutdown on OpenRouter quota exhaustion
    (or Ollama unavailability — see base_agent.py)
  - REPORT.md auto-generation after test evaluation
  - Extraction debug logging: prints a sample prediction on the first
    iteration so you can immediately see what the model outputs vs gold

OCR caching
-----------
The StateManager is passed into the ExtractBenchLoader so extracted PDF
text is persisted in SQLite. On any subsequent run (or after a resume)
the loader skips re-extraction for files whose bytes have not changed,
saving time and API quota.

Judge robustness
----------------
The LLM judge for string_semantic / array_llm parses the model response
with a regex float extractor rather than a bare float() call.  Free-tier
models frequently return verbose text ("I would rate this 0.8 out of 1.0")
instead of a bare float, which causes float() to raise and scores every
stochastic field as 0.0.  The regex extractor finds the first valid float
token in any response format.

When the judge fails entirely, stochastic fields fall back to a
word-overlap F1 score (token-level precision/recall) rather than 0.0 or
binary exact-match.  This gives partial credit for semantically similar
strings and keeps the optimization signal meaningful even without a
working judge.
"""

from __future__ import annotations

import json
import os
import re
import textwrap
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.agents.base_agent import DailyLimitError
from src.agents.critic_mutator import Critic, Extractor, Mutator
from src.core.config_parser import load_config
from src.core.state_manager import StateManager
from src.data.loader import ExtractBenchLoader
from src.data.splitter import deterministic_split
from src.evaluation.scorer import Scorer
from src.optimizer.diff_viewer import DiffViewer


# ---------------------------------------------------------------------------
# Judge helpers
# ---------------------------------------------------------------------------

def _parse_judge_float(text: str) -> Optional[float]:
    """
    Extract a float score from a (possibly verbose) LLM judge response.

    Handles all common free-model response styles:
      "0.8"                           → 0.8
      "0.80"                          → 0.8
      "I would rate this 0.75/1.0"    → 0.75
      "Score: 7/10"                   → 0.7
      "Similarity: 85%"               → 0.85
      "1.0 - identical"               → 1.0

    Returns None if no valid float can be extracted.
    """
    text = text.strip()

    # 1. X/100 pattern — must come before X/10 to avoid "85/100" matching as "85/10"
    match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*100\b', text)
    if match:
        return min(1.0, float(match.group(1)) / 100.0)

    # 2. X/10 pattern
    match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*10\b', text)
    if match:
        return min(1.0, float(match.group(1)) / 10.0)

    # 3. Percentage (e.g. "85%")
    match = re.search(r'(\d+(?:\.\d+)?)\s*%', text)
    if match:
        return min(1.0, float(match.group(1)) / 100.0)

    # 4. Direct float/int in [0, 1]
    match = re.search(r'\b(0(?:\.\d+)?|1(?:\.0*)?)\b', text)
    if match:
        return float(match.group())

    # 5. Any first number in the response (last resort)
    match = re.search(r'(\d+(?:\.\d+)?)', text)
    if match:
        val = float(match.group(1))
        return min(1.0, val / 10.0) if val > 1.0 else val

    return None


def _word_overlap_f1(pred: str, gold: str) -> float:
    """
    Token-level F1 score as a fallback for string_semantic when the judge fails.

    Gives partial credit for semantically similar strings and preserves the
    optimization signal across iterations even on free-tier models that cannot
    produce reliable judge scores.
    """
    pred_tokens = set(str(pred).lower().split())
    gold_tokens = set(str(gold).lower().split())
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    intersection = pred_tokens & gold_tokens
    precision = len(intersection) / len(pred_tokens) if pred_tokens else 0.0
    recall    = len(intersection) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

class OptimizerLoop:
    """Budget-enforced greedy prompt optimization loop."""

    STALL_THRESHOLD = 3  # Iterations without improvement before escalating mutation

    def __init__(self, config_path: str = "config/base_config.yaml"):
        self.config = load_config(config_path)
        self.state  = StateManager()
        self.diff_viewer = DiffViewer()

        cfg = self.config

        self.extractor = Extractor(cfg.models.extractor, self.state)
        self.critic    = Critic(cfg.models.critic,       self.state)
        self.mutator   = Mutator(cfg.models.mutator,     self.state)

        # Build a robust LLM judge callable for stochastic metrics.
        judge_client = self.critic.client
        judge_model  = cfg.models.critic

        def llm_judge(pred, gold, metric: str) -> float:
            return self._call_judge(pred, gold, metric, judge_client, judge_model)

        self.scorer = Scorer(state_manager=self.state, judge_callable=llm_judge)

        # ---- Dataset loading ----
        # Pass the StateManager so the loader can use the OCR cache.
        dataset_base   = os.path.join(cfg.dataset.base_path, "dataset")
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        gemini_key     = os.environ.get("GEMINI_API_KEY")

        loader = ExtractBenchLoader(
            base_path=dataset_base,
            schema_name=cfg.dataset.name,
            vision_model=cfg.vision_model,
            openrouter_key=openrouter_key,
            gemini_key=gemini_key,
            state_manager=self.state,      # ← enables OCR caching
        )
        print(f"\n📂 Loading dataset: {cfg.dataset.name}")
        all_docs = loader.load_all_document_pairs()

        if not all_docs:
            raise RuntimeError(
                "No document pairs could be loaded.\n"
                "Check:\n"
                "  1. data/extract-bench/dataset/<schema_name>/pdf+gold/ exists\n"
                "  2. OPENROUTER_API_KEY is set\n"
                "  3. For scanned PDFs: set GEMINI_API_KEY for best OCR quality\n"
                "  4. The schema name in config matches the folder name exactly"
            )

        self.train_docs, self.val_docs, self.test_docs = deterministic_split(
            all_docs,
            seed=cfg.dataset.split_seed,
            train_ratio=cfg.dataset.train_ratio,
            val_ratio=cfg.dataset.val_ratio,
        )

        if not self.val_docs:
            print(
                "\n  ⚠️  Val set is empty after split. "
                "Falling back to using all loaded docs for validation."
            )
            self.val_docs  = all_docs
            self.test_docs = all_docs

        print(
            f"\n  Split (seed={cfg.dataset.split_seed}): "
            f"{len(self.train_docs)} train | "
            f"{len(self.val_docs)} val | "
            f"{len(self.test_docs)} test"
        )

    # ------------------------------------------------------------------
    # Budget
    # ------------------------------------------------------------------

    def _within_budget(self) -> bool:
        dollar_limit = self.config.budget.max_cost_dollars
        if dollar_limit <= 0.0:
            return True
        spent = self.state.get_total_cost()
        if spent >= dollar_limit:
            print(f"💸 Budget exhausted (${spent:.4f} / ${dollar_limit:.2f}). Stopping.")
            return False
        return True

    # ------------------------------------------------------------------
    # LLM judge — robust float parsing + word-overlap fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _call_judge(pred: object, gold: object, metric: str, client, model: str) -> float:
        """
        Score a (pred, gold) pair for a stochastic metric using an LLM judge.

        Robust to verbose model responses: uses regex float extraction rather
        than a bare float() call.  Falls back to word-overlap F1 if the model
        cannot produce a parseable response.
        """
        try:
            if metric == "string_semantic":
                prompt = (
                    "Score the semantic similarity of these two strings.\n"
                    "Return ONLY a single decimal number between 0.0 and 1.0.\n"
                    "0.0 = completely different, 1.0 = identical meaning.\n\n"
                    f"String A: {pred}\n"
                    f"String B: {gold}\n\n"
                    "Score:"
                )
            else:  # array_llm
                prompt = (
                    "Score the semantic equivalence of these two lists.\n"
                    "Return ONLY a single decimal number between 0.0 and 1.0.\n"
                    "0.0 = nothing in common, 1.0 = semantically identical.\n\n"
                    f"List A: {json.dumps(pred, default=str)}\n"
                    f"List B: {json.dumps(gold, default=str)}\n\n"
                    "Score:"
                )

            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=20,
            )
            raw = (response.choices[0].message.content or "").strip()
            score = _parse_judge_float(raw)
            if score is not None:
                return max(0.0, min(1.0, score))

            return _word_overlap_f1(str(pred), str(gold))

        except Exception:
            return _word_overlap_f1(str(pred), str(gold))

    # ------------------------------------------------------------------
    # Corpus scoring with extraction debug
    # ------------------------------------------------------------------

    def _evaluate_corpus(
        self,
        docs: List[Dict],
        prompt: str,
        debug_first: bool = False,
    ) -> Tuple[float, Dict]:
        """
        Extract + score every document in docs with the current prompt.

        Returns (mean_f1, info_dict).
        """
        total_f1  = 0.0
        breakdown: Dict[str, Dict] = {}
        failed:    List[Dict]       = []

        for i, doc in enumerate(docs):
            prediction = self.extractor.extract(doc["text"], prompt, doc["schema"])

            if debug_first and i == 0:
                print("\n  ── DEBUG: Raw extractor output (first 600 chars) ──")
                print(prediction[:600])
                print("  ── DEBUG: Gold JSON (first 400 chars) ──")
                print(doc["gold_json"][:400])
                print("  ──────────────────────────────────────────────────\n")

            f1, doc_breakdown = self.scorer.score_document(
                pred_json=prediction,
                gold_json=doc["gold_json"],
                schema_str=doc["schema"],
            )
            total_f1 += f1
            breakdown[doc["id"]] = {**doc_breakdown, "f1": f1}

            if f1 < 1.0:
                failed.append({
                    "doc":   doc["text"],
                    "pred":  prediction,
                    "gold":  doc["gold_json"],
                    "score": f1,
                    "id":    doc["id"],
                })

        mean_f1 = total_f1 / max(len(docs), 1)
        return mean_f1, {"docs": breakdown, "failed": failed}

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        cfg = self.config
        print(
            f"\n🚀 Optimizer starting  "
            f"max_iterations={cfg.budget.max_iterations}  "
            f"dataset={cfg.dataset.name}"
        )

        # ---- Resumability ----
        prior_best       = self.state.get_best_state()
        last_done        = self.state.get_last_completed_iteration()
        rejected_history = self.state.get_rejected_prompts()

        if prior_best and last_done >= 0:
            best_prompt     = prior_best["prompt"]
            best_score      = prior_best["val_score"]
            start_iteration = last_done + 1
            current_prompt  = best_prompt
            print(f"♻️  Resuming from iteration {start_iteration} (best val F1: {best_score:.4f})")
        else:
            best_prompt     = cfg.seed_prompt
            best_score      = -1.0
            start_iteration = 0
            current_prompt  = cfg.seed_prompt
            print("🌱 Starting fresh from seed prompt.")

        stall_count     = 0
        seed_test_score = 0.0
        test_score      = 0.0
        test_info: Dict = {"docs": {}, "failed": []}

        try:
            for iteration in range(start_iteration, cfg.budget.max_iterations):
                if not self._within_budget():
                    break

                print(f"\n{'='*60}")
                print(f"  ITERATION {iteration + 1} / {cfg.budget.max_iterations}")
                print(f"{'='*60}")

                debug_this_iter = iteration < 2

                val_score, val_info = self._evaluate_corpus(
                    self.val_docs, current_prompt, debug_first=debug_this_iter
                )
                failed_docs = val_info["failed"]

                print(
                    f"  📊 Val F1: {val_score:.4f}  |  "
                    f"Best so far: {max(best_score, 0.0):.4f}  |  "
                    f"Failed docs: {len(failed_docs)}/{len(self.val_docs)}"
                )

                for doc_id, info in val_info["docs"].items():
                    subtrees = info.get("subtrees", {})
                    if subtrees:
                        field_scores = "  ".join(
                            f"{k}={v['f1']:.2f}" for k, v in subtrees.items()
                        )
                        print(f"    >> {doc_id}: [{field_scores}]")

                accepted = val_score > best_score

                if accepted:
                    self.diff_viewer.generate_diff(best_prompt, current_prompt, iteration)
                    best_score  = val_score
                    best_prompt = current_prompt
                    stall_count = 0
                    print(f"  🏆 New best prompt accepted! (F1={val_score:.4f})")
                else:
                    stall_count += 1
                    print(f"  ❌ Rejected. Stall count: {stall_count}.")
                    if current_prompt not in rejected_history:
                        rejected_history.append(current_prompt)
                    current_prompt = best_prompt

                self.state.log_iteration(
                    iteration=iteration,
                    prompt=current_prompt,
                    val_score=val_score,
                    accepted=accepted,
                    breakdown={
                        "per_doc_f1": {
                            doc_id: info["f1"]
                            for doc_id, info in val_info["docs"].items()
                        }
                    },
                )

                if val_score >= 1.0:
                    print("  ✅ Perfect validation score. Halting early.")
                    break

                if not self._within_budget():
                    break

                if not failed_docs:
                    print("  ℹ️  No failed documents — nothing to critique.")
                    continue

                critique_docs = sorted(failed_docs, key=lambda x: x["score"])[:3]
                critiques: List[str] = []

                for fail in critique_docs:
                    try:
                        critique = self.critic.critique(
                            fail["doc"], fail["pred"], fail["gold"]
                        )
                        critiques.append(critique)
                        print(f"  🔍 Critique generated for: {fail['id']}")
                    except DailyLimitError:
                        raise
                    except Exception as exc:
                        print(f"  ⚠️  Critic failed for '{fail.get('id', '?')}': {exc}")

                if not critiques:
                    print("  ⚠️  All critique attempts failed. Skipping mutation.")
                    continue

                if not self._within_budget():
                    break

                try:
                    current_prompt = self.mutator.mutate(
                        current_prompt=best_prompt,
                        critiques=critiques,
                        rejected_prompts=rejected_history,
                        stall_count=stall_count,
                    )
                    print("  ✏️  Mutator drafted a new prompt proposal.")
                except DailyLimitError:
                    raise
                except Exception as exc:
                    print(f"  ⚠️  Mutator failed: {exc}. Retaining best prompt.")
                    current_prompt = best_prompt

        except DailyLimitError:
            print(
                "\n🚫 API quota exhausted (OpenRouter + Ollama). "
                "State persisted — check your API keys or Ollama setup and re-run."
            )

        # ------------------------------------------------------------------
        # Final test set evaluation
        # ------------------------------------------------------------------
        print(f"\n{'='*60}")
        print("  🧪 FINAL HELD-OUT TEST EVALUATION")
        print(f"{'='*60}")

        try:
            print("  Evaluating seed prompt on test set…")
            seed_test_score, _ = self._evaluate_corpus(self.test_docs, cfg.seed_prompt)
            print(f"  🌱 Seed Test F1   : {seed_test_score:.4f}")

            print("  Evaluating final prompt on test set…")
            test_score, test_info = self._evaluate_corpus(self.test_docs, best_prompt)
            print(f"  ✅ Final Test F1  : {test_score:.4f}")
            print(f"  Best Val F1      : {max(best_score, 0.0):.4f}")

        except DailyLimitError:
            print("  ⚠️  Quota exhausted during final evaluation.")

        print("\n  Per-document breakdown:")
        for doc_id, info in test_info.get("docs", {}).items():
            subtrees = info.get("subtrees", {})
            field_str = "  ".join(
                f"{k}={v['f1']:.2f}" for k, v in subtrees.items()
            ) if subtrees else "n/a"
            print(f"    {doc_id}: F1={info['f1']:.4f}  [{field_str}]")
        print(f"\n  Diffs : logs/diffs/  |  Audit : run_state.db")
        print(f"{'='*60}\n")

        self._write_report(
            seed_prompt=cfg.seed_prompt,
            best_prompt=best_prompt,
            seed_test_score=seed_test_score,
            final_test_score=test_score,
            best_val_score=max(best_score, 0.0),
            test_info=test_info,
        )

    # ------------------------------------------------------------------
    # REPORT.md auto-generation
    # ------------------------------------------------------------------

    def _write_report(
        self,
        seed_prompt: str,
        best_prompt: str,
        seed_test_score: float,
        final_test_score: float,
        best_val_score: float,
        test_info: Dict,
    ) -> None:
        trajectory     = self.state.get_trajectory()
        accepted_iters = [t for t in trajectory if t["accepted"]]

        rows = []
        for doc_id, info in test_info.get("docs", {}).items():
            for field, fs in info.get("subtrees", {}).items():
                rows.append(
                    f"| {doc_id} | {field} | "
                    f"{fs['precision']:.3f} | {fs['recall']:.3f} | {fs['f1']:.3f} |"
                )
        subtree_table = "\n".join(rows) if rows else "| — | — | — | — | — |"

        score_curve = "\n".join(
            f"| {t['iteration']+1:>3} | {t['val_score']:.4f} | "
            f"{'✅' if t['accepted'] else '❌'} |"
            for t in trajectory
        )

        notable = "\n".join(
            f"- **Iteration {t['iteration']+1}** — Val F1: {t['val_score']:.4f}"
            for t in accepted_iters[1:]
        ) or "- No mutations improved over the seed during this run."

        diff_note = (
            "The seed prompt was not improved during this run."
            if seed_prompt.strip() == best_prompt.strip()
            else "See `logs/diffs/` for unified diffs of each accepted mutation."
        )

        # OCR cache stats
        ocr_stats = self.state.get_ocr_cache_stats()
        ocr_note = (
            f"OCR cache: **{ocr_stats['total_cached']}** file(s) cached "
            f"across methods: `{ocr_stats['by_method']}`."
        )

        report = textwrap.dedent(f"""\
            # Prompt Optimization Report

            **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
            **Dataset:** `{self.config.dataset.name}`  
            **Models:** extractor=`{self.config.models.extractor}`  
            critic=`{self.config.models.critic}`  mutator=`{self.config.models.mutator}`

            ---

            ## 1. Test-Set Scores

            | Prompt | Test F1 |
            |--------|---------|
            | Seed   | {seed_test_score:.4f} |
            | Final  | {final_test_score:.4f} |
            | **Δ**  | **{final_test_score - seed_test_score:+.4f}** |

            Best validation F1 achieved during optimization: **{best_val_score:.4f}**

            ---

            ## 2. Per-Subtree Breakdown (Final Prompt, Test Set)

            | Document | Field | Precision | Recall | F1 |
            |----------|-------|-----------|--------|----|
            {subtree_table}

            ---

            ## 3. Optimization Trajectory

            | Iter | Val F1 | Accepted |
            |------|--------|----------|
            {score_curve}

            ---

            ## 4. Notable Accepted Mutations

            {notable}

            ---

            ## 5. Seed Prompt

            ```
            {seed_prompt.strip()}
            ```

            ---

            ## 6. Final Prompt

            ```
            {best_prompt.strip()}
            ```

            ---

            ## 7. Diff Summary

            {diff_note}

            ---

            ## 8. Infrastructure Notes

            **OCR Caching:** {ocr_note}  
            Extracted PDF text is persisted in `run_state.db` (keyed by SHA-256 of
            file bytes). Re-runs and resumed interrupted runs skip re-extraction
            entirely for unchanged files.

            **LLM Fallback:** When OpenRouter hits its daily free-model quota or
            returns repeated errors, agents automatically fall back to a local Ollama
            instance (`OLLAMA_BASE_URL`, default `http://localhost:11434`).  The
            model used is controlled by `OLLAMA_MODEL` (default `llama3.2`).
            Gemini remains the exclusive OCR/vision path regardless of which LLM
            backend is active.

            ---

            ## 9. Limitations

            - **Small dataset:** With only 2–8 documents per schema, validation scores
              are noisy and there is real risk of overfitting the prompt to the 1–2
              validation documents.
            - **Positional array alignment:** Object arrays (workExperience, education)
              are compared positionally.  If predicted ordering differs from gold, items
              are penalised even when content is correct.
            - **Free-tier rate limits:** OpenRouter free models have a daily request cap
              (~50/day).  A paid-tier plan would allow full 20-iteration runs without
              interruption.  Ollama fallback mitigates this for local inference.
            - **No train split used:** The greedy loop uses only the validation set for
              feedback.  Train documents are loaded but not yet exploited for few-shot
              example selection.
            - **Stochastic metric caching:** `string_semantic` and `array_llm` scores
              are cached per (pred, gold) pair within a run.  Initial calls for novel
              pairs are non-deterministic.
            """)

        with open("REPORT.md", "w", encoding="utf-8") as f:
            f.write(report)
        print("  📝 REPORT.md written.")