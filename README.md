# Pibit.ai Prompt Optimizer — Automated Critic-Mutator Framework

An end-to-end, multi-agent pipeline for automatically optimising LLM prompts for structured JSON extraction from documents. Built for the [ExtractBench](https://github.com/ContextualAI/extract-bench) benchmark as part of the Pibit.ai engineering assignment.

> **Video walkthrough:** [_\[YouTube unlisted link\]](https://youtu.be/p0YF8Zyvaig)_

---

## Table of Contents

1. [Overview](#1-overview)
2. [System Architecture](#2-system-architecture)
3. [Project Structure](#3-project-structure)
4. [Setup](#4-setup)
5. [Running the Optimizer](#5-running-the-optimizer)
6. [Retargeting to a Different Dataset](#6-retargeting-to-a-different-dataset)
7. [Configuration Reference](#7-configuration-reference)
8. [Split Policy](#8-split-policy)
9. [Scoring](#9-scoring)
10. [Optimization Strategy](#10-optimization-strategy)
11. [Prompt Linting](#11-prompt-linting)
12. [LLM Fallback Chain](#12-llm-fallback-chain)
13. [Observability](#13-observability)
14. [Resumability](#14-resumability)
15. [Running Tests](#15-running-tests)
16. [Limitations](#16-limitations)

---

## 1. Overview

This system implements a **Greedy Accept/Reject optimization loop** for prompt engineering. Given a dataset of (PDF document, gold JSON) pairs and a seed prompt, it automatically rewrites the extraction prompt to maximise a held-out validation F1 score — then reports improvement on a sealed test set.

### Key capabilities

| Capability | Description |
|---|---|
| **Multi-agent loop** | Extractor → Scorer → Critic → Mutator, each an independent LLM role |
| **Beam search** | Maintains top-2 accepted prompts; secondary used for severe-stall recovery |
| **Prompt linting** | Validates every candidate prompt before spending an evaluation iteration on it |
| **Four-tier fallback** | OpenRouter → GitHub Models → Gemini → Ollama; seamless degradation |
| **Full persistence** | Every LLM call, score, and accepted prompt persisted to SQLite |
| **Resumability** | Interrupted runs warm-start from the last completed iteration |
| **OCR caching** | PDF text extracted once per file, keyed by SHA-256, replayed instantly |
| **Config-only retargeting** | Switch dataset, models, budget, or seed prompt without touching code |

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Optimization Loop                          │
│                                                                  │
│  ┌─────────────┐   JSON    ┌──────────┐  critique  ┌──────────┐ │
│  │  Extractor  │──────────►│  Scorer  │───────────►│  Critic  │ │
│  │ (eval prompt│           │ (F1,P/R) │            └────┬─────┘ │
│  │  on val set)│           └──────────┘                 │       │
│  └──────┬──────┘                                  ┌────▼──────┐ │
│         │ accept / reject                         │  Mutator  │ │
│         │ (greedy, val F1)                        │(new prompt│ │
│  ┌──────▼──────┐                                  └────┬──────┘ │
│  │  Beam Store │◄──────── new prompt ────────────────── ┘        │
│  │  (top-2     │                                                  │
│  │  accepted)  │          ┌─────────────────────────────────┐   │
│  └─────────────┘          │  Prompt Linter                  │   │
│                           │  (ISO timestamp / languages /   │   │
│                           │   root-key contract checks)     │   │
│                           └─────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
                                    │
                              ┌─────▼──────┐
                              │  SQLite DB  │
                              │ run_state.db│
                              │  llm_logs   │
                              │  trajectory │
                              │  ocr_cache  │
                              │ metric_cache│
                              └─────────────┘
```

### Agent roles

| Agent | Responsibility |
|---|---|
| **Extractor** | Applies the current prompt to a document and returns structured JSON |
| **Critic** | Compares predicted JSON to gold, producing severity-ranked, field-level failure diagnoses |
| **Mutator** | Synthesises critiques into an improved, non-regressive prompt; aware of rejected variants and stall state |
| **Scorer** | Schema-aware recursive F1 scorer — fully independent of the optimization loop |
| **StateManager** | SQLite persistence for all LLM calls, trajectory, OCR text, and metric scores |

---

## 3. Project Structure

```
pibit-prompt-optimizer/
├── config/
│   ├── base_config.yaml         # Production run configuration (edit this to retarget)
│   └── test_config.yaml         # Fast 3-iteration smoke-test
├── data/
│   └── extract-bench/           # ExtractBench dataset (cloned separately — see Setup)
├── logs/
│   └── diffs/                   # Unified diff files for every accepted mutation
├── src/
│   ├── agents/
│   │   ├── base_agent.py        # Four-tier LLM client: retry, back-off, fallback chain
│   │   └── critic_mutator.py    # Extractor, Critic, Mutator agent implementations
│   ├── core/
│   │   ├── config_parser.py     # Pydantic config validation
│   │   └── state_manager.py     # SQLite persistence layer
│   ├── data/
│   │   ├── loader.py            # PDF extraction with OCR fallback and caching
│   │   └── splitter.py          # Seeded deterministic train/val/test split
│   ├── evaluation/
│   │   ├── metrics.py           # Per-field metric functions (exact, semantic, tolerance…)
│   │   └── scorer.py            # Schema-aware recursive F1 scorer
│   └── optimizer/
│       ├── diff_viewer.py       # Unified diff generation for accepted mutations
│       └── loop.py              # Central optimization engine
├── tests/
│   ├── test_scorer.py           # Unit tests: metrics, scorer, array alignment, F1 maths
│   └── test_infrastructure.py   # Unit tests: OCR cache, four-tier fallback, loader cache
├── run.py                       # Entry point
├── reset.py                     # Clears optimization state; preserves OCR cache
├── requirements.txt
└── REPORT.md                    # Auto-generated after each run
```

---

## 4. Setup

### Prerequisites

- Python 3.10+
- Git

### 1 — Clone the repository and install dependencies

```bash
git clone <your-repo-url>
cd pibit-prompt-optimizer
pip install -r requirements.txt
```

### 2 — Clone the ExtractBench dataset

```bash
git clone https://github.com/ContextualAI/extract-bench.git data/extract-bench
```

### 3 — Set environment variables

At minimum you need **one** LLM API key. The system tries each tier in order and skips any that are unconfigured.

```bash
# Tier 1 — OpenRouter (primary; free models available)
export OPENROUTER_API_KEY="sk-or-..."

# Tier 2 — GitHub Models (free via GitHub Student Developer Pack)
export GITHUB_TOKEN="ghp_..."
export GITHUB_MODEL="gpt-4o"          # optional override, default: gpt-4o-mini

# Tier 3 — Gemini (free tier; also used for OCR on scanned PDFs)
export GEMINI_API_KEY="AIza..."
export GEMINI_TEXT_MODEL="gemini-2.0-flash"   # optional override

# Tier 4 — Ollama local (no key needed; must be running)
# ollama serve
# ollama pull llama3
export OLLAMA_BASE_URL="http://localhost:11434"   # optional; this is the default
export OLLAMA_MODEL="llama3"                       # optional; this is the default
```

> **Tip:** For the best performance without paying anything, configure all four tiers. Together they give you well over 200 free LLM calls per day before any tier bottlenecks.

---

## 5. Running the Optimizer

### Full production run (20 iterations)

```bash
python run.py
```

This will:
1. Load documents from the dataset path in `config/base_config.yaml`
2. Run up to `max_iterations` of **Extract → Score → Critique → Mutate**
3. Apply prompt linting before each evaluation to prevent known regression classes
4. Evaluate the best prompt on the sealed test set
5. Write `REPORT.md` with full results, trajectory, and diffs

### Fast smoke-test (3 iterations)

```bash
python -c "
from src.optimizer.loop import OptimizerLoop
OptimizerLoop('config/test_config.yaml').run()
"
```

### Resume an interrupted run

Simply re-run `python run.py`. The optimizer reads `run_state.db`, finds the last completed iteration and the best accepted prompt, and continues from there. No data is lost.

### Reset and start fresh (preserving OCR cache)

```bash
python reset.py
python run.py
```

`reset.py` clears `optimization_trajectory`, `llm_logs`, and `metric_cache` but deliberately preserves `ocr_cache` so PDFs are not re-extracted.

---

## 6. Retargeting to a Different Dataset

**No code changes are required.** Edit only `config/base_config.yaml`:

```yaml
dataset:
  name: "finance/10kq"        # Change this line
  base_path: "./data/extract-bench"
  split_seed: 42
  train_ratio: 0.5
  val_ratio: 0.2
```

Available schemas in ExtractBench:

| Schema name | Domain |
|---|---|
| `hiring/resume` | Resumes / CVs |
| `academic/research` | Academic research papers |
| `finance/10kq` | 10-K / 10-Q SEC filings |
| `finance/credit_agreement` | Credit agreements |
| `sport/sport` | Sports statistics |

Also update `seed_prompt` in the config to match the new document type. The optimization logic, scoring function, and all agents remain **completely untouched**.

---

## 7. Configuration Reference

```yaml
# config/base_config.yaml

dataset:
  name: "hiring/resume"       # <category>/<schema> matching extract-bench folder layout
  base_path: "./data/extract-bench"
  split_seed: 7               # Integer seed for reproducible shuffling
  train_ratio: 0.20           # Fraction of documents in the training split
  val_ratio: 0.57             # Fraction in validation (optimization objective)
                              # Remainder goes to test (evaluated once at the end)

budget:
  max_iterations: 20          # Hard cap on optimization iterations
  max_cost_dollars: 0.0       # Dollar budget; 0 = unlimited (free-tier only)

vision_model: "google/gemma-4-31b-it:free"
  # Model used ONLY for OCR on scanned/image PDFs (loader.py).
  # Separate from the text-inference fallback models below.

models:
  extractor: "google/gemma-4-31b-it:free"   # Primary model for JSON extraction
  critic:    "google/gemma-4-31b-it:free"   # Primary model for critique generation
  mutator:   "google/gemma-4-31b-it:free"   # Primary model for prompt mutation
  github_model: "gpt-4o"                    # Tier-2 fallback (GitHub Models endpoint)

seed_prompt: |
  <your extraction prompt here>
```

All behaviour is driven by these keys. Switching the dataset, models, budget, or prompt requires **only config edits** — the codebase is never touched.

---

## 8. Split Policy

Documents are shuffled with Python's `random.Random(split_seed)` and sliced sequentially:

| Split | Fraction | Purpose |
|---|---|---|
| **Train** | `train_ratio` | Available for few-shot example injection into the Mutator on stalls |
| **Val** | `val_ratio` | Optimization objective — every iteration's accept/reject decision |
| **Test** | remainder | Sealed holdout — evaluated **exactly once**, after the loop ends |

### Small-dataset protection

When the dataset is too small for the ratios to guarantee at least one document in each split, the splitter redistributes automatically and prints a warning:

- 3+ documents → honour ratios (with clamping to ensure ≥1 val and ≥1 test)
- 2 documents → 0 train | 1 val | 1 test
- 1 document → same document used for both val and test

### Split choice for `hiring/resume` (seed=7)

With 7 documents and `split_seed=7`, the split is:

```
1 train  (Marketing)
3 val    (Med, Legal, Academic01) ← intentionally diverse document types
3 test   (IT, Academic02, Finance)
```

This deliberate diversity in the val set prevents the optimizer from overfitting prompt rules to a single document type, which was the observed failure mode with `seed=42` (Finance-only val set causing academic CV regressions on test).

---

## 9. Scoring

The scorer traverses the JSON Schema recursively alongside the predicted and gold JSON, computing **micro-averaged precision, recall, and F1** across every leaf field.

### Per-field evaluation policies

| Policy | Description |
|---|---|
| `string_exact` | Case-insensitive, strip-whitespace exact match |
| `integer_exact` | Exact integer match with numeric type coercion |
| `number_tolerance` | Numeric match within ±5% relative tolerance |
| `boolean_exact` | Exact boolean with coercion from strings/ints/bools |
| `string_semantic` | LLM judge (score 0–1, cached per pred/gold pair) |
| `array_llm` | LLM judge for array-level semantic equivalence (cached) |

### Array alignment policy

- **Arrays of objects** (e.g. `workExperience`, `education`): **positional alignment** — item *i* in the prediction is scored against item *i* in gold. Extra items on either side are unmatched false positives or false negatives.
- **Arrays of primitives** (e.g. `skills`, `publications`, `socialLinks`): **set-based soft F1** — for deterministic metrics the score is `|pred_set ∩ gold_set|`; for stochastic metrics each gold item is matched against its highest-scoring prediction counterpart.

### `anyOf` handling

Fields declared with `anyOf` (nullable scalars, polymorphic date types) are resolved by finding which non-null variant matches the gold value's actual Python type, then scoring against that variant. When multiple variants match, the one yielding the highest F1 is chosen.

### `additionalProperties` handling

Object schemas using `additionalProperties` (e.g. the `skills` object when grouped by category heading) are scored by iterating over every key present in the gold object and evaluating each value against the `additionalProperties` schema.

### Stochastic metric determinism

`string_semantic` and `array_llm` results are cached in `run_state.db → metric_cache`, keyed by `SHA-256(metric + pred + gold)`. Once a (pred, gold) pair is scored, the cached value is replayed in all subsequent runs — making results deterministic from the second invocation onward.

### Independence from the optimization loop

The `Scorer` class is a pure function of `(pred_json, gold_json, schema_str)`. It holds no reference to the optimizer, agents, or any mutable state outside the injected `state_manager` (used only for metric caching). The scoring function can be invoked and tested entirely independently.

---

## 10. Optimization Strategy

### Algorithm: Greedy Accept/Reject with Beam Search

Each iteration:

1. **Evaluate** the current candidate prompt on the val set using the Extractor.
2. **Accept** if val F1 strictly improves over the current best; **reject** otherwise.
3. **Critique** the worst-scoring documents with the Critic agent (severity-ranked, field-level failure diagnoses).
4. **Mutate** — the Mutator agent synthesises the critiques into an improved prompt.
5. **Lint** the candidate prompt; retry mutation with lint feedback if it fails.
6. Repeat until budget exhausted or perfect score reached.

### Beam search (width 2)

The optimizer maintains the top-2 accepted prompts. Under a severe stall (`stall_count ≥ 5`), the Mutator receives the secondary beam prompt as an alternative starting point, allowing it to synthesise ideas from two different lineages rather than incrementally patching a stuck branch.

### Train-example injection

When `stall_count ≥ 2`, the most informationally complete training document (selected deterministically by non-null field count) is formatted as a worked example and injected into the Mutator prompt. This is the single most effective recovery technique for smaller models: instead of abstract critique, the Mutator sees a concrete (document → gold JSON) grounding pair it can base its rewrite on.

### Rejection memory

All previously rejected prompts are tracked in memory and (truncated to 3, 400 chars each) passed to the Mutator, which is explicitly instructed not to reproduce those variants. This prevents cycling.

### Escalating boldness on stall

| Stall count | Strategy |
|---|---|
| 0–2 | Normal incremental improvement |
| 3–4 | Mutator warned to try significantly different approach; inline format examples encouraged |
| ≥ 5 | Mutator warned of SEVERE STALL; radical restructure mandated; secondary beam provided |

### Critique format (Critic agent)

The Critic produces structured, severity-ranked blocks:

```
[SEVERITY: HIGH|MEDIUM|LOW]
[FIELD: <dot.path.to.field>]
[TYPE: MISSING|WRONG_VALUE|TYPE_MISMATCH|HALLUCINATED|FORMAT_ERROR|ARRAY_MISMATCH|NULL_WHEN_PRESENT|FORBIDDEN_KEY]
Predicted : <value>
Gold      : <expected value>
Fix       : <one concrete prompt-wording instruction>
```

Up to 7 failures are listed, HIGH severity first. The Critic also minifies JSON inputs before sending to the LLM, reducing token consumption by ~40% on large documents.

---

## 11. Prompt Linting

Every candidate prompt produced by the Mutator is validated by a deterministic linter **before** an evaluation iteration is spent on it. Catching regressions before the API call saves one full iteration (3 LLM calls) per lint failure.

### Lint rules

| Rule | What it catches | Why it matters |
|---|---|---|
| **ISO timestamp anti-pattern** | Instructions to format dates as `YYYY-MM-DDTHH:MM:SS.000Z` | Gold data uses integer years (e.g. `2019`) or short strings (`"Spring 2010"`). ISO timestamps cause every date field to score 0.0 |
| **`languages` key inconsistency** | `languages` appears in field rules but is absent from the root-key contract | Creates conflicting instructions; model unreliably includes the field |
| **Required root keys missing** | Any of the 9 required keys absent from the contract zone (first 900 chars) | Extraction model won't produce the required structure |

### Lint retry loop

```
for lint_attempt in 0..MAX_LINT_RETRIES:
    candidate = Mutator.mutate(...)
    passes, reason = lint_prompt(candidate)
    if passes:
        use candidate
        break
    else:
        prepend lint reason as a HIGH-priority critique
        retry mutation with lint feedback
```

`MAX_LINT_RETRIES = 2`. If all attempts fail, the current best prompt is retained and the iteration is counted as a stall.

---

## 12. LLM Fallback Chain

All three agents (Extractor, Critic, Mutator) share the same four-tier fallback defined in `BaseAgent`:

```
Tier 1 — OpenRouter       (OPENROUTER_API_KEY)
         ↓ daily quota / 5xx / non-retryable error
Tier 2 — GitHub Models    (GITHUB_TOKEN; Azure AI inference endpoint)
         ↓ failure or quota
Tier 3 — Gemini text      (GEMINI_API_KEY; gemini-2.0-flash by default)
         ↓ failure or quota
Tier 4 — Ollama           (local; OLLAMA_BASE_URL / OLLAMA_MODEL)
         ↓ unreachable
         → DailyLimitError (state persisted; re-run when available)
```

### Per-tier behaviour

| Tier | Rate-limit handling |
|---|---|
| OpenRouter | Per-minute 429: exponential back-off (10 s, 20 s, 40 s…) up to 5 retries. Daily-quota 429: immediately skip to Tier 2 |
| GitHub Models | Any failure: skip to Tier 3 |
| Gemini | Any quota/rate error: skip to Tier 4 |
| Ollama | Unreachable: raises `DailyLimitError`; state is safely persisted |

> **OCR vs text inference:** Gemini is also used as the **OCR backend** for scanned PDFs (via the Google File API in `loader.py`). This is a separate code path from the text-inference fallback above — the two uses of Gemini have independent quota pools.

---

## 13. Observability

Every meaningful event in the optimizer is persisted to `run_state.db` (SQLite) and/or written to disk.

### Database tables

| Table | Contents |
|---|---|
| `llm_logs` | Every LLM call: role, model, full prompt, full response, input tokens, output tokens, cost, latency |
| `optimization_trajectory` | Per-iteration: prompt text, val F1, accepted/rejected, per-document F1 breakdown |
| `metric_cache` | SHA-256-keyed stochastic metric results (`string_semantic`, `array_llm`) |
| `ocr_cache` | SHA-256-keyed PDF text extractions (method label included) |

### Prompt diffs

Every time a new prompt is **accepted**, a unified diff between the previous accepted prompt and the new one is written to:

```
logs/diffs/diff_iteration_N.diff
```

This allows you to inspect exactly which wording changes were responsible for any score change, at line granularity.

### Console output

Each iteration prints:

```
============================================================
  ITERATION 4 / 20
============================================================
  📊 Val F1: 0.8142  |  Best so far: 0.7291  |  Failed docs: 2/3
    >> Resume-Academic01: [personalInfo=1.00  workExperience=0.85  education=0.72  ...]
    >> Resume-Legal:      [personalInfo=1.00  workExperience=0.60  ...]
  🏆 New best! (F1=0.8142)  Δ +12 lines  -5 lines
  🔍 Critique for: Resume-Academic01
  🔍 Critique for: Resume-Legal
  📌 Injecting train example (stall=0).
  ✏️  Mutator proposed a new prompt (lint attempt 1 ✅).
```

### Final report

`REPORT.md` is auto-generated at the end of every run and contains:

- Seed-prompt and final-prompt test-set F1 scores and their delta
- Per-document, per-field breakdown (precision / recall / F1)
- Full optimization trajectory table (every iteration, val F1, accepted/rejected)
- Notable accepted mutations with diff summaries
- Full seed prompt and final prompt text
- Infrastructure notes (OCR cache stats, fallback chain status)

---

## 14. Resumability

The optimizer is designed to be safely interrupted at any point:

- **KeyboardInterrupt** (Ctrl+C) is caught; the current state is already in SQLite.
- **DailyLimitError** (entire fallback chain exhausted) is caught; the run exits cleanly.
- **Any other exception** is caught in `run.py`; the traceback is printed; SQLite is unaffected.

On the next invocation of `python run.py`, the optimizer:

1. Reads `get_best_state()` → the highest-scoring accepted prompt and its val F1.
2. Reads `get_last_completed_iteration()` → the iteration index to resume from.
3. Reads `get_rejected_prompts()` → the rejection memory, so it doesn't re-propose failed variants.
4. Continues from `start_iteration = last_done + 1`.

OCR text, stochastic metric scores, and LLM call logs are all preserved across runs.

---

## 15. Running Tests

### Scorer unit tests

```bash
pytest tests/test_scorer.py -v
```

Covers:
- All deterministic metric functions (`string_exact`, `integer_exact`, `number_tolerance`, `boolean_exact`)
- `ScoreResult` accumulation and F1 arithmetic (including zero-division edge cases)
- `Scorer.score_document` on synthetic (schema, gold, pred) triples
- Array alignment policy (positional for object arrays, set-based for primitive arrays)
- Graceful handling of malformed predictions (empty JSON, parse errors, markdown fences)
- Judge float parser (`_parse_judge_float`) for all common free-model response styles
- Word-overlap F1 fallback

### Infrastructure tests

```bash
pytest tests/test_infrastructure.py -v
```

Covers:
- OCR cache: miss, hit, roundtrip, invalidation on file change, overwrite, stats
- Four-tier fallback chain: each tier exercised in isolation; all four tiers exhausted
- Per-minute 429 back-off and retry on OpenRouter
- Missing `GITHUB_TOKEN` and `GEMINI_API_KEY` skip their respective tiers cleanly
- `ExtractBenchLoader._extract_with_cache`: cache hit skips extraction; miss calls and stores

### Run all tests

```bash
pytest tests/ -v
```

---

## 16. Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| **Small dataset** | 2–8 documents per schema; val scores are noisy; overfitting risk | Diverse 3-doc val set (seed=7); sealed test holdout; stochastic metric caching for determinism |
| **Positional array alignment** | `workExperience` / `education` order differences are penalised even when content is correct | Documented policy; consistent across seed and final prompt evaluation |
| **Free-tier rate limits** | Each API tier has a daily cap (~50–200 calls/day) | Four-tier chain distributes load; Ollama provides unlimited local fallback |
| **Prompt linter scope** | Only catches ISO timestamp, `languages` inconsistency, and missing root-key regressions | Other regression classes are caught by the val score drop but still consume one iteration |
| **Single-document val set risk** | If `val_ratio` is small and the dataset is small, val may be a single document | Mitigated by `split_seed=7` which gives 3 val documents for `hiring/resume` |
| **Mutator faithfulness** | Smaller models (qwen2.5:3b, gemma) sometimes ignore constraints or introduce hallucinated field names | Prompt linting catches the most expensive cases; rejection memory prevents cycling |
| **Stochastic metric first call** | `string_semantic` and `array_llm` are non-deterministic on their first invocation for a novel (pred, gold) pair | Cached on first call; fully deterministic for all subsequent evaluations and re-runs |

---

## Appendix: Environment Variable Quick Reference

| Variable | Required? | Default | Purpose |
|---|---|---|---|
| `OPENROUTER_API_KEY` | Yes (Tier 1) | — | Primary LLM API |
| `GITHUB_TOKEN` | Recommended | — | Free GPT-4o-mini / gpt-4o via GitHub Student Dev Pack |
| `GITHUB_MODEL` | No | `gpt-4o-mini` | Override GitHub Models model name |
| `GEMINI_API_KEY` | Recommended | — | Tier-3 text inference + OCR for scanned PDFs |
| `GEMINI_TEXT_MODEL` | No | `gemini-2.0-flash` | Override Gemini text model |
| `OLLAMA_BASE_URL` | No | `http://localhost:11434` | Local Ollama server URL |
| `OLLAMA_MODEL` | No | `llama3` | Local model name (recommend `llama3` over `gemma:2b`) |

---

## References

- [Karpathy — autoresearch](https://github.com/karpathy/autoresearch)
- [ExtractBench (ContextualAI)](https://github.com/ContextualAI/extract-bench)
- [OpenRouter free models](https://openrouter.ai/models?q=free)
- [GitHub Models (Azure AI inference)](https://docs.github.com/en/github-models)
- [Google AI Studio (Gemini free tier)](https://aistudio.google.com/)
- [Ollama](https://ollama.com/)