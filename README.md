# Pibit.ai Prompt Optimizer: Automated Critic-Mutator Framework

A modular, multi-agent pipeline for automatically optimising LLM prompts for structured JSON extraction from documents. Built for the [ExtractBench](https://github.com/ContextualAI/extract-bench) benchmark.

> **Video walkthrough:** _[Link to be added after recording]_

---

## System Architecture

The optimizer implements a **Greedy Accept/Reject loop** with three LLM roles:

```
┌────────────────────────────────────────────────────────────┐
│                    Optimization Loop                        │
│                                                            │
│   ┌──────────┐    JSON     ┌─────────┐   Critique  ┌─────┐│
│   │ Extractor│ ──────────► │  Scorer │ ──────────► │Crit-││
│   │(current  │             │(F1, P/R)│             │ic   ││
│   │ prompt)  │             └─────────┘             └──┬──┘│
│   └──────────┘                                        │   │
│        ▲                                         ┌────▼──┐│
│        │ new prompt                              │Mutator││
│        └────────────────────────────────────────└───────┘│
└────────────────────────────────────────────────────────────┘
```

| Component | Role |
|-----------|------|
| **Extractor** | Applies the current prompt to a document and returns structured JSON |
| **Critic** | Compares predicted JSON to gold standard, producing field-level failure diagnoses |
| **Mutator** | Synthesises critiques into an improved prompt, avoiding previously rejected variants |
| **Scorer** | Schema-aware recursive F1 scorer — fully independent of the optimization loop |
| **StateManager** | SQLite-backed persistence for all LLM calls, scores, and trajectory |

---

## Project Structure

```
pibit-prompt-optimizer/
├── config/
│   ├── base_config.yaml         # Production run configuration
│   └── test_config.yaml         # Fast 3-iteration smoke-test config
├── data/
│   └── extract-bench/           # ExtractBench dataset (cloned separately)
├── src/
│   ├── agents/
│   │   ├── base_agent.py        # LLM client with retry, backoff, daily-limit detection
│   │   └── critic_mutator.py    # Extractor, Critic, Mutator implementations
│   ├── core/
│   │   ├── config_parser.py     # Pydantic configuration validation
│   │   └── state_manager.py     # SQLite persistence: logs, trajectory, metric cache
│   ├── data/
│   │   ├── loader.py            # Multi-stage PDF extraction + vision OCR fallback
│   │   └── splitter.py          # Seeded deterministic train/val/test split
│   ├── evaluation/
...existing code...
```bash
python run.py
```

This will:
1. Load the dataset specified in `config/base_config.yaml`
2. Run up to `max_iterations` of Extract → Score → Critique → Mutate
3. Evaluate the best prompt on the held-out test set
4. Write `REPORT.md` with full results

### Resume an interrupted run

Simply re-run `python run.py`. The optimizer reads `run_state.db` and
continues from the last completed iteration automatically.

### Fast smoke-test (3 iterations)

```bash
python -c "from src.optimizer.loop import OptimizerLoop; OptimizerLoop('config/test_config.yaml').run()"
```

### Run scorer unit tests

```bash
pytest tests/test_scorer.py -v
```

---

## Retargeting to a Different Dataset

To switch from `hiring/resume` to any other ExtractBench schema, edit only
`config/base_config.yaml` — no code changes required:

```yaml
dataset:
  name: "finance/10kq"            # Options: academic/research, finance/10kq,
                                   #          finance/credit_agreement, sport/swimming
  base_path: "./data/extract-bench"
  split_seed: 42
  train_ratio: 0.6
  val_ratio: 0.2
```

Also update the `seed_prompt` in the config to describe the new document type.
The optimization logic, scoring function, and all agents remain untouched.

---

## Split Policy

Documents are shuffled using Python's `random.Random` seeded with `split_seed`
(default `42`), then sliced sequentially:

| Split | Fraction | Purpose |
|-------|----------|---------|
| Train | `train_ratio` | Available for future few-shot selection (loaded but not yet used in the greedy loop) |
| Val | `val_ratio` | Optimization objective — prompt is accepted/rejected based on this score |
| Test | remainder | Held-out; evaluated exactly **once** after the loop completes |

For small datasets (< 3 documents), the splitter redistributes to guarantee
at least 1 document each in val and test, with a printed warning.

---

## Scoring

The scorer traverses the JSON Schema alongside the predicted and gold JSON
objects, computing **micro-averaged precision, recall, and F1** across all
leaf fields. Per-field evaluation policies are honoured exactly as specified:

| Policy | Description |
|--------|-------------|
| `string_exact` | Case-insensitive, whitespace-stripped exact match |
| `integer_exact` | Exact integer match with numeric type coercion |
| `number_tolerance` | Numeric match within 5% relative tolerance |
| `boolean_exact` | Exact boolean match with coercion from strings/ints |
| `string_semantic` | LLM judge (cached per pred/gold pair) |
| `array_llm` | LLM judge for array equivalence (cached) |

**Array alignment policy:**
- *Arrays of objects:* Positional (index-based) alignment
- *Arrays of primitives:* Set-based soft F1

**anyOf fields:** Resolved by matching the gold value's actual Python type
to the appropriate schema variant, then scoring with the matched variant.

Stochastic metric results (`string_semantic`, `array_llm`) are cached in
`run_state.db` keyed by SHA-256 hash of `(metric, pred, gold)`, ensuring
deterministic replay across runs.

---

## Observability

| Artefact | Location | Contents |
|----------|----------|----------|
| LLM call log | `run_state.db` → `llm_logs` | Input, output, tokens, latency, cost per call |
| Optimization trajectory | `run_state.db` → `optimization_trajectory` | Prompt, val F1, accepted/rejected per iteration |
| Metric cache | `run_state.db` → `metric_cache` | Cached stochastic judge scores |
| Prompt diffs | `logs/diffs/diff_iteration_N.diff` | Unified diff between successive accepted prompts |
| Final report | `REPORT.md` | Seed vs final scores, subtree breakdown, trajectory, limitations |

---

## Configuration Reference

```yaml
dataset:
  name: "hiring/resume"          # <category>/<schema> matching extract-bench folder
  base_path: "./data/extract-bench"
  split_seed: 42                 # RNG seed for reproducible splits
  train_ratio: 0.5               # Fraction for training split
  val_ratio: 0.2                 # Fraction for validation split

budget:
  max_iterations: 20             # Hard cap on optimization iterations
  max_cost_dollars: 0.0          # Dollar budget (0 = unlimited / free tier)

vision_model: "google/gemini-2.0-flash-exp:free"  # OCR fallback for scanned PDFs

models:
  extractor: "poolside/laguna-xs.2:free"
  critic:    "poolside/laguna-xs.2:free"
  mutator:   "poolside/laguna-xs.2:free"

seed_prompt: |
  <your extraction prompt here>
```
>>>>>>> 8f9c553 (Initial commit)
