# Reporting Suite

Generates human-readable and machine-readable reports from structured extraction evaluation results.

## Quick Start

```python
from pathlib import Path
from extract_bench import ReportBuilder, ReportConfig

# Configure the report builder
config = ReportConfig(
    output_dir=Path("./reports"),
    output_name="my-evaluation",  # Optional: auto-generated if None
    max_reasoning_length=200,
    top_n_lowest_fields=5,
)

builder = ReportBuilder(config)

# Build report (runs evaluation internally)
report = builder.build(json_schema, gold_json, extracted_json)

# Or async
report = await builder.build_async(json_schema, gold_json, extracted_json)

# Save to disk
output_path = builder.save(report)
# Creates: reports/my-evaluation/
#   - report.json    (full machine-readable report)
#   - summary.txt    (human-readable summary)
#   - fields.csv     (per-field outcomes)
#   - fields.md      (markdown table)
```

## Architecture

```
reporting/
├── __init__.py          # Package exports
├── models.py            # Data structures (dataclasses)
├── schema_stats.py      # Schema shape collector
├── content_stats.py     # Gold/extracted content collector
├── outcome_stats.py     # Evaluation outcomes processor
├── formatters.py        # Output renderers (JSON, text, CSV, markdown)
└── report_builder.py    # Main orchestrator
```

### Data Flow

```
Inputs (schema, gold, extracted)
        │
        ▼
┌───────────────────┐
│ StructuredEvaluator │  ← runs evaluation
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Stats Collectors │
│  - schema_stats   │  ← AST visitor for schema shape
│  - content_stats  │  ← JSON walkers for content
│  - outcome_stats  │  ← processes MetricResult dict
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ EvaluationReport  │  ← aggregated dataclass
└───────────────────┘
        │
        ▼
┌───────────────────┐
│    Formatters     │
│  - JSON           │
│  - Text summary   │
│  - CSV / Markdown │
└───────────────────┘
```

## Module Guide

### models.py

Frozen dataclasses representing report sections. Key types:

| Class              | Purpose                                                   |
| ------------------ | --------------------------------------------------------- |
| `SchemaStats`      | Node counts by type, required/optional keys               |
| `ContentStats`     | Key counts, value type distribution, array stats          |
| `CoverageStats`    | Present in both, missing, spurious fields                 |
| `OutcomeStats`     | Pass/fail by metric, type, required; confusion matrix     |
| `LlmStats`         | LLM call count, score distribution, top lowest fields     |
| `ErrorBreakdown`   | Failures grouped by prefix, metric, reason                |
| `FieldOutcome`     | Per-field result (path, metric, score, passed, reasoning) |
| `EvaluationReport` | Top-level container with all sections                     |

### schema_stats.py

Traverses the schema AST using `AnalyzerVisitor` to count:

- Total nodes and counts by type (object, array, string, etc.)
- Required vs optional keys

Entry point: `collect_schema_stats(schema: RootSchema) -> SchemaStats`

### content_stats.py

Recursively walks raw JSON dicts to compute:

- Total keys (nested)
- Value type distribution
- Array length statistics (min/median/max)
- Coverage comparison between gold and extracted

Entry points:

- `collect_content_stats(data: dict, label: str) -> ContentStats`
- `compute_coverage(gold, extracted, schema) -> CoverageStats`

### outcome_stats.py

Processes the `results: Dict[str, Dict[str, MetricResult]]` from evaluation:

- Filters internal `anyOf[...]` paths
- Normalizes paths (`$.properties.foo.properties.bar` → `foo.bar`)
- Groups pass/fail by metric, schema type, required/optional
- Computes confusion matrix (TP/FP/FN/TN)
- Extracts LLM stats (scores, reasoning snippets)
- Groups errors by path prefix, metric type, reason category

Entry point: `collect_outcome_stats(results, schema, ...) -> (OutcomeStats, List[FieldOutcome])`

### formatters.py

Renders `EvaluationReport` to different formats:

| Function                                | Output                                  |
| --------------------------------------- | --------------------------------------- |
| `format_json(report)`                   | Full JSON with custom dataclass encoder |
| `format_text_summary(report)`           | Human-readable one-page summary         |
| `format_csv(field_outcomes)`            | CSV with all per-field details          |
| `format_markdown_table(field_outcomes)` | Markdown table                          |

### report_builder.py

Main orchestrator that ties everything together:

- `ReportConfig`: output dir, naming, formatting options
- `ReportBuilder.build_async()`: runs evaluation + collects all stats
- `ReportBuilder.save()`: writes all output formats to disk

## Extending the Suite

### Adding a New Statistic

1. Define a new dataclass in `models.py`
2. Create a collector function in the appropriate module (or new module)
3. Call the collector in `report_builder.py` and add to `EvaluationReport`
4. Update formatters to render the new stat

### Adding a New Output Format

1. Add a formatter function in `formatters.py`
2. Add a config flag in `ReportConfig` (e.g., `save_html: bool = True`)
3. Call the formatter in `ReportBuilder.save()`

### Adding a New Metric Category to Outcome Stats

1. Update `LLM_METRIC_IDS` in `outcome_stats.py` if it's an LLM metric
2. Update `_get_reason_category()` for new reason types
3. The rest is automatic since stats are keyed by `metric_id`

## Path Normalization

The suite normalizes JSONPath strings for human readability:

```
$.properties.terms.properties.loan_commitment.properties.amount
                    ↓
terms.loan_commitment.amount
```

Internal `anyOf[...]` paths are filtered from human-facing outputs but preserved in JSON.

## Design Decisions

- **Uniform weighting**: All fields weighted equally for overall score. `aggregation_weight` in schema reserved for future weighted scoring.
- **`NOT_FOUND` as string**: Treated as a literal string value, not a special marker.
- **Frozen dataclasses**: Report structures are immutable for safety.
- **Async-first**: `build_async()` is primary; `build()` is a sync wrapper.
