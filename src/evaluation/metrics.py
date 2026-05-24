"""
Evaluation metrics for ExtractBench per-field evaluation policies.

Supported evaluation_config values (as defined in ExtractBench schemas):
  - string_exact      : Case-insensitive exact string match
  - integer_exact     : Exact integer match with type coercion
  - number_tolerance  : Numeric match within 5% relative tolerance
  - boolean_exact     : Exact boolean match with type coercion
  - string_semantic   : Semantic similarity via LLM judge (cached)
  - array_llm         : LLM-based evaluation of array equivalence (cached)

Deterministic metrics are pure functions. Stochastic metrics
(string_semantic, array_llm) are dispatched via an injected callable
and must be cached externally per (prediction, gold) pair.
"""

import hashlib
import json
from typing import Any


# ---------------------------------------------------------------------------
# Deterministic metrics
# ---------------------------------------------------------------------------

def string_exact(pred: Any, gold: Any) -> float:
    """Case-insensitive, strip-whitespace exact string match."""
    if not isinstance(pred, str) or not isinstance(gold, str):
        return 0.0
    return 1.0 if pred.strip().lower() == gold.strip().lower() else 0.0


def integer_exact(pred: Any, gold: Any) -> float:
    """Exact integer match with numeric type coercion."""
    try:
        return 1.0 if int(pred) == int(gold) else 0.0
    except (ValueError, TypeError):
        return 0.0


def number_tolerance(pred: Any, gold: Any, tolerance: float = 0.05) -> float:
    """
    Numeric match within a relative tolerance band.
    Default tolerance = 5% (matching ExtractBench default).
    When gold == 0, requires pred == 0 for a match.
    """
    try:
        pred_val = float(pred)
        gold_val = float(gold)
        if gold_val == 0:
            return 1.0 if pred_val == 0 else 0.0
        return 1.0 if abs(pred_val - gold_val) / abs(gold_val) <= tolerance else 0.0
    except (ValueError, TypeError):
        return 0.0


def boolean_exact(pred: Any, gold: Any) -> float:
    """
    Exact boolean match with type coercion.
    Handles Python bools, strings ('true'/'false'/'yes'/'no'/'1'/'0'), and integers.
    """
    def to_bool(v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, int):
            return bool(v)
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes")
        return False

    try:
        return 1.0 if to_bool(pred) == to_bool(gold) else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Cache key helpers for stochastic metrics
# ---------------------------------------------------------------------------

def compute_cache_key(metric: str, pred: Any, gold: Any) -> str:
    """
    Deterministic SHA-256 cache key for a (metric, pred, gold) triple.
    Ensures identical inputs always resolve to the same cache entry,
    enabling safe re-use of LLM judge results across runs.
    """
    payload = json.dumps(
        {"metric": metric, "pred": pred, "gold": gold},
        sort_keys=True,
        default=str,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Metric dispatch table (deterministic only; stochastic injected at runtime)
# ---------------------------------------------------------------------------

DETERMINISTIC_METRICS = {
    "string_exact":     string_exact,
    "integer_exact":    integer_exact,
    "number_tolerance": number_tolerance,
    "boolean_exact":    boolean_exact,
}

STOCHASTIC_METRICS = {"string_semantic", "array_llm"}
