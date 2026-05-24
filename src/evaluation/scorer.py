"""
Schema-aware recursive scorer for ExtractBench structured extraction.

Computes per-leaf precision, recall, and F1 by traversing the JSON Schema
alongside the predicted and gold JSON objects. Aggregates micro-averaged
metrics across all leaf fields and provides per-subtree breakdowns.

Schema Format
-------------
ExtractBench schema files wrap the actual JSON Schema inside a
"schema_definition" key. This module automatically unwraps that envelope
before scoring, so callers can pass the raw schema file contents.

Array Alignment Policy
----------------------
- Arrays of objects  : Positional (index-based) alignment. Item i in the
  prediction is compared to item i in the gold. Extra items on either side
  are treated as unmatched (contributing to FP or FN as appropriate).
- Arrays of primitives: Set-based soft F1. For exact metrics the matched
  count is |pred_set ∩ gold_set|. For stochastic metrics each gold item
  is matched against its highest-scoring prediction counterpart.

This policy is documented here and in the README. It is deterministic and
reproducible given the same inputs.

anyOf Handling
--------------
Fields declared with "anyOf" (common in ExtractBench for nullable fields
and polymorphic date/skill types) are resolved by selecting the schema
variant that matches the gold value's actual Python type. When multiple
non-null variants match, the one yielding the highest F1 is chosen.

additionalProperties Handling
------------------------------
Object schemas that use "additionalProperties" (e.g., the grouped skills
object) are scored by iterating over every key present in the gold object
and evaluating each value against the additionalProperties schema.

Stochastic Metrics
------------------
string_semantic and array_llm are evaluated via an LLM judge (injected as
a callable). Results are cached by the StateManager per (pred, gold) pair
using compute_cache_key(), ensuring deterministic replay across runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.evaluation.metrics import (
    DETERMINISTIC_METRICS,
    STOCHASTIC_METRICS,
    compute_cache_key,
    string_exact,
)


# ---------------------------------------------------------------------------
# Score accumulator
# ---------------------------------------------------------------------------

@dataclass
class ScoreResult:
    """
    Micro-averaged precision/recall/F1 accumulator.

    true_positives  : weighted match score (0–1) for each matched field
    predicted_count : number of fields present in prediction
    gold_count      : number of fields present in gold
    subtrees        : nested breakdown by field name
    """
    true_positives: float = 0.0
    predicted_count: int = 0
    gold_count: int = 0
    subtrees: Dict[str, "ScoreResult"] = field(default_factory=dict)

    @property
    def precision(self) -> float:
        return self.true_positives / self.predicted_count if self.predicted_count > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.true_positives / self.gold_count if self.gold_count > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def merge(self, other: "ScoreResult") -> None:
        """Accumulate raw counts from another result (micro-averaging)."""
        self.true_positives += other.true_positives
        self.predicted_count += other.predicted_count
        self.gold_count += other.gold_count

    def to_dict(self) -> Dict:
        result: Dict[str, Any] = {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "true_positives": round(self.true_positives, 4),
            "predicted_count": self.predicted_count,
            "gold_count": self.gold_count,
        }
        if self.subtrees:
            result["subtrees"] = {k: v.to_dict() for k, v in self.subtrees.items()}
        return result


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

class Scorer:
    """
    Schema-aware recursive scorer.

    Parameters
    ----------
    state_manager : StateManager, optional
        Used to cache stochastic metric results. If None, caching is skipped.
    judge_callable : Callable[[Any, Any, str], float], optional
        Function (pred, gold, metric_name) -> float in [0, 1].
        Called for string_semantic and array_llm metrics.
        If None, falls back to string_exact for stochastic fields.
    """

    def __init__(
        self,
        state_manager=None,
        judge_callable: Optional[Callable[[Any, Any, str], float]] = None,
    ):
        self.state_manager = state_manager
        self.judge_callable = judge_callable

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_document(
        self,
        pred_json: str,
        gold_json: str,
        schema_str: str,
    ) -> Tuple[float, Dict]:
        """
        Score a single (prediction, gold) pair against the schema.

        Automatically unwraps the ExtractBench "schema_definition" envelope
        so callers can pass the raw schema file contents directly.

        Returns
        -------
        (aggregate_f1, breakdown_dict)
        """
        try:
            pred = json.loads(self._clean_json(pred_json))
        except (json.JSONDecodeError, Exception):
            pred = {}

        try:
            gold = json.loads(gold_json)
            schema = json.loads(schema_str)
        except (json.JSONDecodeError, Exception) as exc:
            return 0.0, {"error": f"Failed to parse gold/schema: {exc}"}

        # Unwrap ExtractBench schema envelope
        if "schema_definition" in schema:
            schema = schema["schema_definition"]

        result = self._score_object(pred, gold, schema, path="root")
        return result.f1, result.to_dict()

    def score_corpus(
        self,
        doc_results: List[Tuple[ScoreResult, Dict]],
    ) -> Dict:
        """Micro-average raw counts across all documents for corpus-level F1."""
        corpus = ScoreResult()
        for result, _ in doc_results:
            corpus.merge(result)
        return corpus.to_dict()

    def calculate_f1(self, precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    # ------------------------------------------------------------------
    # Internal recursive traversal
    # ------------------------------------------------------------------

    def _score_object(
        self,
        pred: Any,
        gold: Any,
        schema_node: Dict,
        path: str = "root",
    ) -> ScoreResult:
        result = ScoreResult()
        if not isinstance(gold, dict):
            return result

        pred = pred if isinstance(pred, dict) else {}

        # --- Handle schemas using additionalProperties (e.g. grouped skills) ---
        if "additionalProperties" in schema_node and "properties" not in schema_node:
            add_prop_schema = schema_node["additionalProperties"]
            for key, gold_val in gold.items():
                pred_val = pred.get(key)
                field_result = self._score_field(
                    pred_val, gold_val, add_prop_schema, path=f"{path}.{key}"
                )
                result.merge(field_result)
                result.subtrees[key] = field_result
            return result

        # --- Standard property-based object ---
        properties = schema_node.get("properties", {})
        for field_name, field_schema in properties.items():
            gold_val = gold.get(field_name)

            if gold_val is None and field_name not in gold:
                # Field absent in gold entirely — skip
                continue

            field_result = self._score_field(
                pred.get(field_name),
                gold_val,
                field_schema,
                path=f"{path}.{field_name}",
            )
            result.merge(field_result)
            result.subtrees[field_name] = field_result

        return result

    def _score_field(
        self,
        pred_val: Any,
        gold_val: Any,
        field_schema: Dict,
        path: str,
    ) -> ScoreResult:
        # Handle anyOf (nullable fields, polymorphic types)
        if "anyOf" in field_schema:
            return self._score_any_of(pred_val, gold_val, field_schema["anyOf"], path)

        field_type = field_schema.get("type", "string")

        if field_type == "array":
            return self._score_array(pred_val, gold_val, field_schema, path)

        if field_type == "object":
            return self._score_object(pred_val, gold_val, field_schema, path)

        # Leaf scalar
        return self._score_leaf(pred_val, gold_val, field_schema)

    def _score_any_of(
        self,
        pred_val: Any,
        gold_val: Any,
        any_of_schemas: List[Dict],
        path: str,
    ) -> ScoreResult:
        """
        Score a field declared with anyOf.

        Strategy:
        1. If gold is None, nothing to score (field is null in gold).
        2. Identify which non-null schema variant matches gold's actual type.
        3. Score with type-matched candidates; fall back to all non-null if none match.
        4. Return the result with the highest F1 / most gold_count.
        """
        if gold_val is None:
            return ScoreResult()

        non_null = [s for s in any_of_schemas if s.get("type") != "null"]
        if not non_null:
            return ScoreResult()

        # Map Python type to JSON schema type
        _type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }
        gold_schema_type = _type_map.get(type(gold_val), "string")
        matching = [s for s in non_null if s.get("type") == gold_schema_type]
        candidates = matching if matching else non_null

        best_result = ScoreResult()
        best_f1 = -1.0
        for schema in candidates:
            r = self._score_field(pred_val, gold_val, schema, path)
            if r.f1 > best_f1 or (
                r.f1 == best_f1 and r.gold_count > best_result.gold_count
            ):
                best_f1 = r.f1
                best_result = r

        return best_result

    def _score_leaf(
        self,
        pred_val: Any,
        gold_val: Any,
        field_schema: Dict,
    ) -> ScoreResult:
        result = ScoreResult()
        if gold_val is None:
            return result

        result.gold_count = 1
        if pred_val is None:
            return result  # FN: gold present, pred absent

        result.predicted_count = 1
        eval_config = field_schema.get("evaluation_config", "string_exact")
        result.true_positives = self._evaluate_with_config(pred_val, gold_val, eval_config)
        return result

    def _score_array(
        self,
        pred_arr: Any,
        gold_arr: Any,
        field_schema: Dict,
        path: str,
    ) -> ScoreResult:
        result = ScoreResult()
        if not isinstance(gold_arr, list):
            return result

        pred_arr = pred_arr if isinstance(pred_arr, list) else []
        items_schema = field_schema.get("items", {})

        # Resolve items type, handling anyOf on items
        if "anyOf" in items_schema:
            non_null_items = [s for s in items_schema["anyOf"] if s.get("type") != "null"]
            items_type = non_null_items[0].get("type", "string") if non_null_items else "string"
        else:
            items_type = items_schema.get("type", "string")

        eval_config = field_schema.get(
            "evaluation_config",
            items_schema.get("evaluation_config", "string_exact"),
        )

        if items_type == "object":
            # ---- Positional alignment for arrays of objects ----
            max_len = max(len(pred_arr), len(gold_arr))
            for i in range(max_len):
                pred_item = pred_arr[i] if i < len(pred_arr) else None
                gold_item = gold_arr[i] if i < len(gold_arr) else None

                if gold_item is None:
                    # Extra prediction item → count predicted leaves as FP
                    fp_count = self._count_leaves(pred_item or {}, items_schema)
                    result.predicted_count += fp_count
                elif pred_item is None:
                    fn_count = self._count_leaves(gold_item, items_schema)
                    result.gold_count += fn_count
                else:
                    item_result = self._score_object(
                        pred_item, gold_item, items_schema, path=f"{path}[{i}]"
                    )
                    result.merge(item_result)
                    result.subtrees[f"{path}[{i}]"] = item_result
        else:
            # ---- Set-based soft F1 for primitive arrays ----
            result.gold_count = len(gold_arr)
            result.predicted_count = len(pred_arr)

            if eval_config in STOCHASTIC_METRICS:
                # Each gold item matched against best-scoring pred item
                for gold_item in gold_arr:
                    if not pred_arr:
                        break
                    best = max(
                        self._evaluate_with_config(p, gold_item, eval_config)
                        for p in pred_arr
                    )
                    result.true_positives += best
            else:
                # Exact set intersection (normalise to lowercase strings)
                gold_norm = {str(g).strip().lower() for g in gold_arr}
                pred_norm = {str(p).strip().lower() for p in pred_arr}
                result.true_positives = float(len(gold_norm & pred_norm))

        return result

    # ------------------------------------------------------------------
    # Metric dispatch
    # ------------------------------------------------------------------

    def _evaluate_with_config(self, pred: Any, gold: Any, eval_config: str) -> float:
        if eval_config in DETERMINISTIC_METRICS:
            return DETERMINISTIC_METRICS[eval_config](pred, gold)
        if eval_config in STOCHASTIC_METRICS:
            return self._evaluate_stochastic(pred, gold, eval_config)
        # Unknown config — fall back to string_exact
        return string_exact(str(pred), str(gold))

    def _evaluate_stochastic(self, pred: Any, gold: Any, metric: str) -> float:
        """Evaluate a stochastic metric with caching."""
        cache_key = compute_cache_key(metric, pred, gold)

        if self.state_manager:
            cached = self.state_manager.get_metric_cache(cache_key)
            if cached is not None:
                return cached

        if self.judge_callable:
            score = self.judge_callable(pred, gold, metric)
        else:
            # No judge available: graceful fallback to exact match
            score = string_exact(str(pred), str(gold))

        if self.state_manager:
            self.state_manager.set_metric_cache(cache_key, score)

        return score

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_json(text: str) -> str:
        """Strip markdown code fences that some models add around JSON output."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Drop first line (```json or ```) and last line (```)
            inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            text = "\n".join(inner).strip()
        return text

    def _count_leaves(self, obj: Any, schema_node: Dict) -> int:
        """Count the number of scoreable leaf fields in a schema subtree."""
        if not isinstance(obj, dict):
            return 1
        count = 0
        for fname, fschema in schema_node.get("properties", {}).items():
            if fname in obj:
                ftype = fschema.get("type", "string")
                if ftype == "object":
                    count += self._count_leaves(obj[fname], fschema)
                elif ftype == "array":
                    count += len(obj[fname]) if isinstance(obj[fname], list) else 1
                else:
                    count += 1
        return count
