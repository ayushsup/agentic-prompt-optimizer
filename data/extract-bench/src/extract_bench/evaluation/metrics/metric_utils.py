"""Shared utility functions for evaluation metrics."""

from typing import Any, Callable, TypeVar, cast

from ...infra.nodes import ArraySchema, Schema
from ..evaluation_config import MetricConfig
from ..schema_value_instantiator import MISSING
from .base_metric import MetricResult

T = TypeVar("T")


def _is_missing(value: Any) -> bool:
    return value is MISSING


def apply_missing_null_policy(
    *,
    metric_id: str,
    node: Any,
    gold: Any,
    extracted: Any,
) -> tuple[Any, Any, MetricResult | None]:
    """Apply the project policy for MISSING, null, and empty array values.

    Returns: (normalized_gold, normalized_extracted, early_result_or_none)
    """
    schema_node = cast(Schema, node)
    is_array_node = isinstance(schema_node, ArraySchema)

    gold_is_missing = _is_missing(gold)
    extracted_is_missing = _is_missing(extracted)

    gold_is_null = gold is None
    extracted_is_null = extracted is None

    gold_is_empty_array = is_array_node and gold == []
    extracted_is_empty_array = is_array_node and extracted == []

    gold_absent = gold_is_missing or gold_is_null or gold_is_empty_array
    extracted_absent = (
        extracted_is_missing or extracted_is_null or extracted_is_empty_array
    )

    def _absent_reason(
        *, is_missing: bool, is_null: bool, is_empty_array: bool, side: str
    ) -> str:
        if is_missing:
            return f"{side}_missing"
        if is_empty_array:
            return f"{side}_empty_array"
        if is_null:
            return f"{side}_null"
        return f"{side}_absent"

    if gold_absent and extracted_absent:
        if gold_is_missing and extracted_is_missing:
            reason = "both_missing"
        elif gold_is_empty_array and extracted_is_empty_array:
            reason = "both_empty_array"
        elif gold_is_null and extracted_is_null:
            reason = "both_null"
        else:
            reason = "both_absent"
        return (
            None,
            None,
            MetricResult(
                metric_id=metric_id,
                score=1.0,
                passed=True,
                details={"reason": reason},
            ),
        )

    if gold_absent or extracted_absent:
        if gold_absent:
            reason = _absent_reason(
                is_missing=gold_is_missing,
                is_null=gold_is_null,
                is_empty_array=gold_is_empty_array,
                side="gold",
            )
        else:
            reason = _absent_reason(
                is_missing=extracted_is_missing,
                is_null=extracted_is_null,
                is_empty_array=extracted_is_empty_array,
                side="extracted",
            )

        return (
            None,
            None,
            MetricResult(
                metric_id=metric_id,
                score=0.0,
                passed=False,
                details={
                    "reason": reason,
                    "gold": None if gold_is_missing else gold,
                    "extracted": None if extracted_is_missing else extracted,
                },
            ),
        )

    return gold, extracted, None


def get_config_param(
    config: MetricConfig | None,
    param_name: str,
    default_value: T,
    converter: Callable[[Any], T] = lambda x: x,
) -> T:
    """Extract a parameter from metric config with type conversion and fallback."""
    if not config or not config.params or param_name not in config.params:
        return default_value

    try:
        return converter(config.params[param_name])
    except (TypeError, ValueError):
        return default_value
