"""Numeric evaluation metrics (integers and floats)."""

from dataclasses import dataclass
from math import isclose
from typing import Any

from ..evaluation_config import MetricConfig
from .base_metric import BaseMetric, MetricContext, MetricResult
from .metric_descriptors import MetricPromptDescriptor
from .metric_utils import get_config_param
from .policy_metric import PolicyAwareMetric


def _to_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True, slots=True)
class ExactNumberMatchMetric(PolicyAwareMetric, BaseMetric):
    metric_id: str = "number_exact"
    prompt_descriptor = MetricPromptDescriptor(
        metric_id="number_exact",
        summary="Exact numeric equality after coercion to float when possible.",
        pass_rule="pass iff gold_number == predicted_number",
        score_rule="1.0 if pass else 0.0",
    )

    async def _evaluate_values(
        self,
        *,
        node: MetricContext,
        gold: Any,
        extracted: Any,
        config: MetricConfig | None,
    ) -> MetricResult:
        gold_number = _to_number(gold)
        extracted_number = _to_number(extracted)

        passed = gold_number == extracted_number
        score = 1.0 if passed else 0.0
        return MetricResult(
            metric_id=self.metric_id,
            score=score,
            passed=passed,
            details={"gold": gold_number, "extracted": extracted_number},
        )


@dataclass(frozen=True, slots=True)
class NumberToleranceMetric(PolicyAwareMetric, BaseMetric):
    metric_id: str = "number_tolerance"
    default_tolerance: float = 1e-6
    prompt_descriptor = MetricPromptDescriptor(
        metric_id="number_tolerance",
        summary="Numeric match within an absolute tolerance.",
        pass_rule="pass iff |gold - predicted| <= tolerance",
        score_rule="1.0 if pass else max(0.0, 1.0 - |gold-predicted|/tolerance)",
        params={"tolerance": {"type": "number", "meaning": "absolute tolerance"}},
    )

    async def _evaluate_values(
        self,
        *,
        node: MetricContext,
        gold: Any,
        extracted: Any,
        config: MetricConfig | None,
    ) -> MetricResult:
        gold_number = _to_number(gold)
        extracted_number = _to_number(extracted)

        tolerance = get_config_param(config, "tolerance", self.default_tolerance, float)

        if gold_number is None or extracted_number is None:
            passed = gold_number is extracted_number
            score = 1.0 if passed else 0.0
            return MetricResult(
                metric_id=self.metric_id,
                score=score,
                passed=passed,
                details={
                    "gold": gold,
                    "extracted": extracted,
                    "tolerance": tolerance,
                },
            )

        delta = abs(gold_number - extracted_number)
        effective_tolerance = max(tolerance, 1e-12)
        passed = delta <= tolerance or isclose(
            delta, tolerance, rel_tol=1e-9, abs_tol=1e-12
        )
        score = 1.0 if passed else max(0.0, 1.0 - delta / effective_tolerance)
        return MetricResult(
            metric_id=self.metric_id,
            score=score,
            passed=passed,
            details={
                "gold": gold_number,
                "extracted": extracted_number,
                "delta": round(delta, 12),
                "tolerance": tolerance,
            },
        )


@dataclass(frozen=True, slots=True)
class IntegerExactMatchMetric(PolicyAwareMetric, BaseMetric):
    metric_id: str = "integer_exact"
    prompt_descriptor = MetricPromptDescriptor(
        metric_id="integer_exact",
        summary="Exact integer equality after coercion to int when possible.",
        pass_rule="pass iff int(gold) == int(predicted)",
        score_rule="1.0 if pass else 0.0",
    )

    async def _evaluate_values(
        self,
        *,
        node: MetricContext,
        gold: Any,
        extracted: Any,
        config: MetricConfig | None,
    ) -> MetricResult:
        try:
            gold_int = int(gold)
        except (TypeError, ValueError):
            gold_int = None
        try:
            extracted_int = int(extracted)
        except (TypeError, ValueError):
            extracted_int = None

        passed = gold_int == extracted_int
        score = 1.0 if passed else 0.0
        return MetricResult(
            metric_id=self.metric_id,
            score=score,
            passed=passed,
            details={"gold": gold_int, "extracted": extracted_int},
        )
