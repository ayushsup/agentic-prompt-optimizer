"""Boolean evaluation metrics."""

from dataclasses import dataclass
from typing import Any

from ..evaluation_config import MetricConfig
from .base_metric import BaseMetric, MetricContext, MetricResult
from .metric_descriptors import MetricPromptDescriptor
from .policy_metric import PolicyAwareMetric


def _to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


@dataclass(frozen=True, slots=True)
class BooleanExactMatchMetric(PolicyAwareMetric, BaseMetric):
    metric_id: str = "boolean_exact"
    prompt_descriptor = MetricPromptDescriptor(
        metric_id="boolean_exact",
        summary="Exact boolean equality.",
        pass_rule="pass iff gold is True/False and predicted equals gold",
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
        gold_bool = _to_bool(gold)
        extracted_bool = _to_bool(extracted)

        if gold_bool is None or extracted_bool is None:
            passed = gold_bool is extracted_bool
            score = 1.0 if passed else 0.0
            return MetricResult(
                metric_id=self.metric_id,
                score=score,
                passed=passed,
                details={"gold": gold, "extracted": extracted},
            )

        passed = gold_bool == extracted_bool
        score = 1.0 if passed else 0.0
        return MetricResult(
            metric_id=self.metric_id,
            score=score,
            passed=passed,
            details={"gold": gold_bool, "extracted": extracted_bool},
        )
