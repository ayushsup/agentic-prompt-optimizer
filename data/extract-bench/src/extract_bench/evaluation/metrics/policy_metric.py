"""Policy-aware base class for evaluation metrics.

Centralizes the policy that treats MISSING as null-like and
empty arrays as null-like for array nodes.
"""

from typing import Any

from ..evaluation_config import MetricConfig
from .base_metric import MetricContext, MetricResult
from .metric_utils import apply_missing_null_policy


class PolicyAwareMetric:
    """Mixin/base class that applies missing/null policy before metric logic."""

    metric_id: str

    async def evaluate(
        self, node: MetricContext, config: MetricConfig | None = None
    ) -> MetricResult:
        gold_raw = node.get_gold_value()
        extracted_raw = node.get_extracted_value()
        gold, extracted, early = apply_missing_null_policy(
            metric_id=self.metric_id,
            node=node,
            gold=gold_raw,
            extracted=extracted_raw,
        )
        if early is not None:
            return early
        return await self._evaluate_values(
            node=node, gold=gold, extracted=extracted, config=config
        )

    async def _evaluate_values(
        self,
        *,
        node: MetricContext,
        gold: Any,
        extracted: Any,
        config: MetricConfig | None,
    ) -> MetricResult:
        raise NotImplementedError
