"""Base interfaces for evaluation metrics."""

from dataclasses import dataclass
from typing import Any, Dict, Protocol

from ..evaluation_config import MetricConfig


@dataclass(frozen=True, slots=True)
class MetricResult:
    """Structured result returned by metrics."""

    metric_id: str
    score: float
    passed: bool | None = None
    details: dict[str, Any] | None = None


class MetricContext(Protocol):
    """Subset of schema node interface used by metrics."""

    def get_gold_value(self) -> Any: ...

    def get_extracted_value(self) -> Any: ...

    def get_metadata_summary(self) -> Dict[str, Any]: ...


class BaseMetric(Protocol):
    """Common interface implemented by all metrics.

    All metrics are async to enable parallel evaluation.
    """

    metric_id: str
    recurse_into_children: bool = True

    async def evaluate(
        self, node: MetricContext, config: MetricConfig | None = None
    ) -> MetricResult: ...
