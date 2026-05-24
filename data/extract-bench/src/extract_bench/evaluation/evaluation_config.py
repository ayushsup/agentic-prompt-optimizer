"""Evaluation config models used to configure metrics per schema node."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class MetricConfig(BaseModel):
    """Configuration for a single evaluation metric."""

    metric_id: str
    weight: Optional[float] = None
    params: Optional[Dict[str, Any]] = Field(
        default=None, description="Override default metric parameters"
    )

    model_config = ConfigDict(frozen=True, extra="forbid")


class EvaluationConfig(BaseModel):
    """Configuration attached to schema nodes to guide evaluation."""

    metrics: List[MetricConfig] = Field(default_factory=list)
    aggregation_weight: Optional[float] = None

    model_config = ConfigDict(frozen=True, extra="forbid")

    @classmethod
    def from_preset(cls, preset: str) -> "EvaluationConfig":
        """Create evaluation config from a preset string."""
        from .presets import get_preset_config

        return get_preset_config(preset)
