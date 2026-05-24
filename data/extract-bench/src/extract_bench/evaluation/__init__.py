"""Structured extraction evaluation module."""

from .evaluation_config import EvaluationConfig, MetricConfig
from .metric_registry import MetricRegistry, global_metric_registry
from .metrics.base_metric import BaseMetric, MetricResult
from .reporting import EvaluationReport, ReportBuilder, ReportConfig
from .schema_config_helpers import (
    add_evaluation_configs_to_export,
    get_default_evaluation_config,
    get_evaluation_config,
    should_evaluate,
)
from .structured_evaluator import (
    AsyncEvaluationConfig,
    StructuredEvaluator,
    StructuredEvaluatorConfig,
)

__all__ = [
    # Config
    "EvaluationConfig",
    "MetricConfig",
    # Registry
    "MetricRegistry",
    "global_metric_registry",
    # Metrics
    "BaseMetric",
    "MetricResult",
    # Helpers
    "get_evaluation_config",
    "get_default_evaluation_config",
    "should_evaluate",
    "add_evaluation_configs_to_export",
    # Evaluator
    "AsyncEvaluationConfig",
    "StructuredEvaluator",
    "StructuredEvaluatorConfig",
    # Reporting
    "ReportBuilder",
    "ReportConfig",
    "EvaluationReport",
]
