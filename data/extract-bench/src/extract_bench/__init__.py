"""Structured Extraction Evaluation Suite.

A standalone package for evaluating structured extraction quality by comparing
predicted JSON against gold JSON with per-field metrics.
"""

from .evaluation import (
    AsyncEvaluationConfig,
    BaseMetric,
    EvaluationConfig,
    EvaluationReport,
    MetricConfig,
    MetricRegistry,
    MetricResult,
    ReportBuilder,
    ReportConfig,
    StructuredEvaluator,
    StructuredEvaluatorConfig,
    global_metric_registry,
)

__version__ = "0.1.0"

__all__ = [
    # Main evaluator
    "StructuredEvaluator",
    "StructuredEvaluatorConfig",
    "AsyncEvaluationConfig",
    # Config
    "EvaluationConfig",
    "MetricConfig",
    # Registry
    "MetricRegistry",
    "global_metric_registry",
    # Metrics
    "BaseMetric",
    "MetricResult",
    # Reporting
    "ReportBuilder",
    "ReportConfig",
    "EvaluationReport",
]
