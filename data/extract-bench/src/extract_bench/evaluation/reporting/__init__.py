"""Evaluation reporting subpackage."""

from .formatters import (
    format_csv,
    format_json,
    format_markdown_table,
    format_text_summary,
)
from .models import (
    ArrayStats,
    ConfusionCounts,
    ContentStats,
    CoverageStats,
    ErrorBreakdown,
    ErrorGroup,
    EvaluationReport,
    FieldOutcome,
    LlmStats,
    LowScoringField,
    OutcomeStats,
    SchemaStats,
)
from .report_builder import ReportBuilder, ReportConfig

__all__ = [
    # Main classes
    "ReportBuilder",
    "ReportConfig",
    "EvaluationReport",
    # Models
    "SchemaStats",
    "ArrayStats",
    "ContentStats",
    "CoverageStats",
    "ConfusionCounts",
    "LowScoringField",
    "LlmStats",
    "ErrorGroup",
    "ErrorBreakdown",
    "FieldOutcome",
    "OutcomeStats",
    # Formatters
    "format_json",
    "format_text_summary",
    "format_markdown_table",
    "format_csv",
]
