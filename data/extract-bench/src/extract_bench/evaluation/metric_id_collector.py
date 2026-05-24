"""Collect metric IDs and parameter examples from evaluation configs in a schema subtree."""

from typing import Any, Dict

from ..infra.visitors import AnalyzerVisitor
from ..infra.nodes import Schema
from .schema_config_helpers import get_evaluation_config


def collect_metric_ids_with_params(
    *, schema: Schema
) -> dict[str, list[dict[str, Any]]]:
    """Return metric_id -> list of distinct params dicts observed in the subtree."""
    visitor = _MetricIdCollector()
    visitor.visit(schema)
    return visitor.metric_params


class _MetricIdCollector(AnalyzerVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.metric_params: Dict[str, list[dict[str, Any]]] = {}
        self._seen_param_sigs: Dict[str, set[str]] = {}

    def _collect(self, node: Schema) -> None:
        config = get_evaluation_config(node)
        for metric in config.metrics:
            metric_id = metric.metric_id
            params = metric.params or {}
            sig = repr(sorted(params.items()))
            if sig in self._seen_param_sigs.setdefault(metric_id, set()):
                continue
            self._seen_param_sigs[metric_id].add(sig)
            self.metric_params.setdefault(metric_id, []).append(dict(params))

    def pre_visit_children_hook(self, node: Schema) -> None:
        self._collect(node)

    def generic_leaf_visit(self, node: Schema) -> Schema:
        self._collect(node)
        return node
