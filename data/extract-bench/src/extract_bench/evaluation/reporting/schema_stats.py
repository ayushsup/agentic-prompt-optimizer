"""Schema shape statistics collector."""

from collections import defaultdict
from typing import Dict, Set

from ...infra.nodes import (
    AnyOfSchema,
    ArraySchema,
    ObjectSchema,
    RootSchema,
    Schema,
)
from ...infra.visitors import AnalyzerVisitor
from .models import SchemaStats


class _SchemaStatsVisitor(AnalyzerVisitor):
    """Visitor that collects schema shape statistics."""

    def __init__(self):
        self.total_nodes = 0
        self.counts_by_type: Dict[str, int] = defaultdict(int)
        self.required_keys: Set[str] = set()
        self.optional_keys: Set[str] = set()
        self.required_by_type: Dict[str, int] = defaultdict(int)
        self.optional_by_type: Dict[str, int] = defaultdict(int)
        self._path_stack: list[str] = []
        self._required_at_parent: Set[str] = set()

    def _path(self) -> str:
        return ".".join(self._path_stack)

    def _count_node(self, node: Schema) -> None:
        self.total_nodes += 1
        node_type = node.get_type() or node.__class__.__name__.lower().replace(
            "schema", ""
        )
        self.counts_by_type[node_type] += 1

    def visit_rootschema(self, node: RootSchema) -> None:
        self._count_node(node)
        if node.properties:
            required_set = set(node.required or [])
            for key, child in node.properties.items():
                self._path_stack.append(key)
                is_required = key in required_set
                self._track_key(child, is_required)
                child.accept(self)
                self._path_stack.pop()

    def visit_objectschema(self, node: ObjectSchema) -> None:
        self._count_node(node)
        if node.properties:
            required_set = set(node.required or [])
            for key, child in node.properties.items():
                self._path_stack.append(key)
                is_required = key in required_set
                self._track_key(child, is_required)
                child.accept(self)
                self._path_stack.pop()

    def visit_arrayschema(self, node: ArraySchema) -> None:
        self._count_node(node)
        if node.items:
            self._path_stack.append("items")
            node.items.accept(self)
            self._path_stack.pop()

    def visit_anyofschema(self, node: AnyOfSchema) -> None:
        self._count_node(node)
        for i, variant in enumerate(node.any_of or []):
            self._path_stack.append(f"anyOf[{i}]")
            variant.accept(self)
            self._path_stack.pop()

    def generic_leaf_visit(self, node: Schema) -> None:
        self._count_node(node)

    def _track_key(self, node: Schema, is_required: bool) -> None:
        path = self._path()
        node_type = node.get_type() or node.__class__.__name__.lower().replace(
            "schema", ""
        )
        if is_required:
            self.required_keys.add(path)
            self.required_by_type[node_type] += 1
        else:
            self.optional_keys.add(path)
            self.optional_by_type[node_type] += 1


def collect_schema_stats(schema: RootSchema) -> SchemaStats:
    """Collect schema shape statistics from an AST."""
    visitor = _SchemaStatsVisitor()
    schema.accept(visitor)

    return SchemaStats(
        total_nodes=visitor.total_nodes,
        counts_by_type=dict(visitor.counts_by_type),
        required_keys_count=len(visitor.required_keys),
        optional_keys_count=len(visitor.optional_keys),
        required_by_type=dict(visitor.required_by_type),
        optional_by_type=dict(visitor.optional_by_type),
    )
