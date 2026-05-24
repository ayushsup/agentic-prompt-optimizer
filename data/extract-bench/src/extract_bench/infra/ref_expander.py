from .nodes import AnySchema, ReferenceSchema, RootSchema
from .visitors import TransformerVisitor


def expand_refs(schema: RootSchema) -> RootSchema:
    """Replace all $ref nodes with their definitions.

    The resulting tree will have no $defs and no $ref nodes.
    Metadata from $ref nodes (title, description, etc.) is preserved on expanded nodes.

    Note: Cyclic schemas are not supported and will cause RecursionError.

    Args:
        schema: The schema tree to expand refs in

    Returns:
        A new schema tree with all refs expanded and no $defs
    """
    return _RefExpander().visit(schema)


class _RefExpander(TransformerVisitor):
    """Internal visitor for expanding references."""

    def visit_rootschema(self, node: RootSchema) -> RootSchema:
        expanded = super().visit_rootschema(node)
        return expanded.create_copy_with_updated_fields({"defs": None})

    def visit_referenceschema(self, node: ReferenceSchema) -> AnySchema:
        def_schema = node.get_def_schema()
        expanded = def_schema.accept(self)

        metadata_fields = ["title", "description", "default", "example", "nullable"]
        ref_metadata = {}
        for field in metadata_fields:
            ref_value = getattr(node, field, None)
            if ref_value is not None:
                ref_metadata[field] = ref_value

        if ref_metadata:
            expanded = expanded.create_copy_with_updated_fields(ref_metadata)

        return expanded
