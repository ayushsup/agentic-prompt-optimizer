"""Dual-value traversal for gold and extracted values."""

from typing import Any, Dict, List, Union

from .nodes import (
    AnyOfSchema,
    ArraySchema,
    BooleanSchema,
    IntegerSchema,
    NullSchema,
    NumberSchema,
    ObjectSchema,
    ReferenceSchema,
    RootSchema,
    Schema,
    StringSchema,
)


class SchemaInstanceVisitor:
    """Infrastructure to visit both schema and instance jointly.

    Raises ValueError if instance doesn't match schema type.
    """

    def __init__(self):
        self._stack = []
        self.paths = []

    def reset(self) -> None:
        self._stack = []
        self.paths = []

    def _push(self, seg: str) -> None:
        self._stack.append(seg)

    def _pop(self) -> None:
        self._stack.pop()

    def _path(self) -> str:
        return ".".join(self._stack)

    def visit(self, node: Schema, instance: Any) -> Any:
        method_name = f"visit_{node.__class__.__name__.lower()}"
        visitor_method = getattr(self, method_name, None)
        if visitor_method is None:
            raise NotImplementedError(f"{method_name} is not implemented")
        if node.nullable and instance is None:
            return
        return visitor_method(node, instance)

    def visit_rootschema(self, node: RootSchema, instance: Dict[str, Any]) -> None:
        if not isinstance(instance, dict):
            raise ValueError(f"Instance is not a dict at root schema: {self._path()}")
        self._push("$")
        self.visit_objectschema(node, instance)
        self._pop()

    def visit_objectschema(self, node: ObjectSchema, instance: Dict[str, Any]) -> Any:
        if not isinstance(instance, dict):
            raise ValueError(f"Instance is not a dict at object schema: {self._path()}")
        if node.properties:
            for key, value in node.properties.items():
                if key in instance:
                    self._push(f"properties.{key}")
                    self.visit(value, instance[key])
                    self._pop()
        if node.additional_properties and not isinstance(
            node.additional_properties, bool
        ):
            props = node.properties or {}
            for extra_key, extra_value in instance.items():
                if extra_key not in props:
                    self._push(f"additionalProperties.{extra_key}")
                    self.visit(node.additional_properties, extra_value)
                    self._pop()

    def visit_arrayschema(self, node: ArraySchema, instance: List[Any]) -> Any:
        if not isinstance(instance, list):
            raise ValueError(f"Instance is not a list at array schema: {self._path()}")
        for i, item in enumerate(instance):
            self._push(f"items[{i}]")
            self.visit(node.items, item)
            self._pop()

    def visit_referenceschema(self, node: ReferenceSchema, instance: Any) -> Any:
        def_schema = node.get_def_schema()
        self._push(f"ref.{node.get_ref_name()}")
        self.visit(def_schema, instance)
        self._pop()

    def visit_anyofschema(self, node: AnyOfSchema, instance: Any) -> Any:
        raise NotImplementedError(
            "AnyOfSchema is not implemented. This is specific to the final purpose."
        )

    def visit_stringschema(self, node: StringSchema, instance: str) -> Any:
        if not isinstance(instance, str):
            raise ValueError(
                f"Instance is not a string at string schema: {self._path()}"
            )

    def visit_numberschema(
        self, node: NumberSchema, instance: Union[int, float]
    ) -> Any:
        if not isinstance(instance, (int, float)):
            raise ValueError(
                f"Instance is not a number at number schema: {self._path()}"
            )

    def visit_integerschema(self, node: IntegerSchema, instance: int) -> Any:
        if not isinstance(instance, int):
            raise ValueError(
                f"Instance is not an integer at integer schema: {self._path()}"
            )

    def visit_booleanschema(self, node: BooleanSchema, instance: bool) -> Any:
        if not isinstance(instance, bool):
            raise ValueError(
                f"Instance is not a boolean at boolean schema: {self._path()}"
            )

    def visit_nullschema(self, node: NullSchema, instance: None) -> Any:
        if instance is not None:
            raise ValueError(f"Instance is not None at null schema: {self._path()}")
