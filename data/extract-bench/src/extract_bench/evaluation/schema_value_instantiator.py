"""Set gold and extracted values on schema nodes for evaluation.

Given a JSON schema tree, a gold JSON, and an extracted JSON, returns a new
JSON schema tree with the gold and extracted values set on appropriate nodes.
"""

from typing import Any, Dict

from ..infra.ref_expander import expand_refs
from ..infra.schema_instance_visitor import SchemaInstanceVisitor
from ..infra.nodes import (
    AnyOfSchema,
    ArraySchema,
    BooleanSchema,
    IntegerSchema,
    NullSchema,
    NumberSchema,
    ObjectSchema,
    RootSchema,
    StringSchema,
)


class _MissingSentinel:
    """Sentinel value to distinguish between None (explicit null) and missing values."""

    def __repr__(self):
        return "MISSING"

    def __bool__(self):
        return False


MISSING = _MissingSentinel()


class SchemaValueInstantiator(SchemaInstanceVisitor):
    """Traverses schema tree jointly with gold and extracted instances,
    setting values on each relevant node.

    Usage:
        instantiator = SchemaValueInstantiator(schema, gold_json, extracted_json)
        result = instantiator.instantiate()
    """

    def __init__(
        self, schema: RootSchema, gold_instance: Dict, extracted_instance: Dict
    ):
        super().__init__()
        self.schema = expand_refs(schema)
        self.gold_instance = gold_instance
        self.extracted_instance = extracted_instance

    def instantiate(self) -> RootSchema:
        """Main entry point: traverse schema and set gold/extracted values."""
        self._visit_dual(self.schema, self.gold_instance, self.extracted_instance)
        return self.schema

    def _validate_type(
        self, value: Any, expected_types: type | tuple, type_name: str, source: str
    ) -> None:
        if value is not MISSING and not isinstance(value, expected_types):
            raise ValueError(
                f"{source} instance is not {type_name} at {type_name.lower()} schema: {self._path()}"
            )

    def _validate_and_set(
        self,
        node: Any,
        gold_value: Any,
        extracted_value: Any,
        expected_types: type | tuple,
        type_name: str,
    ) -> None:
        self._validate_type(gold_value, expected_types, type_name, "Gold")
        self._validate_type(extracted_value, expected_types, type_name, "Extracted")
        node.set_gold_value(gold_value)
        node.set_extracted_value(extracted_value)

    def _validate_null_and_set(
        self, node: NullSchema, gold_value: Any, extracted_value: Any
    ) -> None:
        if gold_value is not MISSING and gold_value is not None:
            raise ValueError(
                f"Gold instance is not None at null schema: {self._path()}"
            )
        if extracted_value is not MISSING and extracted_value is not None:
            raise ValueError(
                f"Extracted instance is not None at null schema: {self._path()}"
            )
        node.set_gold_value(gold_value)
        node.set_extracted_value(extracted_value)

    def _visit_dual(self, node: Any, gold_value: Any, extracted_value: Any) -> None:
        method_name = f"_visit_dual_{node.__class__.__name__.lower()}"
        visitor_method = getattr(self, method_name, None)
        if visitor_method is None:
            raise NotImplementedError(f"{method_name} is not implemented")

        if node.nullable:
            if gold_value is None or extracted_value is None:
                node.set_gold_value(gold_value)
                node.set_extracted_value(extracted_value)
                return

        return visitor_method(node, gold_value, extracted_value)

    def _visit_dual_rootschema(
        self, node: RootSchema, gold_value: Any, extracted_value: Any
    ) -> None:
        self._push("$")
        self._visit_dual_objectschema(node, gold_value, extracted_value)
        self._pop()

    def _visit_dual_objectschema(
        self, node: ObjectSchema, gold_value: Any, extracted_value: Any
    ) -> None:
        self._validate_and_set(node, gold_value, extracted_value, dict, "a dict")

        if node.properties:
            for key, child_schema in node.properties.items():
                gold_child = (
                    gold_value.get(key, MISSING)
                    if gold_value is not MISSING
                    else MISSING
                )
                extracted_child = (
                    extracted_value.get(key, MISSING)
                    if extracted_value is not MISSING
                    else MISSING
                )
                is_required = node.required and key in node.required
                if is_required:
                    if gold_child is MISSING or extracted_child is MISSING:
                        source = "Gold" if gold_child is MISSING else "Extracted"
                        raise ValueError(
                            f"Required property '{key}' missing in {source} instance at {self._path()}"
                        )

                self._push(f"properties.{key}")
                self._visit_dual(child_schema, gold_child, extracted_child)
                self._pop()

        if node.additional_properties and not isinstance(
            node.additional_properties, bool
        ):
            props = node.properties or {}
            props_keys = set(props.keys())

            all_keys: set[str] = set()
            if gold_value is not MISSING:
                all_keys.update(gold_value.keys())
            if extracted_value is not MISSING:
                all_keys.update(extracted_value.keys())

            for extra_key in all_keys - props_keys:
                gold_extra = (
                    gold_value.get(extra_key, MISSING)
                    if gold_value is not MISSING
                    else MISSING
                )
                extracted_extra = (
                    extracted_value.get(extra_key, MISSING)
                    if extracted_value is not MISSING
                    else MISSING
                )

                if gold_extra is MISSING and extracted_extra is MISSING:
                    continue

                self._push(f"additionalProperties.{extra_key}")
                self._visit_dual(
                    node.additional_properties, gold_extra, extracted_extra
                )
                self._pop()

    def _visit_dual_arrayschema(
        self, node: ArraySchema, gold_value: Any, extracted_value: Any
    ) -> None:
        self._validate_and_set(node, gold_value, extracted_value, list, "a list")

    def _visit_dual_anyofschema(
        self, node: AnyOfSchema, gold_value: Any, extracted_value: Any
    ) -> None:
        node.set_gold_value(gold_value)
        node.set_extracted_value(extracted_value)

    def _visit_dual_stringschema(
        self, node: StringSchema, gold_value: Any, extracted_value: Any
    ) -> None:
        self._validate_and_set(node, gold_value, extracted_value, str, "a string")

    def _visit_dual_numberschema(
        self, node: NumberSchema, gold_value: Any, extracted_value: Any
    ) -> None:
        self._validate_and_set(
            node, gold_value, extracted_value, (int, float), "a number"
        )

    def _visit_dual_integerschema(
        self, node: IntegerSchema, gold_value: Any, extracted_value: Any
    ) -> None:
        self._validate_and_set(node, gold_value, extracted_value, int, "an integer")

    def _visit_dual_booleanschema(
        self, node: BooleanSchema, gold_value: Any, extracted_value: Any
    ) -> None:
        self._validate_and_set(node, gold_value, extracted_value, bool, "a boolean")

    def _visit_dual_nullschema(
        self, node: NullSchema, gold_value: Any, extracted_value: Any
    ) -> None:
        self._validate_null_and_set(node, gold_value, extracted_value)
