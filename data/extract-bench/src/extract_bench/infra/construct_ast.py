from typing import Dict, Union

from .nodes import (
    AnyOfSchema,
    AnySchema,
    ArraySchema,
    BooleanSchema,
    IntegerSchema,
    NullSchema,
    NumberSchema,
    PRIMITIVE_SCHEMA_TYPES,
    ReferenceSchema,
    RootSchema,
    StringSchema,
)


def construct_ast(schema: Dict) -> Union[AnySchema, RootSchema]:
    """Build AST from JSON schema dict. No validation."""
    schema = _convert_array_types_to_anyof(schema)
    result = _build_node(schema)
    if isinstance(result, RootSchema):
        result.set_root_schema_for_children()
    return result


def _convert_array_types_to_anyof(schema: Dict) -> Dict:
    """Convert type arrays to anyOf (e.g., type: ["string", "null"])."""
    if not isinstance(schema, dict):
        return schema

    schema = dict(schema)  # shallow copy to avoid modifying input

    if "type" in schema and isinstance(schema["type"], list):
        if not all(t in PRIMITIVE_SCHEMA_TYPES for t in schema["type"]):
            raise ValueError(
                f"Array types must be primitives. Got types: {schema['type']}"
            )
        schema["anyOf"] = [{"type": t} for t in schema["type"]]
        del schema["type"]

    if "items" in schema:
        schema["items"] = _convert_array_types_to_anyof(schema["items"])
    elif "properties" in schema:
        schema["properties"] = {
            k: _convert_array_types_to_anyof(v) for k, v in schema["properties"].items()
        }
    elif "additionalProperties" in schema and isinstance(
        schema["additionalProperties"], dict
    ):
        schema["additionalProperties"] = _convert_array_types_to_anyof(
            schema["additionalProperties"]
        )

    if "anyOf" in schema:
        schema["anyOf"] = [
            _convert_array_types_to_anyof(item) for item in schema["anyOf"]
        ]

    if "$defs" in schema:
        schema["$defs"] = {
            k: _convert_array_types_to_anyof(v) for k, v in schema["$defs"].items()
        }
    elif "defs" in schema:
        schema["defs"] = {
            k: _convert_array_types_to_anyof(v) for k, v in schema["defs"].items()
        }

    return schema


def _build_node(schema: Dict) -> Union[AnySchema, RootSchema]:
    """Dispatch to appropriate schema class based on schema content."""
    if "$ref" in schema:
        return ReferenceSchema(**schema)
    if "anyOf" in schema:
        return AnyOfSchema(**schema)

    if "type" not in schema:
        raise ValueError("Schema must have a 'type' field")

    schema_type = schema["type"]

    if schema_type == "object":
        return RootSchema(**schema)
    elif schema_type == "array":
        return ArraySchema(**schema)
    elif schema_type == "string":
        return StringSchema(**schema)
    elif schema_type == "number":
        return NumberSchema(**schema)
    elif schema_type == "integer":
        return IntegerSchema(**schema)
    elif schema_type == "boolean":
        return BooleanSchema(**schema)
    elif schema_type == "null":
        return NullSchema(**schema)
    else:
        valid_types = [
            "object",
            "array",
            "string",
            "number",
            "integer",
            "boolean",
            "null",
        ]
        raise ValueError(
            f"Invalid schema type: {schema_type}. Must be one of: {valid_types}"
        )
