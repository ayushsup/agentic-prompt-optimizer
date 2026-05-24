from abc import ABC, abstractmethod
from types import NoneType
from typing import Any, Dict, List, Literal, Optional, Union

import pydantic
from typing_extensions import Annotated, Protocol


PrimitiveSchema = Union[
    "StringSchema",
    "NumberSchema",
    "IntegerSchema",
    "BooleanSchema",
    "NullSchema",
]
NestedSchema = Union[
    "ObjectSchema",
    "ArraySchema",
]
TypedSchema = Annotated[
    Union[
        "PrimitiveSchema",
        "NestedSchema",
    ],
    pydantic.Field(discriminator="type"),
]
CompositeSchema = Union["AnyOfSchema"]
AnySchema = Union[TypedSchema, "ReferenceSchema", "CompositeSchema"]
Schema = Union[AnySchema, "RootSchema"]

PRIMITIVE_SCHEMA_TYPES = ["string", "number", "integer", "boolean", "null"]


class VisitorProtocol(Protocol):
    def visit(self, node: "Schema") -> Any: ...


class AsyncVisitorProtocol(Protocol):
    async def visit(self, node: "Schema") -> Any: ...


class BaseSchema(pydantic.BaseModel, ABC):
    """Base model for common schema attributes.

    Instances are immutable. Use `create_copy_with_updated_fields` to create modified copies.
    """

    # Metadata fields
    title: Optional[str] = None
    description: Optional[str] = None
    default: Optional[Any] = None
    example: Optional[Any] = pydantic.Field(default=None, alias="examples")
    nullable: Optional[bool] = None

    # Evaluation config field (excluded from export by default)
    evaluation_config: Optional[Union[str, Any, Dict[str, Any]]] = pydantic.Field(
        default=None, exclude=True
    )

    model_config = pydantic.ConfigDict(
        populate_by_name=True, extra="forbid", frozen=True
    )

    # Private attributes
    _root_schema: Optional["RootSchema"] = pydantic.PrivateAttr(default=None)
    _gold_value: Optional[Any] = pydantic.PrivateAttr(default=None)
    _extracted_value: Optional[Any] = pydantic.PrivateAttr(default=None)
    _evaluation_result: Optional[Any] = pydantic.PrivateAttr(default=None)

    def create_copy_with_updated_fields(
        self, update: Optional[Dict[str, Any]], fast: bool = True
    ) -> "Schema":
        """Return a new copy with the given field updates. Private attributes are preserved."""
        if not update:
            return self
        if fast:
            for key in update.keys():
                if key not in self.__class__.model_fields:
                    raise ValueError(
                        f"Invalid key: {key} for schema: {self.__class__.__name__}"
                    )
            new_instance = self.model_copy(update=update, deep=False)
        else:
            data = self.model_dump(exclude_unset=True)
            data.update(update)
            new_instance = self.__class__.model_validate(data)

        self._clone_private_attributes(new_instance)
        return new_instance

    @abstractmethod
    def get_children(self) -> List[AnySchema]:
        """Returns a list of immediate children of the schema."""
        raise NotImplementedError

    def export_to_dict(
        self,
        strip_metadata: bool = False,
    ) -> Dict:
        """Export schema to dictionary."""
        exclude_fields = set()
        if strip_metadata and isinstance(self, RootSchema):
            exclude_fields = {"schema_", "id_"}

        result: Dict[str, Any] = self.model_dump(
            by_alias=True, exclude_none=True, exclude=exclude_fields
        )
        return _recursively_strip_none_from_dict(result)

    def accept(self, visitor: VisitorProtocol) -> Any:
        return visitor.visit(self)

    async def accept_async(self, visitor: AsyncVisitorProtocol) -> Any:
        return await visitor.visit(self)

    def is_primitive(self) -> bool:
        return self.get_type() in PRIMITIVE_SCHEMA_TYPES

    def get_type(self) -> str:
        if "type" in self.__class__.model_fields.keys():
            return self.type
        return None

    def get_root_schema(self) -> Union["RootSchema", NoneType]:
        return self._root_schema

    def set_root_schema(self, root_schema: "RootSchema") -> None:
        self._root_schema = root_schema

    def get_metadata_summary(self) -> Dict[str, Any]:
        """Return metadata for LLM prompts."""
        summary = {}
        if self.get_type():
            summary["schema_type"] = self.get_type()
        if self.description:
            summary["description"] = self.description
        if self.example is not None:
            summary["example"] = self.example
        if self.nullable is not None:
            summary["nullable"] = self.nullable
        if self.title:
            summary["title"] = self.title
        return summary

    # Evaluation value accessors
    def set_gold_value(self, gold_value: Any) -> None:
        self._gold_value = gold_value

    def get_gold_value(self) -> Optional[Any]:
        return self._gold_value

    def set_extracted_value(self, extracted_value: Any) -> None:
        self._extracted_value = extracted_value

    def get_extracted_value(self) -> Optional[Any]:
        return self._extracted_value

    def set_evaluation_result(self, evaluation_result: Any) -> None:
        self._evaluation_result = evaluation_result

    def get_evaluation_result(self) -> Optional[Any]:
        return self._evaluation_result

    def _clone_private_attributes(self, target: "BaseSchema") -> None:
        private_attrs = getattr(self, "__pydantic_private__", None)
        if private_attrs is None:
            return
        object.__setattr__(target, "__pydantic_private__", private_attrs.copy())


class BasePrimitiveSchema(BaseSchema):
    def get_children(self) -> List[AnySchema]:
        return []


class StringSchema(BasePrimitiveSchema):
    type: Literal["string"]
    min_length: Optional[int] = pydantic.Field(default=None, alias="minLength")
    max_length: Optional[int] = pydantic.Field(default=None, alias="maxLength")
    pattern: Optional[str] = None
    enum: Optional[List[str]] = None
    format: Optional[str] = None


class IntegerSchema(BasePrimitiveSchema):
    type: Literal["integer"]
    minimum: Optional[int] = None
    maximum: Optional[int] = None
    format: Optional[Literal["int32", "int64"]] = None
    enum: Optional[List[int]] = None


class NumberSchema(BasePrimitiveSchema):
    type: Literal["number"]
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    enum: Optional[List[Union[float, int]]] = None
    format: Optional[Literal["float", "double"]] = None


class BooleanSchema(BasePrimitiveSchema):
    type: Literal["boolean"]


class NullSchema(BasePrimitiveSchema):
    type: Literal["null"]


class ArraySchema(BaseSchema):
    type: Literal["array"]
    items: AnySchema
    min_items: Optional[int] = pydantic.Field(default=None, alias="minItems")
    max_items: Optional[int] = pydantic.Field(default=None, alias="maxItems")

    def get_children(self) -> List[AnySchema]:
        return [self.items]


class ObjectSchema(BaseSchema):
    type: Literal["object"]
    properties: Optional[Dict[str, AnySchema]] = None
    required: Optional[List[str]] = None
    min_properties: Optional[int] = pydantic.Field(default=None, alias="minProperties")
    max_properties: Optional[int] = pydantic.Field(default=None, alias="maxProperties")
    additional_properties: Optional[Union[bool, AnySchema]] = pydantic.Field(
        default=None, alias="additionalProperties"
    )

    def get_children(self) -> List[AnySchema]:
        result = []
        if self.properties:
            result.extend(self.properties.values())
        if self.additional_properties and not isinstance(
            self.additional_properties, bool
        ):
            result.append(self.additional_properties)
        return result


class ReferenceSchema(BaseSchema):
    """Represents a JSON schema reference."""

    ref: str = pydantic.Field(..., alias="$ref")
    summary: Optional[str] = None
    _def_schema: Optional[AnySchema] = pydantic.PrivateAttr(default=None)

    def get_children(self) -> List[AnySchema]:
        return []

    def get_def_schema(self) -> AnySchema:
        if self._def_schema is None:
            root_schema = self.get_root_schema()
            if root_schema is None:
                raise ValueError("Root schema not found for reference schema.")
            all_defs = root_schema.get_defs() or {}
            ref_name = self.get_ref_name()
            if ref_name not in all_defs:
                raise ValueError(
                    f"Definition {ref_name} not found in $defs! "
                    f"Available: {sorted(all_defs.keys())}"
                )
            self._def_schema = all_defs[ref_name]
        return self._def_schema

    def get_ref_name(self) -> str:
        if self.ref.startswith("#/defs/"):
            return self.ref[7:]
        elif self.ref.startswith("#/$defs/"):
            return self.ref[8:]
        else:
            raise ValueError(f"Invalid $ref format: {self.ref}")


class AnyOfSchema(BaseSchema):
    """A value is valid if it validates against any of the subschemas."""

    any_of: List[AnySchema] = pydantic.Field(..., alias="anyOf")

    def is_nullable_anyof(self) -> bool:
        return (
            len(self.any_of) == 2
            and sum(isinstance(schema, NullSchema) for schema in self.any_of) == 1
        )

    def get_non_null_schema(self) -> Optional[AnySchema]:
        if not self.is_nullable_anyof():
            return None
        return (
            self.any_of[1] if isinstance(self.any_of[0], NullSchema) else self.any_of[0]
        )

    def contains_null_branch(self) -> bool:
        return any(isinstance(schema, NullSchema) for schema in self.any_of)

    def get_non_null_schemas(self) -> List[AnySchema]:
        return [schema for schema in self.any_of if not isinstance(schema, NullSchema)]

    def get_children(self) -> List[AnySchema]:
        return self.any_of


class RootSchema(ObjectSchema):
    defs: Optional[Dict[str, AnySchema]] = pydantic.Field(default=None, alias="$defs")
    schema_: Optional[str] = pydantic.Field(
        default=None,
        alias="$schema",
        validation_alias=pydantic.AliasChoices("$schema", "schema"),
    )
    id_: Optional[str] = pydantic.Field(
        default=None,
        alias="$id",
        validation_alias=pydantic.AliasChoices("$id", "id"),
    )

    def set_root_schema_for_children(self) -> None:
        """Sets the root schema for all children."""
        visited = set()

        def recursively_set_root_schema(node: AnySchema) -> None:
            if id(node) in visited:
                return
            visited.add(id(node))
            node.set_root_schema(self)
            for child in node.get_children():
                recursively_set_root_schema(child)

        if self.defs:
            for _, def_schema in self.defs.items():
                def_schema.set_root_schema(self)
                recursively_set_root_schema(def_schema)
        recursively_set_root_schema(self)

    def get_defs(self) -> Dict[str, AnySchema]:
        return self.defs

    @pydantic.model_validator(mode="after")
    def _validate_root_schema_is_not_nullable(self) -> "RootSchema":
        if self.nullable:
            raise ValueError("RootSchema cannot have nullable=True")
        return self

    def get_object_schema_from_root_schema(self) -> ObjectSchema:
        """Get the object schema from the root schema."""
        object_schema_fields = set(ObjectSchema.model_fields.keys())
        field_data = {}
        for field in object_schema_fields:
            if field in self.__pydantic_fields_set__:
                field_data[field] = getattr(self, field)

        obj_schema = ObjectSchema.model_construct(**field_data)
        self._clone_private_attributes(obj_schema)
        return obj_schema


def _recursively_strip_none_from_dict(d: dict) -> dict:
    result = {}
    for k, v in d.items():
        if v is not None:
            if isinstance(v, dict):
                result[k] = _recursively_strip_none_from_dict(v)
            elif isinstance(v, list):
                new_lst = []
                for ls_item in v:
                    if isinstance(ls_item, dict):
                        new_lst.append(_recursively_strip_none_from_dict(ls_item))
                    elif ls_item is not None:
                        new_lst.append(ls_item)
                result[k] = new_lst
            else:
                result[k] = v
    return result


# Rebuild models for forward references
StringSchema.model_rebuild()
NumberSchema.model_rebuild()
IntegerSchema.model_rebuild()
BooleanSchema.model_rebuild()
NullSchema.model_rebuild()
ArraySchema.model_rebuild()
ObjectSchema.model_rebuild()
ReferenceSchema.model_rebuild()
AnyOfSchema.model_rebuild()
RootSchema.model_rebuild()
