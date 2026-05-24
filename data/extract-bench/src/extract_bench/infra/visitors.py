import asyncio
from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

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


class SchemaVisitor(ABC):
    """Base class for synchronous schema visitors."""

    def visit(self, node: Schema) -> Any:
        method_name = f"visit_{node.__class__.__name__.lower()}"
        visitor_method = getattr(self, method_name, None)
        if visitor_method is None:
            raise NotImplementedError(f"{method_name} is not implemented")
        return visitor_method(node)

    @abstractmethod
    def visit_rootschema(self, node: RootSchema) -> Any:
        raise NotImplementedError

    @abstractmethod
    def visit_objectschema(self, node: ObjectSchema) -> Any:
        raise NotImplementedError

    @abstractmethod
    def visit_arrayschema(self, node: ArraySchema) -> Any:
        raise NotImplementedError

    @abstractmethod
    def visit_anyofschema(self, node: AnyOfSchema) -> Any:
        raise NotImplementedError

    def visit_stringschema(self, node: StringSchema) -> Any:
        return self.generic_leaf_visit(node)

    def visit_numberschema(self, node: NumberSchema) -> Any:
        return self.generic_leaf_visit(node)

    def visit_integerschema(self, node: IntegerSchema) -> Any:
        return self.generic_leaf_visit(node)

    def visit_booleanschema(self, node: BooleanSchema) -> Any:
        return self.generic_leaf_visit(node)

    def visit_nullschema(self, node: NullSchema) -> Any:
        return self.generic_leaf_visit(node)

    def visit_referenceschema(self, node: ReferenceSchema) -> Any:
        return self.generic_leaf_visit(node)

    def generic_leaf_visit(self, node: Schema) -> Any:
        return node

    def pre_visit_children_hook(self, node: Schema) -> None:
        pass

    def post_visit_children_hook(self, node: Schema) -> None:
        pass


class AnalyzerVisitor(SchemaVisitor):
    """Read-only traversal without modification."""

    def visit_rootschema(self, node: RootSchema) -> None:
        if node.defs:
            for _, sub_schema in node.defs.items():
                sub_schema.accept(self)
        self.visit_objectschema(node)

    def visit_objectschema(self, node: ObjectSchema) -> None:
        self.pre_visit_children_hook(node)
        if node.properties:
            for _, sub_schema in node.properties.items():
                sub_schema.accept(self)
        if node.additional_properties and not isinstance(
            node.additional_properties, bool
        ):
            node.additional_properties.accept(self)
        self.post_visit_children_hook(node)

    def visit_arrayschema(self, node: ArraySchema) -> None:
        self.pre_visit_children_hook(node)
        if node.items:
            node.items.accept(self)
        self.post_visit_children_hook(node)

    def visit_anyofschema(self, node: AnyOfSchema) -> None:
        self.pre_visit_children_hook(node)
        for schema in node.any_of:
            schema.accept(self)
        self.post_visit_children_hook(node)


class PathAnalyzerVisitor(AnalyzerVisitor):
    """Analyzer with path tracking. Access current path via `_path()`."""

    def __init__(self):
        super().__init__()
        self._stack = ["$"]
        self.paths = []

    def reset(self) -> None:
        self._stack = ["$"]
        self.paths = []

    def _push(self, seg: str) -> None:
        self._stack.append(seg)

    def _pop(self) -> None:
        self._stack.pop()

    def _path(self) -> str:
        return ".".join(self._stack)

    def generic_leaf_visit(self, node: Schema):
        self.paths.append(self._path())
        return node

    def visit_rootschema(self, node: RootSchema) -> None:
        self.paths.append(self._path())
        if node.defs:
            for name, sub in node.defs.items():
                self._push(f"$defs.{name}")
                sub.accept(self)
                self._pop()
        self.visit_objectschema(node)

    def visit_objectschema(self, node: ObjectSchema) -> None:
        self.pre_visit_children_hook(node)
        if node.properties:
            for key, sub in node.properties.items():
                self._push(f"properties.{key}")
                sub.accept(self)
                self._pop()
        if node.additional_properties and not isinstance(
            node.additional_properties, bool
        ):
            self._push("additionalProperties")
            node.additional_properties.accept(self)
            self._pop()
        self.post_visit_children_hook(node)

    def visit_arrayschema(self, node: ArraySchema) -> None:
        self.pre_visit_children_hook(node)
        if node.items:
            self._push("items")
            node.items.accept(self)
            self._pop()
        self.post_visit_children_hook(node)

    def visit_anyofschema(self, node: AnyOfSchema) -> None:
        self.pre_visit_children_hook(node)
        for i, sub in enumerate(node.any_of):
            self._push(f"anyOf[{i}]")
            sub.accept(self)
            self._pop()
        self.post_visit_children_hook(node)


class TransformerVisitor(SchemaVisitor):
    """Visitor that recursively rebuilds the tree with transformations."""

    def __init__(self):
        super().__init__()
        self._paths = []

    def _push(self, node: Schema) -> None:
        self._paths.append(node)

    def _pop(self) -> Schema:
        return self._paths.pop()

    def generic_leaf_visit(self, node: Schema) -> Any:
        return node.model_copy()

    def visit_rootschema(
        self, node: RootSchema, set_root_schema_for_children: bool = True
    ) -> RootSchema:
        update = {}
        if node.defs:
            new_defs = {}
            for key, sub_schema in node.defs.items():
                new_defs[key] = sub_schema.accept(self)
            update["defs"] = new_defs
        new_node = node.create_copy_with_updated_fields(update)
        obj_schema = node.get_object_schema_from_root_schema()
        transformed_object = self.visit_objectschema(obj_schema)
        object_fields = set(ObjectSchema.model_fields.keys())
        update_from_object = {
            k: getattr(transformed_object, k)
            for k in object_fields
            if getattr(transformed_object, k) is not None
        }
        final_node = new_node.create_copy_with_updated_fields(update_from_object)
        if set_root_schema_for_children:
            final_node.set_root_schema_for_children()
        return final_node

    def visit_objectschema(self, node: ObjectSchema) -> ObjectSchema:
        update = {}
        if node.properties:
            new_properties = {}
            for key, sub_schema in node.properties.items():
                new_properties[key] = sub_schema.accept(self)
            update["properties"] = new_properties
        if node.additional_properties is not None and not isinstance(
            node.additional_properties, bool
        ):
            update["additional_properties"] = node.additional_properties.accept(self)
        return node.create_copy_with_updated_fields(update)

    def visit_arrayschema(self, node: ArraySchema) -> ArraySchema:
        update = {}
        if node.items:
            update["items"] = node.items.accept(self)
        return node.create_copy_with_updated_fields(update)

    def visit_anyofschema(self, node: AnyOfSchema) -> AnyOfSchema:
        update = {}
        if node.any_of:
            update["any_of"] = [schema.accept(self) for schema in node.any_of]
        return node.create_copy_with_updated_fields(update)


# --- Async Visitors ---


class AsyncSchemaVisitor(ABC):
    """Base class for async schema visitors."""

    async def visit(self, node: Schema) -> Any:
        method_name = f"visit_{node.__class__.__name__.lower()}"
        visitor_method = getattr(self, method_name, None)
        if visitor_method is None:
            raise NotImplementedError(f"{method_name} is not implemented")
        return await visitor_method(node)

    @abstractmethod
    async def visit_rootschema(self, node: RootSchema) -> Any:
        raise NotImplementedError

    @abstractmethod
    async def visit_objectschema(self, node: ObjectSchema) -> Any:
        raise NotImplementedError

    @abstractmethod
    async def visit_arrayschema(self, node: ArraySchema) -> Any:
        raise NotImplementedError

    @abstractmethod
    async def visit_anyofschema(self, node: AnyOfSchema) -> Any:
        raise NotImplementedError

    async def visit_stringschema(self, node: StringSchema) -> Any:
        return await self.generic_leaf_visit(node)

    async def visit_numberschema(self, node: NumberSchema) -> Any:
        return await self.generic_leaf_visit(node)

    async def visit_integerschema(self, node: IntegerSchema) -> Any:
        return await self.generic_leaf_visit(node)

    async def visit_booleanschema(self, node: BooleanSchema) -> Any:
        return await self.generic_leaf_visit(node)

    async def visit_nullschema(self, node: NullSchema) -> Any:
        return await self.generic_leaf_visit(node)

    async def visit_referenceschema(self, node: ReferenceSchema) -> Any:
        return await self.generic_leaf_visit(node)

    async def generic_leaf_visit(self, node: Schema) -> Any:
        return node

    async def pre_visit_children_hook(self, node: Schema) -> None:
        pass

    async def post_visit_children_hook(self, node: Schema) -> None:
        pass


class AsyncAnalyzerVisitor(AsyncSchemaVisitor):
    """Async analyzer with optional parallel traversal."""

    def __init__(self, parallel_traversal: bool = True):
        self.parallel_traversal = parallel_traversal

    async def visit_rootschema(self, node: RootSchema) -> None:
        if node.defs:
            if self.parallel_traversal:
                await asyncio.gather(
                    *(
                        sub_schema.accept_async(self)
                        for sub_schema in node.defs.values()
                    )
                )
            else:
                for sub_schema in node.defs.values():
                    await sub_schema.accept_async(self)
        await self.visit_objectschema(node)

    async def visit_objectschema(self, node: ObjectSchema) -> None:
        await self.pre_visit_children_hook(node)
        if node.properties:
            if self.parallel_traversal:
                await asyncio.gather(
                    *(
                        sub_schema.accept_async(self)
                        for sub_schema in node.properties.values()
                    )
                )
            else:
                for sub_schema in node.properties.values():
                    await sub_schema.accept_async(self)
        if node.additional_properties and not isinstance(
            node.additional_properties, bool
        ):
            await node.additional_properties.accept_async(self)
        await self.post_visit_children_hook(node)

    async def visit_arrayschema(self, node: ArraySchema) -> None:
        await self.pre_visit_children_hook(node)
        if node.items:
            await node.items.accept_async(self)
        await self.post_visit_children_hook(node)

    async def visit_anyofschema(self, node: AnyOfSchema) -> None:
        await self.pre_visit_children_hook(node)
        if self.parallel_traversal:
            await asyncio.gather(*(schema.accept_async(self) for schema in node.any_of))
        else:
            for schema in node.any_of:
                await schema.accept_async(self)
        await self.post_visit_children_hook(node)


class AsyncPathAnalyzerVisitor(AsyncAnalyzerVisitor):
    """Async path-tracking analyzer using ContextVar for thread-safe path storage."""

    def __init__(self, parallel_traversal: bool = True):
        super().__init__(parallel_traversal)
        self._stack_var = ContextVar(f"async_path_stack_{id(self)}", default=("$",))
        self.paths = []
        self._stack_var.set(("$",))

    def reset(self) -> None:
        self.paths = []
        self._stack_var.set(("$",))

    def _path(self) -> str:
        return ".".join(self._stack_var.get())

    @contextmanager
    def _path_segment(self, seg: str):
        current = self._stack_var.get()
        token = self._stack_var.set(current + (seg,))
        try:
            yield
        finally:
            self._stack_var.reset(token)

    async def generic_leaf_visit(self, node: Schema):
        self.paths.append(self._path())
        return node

    async def visit_rootschema(self, node: RootSchema) -> None:
        self.paths.append(self._path())
        if node.defs:
            tasks = []
            for name, sub in node.defs.items():

                async def visit_def(name=name, sub=sub):
                    with self._path_segment(f"$defs.{name}"):
                        await sub.accept_async(self)

                if self.parallel_traversal:
                    tasks.append(visit_def())
                else:
                    await visit_def()

            if self.parallel_traversal and tasks:
                await asyncio.gather(*tasks)

        await self.visit_objectschema(node)

    async def visit_objectschema(self, node: ObjectSchema) -> None:
        await self.pre_visit_children_hook(node)

        if node.properties:
            tasks = []
            for key, sub in node.properties.items():

                async def visit_prop(key=key, sub=sub):
                    with self._path_segment(f"properties.{key}"):
                        await sub.accept_async(self)

                if self.parallel_traversal:
                    tasks.append(visit_prop())
                else:
                    await visit_prop()

            if self.parallel_traversal and tasks:
                await asyncio.gather(*tasks)

        if node.additional_properties and not isinstance(
            node.additional_properties, bool
        ):
            with self._path_segment("additionalProperties"):
                await node.additional_properties.accept_async(self)

        await self.post_visit_children_hook(node)

    async def visit_arrayschema(self, node: ArraySchema) -> None:
        await self.pre_visit_children_hook(node)
        if node.items:
            with self._path_segment("items"):
                await node.items.accept_async(self)
        await self.post_visit_children_hook(node)

    async def visit_anyofschema(self, node: AnyOfSchema) -> None:
        await self.pre_visit_children_hook(node)
        tasks = []
        for i, sub in enumerate(node.any_of):

            async def visit_anyof(i=i, sub=sub):
                with self._path_segment(f"anyOf[{i}]"):
                    await sub.accept_async(self)

            if self.parallel_traversal:
                tasks.append(visit_anyof())
            else:
                await visit_anyof()

        if self.parallel_traversal and tasks:
            await asyncio.gather(*tasks)

        await self.post_visit_children_hook(node)
