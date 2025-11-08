import inspect
from typing import Any

import pydantic
from langchain_classic.agents import AgentExecutor
from langchain_classic.chains.llm import LLMChain
from langchain_core.runnables import RunnableBinding
from openai import NOT_GIVEN

from typing_extensions import TypeGuard

def to_strict_json_schema(
    model: type[pydantic.BaseModel] | pydantic.TypeAdapter[Any],
) -> dict[str, Any]:
    """NOTE: Similer to the function from OpenAI Library to convert basemodel to json schema"""

    if inspect.isclass(model) and is_basemodel_type(model):
        schema = model.model_json_schema()
    elif isinstance(model, pydantic.TypeAdapter):
        schema = model.json_schema()
    else:
        raise TypeError(
            f"Non BaseModel types are only supported with Pydantic v2 - {model}"
        )

    return _ensure_strict_json_schema(schema, path=(), root=schema)


def _ensure_strict_json_schema(
    json_schema: object,
    *,
    path: tuple[str, ...],
    root: dict[str, object],
) -> dict[str, Any]:
    """Mutates the given JSON schema to ensure it conforms to the `strict` standard
    that the API expects.
    """
    if not is_dict(json_schema):
        raise TypeError(f"Expected {json_schema} to be a dictionary; path={path}")

    defs = json_schema.get("$defs")
    if is_dict(defs):
        for def_name, def_schema in defs.items():
            _ensure_strict_json_schema(
                def_schema, path=(*path, "$defs", def_name), root=root
            )

    definitions = json_schema.get("definitions")
    if is_dict(definitions):
        for definition_name, definition_schema in definitions.items():
            _ensure_strict_json_schema(
                definition_schema,
                path=(*path, "definitions", definition_name),
                root=root,
            )

    typ = json_schema.get("type")
    if typ == "object" and "additionalProperties" not in json_schema:
        json_schema["additionalProperties"] = False

    # object types
    # { 'type': 'object', 'properties': { 'a':  {...} } }
    properties = json_schema.get("properties")
    if is_dict(properties):
        json_schema["required"] = [prop for prop in properties.keys()]
        json_schema["properties"] = {
            key: _ensure_strict_json_schema(
                prop_schema, path=(*path, "properties", key), root=root
            )
            for key, prop_schema in properties.items()
        }

    # arrays
    # { 'type': 'array', 'items': {...} }
    items = json_schema.get("items")
    if is_dict(items):
        json_schema["items"] = _ensure_strict_json_schema(
            items, path=(*path, "items"), root=root
        )

    # unions
    any_of = json_schema.get("anyOf")
    if is_list(any_of):
        json_schema["anyOf"] = [
            _ensure_strict_json_schema(
                variant, path=(*path, "anyOf", str(i)), root=root
            )
            for i, variant in enumerate(any_of)
        ]

    # intersections
    all_of = json_schema.get("allOf")
    if is_list(all_of):
        if len(all_of) == 1:
            json_schema.update(
                _ensure_strict_json_schema(
                    all_of[0], path=(*path, "allOf", "0"), root=root
                )
            )
            json_schema.pop("allOf")
        else:
            json_schema["allOf"] = [
                _ensure_strict_json_schema(
                    entry, path=(*path, "allOf", str(i)), root=root
                )
                for i, entry in enumerate(all_of)
            ]

    # strip `None` defaults as there's no meaningful distinction here
    # the schema will still be `nullable` and the model will default
    # to using `None` anyway
    if json_schema.get("default", NOT_GIVEN) is None:
        json_schema.pop("default")

    # we can't use `$ref`s if there are also other properties defined, e.g.
    # `{"$ref": "...", "description": "my description"}`
    #
    # so we unravel the ref
    # `{"type": "string", "description": "my description"}`
    ref = json_schema.get("$ref")
    if ref and has_more_than_n_keys(json_schema, 1):
        assert isinstance(ref, str), f"Received non-string $ref - {ref}"

        resolved = resolve_ref(root=root, ref=ref)
        if not is_dict(resolved):
            raise ValueError(
                f"Expected `$ref: {ref}` to resolved to a dictionary but got {resolved}"
            )

        # properties from the json schema take priority over the ones on the `$ref`
        json_schema.update({**resolved, **json_schema})
        json_schema.pop("$ref")
        # Since the schema expanded from `$ref` might not have `additionalProperties: false` applied,
        # we call `_ensure_strict_json_schema` again to fix the inlined schema and ensure it's valid.
        return _ensure_strict_json_schema(json_schema, path=path, root=root)

    return json_schema


def is_basemodel_type(typ: type) -> TypeGuard[type[pydantic.BaseModel]]:
    if not inspect.isclass(typ):
        return False
    return issubclass(typ, pydantic.BaseModel)

def resolve_ref(*, root: dict[str, object], ref: str) -> object:
    if not ref.startswith("#/"):
        raise ValueError(f"Unexpected $ref format {ref!r}; Does not start with #/")

    path = ref[2:].split("/")
    resolved = root
    for key in path:
        value = resolved[key]
        assert is_dict(
            value
        ), f"encountered non-dictionary entry while resolving {ref} - {resolved}"
        resolved = value

    return resolved

def is_dict(obj: object) -> TypeGuard[dict[str, object]]:
    return isinstance(obj, dict)

def is_list(obj: object) -> TypeGuard[list[object]]:
    return isinstance(obj, list)

def has_more_than_n_keys(obj: dict[str, object], n: int) -> bool:
    i = 0
    for _ in obj.keys():
        i += 1
        if i > n:
            return True
    return False
def _ensure_llm_and_extra_body(chain):
    """
    Get the actual LLM object from whatever was passed (LLMChain, RunnableBinding,
    AgentExecutor) and make sure `extra_body` is a dict.
    Returns (llm, extra_body_dict).
    """
    llm = None

    # 1) plain LLMChain
    if isinstance(chain, LLMChain):
        llm = chain.llm

    # 2) AgentExecutor
    elif isinstance(chain, AgentExecutor):
        # adjust if your agent shape is different
        llm = chain.agent.runnable.middle[1].bound

    # 3) RunnableBinding (when you did .with_config(...))
    elif isinstance(chain, RunnableBinding):
        llm = chain.llm

    else:
        raise TypeError(f"Cannot extract LLM from chain of type {type(chain)}")

    # now guarantee extra_body is a dict
    extra_body = getattr(llm, "extra_body", None)
    if extra_body is None:
        extra_body = {}
        setattr(llm, "extra_body", extra_body)

    return llm, extra_body


def forbid_extra(schema: Any) -> Any:
    """
    Recursively walk a JSON Schema and, for every object, set
    additionalProperties = False if it's not present.
    """
    if isinstance(schema, dict):
        # if this is an object, force additionalProperties: false
        if schema.get("type") == "object":
            schema.setdefault("additionalProperties", False)

        # recurse into dict values
        for key, value in schema.items():
            schema[key] = forbid_extra(value)

    elif isinstance(schema, list):
        # recurse into list items
        return [forbid_extra(item) for item in schema]

    return schema