import dataclasses
from dataclasses import dataclass, fields
from typing import Any

from pydantic import BaseModel

from mlbricks.configurable import Configurable
from mlbricks.registry import Registry

__all__ = [
    "ConfigInstantiationError",
    "ConfigSerializationError",
    "ParsedNode",
    "RegistryParseError",
    "instantiate",
    "parse",
    "resolve",
    "to_dict",
]


class RegistryParseError(ValueError):
    """Raised when a _registry_ node is structurally malformed."""


class ConfigInstantiationError(RuntimeError):
    """Raised when a class constructor fails during instantiate()."""


class ConfigSerializationError(ValueError):
    """Raised when a value cannot be serialized to a declarative dict."""


@dataclass
class ParsedNode:
    """Intermediate representation of a resolved _registry_ node.

    Produced by parse(); consumed by instantiate(). Carries the original
    registry key string for error messages and debugging.
    """

    key: str
    cls: type
    kwargs: dict[str, Any]


def parse(
    cfg: Any,
    registry: Registry[Any],
) -> Any:
    """DFS-walk cfg, replacing every _registry_ node with a ParsedNode.

    Args:
        cfg: Any Python value -- dict, list, scalar, or a dict with a
            top-level ``_registry_`` key. The latter is converted to a
            ParsedNode directly.
        registry: Registry used to resolve _registry_ keys.

    Returns:
        The input with every _registry_ node replaced by a ParsedNode.
        A top-level _registry_ dict is itself replaced by a ParsedNode.

    Raises:
        RegistryParseError: If a _registry_ value is not a string.
        RegistryKeyError: If a _registry_ key is not found in the registry.

    Example:
        >>> result = parse({"_registry_": "models.mlp", "dim": 64}, registry=reg)
        >>> isinstance(result, ParsedNode)
        True
    """
    if isinstance(cfg, dict):
        return _parse_dict(cfg, registry)
    if isinstance(cfg, list):
        return [parse(item, registry) for item in cfg]
    return cfg


def _parse_dict(
    node: dict[str, Any],
    registry: Registry[Any],
) -> dict[str, Any] | ParsedNode:
    # plain dict -- recurse into values without creating a ParsedNode
    if "_registry_" not in node:
        return {k: parse(v, registry) for k, v in node.items()}

    key = node["_registry_"]
    if not isinstance(key, str):
        raise RegistryParseError(
            f"_registry_ value must be a string, got {type(key).__name__!r}"
        )

    # RegistryKeyError propagates as-is
    cls = registry.get(key)

    # recurse into sibling kwargs before building the node
    kwargs = {k: parse(v, registry) for k, v in node.items() if k != "_registry_"}

    return ParsedNode(key=key, cls=cls, kwargs=kwargs)


def instantiate(parsed: Any) -> Any:
    """Construct all ParsedNodes in a parsed value into Python objects.

    Args:
        parsed: Output of parse(). May be a ParsedNode, dict, list, or scalar.

    Returns:
        The input with every ParsedNode replaced by its constructed object.
        A top-level ParsedNode is itself replaced by the constructed object.

    Raises:
        ConfigInstantiationError: If a constructor raises, wrapping the original
            exception via __cause__.
    """
    if isinstance(parsed, ParsedNode):
        return _instantiate_node(parsed)
    if isinstance(parsed, list):
        return [instantiate(item) for item in parsed]
    if isinstance(parsed, dict):
        return {k: instantiate(v) for k, v in parsed.items()}
    return parsed


def _instantiate_node(node: ParsedNode) -> Any:
    """Instantiate a single ParsedNode by resolving kwargs and calling constructor."""
    # Resolve any nested ParsedNodes in kwargs first (bottom-up)
    resolved_kwargs = {k: instantiate(v) for k, v in node.kwargs.items()}

    try:
        if issubclass(node.cls, Configurable):
            return node.cls.Config(**resolved_kwargs)
        return node.cls(**resolved_kwargs)
    except Exception as exc:
        raise ConfigInstantiationError(f"Failed to instantiate '{node.key}'") from exc


def resolve(
    cfg: Any,
    registry: Registry[Any],
) -> Any:
    """Parse and instantiate a config value in one step.

    A strict inverse of ``to_dict()``: if ``cfg`` is the output of
    ``to_dict(some_config, registry)``, then
    ``resolve(cfg, registry) == some_config``.

    Args:
        cfg: Any Python value. A dict with a top-level ``_registry_`` key
            is instantiated and returned directly. A plain dict (no
            top-level ``_registry_``) is returned as a dict with all
            nested ``_registry_`` nodes resolved. Scalars and lists are
            handled recursively.
        registry: Registry to resolve _registry_ keys against.

    Returns:
        The input with all _registry_ nodes replaced by constructed objects.
        A top-level _registry_ dict yields the object itself, not a dict.

    Example:
        >>> cfg = resolve(to_dict(MLP.Config(hidden_dim=64), reg), reg)
        >>> isinstance(cfg, MLP.Config)
        True
    """
    return instantiate(parse(cfg, registry))


def to_dict(value: Any, registry: Registry[Any]) -> Any:
    """Convert a value to a plain Python dict mirroring the declarative config format.

    Inverse of ``resolve()``. Recursively converts ``Configurable.Config``
    and registered plain dataclass instances to ``{"_registry_": path,
    **fields}`` dicts, recurses into plain dicts and lists, and passes
    scalars through unchanged.

    Args:
        value: A ``Configurable.Config``, registered plain dataclass
            instance, dict, list, scalar, or ``None``.
        registry: Registry used to resolve class paths.

    Returns:
        A plain Python value with all ``Configurable.Config`` and registered
        plain dataclass instances replaced by declarative dicts containing
        ``_registry_`` keys.

    Raises:
        ConfigSerializationError: If any value in the tree cannot be serialized --
            an unregistered owner or class, or an unsupported type.

    Example:
        >>> result = to_dict(MLP.Config(hidden_dim=64), registry=reg)
        >>> result == {"_registry_": "models.mlp", "hidden_dim": 64}
        True
    """
    if isinstance(value, Configurable.Config):
        return _config_to_dict(value, registry)

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _plain_dataclass_to_dict(value, registry)

    if isinstance(value, BaseModel) and not isinstance(value, Configurable.Config):
        # Iterate fields and recurse rather than calling model_dump(): model_dump()
        # recursively serializes nested Configurable.Config instances to plain dicts,
        # losing their type identity and making it impossible to emit _registry_ keys.
        return {
            name: to_dict(getattr(value, name), registry)
            for name in type(value).model_fields
        }

    if isinstance(value, dict):
        return {k: to_dict(v, registry) for k, v in value.items()}

    if isinstance(value, list):
        return [to_dict(item, registry) for item in value]

    if isinstance(value, (int, float, str, bool)) or value is None:
        return value

    raise ConfigSerializationError(
        f"cannot serialize value of type {type(value).__name__!r} --"
        " only Configurable.Config, dict, list, and scalars are supported"
    )


def _config_to_dict(
    cfg: Configurable.Config, registry: Registry[Any]
) -> dict[str, Any]:
    owner = cfg.owner

    if owner is None:
        raise ConfigSerializationError(
            "Config has no owner -- define it inside a Configurable subclass"
        )

    path = registry.path_of(owner)
    if path is None:
        raise ConfigSerializationError(
            f"{owner.__name__} is not registered in the provided registry"
            " -- cannot serialize"
        )

    result: dict[str, Any] = {"_registry_": path}

    for name in type(cfg).model_fields:
        result[name] = to_dict(getattr(cfg, name), registry)

    return result


def _plain_dataclass_to_dict(value: Any, registry: Registry[Any]) -> dict[str, Any]:
    """Serialize a plain registered dataclass instance to a declarative dict.

    Args:
        value: A plain dataclass instance (not a Configurable.Config).
        registry: Registry used for reverse path lookup.

    Returns:
        Dict with a ``_registry_`` key and all field values recursively
        serialized.

    Raises:
        ConfigSerializationError: If the dataclass type is not registered.
    """
    cls = type(value)
    path = registry.path_of(cls)
    if path is None:
        raise ConfigSerializationError(
            f"{cls.__name__} is not registered -- cannot serialize"
        )
    result: dict[str, Any] = {"_registry_": path}
    for fld in fields(value):
        result[fld.name] = to_dict(getattr(value, fld.name), registry)
    return result
