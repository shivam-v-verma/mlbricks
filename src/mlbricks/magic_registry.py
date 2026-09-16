import importlib
import inspect
import typing
from collections.abc import Iterable
from typing import Any, cast

from pydantic import ConfigDict, create_model

from mlbricks.configurable import Configurable

__all__ = [
    "MagicRegistryError",
    "MagicRegistryImportError",
    "MagicRegistryUnknownFieldError",
    "magic_wrap",
]


class MagicRegistryError(ValueError):
    """Base for magic-registry resolution errors."""


class MagicRegistryImportError(MagicRegistryError):
    """Raised when a dotted path cannot be imported or resolved to a class."""


class MagicRegistryUnknownFieldError(MagicRegistryError):
    """Raised when a field key has no matching __init__ parameter on the
    target class and the target has no **kwargs catch-all."""


def _import_dotted(path: str) -> type:
    """Import a dotted path to a class, trying the longest module prefix first.

    Args:
        path: A dotted path, e.g. "torch.optim.Adam".

    Returns:
        The resolved class.

    Raises:
        MagicRegistryImportError: If no prefix of `path` imports as a module,
            an attribute in the remaining chain is missing, or the resolved
            object is not a class.
    """
    parts = path.split(".")

    for split in range(len(parts) - 1, 0, -1):
        module_path = ".".join(parts[:split])
        try:
            obj: Any = importlib.import_module(module_path)
        except ImportError:
            continue

        try:
            for attr in parts[split:]:
                obj = getattr(obj, attr)
        except AttributeError as exc:
            raise MagicRegistryImportError(
                f"'{path}' does not resolve -- '{module_path}' has no"
                f" attribute chain {'.'.join(parts[split:])!r}"
            ) from exc

        if not isinstance(obj, type):
            raise MagicRegistryImportError(f"'{path}' does not resolve to a class")

        return obj

    raise MagicRegistryImportError(f"cannot import any module prefix of '{path}'")


class _MagicConfigBase(Configurable.Config):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


def magic_wrap(path: str, field_keys: Iterable[str]) -> type[Configurable]:
    """Build a Configurable wrapper subclassing the class at `path`.

    The wrapper's Config declares exactly `field_keys` as fields, typed from
    the target's own __init__ signature where possible. Anything not in
    field_keys is never a Config field -- pass it at build() time instead,
    same as any other Configurable's owner kwargs (e.g. build(params=...)).

    Stamped with `_magic_path_` (kept separate from Registry's
    `_registry_path_` -- curated and magic entries are distinct namespaces,
    see docs/registry.md) so resolve_config.to_dict() can round-trip it as
    `_magic_registry_`.

    No caching: each call builds a fresh class. importlib already memoizes
    the underlying import; a small pydantic model + dynamic subclass is
    cheap enough to build per config node.

    Args:
        path: Dotted import path to the target class, e.g. "torch.optim.Adam".
        field_keys: Names of the config node's sibling keys -- the fields
            this Config should declare.

    Returns:
        A new Configurable subclass of the target class.

    Raises:
        MagicRegistryImportError: If `path` can't be imported / isn't a class.
        MagicRegistryUnknownFieldError: If a field key has no matching
            __init__ parameter and the target has no **kwargs catch-all.
    """
    target_cls = _import_dotted(path)
    field_keys = list(field_keys)
    field_set = set(field_keys)
    fields = field_specs(target_cls, field_keys)

    config_cls = create_model(
        f"Magic{target_cls.__name__}Config",
        __base__=_MagicConfigBase,
        **fields,
    )  # ty: ignore[no-matching-overload]

    def __init__(self: Any, cfg: Any, **kwargs: Any) -> None:
        field_values = {name: getattr(cfg, name) for name in type(cfg).model_fields}
        target_cls.__init__(self, **field_values, **kwargs)

    # build() infers legal extra ("owner") kwargs from __init__'s *signature*,
    # not from the **kwargs catch-all above -- expose target_cls's remaining
    # real params (everything not already a Config field) so e.g.
    # build(params=...) is recognized instead of rejected as unexpected.
    try:
        target_sig = inspect.signature(target_cls.__init__)
        remaining = [
            p
            for name, p in target_sig.parameters.items()
            if name not in field_set and name != "self"
        ]
        __init__.__signature__ = inspect.Signature(  # ty: ignore[unresolved-attribute]
            [
                inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                inspect.Parameter("cfg", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                *remaining,
            ]
        )
    except (TypeError, ValueError):
        pass  # unintrospectable __init__ -- fall back to bare **kwargs signature

    wrapper_cls = type(
        f"Magic{target_cls.__name__}",
        (target_cls, Configurable),
        {"Config": config_cls, "__init__": __init__, "_magic_path_": path},
    )
    return cast(type[Configurable], wrapper_cls)


def field_specs(
    target_cls: type, field_keys: Iterable[str]
) -> dict[str, tuple[Any, Any]]:
    """Build create_model field specs for exactly the given keys.

    Types come from target_cls.__init__'s own annotations when resolvable;
    falls back to Any per-field, or wholesale if the signature itself can't
    be introspected (some C-extension __init__s raise on inspect.signature).

    Args:
        target_cls: The class whose __init__ is introspected.
        field_keys: Field names to build specs for -- exactly the keys
            present in the config node, not the full signature.

    Returns:
        Dict mapping field name to a (type, default) tuple for
        pydantic.create_model, all required (default is `...`).

    Raises:
        MagicRegistryUnknownFieldError: If a key has no matching parameter
            on target_cls.__init__ and there's no **kwargs catch-all.
    """
    try:
        sig = inspect.signature(target_cls.__init__)
    except (TypeError, ValueError):
        return dict.fromkeys(field_keys, (Any, ...))

    try:
        hints = typing.get_type_hints(target_cls.__init__)
    except Exception:
        hints = {}

    has_var_keyword = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )

    specs: dict[str, tuple[Any, Any]] = {}
    for key in field_keys:
        param = sig.parameters.get(key)

        if param is None:
            if has_var_keyword:
                specs[key] = (Any, ...)
                continue
            raise MagicRegistryUnknownFieldError(
                f"{target_cls.__module__}.{target_cls.__qualname__} has no"
                f" '{key}' parameter"
            )

        annotation = hints.get(key, param.annotation)
        if annotation is inspect.Parameter.empty or isinstance(annotation, str):
            annotation = Any
        specs[key] = (annotation, ...)

    return specs
