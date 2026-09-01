from collections.abc import Callable
from typing import Any, cast

__all__ = [
    "DuplicateRegistrationError",
    "InvalidRegistryNameError",
    "Registry",
    "RegistryKeyError",
]


class DuplicateRegistrationError(ValueError):
    """Raised when a name is already registered in a Registry."""


class InvalidRegistryNameError(ValueError):
    """Raised when a registry name contains a period."""


class RegistryKeyError(KeyError):
    """Raised when a dotted path cannot be resolved in a Registry."""

    def __str__(self) -> str:
        return self.args[0] if self.args else ""


class Registry[T]:
    """Generic nestable registry mapping string names to uninitialized classes.

    Use ``subgroup()`` to create named sub-registries and ``register()`` to
    decorate classes into a registry level. Resolve dotted paths with ``get()``.

    Example:
        >>> reg: Registry[object] = Registry()
        >>> models = reg.subgroup("models")
        >>> @models.register("mlp")
        ... class MLP: pass
        >>> reg.get("models.mlp") is MLP
        True
    """

    def __init__(self) -> None:
        self._entries: dict[str, type[Any] | Registry[Any]] = {}

    def subgroup[S](self, name: str) -> "Registry[S]":
        """Create a named child registry and store it under ``name``.

        Args:
            name: Single-segment name for this subgroup. Must not contain a
                period.

        Returns:
            A new ``Registry[S]`` stored under ``name`` in this registry.

        Raises:
            InvalidRegistryNameError: If ``name`` contains a period.
            DuplicateRegistrationError: If ``name`` is already registered.
        """
        _validate_name(name)

        if name in self._entries:
            raise DuplicateRegistrationError(f"'{name}' is already registered")

        child: Registry[S] = Registry()
        self._entries[name] = child

        return child

    def register[S](self, name: str) -> Callable[[type[S]], type[S]]:
        """Return a decorator that registers a class under ``name``.

        Args:
            name: Single-segment name. Must not contain a period.

        Returns:
            A decorator that stores the class and returns it unchanged.

        Raises:
            InvalidRegistryNameError: If ``name`` contains a period.
            DuplicateRegistrationError: If ``name`` is already registered in
                this registry (raised at factory-call time, before the
                decorated class is defined).
        """
        _validate_name(name)

        if name in self._entries:
            raise DuplicateRegistrationError(f"'{name}' is already registered")

        def decorator(cls: type[S]) -> type[S]:
            self._entries[name] = cls
            return cls

        return decorator

    def get(self, path: str) -> type[T]:
        """Resolve a dotted path to a registered class.

        Args:
            path: Dot-separated path, e.g. ``"models.gnn.gat"``.

        Returns:
            The uninitialized class registered at ``path``.

        Raises:
            RegistryKeyError: If any segment is missing, an intermediate segment
                is a leaf class, or the final segment is a subgroup.
        """
        parts = path.split(".")
        node: Registry[Any] = self

        # traverse subgroups
        for part in parts[:-1]:
            entry = node._entries.get(part)

            if entry is None:
                raise RegistryKeyError(
                    f"'{part}' not found in registry (path: '{path}')"
                )

            if not isinstance(entry, Registry):
                raise RegistryKeyError(
                    f"'{part}' is not a subgroup -- cannot traverse into a leaf"
                    f" class (path: '{path}')"
                )

            node = entry

        leaf_name = parts[-1]
        result = node._entries.get(leaf_name)

        if result is None:
            raise RegistryKeyError(
                f"'{leaf_name}' not found in registry (path: '{path}')"
            )
        if isinstance(result, Registry):
            raise RegistryKeyError(
                f"'{leaf_name}' is a subgroup, not a registered class (path: '{path}')"
            )

        return cast(type[T], result)

    def __contains__(self, path: str) -> bool:
        try:
            self.get(path)
            return True

        except RegistryKeyError:
            return False

    def leaf_paths(self, prefix: str = "") -> set[str]:
        """Return dotted paths for all leaf classes in this registry.

        Recursively traverses subgroups, building dotted paths like
        ``"models.gnn.gat"``. Subgroups themselves are not included.

        Args:
            prefix: Dot-separated path prefix accumulated by parent calls.
                Pass the default (empty string) at the call site.

        Returns:
            Set of dotted path strings for every registered leaf class.

        Example:
            >>> reg: Registry[object] = Registry()
            >>> models = reg.subgroup("models")
            >>> @models.register("mlp")
            ... class MLP: pass
            >>> reg.leaf_paths() == {"models.mlp"}
            True
        """
        paths: set[str] = set()

        for name, entry in self._entries.items():
            full = f"{prefix}.{name}" if prefix else name

            if isinstance(entry, Registry):
                paths |= entry.leaf_paths(full)
            else:
                paths.add(full)

        return paths

    def path_of(self, cls: type) -> str | None:
        """Return the dotted registry path for ``cls``, or ``None`` if not registered.

        Performs a depth-first tree-walk. Useful for serializing a registered
        class back to its declarative ``_registry_`` key.

        Args:
            cls: The class to look up.

        Returns:
            Dotted path string (e.g. ``"models.gnn.gat"``), or ``None`` if
            ``cls`` is not a leaf in this registry.

        Example:
            >>> reg: Registry[object] = Registry()
            >>> models = reg.subgroup("models")
            >>> @models.register("mlp")
            ... class MLP: pass
            >>> reg.path_of(MLP)
            'models.mlp'
        """
        return self._find_path(cls, prefix="")

    def _find_path(self, cls: type, prefix: str) -> str | None:
        """Recursively search for a class in the registry tree.

        Args:
            cls: The class to look up.
            prefix: Dot-separated path prefix accumulated by parent calls.

        Returns:
            Dotted path string if found, or ``None`` if not found.
        """
        for name, entry in self._entries.items():
            full_name = f"{prefix}.{name}" if prefix else name

            if entry is cls:
                # found answer
                return full_name

            if isinstance(entry, Registry):
                result = entry._find_path(cls, full_name)
                if result is not None:
                    # pass solution up
                    return result

        return None


def _validate_name(name: str) -> None:
    if not name or "." in name:
        raise InvalidRegistryNameError(
            f"Registry name {name!r} must be a non-empty string with no periods"
            " -- periods are reserved as path separators"
        )
