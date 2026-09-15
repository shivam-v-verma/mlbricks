import inspect
import types
from typing import Any, ClassVar, Self

import pydantic
from pydantic import BaseModel, ConfigDict, ValidationInfo, model_validator
from pydantic.fields import FieldInfo

__all__ = [
    "UNSET",
    "BuildTimeField",
    "ConfigFieldNameClashError",
    "ConfigValidationError",
    "Configurable",
    "ConfigurableBuildError",
    "ConfigurableError",
    "MissingCfgParameterError",
    "MissingConfigError",
    "UnexpectedBuildArgError",
    "UnownedConfigError",
    "build_validator",
]


class ConfigurableError(Exception):
    """Base for all Configurable semantic errors."""


class ConfigValidationError(ConfigurableError):
    """Raised when a pydantic validator fails during Config construction or build()."""

    def __init__(self, exc: pydantic.ValidationError) -> None:
        super().__init__(str(exc))
        self.pydantic_error = exc

    def errors(self) -> list[Any]:
        return self.pydantic_error.errors()

    def error_count(self) -> int:
        return self.pydantic_error.error_count()


class ConfigurableBuildError(ConfigurableError):
    """Base for all errors raised by build() due to API wiring mistakes."""


class UnownedConfigError(ConfigurableBuildError):
    """Config was defined outside a Configurable subclass (owner is None)."""


class ConfigFieldNameClashError(ConfigurableBuildError):
    """A Config field name collides with an owner __init__ parameter name."""


class UnexpectedBuildArgError(ConfigurableBuildError):
    """A kwarg passed to build() is not a Config field or owner __init__ parameter."""


class MissingConfigError(ConfigurableError):
    """Concrete Configurable subclass was defined without a nested Config class."""


class MissingCfgParameterError(ConfigurableError):
    """Configurable subclass defines __init__ without a 'cfg' parameter."""


def build_validator(func: types.FunctionType) -> Any:
    """Decorator for Config validators that only run during build().

    Wraps func as a pydantic model_validator gated on {"build": True} context.
    Use for checks that require fully-resolved field values (e.g. deferred fields).
    Raise ValueError inside the decorated method; build() translates to
    ConfigValidationError.

    Note: @wraps is intentionally not used here -- copying func's signature would
    hide the 'info' parameter from pydantic's introspection, causing it to not
    receive ValidationInfo at call time.
    """

    def wrapper(self: Any, info: ValidationInfo) -> Any:
        if info.context and info.context.get("build"):
            return func(self)
        return self

    # Copy name/qualname for debuggability without copying the call signature
    wrapper.__name__ = func.__name__
    wrapper.__qualname__ = func.__qualname__

    return model_validator(mode="after")(wrapper)


class _Unset:
    """Sentinel for a BuildTimeField that has not been resolved yet.

    Distinct from None so a BuildTimeField[T] can hold a T that is itself
    None. Never constructed outside this module.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSET"


UNSET = _Unset()


# Config field annotation for build-time-only, non-serializable values.
#
# Never required at Config construction, never emitted by
# resolve_config.to_dict(), and required at build() time unless the field is
# given its own default (same precedence as any pydantic field). Declare the
# "= UNSET" default explicitly -- it is not implicit.
#
# Example:
#   class Trainer(Configurable):
#       class Config(Configurable.Config["Trainer"]):
#           dataloader: BuildTimeField[DataLoader] = UNSET
#
#       def __init__(self, cfg: "Trainer.Config") -> None:
#           self.loader = cfg.dataloader  # plain DataLoader, no unwrap
#
#   cfg = Trainer.Config()
#   trainer = cfg.build(dataloader=my_loader)
type BuildTimeField[T] = T | _Unset


def is_build_time_field(field: FieldInfo) -> bool:
    """Return True if field's class-declared default is the BuildTimeField sentinel.

    Checks the declared default (``FieldInfo.default``), not the current
    instance value, so this stays correct after build() resolves a real value
    on a validated Config instance.

    Used by build()'s automatic unresolved-field check and by
    resolve_config.to_dict() to skip these fields during serialization.
    """
    return field.default is UNSET


class Configurable:
    """Mixin for classes that are constructed from a nested Config pydantic model.

    Subclasses define a nested ``Config`` that inherits ``Configurable.Config``.
    The config holds all construction parameters; calling ``Config.build()``
    validates and instantiates the owner.

    Always list ``Configurable`` last when combining with other base classes so
    that the other class (e.g. ``nn.Module``) retains MRO priority::

        class MyModel(nn.Module, Configurable): ...  # correct
        class MyModel(Configurable, nn.Module): ...  # wrong

    Example:
        >>> from typing import Self
        >>> from torch import nn
        >>> from pydantic import model_validator
        >>> from mlbricks.configurable import Configurable, build_validator
        >>>
        >>> class MLP(nn.Module, Configurable):
        ...     class Config(Configurable.Config["MLP"]):
        ...         input_dim: int | None = None
        ...         hidden_dim: int = 256
        ...
        ...         @build_validator
        ...         def check_input_dim(self) -> Self:
        ...             if self.input_dim is None:
        ...                 raise ValueError("input_dim required at build time")
        ...             return self
        ...
        ...         @model_validator(mode="after")
        ...         def check_dims(self) -> Self:
        ...             if self.hidden_dim <= 0:
        ...                 raise ValueError("hidden_dim must be positive")
        ...             return self
        ...
        ...     def __init__(self, cfg: "MLP.Config") -> None:
        ...         super().__init__()
        ...         self.fc = nn.Linear(cfg.input_dim, cfg.output_dim)
        >>>
        >>> cfg = MLP.Config(hidden_dim=128)
        >>> mlp = cfg.build(input_dim=64)
    """

    class Config[T: "Configurable"](BaseModel):
        """Base config for all Configurable subclasses.

        Subclass this (subscripted with the owner type) inside each Configurable
        subclass. Use @model_validator(mode="after") for cross-field invariants
        that run at construction and build time. Use @build_validator for checks
        that require fully-resolved values (e.g. deferred fields typed T | None).

        ``owner`` is wired automatically by ``Configurable.__init_subclass__``;
        never set it manually.
        """

        model_config = ConfigDict(arbitrary_types_allowed=True)
        owner: ClassVar[type | None] = None

        def __init__(self, **data: Any) -> None:
            # Wrap pydantic's ValidationError so callers see a uniform exception
            # hierarchy. We deliberately do NOT attempt model_rebuild() here for
            # incomplete models: pydantic's frame-walking heuristic for locally-
            # scoped types breaks when our __init__ adds an extra frame. Instead,
            # test files that define field types inside functions must NOT use
            # `from __future__ import annotations` -- use string literals for
            # forward refs where needed so pydantic resolves types at class-
            # definition time.
            try:
                super().__init__(**data)
            except pydantic.ValidationError as exc:
                raise ConfigValidationError(exc) from exc

        @build_validator
        def _check_build_time_fields(self) -> Self:
            unresolved = [
                name
                for name, field in type(self).model_fields.items()
                if is_build_time_field(field) and getattr(self, name) is UNSET
            ]
            if unresolved:
                raise ValueError(f"BuildTimeField(s) required at build(): {unresolved}")
            return self

        def build(self, **kwargs: Any) -> T:
            """Construct the owning class from this config.

            Args:
                **kwargs: Field overrides, deferred field values, or owner
                    __init__ parameters beyond cfg. Unrecognised keys raise
                    UnexpectedBuildArgError.

            Returns:
                A new instance of the owning class.

            Raises:
                UnownedConfigError: If Config was not defined inside a
                    Configurable subclass.
                ConfigFieldNameClashError: If a field name clashes with an owner
                    __init__ parameter.
                UnexpectedBuildArgError: If an unrecognised kwarg is passed.
                ConfigValidationError: If any validator raises.

            Example:
                >>> cfg = MLP.Config(hidden_dim=128)
                >>> mlp = cfg.build(input_dim=64)
                >>> mlp2 = cfg.build(input_dim=64, hidden_dim=64)
            """
            if self.owner is None:
                raise UnownedConfigError(
                    "Config has no owner -- define it inside a Configurable subclass."
                )

            # Collect extra params accepted by owner.__init__ (beyond self / cfg)
            owner_params = {
                name
                for name, p in inspect.signature(self.owner.__init__).parameters.items()
                if name not in ("self", "cfg")
                and p.kind
                not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                )
            }

            # Guard: field names must not clash with owner __init__ params
            field_names = set(type(self).model_fields)
            name_conflicts = field_names & owner_params
            if name_conflicts:
                raise ConfigFieldNameClashError(
                    f"{type(self).__name__} has field name(s) that clash with "
                    f"{self.owner.__name__}.__init__ parameters: {name_conflicts}"
                )

            # Separate config overrides from owner __init__ kwargs
            config_overrides = {
                k: kwargs.pop(k) for k in list(kwargs) if k in field_names
            }
            owner_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in owner_params}

            if kwargs:
                raise UnexpectedBuildArgError(
                    f"{type(self).__name__}.build() got unexpected kwargs:"
                    f" {set(kwargs)}"
                )

            # getattr over model_fields avoids a serialization roundtrip;
            # nested Config instances stay as their actual types
            merged = {
                name: getattr(self, name) for name in field_names
            } | config_overrides

            try:
                validated = type(self).model_validate(merged, context={"build": True})
            except pydantic.ValidationError as exc:
                raise ConfigValidationError(exc) from exc

            return self.owner(cfg=validated, **owner_kwargs)

    @classmethod
    def validate_config(cls, cfg: object) -> None:
        """Guard that ``cfg`` is the correct Config type for this class.

        Args:
            cfg: The config object to check.

        Raises:
            ConfigurableError: If ``cfg`` is not an instance of ``cls.Config``.
        """
        if not isinstance(cfg, cls.Config):
            raise ConfigurableError(
                f"Expected {cls.Config.__qualname__}, got {type(cfg).__qualname__}"
            )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        if "Config" in cls.__dict__:
            config_cls = cls.__dict__["Config"]
            if isinstance(config_cls, type) and issubclass(
                config_cls, Configurable.Config
            ):
                # Auto-wire owner so build() knows what to instantiate
                config_cls.owner = cls
        elif not inspect.isabstract(cls) and not any(
            "Config" in base.__dict__
            and isinstance(base.__dict__["Config"], type)
            and issubclass(base.__dict__["Config"], Configurable.Config)
            and not inspect.isabstract(base)
            and base is not Configurable
            for base in cls.__mro__[1:]
        ):
            raise MissingConfigError(
                f"{cls.__name__} is a concrete Configurable subclass but defines no"
                " nested Config. Add"
                f" 'class Config(Configurable.Config[\"{cls.__name__}\"])'"
                " (or a subclass of a parent's Config)."
            )

        # If this class defines its own __init__, verify it accepts cfg
        if "__init__" in cls.__dict__:
            params = inspect.signature(cls.__init__).parameters
            if "cfg" not in params:
                raise MissingCfgParameterError(
                    f"{cls.__name__}.__init__ must accept a 'cfg' parameter"
                    " so that build() can pass the validated config."
                )
