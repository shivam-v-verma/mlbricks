from abc import ABC, abstractmethod
from typing import Self

import pytest
from pydantic import Field, model_validator

from mlbricks import (
    UNSET,
    BuildTimeField,
    ConfigFieldNameClashError,
    Configurable,
    ConfigurableError,
    ConfigValidationError,
    MissingCfgParameterError,
    MissingConfigError,
    UnexpectedBuildArgError,
    UnownedConfigError,
    build_validator,
)
from mlbricks.configurable import is_build_time_field


class Plain(Configurable):
    """No validators, no deferred fields."""

    class Config(Configurable.Config["Plain"]):
        hidden_dim: int = 64
        dropout: float = 0.1

    def __init__(self, cfg: "Plain.Config") -> None:
        self.hidden_dim = cfg.hidden_dim
        self.dropout = cfg.dropout


class WithValidate(Configurable):
    """model_validator raises on bad values."""

    class Config(Configurable.Config["WithValidate"]):
        hidden_dim: int = 64

        @model_validator(mode="after")
        def check_hidden_dim(self) -> Self:
            if self.hidden_dim <= 0:
                raise ValueError("hidden_dim must be positive")
            return self

    def __init__(self, cfg: "WithValidate.Config") -> None:
        self.hidden_dim = cfg.hidden_dim


class WithBuildValidator(Configurable):
    """build_validator only fires during build()."""

    class Config(Configurable.Config["WithBuildValidator"]):
        input_dim: int | None = None

        @build_validator
        def check_input_dim(self) -> Self:
            if self.input_dim is None:
                raise ValueError("input_dim required at build time")
            return self

    def __init__(self, cfg: "WithBuildValidator.Config") -> None:
        self.input_dim = cfg.input_dim


class WithExtraInit(Configurable):
    """Owner __init__ accepts an extra param beyond cfg."""

    class Config(Configurable.Config["WithExtraInit"]):
        hidden_dim: int = 64

    def __init__(self, cfg: "WithExtraInit.Config", extra: str) -> None:
        self.hidden_dim = cfg.hidden_dim
        self.extra = extra


class WithBuildTimeField(Configurable):
    """dataloader is build-time only; hidden_dim is a normal field."""

    class Config(Configurable.Config["WithBuildTimeField"]):
        hidden_dim: int = 64
        dataloader: BuildTimeField[str] = UNSET

    def __init__(self, cfg: "WithBuildTimeField.Config") -> None:
        self.hidden_dim = cfg.hidden_dim
        self.dataloader = cfg.dataloader


# ---------------------------------------------------------------------------
# BuildTimeField: annotation shape
# ---------------------------------------------------------------------------


def test_build_time_field_defaults_to_unset() -> None:
    cfg = WithBuildTimeField.Config()
    assert cfg.dataloader is UNSET


def test_build_time_field_construction_does_not_require_value() -> None:
    WithBuildTimeField.Config()  # must not raise


def test_is_build_time_field_true_for_marked_field() -> None:
    field = WithBuildTimeField.Config.model_fields["dataloader"]
    assert is_build_time_field(field)


def test_is_build_time_field_false_for_plain_field() -> None:
    field = WithBuildTimeField.Config.model_fields["hidden_dim"]
    assert not is_build_time_field(field)


def test_build_time_field_accepts_explicit_value_at_construction() -> None:
    cfg = WithBuildTimeField.Config(dataloader="real-loader")
    assert cfg.dataloader == "real-loader"


# ---------------------------------------------------------------------------
# BuildTimeField: enforcement at build()
# ---------------------------------------------------------------------------


def test_build_time_field_raises_when_unresolved_at_build() -> None:
    with pytest.raises(ConfigValidationError):
        WithBuildTimeField.Config().build()


def test_build_time_field_resolves_when_supplied() -> None:
    obj = WithBuildTimeField.Config().build(dataloader="loader")
    assert obj.dataloader == "loader"


def test_build_time_field_override_does_not_bleed() -> None:
    cfg = WithBuildTimeField.Config()
    cfg.build(dataloader="loader")
    with pytest.raises(ConfigValidationError):
        cfg.build()


def test_build_time_field_with_own_default_is_optional() -> None:
    class WithDefault(Configurable):
        class Config(Configurable.Config["WithDefault"]):
            dataloader: BuildTimeField[str] = "fallback"

        def __init__(self, cfg: "WithDefault.Config") -> None:
            self.dataloader = cfg.dataloader

    obj = WithDefault.Config().build()
    assert obj.dataloader == "fallback"


def test_build_time_field_error_names_the_field() -> None:
    with pytest.raises(ConfigValidationError) as exc_info:
        WithBuildTimeField.Config().build()
    assert "dataloader" in str(exc_info.value)


# ---------------------------------------------------------------------------
# __init_subclass__ / owner wiring
# ---------------------------------------------------------------------------


def test_owner_auto_wired() -> None:
    assert Plain.Config.owner is Plain


def test_owner_none_on_base() -> None:
    assert Configurable.Config.owner is None


# ---------------------------------------------------------------------------
# build(): basic construction
# ---------------------------------------------------------------------------


def test_build_returns_owner_instance() -> None:
    obj = Plain.Config().build()
    assert isinstance(obj, Plain)


def test_build_applies_design_time_defaults() -> None:
    obj = Plain.Config().build()
    assert obj.hidden_dim == 64
    assert obj.dropout == 0.1


def test_build_design_time_override() -> None:
    obj = Plain.Config().build(hidden_dim=32)
    assert obj.hidden_dim == 32


def test_build_does_not_mutate_original_config() -> None:
    cfg = Plain.Config(hidden_dim=64)
    cfg.build(hidden_dim=32)
    assert cfg.hidden_dim == 64


# ---------------------------------------------------------------------------
# build(): @model_validator
# ---------------------------------------------------------------------------


def test_model_validator_fires_at_construction() -> None:
    with pytest.raises(ConfigValidationError):
        WithValidate.Config(hidden_dim=-1)


def test_build_calls_model_validator() -> None:
    with pytest.raises(ConfigValidationError):
        WithValidate.Config(hidden_dim=64).build(hidden_dim=-1)


def test_construction_error_raises_config_validation_error() -> None:
    with pytest.raises(ConfigValidationError):
        WithValidate.Config(hidden_dim=-1)


def test_construction_error_has_errors_list() -> None:
    with pytest.raises(ConfigValidationError) as exc_info:
        WithValidate.Config(hidden_dim=-1)
    errs = exc_info.value.errors()
    assert len(errs) == 1
    assert "hidden_dim must be positive" in errs[0]["msg"]


def test_construction_error_has_error_count() -> None:
    with pytest.raises(ConfigValidationError) as exc_info:
        WithValidate.Config(hidden_dim=-1)
    assert exc_info.value.error_count() == 1


def test_build_error_has_errors_list() -> None:
    with pytest.raises(ConfigValidationError) as exc_info:
        WithValidate.Config(hidden_dim=64).build(hidden_dim=-1)
    errs = exc_info.value.errors()
    assert len(errs) == 1
    assert "hidden_dim must be positive" in errs[0]["msg"]


def test_build_error_has_error_count() -> None:
    with pytest.raises(ConfigValidationError) as exc_info:
        WithValidate.Config(hidden_dim=64).build(hidden_dim=-1)
    assert exc_info.value.error_count() == 1


def test_config_validation_error_str_is_human_readable() -> None:
    with pytest.raises(ConfigValidationError) as exc_info:
        WithValidate.Config(hidden_dim=-1)
    assert "hidden_dim" in str(exc_info.value)


# ---------------------------------------------------------------------------
# build(): @build_validator
# ---------------------------------------------------------------------------


def test_build_validator_not_called_at_construction() -> None:
    WithBuildValidator.Config()  # must not raise


def test_build_validator_fires_at_build_time() -> None:
    with pytest.raises(ConfigValidationError):
        WithBuildValidator.Config().build()


def test_build_validator_passes_when_supplied() -> None:
    obj = WithBuildValidator.Config().build(input_dim=64)
    assert obj.input_dim == 64


def test_build_validator_override_does_not_bleed() -> None:
    cfg = WithBuildValidator.Config()
    cfg.build(input_dim=16)
    with pytest.raises(ConfigValidationError):
        cfg.build()


# ---------------------------------------------------------------------------
# build(): extra owner __init__ params
# ---------------------------------------------------------------------------


def test_build_extra_owner_param() -> None:
    obj = WithExtraInit.Config().build(extra="hello")
    assert obj.extra == "hello"


# ---------------------------------------------------------------------------
# build(): error cases
# ---------------------------------------------------------------------------


def test_build_unexpected_kwarg_raises() -> None:
    with pytest.raises(UnexpectedBuildArgError):
        Plain.Config().build(nonexistent=True)


def test_build_name_conflict_raises() -> None:
    class BadModel(Configurable):
        class Config(Configurable.Config["BadModel"]):
            extra: str = "x"

        def __init__(self, cfg: "BadModel.Config", extra: str) -> None:
            pass

    with pytest.raises(ConfigFieldNameClashError):
        BadModel.Config().build(extra="y")


def test_build_time_field_name_conflict_raises() -> None:
    class BadModel(Configurable):
        class Config(Configurable.Config["BadModel"]):
            extra: BuildTimeField[str] = UNSET

        def __init__(self, cfg: "BadModel.Config", extra: str) -> None:
            pass

    with pytest.raises(ConfigFieldNameClashError):
        BadModel.Config().build(extra="y")


def test_build_no_owner_raises() -> None:
    class OrphanConfig(Configurable.Config):
        val: int = 0

    with pytest.raises(UnownedConfigError):
        OrphanConfig().build()


# ---------------------------------------------------------------------------
# validate_config()
# ---------------------------------------------------------------------------


def test_validate_config_correct_type() -> None:
    Plain.validate_config(Plain.Config())


def test_validate_config_wrong_config_type_raises() -> None:
    with pytest.raises(ConfigurableError):
        Plain.validate_config(WithBuildValidator.Config())


def test_validate_config_non_config_raises() -> None:
    with pytest.raises(ConfigurableError):
        Plain.validate_config("not a config")


# ---------------------------------------------------------------------------
# __init_subclass__ hardening
# ---------------------------------------------------------------------------


def test_concrete_configurable_without_config_raises() -> None:
    with pytest.raises(MissingConfigError):

        class NoConfig(Configurable):
            def __init__(self, cfg: object) -> None:
                pass


def test_abstract_configurable_without_config_is_allowed() -> None:
    # Should not raise -- abstract classes are exempt
    class AbstractNoConfig(Configurable, ABC):
        @abstractmethod
        def process(self) -> None: ...


def test_concrete_configurable_with_config_is_allowed() -> None:
    # Should not raise
    class WithConfig(Configurable):
        class Config(Configurable.Config["WithConfig"]):
            dim: int = 32

        def __init__(self, cfg: "WithConfig.Config") -> None:
            pass


def test_init_without_cfg_parameter_raises() -> None:
    with pytest.raises(MissingCfgParameterError):

        class NoCfgParam(Configurable):
            class Config(Configurable.Config["NoCfgParam"]):
                dim: int = 32

            def __init__(self, dim: int) -> None:
                pass


def test_init_with_cfg_parameter_is_allowed() -> None:
    # Should not raise
    class WithCfgParam(Configurable):
        class Config(Configurable.Config["WithCfgParam"]):
            dim: int = 32

        def __init__(self, cfg: "WithCfgParam.Config") -> None:
            pass


def test_no_custom_init_is_allowed() -> None:
    # Should not raise -- inherits __init__, no check needed
    class InheritedInit(Configurable):
        class Config(Configurable.Config["InheritedInit"]):
            dim: int = 32


# ---------------------------------------------------------------------------
# BuildTimeField: nested Configurable.Config composition
# ---------------------------------------------------------------------------


class _InnerWithBuildTimeField(Configurable):
    class Config(Configurable.Config["_InnerWithBuildTimeField"]):
        dataloader: BuildTimeField[str] = UNSET

    def __init__(self, cfg: "_InnerWithBuildTimeField.Config") -> None:
        self.dataloader = cfg.dataloader


class _OuterWithNestedBuildTimeField(Configurable):
    """Outer's __init__ builds the inner Config itself, mirroring the
    documented nested-config composition pattern (each level builds its own
    children)."""

    class Config(Configurable.Config["_OuterWithNestedBuildTimeField"]):
        inner: "_InnerWithBuildTimeField.Config" = Field(
            default_factory=lambda: _InnerWithBuildTimeField.Config()
        )

    def __init__(self, cfg: "_OuterWithNestedBuildTimeField.Config") -> None:
        self.inner = cfg.inner.build()


def test_build_time_field_in_nested_config_raises_if_unresolved() -> None:
    with pytest.raises(ConfigValidationError):
        _OuterWithNestedBuildTimeField.Config().build()


def test_build_time_field_in_nested_config_resolves() -> None:
    cfg = _OuterWithNestedBuildTimeField.Config()
    cfg.inner.dataloader = (
        "loader"  # Config is mutable; no build() override path exists here
    )
    obj = cfg.build()
    assert obj.inner.dataloader == "loader"
