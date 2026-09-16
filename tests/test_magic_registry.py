from typing import Any

import pytest
import torch

import mlbricks
from mlbricks import Configurable, ConfigValidationError
from mlbricks.magic_registry import (
    MagicRegistryImportError,
    MagicRegistryUnknownFieldError,
    field_specs,
    magic_wrap,
)


class _Typed:
    def __init__(self, lr: float = 0.1, tag: str = "x", **extra: Any) -> None:
        pass


class _Untyped:
    def __init__(self, lr, tag="x") -> None:  # no annotations at all
        pass


class _NoKwargsCatchall:
    def __init__(self, lr: float = 0.1) -> None:
        pass


class FakeOptimizer:
    """Stand-in for an external class like torch.optim.Adam."""

    def __init__(self, params: Any, lr: float = 0.1, tag: str = "x") -> None:
        self.params = params
        self.lr = lr
        self.tag = tag


def test_magic_wrap_builds_real_external_class() -> None:
    """End-to-end against an actual third-party class (torch.optim.SGD),
    not a local stand-in -- exercises real union-typed annotations and a
    non-field constructor arg (params) flowing through as an owner kwarg."""
    cls = magic_wrap("torch.optim.SGD", ["lr"])
    assert cls.__name__ == "MagicSGD"
    assert issubclass(cls, torch.optim.SGD)

    cfg = cls.Config(lr=0.05)
    optimizer = cfg.build(params=[torch.nn.Parameter(torch.zeros(1))])

    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.param_groups[0]["lr"] == 0.05


def test_magic_wrap_raises_on_bad_module() -> None:
    with pytest.raises(MagicRegistryImportError):
        magic_wrap("nonexistent_module_xyz.Thing", [])


def test_magic_wrap_raises_on_bad_attribute() -> None:
    with pytest.raises(MagicRegistryImportError):
        magic_wrap("json.NotARealClass", [])


def test_magic_wrap_raises_when_target_is_not_a_class() -> None:
    with pytest.raises(MagicRegistryImportError):
        magic_wrap("json.dumps", [])  # a function, not a class


def test_field_specs_resolves_annotated_type() -> None:
    specs = field_specs(_Typed, ["lr"])
    assert specs == {"lr": (float, ...)}


def test_field_specs_falls_back_to_any_when_unannotated() -> None:
    specs = field_specs(_Untyped, ["lr"])
    assert specs == {"lr": (Any, ...)}


def test_field_specs_unknown_key_with_kwargs_catchall_is_any() -> None:
    specs = field_specs(_Typed, ["made_up_name"])
    assert specs == {"made_up_name": (Any, ...)}


def test_field_specs_unknown_key_without_catchall_raises() -> None:
    with pytest.raises(MagicRegistryUnknownFieldError):
        field_specs(_NoKwargsCatchall, ["made_up_name"])


def test_field_specs_unintrospectable_init_falls_back_to_any() -> None:
    # int.__init__ raises ValueError from inspect.signature()
    specs = field_specs(int, ["whatever"])
    assert specs == {"whatever": (Any, ...)}


def test_magic_wrap_is_configurable_subclass() -> None:
    cls = magic_wrap(f"{__name__}.FakeOptimizer", ["lr"])
    assert issubclass(cls, Configurable)
    assert issubclass(cls, FakeOptimizer)


def test_magic_wrap_owner_is_wired() -> None:
    cls = magic_wrap(f"{__name__}.FakeOptimizer", ["lr"])
    assert cls.Config.owner is cls


def test_magic_wrap_config_has_declared_fields_only() -> None:
    cls = magic_wrap(f"{__name__}.FakeOptimizer", ["lr"])
    assert set(cls.Config.model_fields) == {"lr"}


def test_magic_wrap_build_constructs_real_instance() -> None:
    cls = magic_wrap(f"{__name__}.FakeOptimizer", ["lr"])
    cfg = cls.Config(lr=0.5)
    obj = cfg.build(params=[1, 2, 3])
    assert isinstance(obj, FakeOptimizer)
    assert obj.lr == 0.5
    assert obj.params == [1, 2, 3]
    assert obj.tag == "x"  # not a Config field -- FakeOptimizer's own default


def test_magic_wrap_build_forwards_non_field_kwarg() -> None:
    """A real __init__ param that's not a Config field flows through build()
    as an owner kwarg -- same mechanism any other Configurable uses."""
    cls = magic_wrap(f"{__name__}.FakeOptimizer", ["lr"])
    cfg = cls.Config(lr=0.5)
    obj = cfg.build(params=[], tag="custom")
    assert obj.tag == "custom"


def test_magic_wrap_type_error_on_construction() -> None:
    cls = magic_wrap(f"{__name__}.FakeOptimizer", ["lr"])
    with pytest.raises(ConfigValidationError):
        cls.Config(lr="not-a-float")


def test_magic_wrap_unknown_config_key_rejected() -> None:
    """extra='forbid' -- a key not declared in field_keys at wrap time is
    rejected at Config construction, not silently accepted."""
    cls = magic_wrap(f"{__name__}.FakeOptimizer", ["lr"])
    with pytest.raises(ConfigValidationError):
        cls.Config(lr=0.1, made_up=1)


def test_magic_wrap_no_extra_fields_still_builds() -> None:
    cls = magic_wrap(f"{__name__}.FakeOptimizer", [])
    obj = cls.Config().build(params=[], lr=0.9)
    assert obj.lr == 0.9


def test_magic_registry_errors_exported_from_top_level_package() -> None:
    assert mlbricks.MagicRegistryError is not None
    assert mlbricks.MagicRegistryImportError is not None
    assert mlbricks.MagicRegistryUnknownFieldError is not None
    assert not hasattr(mlbricks, "magic_wrap")
