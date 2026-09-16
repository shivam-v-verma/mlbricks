from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, Field

from mlbricks import (
    UNSET,
    BuildTimeField,
    ConfigInstantiationError,
    ConfigSerializationError,
    Configurable,
    ConfigValidationError,
    Registry,
    RegistryKeyError,
    RegistryParseError,
    resolve,
)
from mlbricks.magic_registry import magic_wrap
from mlbricks.resolve_config import ParsedNode, instantiate, parse, to_dict

# ---------------------------------------------------------------------------
# Fixture classes -- defined at module level so tests can reference them
# in isinstance assertions. Registration happens in the fixture.
# ---------------------------------------------------------------------------


class Plain:
    def __init__(self, hidden_dim: int = 64) -> None:
        self.hidden_dim = hidden_dim


class Cfg(Configurable):
    class Config(Configurable.Config):
        hidden_dim: int = 128

    def __init__(self, cfg: "Cfg.Config") -> None:
        self.hidden_dim = cfg.hidden_dim


class Composite:
    """Accepts an arbitrary sub-object so nested instantiation can be tested."""

    def __init__(self, sub: Any = None, scale: float = 1.0) -> None:
        self.sub = sub
        self.scale = scale


class Outer(Configurable):
    """Configurable whose Config holds a nested Configurable.Config field."""

    class Config(Configurable.Config):
        inner: Cfg.Config = Field(default_factory=Cfg.Config)
        scale: float = 1.0

    def __init__(self, cfg: "Outer.Config") -> None:
        self.scale = cfg.scale


class WithList(Configurable):
    """Configurable whose Config holds a list of Configurable.Config fields."""

    class Config(Configurable.Config):
        items: list[Cfg.Config] = Field(default_factory=list)

    def __init__(self, cfg: "WithList.Config") -> None:
        pass


@dataclass(kw_only=True)
class PlainDC:
    """Plain (non-Configurable) dataclass for serializer tests."""

    width: int = 32
    label: str = "default"


class FakeExternal:
    """Stand-in for an unregistered third-party class, for _magic_registry_ tests."""

    def __init__(self, threshold: float = 0.5, label: str = "x") -> None:
        self.threshold = threshold
        self.label = label


@pytest.fixture
def reg() -> Registry[object]:
    """Fresh isolated registry per test -- never touches the global REGISTRY."""
    reg: Registry[object] = Registry()
    models = reg.subgroup("models")
    models.register("plain")(Plain)
    models.register("cfg")(Cfg)
    models.register("composite")(Composite)
    reg.register("outer")(Outer)
    reg.register("with_list")(WithList)
    reg.register("plain_dc")(PlainDC)
    return reg


# ---------------------------------------------------------------------------
# parse: flat registry node
# ---------------------------------------------------------------------------


def test_parse_flat_registry_node_returns_parsed_node(reg: Registry[object]) -> None:
    cfg = {"model": {"_registry_": "models.plain", "hidden_dim": 32}}
    result = parse(cfg, registry=reg)
    node = result["model"]
    assert isinstance(node, ParsedNode)
    assert node.key == "models.plain"
    assert node.cls is Plain
    assert node.kwargs == {"hidden_dim": 32}


def test_parse_preserves_registry_key_string(reg: Registry[object]) -> None:
    cfg = {"model": {"_registry_": "models.cfg", "hidden_dim": 64}}
    result = parse(cfg, registry=reg)
    assert result["model"].key == "models.cfg"


# ---------------------------------------------------------------------------
# parse: _magic_registry_ node
# ---------------------------------------------------------------------------


def test_parse_magic_registry_node_returns_parsed_node(reg: Registry[object]) -> None:
    cfg = {"_magic_registry_": f"{__name__}.FakeExternal", "threshold": 0.9}
    result = parse(cfg, registry=reg)
    assert isinstance(result, ParsedNode)
    assert result.key == f"{__name__}.FakeExternal"
    assert result.kwargs == {"threshold": 0.9}


def test_resolve_magic_registry_node_end_to_end(reg: Registry[object]) -> None:
    cfg = {"model": {"_magic_registry_": f"{__name__}.FakeExternal", "threshold": 0.9}}
    result = resolve(cfg, registry=reg)
    built = result["model"].build(label="custom")
    assert isinstance(built, FakeExternal)
    assert built.threshold == 0.9
    assert built.label == "custom"


def test_parse_node_with_both_registry_keys_raises(reg: Registry[object]) -> None:
    cfg = {
        "_registry_": "models.plain",
        "_magic_registry_": f"{__name__}.FakeExternal",
    }
    with pytest.raises(RegistryParseError):
        parse(cfg, registry=reg)


# ---------------------------------------------------------------------------
# parse: pass-through cases
# ---------------------------------------------------------------------------


def test_parse_plain_dict_without_registry_passes_through(
    reg: Registry[object],
) -> None:
    cfg = {"settings": {"lr": 0.01, "epochs": 10}}
    assert parse(cfg, registry=reg) == cfg


def test_parse_scalar_values_pass_through(reg: Registry[object]) -> None:
    cfg = {"lr": 0.01, "name": "run1", "enabled": True, "nothing": None}
    assert parse(cfg, registry=reg) == cfg


# ---------------------------------------------------------------------------
# parse: nested registry nodes
# ---------------------------------------------------------------------------


def test_parse_nested_registry_node(reg: Registry[object]) -> None:
    cfg = {
        "model": {
            "_registry_": "models.composite",
            "scale": 2.0,
            "sub": {"_registry_": "models.plain", "hidden_dim": 16},
        }
    }
    result = parse(cfg, registry=reg)
    outer = result["model"]
    assert isinstance(outer, ParsedNode)
    assert outer.key == "models.composite"
    inner = outer.kwargs["sub"]
    assert isinstance(inner, ParsedNode)
    assert inner.key == "models.plain"
    assert inner.kwargs == {"hidden_dim": 16}


# ---------------------------------------------------------------------------
# parse: list recursion
# ---------------------------------------------------------------------------


def test_parse_list_of_registry_nodes(reg: Registry[object]) -> None:
    cfg = {
        "transforms": [
            {"_registry_": "models.plain", "hidden_dim": 16},
            {"_registry_": "models.cfg", "hidden_dim": 32},
        ]
    }
    result = parse(cfg, registry=reg)
    transforms = result["transforms"]
    assert isinstance(transforms[0], ParsedNode)
    assert transforms[0].cls is Plain
    assert isinstance(transforms[1], ParsedNode)
    assert transforms[1].cls is Cfg


def test_parse_list_mixed_elements_pass_through(reg: Registry[object]) -> None:
    cfg = {"items": [{"_registry_": "models.plain"}, 42, "string", None]}
    result = parse(cfg, registry=reg)
    items = result["items"]
    assert isinstance(items[0], ParsedNode)
    assert items[1] == 42
    assert items[2] == "string"
    assert items[3] is None


# ---------------------------------------------------------------------------
# parse: errors
# ---------------------------------------------------------------------------


def test_parse_registry_value_not_string_raises_parse_error(
    reg: Registry[object],
) -> None:
    with pytest.raises(RegistryParseError):
        parse({"x": {"_registry_": 42}}, registry=reg)


def test_parse_unknown_key_propagates_registry_key_error(reg: Registry[object]) -> None:
    with pytest.raises(RegistryKeyError):
        parse({"x": {"_registry_": "nonexistent.key"}}, registry=reg)


# ---------------------------------------------------------------------------
# instantiate: plain class
# ---------------------------------------------------------------------------


def test_instantiate_plain_class_returns_instance() -> None:
    parsed = {
        "model": ParsedNode(key="models.plain", cls=Plain, kwargs={"hidden_dim": 32})
    }
    result = instantiate(parsed)
    assert isinstance(result["model"], Plain)
    assert result["model"].hidden_dim == 32


def test_instantiate_plain_class_default_kwargs() -> None:
    parsed = {"model": ParsedNode(key="models.plain", cls=Plain, kwargs={})}
    result = instantiate(parsed)
    assert result["model"].hidden_dim == 64


# ---------------------------------------------------------------------------
# instantiate: Configurable class -> Config, not instance
# ---------------------------------------------------------------------------


def test_instantiate_configurable_yields_config_not_instance() -> None:
    parsed = {"model": ParsedNode(key="models.cfg", cls=Cfg, kwargs={"hidden_dim": 64})}
    result = instantiate(parsed)
    assert isinstance(result["model"], Cfg.Config)
    assert not isinstance(result["model"], Cfg)


def test_instantiate_configurable_config_field_set() -> None:
    parsed = {"model": ParsedNode(key="models.cfg", cls=Cfg, kwargs={"hidden_dim": 99})}
    result = instantiate(parsed)
    assert result["model"].hidden_dim == 99


# ---------------------------------------------------------------------------
# instantiate: nested ParsedNodes resolved bottom-up
# ---------------------------------------------------------------------------


def test_instantiate_nested_nodes_bottom_up() -> None:
    inner = ParsedNode(key="models.plain", cls=Plain, kwargs={"hidden_dim": 16})
    outer = ParsedNode(
        key="models.composite", cls=Composite, kwargs={"sub": inner, "scale": 3.0}
    )
    result = instantiate({"model": outer})
    composite = result["model"]
    assert isinstance(composite, Composite)
    assert isinstance(composite.sub, Plain)
    assert composite.sub.hidden_dim == 16
    assert composite.scale == 3.0


# ---------------------------------------------------------------------------
# instantiate: list of ParsedNodes
# ---------------------------------------------------------------------------


def test_instantiate_list_of_nodes() -> None:
    parsed = {
        "transforms": [
            ParsedNode(key="models.plain", cls=Plain, kwargs={"hidden_dim": 16}),
            ParsedNode(key="models.plain", cls=Plain, kwargs={"hidden_dim": 32}),
        ]
    }
    result = instantiate(parsed)
    assert len(result["transforms"]) == 2
    assert result["transforms"][0].hidden_dim == 16
    assert result["transforms"][1].hidden_dim == 32


def test_instantiate_list_mixed_passthrough() -> None:
    parsed = {
        "items": [ParsedNode(key="models.plain", cls=Plain, kwargs={}), 42, "hello"]
    }
    result = instantiate(parsed)
    assert isinstance(result["items"][0], Plain)
    assert result["items"][1] == 42
    assert result["items"][2] == "hello"


# ---------------------------------------------------------------------------
# instantiate: errors
# ---------------------------------------------------------------------------


def test_instantiate_bad_kwargs_raises_instantiation_error() -> None:
    parsed = {"x": ParsedNode(key="models.plain", cls=Plain, kwargs={"bad_kwarg": 99})}
    with pytest.raises(ConfigInstantiationError):
        instantiate(parsed)


def test_instantiate_error_chains_original_cause() -> None:
    parsed = {"x": ParsedNode(key="models.plain", cls=Plain, kwargs={"bad_kwarg": 99})}
    with pytest.raises(ConfigInstantiationError) as exc_info:
        instantiate(parsed)
    assert exc_info.value.__cause__ is not None


# ---------------------------------------------------------------------------
# resolve: end-to-end
# ---------------------------------------------------------------------------


def test_resolve_plain_class_end_to_end(reg: Registry[object]) -> None:
    cfg = {"model": {"_registry_": "models.plain", "hidden_dim": 32}}
    result = resolve(cfg, registry=reg)
    assert isinstance(result["model"], Plain)
    assert result["model"].hidden_dim == 32


def test_resolve_configurable_class_end_to_end(reg: Registry[object]) -> None:
    cfg = {"model": {"_registry_": "models.cfg", "hidden_dim": 77}}
    result = resolve(cfg, registry=reg)
    assert isinstance(result["model"], Cfg.Config)
    assert result["model"].hidden_dim == 77


def test_resolve_nested_end_to_end(reg: Registry[object]) -> None:
    cfg = {
        "model": {
            "_registry_": "models.composite",
            "scale": 5.0,
            "sub": {"_registry_": "models.plain", "hidden_dim": 8},
        }
    }
    result = resolve(cfg, registry=reg)
    assert isinstance(result["model"], Composite)
    assert isinstance(result["model"].sub, Plain)
    assert result["model"].sub.hidden_dim == 8


def test_resolve_list_end_to_end(reg: Registry[object]) -> None:
    cfg = {
        "transforms": [
            {"_registry_": "models.plain", "hidden_dim": 4},
            {"_registry_": "models.plain", "hidden_dim": 8},
        ]
    }
    result = resolve(cfg, registry=reg)
    assert len(result["transforms"]) == 2
    assert result["transforms"][0].hidden_dim == 4


# ---------------------------------------------------------------------------
# resolve: top-level _registry_ node (strict inverse of to_dict)
# ---------------------------------------------------------------------------


def test_parse_top_level_registry_node_returns_parsed_node(
    reg: Registry[object],
) -> None:
    result = parse({"_registry_": "models.cfg", "hidden_dim": 64}, registry=reg)
    assert isinstance(result, ParsedNode)
    assert result.cls is Cfg
    assert result.kwargs == {"hidden_dim": 64}


def test_instantiate_top_level_parsed_node_returns_object() -> None:
    node = ParsedNode(key="models.plain", cls=Plain, kwargs={"hidden_dim": 32})
    result = instantiate(node)
    assert isinstance(result, Plain)
    assert result.hidden_dim == 32


def test_resolve_top_level_registry_node_returns_config(
    reg: Registry[object],
) -> None:
    d = {"_registry_": "models.cfg", "hidden_dim": 77}
    result = resolve(d, registry=reg)
    assert isinstance(result, Cfg.Config)
    assert result.hidden_dim == 77


def test_resolve_is_strict_inverse_of_to_dict_flat(reg: Registry[object]) -> None:
    original = Cfg.Config(hidden_dim=99)
    result = resolve(to_dict(original, reg), registry=reg)
    assert isinstance(result, Cfg.Config)
    assert result == original


def test_resolve_is_strict_inverse_of_to_dict_nested(reg: Registry[object]) -> None:
    original = Outer.Config(inner=Cfg.Config(hidden_dim=16), scale=2.0)
    result = resolve(to_dict(original, reg), registry=reg)
    assert isinstance(result, Outer.Config)
    assert isinstance(result.inner, Cfg.Config)
    assert result.inner.hidden_dim == 16
    assert result.scale == 2.0


# ---------------------------------------------------------------------------
# to_dict: scalar pass-through
# ---------------------------------------------------------------------------


def test_to_dict_int_passthrough(reg: Registry[object]) -> None:
    assert to_dict(42, reg) == 42


def test_to_dict_float_passthrough(reg: Registry[object]) -> None:
    assert to_dict(3.14, reg) == 3.14


def test_to_dict_str_passthrough(reg: Registry[object]) -> None:
    assert to_dict("hello", reg) == "hello"


def test_to_dict_bool_passthrough(reg: Registry[object]) -> None:
    assert to_dict(True, reg) is True


def test_to_dict_none_passthrough(reg: Registry[object]) -> None:
    assert to_dict(None, reg) is None


# ---------------------------------------------------------------------------
# to_dict: dict and list recursion
# ---------------------------------------------------------------------------


def test_to_dict_plain_dict_recurses(reg: Registry[object]) -> None:
    assert to_dict({"lr": 0.01, "epochs": 10}, reg) == {"lr": 0.01, "epochs": 10}


def test_to_dict_list_recurses(reg: Registry[object]) -> None:
    assert to_dict([1, 2, 3], reg) == [1, 2, 3]


def test_to_dict_list_of_scalars(reg: Registry[object]) -> None:
    assert to_dict(["a", 1, None], reg) == ["a", 1, None]


# ---------------------------------------------------------------------------
# to_dict: Configurable.Config
# ---------------------------------------------------------------------------


def test_to_dict_configurable_config_flat(reg: Registry[object]) -> None:
    result = to_dict(Cfg.Config(hidden_dim=99), reg)
    assert result == {"_registry_": "models.cfg", "hidden_dim": 99}


def test_to_dict_configurable_config_default_values(reg: Registry[object]) -> None:
    result = to_dict(Cfg.Config(), reg)
    assert result == {"_registry_": "models.cfg", "hidden_dim": 128}


def test_to_dict_registry_key_present(reg: Registry[object]) -> None:
    result = to_dict(Cfg.Config(hidden_dim=64), reg)
    assert "_registry_" in result
    assert result["_registry_"] == "models.cfg"


# ---------------------------------------------------------------------------
# to_dict: nested Configurable.Config
# ---------------------------------------------------------------------------


def test_to_dict_nested_config(reg: Registry[object]) -> None:
    cfg = Outer.Config(inner=Cfg.Config(hidden_dim=16), scale=2.0)
    result = to_dict(cfg, reg)
    assert result == {
        "_registry_": "outer",
        "inner": {"_registry_": "models.cfg", "hidden_dim": 16},
        "scale": 2.0,
    }


# ---------------------------------------------------------------------------
# to_dict: list of Configs
# ---------------------------------------------------------------------------


def test_to_dict_list_of_configs(reg: Registry[object]) -> None:
    cfg = WithList.Config(items=[Cfg.Config(hidden_dim=4), Cfg.Config(hidden_dim=8)])
    result = to_dict(cfg, reg)
    assert result == {
        "_registry_": "with_list",
        "items": [
            {"_registry_": "models.cfg", "hidden_dim": 4},
            {"_registry_": "models.cfg", "hidden_dim": 8},
        ],
    }


# ---------------------------------------------------------------------------
# to_dict: round-trip
# ---------------------------------------------------------------------------


def test_to_dict_round_trip_flat(reg: Registry[object]) -> None:
    original = Cfg.Config(hidden_dim=77)
    restored = resolve(to_dict(original, reg), registry=reg)
    assert isinstance(restored, Cfg.Config)
    assert restored.hidden_dim == 77


def test_to_dict_round_trip_nested(reg: Registry[object]) -> None:
    original = Outer.Config(inner=Cfg.Config(hidden_dim=32), scale=3.0)
    restored = resolve(to_dict(original, reg), registry=reg)
    assert isinstance(restored, Outer.Config)
    assert isinstance(restored.inner, Cfg.Config)
    assert restored.inner.hidden_dim == 32
    assert restored.scale == 3.0


# ---------------------------------------------------------------------------
# to_dict: _magic_registry_ node
# ---------------------------------------------------------------------------


def test_to_dict_magic_registry_node(reg: Registry[object]) -> None:
    cls = magic_wrap(f"{__name__}.FakeExternal", ["threshold"])
    cfg = cls.Config(threshold=0.7)
    result = to_dict(cfg, reg)
    assert result == {
        "_magic_registry_": f"{__name__}.FakeExternal",
        "threshold": 0.7,
    }


def test_to_dict_magic_registry_round_trip(reg: Registry[object]) -> None:
    cls = magic_wrap(f"{__name__}.FakeExternal", ["threshold"])
    original = cls.Config(threshold=0.3)
    restored = resolve(to_dict(original, reg), registry=reg)
    built = restored.build(label="rt")
    assert isinstance(built, FakeExternal)
    assert built.threshold == 0.3
    assert built.label == "rt"


# ---------------------------------------------------------------------------
# to_dict: error cases
# ---------------------------------------------------------------------------


def test_to_dict_unregistered_owner_raises(reg: Registry[object]) -> None:
    class Unregistered(Configurable):
        class Config(Configurable.Config):
            val: int = 0

        def __init__(self, cfg: "Unregistered.Config") -> None:
            pass

    with pytest.raises(ConfigSerializationError):
        to_dict(Unregistered.Config(), reg)


def test_to_dict_orphan_config_raises(reg: Registry[object]) -> None:
    class OrphanConfig(Configurable.Config):
        val: int = 0

    with pytest.raises(ConfigSerializationError):
        to_dict(OrphanConfig(), reg)


def test_to_dict_deferred_field_none_serializes_as_null(reg: Registry[object]) -> None:
    """Deferred fields with None value round-trip as null -- no error."""

    class WithDeferred(Configurable):
        class Config(Configurable.Config):
            hidden_dim: int = 64
            input_dim: int | None = None

        def __init__(self, cfg: "WithDeferred.Config") -> None:
            pass

    reg.register("with_deferred")(WithDeferred)

    result = to_dict(WithDeferred.Config(), reg)
    assert result == {
        "_registry_": "with_deferred",
        "hidden_dim": 64,
        "input_dim": None,
    }


def test_to_dict_round_trip_with_deferred_field(reg: Registry[object]) -> None:
    """Deferred field supplied at construction round-trips correctly."""

    class WithDeferred(Configurable):
        class Config(Configurable.Config):
            hidden_dim: int = 64
            input_dim: int | None = None

        def __init__(self, cfg: "WithDeferred.Config") -> None:
            pass

    reg.register("with_deferred_rt")(WithDeferred)

    original = WithDeferred.Config(hidden_dim=64, input_dim=32)
    restored = resolve(to_dict(original, reg), registry=reg)
    assert isinstance(restored, WithDeferred.Config)
    assert restored.hidden_dim == 64
    assert restored.input_dim == 32


def test_to_dict_skips_build_time_field(reg: Registry[object]) -> None:
    """BuildTimeField is omitted entirely -- not even a null key."""

    class WithBuildTimeField(Configurable):
        class Config(Configurable.Config):
            hidden_dim: int = 64
            dataloader: BuildTimeField[str] = UNSET

        def __init__(self, cfg: "WithBuildTimeField.Config") -> None:
            pass

    reg.register("with_build_time_field")(WithBuildTimeField)

    result = to_dict(WithBuildTimeField.Config(), reg)
    assert result == {
        "_registry_": "with_build_time_field",
        "hidden_dim": 64,
    }
    assert "dataloader" not in result


def test_to_dict_round_trip_omits_build_time_field(reg: Registry[object]) -> None:
    """Round-tripping through to_dict()/resolve() leaves the field UNSET."""

    class WithBuildTimeField(Configurable):
        class Config(Configurable.Config):
            hidden_dim: int = 64
            dataloader: BuildTimeField[str] = UNSET

        def __init__(self, cfg: "WithBuildTimeField.Config") -> None:
            pass

    reg.register("with_build_time_field_rt")(WithBuildTimeField)

    original = WithBuildTimeField.Config(hidden_dim=32)
    restored = resolve(to_dict(original, reg), registry=reg)
    assert isinstance(restored, WithBuildTimeField.Config)
    assert restored.hidden_dim == 32
    with pytest.raises(ConfigValidationError):
        restored.build()  # dataloader was never supplied


def test_to_dict_serializes_build_time_field_with_own_default(
    reg: Registry[object],
) -> None:
    """A BuildTimeField given its own (non-UNSET) default is no longer
    detected as build-time-only -- it serializes like any ordinary field.
    This is a documented consequence of is_build_time_field() sharing one
    detection mechanism between the build() requirement and the to_dict()
    skip, not a separate opt-out flag."""

    class WithDefaultBuildTimeField(Configurable):
        class Config(Configurable.Config):
            logger: BuildTimeField[str] = "default-logger"

        def __init__(self, cfg: "WithDefaultBuildTimeField.Config") -> None:
            pass

    reg.register("with_default_build_time_field")(WithDefaultBuildTimeField)

    result = to_dict(WithDefaultBuildTimeField.Config(), reg)
    assert result == {
        "_registry_": "with_default_build_time_field",
        "logger": "default-logger",
    }


def test_to_dict_round_trip_with_defaulted_build_time_field(
    reg: Registry[object],
) -> None:
    """A BuildTimeField with its own default round-trips like an ordinary
    field: resolve() restores it and build() succeeds without needing an
    override, since it was never UNSET in the first place."""

    class WithDefaultBuildTimeField(Configurable):
        class Config(Configurable.Config):
            logger: BuildTimeField[str] = "default-logger"

        def __init__(self, cfg: "WithDefaultBuildTimeField.Config") -> None:
            self.logger = cfg.logger

    reg.register("with_default_build_time_field_rt")(WithDefaultBuildTimeField)

    original = WithDefaultBuildTimeField.Config()
    restored = resolve(to_dict(original, reg), registry=reg)
    assert isinstance(restored, WithDefaultBuildTimeField.Config)
    assert restored == original
    assert restored.logger == "default-logger"
    obj = restored.build()
    assert obj.logger == "default-logger"


def test_to_dict_unsupported_type_raises(reg: Registry[object]) -> None:
    class Blob:
        pass

    with pytest.raises(ConfigSerializationError):
        to_dict(Blob(), reg)


# ---------------------------------------------------------------------------
# to_dict: plain dataclass
# ---------------------------------------------------------------------------


def test_to_dict_plain_dataclass_registered(reg: Registry[object]) -> None:
    result = to_dict(PlainDC(width=16, label="x"), reg)
    assert result == {"_registry_": "plain_dc", "width": 16, "label": "x"}


def test_to_dict_plain_dataclass_registry_key_present(reg: Registry[object]) -> None:
    result = to_dict(PlainDC(), reg)
    assert result["_registry_"] == "plain_dc"


def test_to_dict_plain_dataclass_default_values(reg: Registry[object]) -> None:
    result = to_dict(PlainDC(), reg)
    assert result == {"_registry_": "plain_dc", "width": 32, "label": "default"}


def test_to_dict_plain_dataclass_unregistered_raises(reg: Registry[object]) -> None:
    @dataclass(kw_only=True)
    class Ghost:
        val: int = 0

    with pytest.raises(ConfigSerializationError):
        to_dict(Ghost(), reg)


def test_to_dict_plain_dataclass_round_trip(reg: Registry[object]) -> None:
    original = PlainDC(width=8, label="rt")
    restored = resolve(to_dict(original, reg), registry=reg)
    assert isinstance(restored, PlainDC)
    assert restored.width == 8
    assert restored.label == "rt"


def test_to_dict_plain_dataclass_nested_in_configurable_config(
    reg: Registry[object],
) -> None:
    """Plain dataclass as a field value inside a Configurable.Config."""

    @dataclass(kw_only=True)
    class Padding:
        left: int = 0
        right: int = 0

    reg.register("padding")(Padding)

    class WithPadding(Configurable):
        class Config(Configurable.Config):
            pad: Padding = Field(default_factory=Padding)
            hidden_dim: int = 64

        def __init__(self, cfg: "WithPadding.Config") -> None:
            pass

    reg.register("with_padding")(WithPadding)

    cfg = WithPadding.Config(pad=Padding(left=4, right=4), hidden_dim=128)
    result = to_dict(cfg, reg)
    assert result == {
        "_registry_": "with_padding",
        "pad": {"_registry_": "padding", "left": 4, "right": 4},
        "hidden_dim": 128,
    }


def test_to_dict_configurable_config_nested_in_plain_dataclass(
    reg: Registry[object],
) -> None:
    """Configurable.Config as a field value inside a plain dataclass."""

    @dataclass(kw_only=True)
    class Wrapper:
        cfg: Cfg.Config = field(default_factory=Cfg.Config)
        name: str = "w"

    reg.register("wrapper")(Wrapper)

    obj = Wrapper(cfg=Cfg.Config(hidden_dim=16), name="test")
    result = to_dict(obj, reg)
    assert result == {
        "_registry_": "wrapper",
        "cfg": {"_registry_": "models.cfg", "hidden_dim": 16},
        "name": "test",
    }


# ---------------------------------------------------------------------------
# to_dict: plain BaseModel
# ---------------------------------------------------------------------------


def test_to_dict_handles_plain_basemodel(reg: Registry[object]) -> None:
    class Point(PydanticBaseModel):
        x: int
        label: str

    result = to_dict(Point(x=3, label="a"), registry=reg)

    assert result == {"x": 3, "label": "a"}


def test_to_dict_basemodel_recurses_into_configurable_config(
    reg: Registry[object],
) -> None:
    class Wrapper(PydanticBaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        name: str
        inner: Cfg.Config

    result = to_dict(Wrapper(name="w", inner=Cfg.Config(hidden_dim=64)), registry=reg)

    assert result == {
        "name": "w",
        "inner": {"_registry_": "models.cfg", "hidden_dim": 64},
    }
