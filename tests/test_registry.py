from typing import Any

import pytest

from mlbricks import (
    DuplicateRegistrationError,
    InvalidRegistryNameError,
    Registry,
    RegistryKeyError,
)


@pytest.fixture
def reg() -> Registry[Any]:
    return Registry()


# ---------------------------------------------------------------------------
# register + get: flat
# ---------------------------------------------------------------------------


def test_register_and_get_flat(reg: Registry[Any]) -> None:
    @reg.register("foo")
    class Foo:
        pass

    assert reg.get("foo") is Foo


def test_register_returns_class_unchanged(reg: Registry[Any]) -> None:
    class Foo:
        pass

    result = reg.register("foo")(Foo)
    assert result is Foo


# ---------------------------------------------------------------------------
# register + get: nested
# ---------------------------------------------------------------------------


def test_register_and_get_nested(reg: Registry[Any]) -> None:
    sub = reg.subgroup("models")

    @sub.register("mlp")
    class MLP:
        pass

    assert reg.get("models.mlp") is MLP


def test_register_and_get_multi_level(reg: Registry[Any]) -> None:
    models = reg.subgroup("models")
    gnn = models.subgroup("gnn")

    @gnn.register("gat")
    class GAT:
        pass

    assert reg.get("models.gnn.gat") is GAT


# ---------------------------------------------------------------------------
# __contains__
# ---------------------------------------------------------------------------


def test_contains_registered_leaf(reg: Registry[Any]) -> None:
    @reg.register("foo")
    class Foo:
        pass

    assert "foo" in reg


def test_contains_missing_path(reg: Registry[Any]) -> None:
    assert "missing" not in reg


def test_contains_nested_leaf(reg: Registry[Any]) -> None:
    sub = reg.subgroup("models")

    @sub.register("mlp")
    class MLP:
        pass

    assert "models.mlp" in reg


def test_contains_subgroup_name_is_false(reg: Registry[Any]) -> None:
    reg.subgroup("models")
    assert "models" not in reg


# ---------------------------------------------------------------------------
# subgroup error cases
# ---------------------------------------------------------------------------


def test_subgroup_duplicate_name_raises(reg: Registry[Any]) -> None:
    reg.subgroup("models")
    with pytest.raises(DuplicateRegistrationError):
        reg.subgroup("models")


def test_subgroup_name_with_period_raises(reg: Registry[Any]) -> None:
    with pytest.raises(InvalidRegistryNameError):
        reg.subgroup("models.gnn")


def test_subgroup_empty_name_raises(reg: Registry[Any]) -> None:
    with pytest.raises(InvalidRegistryNameError):
        reg.subgroup("")


# ---------------------------------------------------------------------------
# register error cases
# ---------------------------------------------------------------------------


def test_register_duplicate_name_raises(reg: Registry[Any]) -> None:
    @reg.register("foo")
    class Foo:
        pass

    with pytest.raises(DuplicateRegistrationError):

        @reg.register("foo")
        class Foo2:
            pass


def test_register_name_with_period_raises(reg: Registry[Any]) -> None:
    with pytest.raises(InvalidRegistryNameError):

        @reg.register("models.mlp")
        class MLP:
            pass


def test_register_empty_name_raises(reg: Registry[Any]) -> None:
    with pytest.raises(InvalidRegistryNameError):

        @reg.register("")
        class Empty:
            pass


# ---------------------------------------------------------------------------
# get error cases
# ---------------------------------------------------------------------------


def test_get_missing_path_raises(reg: Registry[Any]) -> None:
    with pytest.raises(RegistryKeyError):
        reg.get("missing")


def test_get_intermediate_is_leaf_raises(reg: Registry[Any]) -> None:
    @reg.register("foo")
    class Foo:
        pass

    with pytest.raises(RegistryKeyError):
        reg.get("foo.bar")


def test_get_final_is_subgroup_raises(reg: Registry[Any]) -> None:
    reg.subgroup("models")
    with pytest.raises(RegistryKeyError):
        reg.get("models")


# ---------------------------------------------------------------------------
# leaf_paths
# ---------------------------------------------------------------------------


def test_leaf_paths_flat(reg: Registry[Any]) -> None:
    @reg.register("foo")
    class Foo:
        pass

    @reg.register("bar")
    class Bar:
        pass

    assert reg.leaf_paths() == {"foo", "bar"}


def test_leaf_paths_nested(reg: Registry[Any]) -> None:
    sub = reg.subgroup("models")

    @sub.register("mlp")
    class MLP:
        pass

    assert reg.leaf_paths() == {"models.mlp"}


def test_leaf_paths_multi_level(reg: Registry[Any]) -> None:
    models = reg.subgroup("models")
    gnn = models.subgroup("gnn")

    @gnn.register("gat")
    class GAT:
        pass

    @models.register("mlp")
    class MLP:
        pass

    assert reg.leaf_paths() == {"models.gnn.gat", "models.mlp"}


def test_leaf_paths_empty(reg: Registry[Any]) -> None:
    assert reg.leaf_paths() == set()


def test_leaf_paths_subgroup_only_not_included(reg: Registry[Any]) -> None:
    reg.subgroup("models")
    assert reg.leaf_paths() == set()


# ---------------------------------------------------------------------------
# path_of
# ---------------------------------------------------------------------------


def test_path_of_flat(reg: Registry[Any]) -> None:
    @reg.register("foo")
    class Foo:
        pass

    assert reg.path_of(Foo) == "foo"


def test_path_of_nested(reg: Registry[Any]) -> None:
    sub = reg.subgroup("models")

    @sub.register("mlp")
    class MLP:
        pass

    assert reg.path_of(MLP) == "models.mlp"


def test_path_of_multi_level(reg: Registry[Any]) -> None:
    models = reg.subgroup("models")
    gnn = models.subgroup("gnn")

    @gnn.register("gat")
    class GAT:
        pass

    assert reg.path_of(GAT) == "models.gnn.gat"


def test_path_of_unregistered_returns_none(reg: Registry[Any]) -> None:
    class Unknown:
        pass

    assert reg.path_of(Unknown) is None


def test_path_of_subgroup_not_a_leaf(reg: Registry[Any]) -> None:
    # subgroups are not leaves -- should not be returned as a path
    reg.subgroup("models")

    class Models:
        pass

    # "models" maps to a Registry, not this class
    assert reg.path_of(Models) is None
