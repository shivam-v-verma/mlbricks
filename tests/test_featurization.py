from typing import Any

import pytest

from mlbricks import ConfigurableError, Registry
from mlbricks.configurable import ConfigValidationError
from mlbricks.featurization import (
    ConfigurableTransform,
    Context,
    TransformHooks,
    TransformPipeline,
)

type Payload = dict[str, object]
"""A plain dict -- TransformPipeline is generic and needs no domain payload."""


class AlwaysFail(ConfigurableTransform[bool]):
    """A "filter" step: unconditionally rejects, ignoring incoming payload.

    A step that unconditionally passes needs no stub -- it's just NoopTransform
    with Payload=bool (AND-folding True changes nothing).
    """

    class Config(ConfigurableTransform.Config): ...

    def __call__(self, payload: bool, ctx: Context) -> tuple[bool, Context]:
        return False, ctx


class NoopTransform(ConfigurableTransform[Payload]):
    """Transform that returns payload and ctx unchanged."""

    class Config(ConfigurableTransform.Config): ...

    def __call__(self, payload: Payload, ctx: Context) -> tuple[Payload, Context]:
        return payload, ctx


class StepLogTransform(ConfigurableTransform[Payload]):
    """Appends cfg.tag to ctx['log'] -- used for step-ordering assertions."""

    class Config(ConfigurableTransform.Config):
        tag: str = ""

    def __init__(self, cfg: "StepLogTransform.Config") -> None:
        super().__init__(cfg)
        self.tag = cfg.tag

    def __call__(self, payload: Payload, ctx: Context) -> tuple[Payload, Context]:
        ctx.setdefault("log", []).append(self.tag)  # type: ignore[union-attr]
        return payload, ctx


class StepLogBool(ConfigurableTransform[bool]):
    """Appends cfg.tag to ctx['log'] and AND-folds cfg.result into payload."""

    class Config(ConfigurableTransform.Config):
        tag: str = ""
        result: bool = True

    def __init__(self, cfg: "StepLogBool.Config") -> None:
        super().__init__(cfg)
        self.tag = cfg.tag
        self.result = cfg.result

    def __call__(self, payload: bool, ctx: Context) -> tuple[bool, Context]:
        ctx.setdefault("log", []).append(self.tag)  # type: ignore[union-attr]
        return payload and self.result, ctx


class WriteCtxTransform(ConfigurableTransform[Payload]):
    """Writes cfg.value to ctx[cfg.key] -- verifies self.cfg access without __init__."""

    class Config(ConfigurableTransform.Config):
        key: str = "result"
        value: str = "x"

    # Narrows self.cfg to the subclass Config so ty can resolve field access.
    cfg: Config

    def __call__(self, payload: Payload, ctx: Context) -> tuple[Payload, Context]:
        ctx[self.cfg.key] = self.cfg.value
        return payload, ctx


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def reg() -> Registry[Any]:
    """Fresh Registry pre-populated with transform test doubles.

    Use for tests (e.g. get_hash) that must not touch a shared registry.
    """
    fresh: Registry[Any] = Registry()
    transforms_reg = fresh.subgroup("transforms")
    transforms_reg.register("noop")(NoopTransform)
    transforms_reg.register("write_ctx")(WriteCtxTransform)
    return fresh


# ---------------------------------------------------------------------------
# ConfigurableTransform base -- callable behavior
# ---------------------------------------------------------------------------


def test_noop_transform_returns_payload_and_ctx_unchanged() -> None:
    transform = NoopTransform.Config().build()
    payload: Payload = {"key": "val"}
    ctx: Context = {"key": "val"}
    result_payload, result_ctx = transform(payload, ctx)
    assert result_payload is payload
    assert result_ctx is ctx


def test_transform_self_cfg_accessible_without_init() -> None:
    """Config fields on self.cfg work with no __init__ defined on the subclass."""
    transform = WriteCtxTransform.Config(key="out", value="z").build()
    ctx: Context = {}
    transform({}, ctx)
    assert ctx["out"] == "z"


# ---------------------------------------------------------------------------
# TransformPipeline.Config validation
# ---------------------------------------------------------------------------


def test_validate_rejects_duplicate_step_name() -> None:
    with pytest.raises(ConfigValidationError):
        TransformPipeline.Config(
            steps=[
                {"name": "step1", "do": NoopTransform.Config()},
                {"name": "step1", "do": NoopTransform.Config()},
            ]
        )


def test_validate_rejects_missing_name_key() -> None:
    with pytest.raises(ConfigValidationError):
        TransformPipeline.Config(steps=[{"do": NoopTransform.Config()}])


def test_validate_rejects_missing_do_key() -> None:
    with pytest.raises(ConfigValidationError):
        TransformPipeline.Config(steps=[{"name": "step1"}])


# ---------------------------------------------------------------------------
# TransformPipeline call behavior
# ---------------------------------------------------------------------------


def test_pipeline_chains_steps_in_order() -> None:
    pipeline = TransformPipeline.Config(
        steps=[
            {"name": "a", "do": StepLogTransform.Config(tag="a")},
            {"name": "b", "do": StepLogTransform.Config(tag="b")},
        ]
    ).build()
    ctx: Context = {}
    pipeline({}, ctx)
    assert ctx["log"] == ["a", "b"]


def test_pipeline_with_no_steps_returns_payload_unchanged() -> None:
    pipeline = TransformPipeline.Config().build()
    payload: Payload = {}
    result_payload, _ = pipeline(payload, {})
    assert result_payload is payload


# ---------------------------------------------------------------------------
# Filters as Payload=bool transforms: AND-fold, no engine short-circuit
# ---------------------------------------------------------------------------


def test_bool_payload_pipeline_ands_all_step_results() -> None:
    pipeline = TransformPipeline.Config(
        steps=[
            {"name": "a", "do": StepLogBool.Config(tag="a", result=True)},
            {"name": "b", "do": StepLogBool.Config(tag="b", result=False)},
            {"name": "c", "do": StepLogBool.Config(tag="c", result=True)},
        ]
    ).build()
    ctx: Context = {}
    result, _ = pipeline(True, ctx)
    assert result is False
    assert ctx["log"] == ["a", "b", "c"]  # every step still runs, no short-circuit


def test_bool_payload_pipeline_true_when_all_steps_pass() -> None:
    pipeline = TransformPipeline.Config(
        steps=[
            {"name": "a", "do": StepLogBool.Config(tag="a", result=True)},
            {"name": "b", "do": StepLogBool.Config(tag="b", result=True)},
        ]
    ).build()
    result, _ = pipeline(True, {})
    assert result is True


def test_bool_payload_pipeline_always_fail_ignores_incoming_payload() -> None:
    pipeline = TransformPipeline.Config(
        steps=[{"name": "a", "do": AlwaysFail.Config()}]
    ).build()
    result, _ = pipeline(True, {})
    assert result is False


# ---------------------------------------------------------------------------
# TransformHooks.Config validation
# ---------------------------------------------------------------------------


def test_hooks_validate_accepts_start_end_plain_string() -> None:
    TransformHooks.Config(
        entries=[
            {"at": "start", "do": NoopTransform.Config()},
            {"at": "end", "do": NoopTransform.Config()},
        ]
    )  # must not raise


def test_hooks_validate_accepts_before_after_dict() -> None:
    TransformHooks.Config(
        entries=[
            {"at": {"before": "step_a"}, "do": NoopTransform.Config()},
            {"at": {"after": "step_b"}, "do": NoopTransform.Config()},
        ]
    )  # must not raise


def test_hooks_validate_rejects_extra_key_in_entry() -> None:
    with pytest.raises(ConfigValidationError):
        TransformHooks.Config(
            entries=[{"at": "start", "do": NoopTransform.Config(), "extra": "bad"}]
        )


def test_hooks_validate_rejects_missing_at_key() -> None:
    with pytest.raises(ConfigValidationError):
        TransformHooks.Config(entries=[{"do": NoopTransform.Config()}])


def test_hooks_validate_rejects_missing_do_key() -> None:
    with pytest.raises(ConfigValidationError):
        TransformHooks.Config(entries=[{"at": "start"}])


def test_hooks_validate_rejects_unknown_plain_string_at() -> None:
    with pytest.raises(ConfigValidationError):
        TransformHooks.Config(entries=[{"at": "during", "do": NoopTransform.Config()}])


def test_hooks_validate_rejects_multi_key_at_dict() -> None:
    with pytest.raises(ConfigValidationError):
        TransformHooks.Config(
            entries=[
                {"at": {"before": "a", "after": "b"}, "do": NoopTransform.Config()}
            ]
        )


def test_hooks_validate_rejects_unknown_dict_at_key() -> None:
    with pytest.raises(ConfigValidationError):
        TransformHooks.Config(
            entries=[{"at": {"during": "step_a"}, "do": NoopTransform.Config()}]
        )


def test_hooks_validate_rejects_non_string_val_in_before() -> None:
    with pytest.raises(ConfigValidationError):
        TransformHooks.Config(
            entries=[{"at": {"before": 123}, "do": NoopTransform.Config()}]
        )


# ---------------------------------------------------------------------------
# TransformHooks.get() lookup
# ---------------------------------------------------------------------------


def test_hooks_get_start_returns_hooks_in_definition_order() -> None:
    cfg = TransformHooks.Config(
        entries=[
            {"at": "start", "do": StepLogTransform.Config(tag="a")},
            {"at": "start", "do": StepLogTransform.Config(tag="b")},
        ]
    )
    hooks = cfg.build()
    ctx: Context = {}
    payload: Payload = {}
    for fn in hooks.get("start"):
        payload, ctx = fn(payload, ctx)
    assert ctx["log"] == ["a", "b"]


def test_hooks_get_before_scoped_to_named_step() -> None:
    cfg = TransformHooks.Config(
        entries=[{"at": {"before": "step_x"}, "do": NoopTransform.Config()}]
    )
    hooks = cfg.build()
    assert len(hooks.get("before", "step_x")) == 1
    assert len(hooks.get("before", "other_step")) == 0
    assert len(hooks.get("after", "step_x")) == 0


def test_hooks_get_returns_empty_list_for_unregistered_point() -> None:
    hooks = TransformHooks.Config().build()
    assert hooks.get("start") == []
    assert hooks.get("end") == []
    assert hooks.get("before", "step_x") == []


# ---------------------------------------------------------------------------
# TransformHooks.named_refs()
# ---------------------------------------------------------------------------


def test_named_refs_collects_before_and_after_values() -> None:
    cfg = TransformHooks.Config(
        entries=[
            {"at": {"before": "step_a"}, "do": NoopTransform.Config()},
            {"at": {"after": "step_b"}, "do": NoopTransform.Config()},
            {"at": "start", "do": NoopTransform.Config()},
        ]
    )
    hooks = cfg.build()
    assert hooks.named_refs() == {"step_a", "step_b"}


def test_named_refs_excludes_start_end() -> None:
    cfg = TransformHooks.Config(
        entries=[
            {"at": "start", "do": NoopTransform.Config()},
            {"at": "end", "do": NoopTransform.Config()},
        ]
    )
    assert cfg.build().named_refs() == set()


def test_named_refs_empty_hooks_returns_empty_set() -> None:
    assert TransformHooks.Config().build().named_refs() == set()


# ---------------------------------------------------------------------------
# with_hooks integration: validation + lifecycle bracketing
# ---------------------------------------------------------------------------


def test_with_hooks_raises_for_unknown_step_reference() -> None:
    pipeline = TransformPipeline.Config(
        steps=[{"name": "real_step", "do": NoopTransform.Config()}]
    ).build()
    hooks_cfg = TransformHooks.Config(
        entries=[{"at": {"before": "ghost"}, "do": NoopTransform.Config()}]
    )
    with pytest.raises(ConfigurableError):
        pipeline.with_hooks(hooks_cfg)


def test_with_hooks_accepts_valid_named_refs() -> None:
    pipeline = TransformPipeline.Config(
        steps=[{"name": "real_step", "do": NoopTransform.Config()}]
    ).build()
    hooks_cfg = TransformHooks.Config(
        entries=[{"at": {"before": "real_step"}, "do": NoopTransform.Config()}]
    )
    pipeline.with_hooks(hooks_cfg)  # must not raise


def test_with_hooks_start_end_bracket_steps() -> None:
    pipeline = TransformPipeline.Config(
        steps=[{"name": "the_step", "do": StepLogTransform.Config(tag="step")}]
    ).build()
    hooks_cfg = TransformHooks.Config(
        entries=[
            {"at": "start", "do": StepLogTransform.Config(tag="start")},
            {"at": "end", "do": StepLogTransform.Config(tag="end")},
        ]
    )
    pipeline.with_hooks(hooks_cfg)
    ctx: Context = {}
    pipeline({}, ctx)
    assert ctx["log"] == ["start", "step", "end"]


def test_with_hooks_before_after_fire_around_named_step() -> None:
    pipeline = TransformPipeline.Config(
        steps=[{"name": "my_step", "do": StepLogTransform.Config(tag="step")}]
    ).build()
    hooks_cfg = TransformHooks.Config(
        entries=[
            {"at": {"before": "my_step"}, "do": StepLogTransform.Config(tag="before")},
            {"at": {"after": "my_step"}, "do": StepLogTransform.Config(tag="after")},
        ]
    )
    pipeline.with_hooks(hooks_cfg)
    ctx: Context = {}
    pipeline({}, ctx)
    assert ctx["log"] == ["before", "step", "after"]


# ---------------------------------------------------------------------------
# get_hash
# ---------------------------------------------------------------------------


def test_get_hash_returns_eight_char_hex_string(reg: Registry[Any]) -> None:
    pipeline = TransformPipeline.Config().build()
    h = pipeline.get_hash(reg)
    assert len(h) == 8
    assert all(c in "0123456789abcdef" for c in h)


def test_get_hash_same_config_same_hash(reg: Registry[Any]) -> None:
    h1 = TransformPipeline.Config().build().get_hash(reg)
    h2 = TransformPipeline.Config().build().get_hash(reg)
    assert h1 == h2


def test_get_hash_no_hooks_vs_empty_hooks_differ(reg: Registry[Any]) -> None:
    no_hooks = TransformPipeline.Config().build()
    empty_hooks = TransformPipeline.Config().build()
    empty_hooks.with_hooks(TransformHooks.Config())
    assert no_hooks.get_hash(reg) != empty_hooks.get_hash(reg)


def test_get_hash_same_hooks_same_hash(reg: Registry[Any]) -> None:
    hooks_cfg = TransformHooks.Config()
    p1 = TransformPipeline.Config().build()
    p2 = TransformPipeline.Config().build()
    p1.with_hooks(hooks_cfg)
    p2.with_hooks(hooks_cfg)
    assert p1.get_hash(reg) == p2.get_hash(reg)


def test_get_hash_reflects_current_state_not_build_time(reg: Registry[Any]) -> None:
    """Calling get_hash() before vs after with_hooks() gives different results --
    callers own cache-key semantics by choosing when to call it."""
    pipeline = TransformPipeline.Config().build()
    before = pipeline.get_hash(reg)
    pipeline.with_hooks(TransformHooks.Config())
    after = pipeline.get_hash(reg)
    assert before != after


def test_get_hash_does_not_require_pipeline_or_hooks_registration() -> None:
    """get_hash hashes steps/hook entries, not the wrapper Configs -- only
    each step's transform class needs to be registered, not TransformPipeline
    or TransformHooks themselves."""
    only_transforms: Registry[Any] = Registry()
    only_transforms.subgroup("transforms").register("noop")(NoopTransform)

    pipeline = TransformPipeline.Config(
        steps=[{"name": "noop", "do": NoopTransform.Config()}]
    ).build()
    pipeline.with_hooks(
        TransformHooks.Config(entries=[{"at": "start", "do": NoopTransform.Config()}])
    )

    h = pipeline.get_hash(only_transforms)
    assert len(h) == 8
