"""Config-driven pipelines of transform steps, built from serializable configs.

ConfigurableTransform is the base class for a single step. TransformPipeline
runs an ordered, named sequence of inference-only steps, so it can be
serialized alongside a model. TransformHooks attaches extra training-only
steps (e.g. label attachment, augmentation) without changing the pipeline
itself, so they can be serialized alongside the data pipeline instead.

A filter pipeline is just a TransformPipeline[bool].
"""

import hashlib
import json
from collections import defaultdict
from typing import Any, Literal, Self, assert_never

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mlbricks.configurable import Configurable, ConfigurableError
from mlbricks.registry import Registry
from mlbricks.resolve_config import to_dict

__all__ = [
    "ConfigurableTransform",
    "Context",
    "NamedTransformStep",
    "TransformHookEntry",
    "TransformHooks",
    "TransformPipeline",
]

# Mutable state dict threaded through pipeline steps within a single call.
type Context = dict[str, Any]

# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------


class ConfigurableTransform[P](Configurable):
    """Base class for a single pipeline step. Subclass and override __call__.

    Example (filtering, Payload=bool):
        >>> class MinLength(ConfigurableTransform[bool]):
        ...     class Config(ConfigurableTransform.Config):
        ...         min_length: int = 10
        ...
        ...     def __call__(self, payload: bool, ctx: Context) -> tuple[bool, Context]:
        ...         return payload and len(ctx["sequence"]) >= self.cfg.min_length, ctx

    Example (computing a feature):
        >>> class AddValueSum(ConfigurableTransform[dict[str, float]]):
        ...     class Config(ConfigurableTransform.Config):
        ...         key: str = "value_sum"
        ...
        ...     def __call__(
        ...         self, payload: dict[str, float], ctx: Context
        ...     ) -> tuple[dict[str, float], Context]:
        ...         payload[self.cfg.key] = sum(ctx["values"])
        ...         return payload, ctx
    """

    class Config(Configurable.Config): ...

    def __init__(self, cfg: "ConfigurableTransform.Config") -> None:
        self.cfg = cfg

    def __call__(self, payload: P, ctx: Context) -> tuple[P, Context]:
        """Apply this transform, returning the (possibly modified) payload and ctx."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


class _BeforeRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    before: str


class _AfterRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    after: str


class TransformHookEntry(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    at: Literal["start", "end"] | _BeforeRef | _AfterRef
    do: ConfigurableTransform.Config


class TransformHooks(Configurable):
    """Training-only steps attached separately from the inference pipeline.

    Hooks attach relative to inference steps for training-only tasks like
    attaching labels or augmenting data.
    """

    class Config(Configurable.Config):
        entries: list[TransformHookEntry] = Field(default_factory=list)

    def __init__(self, cfg: "TransformHooks.Config") -> None:
        # eg {("start", None): [<Transforms>]}
        # or {("after", <step_name>): [<Transforms>]}
        self._index: defaultdict[
            tuple[str, str | None],
            list[ConfigurableTransform],
        ] = defaultdict(list)

        for entry in cfg.entries:
            if isinstance(entry.at, str):
                key = (entry.at, None)

            elif isinstance(entry.at, _BeforeRef):
                key = ("before", entry.at.before)

            elif isinstance(entry.at, _AfterRef):
                key = ("after", entry.at.after)

            else:
                assert_never(entry.at)

            self._index[key].append(entry.do.build())

    def get(self, at: str, val: str | None = None) -> list[ConfigurableTransform]:
        """Look up hooks registered at a given lifecycle position.

        Args:
            at: The lifecycle position: "start", "end", "before", or "after".
            val: For "before"/"after", the name of the step being referenced.
                Ignored (leave as None) for "start"/"end".

        Returns:
            Hooks registered at this position, in registration order. Empty
            list if none are registered.
        """
        return self._index.get((at, val), [])

    def named_refs(self) -> set[str]:
        """Return the step names these hooks reference."""
        return {
            val
            for (at, val) in self._index
            if at in ("before", "after") and val is not None
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class NamedTransformStep(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    do: ConfigurableTransform.Config


class TransformPipeline[P](Configurable):
    """Ordered, named ConfigurableTransform steps that run as one pipeline.

    Example:
        >>> pipeline = TransformPipeline.Config(
        ...     steps=[{"name": "add_value_sum", "do": AddValueSum.Config()}]
        ... ).build()
        >>> pipeline({}, {"values": [1.0, 2.0]})
        # ({"value_sum": 3.0}, {"values": [1.0, 2.0]})

    Hooks attach training-only steps without changing the pipeline itself:

        >>> pipeline.with_hooks(
        ...     TransformHooks.Config(
        ...         entries=[{"at": "end", "do": AddValueSum.Config(key="value_sum_2")}]
        ...     )
        ... )
        >>> pipeline({}, {"values": [1.0, 2.0]})
        # ({"value_sum": 3.0, "value_sum_2": 3.0}, {"values": [1.0, 2.0]})

    A filter pipeline is just TransformPipeline[bool]: each step ANDs its
    result into the payload, so the pipeline overall reads as one predicate.

        >>> pipeline = TransformPipeline.Config(
        ...     steps=[{"name": "min_length", "do": MinLength.Config(min_length=10)}]
        ... ).build()
        >>> pipeline(True, {"sequence": "abc"})
        # (False, {"sequence": "abc"})
    """

    class Config(Configurable.Config["TransformPipeline"]):
        steps: list[NamedTransformStep] = Field(default_factory=list)

        @model_validator(mode="after")
        def _check_unique_names(self) -> Self:
            names = [step.name for step in self.steps]

            duplicates = {name for name in names if names.count(name) > 1}
            if duplicates:
                raise ValueError(f"duplicate step name(s) {duplicates!r}")

            return self

    def __init__(self, cfg: "TransformPipeline.Config") -> None:
        self._cfg = cfg
        self._steps = [(step.name, step.do.build()) for step in cfg.steps]

        # Default hook source before with_hooks() is called: no-op lookups.
        self._hooks = TransformHooks(TransformHooks.Config())
        self._hooks_cfg: TransformHooks.Config | None = None

    def __call__(self, payload: P, ctx: Context) -> tuple[P, Context]:
        """Run every step, threading payload and ctx through each.

        Run order:
            1. start hooks
            2. for each step: before-hooks, the step, after-hooks
            3. end hooks

        Args:
            payload: The dataset sample or training input being transformed
                as it moves through the pipeline, e.g. a bool for a filtering
                pipeline, or a tensor/Data object being featurized.
            ctx: Scratch pad threaded alongside payload. Holds raw input data
                and intermediate calculations any step may need.

        Returns:
            The final (payload, ctx) after all steps and hooks have run.
        """
        for hook in self._hooks.get("start"):
            payload, ctx = hook(payload, ctx)

        for name, step in self._steps:
            for hook in self._hooks.get("before", name):
                payload, ctx = hook(payload, ctx)

            # transform between hooks
            payload, ctx = step(payload, ctx)

            for hook in self._hooks.get("after", name):
                payload, ctx = hook(payload, ctx)

        for hook in self._hooks.get("end"):
            payload, ctx = hook(payload, ctx)

        return payload, ctx

    def with_hooks(self, hooks_cfg: "TransformHooks.Config") -> None:
        """Attach hooks to this pipeline, replacing any previously attached.

        Args:
            hooks_cfg: Config for the hooks to attach.

        Raises:
            ConfigurableError: If a hook's before/after entry names a step
                that does not exist in this pipeline.
        """
        hooks = hooks_cfg.build()

        valid = {name for name, _ in self._steps}
        unknown = hooks.named_refs() - valid
        if unknown:
            raise ConfigurableError(f"hook references unknown step(s) {unknown!r}")

        self._hooks = hooks
        self._hooks_cfg = hooks_cfg

    def get_hash(self, registry: Registry[object]) -> str:
        """Return an 8-char hex digest identifying this pipeline's run state.

        Hashes pipeline execution, i.e. the base steps plus whatever hooks
        are currently attached. Call before with_hooks() to exclude hooks
        from the hash, or after to include them.

        Hashes the steps/hook entries themselves rather than the pipeline's
        or hooks' own Config -- only each step's transform class needs to be
        registered, not TransformPipeline/TransformHooks.

        Args:
            registry: Registry used to resolve step/hook config classes for
                hashing.

        Returns:
            An 8-character hash.
        """
        payload = {
            "pipeline": to_dict(self._cfg.steps, registry=registry),
            "hooks": to_dict(self._hooks_cfg.entries, registry=registry)
            if self._hooks_cfg is not None
            else None,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode())
        return digest.hexdigest()[:8]
