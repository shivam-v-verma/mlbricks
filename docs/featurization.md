# Featurization

Source: `src/mlbricks/featurization.py`

## Motivation

Featurization is the step between raw input and whatever the model consumes:
tokenizing text, cropping images, computing molecular graph features. It
typically breaks down into an ordered sequence of small steps, each one
reading prior state and producing a further piece of the object the model
needs.

Two things follow from that:

- The sequence should be declarable, not hardcoded, so a different pipeline
  is a config change rather than a code change (see `Configurable`,
  `Registry`).
- Training adds extra steps -- label attachment, augmentation, normalization
  -- that should not change what the model expects to receive at inference
  time.

Each step has the same signature regardless of domain: it takes a `payload`
and a `ctx`, and returns both. `ConfigurableTransform` and `TransformPipeline`
implement that as an ordered, named sequence of steps, declared through
`Configurable` like any other component.

## Design principles

**`ctx` is a scratchpad; `payload` is the thing being built.** The caller
preloads `ctx` with whatever external inputs the pipeline needs -- a
dataframe row, a raw string, label values. Steps read and write `ctx` as they
run, using it to hold intermediate calculations that later steps consume.
`payload` is the object actually under construction -- a tensor, a tuple of
tensors, a graph `Data` object, or, for a filter, a `bool`. Every step's
signature is `(payload, ctx) -> (payload, ctx)`; `ctx` does not outlive the
call, `payload` is what the pipeline returns.

**The model owns inference; hooks own training.** A model's forward pass is
defined for inputs featurized a specific way, so the pipeline that produces
those inputs is part of the model's identity: it is serialized with the model
and read by any consumer -- dataset, dataloader, inference server -- rather
than reimplemented by each one. Steps needed only during training -- label
attachment, augmentation, output normalization -- are not part of that
identity: they are attached separately, at training time, and dropped
afterward. `TransformHooks` implements that separation: hooks are attached to
a built pipeline via `with_hooks()` and are not part of the pipeline's own
config or serialized form. Rule of thumb: a step the model needs to produce
correct output belongs in the pipeline; a step needed only during training
belongs in hooks.

**`get_hash()` identifies whether featurization would produce the same
data.** Datasets cache featurization output because it is often the expensive
part of a run; that cache is valid only for as long as the featurization
producing it is unchanged. `get_hash()` hashes the pipeline's config together
with whatever hooks are currently attached into an 8-character digest, usable
as (part of) a cache key. See `StorageCache.get_or_compute()` in
[storage.md](storage.md).

---

## Concept

`ConfigurableTransform[P]` is the base class for a single step: a
`Configurable` whose `__call__` takes `(payload: P, ctx: Context)` and
returns `(payload, ctx)`. `P` is the payload type -- the same base class
works for any of it, for example:

- a tensor, e.g. accumulating image features
- a graph `Data` object, e.g. accumulating molecular features
- a `bool`, for a filtering decision

`TransformPipeline[P]` runs an ordered list of named `ConfigurableTransform[P]`
steps in sequence, threading `payload` and `ctx` through each in turn. Steps
are named (`{"name": ..., "do": ...}`) so hooks can address them by name.

`TransformHooks` holds steps attached to specific lifecycle points around a
pipeline -- `"start"`, `"end"`, `{"before": step_name}`, `{"after": step_name}`
-- kept separate from the pipeline's own config. `pipeline.with_hooks(hooks_cfg)`
attaches them; `pipeline.get_hash()` reflects whichever hooks are currently
attached at the time it's called.

---

## Full example

```python
# transforms.py
from mlbricks import ConfigurableTransform
from mlbricks.featurization import Context
from myproject.registry import TRANSFORMS_REGISTRY

type Payload = dict[str, float]


@TRANSFORMS_REGISTRY.register("normalize")
class Normalize(ConfigurableTransform[Payload]):
    """Reads ctx[input_key], normalizes it, writes payload[output_key]."""

    class Config(ConfigurableTransform.Config):
        input_key: str = "value"
        output_key: str = "x"
        mean: float = 0.0
        std: float = 1.0

    cfg: Config

    def __call__(self, payload: Payload, ctx: Context) -> tuple[Payload, Context]:
        payload[self.cfg.output_key] = (
            ctx[self.cfg.input_key] - self.cfg.mean
        ) / self.cfg.std
        return payload, ctx


@TRANSFORMS_REGISTRY.register("attach_scalar")
class AttachScalar(ConfigurableTransform[Payload]):
    """Copies ctx[input_key] straight to payload[output_key], unnormalized."""

    class Config(ConfigurableTransform.Config):
        input_key: str
        output_key: str = "y"

    cfg: Config

    def __call__(self, payload: Payload, ctx: Context) -> tuple[Payload, Context]:
        payload[self.cfg.output_key] = ctx[self.cfg.input_key]
        return payload, ctx
```

```yaml
# pipeline.yaml -- inference-time; belongs to the model, gets serialized with it
pipeline:
  _registry_: pipelines.transform
  steps:
    - name: normalize
      do:
        _registry_: transforms.normalize
        input_key: value
        output_key: x
        mean: 12.0
        std: 4.0
```

```yaml
# hooks.yaml -- training-time only; never serialized with the model
hooks:
  _registry_: hooks.transform
  entries:
    - at: end
      do:
        _registry_: transforms.attach_scalar
        input_key: label
        output_key: y
```

```python
from mlbricks import resolve
from myproject.registry import REGISTRY
import myproject.transforms  # populate REGISTRY as a side effect
from omegaconf import OmegaConf

pipeline = resolve(
    OmegaConf.to_object(OmegaConf.load("pipeline.yaml")), registry=REGISTRY
)["pipeline"].build()

ctx = {"value": 14.0, "label": 1.0}
payload, ctx = pipeline({}, ctx)
# payload == {"x": 0.5}  -- what the model will see at inference

hooks_cfg = resolve(
    OmegaConf.to_object(OmegaConf.load("hooks.yaml")), registry=REGISTRY
)["hooks"]
pipeline.with_hooks(hooks_cfg)
payload, ctx = pipeline({}, ctx)
# payload == {"x": 0.5, "y": 1.0}  -- "y" only appears once hooks are attached
```

The pipeline config is what a model checkpoint would serialize; the hooks
config never appears in it. Swapping `pipeline.yaml` for a different one
changes what the model expects, and is a config change, not a code change.

`TransformPipeline` isn't tied to this payload shape either -- a
`TransformPipeline[bool]` built from the same `Normalize`-style steps is a
filter pipeline; see [Filters](#filters) below.

---

## Things to note

1. **Steps are addressable by name.** `{"name": ..., "do": ...}` in
   `TransformPipeline.Config.steps` is what lets `TransformHooks` target
   `{"before": name}` / `{"after": name}`; duplicate names are rejected at
   config-construction time.
2. **`with_hooks()` mutates the built pipeline, not its config.** It
   validates that every `before`/`after` reference in the hooks names an
   actual step (raising `ConfigurableError` otherwise), then replaces any
   previously attached hooks. The pipeline's own `Config` is untouched --
   hooks never enter it.
3. **The same pipeline object answers differently before and after
   `with_hooks()`.** Calling it runs whatever hooks are currently attached;
   calling `get_hash()` reflects that same current state. There's no
   separate "hooked" vs "unhooked" type -- one object, two states over its
   lifetime (build, then optionally `with_hooks()`).

---

## Special cases

### Filters

A filter is `TransformPipeline[bool]` -- there's no separate filter class.
Call it starting from `payload=True`; each step ANDs its own result into the
running payload:

```python
class MinLength(ConfigurableTransform[bool]):
    class Config(ConfigurableTransform.Config):
        min_length: int = 10

    cfg: Config

    def __call__(self, payload: bool, ctx: Context) -> tuple[bool, Context]:
        return payload and len(ctx["value"]) >= self.cfg.min_length, ctx


keep, ctx = filter_pipeline(True, ctx)
```

Starting from `True` looks a little odd next to a dedicated `.matches(ctx)`-
style entrypoint, but there isn't one -- a filter pipeline has the same
`__call__(payload, ctx)` signature as any other `TransformPipeline`, called
with the identity value for AND.

### Composing multiple stages

`TransformPipeline` has no notion of "prefilter", "pretransform", or
"transform" -- those are stage names a caller assigns to *separate*
`TransformPipeline` instances (a `TransformPipeline[bool]` to decide
inclusion, one or more `TransformPipeline[SomeDataT]` to build the object),
orchestrated by a bit of code above this library. Compose as many stages as
the domain needs -- one, three, or any other number.

### `get_hash()` and caching

`get_hash(registry)` hashes `to_dict()` of the pipeline's config together
with the currently-attached hooks' config (or `None`) into an 8-character
digest. Two pipelines built from equal configs with equal hooks attached hash
identically. It identifies the *featurization*, not the data it's applied to
-- two different input rows run through the same pipeline hash identically.
Combine it with something that identifies the input (e.g.
`uri_slug(data_uri)`) to build a correct cache key for
`StorageCache.get_or_compute()`:

```python
key = f"{uri_slug(data_uri)}-{pipeline.get_hash(registry)}"
data = cache.get_or_compute(key, lambda: pipeline(payload, ctx))
```

Because hooks change the hash, a training run with label-attachment or
augmentation hooks attached caches separately from one without them, and from
a run using a different pipeline entirely.

---

## When to use

| If you need to... | Use... |
|---|---|
| Decide whether a sample belongs in a dataset | A `TransformPipeline[bool]` (a filter) |
| Build the object a model consumes, step by step | A `TransformPipeline[SomeDataT]` |
| Attach labels, augment, or normalize only during training | `TransformHooks` + `pipeline.with_hooks()` |
| Cache-key a pipeline's expensive output | `pipeline.get_hash(registry)` |

If a processing step is fixed for the life of the project and never swapped,
a plain function is fine -- `ConfigurableTransform`/`TransformPipeline` pay
off once the sequence needs to travel through a config, or training needs to
add steps the model itself should never see.
