# Config Resolution

Source: `src/mlbricks/resolve_config.py`

## Motivation

YAML configs describe components by name, not by import path. A training
config might say `_registry_: models.trunks.gat` without the training script
knowing where `GAT` lives or how to import it.

`resolve()` bridges that gap. It walks a plain Python dict (e.g. from
`OmegaConf.to_object()`), finds every `{"_registry_": "...", ...}` node,
looks up the class in the provided registry, and constructs it -- returning
a dict where every `_registry_` node has been replaced by a constructed object.

---

## Full example

```python
from mlbricks import resolve
from myproject.registry import REGISTRY
import myproject.models, myproject.losses  # populate REGISTRY as a side effect

cfg = resolve(
    {
        "model": {
            "_registry_": "models.trunks.gat",
            "hidden_dim": 128,
            "num_layers": 3,
        },
        "loss": {
            "_registry_": "losses.binary_cross_entropy",
        },
        "seed": 42,
    },
    registry=REGISTRY,
)

# cfg["model"] is a GAT.Config(hidden_dim=128, num_layers=3)
# cfg["loss"] is a BinaryCrossEntropy.Config()
# cfg["seed"] is 42 -- plain values pass through unchanged
```

Three things to note:

1. **The modules that populate your registry must be imported first.**
   Registration happens at import time; if a registry key is not yet
   imported, `resolve()` raises `RegistryKeyError`.
2. **`Configurable` subclasses yield `Config` instances, not live objects.**
   Call `.build()` yourself -- usually inside the owner's `__init__` -- to
   control when runtime fields (e.g. `input_dim`) are resolved.
3. **Plain values pass through.** Scalars, lists, and nested dicts without
   `_registry_` are returned unchanged.

---

## Error types

| Exception | When it fires |
|---|---|
| `RegistryKeyError` | A `_registry_` key is not found in the registry. |
| `RegistryParseError` | A `_registry_`/`_magic_registry_` value is not a string, or a node has both keys. |
| `ConfigInstantiationError` | A class constructor raises during resolution. Wraps the original exception via `__cause__`. |
| `ConfigSerializationError` | `to_dict()` encounters an unregistered owner or an unsupported type. |
| `MagicRegistryImportError` | A `_magic_registry_` path can't be imported, or doesn't resolve to a class. |
| `MagicRegistryUnknownFieldError` | A `_magic_registry_` field has no matching `__init__` parameter on the target class. |

`RegistryKeyError` is most commonly caused by a missing import of the module
that registers the class, before calling `resolve()`.

---

## When to use `resolve()`

Call `resolve()` at the top of a training script, after loading your YAML
config into a plain dict:

```python
from mlbricks import resolve
from myproject.registry import REGISTRY
from omegaconf import OmegaConf

raw = OmegaConf.to_object(OmegaConf.load("config.yaml"))
cfg = resolve(raw, registry=REGISTRY)
```

YAML-driven resolution is the standard pattern for reproducible training
runs. Plain imports and direct construction are fine for quick local
experiments, but should be migrated to YAML configs before results are
tracked -- a config file is the authoritative record of what ran.

---

## Serializing configs with `to_dict()`

`to_dict(value, registry)` is the strict inverse of `resolve()` -- it converts
a `Configurable.Config` back to the declarative dict format, and `resolve()`
returns the Config object directly when given that format back. Scalars, dicts,
and lists are handled recursively so the entire config tree serializes in one
call. The primary use case is checkpointing: frameworks like Lightning let you
save hyperparameters as a plain dict and reconstruct the full config on load.

```python
# direct round-trip
serialized = to_dict(original, registry=REGISTRY)
restored = resolve(serialized, registry=REGISTRY)
# isinstance(restored, type(original))  # True
```

Both forms are valid. The envelope form -- `{"model": to_dict(cfg, reg)}` --
is the standard YAML config shape and still works: `resolve()` returns a plain
dict when the top-level dict has no `_registry_` key.

### Limitations

**The owner class must appear in the registry you pass.** `to_dict()` looks
up the dotted path for each `Configurable` subclass in the provided registry
to reconstruct the `_registry_` key. If you pass a different registry than
the one used at `resolve()` time -- or a class was never registered -- it
raises `ConfigSerializationError`.

**Deferred fields serialize as `null` until you assign them.** Fields
declared as `T | None = None` serialize as `null` until set on the Config
instance. To capture a resolved value in the serialized output, assign it
directly before calling `to_dict()` -- `Config` is mutable by default:

```python
cfg = MLP.Config(hidden_dim=128)
cfg.input_dim = 64  # set deferred field directly
serialized = to_dict(cfg, registry=REGISTRY)
# serialized["input_dim"] == 64
mlp = cfg.build()  # also works -- build() sees the assigned value
```

Note: direct assignment does not re-run field validators (`validate_assignment`
is not enabled). Type enforcement happens at `build()` time via `@build_validator`.

**`BuildTimeField[T]` fields are omitted entirely, not serialized as `null`.**
Unlike `T | None` deferred fields, a field declared with `BuildTimeField[T]`
never appears in `to_dict()`'s output -- there is no key to omit-or-include,
by design, since the field is meant to hold a live object, not something
intended to survive a round trip. See
[Build-time-only fields](configurable.md#build-time-only-fields) for the full
contract.

**Serializable types are `Configurable.Config`, plain `BaseModel` subclasses,
registered plain dataclasses, `dict`, `list`, and scalars.** Anything outside
this set -- an unregistered class or dataclass, a live model instance, a
`torch.Tensor` -- raises `ConfigSerializationError`. Keep config trees to plain data.

Plain `BaseModel` subclasses (those that do not inherit from `Configurable.Config`)
are serialized by iterating their declared fields and recursing into each value.
This covers inline schema types like entry models that appear as list elements
in `Configurable.Config` fields.

---

## Magic registry entries

`_magic_registry_` is `_registry_`'s counterpart for classes you haven't
registered -- see [External classes](registry.md#external-classes) for the
motivating example. The two keys are mutually exclusive on one node;
supplying both raises `RegistryParseError`.

Differences from `_registry_` nodes:

- The value is a dotted **import path** (`"torch.optim.Adam"`), not a name
  registered via `Registry.register()`. No import-order requirement like
  `_registry_` has (see [Import order](registry.md#import-order)) -- the
  class is imported on demand by `resolve()` itself.
- The Config is synthesized on the fly with exactly the sibling keys present
  in the node as fields -- there's no fixed schema to consult ahead of time.
  A key with no matching `__init__` parameter (and no `**kwargs` catch-all on
  the target) raises `MagicRegistryUnknownFieldError` immediately.
- Fields are typed from the target's own `__init__` annotations when
  resolvable, so a type mismatch (`lr: "oops"` against `lr: float`) raises
  `ConfigValidationError` at `Config()` construction -- before the target's
  constructor ever runs. (Through `resolve()`, this surfaces as
  `ConfigInstantiationError` with the `ConfigValidationError` as `__cause__`,
  same as any other instantiation failure -- see the error table above.)
  Unannotated or unintrospectable parameters fall back to unvalidated `Any`.
- Round-trips as `_magic_registry_` in `to_dict()`'s output, not `_registry_`
  -- no `Registry.path_of()` lookup involved, so it works even against a
  `Registry` instance that never saw the class.

**Trust assumption:** unlike `_registry_`, which only reaches classes an
owning project explicitly `.register()`s, `_magic_registry_` will import and
instantiate *any* dotted path reachable in the running interpreter. Treat
config trees the same way you'd treat code -- developer-authored and
reviewed, not sourced from untrusted input.
