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
| `RegistryParseError` | A `_registry_` value is not a string. |
| `ConfigInstantiationError` | A class constructor raises during resolution. Wraps the original exception via `__cause__`. |
| `ConfigSerializationError` | `to_dict()` encounters an unregistered owner or an unsupported type. |

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

**Serializable types are `Configurable.Config`, plain `BaseModel` subclasses,
registered plain dataclasses, `dict`, `list`, and scalars.** Anything outside
this set -- an unregistered class or dataclass, a live model instance, a
`torch.Tensor` -- raises `ConfigSerializationError`. Keep config trees to plain data.

Plain `BaseModel` subclasses (those that do not inherit from `Configurable.Config`)
are serialized by iterating their declared fields and recursing into each value.
This covers inline schema types like entry models that appear as list elements
in `Configurable.Config` fields.
