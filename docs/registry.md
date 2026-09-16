# Registry

Source: `src/mlbricks/registry.py`

## Motivation

`Registry` allows us to swap components frequently in experiments, eg a
different GNN encoder, a new loss function, an alternative featurizer. If
consuming code imports those classes directly, every swap requires a code
change.

Using `Registry` we can register components under a string name at import time.
Consuming code resolves a name to a class at runtime. The string can come from
a config object, a CLI flag, or anywhere else -- the lookup site does not need
to know which module the class lives in.

## Concept

A `Registry[T]` holds two kinds of entries:

- **Leaf classes** -- concrete implementations registered with `.register()`.
- **Subgroups** -- child `Registry` instances created with `.subgroup()`,
  used to organize entries into namespaces.

Entries are addressed with dotted paths: `"models.gat"` traverses the
`models` subgroup and returns the `GAT` class.

`mlbricks` does not ship a pre-built singleton registry. Construct one
`Registry[Any]` instance as your project's root and pass it around; every
consumer owns its own instance.

---

## Full example

```python
# myproject/registry.py
from typing import Any

from mlbricks import Registry

REGISTRY: Registry[Any] = Registry()
MODEL_REGISTRY = REGISTRY.subgroup("models")
```

```python
# myproject/models/gat.py
from torch import nn

from mlbricks import Configurable
from myproject.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("gat")
class GAT(nn.Module, Configurable):
    class Config(Configurable.Config["GAT"]):
        hidden_dim: int = 64

    def __init__(self, cfg: "GAT.Config") -> None:
        super().__init__()
        self.hidden_dim = cfg.hidden_dim
```

```python
# myproject/models/__init__.py
from myproject.models.gat import GAT  # re-export; registration is a side effect
```

```python
# consuming code
from myproject.registry import REGISTRY
import myproject.models  # populates REGISTRY as a side effect

cls = REGISTRY.get("models.gat")  # returns GAT (uninstantiated)
model = cls.Config(hidden_dim=64).build()
```

Three things to note:

1. **Subgroup globals are uppercased** (`MODEL_REGISTRY`), consistent with
   the root `REGISTRY`. They are module-level singletons owned by your
   project.
2. **`.get()` always returns the raw registered class, never a `Config`.**
   `Registry` has no awareness of `Configurable` -- for a `Configurable`
   subclass you build a `Config` from the returned class yourself (as
   above); for a plain class you call it directly. Only `resolve()`
   (see [Config Resolution](config-resolution.md)) special-cases
   `Configurable` subclasses to build `cls.Config(**kwargs)` for you.
3. **Registration happens at import time.** Something in your project must
   import each module that registers a class before `.get()` can resolve
   it -- see [Import order](#import-order) below.

---

## Special cases

### `__contains__`

`path in registry` returns `True` only for resolved leaf classes -- not for
subgroup names.

```python
"models.gat" in REGISTRY  # True  -- GAT is a leaf
"models" in REGISTRY  # False -- models is a subgroup
"models.xyz" in REGISTRY  # False -- not registered
```

Use this to check membership before calling `.get()` when presence is uncertain.

### Type constraint

`Registry.register()` places no runtime type constraint on `cls` -- any class
can be registered.

If you also use `resolve_config.to_dict()` to serialize configs back to their
declarative `_registry_` form (see [Config Resolution](config-resolution.md)),
keep registered classes to `Configurable` subclasses and plain dataclasses:
those are the only types `to_dict()` knows how to round-trip, and it raises
`ConfigSerializationError` for anything else. `mlbricks` does not enforce this
for you; a project-level test that walks its own registry tree is one way to
catch violations early. For external classes you don't own, see
[External classes](#external-classes) below instead of registering a
hand-rolled subclass.

### Import order

Registration happens at import time. If a module that populates your registry
has not been imported yet, `.get()` will raise `RegistryKeyError` even if the
class exists somewhere in the codebase.

**Convention:** each package that owns a registry subgroup can follow a
three-file pattern:

- `registry.py` -- creates the subgroup and exports it as an uppercase global
  (e.g. `MODEL_REGISTRY`). Imports only the root `Registry` instance.
- Class modules (e.g. `gat.py`) -- import the subgroup from `registry.py` and
  use `@MODEL_REGISTRY.register(...)` to decorate their classes.
- `__init__.py` -- re-exports the public classes from the class modules.
  Registration is a side effect of those imports.
- Your project's top-level `__init__.py` imports each subpackage (e.g.
  `from myproject import models`), chaining the re-export upward so that
  importing your package alone populates the entire registry.

This keeps the dependency chain acyclic:

```
myproject/__init__.py  ->  models/__init__.py  ->  gat.py  ->  models/registry.py  ->  myproject.registry
```

`registry.py` is also a single place to see every subgroup the package owns. See the [full example](#full-example) above for a complete illustration.

### Nested subgroups

Subgroups can themselves own subgroups, creating multi-level namespaces. This
is useful when a subsystem has several distinct component categories that each
need their own registry.

```python
# myproject/featurization/registry.py
from myproject.registry import REGISTRY

FEATURIZATION_REGISTRY = REGISTRY.subgroup("featurization")
FEATURIZER_REGISTRY = FEATURIZATION_REGISTRY.subgroup("featurizers")
HOOKS_REGISTRY = FEATURIZATION_REGISTRY.subgroup("hooks")
TRANSFORMS_REGISTRY = FEATURIZATION_REGISTRY.subgroup("transforms")
FILTERS_REGISTRY = FEATURIZATION_REGISTRY.subgroup("filters")
```

```python
# myproject/featurization/featurizers/geometric.py
from myproject.featurization.registry import FEATURIZER_REGISTRY


@FEATURIZER_REGISTRY.register("geometric")
class GeometricFeaturizer: ...
```

The resulting dotted path is `"featurization.featurizers.geometric"` -- one
segment per level of nesting. The same `REGISTRY.get()` call resolves it:

```python
cls = REGISTRY.get("featurization.featurizers.geometric")
```

All subgroup globals follow the same uppercase naming convention as the root
(`FEATURIZER_REGISTRY`, `HOOKS_REGISTRY`, etc.). Collect them in a single
`registry.py` at the subsystem root so every subgroup the package owns is
visible in one place.

### Error types

| Exception | When it fires |
|---|---|
| `InvalidRegistryNameError` | A name passed to `.register()` or `.subgroup()` is empty or contains a period. |
| `DuplicateRegistrationError` | A name is already registered at that registry level -- whether as a leaf or a subgroup -- or the same class is registered under a second path. |
| `RegistryKeyError` | A dotted path cannot be resolved: a segment is missing, an intermediate segment is a leaf class, or the final segment is a subgroup rather than a leaf. |

`RegistryKeyError` is a subclass of `KeyError`. Its `__str__` returns the
message directly (without `KeyError`'s extra quoting), so it prints cleanly in
tracebacks.

---

## External classes

For a class you don't own -- `torch.optim.Adam`, `sklearn`'s estimators,
anything from a third-party library -- hand-writing a `Configurable` subclass
just to wire a handful of constructor kwargs is unnecessary boilerplate. Use
`_magic_registry_` instead of `_registry_`:

```yaml
optimizer:
  _magic_registry_: torch.optim.Adam
  lr: 0.001
  weight_decay: 0.0001
```

`_magic_registry_` takes a dotted **import path**, not a curated registry
name -- no `.register()` call needed. `resolve()` imports the class, builds a
`Configurable` wrapper on the fly with a `Config` field for each YAML key
(typed from `Adam.__init__`'s own annotations, so `lr: "oops"` fails
validation before `Adam.__init__` ever runs), and subclasses `Adam` directly
-- the built object *is* a real `torch.optim.Adam`, not a wrapper around one.

Anything the class needs that isn't in the YAML -- `params`, most commonly,
since parameters are a live object, not config -- is supplied at `.build()`
time, exactly like any other `Configurable`'s owner kwargs:

```python
from mlbricks import resolve

cfg = resolve(
    {"optimizer": {"_magic_registry_": "torch.optim.Adam", "lr": 1e-3}},
    registry=REGISTRY,
)
optimizer = cfg["optimizer"].build(params=model.parameters())
```

It round-trips through `to_dict()` the same way `_registry_` entries do:

```python
to_dict(cfg["optimizer"], REGISTRY)
# {"_magic_registry_": "torch.optim.Adam", "lr": 0.001}
```

See [Config Resolution](config-resolution.md#magic-registry-entries) for the
full contract and error types.

### When `_magic_registry_` isn't enough

Fall back to a hand-written `Configurable` subclass (see the
[full example](#full-example) above) when:

- The external class's `__init__` isn't introspectable (e.g. some
  C-extension types) *and* you need real type validation on the fields --
  `_magic_registry_` still works here, but silently drops to unvalidated
  `Any` fields.
- You need cross-field validation (`@model_validator`), deferred fields
  (`T | None`), or `BuildTimeField[T]` semantics -- `_magic_registry_`
  Configs are flat, one field per YAML key, no validators.
- The external class doesn't compose cleanly with multiple inheritance
  (unusual `__new__`/metaclass behavior) -- `_magic_registry_` subclasses
  `(target_cls, Configurable)` directly, which assumes a plain-Python
  `__init__` path like `nn.Module` subclasses have.

---

## When to use `Registry`

Use a registry for any class that needs to be selected by name at runtime --
when the choice of implementation is deferred to config time rather than baked
into the calling code.

Common cases:

| Component type | Example names |
|---|---|
| GNN encoders | `"models.trunks.gat"` |
| MLP heads | `"models.mlp"` |
| Loss functions | `"losses.binary_cross_entropy"` |
| Featurizers | `"featurization.featurizers.geometric"` |
| Hooks | `"featurization.hooks.geometric"` |

If you always know which class to use at write time, a plain import is fine.
`Registry` pays off when the class name needs to travel -- through a config
object, a sweep parameter, or a CLI flag -- and the lookup site should not care
where the class lives.
