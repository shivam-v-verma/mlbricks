# Configurable

Source: `src/mlbricks/configurable.py`

## Motivation

ML objects -- models, datasets, losses, optimizers -- have many construction
parameters. We want to specify them declaratively: in a YAML file, a script,
or a config object passed around the codebase, rather than threading `__init__`
kwargs everywhere.

`Configurable` is the pattern that bridges that gap. A config object fully
describes how to build a class. It can be serialized, composed with other
configs, and converted to a live object on demand by calling `.build()`.

## Concept

Every `Configurable` class owns a nested `Config` pydantic model:

- The **config** holds all construction parameters. It is what you store in a
  YAML file and pass around the codebase.
- Calling `Config.build()` validates the config and instantiates the owner.

Use pydantic's native validation decorators for invariants:

- `@field_validator` and `@model_validator(mode="after")` run at both
  construction and build time -- use these for type coercions and cross-field
  invariants.
- `@build_validator` runs only during `build()` -- use it for checks that
  require fully-resolved values, such as deferred fields.

---

## Full example

```python
from typing import Self
from torch import nn
from pydantic import model_validator
from mlbricks import Configurable, ConfigValidationError


class MLP(nn.Module, Configurable):  # Configurable last -- preserves nn.Module MRO
    class Config(Configurable.Config["MLP"]):
        hidden_dim: int = 256
        output_dim: int = 128

        @model_validator(mode="after")
        def check_dims(self) -> Self:
            if self.hidden_dim <= 0:
                raise ValueError("hidden_dim must be positive")
            if self.output_dim <= 0:
                raise ValueError("output_dim must be positive")
            return self

    def __init__(self, cfg: "MLP.Config") -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.hidden_dim, cfg.output_dim)


# Construct the config (design time)
cfg = MLP.Config(hidden_dim=128)

# Build the object (runtime)
mlp = cfg.build()
```

Three things to note:

1. **`Configurable` comes last** in the base class list. Other bases like
   `nn.Module` must retain MRO priority, or their `__init__` machinery breaks.
2. **Validators** use pydantic's `@model_validator` or `@build_validator`.
   Raise `ValueError` inside validators; both Config construction and `build()`
   translate any `pydantic.ValidationError` to `ConfigValidationError`.
   `ConfigValidationError` exposes `.errors()` and `.error_count()` for
   structured field-level inspection.
3. **`__init__` takes `cfg`** as its sole construction argument (beyond `self`).
   Unpack what you need from it.

---

## Special cases

### Deferred fields

Some fields are data-dependent and unknown until build time -- for example,
`input_dim` that comes from inspecting a dataset. Declare them as
`T | None = None` and supply them as kwargs to `.build()`. Use `@build_validator`
to enforce that they are non-None at build time.

```python
from typing import Self
from torch import nn
from mlbricks import Configurable, build_validator


class MLP(nn.Module, Configurable):
    class Config(Configurable.Config["MLP"]):
        input_dim: int | None = None  # supplied at build time
        hidden_dim: int = 256
        output_dim: int = 128

        @build_validator
        def check_input_dim(self) -> Self:
            if self.input_dim is None:
                raise ValueError("input_dim required at build time")
            return self

    def __init__(self, cfg: "MLP.Config") -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.input_dim, cfg.output_dim)


cfg = MLP.Config(hidden_dim=128)
mlp = cfg.build(input_dim=64)  # input_dim resolved here
```

If a required deferred field is not supplied, `build()` raises
`ConfigValidationError`. Call `.errors()` on it for field-level detail.

### Build-time-only fields

Some fields hold live, non-serializable objects -- a `DataLoader`, an
`nn.Module`, a logger -- that have no business round-tripping through YAML.
(If you're wrapping an *external* class you don't own just to get here, check
[External classes](registry.md#external-classes) first -- `_magic_registry_`
covers the common case without a hand-written subclass.) Declare them with
`BuildTimeField[T]` instead of `T | None = None`:

```python
from mlbricks import UNSET, BuildTimeField, Configurable


class Trainer(Configurable):
    class Config(Configurable.Config["Trainer"]):
        hidden_dim: int = 256
        dataloader: BuildTimeField[DataLoader] = UNSET  # explicit default required

    def __init__(self, cfg: "Trainer.Config") -> None:
        self.loader = cfg.dataloader  # plain DataLoader, no unwrapping
```

Calling `.build()` without it raises, unless it's supplied:

```python
cfg = Trainer.Config(hidden_dim=128)
cfg.build()  # ConfigValidationError -- dataloader required at build()
```

```python
trainer = cfg.build(dataloader=my_loader)  # resolved only for this call
```

Unlike `T | None` deferred fields, `BuildTimeField[T]` requires no manual
`@build_validator` -- `build()` raises automatically if it's still unresolved.
It's also invisible to `to_dict()`: the field is omitted from the serialized
dict entirely, not even as `null`. See
[Serializing configs with `to_dict()`](config-resolution.md#serializing-configs-with-to_dict)
for the serialization contract.

A `BuildTimeField[T]` can opt out of the "required at build" check by giving
it its own default, exactly like any other field:

```python
logger: BuildTimeField[Logger] = default_logger
```

Detection of "is this a `BuildTimeField`" is the class-declared default being
`UNSET` -- the same mechanism drives both the required-at-build check and the
`to_dict()` skip. Giving a field its own default opts it out of *both*: it's
no longer required at build, but it's also no longer omitted from
`to_dict()` -- it serializes like any ordinary field. Only give a
`BuildTimeField[T]` its own default when that value is itself serializable,
or when you never call `to_dict()` on that `Config`.

### Field constraints

Every `Config` is a Pydantic model, so type-level constraints are free.
Prefer them over manual validators. For build-time-only, non-serializable
values, see [Build-time-only fields](#build-time-only-fields) above instead
-- it isn't a constraint mechanism. Priority order:

1. **Declarative annotation** -- express bounds and length checks directly
   in the type:
   ```python
   from typing import Annotated

   from pydantic import Field, PositiveInt


   class MyModel(Configurable):
       class Config(Configurable.Config["MyModel"]):
           n_layers: PositiveInt = 3  # pydantic builtin
           dropout: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0  # [0, 1]
           val_frac: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.2  # (0, 1)
           name: Annotated[str, Field(min_length=1)]
   ```

2. **`@field_validator`** -- for single-field parse or coerce logic that
   cannot be expressed as a type:
   ```python
   @field_validator("betas")
   @classmethod
   def check_betas_length(cls, val: list[float]) -> list[float]:
       if len(val) != 2:
           raise ValueError("betas must have exactly 2 elements")
       return val
   ```

3. **`@model_validator(mode="after")`** -- for static cross-field invariants
   that must hold at both construction and build time:
   ```python
   @model_validator(mode="after")
   def check_hidden_divisible_by_heads(self) -> Self:
       if self.hidden_dim % self.n_att_heads != 0:
           raise ValueError("hidden_dim must be divisible by n_att_heads")
       return self
   ```

4. **`@build_validator`** -- only for checks that require a deferred field
   to be resolved (i.e. when a `T | None` field must be non-None at build
   time). See `Deferred fields` above.

Pydantic ships many constrained scalar types (`PositiveInt`, `NonNegativeFloat`,
etc.). For patterns that recur across your own codebase, define your own
aliases the same way (e.g. `UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]`)
and reuse them. Use `Annotated[T, Field(...)]` directly for anything else.

**Testing convention**: do not write tests for pydantic's own enforcement.
Only test: cross-field `@model_validator` invariants, deferred-field
`@build_validator` contracts, and `@field_validator` with non-trivial logic.
Apply the same exception to any custom constrained-type module you maintain
-- test boundary conditions there, since those definitions are your own code.

### Overriding fields at build time

Any field -- not just deferred ones -- can be overridden as a kwarg to
`.build()`. The override applies only to that one build call; the config object
is not mutated.

```python
cfg = MLP.Config(hidden_dim=256)

mlp_small = cfg.build(input_dim=64, hidden_dim=64)  # one-off override
mlp_large = cfg.build(input_dim=64)  # original hidden_dim=256
```

### Nested configs

Configs compose naturally: a parent config holds child configs as fields. Each
`__init__` assembles its own children by calling their `.build()`.

```python
from collections.abc import Iterable
from typing import Any, Self
from pydantic import Field
from mlbricks import Configurable, build_validator
# MLP defined as in the Deferred fields section above


class Optimizer(Configurable):
    class Config(Configurable.Config["Optimizer"]):
        lr: float = 1e-3

    def __init__(self, cfg: "Optimizer.Config", params: Iterable[Any]) -> None:
        self.lr = cfg.lr
        self.params = list(params)


class Trainer(Configurable):
    class Config(Configurable.Config["Trainer"]):
        mlp: MLP.Config = Field(default_factory=MLP.Config)
        optimizer: Optimizer.Config = Field(
            default_factory=lambda: Optimizer.Config(lr=1e-3)
        )
        input_dim: int | None = None

        @build_validator
        def check_input_dim(self) -> Self:
            if self.input_dim is None:
                raise ValueError("input_dim required at build time")
            return self

    def __init__(self, cfg: "Trainer.Config") -> None:
        self.mlp = cfg.mlp.build(input_dim=cfg.input_dim)
        self.optimizer = cfg.optimizer.build(params=self.mlp.parameters())
        self.cfg = cfg


cfg = Trainer.Config(
    mlp=MLP.Config(hidden_dim=128, output_dim=64),
    optimizer=Optimizer.Config(lr=1e-3),
)
trainer = cfg.build(input_dim=784)
```

`Trainer.Config` holds `MLP.Config` as a plain field -- not an assembled `MLP`.
`build()` does not know about nesting; it is `__init__` that calls
`cfg.mlp.build()` to assemble the child. Each level assembles its own children.

This is the primary composability pattern for any config tree -- models,
datasets, losses, optimizers, or anything else built from `Configurable`.

### Generic `Configurable` subclasses

Some classes are generic over their data type -- for example, a filter that
works on any `DataT`. Use PEP 695 syntax for the type parameter on the outer
class. The inner `Config` does **not** get a type parameter; inherit it plain.

```python
from mlbricks import Configurable


class ConfigurableFilter[T](Configurable):
    class Config(Configurable.Config):  # no subscript on the outer class
        pass

    def __init__(self, cfg: "ConfigurableFilter.Config") -> None:
        pass

    def __call__(self, data: T) -> bool:
        raise NotImplementedError
```

**Why no subscript on the inner `Config` base?**

`ConfigurableFilter[T]` is a PEP 695 generic class. Python evaluates base
class expressions immediately when the class body runs. Writing
`class Config(ConfigurableFilter.Config["ClassName"])` would try to subscript
`ConfigurableFilter` at class-definition time and raise `TypeError` -- even
though `Configurable.Config["OwnerClass"]` works fine on the root class (which
uses a `ClassVar` trick internally, not a type parameter).

The rule: subscript `Configurable.Config["Owner"]` only when inheriting
directly from the root `Configurable`. If the outer class itself carries a type
parameter, inherit plain.

### `from __future__ import annotations` is not compatible

Do not add `from __future__ import annotations` to files that define
`Configurable.Config` subclasses. That import makes all annotations lazy
strings, which breaks pydantic's type resolution for locally-scoped types
(types defined inside a function or test).

When pydantic encounters a lazy string annotation it cannot resolve at
class-definition time, it defers resolution by walking the Python call
stack. The `Config.__init__` wrapper adds a frame that throws off that
stack walk, causing `PydanticUserError` at construction time.

**Instead, use string literals for forward references:**

```python
# Bad -- from __future__ import annotations breaks Config subclasses
from __future__ import annotations


class MyModel(Configurable):
    class Config(Configurable.Config["MyModel"]):
        hidden_dim: int = 128

    def __init__(self, cfg: MyModel.Config) -> None:  # NameError at class time
        ...


# Good -- quote only the forward references that need it
class MyModel(Configurable):
    class Config(Configurable.Config["MyModel"]):
        hidden_dim: int = 128

    def __init__(self, cfg: "MyModel.Config") -> None:  # string literal, fine
        ...
```

This applies to production code and test files alike. If ruff reports
`F821 Undefined name` on an annotation inside a `Configurable` subclass,
the fix is to quote that annotation, not to add the future import.

### Subclass requirements (enforced at class-definition time)

Two mistakes are caught immediately when Python evaluates the class body,
not deferred until `build()` is called:

**Missing nested `Config`** -- every concrete (non-abstract) `Configurable`
subclass must define its own `Config` in its own `__dict__`. Inheriting a
parent's `Config` without redefining it means `Config.owner` points at the
parent, and `build()` would silently instantiate the wrong class.

```python
# Bad -- raises MissingConfigError at class-definition time
class MyModel(Configurable):
    def __init__(self, cfg: object) -> None:
        pass


# Good
class MyModel(Configurable):
    class Config(Configurable.Config["MyModel"]):
        hidden_dim: int = 128

    def __init__(self, cfg: "MyModel.Config") -> None: ...
```

Abstract classes (those with unimplemented `@abstractmethod`s) are exempt --
they cannot be instantiated directly and may leave `Config` to concrete
subclasses.

**`__init__` without `cfg`** -- if a subclass defines its own `__init__`,
it must accept `cfg` as a parameter. `build()` always passes the validated
config as `cfg=...`, so omitting the parameter causes a `TypeError` at
build time that is hard to diagnose.

```python
# Bad -- raises MissingCfgParameterError at class-definition time
class MyModel(Configurable):
    class Config(Configurable.Config["MyModel"]):
        hidden_dim: int = 128

    def __init__(self, hidden_dim: int) -> None:  # missing cfg
        ...


# Good
class MyModel(Configurable):
    class Config(Configurable.Config["MyModel"]):
        hidden_dim: int = 128

    def __init__(self, cfg: "MyModel.Config") -> None: ...
```

Classes that do not define their own `__init__` (inheriting one from a
parent) are not checked -- the inherited `__init__` is assumed correct.

### Validating config type at entrypoints

At script entrypoints, use `validate_config()` to guard against being passed
the wrong config type before calling `.build()`.

```python
def train(cfg: object) -> None:
    Trainer.validate_config(cfg)  # raises ConfigurableError if wrong type
    assert isinstance(cfg, Trainer.Config)  # narrows type for the checker
    trainer = cfg.build(input_dim=784)
    ...
```

`validate_config()` raises at runtime but does not narrow the type. Follow it
with `assert isinstance(...)` so the type checker knows `cfg` is a
`Trainer.Config` for the rest of the function.

---

## When to use `Configurable`

Use it for any class that:

- Has construction parameters you want to specify declaratively
- May be instantiated from a YAML file or composed into a larger config
- Benefits from separating "what are the parameters" from "build the object"

Common cases:

| Class type | Example parameters |
|---|---|
| `nn.Module` | layer dims, activation, dropout |
| `Dataset` | data path, split, featurizer settings |
| Loss | loss weights, margin, reduction mode |
| Optimizer | learning rate, weight decay, scheduler |

If a class is only ever constructed in one place with hardcoded arguments, a
plain `__init__` is fine. `Configurable` pays off when the config needs to
travel -- across files, into a YAML, or into a sweep.
