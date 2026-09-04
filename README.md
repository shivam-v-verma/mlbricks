# mlbricks

Helper utilities for declarative ML configuration.

## Why

Declarative experiments (e.g. describing a run in YAML instead of
hardcoding it) keep research code clean: swap a model, a loss, an
optimizer, without touching the training script. Most ways of getting
there ask for a lot: a whole framework, a specific project layout, magic
behind the YAML. `mlbricks` is a few small, composable pieces that suggest
a way to specify experiments declaratively without too much magic.

## What's in the box

- **[`Configurable`](docs/configurable.md)** -- turn a class's constructor
  arguments into a typed config validated with pydantic, so anything you'd
  build in code is also expressible in YAML. Useful well beyond top-level
  models: optimizers, schedulers, losses, callbacks, data transforms, or
  anywhere the choice affects a run and you want it swept in HPO or logged
  for reproducibility, wrapping it as a `Configurable` gives you
  (de)serialization.
- **[`Registry`](docs/registry.md)** -- a name-to-class lookup, so YAML can
  say *which* `Configurable` (or plain class) to use, not just its kwargs.
- **[`resolve()` and `to_dict()`](docs/config-resolution.md)** -- because
  `Configurable` and `Registry` already give you declarative kwargs and
  name lookup, going from a plain dict to a built config tree (and back
  again) falls out almost for free. `resolve()` walks a YAML-shaped dict
  and constructs it; `to_dict()` is the inverse, for logging or
  checkpointing what actually ran.
- **[`StorageDriver` and `StorageCache`](docs/storage.md)** -- a small
  local/S3-agnostic file interface and a caching layer on top, so dataset
  code doesn't need to branch on where data lives.
- **[`ConfigurableTransform` and `TransformPipeline`](docs/featurization.md)**
  -- an ordered, named sequence of config-driven steps for building whatever
  a model consumes (features, filters), plus `TransformHooks` for splicing
  in training-only steps without changing the pipeline the model owns.

## A minimal taste

```yaml
# config.yaml
model:
  _registry_: models.mlp
  in_dim: 784
  hidden_dim: 128
  out_dim: 10
seed: 0
```

```python
import torch.nn as nn
from pydantic import PositiveInt
import yaml
from mlbricks import Configurable, Registry, resolve, to_dict

REGISTRY = Registry()
MODEL_REGISTRY = REGISTRY.subgroup("models")


@MODEL_REGISTRY.register("mlp")
class MLP(nn.Module, Configurable):
    class Config(Configurable.Config["MLP"]):
        in_dim: PositiveInt
        hidden_dim: PositiveInt
        out_dim: PositiveInt

    def __init__(self, cfg: "MLP.Config") -> None:
        super().__init__()
        self.hparams = to_dict(cfg, registry=REGISTRY)
        self.layers = nn.Sequential(
            nn.Linear(cfg.in_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.out_dim),
        )

    def forward(self, x):
        return self.layers(x)


raw = yaml.safe_load(open("config.yaml"))  # or OmegaConf.to_object(OmegaConf.load(...))
cfg = resolve(raw, registry=REGISTRY)
model = cfg["model"].build()
print(model.hparams)
# {'_registry_': 'models.mlp', 'in_dim': 784, 'hidden_dim': 128, 'out_dim': 10}
```

## Installation

```bash
uv add mlbricks-core
```

or

```bash
pip install mlbricks-core
```

## Scope

This is intentionally lightweight, meant to complement whatever training
framework you're already using rather than replace it. For things like
CLIs, experiment tracking, or distributed orchestration, we defer to those
frameworks' opinions -- `mlbricks` just helps with configuration
management.

## License

MIT. See [LICENSE](LICENSE).
