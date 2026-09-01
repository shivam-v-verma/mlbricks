# Storage Layer

`StorageDriver` and `StorageCache` solve two related but distinct problems. `StorageDriver` provides a unified interface over local filesystem and S3 operations, so datasets and training scripts never branch on where data lives. `StorageCache` builds on top of it to offer distributed caching — processed data written after one run can be retrieved by a future run or a different transient worker, avoiding redundant computation across the cluster. Neither belongs in a YAML config; they govern *where and how* execution runs rather than what it computes, so they're built from your own process-level settings and injected as dependencies.

---

## StorageDriver

Source: `src/mlbricks/storage.py`

Training scripts and datasets need to read raw files and write processed caches without caring whether that data lives on a local disk or in S3. `StorageDriver` provides a single `get` / `put` / `list` interface that dispatches to local or S3 storage based on URI prefix. The core abstraction is a `data_dir` — a local root directory that all relative paths resolve against and that S3 downloads are staged into. This means the same code works whether `data_dir` points to a fast scratch volume, a network mount, or a temp directory on a transient worker; only the process-level configuration changes.

### URI convention

All URI parameters follow one rule:

| URI form | Behaviour |
|---|---|
| `s3://bucket/key` | Routed to S3 |
| `relative/path` | Resolved against `data_dir` |
| `/absolute/path` | **Raises `ValueError`** |

`src: Path` passed to `put` is exempt — it is the caller's source file on disk and may be any valid path. All data paths in YAML configs must follow this convention.

### API

**`get(uri: str) -> Path | None`**

Returns a local `Path` for the resource, or `None` if it doesn't exist. For S3 URIs, the file is downloaded and staged under `data_dir` on first access; subsequent calls return the staged path without re-downloading. This means a transient worker that fetches the same file twice pays the download cost only once per process lifetime.

**`put(src: Path, dst_uri: str) -> None`**

Writes `src` to `dst_uri`. For local destinations, parent directories are created as needed. For S3, the file is uploaded directly.

**`list(uri_glob: str) -> list[str]`**

Returns sorted URI strings matching `uri_glob` without downloading anything. Return values follow the same URI convention as inputs and can be passed directly to `get` or `put`.

`StorageDriver` is picklable, so it can be held by dataset classes that are serialized into DataLoader worker processes.

---

## StorageCache

Source: `src/mlbricks/cache.py`

Preprocessing raw data can be expensive. Without caching, every training run and every transient worker within a run would repeat the same computation on the same inputs. `StorageCache` wraps a `StorageDriver` and a `cache_uri` to provide a single `get_or_compute` interface: on a hit the cached result is returned directly; on a miss the computation runs, the result is persisted, and future runs including runs on different machines can retrieve it. Because `cache_uri` follows the same URI convention as `StorageDriver`, pointing it at an S3 URI makes the cache durable across the cluster with no changes to calling code.

### API

**`get_or_compute(key: str, fn: Callable[[], T]) -> T`**

Returns the cached value for `key` if it exists, otherwise calls `fn()`, persists the result, and returns it. `fn` is a closure typically wrapping featurization on a predefined input and is only evaluated on a cache miss. The cache is keyed by `cache_uri/key`; callers own key construction and should not include `cache_uri` in the key. A good key captures the inputs that determine the output, typically the source data URI and a featurizer hash, so that a cache entry is reused exactly when the result would be identical.

`StorageCache` is picklable, so it can be held by dataset classes that are serialized into DataLoader worker processes.

---

## Usage

`StorageDriver` and `StorageCache` are constructed once at the script entrypoint from your own process-level settings and injected into any component that needs them. `mlbricks` does not ship a `Settings` class -- construct them from wherever your entrypoint config lives:

```python
from mlbricks import StorageCache, StorageDriver

from myproject.settings import get_settings

settings = get_settings()
storage = StorageDriver(data_dir=settings.data_dir)
cache = StorageCache(storage=storage, cache_uri=settings.cache_uri)

dataset = cfg["dataset"].build(storage=storage, cache=cache)
```

`cache_uri` follows the same URI convention as `StorageDriver`:

| `cache_uri` value | Resolves to |
|---|---|
| `"processed"` | `data_dir/processed/` |
| `"s3://my-project-cache"` | S3 bucket, shared across workers and persistent between runs |

### `uri_slug`

`uri_slug` (from `mlbricks`) is a companion utility for constructing cache keys. It converts a data URI to a flat slug by stripping the file extension and replacing path separators with `--`:

```python
from mlbricks import uri_slug

uri_slug("raw/adme.parquet")  # -> "raw--adme"
uri_slug(
    "s3://my-project-cache/raw/adme.parquet"
)  # -> "s3--my-project-cache--raw--adme"
```
