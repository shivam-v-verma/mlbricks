"""Typed persistent cache backed by StorageDriver."""

from collections.abc import Callable
from typing import Any

import torch

from mlbricks.storage import StorageDriver

__all__ = ["StorageCache"]


class StorageCache:
    """Typed persistent cache backed by StorageDriver.

    Wraps StorageDriver and cache_uri into a single injectable that handles
    cache hit/miss, torch serialisation, and tempfile lifecycle.

    Constructed once at the script entrypoint from Settings:

        storage = StorageDriver(data_dir=settings.data_dir)
        cache = StorageCache(storage=storage, cache_uri=settings.cache_uri)

    Not Configurable -- constructed directly from Settings like StorageDriver.

    Picklable for DataLoader worker processes (holds StorageDriver + str).
    """

    def __init__(self, storage: StorageDriver, cache_uri: str) -> None:
        self._storage = storage
        self._cache_uri = cache_uri

    def get_or_compute[T](
        self,
        key: str,
        fn: Callable[[], T],
        *,
        map_location: Any = None,
        mmap: bool = False,
    ) -> T:
        """Return cached value for key, or compute, cache, and return fn().

        On hit, deserialises and returns the cached value without calling fn.
        On miss, calls fn(), persists the result, and returns it -- reloaded
        from disk when mmap=True, so the return value is always file-backed
        (shareable/evictable across worker processes) rather than a plain
        in-memory object.

        Note: on a plain miss (mmap=False), map_location is ignored -- fn()'s
        result is returned as-is, so device/backing can differ from a hit.

        Args:
            key: Cache key; cache_uri is prepended internally.
            fn: Called only on a cache miss; result is serialised and stored.
            map_location: Passed to torch.load; see torch.load.
            mmap: Passed to torch.load; see torch.load. Also forces a
                file-backed reload on a miss.

        Returns:
            Deserialised cached value on hit, or fn() result (file-backed if
            mmap=True) on miss.

        Example:
            key = f"{uri_slug(parquet_path)}--{featurizer.get_hash()}.pt"
            data = cache.get_or_compute(key, lambda: featurize(parquet_path))
        """
        full_key = f"{self._cache_uri}/{key}"
        local_path = self._storage.get(full_key)

        # Cache hit: load straight from the storage-resolved path.
        if local_path is not None:
            return torch.load(  # type: ignore[return-value]
                local_path,
                weights_only=False,
                map_location=map_location,
                mmap=mmap,
            )

        # Cache miss: compute it
        result = fn()

        # Save to the path that would be used by storage
        local_path = self._storage.local_path(full_key)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, local_path)
        self._storage.put(local_path, full_key)

        if not mmap:
            # Return from memory, skip reload
            return result

        # Reload to preserve mmap=True behavior
        return torch.load(  # type: ignore[return-value]
            local_path,
            weights_only=False,
            map_location=map_location,
            mmap=mmap,
        )
