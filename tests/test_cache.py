"""Tests for StorageCache."""

import pickle
from pathlib import Path
from typing import Any

import pytest
import torch

from mlbricks import StorageCache, StorageDriver


@pytest.fixture()
def cache(tmp_path: Path) -> StorageCache:
    return StorageCache(storage=StorageDriver(data_dir=tmp_path), cache_uri="cache")


def test_miss_calls_fn_and_returns_result(cache: StorageCache) -> None:
    calls: list[int] = []

    def fn() -> list[int]:
        calls.append(1)
        return [42]

    result = cache.get_or_compute("key.pt", fn)

    assert result == [42]
    assert len(calls) == 1


def test_hit_returns_cached_result_without_calling_fn(
    cache: StorageCache,
) -> None:
    cache.get_or_compute("key.pt", lambda: [42])

    calls: list[int] = []

    def second_fn() -> list[int]:
        calls.append(1)
        return [99]

    result = cache.get_or_compute("key.pt", second_fn)

    assert result == [42]
    assert len(calls) == 0


def test_fn_called_exactly_once_across_two_calls(cache: StorageCache) -> None:
    calls: list[int] = []

    def fn() -> list[int]:
        calls.append(1)
        return [42]

    cache.get_or_compute("key.pt", fn)
    cache.get_or_compute("key.pt", fn)

    assert len(calls) == 1


def test_fn_exception_propagates(cache: StorageCache) -> None:
    def fn() -> list[int]:
        raise RuntimeError("compute failed")

    with pytest.raises(RuntimeError):
        cache.get_or_compute("key.pt", fn)


def test_key_prefixed_with_cache_uri(tmp_path: Path) -> None:
    cache = StorageCache(storage=StorageDriver(data_dir=tmp_path), cache_uri="myprefix")
    cache.get_or_compute("file.pt", lambda: [1, 2, 3])

    assert (tmp_path / "myprefix" / "file.pt").exists()


def test_is_picklable(cache: StorageCache) -> None:
    restored = pickle.loads(pickle.dumps(cache))
    assert isinstance(restored, StorageCache)


def test_cached_value_survives_torch_roundtrip(tmp_path: Path) -> None:
    """Value returned on hit is loaded via torch.load -- not returned from memory."""
    cache = StorageCache(storage=StorageDriver(data_dir=tmp_path), cache_uri="cache")
    original = [torch.tensor([1.0, 2.0])]
    cache.get_or_compute("key.pt", lambda: original)

    result = cache.get_or_compute("key.pt", lambda: [])

    assert torch.allclose(result[0], original[0])


def test_map_location_and_mmap_passthrough_on_hit(tmp_path: Path) -> None:
    """map_location/mmap kwargs are threaded through to torch.load on a hit."""
    cache = StorageCache(storage=StorageDriver(data_dir=tmp_path), cache_uri="cache")
    cache.get_or_compute("key.pt", lambda: [torch.tensor([1.0, 2.0])])

    result = cache.get_or_compute("key.pt", lambda: [], map_location="cpu", mmap=True)

    assert torch.allclose(result[0], torch.tensor([1.0, 2.0]))


def test_miss_with_mmap_returns_file_backed_reload_not_raw_result(
    tmp_path: Path,
) -> None:
    """On a miss with mmap=True, the result is reloaded from disk, not returned
    as-is -- mutating the original object after the call must not affect the
    returned value.
    """
    cache = StorageCache(storage=StorageDriver(data_dir=tmp_path), cache_uri="cache")
    original = [torch.tensor([1.0, 2.0])]

    result = cache.get_or_compute("key.pt", lambda: original, mmap=True)
    original[0].add_(1.0)

    assert torch.allclose(result[0], torch.tensor([1.0, 2.0]))


# ---------------------------------------------------------------------------
# S3-backed cache
# ---------------------------------------------------------------------------


def test_s3_miss_persists_to_bucket(tmp_path: Path, s3_bucket: Any) -> None:
    """A cache miss against an s3:// cache_uri uploads the result to S3."""
    cache = StorageCache(
        storage=StorageDriver(data_dir=tmp_path), cache_uri="s3://test-bucket/cache"
    )
    cache.get_or_compute("file.pt", lambda: [42])

    obj = s3_bucket.get_object(Bucket="test-bucket", Key="cache/file.pt")
    assert obj["Body"].read()


def test_s3_hit_from_fresh_worker_without_recompute(
    tmp_path: Path, s3_bucket: Any
) -> None:
    """A different worker (fresh data_dir, no local staging) reuses the cache
    entry from S3 instead of recomputing.
    """
    producer = StorageCache(
        storage=StorageDriver(data_dir=tmp_path / "producer"),
        cache_uri="s3://test-bucket/cache",
    )
    producer.get_or_compute("file.pt", lambda: [42])

    consumer = StorageCache(
        storage=StorageDriver(data_dir=tmp_path / "consumer"),
        cache_uri="s3://test-bucket/cache",
    )

    def fn_should_not_be_called() -> list[int]:
        raise AssertionError("fn must not be called on a cache hit")

    result = consumer.get_or_compute("file.pt", fn_should_not_be_called)

    assert result == [42]


def test_s3_miss_with_mmap_returns_correct_value(
    tmp_path: Path, s3_bucket: Any
) -> None:
    """The mmap-reload path on a miss also works with an s3:// cache_uri."""
    cache = StorageCache(
        storage=StorageDriver(data_dir=tmp_path), cache_uri="s3://test-bucket/cache"
    )
    original = [torch.tensor([1.0, 2.0])]

    result = cache.get_or_compute("key.pt", lambda: original, mmap=True)

    assert torch.allclose(result[0], torch.tensor([1.0, 2.0]))


def test_s3_miss_survives_bucket_deletion_via_local_staging(
    tmp_path: Path, s3_bucket: Any
) -> None:
    """After a miss, the local staging file left behind means a later access
    on the same worker does not need to re-fetch from S3.
    """
    cache = StorageCache(
        storage=StorageDriver(data_dir=tmp_path), cache_uri="s3://test-bucket/cache"
    )
    cache.get_or_compute("file.pt", lambda: [42])

    s3_bucket.delete_object(Bucket="test-bucket", Key="cache/file.pt")

    def fn_should_not_be_called() -> list[int]:
        raise AssertionError("fn must not be called on a cache hit")

    result = cache.get_or_compute("file.pt", fn_should_not_be_called)

    assert result == [42]
