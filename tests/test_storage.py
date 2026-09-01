"""Tests for Storage ABC, LocalStorage, S3Storage, StorageDriver."""

import pickle
from pathlib import Path
from typing import Any

import pytest

from mlbricks import StorageDriver
from mlbricks.storage import LocalStorage, S3Storage

# ---------------------------------------------------------------------------
# URI validation (via StorageDriver)
# ---------------------------------------------------------------------------


def test_absolute_uri_raises_on_get(tmp_path: Path) -> None:
    driver = StorageDriver(data_dir=tmp_path)
    with pytest.raises(ValueError):
        driver.get("/absolute/path/file.pt")


def test_absolute_uri_raises_on_put(tmp_path: Path) -> None:
    src = tmp_path / "src.pt"
    src.write_bytes(b"data")
    driver = StorageDriver(data_dir=tmp_path)
    with pytest.raises(ValueError):
        driver.put(src, "/absolute/dst.pt")


def test_absolute_uri_raises_on_list(tmp_path: Path) -> None:
    driver = StorageDriver(data_dir=tmp_path)
    with pytest.raises(ValueError):
        driver.list("/absolute/**/*.pt")


# ---------------------------------------------------------------------------
# LocalStorage
# ---------------------------------------------------------------------------


def test_local_get_returns_path_on_hit(tmp_path: Path) -> None:
    (tmp_path / "file.pt").write_bytes(b"data")
    storage = LocalStorage(data_dir=tmp_path)
    result = storage.get("file.pt")
    assert result == tmp_path / "file.pt"


def test_local_get_returns_none_on_miss(tmp_path: Path) -> None:
    storage = LocalStorage(data_dir=tmp_path)
    assert storage.get("missing.pt") is None


def test_local_put_copies_file(tmp_path: Path) -> None:
    src = tmp_path / "src.pt"
    src.write_bytes(b"payload")
    storage = LocalStorage(data_dir=tmp_path)
    storage.put(src, "dst.pt")
    assert (tmp_path / "dst.pt").read_bytes() == b"payload"


def test_local_path_no_io(tmp_path: Path) -> None:
    """local_path is a pure path computation -- no existence check."""
    storage = LocalStorage(data_dir=tmp_path)
    assert storage.local_path("missing.pt") == tmp_path / "missing.pt"


def test_local_put_noop_when_src_is_dst(tmp_path: Path) -> None:
    """put is a no-op when src already is the destination file."""
    storage = LocalStorage(data_dir=tmp_path)
    dst = storage.local_path("file.pt")
    dst.write_bytes(b"payload")
    storage.put(dst, "file.pt")
    assert dst.read_bytes() == b"payload"


def test_local_put_creates_parent_dirs(tmp_path: Path) -> None:
    src = tmp_path / "src.pt"
    src.write_bytes(b"payload")
    storage = LocalStorage(data_dir=tmp_path)
    storage.put(src, "nested/deep/dst.pt")
    assert (tmp_path / "nested" / "deep" / "dst.pt").exists()


def test_local_list_returns_sorted_relative_paths(tmp_path: Path) -> None:
    (tmp_path / "a.pt").write_bytes(b"")
    (tmp_path / "b.pt").write_bytes(b"")
    storage = LocalStorage(data_dir=tmp_path)
    result = storage.list("*.pt")
    assert result == ["a.pt", "b.pt"]


def test_local_list_returns_empty_on_no_match(tmp_path: Path) -> None:
    storage = LocalStorage(data_dir=tmp_path)
    assert storage.list("*.pt") == []


# ---------------------------------------------------------------------------
# S3Storage
# ---------------------------------------------------------------------------


def test_s3_get_downloads_file(tmp_path: Path, s3_bucket: Any) -> None:
    s3_bucket.put_object(Bucket="test-bucket", Key="data/file.pt", Body=b"content")
    storage = S3Storage(data_dir=tmp_path)
    result = storage.get("s3://test-bucket/data/file.pt")
    assert result is not None
    assert result.read_bytes() == b"content"


def test_s3_get_returns_none_on_miss(tmp_path: Path, s3_bucket: Any) -> None:
    storage = S3Storage(data_dir=tmp_path)
    assert storage.get("s3://test-bucket/missing.pt") is None


def test_s3_get_uses_staging_on_second_call(tmp_path: Path, s3_bucket: Any) -> None:
    """Second get returns staging path without re-downloading."""
    s3_bucket.put_object(Bucket="test-bucket", Key="file.pt", Body=b"content")
    storage = S3Storage(data_dir=tmp_path)

    first = storage.get("s3://test-bucket/file.pt")
    assert first is not None

    # Delete from mock S3 -- second call must use staging, not re-download
    s3_bucket.delete_object(Bucket="test-bucket", Key="file.pt")
    second = storage.get("s3://test-bucket/file.pt")
    assert second == first


def test_s3_staging_path_derivation(tmp_path: Path, s3_bucket: Any) -> None:
    """Staging filename is s3--bucket--prefix--file.pt."""
    s3_bucket.put_object(Bucket="test-bucket", Key="prefix/file.pt", Body=b"content")
    storage = S3Storage(data_dir=tmp_path)
    result = storage.get("s3://test-bucket/prefix/file.pt")
    assert result == tmp_path / "s3--test-bucket--prefix--file.pt"


def test_s3_local_path_matches_staging_derivation(tmp_path: Path) -> None:
    """local_path matches get's staging path, without any network call."""
    storage = S3Storage(data_dir=tmp_path)
    result = storage.local_path("s3://test-bucket/prefix/file.pt")
    assert result == tmp_path / "s3--test-bucket--prefix--file.pt"


def test_s3_put_uploads_file(tmp_path: Path, s3_bucket: Any) -> None:
    src = tmp_path / "local.pt"
    src.write_bytes(b"payload")
    storage = S3Storage(data_dir=tmp_path)
    storage.put(src, "s3://test-bucket/uploaded.pt")
    obj = s3_bucket.get_object(Bucket="test-bucket", Key="uploaded.pt")
    assert obj["Body"].read() == b"payload"


def test_s3_list_returns_matching_uris(tmp_path: Path, s3_bucket: Any) -> None:
    for key in ["train/a.parquet", "train/b.parquet", "val/c.parquet"]:
        s3_bucket.put_object(Bucket="test-bucket", Key=key, Body=b"")
    storage = S3Storage(data_dir=tmp_path)
    result = storage.list("s3://test-bucket/train/*.parquet")
    assert result == [
        "s3://test-bucket/train/a.parquet",
        "s3://test-bucket/train/b.parquet",
    ]


def test_s3_list_returns_empty_on_no_match(tmp_path: Path, s3_bucket: Any) -> None:
    storage = S3Storage(data_dir=tmp_path)
    assert storage.list("s3://test-bucket/missing/*.parquet") == []


def test_s3_storage_is_picklable(tmp_path: Path) -> None:
    """S3Storage must be picklable for DataLoader worker processes."""
    storage = S3Storage(data_dir=tmp_path)
    restored = pickle.loads(pickle.dumps(storage))
    assert isinstance(restored, S3Storage)


# ---------------------------------------------------------------------------
# StorageDriver dispatch
# ---------------------------------------------------------------------------


def test_driver_routes_relative_to_local(tmp_path: Path) -> None:
    (tmp_path / "file.pt").write_bytes(b"data")
    driver = StorageDriver(data_dir=tmp_path)
    assert driver.get("file.pt") == tmp_path / "file.pt"


def test_driver_local_path_routes_relative_to_local(tmp_path: Path) -> None:
    driver = StorageDriver(data_dir=tmp_path)
    assert driver.local_path("file.pt") == tmp_path / "file.pt"


def test_driver_local_path_routes_s3_uri_to_s3(tmp_path: Path) -> None:
    driver = StorageDriver(data_dir=tmp_path)
    result = driver.local_path("s3://test-bucket/file.pt")
    assert result == tmp_path / "s3--test-bucket--file.pt"


def test_driver_local_path_raises_on_absolute(tmp_path: Path) -> None:
    driver = StorageDriver(data_dir=tmp_path)
    with pytest.raises(ValueError):
        driver.local_path("/absolute/path/file.pt")


def test_driver_routes_s3_uri_to_s3(tmp_path: Path, s3_bucket: Any) -> None:
    s3_bucket.put_object(Bucket="test-bucket", Key="file.pt", Body=b"data")
    driver = StorageDriver(data_dir=tmp_path)
    assert driver.get("s3://test-bucket/file.pt") is not None


def test_driver_is_picklable(tmp_path: Path) -> None:
    driver = StorageDriver(data_dir=tmp_path)
    restored = pickle.loads(pickle.dumps(driver))
    assert isinstance(restored, StorageDriver)
