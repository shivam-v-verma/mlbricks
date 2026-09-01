"""Transparent local/S3 file I/O for datasets and training scripts.

Public API: ``StorageDriver`` only. ``Storage``, ``LocalStorage``, and
``S3Storage`` are private implementation details.
"""

import fnmatch
import glob as glob_module
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

from mlbricks.utils import path_slug

__all__ = ["StorageDriver"]


def _validate_uri(uri: str) -> None:
    """Raise ValueError if uri is an absolute local path."""
    if not uri.startswith("s3://") and Path(uri).is_absolute():
        raise ValueError(
            f"absolute local paths are not allowed; use a path relative to "
            f"data_dir or an s3:// URI. Got: {uri!r}"
        )


class Storage(ABC):
    """Private ABC shared by LocalStorage, S3Storage, and StorageDriver."""

    @abstractmethod
    def local_path(self, uri: str) -> Path:
        """Return the local path uri resolves or stages to.

        Pure path computation -- no existence check, no I/O, no network
        call. The file may not exist yet.

        Args:
            uri: Relative path (resolved against data_dir) or s3:// URI.

        Returns:
            Local Path uri resolves to (local) or stages to (S3).
        """

    @abstractmethod
    def get(self, uri: str) -> Path | None:
        """Return a local Path for uri, or None on miss.

        For S3 URIs, checks local staging before downloading -- repeated
        calls for the same URI are free after the first download.

        Args:
            uri: Relative path (resolved against data_dir) or s3:// URI.

        Returns:
            Local Path if the resource exists, None otherwise.
        """

    @abstractmethod
    def put(self, src: Path, dst_uri: str) -> None:
        """Write src to dst_uri.

        Args:
            src: Source file (any valid Path; may be relative or absolute).
            dst_uri: Relative path or s3:// URI. Absolute local paths raise.
        """

    @abstractmethod
    def list(self, uri_glob: str) -> list[str]:
        """Return sorted URI strings matching uri_glob without downloading.

        Args:
            uri_glob: Glob pattern as a relative path or s3:// URI prefix.

        Returns:
            Sorted list of URI strings; empty list if no matches.
        """


class LocalStorage(Storage):
    """Local filesystem storage, rooted at data_dir."""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir

    def local_path(self, uri: str) -> Path:
        return self._data_dir / uri

    def get(self, uri: str) -> Path | None:
        path = self.local_path(uri)
        return path if path.exists() else None

    def put(self, src: Path, dst_uri: str) -> None:
        dst = self.local_path(dst_uri)
        if src.resolve() == dst.resolve():
            return

        dst.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(src, dst)

    def list(self, uri_glob: str) -> list[str]:
        pattern = str(self._data_dir / uri_glob)
        return sorted(
            str(Path(p).relative_to(self._data_dir)) for p in glob_module.glob(pattern)
        )


class S3Storage(Storage):
    """S3 storage with local staging under data_dir.

    boto3 client is created lazily on first use so instances remain
    picklable for DataLoader worker processes.
    """

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        # Lazy: created on first use to keep instances picklable
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = boto3.client("s3")

        return self._client

    @staticmethod
    def _staging_path(data_dir: Path, uri: str) -> Path:
        """Derive flat local staging path from an s3:// URI.

        Example:
            s3://my-bucket/raw/data.parquet
            -> data_dir/s3--my-bucket--raw--data.parquet
        """
        stripped = uri.removeprefix("s3://")
        slug = "s3--" + path_slug(Path(stripped))

        return data_dir / slug

    @staticmethod
    def _parse_uri(uri: str) -> tuple[str, str]:
        """Split s3://bucket/key into (bucket, key)."""
        without_scheme = uri.removeprefix("s3://")
        bucket, _, key = without_scheme.partition("/")

        return bucket, key

    def local_path(self, uri: str) -> Path:
        return self._staging_path(self._data_dir, uri)

    def get(self, uri: str) -> Path | None:
        staging = self.local_path(uri)

        # Return staging if already downloaded
        if staging.exists():
            return staging

        bucket, key = self._parse_uri(uri)
        client = self._get_client()

        try:
            staging.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(staging))
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("404", "NoSuchKey"):
                return None
            raise

        return staging

    def put(self, src: Path, dst_uri: str) -> None:
        bucket, key = self._parse_uri(dst_uri)
        self._get_client().upload_file(str(src), bucket, key)

    def list(self, uri_glob: str) -> list[str]:
        bucket, pattern = self._parse_uri(uri_glob)
        # Use the non-wildcard prefix to narrow the list call
        prefix = pattern.split("*")[0]

        client = self._get_client()
        paginator = client.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if fnmatch.fnmatch(obj["Key"], pattern):
                    keys.append(f"s3://{bucket}/{obj['Key']}")

        return sorted(keys)


class StorageDriver(Storage):
    """Dispatcher: routes get/put/list to LocalStorage or S3Storage.

    Constructed once at the script entrypoint and injected as a dependency.
    Not Configurable -- built directly from Settings:

        storage = StorageDriver(data_dir=settings.data_dir)

    URI convention:
    - ``s3://...``        -> S3Storage
    - relative path       -> LocalStorage (resolved against data_dir)
    - absolute local path -> raises ValueError
    """

    def __init__(self, data_dir: Path) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = data_dir

        self._local = LocalStorage(data_dir)
        self._s3 = S3Storage(data_dir)

    def local_path(self, uri: str) -> Path:
        _validate_uri(uri)
        return (
            self._s3.local_path(uri)
            if uri.startswith("s3://")
            else self._local.local_path(uri)
        )

    def get(self, uri: str) -> Path | None:
        _validate_uri(uri)
        return self._s3.get(uri) if uri.startswith("s3://") else self._local.get(uri)

    def put(self, src: Path, dst_uri: str) -> None:
        _validate_uri(dst_uri)
        if dst_uri.startswith("s3://"):
            self._s3.put(src, dst_uri)
        else:
            self._local.put(src, dst_uri)

    def list(self, uri_glob: str) -> list[str]:
        _validate_uri(uri_glob)
        return (
            self._s3.list(uri_glob)
            if uri_glob.startswith("s3://")
            else self._local.list(uri_glob)
        )
