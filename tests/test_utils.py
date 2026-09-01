"""Tests for mlbricks.utils."""

from pathlib import Path

import pytest

from mlbricks import path_slug, uri_slug

# ---------------------------------------------------------------------------
# path_slug
# ---------------------------------------------------------------------------


def test_path_slug_single_component() -> None:
    assert path_slug(Path("file")) == "file"


def test_path_slug_replaces_separators() -> None:
    assert path_slug(Path("data/train/shard-0000")) == "data--train--shard-0000"


def test_path_slug_two_components() -> None:
    assert path_slug(Path("a/b")) == "a--b"


def test_path_slug_rejects_absolute_path() -> None:
    with pytest.raises(ValueError, match="absolute"):
        path_slug(Path("/data/train/shard"))


# ---------------------------------------------------------------------------
# uri_slug
# ---------------------------------------------------------------------------


def test_uri_slug_relative_path() -> None:
    assert uri_slug("raw/data.parquet") == "raw--data"


def test_uri_slug_s3_uri() -> None:
    assert uri_slug("s3://my-bucket/raw/data.parquet") == "s3--my-bucket--raw--data"


def test_uri_slug_strips_extension() -> None:
    assert uri_slug("train/shard-0.parquet") == "train--shard-0"
