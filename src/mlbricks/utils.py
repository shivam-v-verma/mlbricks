"""Path-slugging helpers shared by the storage layer."""

from pathlib import Path

__all__ = ["path_slug", "uri_slug"]


def path_slug(path: Path) -> str:
    """Convert a relative path to a flat slug by replacing separators with --.

    Args:
        path: Relative Path with no suffix. Absolute paths are rejected.

    Returns:
        Flat slug string, e.g. ``Path("data/train/shard-0000")``
        becomes ``"data--train--shard-0000"``.

    Raises:
        ValueError: If path is absolute.
    """
    if path.is_absolute():
        raise ValueError(f"path must be relative, got absolute path: {path}")
    return str(path).replace("/", "--")


def uri_slug(uri: str) -> str:
    """Convert a data URI to a flat slug for cache key construction.

    Strips the file extension and replaces path separators with --.
    Handles both relative paths and s3:// URIs.

    Args:
        uri: Relative path string or s3:// URI.

    Returns:
        Flat slug with no extension.

    Example:
        uri_slug("raw/adme.parquet")  # -> "raw--adme"
        uri_slug("s3://bucket/raw/adme.parquet")  # -> "s3--bucket--raw--adme"
    """
    if uri.startswith("s3://"):
        stripped = uri.removeprefix("s3://")
        return "s3--" + path_slug(Path(stripped).with_suffix(""))
    return path_slug(Path(uri).with_suffix(""))
