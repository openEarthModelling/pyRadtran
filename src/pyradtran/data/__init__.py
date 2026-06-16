"""Bundled libRadtran data + access layer.

Provides :class:`DataResolver` for locating data files with a three-tier
priority (explicit path > environment variable > bundled subset) and an
optional ``bundled_only`` mode for reproducible runs.
"""

from __future__ import annotations

from pyradtran.data.manifest import Asset, ValidationIssue, load_manifest
from pyradtran.data.resolver import DataResolver

__all__ = [
    "Asset",
    "ValidationIssue",
    "DataResolver",
    "load_manifest",
    "list_bundled",
    "get_data_root",
    "resolve",
]


def get_data_root(*, bundled_only: bool = False):
    """Return the effective data_files_path (convenience wrapper)."""
    return DataResolver(bundled_only=bundled_only).data_root


def list_bundled(category: str | None = None):
    """List bundled assets (convenience wrapper)."""
    return DataResolver().list_bundled(category)


def resolve(category: str, name: str, *, bundled_only: bool = False):
    """Resolve a logical reference to an absolute path (convenience wrapper)."""
    return DataResolver(bundled_only=bundled_only).resolve(category, name)
