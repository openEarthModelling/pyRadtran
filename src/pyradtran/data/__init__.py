"""Bundled libRadtran data + access layer.

Provides :class:`DataResolver` for locating data files with a three-tier
priority (explicit path > environment variable > bundled subset) and an
optional ``bundled_only`` mode for reproducible runs.
"""

from pyradtran.data.manifest import Asset, ValidationIssue, load_manifest
from pyradtran.data.resolver import DataResolver

__all__ = ["Asset", "ValidationIssue", "DataResolver", "load_manifest"]
