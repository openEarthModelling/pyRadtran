"""Bundled libRadtran data + access layer.

Provides :class:`DataResolver` for locating data files with a three-tier
priority (explicit path > environment variable > bundled subset) and an
optional ``bundled_only`` mode for reproducible runs.
"""

from pyradtran.data.manifest import Asset, load_manifest

__all__ = ["Asset", "load_manifest"]
