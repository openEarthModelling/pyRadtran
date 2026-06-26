"""Consistency between MANIFEST.toml and the bundled assets/ directory."""

from pathlib import Path

import pytest

from pyradtran.data.manifest import check_consistency, load_manifest

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "src" / "pyradtran" / "data" / "assets"


def _has_real_files(d: Path) -> bool:
    return any(f.is_file() and f.name != ".gitkeep" for f in d.rglob("*"))


def test_manifest_and_assets_are_consistent():
    assets = load_manifest()
    if not _has_real_files(_ASSETS_DIR):
        pytest.skip("assets/ empty (Phase A) — consistency checked once data lands")
    messages = check_consistency(assets, _ASSETS_DIR)
    assert messages == [], "Manifest/assets mismatch:\n" + "\n".join(messages)
