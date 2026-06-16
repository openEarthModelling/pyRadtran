"""Tests for the data access layer (DataResolver + manifest)."""

from pathlib import Path

import pytest

from pyradtran.data.manifest import Asset, load_manifest
from pyradtran.data.resolver import DataResolver


def test_asset_is_frozen_dataclass():
    a = Asset(
        category="solar_flux",
        name="kurudz_1.0nm.dat",
        uvspec_keyword="source",
        paths=("solar_flux/kurudz_1.0nm.dat",),
    )
    assert a.category == "solar_flux"
    assert a.paths == ("solar_flux/kurudz_1.0nm.dat",)


def test_load_manifest_is_populated():
    """Phase B: the manifest declares the curated bundled data subset."""
    assets = load_manifest()
    assert len(assets) > 0
    for a in assets:
        assert isinstance(a, Asset)
    names = {a.name for a in assets}
    assert "US-standard" in names
    assert "kurudz_1.0nm.dat" in names


def test_load_manifest_returns_list_of_assets():
    # Just assert the return type contract.
    assets = load_manifest()
    assert isinstance(assets, list)
    for a in assets:
        assert isinstance(a, Asset)


@pytest.fixture
def _clean_env(monkeypatch):
    """Ensure neither libRadtran env var leaks into resolution tests."""
    monkeypatch.delenv("LIBRADTRAN_DATA_FILES", raising=False)
    monkeypatch.delenv("LIBRADTRANDIR", raising=False)


def test_data_root_explicit_wins(_clean_env, tmp_path):
    root = tmp_path / "explicit"
    root.mkdir()
    r = DataResolver(data_root=root)
    assert r.data_root == root.resolve()


def test_data_root_env_var_when_no_explicit(_clean_env, tmp_path, monkeypatch):
    root = tmp_path / "envdata"
    root.mkdir()
    monkeypatch.setenv("LIBRADTRAN_DATA_FILES", str(root))
    r = DataResolver()
    assert r.data_root == root.resolve()


def test_data_root_libradtrandir_appends_data(_clean_env, tmp_path, monkeypatch):
    base = tmp_path / "librt"
    (base / "data").mkdir(parents=True)
    monkeypatch.setenv("LIBRADTRANDIR", str(base))
    r = DataResolver()
    assert r.data_root == (base / "data").resolve()


def test_data_root_falls_back_to_bundled(_clean_env):
    r = DataResolver()
    assert r.data_root.name == "assets"
    assert r.data_root.parent.name == "data"


def test_resolve_missing_raises(_clean_env):
    r = DataResolver()
    with pytest.raises(FileNotFoundError, match="definitely_not_bundled.dat"):
        r.resolve("solar_flux", "definitely_not_bundled.dat")


def test_is_available_unknown_name_is_permissive(_clean_env):
    """Assets absent from the bundled manifest are assumed externally available."""
    r = DataResolver()
    assert r.is_available("solar_flux", "whatever_not_in_manifest") is True


def test_list_bundled_returns_assets(_clean_env):
    r = DataResolver()
    all_assets = r.list_bundled()
    assert len(all_assets) > 0
    profiles = r.list_bundled("atmosphere_profile")
    assert all(a.category == "atmosphere_profile" for a in profiles)
    assert len(profiles) >= 6


def test_bundled_only_ignores_explicit_root(_clean_env, tmp_path):
    root = tmp_path / "explicit"
    root.mkdir()
    r = DataResolver(data_root=root, bundled_only=True)
    assert r.data_root == _bundled_root_path()
    assert r.data_root != root.resolve()


def test_bundled_only_ignores_env_var(_clean_env, tmp_path, monkeypatch):
    monkeypatch.setenv("LIBRADTRAN_DATA_FILES", str(tmp_path))
    r = DataResolver(bundled_only=True)
    assert r.data_root == _bundled_root_path()


def _bundled_root_path() -> Path:
    from pyradtran.data.resolver import _BUNDLED_ROOT

    return _BUNDLED_ROOT


def test_public_api_exports():
    import pyradtran

    assert hasattr(pyradtran, "DataResolver")
    from pyradtran.data import (
        Asset,
        DataResolver,
        ValidationIssue,
        get_data_root,
        list_bundled,
    )

    # Importing verifies these are exported; the asserts also use the names.
    assert Asset is not None
    assert DataResolver is not None
    assert ValidationIssue is not None
    assert callable(list_bundled)
    assert callable(get_data_root)
