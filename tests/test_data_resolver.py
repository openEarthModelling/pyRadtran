"""Tests for the data access layer (DataResolver + manifest)."""

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


def test_load_manifest_phase_a_is_empty():
    """Phase A ships an empty manifest (no data files committed yet)."""
    assets = load_manifest()
    assert assets == []


def test_load_manifest_returns_list_of_assets():
    # Manifest is empty in phase A; just assert return type contract.
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
    with pytest.raises(FileNotFoundError, match="kurudz_1.0nm.dat"):
        r.resolve("solar_flux", "kurudz_1.0nm.dat")  # not in phase-A manifest


def test_is_available_unknown_name_is_permissive(_clean_env):
    """Assets absent from the bundled manifest are assumed externally available."""
    r = DataResolver()
    assert r.is_available("solar_flux", "whatever_not_in_manifest") is True


def test_list_bundled_empty_in_phase_a(_clean_env):
    r = DataResolver()
    assert r.list_bundled() == []
