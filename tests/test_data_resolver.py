"""Tests for the data access layer (DataResolver + manifest)."""

from pyradtran.data.manifest import Asset, load_manifest


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
