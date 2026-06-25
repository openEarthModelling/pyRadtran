"""Tests for DataResolver.validate_scene (data-reference pre-checks)."""

import pytest

from pyradtran import Scene
from pyradtran.data.manifest import ValidationIssue
from pyradtran.data.resolver import DataResolver


@pytest.fixture
def _clean_env(monkeypatch):
    monkeypatch.delenv("LIBRADTRAN_DATA_FILES", raising=False)
    monkeypatch.delenv("LIBRADTRANDIR", raising=False)


def _scene(**kwargs):
    return (
        Scene()
        .set_atmosphere(profile="us")
        .set_source_solar(sza=30.0)
        .set_wavelength(300.0, 400.0)
        .set_solver(method="disort", streams=16)
        .set_output(quantities=["lambda", "edir"], format="ascii")
    )


def test_validation_issue_is_frozen():
    v = ValidationIssue(severity="warning", category="ckd", name="reptran coarse", message="x")
    assert v.severity == "warning"


def test_validate_scene_no_issues_when_all_permissive(_clean_env):
    """Phase A: empty manifest -> all refs permissive -> no issues."""
    r = DataResolver()
    issues = r.validate_scene(_scene())
    assert issues == []


def test_validate_scene_flags_missing_profile(_clean_env, tmp_path):
    """When a known manifest asset is missing on disk, an issue is raised."""
    from pyradtran.data.manifest import Asset

    r = DataResolver(data_root=tmp_path)
    r._manifest = [
        Asset(
            category="atmosphere_profile",
            name="US-standard",
            uvspec_keyword="atmosphere_file",
            paths=("atmmod/afglus.dat",),
        )
    ]
    issues = r.validate_scene(_scene())
    assert any(i.category == "atmosphere_profile" and i.name == "US-standard" for i in issues)


def test_validate_scene_skips_custom_profile_path(_clean_env, tmp_path):
    """A profile that looks like a file path is user-provided; not validated."""
    r = DataResolver(data_root=tmp_path)
    r._manifest = []
    scene = (
        Scene()
        .set_atmosphere(profile="/abs/path/custom.dat")
        .set_source_solar(sza=30.0)
        .set_wavelength(300.0, 400.0)
        .set_solver(method="disort", streams=16)
        .set_output(quantities=["lambda", "edir"], format="ascii")
    )
    assert r.validate_scene(scene) == []


def test_validate_scene_checks_solar_flux(_clean_env, tmp_path):
    from pyradtran.data.manifest import Asset

    r = DataResolver(data_root=tmp_path)
    r._manifest = [
        Asset(
            category="solar_flux",
            name="kurudz_1.0nm.dat",
            uvspec_keyword="source",
            paths=("solar_flux/kurudz_1.0nm.dat",),
        )
    ]
    scene = (
        Scene()
        .set_atmosphere(profile="us")
        .set_source_solar(sza=30.0, solar_flux_file="kurudz_1.0nm.dat")
        .set_wavelength(300.0, 400.0)
        .set_solver(method="disort", streams=16)
        .set_output(quantities=["lambda", "edir"], format="ascii")
    )
    issues = r.validate_scene(scene)
    assert any(i.category == "solar_flux" for i in issues)


def test_validate_scene_checks_mol_abs_param(_clean_env, tmp_path):
    from pyradtran.data.manifest import Asset

    r = DataResolver(data_root=tmp_path)
    r._manifest = [
        Asset(
            category="ckd",
            name="reptran coarse",
            uvspec_keyword="mol_abs_param",
            paths=("correlated_k/reptran/x.cdf",),
        )
    ]
    scene = (
        Scene()
        .set_atmosphere(profile="us", mol_abs_param="reptran coarse")
        .set_source_solar(sza=30.0)
        .set_wavelength(300.0, 400.0)
        .set_solver(method="disort", streams=16)
        .set_output(quantities=["lambda", "edir"], format="ascii")
    )
    issues = r.validate_scene(scene)
    assert any(i.category == "ckd" and i.name == "reptran coarse" for i in issues)
