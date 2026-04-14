"""Tests for Scene builder with immutable chain API."""

import pytest

from pyradtran.scene import Scene


class TestSceneBuilder:
    def test_empty_scene_raises_on_build(self):
        scene = Scene()
        with pytest.raises(ValueError):
            scene.build_input()

    def test_set_atmosphere_returns_new_scene(self):
        s1 = Scene()
        s2 = s1.set_atmosphere(profile="us")
        assert s1 is not s2
        assert s1.atmosphere is None
        assert s2.atmosphere is not None

    def test_chain_api(self):
        scene = (
            Scene()
            .set_atmosphere(profile="us", altitude=2.663)
            .set_source_solar(sza=30.0)
            .set_wavelength(250.0, 1200.0)
            .set_solver(method="disort", streams=16)
            .set_output(quantities=["lambda", "edir"])
        )
        assert scene.atmosphere.profile == "us"
        assert scene.atmosphere.altitude == 2.663
        assert scene.source.sza == 30.0
        assert scene.solver.method == "disort"
        assert scene.solver.streams == 16

    def test_immutability(self):
        s1 = Scene().set_atmosphere(profile="us")
        s2 = s1.set_source_solar(sza=30.0)
        assert s1.source is None
        assert s2.source is not None

    def test_clone(self):
        s1 = Scene().set_atmosphere(profile="us").set_source_solar(sza=30.0)
        s2 = s1.clone()
        assert s1 is not s2
        assert s1.atmosphere is not s2.atmosphere
        assert s2.atmosphere.profile == "us"

    def test_set_mol_modify(self):
        scene = (
            Scene()
            .set_atmosphere(profile="us")
            .set_mol_modify("H2O", 5.0, "MM")
            .set_mol_modify("O3", 300.0, "DU")
        )
        assert len(scene.atmosphere.mol_modify) == 2

    def test_add_raw_keyword(self):
        scene = (
            Scene()
            .set_atmosphere(profile="us")
            .set_source_solar(sza=30.0)
            .set_wavelength(250.0, 1200.0)
            .set_solver(method="disort", streams=16)
            .set_output(quiet=True)
            .add_raw_keyword("verbose", "")
        )
        assert ("verbose", "") in scene.raw_keywords

    def test_set_surface(self):
        scene = Scene().set_surface(albedo=0.2)
        assert scene.surface.albedo == 0.2

    def test_set_aerosol_default(self):
        scene = Scene().set_aerosol(default=True, angstrom_alpha=1.3, angstrom_beta=0.08)
        assert scene.aerosol is not None
        assert scene.aerosol.default is True

    def test_build_input_returns_string(self):
        scene = (
            Scene()
            .set_atmosphere(profile="us")
            .set_source_solar(sza=30.0)
            .set_wavelength(300.0, 400.0)
            .set_solver(method="disort", streams=8)
            .set_output(quiet=True, format="ascii")
            .set_surface(albedo=0.2)
        )
        text = scene.build_input()
        assert isinstance(text, str)
        assert "atmosphere_file" in text
        assert "source solar" in text

    def test_missing_source_raises(self):
        scene = Scene().set_atmosphere(profile="us").set_wavelength(300.0, 400.0).set_solver()
        with pytest.raises(ValueError, match="source"):
            scene.build_input()

    def test_missing_wavelength_raises(self):
        scene = Scene().set_atmosphere(profile="us").set_source_solar(sza=30.0).set_solver()
        with pytest.raises(ValueError, match="wavelength"):
            scene.build_input()

    def test_missing_solver_raises(self):
        scene = (
            Scene()
            .set_atmosphere(profile="us")
            .set_source_solar(sza=30.0)
            .set_wavelength(300.0, 400.0)
        )
        with pytest.raises(ValueError, match="solver"):
            scene.build_input()

    def test_repr(self):
        scene = Scene().set_atmosphere(profile="us").set_source_solar(sza=30.0)
        r = repr(scene)
        assert "Scene(" in r
        assert "atmosphere" in r
        assert "source" in r


# --- Phase 2 tests ---


def test_set_aerosol_modify():
    scene = Scene().set_atmosphere(profile="us").set_aerosol(default=True)
    scene2 = scene.set_aerosol_modify("ssa", "scale", 0.85)
    lines = scene2.aerosol.to_uvspec_lines()
    assert "aerosol_modify ssa scale 0.85" in lines


def test_set_aerosol_modify_multiple():
    scene = Scene().set_atmosphere(profile="us").set_aerosol(default=True)
    scene2 = scene.set_aerosol_modify("ssa", "scale", 0.85)
    scene3 = scene2.set_aerosol_modify("gg", "set", 0.7)
    lines = scene3.aerosol.to_uvspec_lines()
    assert "aerosol_modify ssa scale 0.85" in lines
    assert "aerosol_modify gg set 0.70" in lines


def test_set_cloud_water():
    scene = Scene().set_atmosphere(profile="us")
    scene2 = scene.set_cloud(wc_properties="hu")
    lines = scene2.cloud.to_uvspec_lines()
    assert "wc_properties hu" in lines


def test_set_cloud_ice():
    scene = Scene().set_atmosphere(profile="us")
    scene2 = scene.set_cloud(ic_properties="fu", ic_habit="rosette-6")
    lines = scene2.cloud.to_uvspec_lines()
    assert "ic_properties fu" in lines
    assert "ic_habit rosette-6" in lines


def test_set_surface_brdf():
    scene = Scene().set_atmosphere(profile="us")
    scene2 = scene.set_surface(brdf_hapke={"w": 0.4, "b0": 1.0, "h": 0.06})
    lines = scene2.surface.to_uvspec_lines()
    assert "brdf_hapke w 0.4" in lines


def test_immutable_set_aerosol_modify():
    scene = Scene().set_atmosphere(profile="us").set_aerosol(default=True)
    scene2 = scene.set_aerosol_modify("ssa", "scale", 0.85)
    assert len(scene.aerosol.modify) == 0
    assert len(scene2.aerosol.modify) == 1
