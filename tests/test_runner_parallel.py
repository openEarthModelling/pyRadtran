"""Tests for Runner.execute_many parallel behavior: fail-fast + no pickling.

``execute_many`` runs scenes on a *thread* pool: uvspec executes as a
subprocess (the GIL is released while waiting on it), so threads parallelize
the actual work while scenes stay in-process and never need to be pickled.
Failures must surface immediately as ``RuntimeError`` naming the failing
scene index instead of being swallowed into the result list.
"""

import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from pyradtran import Scene
from pyradtran.core.runner import Runner
from pyradtran.models.aerosol_composite import (
    BulkSpecies,
    CompositeAerosol,
    MieSpecies,
    RefractiveIndex,
    SizeDistribution,
)
from pyradtran.models.blocks import MassProfile, PlacedBlock


def _simple_scene() -> Scene:
    return (
        Scene().set_atmosphere(profile="us").set_source_solar(sza=30.0).set_wavelength(400.0, 700.0)
    )


def _composite_scene() -> Scene:
    """Scene carrying a CompositeAerosol with one Mie block (the docs case)."""
    ri = RefractiveIndex(wavelength_um=[0.40, 0.55], n_real=[1.5, 1.5], k_imag=[0.0, 0.0])
    sp = MieSpecies(
        refractive_index=ri,
        size_distribution=SizeDistribution(
            kind="lognormal", params={"r_g_um": 0.1, "sigma_g": 2.0}
        ),
        particle_density_kg_m3=1800.0,
        name="t",
    )
    comp = CompositeAerosol(
        pieces=[PlacedBlock(block=sp, profile=MassProfile(kg_m3_per_layer=(1e-8, 1e-9)))],
        wavelength_grid_um=[0.4, 0.55],
        altitude_grid_km=[1.0, 0.0],
        n_legendre=32,
        output_dir=None,
    )
    return _simple_scene().set_aerosol(comp)


def _unpicklable_composite_scene() -> Scene:
    """Scene carrying a composite whose payload cannot pickle.

    Reproduces the multicomponent-example failure: an aerosol3D-backed
    ``BulkSpecies`` stores a local closure (``SizeDistribution.lognormal``'s
    ``_pdf``) on the bulk object, so pickling the scene raises.
    """

    def _pdf(r):  # local closure -> unpicklable, like aerosol3D SizeDistribution
        return r

    l_vals = np.arange(8)
    bulk = SimpleNamespace(
        wavelength_nm=np.linspace(400.0, 600.0, 3),
        C_ext=np.full(3, 10.0),
        C_sca=np.full(3, 9.0),
        SSA=np.full(3, 0.9),
        g=np.full(3, 0.5),
        beta=np.tile((2 * l_vals + 1) * 0.5**l_vals, (3, 1)),
        legendre_moments_beta=np.tile(0.5**l_vals, (3, 1)),
        size_distribution=SimpleNamespace(moment=lambda p: 1.0, _pdf=_pdf),
        effective_density_kg_m3=1800.0,
    )
    comp = CompositeAerosol(
        pieces=[
            PlacedBlock(
                block=BulkSpecies(bulk=bulk, name="b"),
                profile=MassProfile(kg_m3_per_layer=(1e-8, 1e-9)),
            )
        ],
        wavelength_grid_um=[0.4, 0.55],
        altitude_grid_km=[1.0, 0.0],
        n_legendre=32,
        output_dir=None,
    )
    return _simple_scene().set_aerosol(comp)


class TestExecuteManyFailFast:
    def test_first_scene_failure_raises_naming_index(self, monkeypatch):
        """A failing scene must raise RuntimeError naming its index, with the
        original exception chained — not be swallowed into the results."""

        def fake_execute(scene, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(Runner, "execute", staticmethod(fake_execute))

        with pytest.raises(RuntimeError, match="scene 0") as excinfo:
            Runner.execute_many([_simple_scene(), _simple_scene()], max_workers=2)
        assert isinstance(excinfo.value.__cause__, RuntimeError)
        assert "boom" in str(excinfo.value.__cause__)

    def test_later_scene_failure_raises_naming_index(self, monkeypatch):
        """Only the second scene fails -> the error names scene 1."""

        def fake_execute(scene, **kwargs):
            if getattr(scene, "_tag", None) == 0:
                return xr.Dataset({"x": ("zout", [0])})
            raise ValueError("late boom")

        monkeypatch.setattr(Runner, "execute", staticmethod(fake_execute))
        scenes = [_simple_scene(), _simple_scene()]
        for i, s in enumerate(scenes):
            s._tag = i

        with pytest.raises(RuntimeError, match="scene 1") as excinfo:
            Runner.execute_many(scenes, max_workers=2)
        assert isinstance(excinfo.value.__cause__, ValueError)

    def test_both_succeed_returns_datasets_in_order(self, monkeypatch):
        """Two succeeding scenes -> both Datasets returned in submission order."""

        def fake_execute(scene, **kwargs):
            return xr.Dataset({"x": ("zout", [scene._tag])})

        monkeypatch.setattr(Runner, "execute", staticmethod(fake_execute))
        scenes = [_simple_scene(), _simple_scene()]
        for i, s in enumerate(scenes):
            s._tag = i

        results = Runner.execute_many(scenes, max_workers=2)

        assert len(results) == 2
        assert all(isinstance(r, xr.Dataset) for r in results)
        assert results[0]["x"].item() == 0
        assert results[1]["x"].item() == 1


class TestExecuteManyNoPickling:
    def test_composite_scene_accepted(self, monkeypatch):
        """A Scene carrying a CompositeAerosol runs without pickling errors."""

        def fake_execute(scene, **kwargs):
            return xr.Dataset({"x": ("zout", [42])})

        monkeypatch.setattr(Runner, "execute", staticmethod(fake_execute))
        scene = _composite_scene()

        results = Runner.execute_many([scene, scene], max_workers=2)

        assert len(results) == 2
        for r in results:
            assert isinstance(r, xr.Dataset)
            assert r["x"].item() == 42

    def test_unpicklable_composite_scene_accepted(self, monkeypatch):
        """Regression (multicomponent example): a scene whose composite payload
        cannot pickle (aerosol3D bulk closures) must still run — execute_many
        must not put scenes on a process boundary."""

        def fake_execute(scene, **kwargs):
            return xr.Dataset({"x": ("zout", [7])})

        monkeypatch.setattr(Runner, "execute", staticmethod(fake_execute))
        scene = _unpicklable_composite_scene()

        # Guard: the fixture genuinely fails to pickle (the original bug).
        with pytest.raises(Exception):
            pickle.dumps(scene)

        results = Runner.execute_many([scene, scene], max_workers=2)

        assert len(results) == 2
        for r in results:
            assert isinstance(r, xr.Dataset)
            assert r["x"].item() == 7
