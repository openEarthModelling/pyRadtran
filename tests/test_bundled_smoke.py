"""End-to-end smoke test: a typical scene runs using ONLY bundled data."""

import pytest
import xarray as xr

from pyradtran import Runner, RunnerConfig, Scene
from pyradtran.data.resolver import DataResolver

pytestmark = pytest.mark.slow


def _has_real_assets():
    from pathlib import Path

    d = Path(__file__).resolve().parent.parent / "src" / "pyradtran" / "data" / "assets"
    return d.is_dir() and any(f.is_file() and f.name != ".gitkeep" for f in d.rglob("*"))


@pytest.fixture
def _no_external(monkeypatch):
    monkeypatch.delenv("LIBRADTRAN_DATA_FILES", raising=False)
    monkeypatch.delenv("LIBRADTRANDIR", raising=False)


def test_typical_scene_runs_bundled_only(_no_external, uvspec_exe):
    if not _has_real_assets():
        pytest.skip("bundled assets not yet committed (Phase B)")
    scene = (
        Scene()
        .set_atmosphere(profile="us")
        .set_source_solar(sza=30.0)
        .set_wavelength(300.0, 400.0)
        .set_solver(method="disort", streams=16)
        .set_output(quantities=["lambda", "edir"], format="ascii")
        .set_surface(albedo=0.2)
    )
    cfg = RunnerConfig(uvspec_exe=uvspec_exe, bundled_only=True)
    result = Runner.execute(scene, config=cfg)
    assert isinstance(result, xr.Dataset)
    assert "wavelength" in result.sizes
    # Confirm it actually used the bundled root.
    assert DataResolver(bundled_only=True).data_root.name == "assets"


def test_reptran_coarse_scene_runs_bundled_only(_no_external, uvspec_exe):
    """Exercises the (largest) bundled CKD data: reptran coarse."""
    if not _has_real_assets():
        pytest.skip("bundled assets not yet committed (Phase B)")
    scene = (
        Scene()
        .set_atmosphere(profile="us", mol_abs_param="reptran coarse")
        .set_source_solar(sza=30.0)
        .set_wavelength(300.0, 400.0)
        .set_solver(method="disort", streams=16)
        .set_output(quantities=["lambda", "edir"], format="ascii")
        .set_surface(albedo=0.2)
    )
    cfg = RunnerConfig(uvspec_exe=uvspec_exe, bundled_only=True)
    result = Runner.execute(scene, config=cfg)
    assert result.edir.min() >= 0
