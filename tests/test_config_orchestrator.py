"""T3: orchestrator unit tests (synthetic data — no uvspec needed).

End-to-end orchestration is exercised by the canonical YAML example (T6) and
the uvspec-gated regression suite.
"""

import numpy as np
import pytest
import xarray as xr

from pyradtran.config.orchestrator import (
    RunResult,
    _compute_drf,
    _heating_section,
    _make_plots,
    _scene_albedo,
)
from pyradtran.config.schema import SceneSection


def _scene_section(output=None, surface=None) -> SceneSection:
    return SceneSection(
        atmosphere={"profile": "us", "altitude": 0.0},
        source={"sza": 30.0},
        wavelength={"min_nm": 400.0, "max_nm": 700.0},
        surface=surface,
        output=output if output is not None else {},
    )


def _synthetic_rt(**extra) -> xr.Dataset:
    """2 wl × 2 zout flux dataset with hand-checkable values."""
    wl = [400.0, 550.0]
    zout = [0.0, 120.0]  # index 0 = surface, 1 = TOA
    edir = xr.DataArray([[100.0, 1000.0], [80.0, 800.0]], dims=("wavelength", "zout"))
    edn = xr.DataArray([[50.0, 0.0], [40.0, 0.0]], dims=("wavelength", "zout"))
    eup = xr.DataArray([[30.0, 200.0], [24.0, 160.0]], dims=("wavelength", "zout"))
    return xr.Dataset(
        {
            "edir": (("wavelength", "zout"), edir.data),
            "edn": (("wavelength", "zout"), edn.data),
            "eup": (("wavelength", "zout"), eup.data),
        },
        coords={"wavelength": wl, "zout": zout},
        **extra,
    )


class TestHeatingSection:
    def test_drops_quantities(self):
        out = {"quantities": ["lambda", "edir"], "format": "ascii", "heating_rate": "local"}
        section = _scene_section(output=out)
        heat = _heating_section(section)
        assert "quantities" not in heat.output
        assert heat.output["heating_rate"] == "local"
        assert heat.output["format"] == "ascii"

    def test_original_untouched(self):
        out = {"quantities": ["lambda"], "heating_rate": "local"}
        section = _scene_section(output=out)
        _heating_section(section)
        assert section.output["quantities"] == ["lambda"]


class TestSceneAlbedo:
    def test_from_surface(self):
        assert _scene_albedo(_scene_section(surface={"albedo": 0.2})) == 0.2

    def test_missing_raises(self):
        with pytest.raises(ValueError, match="albedo"):
            _scene_albedo(_scene_section())


class TestComputeDrf:
    def test_hand_computed(self):
        # With aerosol (wl=550): net_toa = 800-160 = 640, net_sfc = 80+40-24 = 96.
        # No aerosol: edir+20/edn+10/eup+6 everywhere, plus edir+100 and eup+50
        # at TOA → net_toa = 920-216 = 704, net_sfc = 100+50-30 = 120.
        rt_noaer = _synthetic_rt()
        for var, delta in (("edir", 20.0), ("edn", 10.0), ("eup", 6.0)):
            rt_noaer[var].data += delta
        rt_noaer["edir"].data[:, 1] += 100.0  # extra beam at TOA
        rt_noaer["eup"].data[:, 1] += 50.0
        drf = _compute_drf(_synthetic_rt(), rt_noaer)
        assert np.allclose(drf.toa, 640.0 - 704.0)  # -64 W/m² (both wl)
        assert np.allclose(drf.surface, 96.0 - 120.0)  # -24 W/m²
        assert np.allclose(drf.atmosphere, drf.toa - drf.surface)
        assert np.allclose(drf.wavelength_nm, [400.0, 550.0])


class TestMakePlots:
    def test_unknown_plot_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="unknown plot"):
            _make_plots(["not_a_plot"], RunResult(rt=_synthetic_rt()), tmp_path)

    def test_drf_plot_requires_drf(self, tmp_path):
        with pytest.raises(ValueError, match="analysis.drf"):
            _make_plots(["drf_spectral"], RunResult(rt=_synthetic_rt()), tmp_path)

    def test_heating_plot_requires_heating(self, tmp_path):
        with pytest.raises(ValueError, match="analysis.heating"):
            _make_plots(["rt_heating_rate"], RunResult(rt=_synthetic_rt()), tmp_path)

    def test_renders_rt_spectral(self, tmp_path):
        paths = _make_plots(["rt_spectral"], RunResult(rt=_synthetic_rt()), tmp_path)
        assert paths == [tmp_path / "rt_spectral.png"]
        assert (tmp_path / "rt_spectral.png").stat().st_size > 0
