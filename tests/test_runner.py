"""Tests for Runner (subprocess execution)."""

import pytest
import xarray as xr

from pyradtran import Scene
from pyradtran.core.runner import Runner, RunnerConfig


class TestRunnerConfig:
    def test_default_config(self):
        cfg = RunnerConfig()
        assert cfg.uvspec_exe is None
        assert cfg.data_path is None
        assert cfg.max_workers == 1
        assert cfg.keep_temp is False

    def test_custom_config(self):
        cfg = RunnerConfig(
            uvspec_exe="/usr/local/bin/uvspec",
            data_path="/data/libradtran/data",
            max_workers=4,
            keep_temp=True,
        )
        assert cfg.uvspec_exe == "/usr/local/bin/uvspec"
        assert cfg.max_workers == 4


class TestRunnerExecute:
    def test_single_execution_returns_dataset(self, has_uvspec, data_path):
        scene = (
            Scene()
            .set_atmosphere(profile="us")
            .set_source_solar(sza=30.0)
            .set_wavelength(300.0, 400.0)
            .set_solver(method="disort", streams=16)
            .set_output(quantities=["lambda", "edir"], format="ascii")
            .set_surface(albedo=0.2)
        )
        result = Runner.execute(scene, data_path=data_path, uvspec_exe=has_uvspec)
        assert isinstance(result, xr.Dataset)
        assert "wavelength" in result.dims

    def test_solar_transmittance_values(self, has_uvspec, data_path):
        scene = (
            Scene()
            .set_atmosphere(profile="us")
            .set_source_solar(sza=30.0)
            .set_wavelength(300.0, 400.0)
            .set_solver(method="disort", streams=16)
            .set_output(quantity="transmittance", format="ascii")
            .set_surface(albedo=0.2)
        )
        result = Runner.execute(scene, data_path=data_path, uvspec_exe=has_uvspec)
        assert result.edir.min() >= 0

    def test_invalid_scene_raises(self, has_uvspec, data_path):
        scene = Scene()
        with pytest.raises(Exception):
            Runner.execute(scene, data_path=data_path, uvspec_exe=has_uvspec)
