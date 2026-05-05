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


class TestRunnerGlobalConfig:
    """Tests for Runner global default configuration."""

    @pytest.fixture(autouse=True)
    def _reset_global_config(self):
        """Reset Runner._config to defaults after each test."""
        original = Runner._config
        Runner._config = RunnerConfig()
        yield
        Runner._config = original

    def test_configure_sets_global_defaults(self):
        cfg = Runner.configure(
            uvspec_exe="/opt/uvspec",
            data_path="/opt/data",
            max_workers=8,
            keep_temp=True,
            timeout=120,
        )
        assert Runner._config.uvspec_exe == "/opt/uvspec"
        assert Runner._config.data_path == "/opt/data"
        assert Runner._config.max_workers == 8
        assert Runner._config.keep_temp is True
        assert Runner._config.timeout == 120
        assert cfg is Runner._config

    def test_configure_accepts_config_object(self):
        cfg = RunnerConfig(uvspec_exe="/foo/uvspec", data_path="/foo/data")
        returned = Runner.configure(config=cfg)
        assert Runner._config is cfg
        assert returned is cfg

    def test_global_defaults_used_by_execute(self, has_uvspec, data_path):
        Runner.configure(uvspec_exe=has_uvspec, data_path=data_path)
        scene = (
            Scene()
            .set_atmosphere(profile="us")
            .set_source_solar(sza=30.0)
            .set_wavelength(300.0, 400.0)
            .set_solver(method="disort", streams=16)
            .set_output(quantities=["lambda", "edir"], format="ascii")
            .set_surface(albedo=0.2)
        )
        result = Runner.execute(scene)
        assert isinstance(result, xr.Dataset)
        assert result.attrs["uvspec_exe"] == has_uvspec

    def test_explicit_params_override_global_defaults(self, has_uvspec, data_path):
        Runner.configure(uvspec_exe="/nonexistent/uvspec", data_path="/nonexistent/data")
        scene = (
            Scene()
            .set_atmosphere(profile="us")
            .set_source_solar(sza=30.0)
            .set_wavelength(300.0, 400.0)
            .set_solver(method="disort", streams=16)
            .set_output(quantities=["lambda", "edir"], format="ascii")
            .set_surface(albedo=0.2)
        )
        result = Runner.execute(scene, uvspec_exe=has_uvspec, data_path=data_path)
        assert isinstance(result, xr.Dataset)
        assert result.attrs["uvspec_exe"] == has_uvspec

    def test_config_param_overrides_global_defaults(self, has_uvspec, data_path):
        Runner.configure(uvspec_exe="/nonexistent/uvspec", data_path="/nonexistent/data")
        scene = (
            Scene()
            .set_atmosphere(profile="us")
            .set_source_solar(sza=30.0)
            .set_wavelength(300.0, 400.0)
            .set_solver(method="disort", streams=16)
            .set_output(quantities=["lambda", "edir"], format="ascii")
            .set_surface(albedo=0.2)
        )
        local_cfg = RunnerConfig(uvspec_exe=has_uvspec, data_path=data_path)
        result = Runner.execute(scene, config=local_cfg)
        assert isinstance(result, xr.Dataset)
        assert result.attrs["uvspec_exe"] == has_uvspec


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
