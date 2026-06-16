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
        assert cfg.max_workers == 4
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

    def test_bundled_only_field_exists(self):
        cfg = RunnerConfig(bundled_only=True)
        assert cfg.bundled_only is True


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

    def test_execute_many_uses_global_defaults(self, has_uvspec, data_path):
        Runner.configure(uvspec_exe=has_uvspec, data_path=data_path)
        scenes = [
            Scene()
            .set_atmosphere(profile="us")
            .set_source_solar(sza=30.0)
            .set_wavelength(300.0, 400.0)
            .set_solver(method="disort", streams=16)
            .set_output(quantities=["lambda", "edir"], format="ascii")
            .set_surface(albedo=0.2)
            for _ in range(2)
        ]
        results = Runner.execute_many(scenes, max_workers=2)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, xr.Dataset)
            assert r.attrs["uvspec_exe"] == has_uvspec

    def test_execute_many_config_param_overrides_global(self, has_uvspec, data_path):
        Runner.configure(uvspec_exe="/nonexistent/uvspec", data_path="/nonexistent/data")
        scenes = [
            Scene()
            .set_atmosphere(profile="us")
            .set_source_solar(sza=30.0)
            .set_wavelength(300.0, 400.0)
            .set_solver(method="disort", streams=16)
            .set_output(quantities=["lambda", "edir"], format="ascii")
            .set_surface(albedo=0.2)
            for _ in range(2)
        ]
        local_cfg = RunnerConfig(uvspec_exe=has_uvspec, data_path=data_path)
        results = Runner.execute_many(scenes, max_workers=2, config=local_cfg)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, xr.Dataset)
            assert r.attrs["uvspec_exe"] == has_uvspec


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


class TestRunnerBundledData:
    """Runner uses DataResolver for data-path resolution."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        monkeypatch.delenv("LIBRADTRAN_DATA_FILES", raising=False)
        monkeypatch.delenv("LIBRADTRANDIR", raising=False)

    def test_data_resolver_used_for_data_root(self, monkeypatch):
        """Runner delegates data-path resolution to DataResolver with the right args.

        Default config -> resolver gets data_root=None, bundled_only=False. The
        fake uvspec binary does not exist, so subprocess raises FileNotFoundError
        AFTER the resolver is constructed; we assert the captured kwargs.
        """
        from pyradtran.core import runner as runner_mod

        captured: dict = {}

        class FakeResolver:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            @property
            def data_root(self):
                return "/fake/bundled/root"

            def validate_scene(self, scene):
                return []

        monkeypatch.setattr(runner_mod, "DataResolver", FakeResolver)
        # _find_uvspec(None) falls back to shutil.which; stub so no real binary is needed.
        monkeypatch.setattr(runner_mod.shutil, "which", lambda name: "/fake/uvspec")

        scene = (
            Scene()
            .set_atmosphere(profile="us")
            .set_source_solar(sza=30.0)
            .set_wavelength(300.0, 400.0)
            .set_solver(method="disort", streams=16)
            .set_output(quantities=["lambda", "edir"], format="ascii")
        )

        # Fake uvspec binary cannot run -> subprocess raises FileNotFoundError,
        # but only AFTER DataResolver has been constructed.
        with pytest.raises(FileNotFoundError):
            runner_mod.Runner.execute(scene)

        assert captured.get("data_root") is None
        assert captured.get("bundled_only") is False

    def test_execute_forwards_explicit_data_path_to_resolver(self, monkeypatch):
        """An explicit data_path is forwarded to DataResolver as data_root."""
        from pyradtran.core import runner as runner_mod

        captured: dict = {}

        class FakeResolver:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            @property
            def data_root(self):
                return "/fake/root"

            def validate_scene(self, scene):
                return []

        monkeypatch.setattr(runner_mod, "DataResolver", FakeResolver)
        monkeypatch.setattr(runner_mod.shutil, "which", lambda name: "/fake/uvspec")

        scene = (
            Scene()
            .set_atmosphere(profile="us")
            .set_source_solar(sza=30.0)
            .set_wavelength(300.0, 400.0)
            .set_solver(method="disort", streams=16)
            .set_output(quantities=["lambda", "edir"], format="ascii")
        )

        with pytest.raises(FileNotFoundError):
            runner_mod.Runner.execute(scene, data_path="/explicit/data")

        assert captured.get("data_root") == "/explicit/data"

    def test_execute_warns_on_missing_data(self, monkeypatch, tmp_path, caplog):
        """strict=False (default): a missing bundled asset logs a warning; the run
        proceeds and fails later (fake uvspec), but the warning is captured."""
        import logging

        from pyradtran.core import runner as runner_mod
        from pyradtran.data import resolver as resolver_mod
        from pyradtran.data.manifest import Asset

        # Inject a manifest so the resolver "knows" US-standard but cannot find it
        # under the empty tmp_path data root.
        monkeypatch.setattr(
            resolver_mod,
            "load_manifest",
            lambda: [
                Asset(
                    category="atmosphere_profile",
                    name="US-standard",
                    uvspec_keyword="atmosphere_file",
                    paths=("atmmod/afglus.dat",),
                )
            ],
        )
        # _find_uvspec(None) falls back to shutil.which; stub so no real binary is needed.
        monkeypatch.setattr(runner_mod.shutil, "which", lambda name: "/fake/uvspec")

        scene = (
            Scene()
            .set_atmosphere(profile="us")
            .set_source_solar(sza=30.0)
            .set_wavelength(300.0, 400.0)
            .set_solver(method="disort", streams=16)
            .set_output(quantities=["lambda", "edir"], format="ascii")
        )

        # Fake uvspec binary cannot run -> FileNotFoundError, but only AFTER
        # validate_scene logs the missing-asset warning.
        with (
            caplog.at_level(logging.WARNING, logger="pyradtran"),
            pytest.raises(FileNotFoundError),
        ):
            runner_mod.Runner.execute(scene, data_path=str(tmp_path))

        assert any("US-standard" in rec.message for rec in caplog.records)

    def test_execute_raises_when_strict_and_missing_data(self, monkeypatch, tmp_path):
        """strict=True: missing bundled asset raises FileNotFoundError before uvspec runs."""
        from pyradtran.core import runner as runner_mod
        from pyradtran.data import resolver as resolver_mod
        from pyradtran.data.manifest import Asset

        monkeypatch.setattr(
            resolver_mod,
            "load_manifest",
            lambda: [
                Asset(
                    category="atmosphere_profile",
                    name="US-standard",
                    uvspec_keyword="atmosphere_file",
                    paths=("atmmod/afglus.dat",),
                )
            ],
        )
        monkeypatch.setattr(runner_mod.shutil, "which", lambda name: "/fake/uvspec")

        scene = (
            Scene()
            .set_atmosphere(profile="us")
            .set_source_solar(sza=30.0)
            .set_wavelength(300.0, 400.0)
            .set_solver(method="disort", streams=16)
            .set_output(quantities=["lambda", "edir"], format="ascii")
        )

        with pytest.raises(FileNotFoundError, match="strict mode"):
            runner_mod.Runner.execute(scene, data_path=str(tmp_path), strict=True)
