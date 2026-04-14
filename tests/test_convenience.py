"""Tests for convenience functions."""

import xarray as xr

from pyradtran.convenience import _airmass_to_sza, run_solar_transmittance


class TestAirmassConversion:
    def test_airmass_1(self):
        sza = _airmass_to_sza(1.0)
        assert abs(sza - 0.0) < 0.01

    def test_airmass_2(self):
        sza = _airmass_to_sza(2.0)
        assert abs(sza - 60.0) < 0.5

    def test_airmass_very_large(self):
        sza = _airmass_to_sza(38.0)
        assert 88.0 < sza < 90.0


class TestRunSolarTransmittance:
    def test_returns_dataset(self, has_uvspec, data_path):
        result = run_solar_transmittance(
            airmass=1.5, pwv=5.0, ozone=300.0, data_path=data_path, uvspec_exe=has_uvspec,
        )
        assert isinstance(result, xr.Dataset)

    def test_with_altitude(self, has_uvspec, data_path):
        result = run_solar_transmittance(
            airmass=1.0, pwv=3.0, ozone=300.0, altitude="LSST",
            data_path=data_path, uvspec_exe=has_uvspec,
        )
        assert isinstance(result, xr.Dataset)


class TestRunThermalBrightness:
    def test_returns_dataset(self, has_uvspec, data_path):
        from pyradtran.convenience import run_thermal_brightness
        result = run_thermal_brightness(
            pwv=10.0, altitude=2.2, data_path=data_path, uvspec_exe=has_uvspec,
        )
        assert isinstance(result, xr.Dataset)


# --- Phase 2 tests ---


def test_run_solar_radiance_creates_scene():
    """Verify run_solar_radiance builds a valid Scene with radiance output."""
    from unittest.mock import patch, MagicMock
    from pyradtran.convenience import run_solar_radiance

    mock_dataset = MagicMock()
    with patch("pyradtran.convenience.Runner.execute", return_value=mock_dataset):
        result = run_solar_radiance(sza=60.0, airmass=2.0)
        assert result is mock_dataset

    with patch("pyradtran.convenience.Runner.execute", return_value=mock_dataset) as mock_exec:
        run_solar_radiance(sza=60.0, aerosol_tau=0.1)
        scene_arg = mock_exec.call_args[1]["scene"] if "scene" in mock_exec.call_args[1] else mock_exec.call_args[0][0]
        assert scene_arg.aerosol is not None


def test_run_with_aerosol_creates_scene():
    """Verify run_with_aerosol passes aerosol config correctly."""
    from unittest.mock import patch, MagicMock
    from pyradtran.convenience import run_with_aerosol

    mock_dataset = MagicMock()
    with patch("pyradtran.convenience.Runner.execute", return_value=mock_dataset) as mock_exec:
        run_with_aerosol(
            aerosol_file_type="explicit",
            aerosol_file_path="/data/profile.dat",
            sza=45.0,
        )
        scene_arg = mock_exec.call_args[1]["scene"] if "scene" in mock_exec.call_args[1] else mock_exec.call_args[0][0]
        assert scene_arg.aerosol.file == ("explicit", "/data/profile.dat")


def test_run_cloudy_scene_creates_scene():
    """Verify run_cloudy_scene passes cloud config correctly."""
    from unittest.mock import patch, MagicMock
    from pyradtran.convenience import run_cloudy_scene

    mock_dataset = MagicMock()
    with patch("pyradtran.convenience.Runner.execute", return_value=mock_dataset) as mock_exec:
        run_cloudy_scene(
            ic_properties="fu",
            ic_tau=5.0,
            sza=30.0,
        )
        scene_arg = mock_exec.call_args[1]["scene"] if "scene" in mock_exec.call_args[1] else mock_exec.call_args[0][0]
        assert scene_arg.cloud is not None
        assert scene_arg.cloud.ic_properties == "fu"


# --- Phase 3 tests ---


def test_run_lidar_creates_scene():
    """Verify run_lidar builds a valid Scene with sslidar config."""
    from unittest.mock import patch, MagicMock
    from pyradtran.convenience import run_lidar

    mock_dataset = MagicMock()
    with patch("pyradtran.convenience.Runner.execute", return_value=mock_dataset) as mock_exec:
        run_lidar(area=2.0, E0=0.2, n_ranges=50)
        scene_arg = mock_exec.call_args[1]["scene"] if "scene" in mock_exec.call_args[1] else mock_exec.call_args[0][0]
        assert scene_arg.sslidar is not None
        assert scene_arg.sslidar.area == 2.0
        assert scene_arg.sslidar.E0 == 0.2
        assert scene_arg.sslidar.n_ranges == 50
        assert scene_arg.solver.method == "sslidar"


def test_run_polarized_creates_scene():
    """Verify run_polarized builds a valid Scene with MC polarisation."""
    from unittest.mock import patch, MagicMock
    from pyradtran.convenience import run_polarized

    mock_dataset = MagicMock()
    with patch("pyradtran.convenience.Runner.execute", return_value=mock_dataset) as mock_exec:
        run_polarized(photons=50000, sza=45.0)
        scene_arg = mock_exec.call_args[1]["scene"] if "scene" in mock_exec.call_args[1] else mock_exec.call_args[0][0]
        assert scene_arg.mc is not None
        assert scene_arg.mc.photons == 50000
        assert scene_arg.mc.polarisation is True
        assert scene_arg.solver.method == "mystic"
        assert scene_arg.source.sza == 45.0
