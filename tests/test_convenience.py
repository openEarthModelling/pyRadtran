"""Tests for convenience functions."""

import pytest
import xarray as xr

from pyradtran.convenience import _airmass_to_sza, run_solar_transmittance


def _need_opac_data():
    """Skip if OPAC ingredient data is not resolvable on disk."""
    from pyradtran.optics import opac

    try:
        root = opac._opac_root(None)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"OPAC data unavailable: {e}")
    if not (root / "size_distr.cfg").is_file():
        pytest.skip("OPAC data not bundled")


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
            airmass=1.5,
            pwv=5.0,
            ozone=300.0,
            data_path=data_path,
            uvspec_exe=has_uvspec,
        )
        assert isinstance(result, xr.Dataset)

    def test_with_altitude(self, has_uvspec, data_path):
        result = run_solar_transmittance(
            airmass=1.0,
            pwv=3.0,
            ozone=300.0,
            altitude="LSST",
            data_path=data_path,
            uvspec_exe=has_uvspec,
        )
        assert isinstance(result, xr.Dataset)


class TestRunThermalBrightness:
    def test_returns_dataset(self, has_uvspec, data_path):
        from pyradtran.convenience import run_thermal_brightness

        result = run_thermal_brightness(
            pwv=10.0,
            altitude=2.2,
            data_path=data_path,
            uvspec_exe=has_uvspec,
        )
        assert isinstance(result, xr.Dataset)


# --- Phase 2 tests ---


def _get_scene_arg(mock_exec):
    """Extract the Scene argument from a mocked Runner.execute call."""
    ca = mock_exec.call_args
    return ca[1]["scene"] if "scene" in ca[1] else ca[0][0]


def test_run_solar_radiance_creates_scene():
    """Verify run_solar_radiance builds a valid Scene with radiance output."""
    from unittest.mock import MagicMock, patch

    from pyradtran.convenience import run_solar_radiance

    mock_dataset = MagicMock()
    with patch("pyradtran.convenience.Runner.execute", return_value=mock_dataset):
        result = run_solar_radiance(sza=60.0, airmass=2.0)
        assert result is mock_dataset


def test_run_cloudy_scene_creates_scene():
    """Verify run_cloudy_scene passes cloud config correctly."""
    from unittest.mock import MagicMock, patch

    from pyradtran.convenience import run_cloudy_scene

    mock_dataset = MagicMock()
    with patch("pyradtran.convenience.Runner.execute", return_value=mock_dataset) as mock_exec:
        run_cloudy_scene(
            ic_properties="fu",
            ic_tau=5.0,
            sza=30.0,
        )
        scene_arg = _get_scene_arg(mock_exec)
        assert scene_arg.cloud is not None
        assert scene_arg.cloud.ic_properties == "fu"


# --- Phase 3 tests ---


def test_run_lidar_creates_scene():
    """Verify run_lidar builds a valid Scene with sslidar config."""
    from unittest.mock import MagicMock, patch

    from pyradtran.convenience import run_lidar

    mock_dataset = MagicMock()
    with patch("pyradtran.convenience.Runner.execute", return_value=mock_dataset) as mock_exec:
        run_lidar(area=2.0, E0=0.2, n_ranges=50)
        scene_arg = _get_scene_arg(mock_exec)
        assert scene_arg.sslidar is not None
        assert scene_arg.sslidar.area == 2.0
        assert scene_arg.sslidar.E0 == 0.2
        assert scene_arg.sslidar.n_ranges == 50
        assert scene_arg.solver.method == "sslidar"


def test_run_polarized_creates_scene():
    """Verify run_polarized builds a valid Scene with MC polarisation."""
    _need_opac_data()
    from unittest.mock import MagicMock, patch

    from pyradtran.convenience import run_polarized
    from pyradtran.models.aerosol_composite import CompositeAerosol

    mock_dataset = MagicMock()
    with patch("pyradtran.convenience.Runner.execute", return_value=mock_dataset) as mock_exec:
        run_polarized(photons=50000, sza=45.0, wl_min=545.0, wl_max=555.0)
        scene_arg = _get_scene_arg(mock_exec)
        assert scene_arg.mc is not None
        assert scene_arg.mc.photons == 50000
        assert scene_arg.mc.polarisation is True
        assert scene_arg.mc.backward is True
        assert scene_arg.aerosol is not None
        assert isinstance(scene_arg.aerosol, CompositeAerosol)
        assert scene_arg.solver.method == "mystic"
        assert scene_arg.source.sza == 45.0


class TestRun3D:
    def test_returns_dataset(self, uvspec_exe, data_path):
        from pyradtran.convenience import run_3d

        result = run_3d(
            data_path=data_path,
            uvspec_exe=uvspec_exe,
            photons=1000,
        )
        assert isinstance(result, xr.Dataset)


def _need_mps(data_path):
    """Skip unless libRadtran's MPS satellite-geometry netCDF is available.

    MPS ships with libRadtran (not bundled here). Gate the satellite test so
    CI without the full libRadtran data tree skips cleanly instead of failing.
    """
    import os

    for root in (data_path, os.environ.get("PYRADTRAN_DATA_PATH")):
        if root and os.path.isfile(os.path.join(root, "MPS")):
            return
    pytest.skip("libRadtran MPS satellite-geometry file not available")


class TestRunSatellite:
    def test_returns_dataset(self, uvspec_exe, data_path):
        _need_mps(data_path)
        from pyradtran.convenience import run_satellite

        result = run_satellite(
            geometry="MPS",
            pixel=(10, 20),
            sza=60.0,
            data_path=data_path,
            uvspec_exe=uvspec_exe,
        )
        assert isinstance(result, xr.Dataset)


def test_run_with_opac_preset_creates_scene():
    """Verify run_with_opac_preset builds a CompositeAerosol via the OPAC factory."""
    _need_opac_data()
    from unittest.mock import MagicMock, patch

    from pyradtran.convenience import run_with_opac_preset
    from pyradtran.models.aerosol_composite import CompositeAerosol

    mock_dataset = MagicMock()
    with patch("pyradtran.convenience.Runner.execute", return_value=mock_dataset) as mock_exec:
        run_with_opac_preset(preset="maritime_clean", sza=45.0, wl_min=545.0, wl_max=555.0)
        scene_arg = _get_scene_arg(mock_exec)
        assert scene_arg.aerosol is not None
        assert isinstance(scene_arg.aerosol, CompositeAerosol)
        assert len(scene_arg.aerosol.pieces) >= 1


def test_run_with_opac_custom_creates_scene():
    """Verify run_with_opac_custom builds a CompositeAerosol from a profile file."""
    _need_opac_data()
    from unittest.mock import MagicMock, patch

    from pyradtran.convenience import run_with_opac_custom
    from pyradtran.models.aerosol_composite import CompositeAerosol
    from pyradtran.optics import opac

    species_file = str(opac._opac_root(None) / "standard_aerosol_files" / "continental_average.dat")

    mock_dataset = MagicMock()
    with patch("pyradtran.convenience.Runner.execute", return_value=mock_dataset) as mock_exec:
        run_with_opac_custom(species_file=species_file, sza=45.0, wl_min=545.0, wl_max=555.0)
        scene_arg = _get_scene_arg(mock_exec)
        assert scene_arg.aerosol is not None
        assert isinstance(scene_arg.aerosol, CompositeAerosol)
        assert len(scene_arg.aerosol.pieces) >= 1


def test_run_with_opac_preset_sets_disort_intcor_moments():
    """OPAC folded phase functions are Legendre moments; DISORT needs disort_intcor='moments'."""
    _need_opac_data()
    from unittest.mock import MagicMock, patch

    from pyradtran.convenience import run_with_opac_preset

    mock_dataset = MagicMock()
    with patch("pyradtran.convenience.Runner.execute", return_value=mock_dataset) as mock_exec:
        run_with_opac_preset(preset="maritime_clean", sza=45.0, wl_min=545.0, wl_max=555.0)
        scene_arg = _get_scene_arg(mock_exec)
        assert scene_arg.solver.disort_intcor == "moments"


def test_run_with_opac_custom_sets_disort_intcor_moments():
    """OPAC custom profile also folds Mie moments; DISORT needs disort_intcor='moments'."""
    _need_opac_data()
    from unittest.mock import MagicMock, patch

    from pyradtran.convenience import run_with_opac_custom
    from pyradtran.optics import opac

    species_file = str(opac._opac_root(None) / "standard_aerosol_files" / "continental_average.dat")

    mock_dataset = MagicMock()
    with patch("pyradtran.convenience.Runner.execute", return_value=mock_dataset) as mock_exec:
        run_with_opac_custom(species_file=species_file, sza=45.0, wl_min=545.0, wl_max=555.0)
        scene_arg = _get_scene_arg(mock_exec)
        assert scene_arg.solver.disort_intcor == "moments"
