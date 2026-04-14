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
