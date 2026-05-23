"""Tests for uvspec output parsing (ASCII and NetCDF)."""

import numpy as np
import xarray as xr

from pyradtran.core.output_parser import parse_output


class TestParseAsciiOutput:
    def test_single_level_basic(self, tmp_path):
        out_file = tmp_path / "uvspec.out"
        content = (
            "# comment line\n"
            "300.0 1.23 0.45 0.01 2.0 1.0 0.5\n"
            "400.0 5.67 1.23 0.03 8.0 2.0 1.0\n"
            "500.0 10.0  2.00 0.05 15.0 3.0 1.5\n"
        )
        out_file.write_text(content)
        ds = parse_output(out_file, format="ascii")
        assert isinstance(ds, xr.Dataset)
        assert "wavelength" in ds.dims
        assert len(ds.wavelength) == 3
        assert "edir" in ds.data_vars

    def test_single_level_extracts_columns(self, tmp_path):
        out_file = tmp_path / "uvspec.out"
        content = "300.0 1.23 0.45 0.01 2.0 1.0 0.5\n"
        out_file.write_text(content)
        ds = parse_output(out_file, format="ascii")
        assert np.isclose(ds.edir.values[0], 1.23)
        assert np.isclose(ds.edn.values[0], 0.45)
        assert np.isclose(ds.eup.values[0], 0.01)
        assert np.isclose(ds.udir.values[0], 2.0)
        assert np.isclose(ds.udn.values[0], 1.0)
        assert np.isclose(ds.uup.values[0], 0.5)

    def test_two_levels(self, tmp_path):
        out_file = tmp_path / "uvspec.out"
        content = (
            "300.0 1.0 0.3 0.01 2.0 1.0 0.5\n"
            "300.0 0.8 0.2 0.00 1.5 0.8 0.3\n"
            "400.0 5.0 1.0 0.03 8.0 2.0 1.0\n"
            "400.0 4.0 0.8 0.02 6.0 1.5 0.7\n"
        )
        out_file.write_text(content)
        ds = parse_output(out_file, format="ascii", n_zout=2)
        assert "zout" in ds.dims
        assert len(ds.zout) == 2

    def test_output_user_custom_columns(self, tmp_path):
        out_file = tmp_path / "uvspec.out"
        content = "300.0 0.95\n400.0 0.88\n500.0 0.80\n"
        out_file.write_text(content)
        ds = parse_output(out_file, format="ascii", column_names=["wavelength", "transmittance"])
        assert "transmittance" in ds.data_vars


class TestParseNetcdfOutput:
    def test_netcdf_basic(self, tmp_path):
        wl = np.array([300.0, 400.0, 500.0])
        edir = np.array([1.23, 5.67, 10.0])
        ds_in = xr.Dataset(
            {"edir": ("wavelength", edir)},
            coords={"wavelength": wl},
        )
        nc_file = tmp_path / "uvspec.nc"
        ds_in.to_netcdf(nc_file)
        ds = parse_output(nc_file, format="netcdf")
        assert isinstance(ds, xr.Dataset)
        assert len(ds.wavelength) == 3
        assert np.isclose(ds.edir.values[1], 5.67)
