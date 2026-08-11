"""T1 follow-up: parse_heating_ascii handles uvspec's wide heating-rate format.

libRadtran heating_rate mode emits: 1 header row (0.0 marker + zout levels),
then per-wavelength rows (wavelength + K/day at each zout). Distinct from the
standard long-format flux output.
"""

import numpy as np

from pyradtran.core.output_parser import parse_heating_ascii


def test_parse_heating_ascii_wide_format(tmp_path):
    p = tmp_path / "heat.out"
    p.write_text(
        "0.000e+00  0.000000  2.000000  5.000000\n"
        "400.120  5.99e-06  6.63e-06  7.09e-06\n"
        "400.360  6.99e-06  7.64e-06  8.10e-06\n"
    )
    ds = parse_heating_ascii(p, zout_levels_km=[0.0, 2.0, 5.0])
    assert "heating_rate" in ds.data_vars
    assert ds["heating_rate"].dims == ("wavelength", "zout")
    assert ds.sizes["wavelength"] == 2
    assert ds.sizes["zout"] == 3
    assert list(ds["zout"].values) == [0.0, 2.0, 5.0]
    assert np.isclose(float(ds["heating_rate"].isel(wavelength=0, zout=1).values), 6.63e-06)
    assert np.isclose(float(ds["heating_rate"].isel(wavelength=1, zout=2).values), 8.10e-06)


def test_parse_heating_ascii_raises_on_column_count_mismatch(tmp_path):
    import pytest

    p = tmp_path / "heat.out"
    p.write_text(
        "0.000e+00  0.000000  2.000000\n"
        "400.120  5.99e-06  6.63e-06\n"
    )
    with pytest.raises(ValueError, match="per-level columns"):
        parse_heating_ascii(p, zout_levels_km=[0.0, 2.0, 5.0])  # 3 zout, file has 2


def test_parse_heating_ascii_empty_file_raises(tmp_path):
    import pytest

    p = tmp_path / "heat.out"
    p.write_text("")
    with pytest.raises(ValueError, match="Empty"):
        parse_heating_ascii(p, zout_levels_km=[0.0, 2.0])


def test_parse_heating_ascii_handles_no_header_marker(tmp_path):
    """If libRadtran ever omits the marker row, still parse as wide data."""
    p = tmp_path / "heat.out"
    p.write_text("400.120  5.99e-06  6.63e-06\n400.360  6.99e-06  7.64e-06\n")
    ds = parse_heating_ascii(p, zout_levels_km=[0.0, 2.0])
    assert ds.sizes["wavelength"] == 2
    assert float(ds["heating_rate"].isel(wavelength=0, zout=0).values) == 5.99e-06
