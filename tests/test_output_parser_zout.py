"""Tests for physical zout-altitude retention and the heating-rate column."""
from __future__ import annotations

from pathlib import Path

from pyradtran.core.output_parser import (
    HEATING_RATE_COLUMN,
    parse_output,
    resolve_zout_tokens,
)


def _write_ascii(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "out.dat"
    p.write_text(text)
    return p


def test_resolve_zout_tokens_handles_strings_and_floats():
    assert resolve_zout_tokens([0, "toa"], atmosphere_top_km=120.0) == [0.0, 120.0]
    assert resolve_zout_tokens(["surface", "top", 5.0], atmosphere_top_km=100.0) == [
        0.0,
        100.0,
        5.0,
    ]


def test_parse_ascii_uses_physical_zout_levels(tmp_path):
    # 2 wavelengths x 2 zout levels, standard 7 columns.
    # Each wavelength's zout levels are consecutive; zout index 0 = 0.0 km
    # (surface, attenuated), index 1 = 120 km (TOA, unattenuated).
    text = "\n".join(
        [
            "300 0.8 0.4 0.1 0.0 0.0 0.0",
            "300 1.0 0.5 0.1 0.0 0.0 0.0",
            "500 0.9 0.5 0.2 0.0 0.0 0.0",
            "500 1.0 0.6 0.2 0.0 0.0 0.0",
        ]
    )
    path = _write_ascii(tmp_path, text)
    ds = parse_output(
        path,
        format="ascii",
        n_zout=2,
        zout_levels_km=[0.0, 120.0],
    )
    assert list(ds["zout"].values) == [0.0, 120.0]
    assert ds["edir"].sel(zout=0.0).values.tolist() == [0.8, 0.9]  # surface = low z


def test_parse_ascii_falls_back_to_integer_index_when_no_levels(tmp_path):
    text = "\n".join(
        [
            "300 1.0 0.5 0.1 0.0 0.0 0.0",
            "300 0.8 0.4 0.1 0.0 0.0 0.0",
        ]
    )
    path = _write_ascii(tmp_path, text)
    ds = parse_output(path, format="ascii", n_zout=2)
    assert list(ds["zout"].values) == [0, 1]  # legacy integer index preserved


def test_parse_ascii_heating_rate_column(tmp_path):
    # 1 wavelength x 1 zout, 8 columns (standard 7 + heating_rate).
    text = "300 1.0 0.5 0.1 0.0 0.0 0.0 0.234\n"
    path = _write_ascii(tmp_path, text)
    ds = parse_output(
        path,
        format="ascii",
        n_zout=1,
        column_names=[
            "wavelength",
            "edir",
            "edn",
            "eup",
            "udir",
            "udn",
            "uup",
            HEATING_RATE_COLUMN,
        ],
    )
    assert HEATING_RATE_COLUMN in ds.data_vars
    assert float(ds[HEATING_RATE_COLUMN].values.item()) == 0.234
