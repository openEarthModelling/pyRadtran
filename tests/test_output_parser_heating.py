"""T1: 8-column uvspec ASCII output (fluxes + heating_rate) must name the 8th
column 'heating_rate', not 'col_7'."""

from pathlib import Path

from pyradtran.core.output_parser import HEATING_RATE_COLUMN, parse_output


def _write_ascii(tmp_path: Path, n_zout_rows: int, n_cols: int) -> Path:
    p = tmp_path / "out.dat"
    lines = []
    for w in range(500, 510):  # 10 wavelengths
        for _ in range(n_zout_rows):
            vals = [float(w)] + [0.1 * (c + 1) for c in range(n_cols - 1)]
            lines.append(" ".join(f"{v:.4f}" for v in vals))
    p.write_text("\n".join(lines) + "\n")
    return p


def test_8_column_ascii_names_heating_rate(tmp_path):
    p = _write_ascii(tmp_path, n_zout_rows=1, n_cols=8)
    ds = parse_output(p, format="ascii", n_zout=1)
    assert HEATING_RATE_COLUMN in ds.data_vars, (
        f"expected '{HEATING_RATE_COLUMN}' in {list(ds.data_vars)}"
    )
    # Standard 7 flux columns still present.
    for col in ("wavelength", "edir", "edn", "eup", "udir", "udn", "uup"):
        assert col in ds.data_vars or col in ds.coords


def test_7_column_ascii_unchanged(tmp_path):
    p = _write_ascii(tmp_path, n_zout_rows=1, n_cols=7)
    ds = parse_output(p, format="ascii", n_zout=1)
    assert HEATING_RATE_COLUMN not in ds.data_vars
    assert "col_0" not in ds.data_vars  # standard naming kicked in
