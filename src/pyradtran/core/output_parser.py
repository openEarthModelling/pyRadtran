"""Parse uvspec output files into xarray.Dataset.

Supports both ASCII (default DISORT 7-column and output_user custom)
and NetCDF output formats.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import xarray as xr

#: Name of the heating-rate data variable when libRadtran's heat output is
#: requested (e.g. via ``solver.dynamic_heat_unit``). The ASCII parser handles
#: arbitrary column names; this constant pins the convention used across
#: pyRadtran so callers and the viz layer agree on the variable name.
HEATING_RATE_COLUMN = "heating_rate"


def resolve_zout_tokens(
    zout: Sequence[float | str], atmosphere_top_km: float = 120.0
) -> list[float]:
    """Resolve uvspec ``zout`` tokens (which may contain ``"toa"``/``"surface"``)
    into numeric altitudes in km above ground level.

    Args:
        zout: Output levels as given to uvspec (floats and/or keyword strings).
        atmosphere_top_km: Altitude (km) to which ``"toa"``/``"top"`` resolve.

    Returns:
        List of float altitudes in km, same length and order as ``zout``.
    """
    resolved: list[float] = []
    for z in zout:
        if isinstance(z, str):
            token = z.strip().lower()
            if token in ("toa", "top"):
                resolved.append(float(atmosphere_top_km))
            elif token in ("surface", "ground", "sfc", "0"):
                resolved.append(0.0)
            else:
                resolved.append(float(z))
        else:
            resolved.append(float(z))
    return resolved


def parse_output(
    output_path: str | Path,
    format: str = "netcdf",
    n_zout: int = 1,
    column_names: list[str] | None = None,
    zout_levels_km: Sequence[float] | None = None,
) -> xr.Dataset:
    """Parse uvspec output file into xarray.Dataset.

    Args:
        output_path: Path to uvspec output file.
        format: Output format — "netcdf" or "ascii".
        n_zout: Number of zout levels in output (for ASCII multi-level).
        column_names: Custom column names for output_user ASCII output.
        zout_levels_km: Physical zout altitudes in km. When provided (ASCII path),
            used as the ``zout`` coordinate instead of an integer index.

    Returns:
        xarray.Dataset with wavelength (and optionally zout) as coordinates.
    """
    output_path = Path(output_path)

    if format == "netcdf":
        return _parse_netcdf(output_path)
    else:
        return _parse_ascii(
            output_path,
            n_zout=n_zout,
            column_names=column_names,
            zout_levels_km=zout_levels_km,
        )


def _parse_netcdf(path: Path) -> xr.Dataset:
    """Read a NetCDF output file produced by uvspec.

    libRadtran builds whose NetCDF support is broken at runtime (e.g. an ABI
    mismatch with the system libnetcdf) write a 0-byte .nc; xarray then fails
    with a cryptic "did not find a match in any of ... IO backends". Detect
    empty/missing output and point the user at ``format="ascii"``, which works
    everywhere and yields an equivalent xarray.Dataset.
    """
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError(
            f"uvspec produced an empty or missing NetCDF file ({path}). The "
            "libRadtran build may not support NetCDF output — use format='ascii' "
            "instead (OutputConfig(format='ascii') or .set_output(format='ascii')), "
            "which yields an equivalent xarray.Dataset."
        )
    ds = xr.open_dataset(path)
    ds.load()
    ds.close()
    return ds


_STANDARD_COLUMNS = ["wavelength", "edir", "edn", "eup", "udir", "udn", "uup"]


def _parse_ascii(
    path: Path,
    n_zout: int = 1,
    column_names: list[str] | None = None,
    zout_levels_km: Sequence[float] | None = None,
) -> xr.Dataset:
    """Read ASCII uvspec output.

    For multi-level output (n_zout > 1), uvspec interleaves rows by wavelength.
    """
    lines = []
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                lines.append(stripped)

    if not lines:
        raise ValueError(f"Empty output file: {path}")

    data = np.loadtxt(lines)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    n_rows, n_cols = data.shape

    if column_names is None:
        if n_cols == 7:
            column_names = _STANDARD_COLUMNS
        elif n_cols == 2:
            column_names = ["wavelength", "value"]
        else:
            column_names = [f"col_{i}" for i in range(n_cols)]

    if n_zout == 1:
        data_vars = {}
        wavelength = None
        for i, name in enumerate(column_names):
            if name == "wavelength":
                wavelength = data[:, i]
            else:
                data_vars[name] = ("wavelength", data[:, i])
        if wavelength is None:
            wavelength = data[:, 0]
        return xr.Dataset(data_vars, coords={"wavelength": wavelength})
    else:
        n_wl = n_rows // n_zout
        if n_rows != n_wl * n_zout:
            raise ValueError(
                f"Expected {n_wl * n_zout} rows for {n_wl} wavelengths x {n_zout} zout levels, "
                f"got {n_rows}"
            )
        reshaped = data.reshape(n_wl, n_zout, n_cols)
        data_vars = {}
        coords = {}
        for i, name in enumerate(column_names):
            if name == "wavelength":
                coords["wavelength"] = reshaped[:, 0, i]
            else:
                data_vars[name] = (("wavelength", "zout"), reshaped[:, :, i])
        if zout_levels_km is not None:
            coords["zout"] = np.asarray(zout_levels_km, dtype=float)
        else:
            coords["zout"] = np.arange(n_zout)
        return xr.Dataset(data_vars, coords=coords)
