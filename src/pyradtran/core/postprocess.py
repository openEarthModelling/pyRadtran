"""Pure dataset post-processing for RT outputs (no Runner / no libRadtran).

Transforms raw parsed flux datasets into physically meaningful quantities.
Kept separate from :mod:`pyradtran.viz` so the same derivations serve numeric
use (logging, NetCDF export, comparisons) without importing matplotlib.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class BudgetResult:
    """Spectral transmittance / reflectance / absorptance of the column."""

    transmittance: np.ndarray
    reflectance: np.ndarray
    absorptance: np.ndarray
    wavelength: np.ndarray


def _boundary_indices(zout: np.ndarray) -> tuple[int, int]:
    """Return ``(surface_idx, toa_idx)`` — the min- and max-altitude zout levels."""
    z = np.asarray(zout, dtype=float)
    return int(np.argmin(z)), int(np.argmax(z))


def _budget_arrays(ds: xr.Dataset, *, f_incident: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "zout" not in ds.dims or ds.sizes["zout"] < 2:
        raise ValueError(
            "Budget derivation requires a 2-D dataset (wavelength, zout) with at "
            "least surface + TOA levels."
        )
    surface, toa = _boundary_indices(ds["zout"].values)
    f_inc = ds[f_incident].isel(zout=toa).values
    f_down_sfc = (ds["edir"].isel(zout=surface) + ds["edn"].isel(zout=surface)).values
    f_up_toa = ds["eup"].isel(zout=toa).values
    with np.errstate(divide="ignore", invalid="ignore"):
        trans = np.where(f_inc != 0, f_down_sfc / f_inc, np.nan)
        refl = np.where(f_inc != 0, f_up_toa / f_inc, np.nan)
    absorp = 1.0 - trans - refl
    return trans, refl, absorp


def add_budget_vars(ds: xr.Dataset, *, f_incident: str = "edir") -> xr.Dataset:
    """Return a copy of ``ds`` with ``transmittance``/``reflectance``/``absorptance``.

    Conventions: ``T = (edir+edn)@surface / f_incident@TOA``,
    ``R = eup@TOA / f_incident@TOA``, ``A = 1 - T - R``. Surface = lowest zout
    level, TOA = highest zout level.
    """
    trans, refl, absorp = _budget_arrays(ds, f_incident=f_incident)
    out = ds.copy()
    out["transmittance"] = ("wavelength", trans)
    out["reflectance"] = ("wavelength", refl)
    out["absorptance"] = ("wavelength", absorp)
    return out


def compute_budget(ds: xr.Dataset, *, f_incident: str = "edir") -> BudgetResult:
    """Typed companion to :func:`add_budget_vars` returning a :class:`BudgetResult`."""
    trans, refl, absorp = _budget_arrays(ds, f_incident=f_incident)
    return BudgetResult(
        transmittance=trans,
        reflectance=refl,
        absorptance=absorp,
        wavelength=np.asarray(ds["wavelength"].values, dtype=float),
    )


def _layer_centers(z_km: np.ndarray) -> np.ndarray:
    z = np.asarray(z_km, dtype=float)
    return 0.5 * (z[:-1] + z[1:])


def evaluate_composite_on_grid(
    comp, wl_um, z_km, n_legendre: int = 32
) -> xr.Dataset:
    """Evaluate the mixed composite optics on a (wavelength, layer) grid.

    Calls ``comp.evaluate(...)`` (pure analytic mixing — no RT) and wraps the
    resulting :class:`LayerOptics` as an :class:`xarray.Dataset` with a
    ``layer`` dimension indexed by layer-center altitude.
    """
    wl = np.asarray(wl_um, dtype=float)
    z = np.asarray(z_km, dtype=float)
    lo = comp.evaluate(wl_um=wl, z_km=z, n_legendre=n_legendre)
    centers = _layer_centers(z)
    return xr.Dataset(
        {
            "tau": (("wavelength", "layer"), lo.tau),
            "ssa": (("wavelength", "layer"), lo.ssa),
            "g": (("wavelength", "layer"), lo.g),
        },
        coords={
            "wavelength": wl,
            "layer": np.arange(centers.size),
            "altitude_km": ("layer", centers),
        },
    )


def evaluate_blocks_on_grid(
    comp, wl_um, z_km, n_legendre: int = 32
) -> dict[str, xr.Dataset]:
    """Per-piece ``(tau, rho)`` on the grid; one dataset per block name.

    ``tau`` comes from each piece's ``to_layer_optics``; ``rho_kg_m3`` is added
    only for pieces that carry a ``profile`` (e.g. :class:`PlacedBlock`).
    """
    wl = np.asarray(wl_um, dtype=float)
    z = np.asarray(z_km, dtype=float)
    centers = _layer_centers(z)
    out: dict[str, xr.Dataset] = {}
    for piece in comp.pieces:
        lo = piece.to_layer_optics(wl, z, n_legendre=n_legendre)
        name = getattr(piece, "name", type(piece).__name__)
        ds = xr.Dataset(
            {"tau": (("wavelength", "layer"), lo.tau)},
            coords={
                "wavelength": wl,
                "layer": np.arange(centers.size),
                "altitude_km": ("layer", centers),
            },
        )
        profile = getattr(piece, "profile", None)
        if profile is not None:
            ds["rho_kg_m3"] = ("layer", np.asarray(profile.evaluate(centers), dtype=float))
        out[name] = ds
    return out
