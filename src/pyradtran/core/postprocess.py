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


@dataclass(frozen=True)
class EnergyBudget:
    """Spectral column energy budget (W/m² per wavelength).

    Identity holds by construction:
    ``f_incident == f_up_toa + f_abs_surface + f_abs_atm``. ``f_abs_atm`` is the
    atmospheric-absorption residual; physically it must be >= 0.
    """

    f_incident: np.ndarray    # edir@toa = F0 * mu0
    f_up_toa: np.ndarray      # eup@toa (reflected + back-scattered out the top)
    f_abs_surface: np.ndarray  # (1 - albedo) * (edir + edn)@surface
    f_abs_atm: np.ndarray      # residual = f_incident - f_up_toa - f_abs_surface
    wavelength: np.ndarray


def compute_energy_budget(
    ds: xr.Dataset, albedo: float, *, f_incident: str = "edir"
) -> EnergyBudget:
    """Per-wavelength column energy budget in W/m².

    Convention: ``f_incident = edir @ TOA`` (the incident solar beam, = F0·μ0;
    matches :func:`add_budget_vars`). Surface = lowest zout level, TOA = highest.
    """
    if "zout" not in ds.dims or ds.sizes["zout"] < 2:
        raise ValueError(
            "Energy budget requires a 2-D dataset (wavelength, zout) with at "
            "least surface + TOA levels."
        )
    surface, toa = _boundary_indices(ds["zout"].values)
    f_inc = ds[f_incident].isel(zout=toa).values
    f_up_toa = ds["eup"].isel(zout=toa).values
    f_down_sfc = (ds["edir"].isel(zout=surface) + ds["edn"].isel(zout=surface)).values
    f_abs_surface = (1.0 - albedo) * f_down_sfc
    f_abs_atm = f_inc - f_up_toa - f_abs_surface
    return EnergyBudget(
        f_incident=f_inc,
        f_up_toa=f_up_toa,
        f_abs_surface=f_abs_surface,
        f_abs_atm=f_abs_atm,
        wavelength=np.asarray(ds["wavelength"].values, dtype=float),
    )


def assert_energy_conservation(
    ds: xr.Dataset, albedo: float, *, tol: float = 0.05, f_incident: str = "edir"
) -> EnergyBudget:
    """Compute the budget and assert physical bounds.

    Hard checks (exact physics, within ``tol`` for numerical noise):
      1. ``f_abs_atm >= -tol * f_incident``  (atmosphere cannot create energy)
      2. ``f_up_toa <= f_incident * (1 + tol)`` (cannot reflect more than incident)

    The identity ``f_inc == f_up_toa + f_abs_surface + f_abs_atm`` holds by
    construction (f_abs_atm is defined as the residual) and is not re-asserted.
    Returns the budget for logging / plotting. Raises AssertionError on violation.
    """
    b = compute_energy_budget(ds, albedo, f_incident=f_incident)
    eps = np.maximum(tol * b.f_incident, 1e-9)
    # Check the more specific physical impossibility first: eup@toa cannot
    # exceed f_incident. (When it does, the residual f_abs_atm is also negative,
    # so this must precede the absorption check to report the right root cause.)
    if np.any(b.f_up_toa > b.f_incident + eps):
        worst = float(np.max(b.f_up_toa - b.f_incident))
        raise AssertionError(
            f"TOA upwelling exceeds incident ({worst:.3f} W/m² over F_inc): "
            "physically impossible — check eup/edir column mapping."
        )
    if np.any(b.f_abs_atm < -eps):
        worst = float(np.min(b.f_abs_atm))
        raise AssertionError(
            f"Atmospheric absorption negative ({worst:.3f} W/m² < -{tol:.0%}·F_inc): "
            "column creates energy — check flux parsing / zout ordering."
        )
    return b


def _layer_centers(z_km: np.ndarray) -> np.ndarray:
    z = np.asarray(z_km, dtype=float)
    return 0.5 * (z[:-1] + z[1:])


def evaluate_composite_on_grid(comp, wl_um, z_km, n_legendre: int = 32) -> xr.Dataset:
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


def evaluate_blocks_on_grid(comp, wl_um, z_km, n_legendre: int = 32) -> dict[str, xr.Dataset]:
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
        profile = getattr(piece, "profile", None)
        if profile is not None:
            ds["rho_kg_m3"] = ("layer", np.asarray(profile.evaluate(centers), dtype=float))
        out[name] = ds
    return out
