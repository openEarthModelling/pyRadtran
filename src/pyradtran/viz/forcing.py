"""Climate-forcing plots: direct radiative forcing (DRF) + spectral attribution.

DRF sign convention follows IPCC: negative = cooling (aerosol reflects/absorbs
more than the clean column). DRF_atm (shaded) = atmospheric-absorption forcing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pyradtran.viz._style import get_palette, require_mpl, save, set_theme


def _ensure_axes(ax=None):
    require_mpl()
    import matplotlib.pyplot as plt

    set_theme()
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    return fig, ax


def plot_drf_spectral(
    wavelength_nm: np.ndarray,
    drf_toa: np.ndarray,
    drf_surf: np.ndarray,
    ax=None,
    save_path=None,
):
    """Two lines (TOA, surface DRF) + shaded atmospheric-absorption forcing band.

    Args:
        wavelength_nm: wavelength axis in nm.
        drf_toa, drf_surf: per-wavelength DRF in W/m² (negative = cooling).
    """
    fig, ax = _ensure_axes(ax)
    ax.axhline(0.0, color="k", linewidth=0.8, linestyle="--")
    ax.plot(wavelength_nm, drf_toa, color="#1f77b4", linewidth=1.8, label="TOA")
    ax.plot(wavelength_nm, drf_surf, color="#d62728", linewidth=1.8, label="Surface")
    ax.fill_between(
        wavelength_nm,
        drf_toa,
        drf_surf,
        color="#ff7f0e",
        alpha=0.25,
        label="Atmosphere",
    )
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Direct radiative forcing (W/m²)")
    ax.legend(loc="best")
    if save_path is not None:
        save(fig, Path(save_path))
    return fig, ax


def plot_spectral_attribution(
    result,
    variable: str = "edir",
    level: str = "surface",
    ax=None,
    save_path=None,
):
    """Stacked area of per-block contribution(λ) for one variable/level.

    Args:
        result: :class:`pyradtran.workflow.AttributionResult` (contributions are
            already spectral xr.Datasets).
        variable: flux variable to attribute (edir, edn, eup, ...).
        level: "surface" (min zout) or "toa" (max zout).
    """
    fig, ax = _ensure_axes(ax)
    zout = result.full["zout"].values
    level_idx = int(np.argmin(zout)) if level == "surface" else int(np.argmax(zout))
    wl = np.asarray(result.full["wavelength"].values, dtype=float)

    names = list(result.contributions)
    colors = get_palette(len(names))
    stack = np.zeros_like(wl, dtype=float)
    for color, name in zip(colors, names, strict=True):
        c = np.asarray(
            result.contributions[name][variable].isel(zout=level_idx).values, dtype=float
        )
        ax.fill_between(wl, stack, stack + c, color=color, alpha=0.55, label=name, linewidth=0)
        stack = stack + c
    ax.plot(wl, stack, color="k", linewidth=1.0, linestyle=":", label="Σ contributions")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(f"Attribution of {variable} @ {level} (W/m²)")
    ax.legend(loc="best")
    if save_path is not None:
        save(fig, Path(save_path))
    return fig, ax
