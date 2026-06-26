"""RT-result plots consuming xarray.Dataset (pure data-in -> fig-out)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from pyradtran.core.output_parser import HEATING_RATE_COLUMN
from pyradtran.core.postprocess import add_budget_vars
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


def _maybe_save(fig, save_path):
    if save_path is not None:
        save(fig, Path(save_path))


def _surface_index(ds: xr.Dataset) -> int:
    """Index of the surface zout level (lowest altitude), or 0 for 1-D datasets."""
    if "zout" not in ds.dims:
        return 0
    z = np.asarray(ds["zout"].values, dtype=float)
    return int(np.argmin(z))


def plot_spectral(
    ds: xr.Dataset,
    *,
    variables=("edir", "edn", "eup"),
    level="surface",
    ax=None,
    save_path=None,
):
    """Plot selected flux variables vs wavelength at a single zout level.

    For 2-D datasets, ``level="surface"`` selects the lowest-altitude zout level.
    """
    fig, ax = _ensure_axes(ax)
    wl = np.asarray(ds["wavelength"].values, dtype=float)
    idx = _surface_index(ds) if level == "surface" else int(level)
    colors = get_palette(len(variables))
    for color, var in zip(colors, variables, strict=True):
        y = ds[var].values
        y = y[:, idx] if y.ndim == 2 else y
        ax.plot(wl, y, color=color, label=var, linewidth=1.5)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Irradiance")
    ax.legend(loc="best")
    _maybe_save(fig, save_path)
    return fig, ax


def plot_flux_profile(
    ds: xr.Dataset,
    *,
    variable: str = "edir",
    wavelength_nm: float = 550.0,
    ax=None,
    save_path=None,
):
    """Plot a flux variable vs altitude (km) at the wavelength nearest ``wavelength_nm``."""
    if "zout" not in ds.dims:
        raise ValueError("plot_flux_profile requires a 2-D dataset with a zout dimension.")
    fig, ax = _ensure_axes(ax)
    wl = np.asarray(ds["wavelength"].values, dtype=float)
    z = np.asarray(ds["zout"].values, dtype=float)
    i = int(np.argmin(np.abs(wl - wavelength_nm)))
    ax.plot(ds[variable].isel(wavelength=i).values, z, linewidth=1.5)
    ax.set_xlabel(f"{variable} @ {wl[i]:.0f} nm")
    ax.set_ylabel("Altitude (km)")
    _maybe_save(fig, save_path)
    return fig, ax


def plot_heating_rate(
    ds: xr.Dataset,
    *,
    wavelength_nm: float | None = None,
    ax=None,
    save_path=None,
):
    """Plot heating rate vs altitude (km).

    If ``wavelength_nm`` is None, plot all wavelengths as faint background
    lines; otherwise plot the nearest-wavelength line bold.
    """
    if HEATING_RATE_COLUMN not in ds.data_vars:
        raise ValueError(
            f"Dataset has no '{HEATING_RATE_COLUMN}' variable; request heating-rate "
            "output from libRadtran (set dynamic_heat_unit)."
        )
    if "zout" not in ds.dims:
        raise ValueError("plot_heating_rate requires a 2-D dataset with a zout dimension.")
    fig, ax = _ensure_axes(ax)
    wl = np.asarray(ds["wavelength"].values, dtype=float)
    z = np.asarray(ds["zout"].values, dtype=float)
    heat = ds[HEATING_RATE_COLUMN].values
    if wavelength_nm is None:
        for i in range(wl.size):
            ax.plot(heat[i, :], z, linewidth=0.5, alpha=0.4)
    else:
        i = int(np.argmin(np.abs(wl - wavelength_nm)))
        ax.plot(heat[i, :], z, linewidth=1.5)
        ax.set_xlabel(f"Heating rate @ {wl[i]:.0f} nm")
    ax.set_ylabel("Altitude (km)")
    if wavelength_nm is None:
        ax.set_xlabel("Heating rate (all wavelengths)")
    _maybe_save(fig, save_path)
    return fig, ax


def plot_budget(
    ds: xr.Dataset,
    *,
    components=("transmittance", "reflectance", "absorptance"),
    ax=None,
    save_path=None,
):
    """Stacked-area plot of T/R/A vs wavelength. Requires a budget-enriched dataset."""
    missing = [c for c in components if c not in ds.data_vars]
    if missing:
        raise ValueError(
            f"Dataset missing budget variables {missing}; run add_budget_vars(ds) first."
        )
    fig, ax = _ensure_axes(ax)
    wl = np.asarray(ds["wavelength"].values, dtype=float)
    colors = get_palette(len(components))
    values = np.vstack([np.asarray(ds[c].values, dtype=float) for c in components])
    ax.stackplot(wl, values, labels=list(components), colors=colors, alpha=0.85)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Fraction of incident flux")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="best")
    _maybe_save(fig, save_path)
    return fig, ax


def plot_rt_overview(ds: xr.Dataset, *, wavelength_nm: float = 550.0):
    """Convenience multi-panel: spectral (surface) + flux profile + budget."""
    require_mpl()
    import matplotlib.pyplot as plt

    set_theme()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    plot_spectral(ds, ax=axes[0])
    if "zout" in ds.dims:
        plot_flux_profile(ds, wavelength_nm=wavelength_nm, ax=axes[1])
    else:
        axes[1].text(0.5, 0.5, "no zout dim", ha="center", va="center")
    plot_budget(add_budget_vars(ds), ax=axes[2])
    fig.tight_layout()
    return fig, axes
