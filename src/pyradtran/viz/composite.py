"""Composite and per-block diagnostic plots (pure data-in -> fig-out)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from pyradtran.viz._style import get_palette, require_mpl, save, set_theme

_QUANTITY_LABELS = {
    "tau": "Optical depth τ",
    "ssa": "Single-scattering albedo",
    "g": "Asymmetry parameter g",
}
_BLOCK_QUANTITY_LABELS = {
    "tau": "Optical depth τ (spectral sum)",
    "rho": "Mass concentration (kg/m³)",
}


def _ensure_axes(ax=None):
    require_mpl()
    import matplotlib.pyplot as plt

    set_theme()
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    return fig, ax


def plot_composite_optics(ds: xr.Dataset, *, quantity: str = "tau", ax=None, save_path=None):
    """pcolormesh of τ / ssa / g over (wavelength, altitude) for the mixed composite."""
    if quantity not in ds.data_vars:
        raise ValueError(f"Dataset has no '{quantity}' variable.")
    fig, ax = _ensure_axes(ax)
    wl = np.asarray(ds["wavelength"].values, dtype=float)
    alt = np.asarray(ds["altitude_km"].values, dtype=float)
    mesh = ax.pcolormesh(wl, alt, ds[quantity].values.T, shading="auto")
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title(_QUANTITY_LABELS.get(quantity, quantity))
    fig.colorbar(mesh, ax=ax, label=_QUANTITY_LABELS.get(quantity, quantity))
    if save_path is not None:
        save(fig, Path(save_path))
    return fig, ax


def plot_block_profiles(
    per_block_ds_dict: dict[str, xr.Dataset],
    *,
    quantity: str = "tau",
    ax=None,
    save_path=None,
):
    """Per-block τ(z) (spectrally summed over wavelength) or ρ(z) vs altitude.

    The τ curve is ``tau.sum(axis=0)`` — the per-layer sum over wavelength, a
    rough "spectral optical burden per layer" overview. For a physically
    meaningful τ(z) at a specific band, slice the block dataset to that
    wavelength before passing it in.
    """
    fig, ax = _ensure_axes(ax)
    colors = get_palette(len(per_block_ds_dict))
    for color, (name, bds) in zip(colors, per_block_ds_dict.items(), strict=True):
        alt = np.asarray(bds["altitude_km"].values, dtype=float)
        if quantity == "tau":
            y = np.asarray(bds["tau"].values, dtype=float).sum(axis=0)  # sum over wavelength
        elif quantity == "rho":
            if "rho_kg_m3" not in bds.data_vars:
                continue  # block has no mass profile (e.g. DirectLayerOpticsBlock)
            y = np.asarray(bds["rho_kg_m3"].values, dtype=float)
        else:
            raise ValueError(f"Unknown quantity: {quantity!r} (use 'tau' or 'rho')")
        ax.plot(y, alt, color=color, label=name, linewidth=1.5, marker="o", markersize=3)
    ax.set_xlabel(_BLOCK_QUANTITY_LABELS.get(quantity, quantity))
    ax.set_ylabel("Altitude (km)")
    ax.legend(loc="best")
    if save_path is not None:
        save(fig, Path(save_path))
    return fig, ax
