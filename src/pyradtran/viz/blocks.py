"""Per-block spectral optics plot: tau / ssa / g vs wavelength, one line per block.

For tau: column-integrated (sum over layers). For ssa/g: tau-weighted column
average (the physically meaningful column value).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from pyradtran.viz._style import get_palette, require_mpl, save, set_theme

_VALID = {"tau", "ssa", "g"}
_LABELS = {
    "tau": "Column optical depth τ(λ)",
    "ssa": "Single-scattering albedo ⟨ω⟩(λ)",
    "g": "Asymmetry parameter ⟨g⟩(λ)",
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


def _column_reduce(ds: xr.Dataset, quantity: str) -> np.ndarray:
    """Reduce (wavelength, layer) -> (wavelength,) for plotting."""
    tau = np.asarray(ds["tau"].values, dtype=float)  # (wl, layer)
    if quantity == "tau":
        return tau.sum(axis=1)
    q = np.asarray(ds[quantity].values, dtype=float)  # (wl, layer)
    w = np.abs(tau)
    wsum = w.sum(axis=1, keepdims=True)
    safe = np.where(wsum > 0, wsum, 1.0)
    return (q * w / safe).sum(axis=1)  # tau-weighted mean over layers


def plot_block_spectral_optics(
    per_block_ds_dict: dict[str, xr.Dataset],
    *,
    quantity: str = "tau",
    ax=None,
    save_path=None,
):
    """One line per block of column tau / ssa / g vs wavelength.

    Args:
        per_block_ds_dict: ``{block_name: xr.Dataset}`` from
            :func:`pyradtran.core.postprocess.evaluate_blocks_on_grid`. Each
            dataset must carry the requested quantity (tau/ssa/g) on a
            (wavelength, layer) grid.
        quantity: one of ``"tau"`` (column sum), ``"ssa"`` or ``"g"``
            (tau-weighted column mean).
    """
    if quantity not in _VALID:
        raise ValueError(f"Unknown quantity {quantity!r}; choose from {sorted(_VALID)}")
    fig, ax = _ensure_axes(ax)
    colors = get_palette(len(per_block_ds_dict))
    for color, (name, bds) in zip(colors, per_block_ds_dict.items(), strict=True):
        if quantity not in bds.data_vars:
            continue
        wl = np.asarray(bds["wavelength"].values, dtype=float)
        y = _column_reduce(bds, quantity)
        ax.plot(wl, y, color=color, label=name, linewidth=1.5, marker="o", markersize=3)
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel(_LABELS[quantity])
    ax.legend(loc="best")
    if save_path is not None:
        save(fig, Path(save_path))
    return fig, ax
