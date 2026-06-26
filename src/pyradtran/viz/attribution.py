"""Component-attribution plot (the LEGO killer plot).

Consumes an AttributionResult by duck-typing (AttributionLike). This module
intentionally does NOT import pyradtran.workflow — the workflow produces the
result, viz renders it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import xarray as xr

from pyradtran.viz._style import get_palette, require_mpl, save, set_theme


@runtime_checkable
class AttributionLike(Protocol):
    full: xr.Dataset
    contributions: dict[str, xr.Dataset]


def _get(obj, key):
    """Read ``key`` from an attribute-style (``.full``) or mapping (``["full"]``) object."""
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, dict):
        return obj[key]
    raise TypeError(f"Expected an object with .{key} or ['{key}']; got {type(obj).__name__}")


def plot_component_attribution(
    result: AttributionLike,
    *,
    variable: str = "edir",
    level="surface",
    ax=None,
    save_path=None,
):
    """Plot per-block contributions (full - leave_one_out) plus the full curve."""
    require_mpl()
    import matplotlib.pyplot as plt

    set_theme()
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    full = _get(result, "full")
    contributions = _get(result, "contributions")
    wl = np.asarray(full["wavelength"].values, dtype=float)

    idx = 0
    if "zout" in full[variable].dims:
        z = np.asarray(full["zout"].values, dtype=float)
        idx = int(np.argmin(z)) if level == "surface" else int(level)
        full_y = full[variable].values[:, idx]
    else:
        full_y = np.asarray(full[variable].values, dtype=float)

    n = 1 + len(contributions)
    colors = get_palette(n)
    ax.plot(wl, full_y, color=colors[0], linewidth=2.0, label="full")
    for color, (name, ds_c) in zip(colors[1:], contributions.items(), strict=True):
        y = np.asarray(ds_c[variable].values, dtype=float)
        if "zout" in ds_c[variable].dims:
            y = y[:, idx]
        ax.plot(wl, y, color=color, linewidth=1.5, label=name)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(f"{variable} contribution")
    ax.legend(loc="best")
    if save_path is not None:
        save(fig, Path(save_path))
    return fig, ax
