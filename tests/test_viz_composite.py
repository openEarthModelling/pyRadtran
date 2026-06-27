"""Tests for composite/block diagnostic plots (headless, synthetic data)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from pyradtran.viz.composite import plot_block_profiles, plot_composite_optics


def _composite_ds() -> xr.Dataset:
    wl = np.array([0.3, 0.5, 0.8])
    alt = np.array([4.0, 2.0, 0.5])
    tau = np.array([[0.9, 0.5, 0.1], [0.8, 0.4, 0.08], [0.7, 0.3, 0.06]])
    return xr.Dataset(
        {
            "tau": (("wavelength", "layer"), tau),
            "ssa": (("wavelength", "layer"), np.full((3, 3), 0.9)),
            "g": (("wavelength", "layer"), np.full((3, 3), 0.6)),
        },
        coords={"wavelength": wl, "layer": np.arange(3), "altitude_km": ("layer", alt)},
    )


def _block_dict() -> dict[str, xr.Dataset]:
    alt = np.array([4.0, 2.0, 0.5])
    return {
        "soot": xr.Dataset(
            {
                "tau": (("wavelength", "layer"), np.full((2, 3), 0.5)),
                "rho_kg_m3": ("layer", np.array([1e-6, 5e-7, 1e-7])),
            },
            coords={
                "wavelength": np.array([0.3, 0.5]),
                "layer": np.arange(3),
                "altitude_km": ("layer", alt),
            },
        ),
        "dust": xr.Dataset(
            {
                "tau": (("wavelength", "layer"), np.full((2, 3), 0.2)),
                "rho_kg_m3": ("layer", np.array([2e-6, 1e-6, 2e-7])),
            },
            coords={
                "wavelength": np.array([0.3, 0.5]),
                "layer": np.arange(3),
                "altitude_km": ("layer", alt),
            },
        ),
    }


def test_plot_composite_optics_draws_pcolormesh():
    fig, ax = plot_composite_optics(_composite_ds(), quantity="tau")
    assert len(ax.collections) >= 1  # a QuadMesh was added
    plt.close(fig)


def test_plot_block_profiles_draws_one_line_per_block():
    fig, ax = plot_block_profiles(_block_dict(), quantity="tau")
    # 2 blocks, summed over wavelength -> 2 lines (one per block).
    assert len(ax.lines) == 2
    plt.close(fig)
