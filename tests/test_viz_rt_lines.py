"""Tests for RT line/profile plots (headless, synthetic data)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from pyradtran.core.output_parser import HEATING_RATE_COLUMN
from pyradtran.viz.rt import plot_flux_profile, plot_heating_rate, plot_spectral


def _spectral_ds() -> xr.Dataset:
    wl = np.array([300.0, 500.0, 800.0])
    return xr.Dataset(
        {
            "edir": ("wavelength", np.array([1.0, 0.9, 0.8])),
            "edn": ("wavelength", np.array([0.1, 0.15, 0.2])),
            "eup": ("wavelength", np.array([0.05, 0.08, 0.1])),
        },
        coords={"wavelength": wl},
    )


def _profile_ds() -> xr.Dataset:
    wl = np.array([300.0, 550.0, 800.0])
    zout = np.array([0.0, 4.0, 120.0])  # surface, mid, TOA
    edir = np.array([[0.5, 0.7, 1.0], [0.6, 0.8, 1.0], [0.7, 0.9, 1.0]])
    heat = np.array([[0.1, 0.05, 0.0], [0.12, 0.06, 0.0], [0.14, 0.07, 0.0]])
    return xr.Dataset(
        {
            "edir": (("wavelength", "zout"), edir),
            HEATING_RATE_COLUMN: (("wavelength", "zout"), heat),
        },
        coords={"wavelength": wl, "zout": zout},
    )


def test_plot_spectral_draws_one_line_per_variable():
    fig, ax = plot_spectral(_spectral_ds(), variables=("edir", "edn", "eup"))
    assert len(ax.lines) == 3
    assert "wavelength" in ax.get_xlabel().lower()
    plt.close(fig)


def test_plot_flux_profile_uses_physical_altitude_axis():
    ds = _profile_ds()
    fig, ax = plot_flux_profile(ds, variable="edir", wavelength_nm=550.0)
    assert len(ax.lines) == 1
    # y-axis is altitude in km -> max ~120
    ylim = ax.get_ylim()
    assert max(abs(v) for v in ylim) >= 100.0
    plt.close(fig)


def test_plot_heating_rate_profiles_against_altitude():
    ds = _profile_ds()
    fig, ax = plot_heating_rate(ds, wavelength_nm=550.0)
    assert len(ax.lines) == 1
    plt.close(fig)
