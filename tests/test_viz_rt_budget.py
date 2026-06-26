"""Tests for budget and overview plots (headless, synthetic data)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from pyradtran.core.postprocess import add_budget_vars
from pyradtran.viz.rt import plot_budget, plot_rt_overview


def _flux_ds() -> xr.Dataset:
    wl = np.array([300.0, 500.0, 800.0])
    zout = np.array([0.0, 120.0])
    edir = np.array([[0.5, 1.0], [0.6, 1.0], [0.7, 1.0]])
    edn = np.array([[0.1, 0.0], [0.1, 0.0], [0.1, 0.0]])
    eup = np.array([[0.05, 0.2], [0.06, 0.2], [0.07, 0.2]])
    return xr.Dataset(
        {
            "edir": (("wavelength", "zout"), edir),
            "edn": (("wavelength", "zout"), edn),
            "eup": (("wavelength", "zout"), eup),
        },
        coords={"wavelength": wl, "zout": zout},
    )


def test_plot_budget_requires_enriched_dataset_and_draws_components():
    ds = add_budget_vars(_flux_ds())
    fig, ax = plot_budget(ds)
    # stackplot + explicit legend -> one legend entry per component (3).
    assert len(ax.legend_.get_texts()) == 3
    plt.close(fig)


def test_plot_budget_raises_on_unenriched_dataset():
    with pytest.raises(ValueError):
        plot_budget(_flux_ds())


def test_plot_rt_overview_has_three_panels():
    fig, ax = plot_rt_overview(_flux_ds(), wavelength_nm=500.0)
    # plot_rt_overview returns the figure and a numpy array (or tuple) of 3 axes.
    axes = fig.get_axes()
    assert len(axes) >= 3
    plt.close(fig)
