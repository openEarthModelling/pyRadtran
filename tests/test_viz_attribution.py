"""Tests for the component-attribution plot (headless, synthetic data)."""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from pyradtran.viz.attribution import plot_component_attribution


@dataclass(frozen=True)
class _Result:
    full: xr.Dataset
    contributions: dict


def _ds(value: float) -> xr.Dataset:
    return xr.Dataset(
        {"edir": ("wavelength", np.array([value]))},
        coords={"wavelength": np.array([500.0])},
    )


def test_plot_component_attribution_draws_full_plus_each_block():
    result = _Result(
        full=_ds(1.0),
        contributions={"soot": _ds(0.6), "dust": _ds(0.4)},
    )
    fig, ax = plot_component_attribution(result, variable="edir")
    # 1 full line + 2 contribution lines = 3 lines.
    assert len(ax.lines) == 3
    plt.close(fig)


def test_plot_component_attribution_works_with_plain_dict_duck_type():
    result = {"full": _ds(1.0), "contributions": {"a": _ds(1.0)}}
    fig, ax = plot_component_attribution(result, variable="edir")
    assert len(ax.lines) == 2
    plt.close(fig)
