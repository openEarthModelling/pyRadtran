"""Unit tests for forcing plots: DRF spectral + spectral attribution."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import xarray as xr

from pyradtran.workflow import AttributionResult


def test_plot_drf_spectral_draws_two_lines_and_fill():
    from pyradtran.viz.forcing import plot_drf_spectral

    wl = np.array([400.0, 550.0, 700.0])
    drf_toa = np.array([-1.0, -1.5, -0.8])
    drf_surf = np.array([-2.0, -3.0, -1.5])
    fig, ax = plot_drf_spectral(wl, drf_toa, drf_surf)
    assert len(ax.lines) >= 2
    assert len(ax.collections) >= 1  # the fill_between band
    assert "Wavelength" in ax.get_xlabel()


def test_plot_drf_spectral_zero_line_drawn():
    from pyradtran.viz.forcing import plot_drf_spectral

    wl = np.array([400.0, 700.0])
    fig, ax = plot_drf_spectral(wl, np.array([-1.0, -0.5]), np.array([-2.0, -1.0]))
    ydata = [list(l.get_ydata()) for l in ax.lines]
    assert [0.0, 0.0] in ydata  # zero reference line


def _attribution_result_spectral():
    wl = np.array([400.0, 550.0, 700.0])
    zout = [0.0, 120.0]
    full = xr.Dataset(
        {"edir": (("wavelength", "zout"), [[500.0, 800.0], [450.0, 800.0], [480.0, 800.0]])},
        coords={"wavelength": wl, "zout": zout},
    )
    contributions = {
        "bc": full - xr.Dataset(
            {"edir": (("wavelength", "zout"), [[510.0, 800.0]] * 3)},
            coords={"wavelength": wl, "zout": zout},
        ),
        "dust": full - xr.Dataset(
            {"edir": (("wavelength", "zout"), [[520.0, 800.0]] * 3)},
            coords={"wavelength": wl, "zout": zout},
        ),
    }
    return AttributionResult(full=full, contributions=contributions)


def test_plot_spectral_attribution_draws_one_stack_per_block():
    from pyradtran.viz.forcing import plot_spectral_attribution

    result = _attribution_result_spectral()
    fig, ax = plot_spectral_attribution(result, variable="edir", level="surface")
    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "bc" in legend_texts and "dust" in legend_texts
