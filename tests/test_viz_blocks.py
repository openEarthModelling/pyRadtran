"""Unit tests for per-block spectral optics plot + evaluate_blocks_on_grid ssa/g."""

import matplotlib

matplotlib.use("Agg")

import numpy as np

from pyradtran.core.postprocess import evaluate_blocks_on_grid
from pyradtran.models.aerosol_composite import (
    CompositeAerosol,
    IntegrationConfig,
    MieSpecies,
    RefractiveIndex,
    SizeDistribution,
)
from pyradtran.models.blocks import MassProfile, PlacedBlock

WL = [0.40, 0.55, 0.70]
ALT = [2.0, 0.0]


def _composite():
    ri = RefractiveIndex(
        wavelength_um=WL, n_real=[1.95, 1.95, 1.95], k_imag=[0.79, 0.79, 0.79]
    )
    sd = SizeDistribution(kind="lognormal", params={"r_g_um": 0.1, "sigma_g": 2.0})
    sp = MieSpecies(
        refractive_index=ri,
        size_distribution=sd,
        particle_density_kg_m3=1800.0,
        integration_config=IntegrationConfig(),
        name="bc",
    )
    return CompositeAerosol(
        pieces=[PlacedBlock(block=sp, profile=MassProfile(kg_m3_per_layer=(1e-7,)))],
        wavelength_grid_um=WL,
        altitude_grid_km=ALT,
        n_legendre=4,
        output_dir=".",
    )


def test_evaluate_blocks_on_grid_now_has_ssa_and_g():
    blocks = evaluate_blocks_on_grid(_composite(), WL, ALT, n_legendre=4)
    ds = blocks["bc"]
    assert "ssa" in ds.data_vars
    assert "g" in ds.data_vars
    assert ds["ssa"].shape == (3, 1)  # (wavelength, layer)


def test_plot_block_spectral_optics_tau_one_line_per_block():
    from pyradtran.viz.blocks import plot_block_spectral_optics

    blocks = evaluate_blocks_on_grid(_composite(), WL, ALT, n_legendre=4)
    fig, ax = plot_block_spectral_optics(blocks, quantity="tau")
    assert len(ax.lines) == 1
    assert "Wavelength" in ax.get_xlabel()


def test_plot_block_spectral_optics_ssa_in_unit_interval():
    from pyradtran.viz.blocks import plot_block_spectral_optics

    blocks = evaluate_blocks_on_grid(_composite(), WL, ALT, n_legendre=4)
    fig, ax = plot_block_spectral_optics(blocks, quantity="ssa")
    line = ax.lines[0]
    y = line.get_ydata()
    assert np.all(y >= 0.0) and np.all(y <= 1.0)


def test_plot_block_spectral_optics_bad_quantity_raises():
    import pytest

    from pyradtran.viz.blocks import plot_block_spectral_optics

    blocks = evaluate_blocks_on_grid(_composite(), WL, ALT, n_legendre=4)
    with pytest.raises(ValueError):
        plot_block_spectral_optics(blocks, quantity="nope")
