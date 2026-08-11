"""Unit tests for input-diagnostic viz (size dist, phase function, Legendre decay)."""

import matplotlib

matplotlib.use("Agg")

from pyradtran.models.aerosol_composite import (
    CompositeAerosol,
    IntegrationConfig,
    MieSpecies,
    RefractiveIndex,
    SizeDistribution,
)
from pyradtran.models.blocks import MassProfile, PlacedBlock


def _blocks():
    ri = RefractiveIndex(wavelength_um=[0.40, 0.70], n_real=[1.95, 1.95], k_imag=[0.79, 0.79])
    sd = SizeDistribution(kind="lognormal", params={"r_g_um": 0.1, "sigma_g": 2.0})
    return {"bc": sd}, {"bc": (ri, sd)}


def test_plot_size_distributions_draws_one_line_per_block():
    from pyradtran.viz.inputs import plot_size_distributions

    blocks, _ = _blocks()
    fig, ax = plot_size_distributions(blocks)
    assert len(ax.lines) == 1
    assert ax.get_xscale() == "log"
    assert "Radius" in ax.get_xlabel()


def test_plot_phase_functions_draws_one_line_per_block_on_log_y():
    from pyradtran.viz.inputs import plot_phase_functions

    _, ri_sd = _blocks()
    fig, ax = plot_phase_functions(ri_sd, wavelength_um=0.55)
    assert len(ax.lines) == 1
    assert ax.get_yscale() == "log"
    assert "Scattering angle" in ax.get_xlabel()


def test_plot_phase_functions_raises_on_discrete_unsupported():
    """Discrete SD has no single characteristic radius — raise clearly."""
    import pytest

    from pyradtran.viz.inputs import plot_phase_functions

    ri = RefractiveIndex(wavelength_um=[0.40, 0.70], n_real=[1.5, 1.5], k_imag=[0.0, 0.0])
    sd = SizeDistribution(
        kind="discrete", params={"radius_um": [0.1, 1.0], "weights": [1.0, 1.0]}
    )
    with pytest.raises(NotImplementedError):
        plot_phase_functions({"d": (ri, sd)})


def test_plot_legendre_decay_draws_stem():
    from pyradtran.viz.inputs import plot_legendre_decay

    ri = RefractiveIndex(wavelength_um=[0.40, 0.70], n_real=[1.5, 1.5], k_imag=[0.01, 0.01])
    sd = SizeDistribution(kind="lognormal", params={"r_g_um": 0.5, "sigma_g": 2.0})
    sp = MieSpecies(
        refractive_index=ri,
        size_distribution=sd,
        particle_density_kg_m3=1000.0,
        integration_config=IntegrationConfig(),
        name="x",
    )
    comp = CompositeAerosol(
        pieces=[PlacedBlock(block=sp, profile=MassProfile(kg_m3_per_layer=(1e-7,)))],
        wavelength_grid_um=[0.55],
        altitude_grid_km=[2.0, 0.0],
        n_legendre=8,
        output_dir=".",
    )
    fig, ax = plot_legendre_decay(comp, wavelength_um=0.55, n_legendre=8)
    assert "moment" in ax.get_xlabel().lower() or "l" in ax.get_xlabel().lower()
