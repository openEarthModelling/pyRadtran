"""B1: multi-block external-mixture column tau must equal sum of target taus.

Validates od_to_mass_profile (per-block tau->mass inversion) + composite
stacking: 3 Mie blocks with target tau@550 = {0.15, 0.15, 0.20} must yield a
composite column tau@550 ~= 0.50. No libRadtran — pure analytic Mie mixing.
"""

import numpy as np

from pyradtran.core.postprocess import evaluate_blocks_on_grid, evaluate_composite_on_grid
from pyradtran.models.aerosol_composite import (
    CompositeAerosol,
    IntegrationConfig,
    MieSpecies,
    RefractiveIndex,
    SizeDistribution,
)
from pyradtran.models.blocks import PlacedBlock, od_to_mass_profile

WL = [0.40, 0.55, 0.70]
ALT = [10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.0]
_TARGETS = [
    {"name": "bc",      "n": 1.95, "k": 0.79,  "r_g": 0.10, "sigma": 2.0, "rho": 1800.0, "H": 1.5, "tau": 0.15},
    {"name": "sulfate", "n": 1.53, "k": 0.0,   "r_g": 0.15, "sigma": 1.7, "rho": 1770.0, "H": 2.0, "tau": 0.15},
    {"name": "dust",    "n": 1.53, "k": 0.008, "r_g": 0.50, "sigma": 2.2, "rho": 2600.0, "H": 3.0, "tau": 0.20},
]


def _build_composite():
    pieces = []
    for b in _TARGETS:
        ri = RefractiveIndex(wavelength_um=WL, n_real=[b["n"]] * 3, k_imag=[b["k"]] * 3)
        sd = SizeDistribution(kind="lognormal", params={"r_g_um": b["r_g"], "sigma_g": b["sigma"]})
        species = MieSpecies(
            refractive_index=ri,
            size_distribution=sd,
            particle_density_kg_m3=b["rho"],
            integration_config=IntegrationConfig(),
            name=b["name"],
        )
        profile = od_to_mass_profile(
            species, tau_ref=b["tau"], ref_nm=550.0, altitude_km=ALT, scale_height_km=b["H"]
        )
        pieces.append(PlacedBlock(block=species, profile=profile))
    return CompositeAerosol(
        pieces=pieces, wavelength_grid_um=WL, altitude_grid_km=ALT, n_legendre=4, output_dir="."
    )


def test_column_tau_550_equals_sum_of_targets():
    comp = _build_composite()
    grid = evaluate_composite_on_grid(comp, WL, ALT, n_legendre=4)
    col_tau_550 = float(np.sum(grid["tau"].sel(wavelength=0.55).values))
    expected = sum(b["tau"] for b in _TARGETS)
    assert np.isclose(col_tau_550, expected, rtol=0.01), (
        f"column tau@550={col_tau_550:.4f}, expected sum={expected:.4f} (1% tol)"
    )


def test_per_block_column_tau_550_matches_own_target():
    """Each block alone also inverts to its own target (single-block sanity)."""
    comp = _build_composite()
    blocks = evaluate_blocks_on_grid(comp, WL, ALT, n_legendre=4)
    for b in _TARGETS:
        col = float(np.sum(blocks[b["name"]]["tau"].sel(wavelength=0.55).values))
        assert np.isclose(col, b["tau"], rtol=0.01), (
            f"{b['name']}: column tau@550={col:.4f}, target={b['tau']:.4f}"
        )
