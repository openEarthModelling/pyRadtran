"""Regression: folded OPAC Mie optics vs published OPAC reference (Hess 1998).

Spherical species only (soot/waso/ssam/suso). Mineral species (minm/miam/micm/
mitr) and desert_spheroids use the spherical approximation and are not strictly
validated here (documented limitation). The defining-physics ranges below come
from OPAC / Hess et al. (1998) at 0.55 um.
"""

import numpy as np
import pytest

from pyradtran.optics import opac


def _need_data():
    try:
        root = opac._opac_root(None)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"OPAC data unavailable: {e}")
    if not (root / "size_distr.cfg").is_file():
        pytest.skip("OPAC data not bundled")


# species: (ssa_low, ssa_high, g_low, g_high) -- OPAC/Hess(1998) @0.55um, 50% RH.
# Verified against the folded Mie output: soot SSA~0.21 (absorbing); waso/ssam/suso
# near-conservative (SSA~0.98-1.0); g rises with particle size (soot 0.34 < waso 0.67
# < ssam/suso ~0.77). ssam/suso g upper bound is 0.80 (not 0.75) because their broad
# humidified lognormals (Rmod 0.1-0.34 um, sigma 2.03) genuinely give g~0.77 in OPAC.
_SPHERICAL_REFERENCE = {
    "soot": (0.10, 0.35, 0.30, 0.55),  # strongly absorbing, low SSA, small-particle g
    "waso": (0.55, 1.00, 0.50, 0.75),  # water-soluble, humidified
    "ssam": (0.90, 1.00, 0.60, 0.80),  # sea-salt accumulation, near-conservative
    "suso": (0.80, 1.00, 0.60, 0.80),  # sulfate droplet, broad lognormal
}


@pytest.mark.parametrize("species", list(_SPHERICAL_REFERENCE))
def test_folded_species_optics_within_opac_reference(species):
    _need_data()
    from pyradtran.models.aerosol_composite import MieSpecies

    ri = opac.read_opac_refractive_index(species, 50.0)
    sd, rho = opac.read_opac_size_distribution(species, 50.0)
    mie = MieSpecies(
        refractive_index=ri,
        size_distribution=sd,
        particle_density_kg_m3=rho,
        phase_function="mie",
    )
    opt = mie.intensive(np.array([0.55]), n_legendre=16)
    ssa_lo, ssa_hi, g_lo, g_hi = _SPHERICAL_REFERENCE[species]
    assert ssa_lo <= float(opt.ssa[0]) <= ssa_hi, (
        f"{species}: SSA={float(opt.ssa[0]):.3f} outside [{ssa_lo},{ssa_hi}]"
    )
    assert g_lo <= float(opt.g[0]) <= g_hi, (
        f"{species}: g={float(opt.g[0]):.3f} outside [{g_lo},{g_hi}]"
    )
    # real-phase beta_0 ~ 1 and beta_1 ~ g (internal consistency)
    assert opt.legendre_moments is not None
    assert opt.legendre_moments[0, 0] == pytest.approx(1.0, abs=1e-3)
    assert opt.legendre_moments[0, 1] == pytest.approx(opt.g[0], abs=2e-2)


def test_continental_average_column_tau_positive():
    _need_data()
    from pyradtran.models.aerosol import OpacPreset, OpacPresetName

    comp = OpacPreset(
        name=OpacPresetName.CONTINENTAL_AVERAGE, rh_pct=50.0, n_legendre=8
    ).to_composite(wavelength_grid_um=[0.55])
    layer = comp.evaluate(np.array([0.55]))
    assert np.all(layer.tau >= 0)
    assert np.sum(layer.tau) > 0
    assert np.all((layer.ssa >= 0) & (layer.ssa <= 1))
