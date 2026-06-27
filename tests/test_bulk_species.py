"""Tests for BulkSpecies (wraps aerosol3D BulkAerosolOpticsData)."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("Aerosol3D")  # needs the aerosol3d sibling repo (absent in CI)
from Aerosol3D.bulk.datastructs import SizeDistribution

from pyradtran.models.aerosol_composite import BulkSpecies


def _bulk(g=0.5, density=1800.0):
    n_wl, n_l = 3, 8
    l_vals = np.arange(n_l)
    beta = (2 * l_vals + 1) * g**l_vals  # k_l form
    legendre_moments_beta = g**l_vals  # g_l form
    sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.4)
    return SimpleNamespace(
        wavelength_nm=np.linspace(400.0, 600.0, n_wl),
        C_ext=np.full(n_wl, 10.0),  # nm^2/particle
        C_sca=np.full(n_wl, 9.0),
        SSA=np.full(n_wl, 0.9),
        g=np.full(n_wl, g),
        beta=np.tile(beta, (n_wl, 1)),
        legendre_moments_beta=np.tile(legendre_moments_beta, (n_wl, 1)),
        n_legendre=n_l,
        size_distribution=sd,
        effective_density_kg_m3=density,
    )


def test_bulk_species_intensive_units_and_moments():
    sp = BulkSpecies(bulk=_bulk(g=0.5, density=1800.0))
    wl_um = np.array([0.45, 0.55])
    out = sp.intensive(wl_um, n_legendre=8)
    # mass per particle = rho * (4/3) pi <r^3>; <r^3> = moment(3) in nm^3
    r3 = 100.0**3 * np.exp(0.5 * (3 * 0.4) ** 2)  # lognormal moment(3) analytic
    vol_m3 = (4.0 / 3.0) * np.pi * r3 * 1e-27
    mass = 1800.0 * vol_m3
    expected_beta = 10.0 * 1e-6 * 1e-12 / mass  # nm^2 -> um^2 -> m^2 / kg
    np.testing.assert_allclose(out.beta_ext_per_mass, expected_beta, rtol=1e-9)
    np.testing.assert_allclose(out.ssa, 0.9, rtol=1e-9)
    np.testing.assert_allclose(out.g, 0.5, rtol=1e-9)
    # g_l form selected (hypothesis): moment[1] == g
    assert out.legendre_moments.shape == (2, 8)
    np.testing.assert_allclose(out.legendre_moments[:, 1], 0.5, rtol=1e-9)


def test_bulk_species_n_legendre_truncation():
    sp = BulkSpecies(bulk=_bulk(g=0.5))
    out = sp.intensive(np.array([0.5]), n_legendre=4)
    assert out.legendre_moments.shape == (1, 4)
