"""Tests for phase_function_to_legendre (real Mie phase function -> PMOM moments)."""

import numpy as np
import pytest

from pyradtran.optics.mie import bhmie, phase_function_to_legendre


def test_isotropic_input_is_flat():
    """Constant S1/S2 -> phase function constant -> beta_0=1, beta_l>0=0."""
    n = 181
    angles = np.linspace(0.0, 180.0, n)
    s = np.ones(n, dtype=complex)
    beta = phase_function_to_legendre(s, s, angles, n_legendre=8)
    assert beta[0] == pytest.approx(1.0, abs=1e-6)
    # Isotropic moments vanish by orthogonality; the residual is the trapezoidal
    # discretisation error of integral P_l over the finite angle grid (~1e-4).
    assert np.allclose(beta[1:], 0.0, atol=1e-3)


def test_normalisation_beta0_is_one():
    res = bhmie(5.0, 1.33 + 0.0j, n_angles=181)
    beta = phase_function_to_legendre(res["S1"], res["S2"], res["angles_deg"], n_legendre=16)
    assert beta[0] == pytest.approx(1.0, abs=1e-3)


def test_beta1_matches_bhmie_g():
    """The l=1 moment must recover the asymmetry parameter bhmie reports."""
    res = bhmie(5.0, 1.5 + 0.01j, n_angles=361)
    beta = phase_function_to_legendre(res["S1"], res["S2"], res["angles_deg"], n_legendre=8)
    assert beta[1] == pytest.approx(res["g"], abs=2e-2)


def test_rayleigh_limit_near_isotropic():
    res = bhmie(0.01, 1.5 + 0.0j, n_angles=181)
    beta = phase_function_to_legendre(res["S1"], res["S2"], res["angles_deg"], n_legendre=4)
    assert beta[0] == pytest.approx(1.0, abs=1e-3)
    assert abs(beta[1]) < 1e-2


def test_forward_peaked_large_x_has_positive_g():
    res = bhmie(50.0, 1.33 + 0.0j, n_angles=361)
    beta = phase_function_to_legendre(res["S1"], res["S2"], res["angles_deg"], n_legendre=8)
    assert beta[1] > 0.5
    assert beta[1] == pytest.approx(res["g"], abs=2e-2)
