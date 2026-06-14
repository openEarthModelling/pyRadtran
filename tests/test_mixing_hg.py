"""Test _fill_hg_moments returns g_l (PMOM) convention moments."""

import numpy as np

from pyradtran.optics.mixing import _fill_hg_moments


def test_hg_moments_are_g_to_the_l():
    """HG Legendre moments must be g^l (g_l / PMOM form), not (2l+1)*g^l (k_l)."""
    g = np.array([[0.5]])
    m = _fill_hg_moments(g, n_legendre=5)
    expected = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
    np.testing.assert_allclose(m[0, 0, :], expected)
