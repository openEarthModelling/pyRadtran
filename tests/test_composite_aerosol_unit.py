import numpy as np
import pytest

from pyradtran.models.aerosol_composite import ParticleOptics, RefractiveIndex


class TestRefractiveIndex:
    def test_at_interpolation(self):
        ri = RefractiveIndex(
            wavelength_um=[0.4, 0.55, 0.7],
            n_real=[1.5, 1.45, 1.42],
            k_imag=[0.01, 0.005, 0.003],
        )
        result = ri.at(np.array([0.4, 0.55, 0.7]))
        assert np.allclose(result.real, [1.5, 1.45, 1.42])
        assert np.all(result.imag >= 0)

    def test_at_out_of_range_raises(self):
        ri = RefractiveIndex(
            wavelength_um=[0.4, 0.7],
            n_real=[1.5, 1.42],
            k_imag=[0.01, 0.003],
        )
        with pytest.raises(ValueError):
            ri.at(np.array([0.3]))
        with pytest.raises(ValueError):
            ri.at(np.array([0.8]))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            RefractiveIndex(
                wavelength_um=[0.4, 0.55],
                n_real=[1.5],
                k_imag=[0.01, 0.005],
            )

    def test_negative_k_raises(self):
        with pytest.raises(ValueError):
            RefractiveIndex(
                wavelength_um=[0.4, 0.55],
                n_real=[1.5, 1.45],
                k_imag=[0.01, -0.001],
            )


class TestParticleOptics:
    def test_from_cross_sections(self):
        wl = [0.5, 0.6]
        r = [0.1, 1.0]
        Cext = np.array([[0.0314, 3.14], [0.0452, 4.52]])
        Csca = np.array([[0.0251, 2.51], [0.0362, 3.62]])
        g = np.array([[0.1, 0.7], [0.15, 0.65]])
        po = ParticleOptics.from_cross_sections(
            wavelength_um=wl, radius_um=r, Cext_um2=Cext, Csca_um2=Csca, g=g
        )
        assert po.Qext.shape == (2, 2)
        assert np.isclose(po.Qext[0, 0], 1.0, atol=1e-3)

    def test_qsca_greater_than_qext_raises(self):
        with pytest.raises(ValueError):
            ParticleOptics(
                wavelength_um=[0.5],
                radius_um=[0.1],
                Qext=np.array([[1.0]]),
                Qsca=np.array([[1.1]]),
                g=np.array([[0.0]]),
            )

    def test_g_out_of_range_raises(self):
        with pytest.raises(ValueError):
            ParticleOptics(
                wavelength_um=[0.5],
                radius_um=[0.1],
                Qext=np.array([[1.0]]),
                Qsca=np.array([[0.5]]),
                g=np.array([[1.1]]),
            )
