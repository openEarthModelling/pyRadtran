import numpy as np
import pytest

from pyradtran.models.aerosol_composite import (
    IntegrationConfig,
    MieSpecies,
    ParticleOptics,
    RefractiveIndex,
    SizeDistribution,
)


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


class TestSizeDistribution:
    def test_lognormal_normalization(self):
        sd = SizeDistribution(
            kind="lognormal", params={"r_g_um": 0.5, "sigma_g": 2.0}
        )
        r = np.logspace(-2, 2, 10000)
        dn = sd.evaluate(r)
        total = np.trapezoid(dn, r)
        assert np.isclose(total, 1.0, rtol=0.01)

    def test_monodisperse_peak_location(self):
        sd = SizeDistribution(
            kind="monodisperse", params={"radius_um": 1.0}
        )
        r = np.linspace(0.5, 1.5, 1000)
        dn = sd.evaluate(r)
        assert r[np.argmax(dn)] == pytest.approx(1.0, abs=0.02)

    def test_invalid_sigma_g_raises(self):
        with pytest.raises(ValueError):
            SizeDistribution(
                kind="lognormal", params={"r_g_um": 0.5, "sigma_g": 1.0}
            )

    def test_integration_config_defaults(self):
        cfg = IntegrationConfig()
        assert cfg.n_radius_grid == 200
        assert cfg.radius_min_um == 0.001
        assert cfg.radius_max_um == 100.0


from pyradtran.optics.mie import bhmie


class TestBhmie:
    """Tests against published Mie reference values."""

    def test_zero_size_parameter(self):
        result = bhmie(0.0, 1.5 + 0.0j)
        assert result["Qext"] == 0.0
        assert result["Qsca"] == 0.0
        assert result["g"] == 0.0

    def test_rayleigh_limit_small_x(self):
        """For x << 1, Qsca ~ (8/3) * x^4 * |(m^2-1)/(m^2+2)|^2."""
        x = 0.01
        m = 1.5 + 0.0j
        result = bhmie(x, m)
        expected_Qsca = (8.0 / 3.0) * x**4 * abs((m**2 - 1) / (m**2 + 2)) ** 2
        assert result["Qsca"] == pytest.approx(expected_Qsca, rel=0.01)

    def test_geometric_optics_large_x(self):
        """For x >> 1, Qext -> 2 (with ripple-structure tolerance)."""
        x = 100.0
        m = 1.33 + 0.0j
        result = bhmie(x, m)
        assert result["Qext"] == pytest.approx(2.0, abs=0.15)

    def test_standard_sphere_bohren_huffman(self):
        """B&H Table 4.1: x=1, m=1.5+1.0i -> Qext~2.336, Qsca~0.663."""
        result = bhmie(1.0, 1.5 + 1.0j)
        assert result["Qext"] == pytest.approx(2.336, abs=0.05)
        assert result["Qsca"] == pytest.approx(0.663, abs=0.05)
        assert abs(result["g"]) <= 1.0

    def test_absorbing_particle(self):
        """Absorbing particle: Qext > Qsca."""
        result = bhmie(5.0, 1.5 + 0.5j)
        assert result["Qext"] > result["Qsca"]
        assert result["Qsca"] >= 0.0
        assert result["Qext"] >= 0.0

    def test_angular_output_shape(self):
        result = bhmie(5.0, 1.33 + 0.0j, n_angles=37)
        assert "S1" in result
        assert "S2" in result
        assert len(result["S1"]) == 37
        assert len(result["angles_deg"]) == 37


class TestMieSpecies:
    def test_basic_mie_species(self):
        ri = RefractiveIndex(
            wavelength_um=[0.5, 0.55],
            n_real=[1.5, 1.5],
            k_imag=[0.01, 0.01],
        )
        sd = SizeDistribution(
            kind="monodisperse", params={"radius_um": 0.5}
        )
        species = MieSpecies(
            refractive_index=ri,
            size_distribution=sd,
            particle_density_kg_m3=1000.0,
            integration_config=IntegrationConfig(n_radius_grid=50),
        )
        wl = np.array([0.5, 0.55])
        optics = species.intensive(wl)
        assert optics.beta_ext_per_mass.shape == (2,)
        assert optics.ssa.shape == (2,)
        assert optics.g.shape == (2,)
        assert np.all(optics.beta_ext_per_mass >= 0)
        assert np.all((optics.ssa >= 0) & (optics.ssa <= 1))
        assert np.all(np.abs(optics.g) <= 1.0)
        assert optics.legendre_moments is not None
        assert optics.legendre_moments.shape == (2, 32)
        assert np.isclose(optics.legendre_moments[0, 0], 1.0)

    def test_mie_species_n_legendre(self):
        ri = RefractiveIndex(
            wavelength_um=[0.5, 0.55],
            n_real=[1.5, 1.5],
            k_imag=[0.01, 0.01],
        )
        sd = SizeDistribution(
            kind="monodisperse", params={"radius_um": 0.5}
        )
        species = MieSpecies(
            refractive_index=ri,
            size_distribution=sd,
            particle_density_kg_m3=1000.0,
            integration_config=IntegrationConfig(n_radius_grid=50),
        )
        wl = np.array([0.5, 0.55])
        optics = species.intensive(wl, n_legendre=16)
        assert optics.legendre_moments is not None
        assert optics.legendre_moments.shape == (2, 16)

    def test_lognormal_mie_species(self):
        ri = RefractiveIndex(
            wavelength_um=[0.55, 0.6],
            n_real=[1.5, 1.5],
            k_imag=[0.01, 0.01],
        )
        sd = SizeDistribution(
            kind="lognormal", params={"r_g_um": 0.3, "sigma_g": 1.5}
        )
        species = MieSpecies(
            refractive_index=ri,
            size_distribution=sd,
            particle_density_kg_m3=2000.0,
            integration_config=IntegrationConfig(n_radius_grid=50),
        )
        wl = np.array([0.55])
        optics = species.intensive(wl)
        assert optics.beta_ext_per_mass[0] > 0
        assert 0 < optics.ssa[0] <= 1
