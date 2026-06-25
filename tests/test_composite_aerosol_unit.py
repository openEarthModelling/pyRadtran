import numpy as np
import pytest

from pyradtran.models.aerosol_composite import (
    IntegrationConfig,
    LayerOptics,
    MieSpecies,
    RefractiveIndex,
    SizeDistribution,
)
from pyradtran.optics.mixing import _fill_hg_moments, combine_sources


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


class TestSizeDistribution:
    def test_lognormal_normalization(self):
        sd = SizeDistribution(kind="lognormal", params={"r_g_um": 0.5, "sigma_g": 2.0})
        r = np.logspace(-2, 2, 10000)
        dn = sd.evaluate(r)
        total = np.trapz(dn, r)
        assert np.isclose(total, 1.0, rtol=0.01)

    def test_monodisperse_peak_location(self):
        sd = SizeDistribution(kind="monodisperse", params={"radius_um": 1.0})
        r = np.linspace(0.5, 1.5, 1000)
        dn = sd.evaluate(r)
        assert r[np.argmax(dn)] == pytest.approx(1.0, abs=0.02)

    def test_invalid_sigma_g_raises(self):
        with pytest.raises(ValueError):
            SizeDistribution(kind="lognormal", params={"r_g_um": 0.5, "sigma_g": 1.0})

    def test_integration_config_defaults(self):
        cfg = IntegrationConfig()
        assert cfg.n_radius_grid == 200
        assert cfg.radius_min_um == 0.001
        assert cfg.radius_max_um == 100.0

    def test_modified_gamma_normalization(self):
        sd = SizeDistribution(
            kind="modified_gamma",
            params={"alpha": 2.0, "gamma": 1.0, "r_c_um": 0.5},
        )
        r = np.logspace(-2, 2, 10000)
        dn = sd.evaluate(r)
        total = np.trapz(dn, r)
        assert np.isclose(total, 1.0, rtol=0.01)

    def test_discrete_normalization(self):
        sd = SizeDistribution(
            kind="discrete",
            params={"radius_um": [0.1, 1.0, 10.0], "weights": [1.0, 2.0, 1.0]},
        )
        r = np.logspace(-2, 2, 10000)
        dn = sd.evaluate(r)
        total = np.trapz(dn, r)
        assert np.isclose(total, 1.0, rtol=0.01)

    def test_modified_gamma_invalid_params_raises(self):
        with pytest.raises(ValueError):
            SizeDistribution(kind="modified_gamma", params={"alpha": 2.0})

    def test_discrete_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            SizeDistribution(
                kind="discrete",
                params={"radius_um": [0.1, 1.0], "weights": [1.0]},
            )


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
        """B&H Table 4.2: x=1, m=1.5+1.0i -> Qext~2.336, Qsca~0.663, g~0.192."""
        result = bhmie(1.0, 1.5 + 1.0j)
        assert result["Qext"] == pytest.approx(2.336, abs=0.05)
        assert result["Qsca"] == pytest.approx(0.663, abs=0.05)
        assert result["g"] == pytest.approx(0.192, abs=0.01)

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
        sd = SizeDistribution(kind="monodisperse", params={"radius_um": 0.5})
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
        sd = SizeDistribution(kind="monodisperse", params={"radius_um": 0.5})
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
        sd = SizeDistribution(kind="lognormal", params={"r_g_um": 0.3, "sigma_g": 1.5})
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

    def test_mie_species_real_phase_function(self):
        """phase_function='mie' -> real Legendre moments (beta_0=1, beta_1~g)."""
        ri = RefractiveIndex(
            wavelength_um=[0.55, 0.6],
            n_real=[1.5, 1.5],
            k_imag=[0.01, 0.01],
        )
        sd = SizeDistribution(kind="monodisperse", params={"radius_um": 0.5})
        species = MieSpecies(
            refractive_index=ri,
            size_distribution=sd,
            particle_density_kg_m3=1000.0,
            integration_config=IntegrationConfig(n_radius_grid=20),
            phase_function="mie",
        )
        optics = species.intensive(np.array([0.55]), n_legendre=16)
        assert optics.legendre_moments is not None
        assert optics.legendre_moments.shape == (1, 16)
        assert optics.legendre_moments[0, 0] == pytest.approx(1.0, abs=1e-3)
        assert optics.legendre_moments[0, 1] == pytest.approx(optics.g[0], abs=2e-2)

    def test_mie_species_default_is_hg(self):
        ri = RefractiveIndex(wavelength_um=[0.55, 0.6], n_real=[1.5, 1.5], k_imag=[0.01, 0.01])
        sd = SizeDistribution(kind="monodisperse", params={"radius_um": 0.5})
        species = MieSpecies(
            refractive_index=ri,
            size_distribution=sd,
            particle_density_kg_m3=1000.0,
            integration_config=IntegrationConfig(n_radius_grid=20),
        )
        assert species.phase_function == "hg"


from pyradtran.models.blocks import MassProfile, PlacedBlock


class TestPlacedBlock:
    def test_to_layer_optics_uniform_layer(self):
        """tau = beta_ext * mass * dz for a uniform layer."""
        ri = RefractiveIndex(
            wavelength_um=[0.55, 0.6],
            n_real=[1.5, 1.5],
            k_imag=[0.01, 0.01],
        )
        sd = SizeDistribution(kind="monodisperse", params={"radius_um": 0.5})
        species = MieSpecies(
            refractive_index=ri,
            size_distribution=sd,
            particle_density_kg_m3=1000.0,
            integration_config=IntegrationConfig(n_radius_grid=50),
        )
        placed = PlacedBlock(
            block=species,
            profile=MassProfile(kg_m3_per_layer=(0.001,)),  # kg/m^3
        )
        wl = np.array([0.55])
        z = np.array([10.0, 0.0])
        layer = placed.to_layer_optics(wl, z)
        assert layer.tau.shape == (1, 1)
        assert layer.ssa.shape == (1, 1)
        assert layer.g.shape == (1, 1)
        assert layer.legendre_moments.shape == (1, 1, 32)
        assert layer.tau[0, 0] > 0


class TestMixing:
    def test_two_source_mixing(self):
        """Mix two equal sources: tau doubles, ssa/g unchanged."""
        n_wl, n_layer, n_leg = 2, 3, 4
        tau = np.ones((n_wl, n_layer)) * 0.1
        ssa = np.ones((n_wl, n_layer)) * 0.9
        g = np.ones((n_wl, n_layer)) * 0.7
        moments = np.ones((n_wl, n_layer, n_leg))
        moments[:, :, 0] = 1.0

        src1 = LayerOptics(tau=tau, ssa=ssa, g=g, legendre_moments=moments)
        src2 = LayerOptics(tau=tau, ssa=ssa, g=g, legendre_moments=moments)

        result = combine_sources([src1, src2], n_legendre=n_leg)
        assert np.allclose(result["tau"], 0.2)
        assert np.allclose(result["ssa"], 0.9)
        assert np.allclose(result["g"], 0.7)

    def test_hg_moment_fill(self):
        g = np.full((1, 1), 0.7)
        moments = _fill_hg_moments(g, n_legendre=4)
        assert moments[0, 0, 0] == 1.0
        assert moments[0, 0, 1] == pytest.approx(0.7)
        assert moments[0, 0, 2] == pytest.approx(0.7**2)
        assert moments[0, 0, 3] == pytest.approx(0.7**3)

    def test_zero_tau_layer(self):
        """Layer with zero tau should have ssa=0, g=0."""
        tau = np.zeros((1, 1))
        ssa = np.ones((1, 1)) * 0.9
        g = np.ones((1, 1)) * 0.7
        moments = np.ones((1, 1, 4))
        src = LayerOptics(tau=tau, ssa=ssa, g=g, legendre_moments=moments)
        result = combine_sources([src], n_legendre=4)
        assert result["ssa"][0, 0] == 0.0
        assert result["g"][0, 0] == 0.0


from pyradtran.models.aerosol_composite import CompositeAerosol


class TestCompositeAerosol:
    def _mie_placed(self):
        ri = RefractiveIndex(wavelength_um=[0.55, 0.6], n_real=[1.5, 1.5], k_imag=[0.01, 0.01])
        mie = MieSpecies(
            refractive_index=ri,
            size_distribution=SizeDistribution(kind="monodisperse", params={"radius_um": 0.5}),
            particle_density_kg_m3=1000.0,
            integration_config=IntegrationConfig(n_radius_grid=50),
        )
        return PlacedBlock(block=mie, profile=MassProfile(kg_m3_per_layer=(0.001,)))

    def test_single_piece_explicit_path(self):
        comp = CompositeAerosol(
            pieces=[self._mie_placed()],
            wavelength_grid_um=[0.55],
            altitude_grid_km=[10.0, 0.0],
            n_legendre=4,
        )
        lines = comp.to_uvspec_lines()
        assert lines[0] == "aerosol_default"
        assert lines[1].startswith("aerosol_file explicit ")

    def test_any_mix_single_path(self):
        """Two PlacedBlocks mix through the single explicit-file path."""
        comp = CompositeAerosol(
            pieces=[self._mie_placed(), self._mie_placed()],
            wavelength_grid_um=[0.55],
            altitude_grid_km=[10.0, 0.0],
            n_legendre=4,
        )
        lines = comp.to_uvspec_lines()
        assert lines[1].startswith("aerosol_file explicit ")

    def test_evaluate_doubles_tau_for_two_equal_pieces(self):
        import tempfile
        from pathlib import Path

        comp = CompositeAerosol(
            pieces=[self._mie_placed(), self._mie_placed()],
            wavelength_grid_um=[0.55],
            altitude_grid_km=[10.0, 0.0],
            n_legendre=4,
            output_dir=Path(tempfile.mkdtemp()),
        )
        wl = np.array([0.55])
        z = np.array([10.0, 0.0])
        mixed = comp.evaluate(wl, z, n_legendre=4)
        single = self._mie_placed().to_layer_optics(wl, z, n_legendre=4)
        assert np.isclose(mixed.tau[0, 0], 2.0 * single.tau[0, 0], rtol=1e-9)

    def test_rejects_ascending_altitude(self):
        with pytest.raises(ValueError):
            CompositeAerosol(
                pieces=[self._mie_placed()],
                wavelength_grid_um=[0.55],
                altitude_grid_km=[0.0, 10.0],
            )

    def test_opacpreset_is_not_a_piece(self):
        """OpacPreset stays a standalone AerosolModel; it is not a Piece and must be rejected."""
        from pyradtran.models.aerosol import OpacPreset, OpacPresetName

        with pytest.raises(ValueError):
            CompositeAerosol(
                pieces=[OpacPreset(name=OpacPresetName.CONTINENTAL_AVERAGE)],
                wavelength_grid_um=[0.55],
                altitude_grid_km=[10.0, 0.0],
            )


class TestSpeciesBlockFields:
    """Task 1: species blocks expose ``name`` and ``mass_per_particle_kg``."""

    def test_mie_species_mass_per_particle(self):
        ri = RefractiveIndex(wavelength_um=[0.55, 0.6], n_real=[1.5, 1.5], k_imag=[0.01, 0.01])
        sd = SizeDistribution(kind="monodisperse", params={"radius_um": 0.5})
        species = MieSpecies(
            refractive_index=ri, size_distribution=sd, particle_density_kg_m3=1000.0
        )
        # monodisperse r=0.5um, rho=1000 -> mass = rho*(4/3)*pi*r^3
        expected = 1000.0 * (4.0 / 3.0) * np.pi * (0.5e-6) ** 3
        assert species.mass_per_particle_kg == pytest.approx(expected, rel=0.1)
        assert species.name == "MieSpecies"

    def test_bulk_species_name_and_mass(self):
        from pyradtran.models.aerosol_composite import BulkSpecies

        class _StubSD:
            @staticmethod
            def moment(order):
                return 100.0**order  # nm^order; moment(3) = 1e6 nm^3

        class _StubBulk:
            wavelength_nm = np.array([550.0])
            effective_density_kg_m3 = 1800.0
            size_distribution = _StubSD()

        bs = BulkSpecies(bulk=_StubBulk())
        assert bs.name == "BulkSpecies"
        expected = 1800.0 * (4.0 / 3.0) * np.pi * (100.0**3) * 1e-27
        assert bs.mass_per_particle_kg == pytest.approx(expected, rel=1e-9)
