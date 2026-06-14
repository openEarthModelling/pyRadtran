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


from pyradtran.models.aerosol_composite import LoadedSpecies


class TestLoadedSpecies:
    def test_evaluate_uniform_layer(self):
        """tau = beta_ext * mass * dz for uniform layer."""
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
        loaded = LoadedSpecies(
            species=species,
            mass_profile_kg_m3=[0.001],  # kg/m^3
            altitude_km=[10.0, 0.0],
        )
        wl = np.array([0.55])
        z = np.array([10.0, 0.0])
        layer = loaded.evaluate(wl, z)
        assert layer.tau.shape == (1, 1)
        assert layer.ssa.shape == (1, 1)
        assert layer.g.shape == (1, 1)
        assert layer.legendre_moments.shape == (1, 1, 32)
        assert layer.tau[0, 0] > 0

    def test_altitude_must_be_descending(self):
        ri = RefractiveIndex(wavelength_um=[0.55, 0.6], n_real=[1.5, 1.5], k_imag=[0.01, 0.01])
        sd = SizeDistribution(kind="monodisperse", params={"radius_um": 0.5})
        mie = MieSpecies(
            refractive_index=ri,
            size_distribution=sd,
            particle_density_kg_m3=1000.0,
        )
        with pytest.raises(ValueError):
            LoadedSpecies(
                species=mie,
                mass_profile_kg_m3=[0.001],
                altitude_km=[0.0, 10.0],
            )


import os
import tempfile

import netCDF4

from pyradtran.models.aerosol_composite import OPACSpecies


class TestOPACSpecies:
    def _make_synthetic_nc(self, path: str, nlyr: int = 5, nmom: int = 32):
        with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
            ds.createDimension("nlyr", nlyr)
            ds.createDimension("nphamat", 1)
            ds.createDimension("nmom+1", nmom)

            dtauc = ds.createVariable("output_dtauc", "f8", ("nlyr",))
            ssalb = ds.createVariable("output_ssalb", "f8", ("nlyr",))
            pmom = ds.createVariable("output_pmom", "f8", ("nlyr", "nphamat", "nmom+1"))

            dtauc[:] = np.linspace(0.01, 0.05, nlyr)
            ssalb[:] = np.full(nlyr, 0.95)
            pmom_data = np.zeros((nlyr, 1, nmom))
            pmom_data[:, 0, 0] = 1.0
            pmom_data[:, 0, 1] = 0.7  # g ≈ pmom[1]/3
            pmom[:] = pmom_data

    def test_read_synthetic_opac(self):
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            path = f.name
        try:
            self._make_synthetic_nc(path)
            species = OPACSpecies(netcdf_path=path)
            wl = np.array([0.55])
            optics = species.intensive(wl)
            assert optics.beta_ext_per_mass[0] > 0
            assert 0 < optics.ssa[0] <= 1
        finally:
            os.unlink(path)


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
    def test_single_loaded_source(self):
        ri = RefractiveIndex(wavelength_um=[0.55, 0.6], n_real=[1.5, 1.5], k_imag=[0.01, 0.01])
        sd = SizeDistribution(kind="monodisperse", params={"radius_um": 0.5})
        mie = MieSpecies(
            refractive_index=ri,
            size_distribution=sd,
            particle_density_kg_m3=1000.0,
            integration_config=IntegrationConfig(n_radius_grid=50),
        )
        loaded = LoadedSpecies(
            species=mie,
            mass_profile_kg_m3=[0.001],
            altitude_km=[10.0, 0.0],
        )
        comp = CompositeAerosol(
            sources=[loaded],
            wavelength_grid_um=[0.55],
            altitude_grid_km=[10.0, 0.0],
            n_legendre=4,
        )
        lines = comp.to_uvspec_lines()
        assert len(lines) == 2
        assert lines[1].startswith("aerosol_file explicit ")

    def test_single_preset_shortcut(self):
        from pyradtran.models.aerosol import OpacPreset, OpacPresetName

        preset = OpacPreset(name=OpacPresetName.CONTINENTAL_AVERAGE)
        comp = CompositeAerosol(
            sources=[preset],
            wavelength_grid_um=[0.55],
            altitude_grid_km=[10.0, 0.0],
        )
        lines = comp.to_uvspec_lines()
        # Single preset should delegate directly, not go through explicit file
        assert len(lines) >= 1
        assert any("aerosol_species" in line for line in lines)

    def test_mixed_sources_raises(self):
        from pyradtran.models.aerosol import OpacPreset, OpacPresetName

        ri = RefractiveIndex(wavelength_um=[0.55, 0.6], n_real=[1.5, 1.5], k_imag=[0.01, 0.01])
        sd = SizeDistribution(kind="monodisperse", params={"radius_um": 0.5})
        mie = MieSpecies(
            refractive_index=ri,
            size_distribution=sd,
            particle_density_kg_m3=1000.0,
        )
        loaded = LoadedSpecies(
            species=mie,
            mass_profile_kg_m3=[0.001],
            altitude_km=[10.0, 0.0],
        )
        preset = OpacPreset(name=OpacPresetName.CONTINENTAL_AVERAGE)

        with pytest.raises(ValueError):
            CompositeAerosol(
                sources=[loaded, preset],
                wavelength_grid_um=[0.55],
                altitude_grid_km=[10.0, 0.0],
            )
