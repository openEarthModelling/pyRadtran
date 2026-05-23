"""End-to-end tests requiring libRadtran/uvspec.

These are marked slow and skipped when uvspec is absent.
"""

import tempfile
from pathlib import Path

import pytest

from pyradtran.models.aerosol import OpacPreset, OpacPresetName
from pyradtran.models.aerosol_composite import (
    CompositeAerosol,
    IntegrationConfig,
    LoadedSpecies,
    MieSpecies,
    RefractiveIndex,
    SizeDistribution,
)
from pyradtran.scene import Scene

pytestmark = pytest.mark.slow


class TestCompositeE2E:
    def test_composite_builds_valid_input(self):
        """CompositeAerosol produces valid uvspec input with explicit file."""
        wl = [0.55, 0.60]
        alt = [100.0, 0.0]
        ri = RefractiveIndex(wavelength_um=wl, n_real=[1.53, 1.53], k_imag=[0.008, 0.008])
        sd = SizeDistribution(kind="lognormal", params={"r_g_um": 0.5, "sigma_g": 2.0})
        mie = MieSpecies(
            refractive_index=ri,
            size_distribution=sd,
            particle_density_kg_m3=2500.0,
            integration_config=IntegrationConfig(n_radius_grid=30),
        )
        loaded = LoadedSpecies(
            species=mie,
            mass_profile_kg_m3=[1e-6],  # very small perturbation
            altitude_km=alt,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            comp = CompositeAerosol(
                sources=[loaded],
                wavelength_grid_um=wl,
                altitude_grid_km=alt,
                n_legendre=16,
                output_dir=Path(tmpdir),
            )
            scene = (
                Scene()
                .set_atmosphere(profile="us", altitude=0.0)
                .set_source_solar(sza=30.0)
                .set_wavelength(550.0, 550.0)
                .set_solver(method="disort", streams=16)
                .set_output(quantities=["lambda", "edir"])
                .set_aerosol(comp)
            )

            input_text = scene.build_input()
            assert "aerosol_file explicit" in input_text
            assert "aerosol_file explicit" in input_text

    def test_preset_baseline_builds_valid_input(self):
        """Baseline OpacPreset still works."""
        scene = (
            Scene()
            .set_atmosphere(profile="us", altitude=0.0)
            .set_source_solar(sza=30.0)
            .set_wavelength(550.0, 550.0)
            .set_solver(method="disort", streams=16)
            .set_output(quantities=["lambda", "edir"])
            .set_aerosol(OpacPreset(name=OpacPresetName.CONTINENTAL_AVERAGE))
        )
        input_text = scene.build_input()
        assert "aerosol_species_file" in input_text
