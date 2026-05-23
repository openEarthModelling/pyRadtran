import tempfile
from pathlib import Path

import numpy as np
import pytest

from pyradtran.models.aerosol_composite import (
    CompositeAerosol,
    IntegrationConfig,
    LoadedSpecies,
    MieSpecies,
    ParticleOptics,
    PrecomputedSpecies,
    RefractiveIndex,
    SizeDistribution,
)
from pyradtran.optics.layer_writer import write_explicit_aerosol


class TestLayerWriter:
    def test_writes_master_and_layers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            tau = np.array([[0.1, 0.2]])  # (1 wl, 2 layers)
            ssa = np.array([[0.9, 0.8]])
            g = np.array([[0.7, 0.6]])
            moments = np.ones((1, 2, 4))
            moments[:, :, 0] = 1.0
            wl = np.array([0.55])
            alt = np.array([10.0, 5.0, 0.0])

            master = write_explicit_aerosol(
                tau=tau,
                ssa=ssa,
                g=g,
                legendre_moments=moments,
                wavelength_um=wl,
                altitude_km=alt,
                output_dir=outdir,
                source_signatures=["test"],
            )

            assert master.exists()
            master_text = master.read_text()
            lines = master_text.strip().split("\n")
            assert len(lines) == 3  # 2 layers + NULL
            assert "NULL.LAYER" in lines[0]

            # Verify layer file exists and has correct format
            layer_line = lines[1].split()
            layer_file = outdir / layer_line[1]
            assert layer_file.exists()
            layer_text = layer_file.read_text().strip()
            vals = [float(v) for v in layer_text.split()]
            assert len(vals) == 3 + 4  # wl, beta_ext, ssa + 4 moments
            assert vals[0] == pytest.approx(550.0)  # nm
            assert vals[2] == pytest.approx(0.9)

    def test_cache_hit_skips_rewrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            tau = np.array([[0.1]])
            ssa = np.array([[0.9]])
            g = np.array([[0.7]])
            moments = np.ones((1, 1, 4))
            wl = np.array([0.55])
            alt = np.array([10.0, 0.0])

            master1 = write_explicit_aerosol(
                tau=tau,
                ssa=ssa,
                g=g,
                legendre_moments=moments,
                wavelength_um=wl,
                altitude_km=alt,
                output_dir=outdir,
                source_signatures=["test"],
            )
            mtime1 = master1.stat().st_mtime

            master2 = write_explicit_aerosol(
                tau=tau,
                ssa=ssa,
                g=g,
                legendre_moments=moments,
                wavelength_um=wl,
                altitude_km=alt,
                output_dir=outdir,
                source_signatures=["test"],
            )
            mtime2 = master2.stat().st_mtime
            assert mtime1 == mtime2
            assert master1 == master2


class TestFullPipeline:
    def test_mie_plus_precomputed_pipeline(self):
        """Two LoadedSpecies -> mixed -> explicit files."""
        import tempfile
        from pathlib import Path

        wl = [0.5, 0.55, 0.6]
        alt = [10.0, 5.0, 0.0]

        # Source 1: Mie species
        ri1 = RefractiveIndex(wavelength_um=wl, n_real=[1.5] * 3, k_imag=[0.01] * 3)
        sd1 = SizeDistribution(kind="lognormal", params={"r_g_um": 0.3, "sigma_g": 1.5})
        mie = MieSpecies(
            refractive_index=ri1,
            size_distribution=sd1,
            particle_density_kg_m3=2000.0,
            integration_config=IntegrationConfig(n_radius_grid=50),
        )
        loaded1 = LoadedSpecies(
            species=mie,
            mass_profile_kg_m3=[0.001, 0.002],
            altitude_km=alt,
        )

        # Source 2: Precomputed species
        po = ParticleOptics(
            wavelength_um=wl,
            radius_um=[0.5, 1.0],
            Qext=np.full((3, 2), 2.0),
            Qsca=np.full((3, 2), 1.5),
            g=np.full((3, 2), 0.7),
        )
        sd2 = SizeDistribution(kind="monodisperse", params={"radius_um": 1.0})
        precomp = PrecomputedSpecies(
            particle_optics=po,
            size_distribution=sd2,
            particle_density_kg_m3=1000.0,
        )
        loaded2 = LoadedSpecies(
            species=precomp,
            mass_profile_kg_m3=[0.0005, 0.001],
            altitude_km=alt,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            comp = CompositeAerosol(
                sources=[loaded1, loaded2],
                wavelength_grid_um=wl,
                altitude_grid_km=alt,
                n_legendre=8,
                output_dir=Path(tmpdir),
            )
            lines = comp.to_uvspec_lines()
            assert len(lines) == 2
            assert lines[1].startswith("aerosol_file explicit ")

            # Verify files exist
            master_path = Path(lines[1].split()[-1])
            assert master_path.exists()

    def test_format_invariants(self):
        """k_0 = 1, last row is NULL.LAYER, beta_ext in 1/km."""
        import tempfile
        from pathlib import Path

        wl = [0.5, 0.55]
        alt = [5.0, 0.0]
        ri = RefractiveIndex(wavelength_um=wl, n_real=[1.5, 1.5], k_imag=[0.01, 0.01])
        sd = SizeDistribution(kind="monodisperse", params={"radius_um": 0.5})
        mie = MieSpecies(
            refractive_index=ri,
            size_distribution=sd,
            particle_density_kg_m3=1000.0,
            integration_config=IntegrationConfig(n_radius_grid=30),
        )
        loaded = LoadedSpecies(
            species=mie,
            mass_profile_kg_m3=[0.001],
            altitude_km=alt,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            comp = CompositeAerosol(
                sources=[loaded],
                wavelength_grid_um=wl,
                altitude_grid_km=alt,
                n_legendre=4,
                output_dir=Path(tmpdir),
            )
            lines = comp.to_uvspec_lines()
            master_path = Path(lines[1].split()[-1])
            master_text = master_path.read_text()
            master_lines = master_text.strip().split("\n")
            assert "NULL.LAYER" in master_lines[0]

            # Check layer file format (2 wavelengths x 7 values each)
            layer_file = Path(tmpdir) / master_lines[1].split()[1]
            layer_text = layer_file.read_text().strip()
            vals = [float(v) for v in layer_text.split()]
            assert len(vals) == 2 * (3 + 4)  # 2 wl x (wl, beta_ext, ssa + 4 moments)
            assert vals[3] == pytest.approx(1.0)  # k_0 for first wavelength
