import tempfile
from pathlib import Path

import numpy as np
import pytest

from pyradtran.models.aerosol_composite import (
    CompositeAerosol,
    IntegrationConfig,
    MieSpecies,
    RefractiveIndex,
    SizeDistribution,
)
from pyradtran.models.blocks import MassProfile, PlacedBlock
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
        loaded = PlacedBlock(
            block=mie,
            profile=MassProfile(kg_m3_per_layer=(0.001,)),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            comp = CompositeAerosol(
                pieces=[loaded],
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
