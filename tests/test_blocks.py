"""Tests for pyradtran.models.blocks (Task 2: profiles + AerosolBlock protocol)."""

import numpy as np
import pytest

from pyradtran.models.aerosol_composite import (
    IntegrationConfig,
    MieSpecies,
    RefractiveIndex,
    SizeDistribution,
)
from pyradtran.models.blocks import (
    AerosolBlock,
    ExponentialProfile,
    MassProfile,
    VerticalProfile,
    od_to_mass_profile,
)


def _mie_block() -> MieSpecies:
    ri = RefractiveIndex(wavelength_um=[0.55, 0.6], n_real=[1.5, 1.5], k_imag=[0.01, 0.01])
    sd = SizeDistribution(kind="monodisperse", params={"radius_um": 0.5})
    return MieSpecies(
        refractive_index=ri,
        size_distribution=sd,
        particle_density_kg_m3=1000.0,
        integration_config=IntegrationConfig(n_radius_grid=50),
    )


class TestMassProfile:
    def test_evaluate_returns_input(self):
        p = MassProfile(kg_m3_per_layer=(1e-9, 2e-9, 3e-9))
        out = p.evaluate([10.0, 5.0, 0.0])
        assert out.shape == (3,)
        assert np.allclose(out, [1e-9, 2e-9, 3e-9])
        assert isinstance(p, VerticalProfile)


class TestExponentialProfile:
    def test_exponential_decay(self):
        p = ExponentialProfile(rho0_kg_m3=1e-8, scale_height_km=2.0)
        alt = np.array([4.0, 2.0, 0.0])
        out = p.evaluate(alt)
        expected = 1e-8 * np.exp(-alt / 2.0)
        assert np.allclose(out, expected)
        assert isinstance(p, VerticalProfile)


class TestOdToMassProfile:
    def test_inverts_target_od_discrete(self):
        """Per-layer masses must sum (with block beta_ext and dz) to tau_ref."""
        block = _mie_block()
        alt = [4.0, 2.0, 0.0]  # 2 layers
        mp = od_to_mass_profile(
            block, tau_ref=0.5, ref_nm=550.0,
            altitude_km=alt, scale_height_km=2.0,
        )
        assert isinstance(mp, MassProfile)
        dz_m = -np.diff(alt) * 1000.0  # layer thicknesses in m
        rho = np.asarray(mp.kg_m3_per_layer)
        beta_ext = float(block.intensive(np.array([0.55])).beta_ext_per_mass[0])  # m^2/kg
        tau_col = float(np.sum(rho * beta_ext * dz_m))
        assert tau_col == pytest.approx(0.5, rel=0.02)

    def test_raises_on_nonpositive_beta(self):
        # A block whose intensive beta_ext is <= 0 cannot be inverted; hard to build
        # from MieSpecies, so check the guard via a tiny stub block.
        class _ZeroBlock:
            name = "zero"
            def intensive(self, wl_um, n_legendre=32):
                from pyradtran.models.aerosol_composite import SpeciesOptics
                return SpeciesOptics(beta_ext_per_mass=np.array([0.0]),
                                     ssa=np.array([0.0]), g=np.array([0.0]))
            @property
            def mass_per_particle_kg(self) -> float:
                return 1.0

        with pytest.raises(ValueError):
            od_to_mass_profile(_ZeroBlock(), tau_ref=0.5, ref_nm=550.0,
                               altitude_km=[4.0, 0.0], scale_height_km=2.0)


class TestAerosolBlockProtocol:
    def test_mie_satisfies_protocol(self):
        b = _mie_block()
        assert hasattr(b, "intensive")
        assert hasattr(b, "mass_per_particle_kg")
        assert hasattr(b, "name")
        # runtime_checkable protocol: structural isinstance
        assert isinstance(b, AerosolBlock)


from pyradtran.models.blocks import PlacedBlock  # noqa: E402


class TestPlacedBlock:
    def test_to_layer_optics_matches_intensity_times_profile(self):
        block = _mie_block()
        alt = [10.0, 0.0]
        pb = PlacedBlock(block=block, profile=MassProfile(kg_m3_per_layer=(1e-3,)))
        wl = np.array([0.55])
        layer = pb.to_layer_optics(wl, alt, n_legendre=8)
        # tau = beta_ext * rho * dz ; dz = 10 km = 1e4 m
        beta = block.intensive(wl, n_legendre=8).beta_ext_per_mass[0]
        expected_tau = beta * 1e-3 * 1e4
        assert layer.tau.shape == (1, 1)
        assert layer.tau[0, 0] == pytest.approx(expected_tau, rel=0.01)
        assert layer.legendre_moments.shape == (1, 1, 8)
        assert layer.ssa[0, 0] > 0
        assert pb.name == "MieSpecies"

    def test_modify_tau_scale(self):
        from pyradtran.models.aerosol import AerosolModifyEntry

        block = _mie_block()
        alt = [10.0, 0.0]
        wl = np.array([0.55])
        base = PlacedBlock(
            block=block, profile=MassProfile(kg_m3_per_layer=(1e-3,))
        ).to_layer_optics(wl, alt, n_legendre=8)
        scaled = PlacedBlock(
            block=block,
            profile=MassProfile(kg_m3_per_layer=(1e-3,)),
            modify=(AerosolModifyEntry(variable="tau", action="scale", value=2.0),),
        ).to_layer_optics(wl, alt, n_legendre=8)
        assert scaled.tau[0, 0] == pytest.approx(2.0 * base.tau[0, 0], rel=1e-9)


class TestDirectLayerOpticsBlock:
    def test_roundtrip_write_read(self):
        """write_explicit_aerosol -> DirectLayerOpticsBlock recovers tau/ssa."""
        import tempfile
        from pathlib import Path

        from pyradtran.models.blocks import DirectLayerOpticsBlock
        from pyradtran.optics.layer_writer import write_explicit_aerosol

        # Block whose refractive index covers the test wavelengths.
        ri = RefractiveIndex(
            wavelength_um=[0.45, 0.55, 0.65], n_real=[1.5, 1.5, 1.5], k_imag=[0.01, 0.01, 0.01]
        )
        sd = SizeDistribution(kind="monodisperse", params={"radius_um": 0.5})
        block = MieSpecies(
            refractive_index=ri, size_distribution=sd, particle_density_kg_m3=1000.0,
            integration_config=IntegrationConfig(n_radius_grid=30),
        )
        alt = [6.0, 4.0, 2.0, 0.0]  # 3 layers
        wl = np.array([0.45, 0.55, 0.65])
        original = PlacedBlock(
            block=block, profile=MassProfile(kg_m3_per_layer=(5e-4, 5e-4, 5e-4))
        ).to_layer_optics(wl, alt, n_legendre=8)

        with tempfile.TemporaryDirectory() as d:
            master = write_explicit_aerosol(
                tau=original.tau, ssa=original.ssa, g=original.g,
                legendre_moments=original.legendre_moments,
                wavelength_um=wl, altitude_km=np.asarray(alt), output_dir=Path(d),
                source_signatures=["test"],
            )
            recovered = DirectLayerOpticsBlock(
                master_path=str(master), name="testfile"
            ).to_layer_optics(wl, alt, n_legendre=8)

        assert recovered.tau.shape == original.tau.shape
        assert np.allclose(recovered.tau, original.tau, rtol=1e-4, atol=1e-12)
        assert np.allclose(recovered.ssa, original.ssa, atol=1e-4)
