"""Tests for Pydantic model validation and serialization.

Covers Tasks 2, 3, and 4 of Phase 1:
  - UvspecOption base class
  - AtmosphereConfig
  - SourceConfig, WavelengthConfig, SolverConfig, OutputConfig
  - SurfaceConfig, AerosolConfig
  - Placeholder models (CloudConfig, McConfig, AdvancedConfig)
"""

import pytest

from pyradtran.models.advanced import AdvancedConfig
from pyradtran.models.base import UvspecOption
from pyradtran.models.mc import McConfig
from pyradtran.models.output import OutputConfig

# ---------------------------------------------------------------------------
# Task 2: UvspecOption base class tests
# ---------------------------------------------------------------------------


class FakeOption(UvspecOption):
    """Minimal concrete model for testing."""

    wavelength: float = 550.0
    sza: float | None = None

    def to_uvspec_lines(self) -> list[str]:
        lines = []
        lines.append(f"wavelength {self.wavelength}")
        if self.sza is not None:
            lines.append(f"sza {self.sza}")
        return lines


class TestUvspecOption:
    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            FakeOption(wavelength=550.0, nonexistent_field=1.0)

    def test_to_uvspec_lines_basic(self):
        opt = FakeOption(wavelength=550.0, sza=30.0)
        assert opt.to_uvspec_lines() == ["wavelength 550.0", "sza 30.0"]

    def test_to_uvspec_lines_optional_none(self):
        opt = FakeOption(wavelength=550.0)
        assert opt.to_uvspec_lines() == ["wavelength 550.0"]

    def test_default_values(self):
        opt = FakeOption()
        assert opt.wavelength == 550.0
        assert opt.sza is None

    def test_frozen_instance(self):
        """UvspecOption should be immutable (frozen=True)."""
        opt = FakeOption()
        with pytest.raises(Exception):
            opt.wavelength = 600.0


# ---------------------------------------------------------------------------
# Task 3: AtmosphereConfig tests
# ---------------------------------------------------------------------------


class TestAtmosphereConfig:
    def test_minimal(self):
        from pyradtran.models.atmosphere import AtmosphereConfig

        a = AtmosphereConfig(profile="us")
        lines = a.to_uvspec_lines()
        assert "atmosphere_file US-standard" in lines

    def test_custom_file(self):
        from pyradtran.models.atmosphere import AtmosphereConfig

        a = AtmosphereConfig(profile="/path/to/atm.dat")
        lines = a.to_uvspec_lines()
        assert "atmosphere_file /path/to/atm.dat" in lines

    def test_altitude_and_pressure(self):
        from pyradtran.models.atmosphere import AtmosphereConfig

        a = AtmosphereConfig(profile="us", altitude=2.663, pressure=731.5)
        lines = a.to_uvspec_lines()
        assert "altitude 2.663" in lines
        assert "pressure 731.5" in lines

    def test_mol_modify(self):
        from pyradtran.models.atmosphere import AtmosphereConfig

        a = AtmosphereConfig(
            profile="us",
            mol_modify=[("H2O", 5.0, "MM"), ("O3", 300.0, "DU")],
        )
        lines = a.to_uvspec_lines()
        assert "mol_modify H2O 5.0 MM" in lines
        assert "mol_modify O3 300.0 DU" in lines

    def test_mol_modify_units_validated(self):
        from pyradtran.models.atmosphere import AtmosphereConfig

        with pytest.raises(Exception):
            AtmosphereConfig(profile="us", mol_modify=[("O3", 300.0, "XYZ")])

    def test_pressure_valid_range(self):
        from pyradtran.models.atmosphere import AtmosphereConfig

        with pytest.raises(Exception):
            AtmosphereConfig(profile="us", pressure=-1.0)

    def test_altitude_valid_range(self):
        from pyradtran.models.atmosphere import AtmosphereConfig

        with pytest.raises(Exception):
            AtmosphereConfig(profile="us", altitude=-1e7)

    def test_mol_abs_param_reptran(self):
        from pyradtran.models.atmosphere import AtmosphereConfig

        a = AtmosphereConfig(profile="us", mol_abs_param="reptran fine")
        lines = a.to_uvspec_lines()
        assert "mol_abs_param reptran fine" in lines

    def test_profile_not_set_raises(self):
        from pyradtran.models.atmosphere import AtmosphereConfig

        with pytest.raises(Exception):
            AtmosphereConfig()

    def test_extra_field_forbidden(self):
        from pyradtran.models.atmosphere import AtmosphereConfig

        with pytest.raises(Exception):
            AtmosphereConfig(profile="us", fake_option=1)

    # --- Phase 2 tests ---

    def test_crs_model(self):
        from pyradtran.models.atmosphere import AtmosphereConfig
        a = AtmosphereConfig(
            profile="us",
            crs_model={"species": "o3", "model": "Serdyuchenko"},
        )
        lines = a.to_uvspec_lines()
        assert "crs_model o3 Serdyuchenko" in lines

    def test_crs_model_multiple(self):
        from pyradtran.models.atmosphere import AtmosphereConfig
        a = AtmosphereConfig(
            profile="us",
            crs_model=[
                {"species": "o3", "model": "Serdyuchenko"},
                {"species": "no2", "model": "Bogumil"},
            ],
        )
        lines = a.to_uvspec_lines()
        assert "crs_model o3 Serdyuchenko" in lines
        assert "crs_model no2 Bogumil" in lines

    def test_mol_abs_param_all_variants(self):
        from pyradtran.models.atmosphere import AtmosphereConfig
        for param in [
            "reptran", "reptran fine", "reptran coarse", "reptran medium",
            "lowtran", "kato", "kato2", "fu", "crs",
        ]:
            a = AtmosphereConfig(profile="us", mol_abs_param=param)
            lines = a.to_uvspec_lines()
            assert f"mol_abs_param {param}" in lines

    def test_crs_model_invalid_species(self):
        from pyradtran.models.atmosphere import AtmosphereConfig
        with pytest.raises(Exception):
            AtmosphereConfig(
                profile="us",
                crs_model={"species": "bogus", "model": "Serdyuchenko"},
            )


# ---------------------------------------------------------------------------
# Task 4: SourceConfig tests
# ---------------------------------------------------------------------------


class TestSourceConfig:
    def test_solar(self):
        from pyradtran.models.source import SourceConfig

        s = SourceConfig(source="solar", sza=30.0)
        lines = s.to_uvspec_lines()
        assert "source solar" in lines
        assert "sza 30.0" in lines

    def test_thermal(self):
        from pyradtran.models.source import SourceConfig

        s = SourceConfig(source="thermal")
        lines = s.to_uvspec_lines()
        assert "source thermal" in lines

    def test_with_phi0_and_doy(self):
        from pyradtran.models.source import SourceConfig

        s = SourceConfig(source="solar", sza=60.0, phi0=180.0, day_of_year=172)
        lines = s.to_uvspec_lines()
        assert "phi0 180.0" in lines
        assert "day_of_year 172" in lines

    def test_solar_flux_file(self):
        from pyradtran.models.source import SourceConfig

        s = SourceConfig(source="solar", sza=30.0, solar_flux_file="/data/kurudz.dat")
        lines = s.to_uvspec_lines()
        assert "source solar /data/kurudz.dat" in lines

    def test_sza_valid_range(self):
        from pyradtran.models.source import SourceConfig

        with pytest.raises(Exception):
            SourceConfig(source="solar", sza=200.0)

    def test_phi0_valid_range(self):
        from pyradtran.models.source import SourceConfig

        with pytest.raises(Exception):
            SourceConfig(source="solar", sza=30.0, phi0=400.0)

    def test_solar_requires_sza(self):
        from pyradtran.models.source import SourceConfig

        with pytest.raises(Exception):
            SourceConfig(source="solar")


class TestSourceConfigAdvanced:
    def test_umu_single(self):
        from pyradtran.models.source import SourceConfig

        s = SourceConfig(source="solar", sza=30.0, umu=[1.0])
        lines = s.to_uvspec_lines()
        assert "umu 1.0" in lines

    def test_umu_multiple(self):
        from pyradtran.models.source import SourceConfig

        s = SourceConfig(source="solar", sza=30.0, umu=[1.0, 0.5, -1.0])
        lines = s.to_uvspec_lines()
        assert "umu 1.0 0.5 -1.0" in lines

    def test_phi(self):
        from pyradtran.models.source import SourceConfig

        s = SourceConfig(source="solar", sza=30.0, phi=[0.0, 90.0, 180.0])
        lines = s.to_uvspec_lines()
        assert "phi 0.0 90.0 180.0" in lines

    def test_umu_phi_combined(self):
        from pyradtran.models.source import SourceConfig

        s = SourceConfig(source="solar", sza=60.0, umu=[1.0, 0.8], phi=[0.0, 180.0])
        lines = s.to_uvspec_lines()
        assert "umu 1.0 0.8" in lines
        assert "phi 0.0 180.0" in lines


# ---------------------------------------------------------------------------
# Task 5: SourceConfig satellite geometry tests
# ---------------------------------------------------------------------------


class TestSourceConfigSatellite:
    """Tests for satellite geometry options (Phase 4)."""

    def test_satellite_geometry(self):
        from pyradtran.models.source import SourceConfig

        s = SourceConfig(source="solar", sza=60.0, satellite_geometry="SENTINEL2A")
        lines = s.to_uvspec_lines()
        assert "satellite_geometry SENTINEL2A" in lines

    def test_satellite_pixel(self):
        from pyradtran.models.source import SourceConfig

        s = SourceConfig(source="solar", sza=60.0, satellite_pixel=(100, 200))
        lines = s.to_uvspec_lines()
        assert "satellite_pixel 100 200" in lines

    def test_satellite_pixel_negative_coords(self):
        from pyradtran.models.source import SourceConfig

        s = SourceConfig(source="solar", sza=0.0, satellite_pixel=(-50, 100))
        lines = s.to_uvspec_lines()
        assert "satellite_pixel -50 100" in lines

    def test_satellite_geometry_and_pixel(self):
        from pyradtran.models.source import SourceConfig

        s = SourceConfig(source="solar", sza=60.0, satellite_geometry="MPS",
                          satellite_pixel=(10, 20))
        lines = s.to_uvspec_lines()
        assert "satellite_geometry MPS" in lines
        assert "satellite_pixel 10 20" in lines


# ---------------------------------------------------------------------------
# Task 4: WavelengthConfig tests
# ---------------------------------------------------------------------------


class TestWavelengthConfig:
    def test_range(self):
        from pyradtran.models.wavelength import WavelengthConfig

        w = WavelengthConfig(wavelength_min=250.0, wavelength_max=1200.0)
        lines = w.to_uvspec_lines()
        assert "wavelength 250.0 1200.0" in lines

    def test_wavenumber_range(self):
        from pyradtran.models.wavelength import WavelengthConfig

        w = WavelengthConfig(wavelength_min=10000.0, wavelength_max=500.0, unit="cm-1")
        lines = w.to_uvspec_lines()
        assert "wavelength 10000.0 500.0 cm-1" in lines

    def test_with_spline(self):
        from pyradtran.models.wavelength import WavelengthConfig

        w = WavelengthConfig(wavelength_min=300.0, wavelength_max=2500.0, spline="0.05 1.0 0.05")
        lines = w.to_uvspec_lines()
        assert "spline 0.05 1.0 0.05" in lines

    def test_filter_function_file(self):
        from pyradtran.models.wavelength import WavelengthConfig

        w = WavelengthConfig(
            wavelength_min=300.0,
            wavelength_max=1200.0,
            filter_function_file="/data/filter.dat",
        )
        lines = w.to_uvspec_lines()
        assert "filter_function_file /data/filter.dat" in lines

    def test_wavelength_valid_range(self):
        from pyradtran.models.wavelength import WavelengthConfig

        with pytest.raises(Exception):
            WavelengthConfig(wavelength_min=-1.0, wavelength_max=1000.0)


# ---------------------------------------------------------------------------
# Task 4: SolverConfig tests
# ---------------------------------------------------------------------------


class TestSolverConfig:
    def test_disort(self):
        from pyradtran.models.solver import SolverConfig

        from pyradtran.models.solver import SolverConfig
        s = SolverConfig(method="disort", streams=16)
        lines = s.to_uvspec_lines()
        assert "rte_solver disort" in lines
        assert "number_of_streams 16" in lines

    def test_twostr(self):
        from pyradtran.models.solver import SolverConfig

        from pyradtran.models.solver import SolverConfig
        s = SolverConfig(method="twostr")
        lines = s.to_uvspec_lines()
        assert "rte_solver twostr" in lines

    def test_pseudospherical(self):
        from pyradtran.models.solver import SolverConfig

        from pyradtran.models.solver import SolverConfig
        s = SolverConfig(method="disort", streams=8, pseudospherical=True)
        lines = s.to_uvspec_lines()
        assert "pseudospherical" in lines

    def test_deltam(self):
        from pyradtran.models.solver import SolverConfig

        from pyradtran.models.solver import SolverConfig
        s = SolverConfig(method="disort", streams=16, deltam=True)
        lines = s.to_uvspec_lines()
        assert "deltam" in lines

    def test_streams_valid_range(self):
        from pyradtran.models.solver import SolverConfig

        with pytest.raises(Exception):
            SolverConfig(method="disort", streams=0)

    def test_invalid_solver(self):
        from pyradtran.models.solver import SolverConfig

        with pytest.raises(Exception):
            SolverConfig(method="bogus_solver")


# ---------------------------------------------------------------------------
# Task 4: OutputConfig tests
# ---------------------------------------------------------------------------


class TestOutputConfig:
    def test_output_user(self):
        from pyradtran.models.output import OutputConfig

        o = OutputConfig(quantities=["lambda", "edir", "edn"])
        lines = o.to_uvspec_lines()
        assert "output_user lambda edir edn" in lines

    def test_output_quantity(self):
        from pyradtran.models.output import OutputConfig

        o = OutputConfig(quantity="transmittance")
        lines = o.to_uvspec_lines()
        assert "output_quantity transmittance" in lines

    def test_netcdf_format(self):
        from pyradtran.models.output import OutputConfig

        o = OutputConfig(format="netcdf")
        lines = o.to_uvspec_lines()
        assert "output_format netcdf" in lines

    def test_zout(self):
        from pyradtran.models.output import OutputConfig

        o = OutputConfig(zout=[0, 100])
        lines = o.to_uvspec_lines()
        zout_lines = [line for line in lines if line.startswith("zout")]
        assert len(zout_lines) == 1
        parts = zout_lines[0].split()
        assert parts[0] == "zout"
        assert float(parts[1]) == 0.0
        assert float(parts[2]) == 100.0

    def test_quiet(self):
        from pyradtran.models.output import OutputConfig

        o = OutputConfig(quiet=True)
        lines = o.to_uvspec_lines()
        assert "quiet" in lines

    def test_output_file(self):
        from pyradtran.models.output import OutputConfig

        o = OutputConfig(output_file="result.nc")
        lines = o.to_uvspec_lines()
        assert "output_file result.nc" in lines

    def test_quiet_verbose_mutual_exclusion(self):
        from pyradtran.models.output import OutputConfig

        with pytest.raises(Exception):
            OutputConfig(quiet=True, verbose=True)

    def test_invalid_format(self):
        from pyradtran.models.output import OutputConfig

        with pytest.raises(Exception):
            OutputConfig(format="json")


# ---------------------------------------------------------------------------
# Task 3: OutputConfig advanced (heating_rate, output_process) tests
# ---------------------------------------------------------------------------


class TestOutputConfigAdvanced:
    def test_heating_rate_local(self):
        o = OutputConfig(heating_rate="local")
        lines = o.to_uvspec_lines()
        assert "heating_rate local" in lines

    def test_heating_rate_layer_fd(self):
        o = OutputConfig(heating_rate="layer_fd")
        lines = o.to_uvspec_lines()
        assert "heating_rate layer_fd" in lines

    def test_heating_rate_none(self):
        o = OutputConfig(heating_rate="none")
        lines = o.to_uvspec_lines()
        assert "heating_rate none" in lines

    def test_heating_rate_invalid(self):
        with pytest.raises(Exception):
            OutputConfig(heating_rate="invalid")

    def test_output_process_integrate(self):
        o = OutputConfig(process="integrate")
        lines = o.to_uvspec_lines()
        assert "output_process integrate" in lines

    def test_output_process_per_nm(self):
        o = OutputConfig(process="per_nm")
        lines = o.to_uvspec_lines()
        assert "output_process per_nm" in lines


# ---------------------------------------------------------------------------
# Task 4: SurfaceConfig tests
# ---------------------------------------------------------------------------


class TestSurfaceConfig:
    def test_albedo(self):
        from pyradtran.models.surface import SurfaceConfig

        s = SurfaceConfig(albedo=0.2)
        lines = s.to_uvspec_lines()
        assert "albedo 0.2" in lines

    def test_albedo_valid_range(self):
        from pyradtran.models.surface import SurfaceConfig

        with pytest.raises(Exception):
            SurfaceConfig(albedo=1.5)

    def test_albedo_file(self):
        from pyradtran.models.surface import SurfaceConfig

        s = SurfaceConfig(albedo_file="/data/albedo.dat")
        lines = s.to_uvspec_lines()
        assert "albedo_file /data/albedo.dat" in lines

    def test_surface_temperature(self):
        from pyradtran.models.surface import SurfaceConfig

        s = SurfaceConfig(albedo=0.1, sur_temperature=288.0)
        lines = s.to_uvspec_lines()
        assert "sur_temperature 288.0" in lines

    def test_albedo_and_albedo_file_mutual_exclusion(self):
        from pyradtran.models.surface import SurfaceConfig

        with pytest.raises(Exception):
            SurfaceConfig(albedo=0.2, albedo_file="/data/albedo.dat")

    # --- Phase 2 tests ---

    def test_brdf_ambrals(self):
        from pyradtran.models.surface import SurfaceConfig
        s = SurfaceConfig(brdf_ambrals={"iso": 0.3, "vol": 0.1, "geo": 0.05})
        lines = s.to_uvspec_lines()
        assert "brdf_ambrals iso 0.3" in lines
        assert "brdf_ambrals vol 0.1" in lines
        assert "brdf_ambrals geo 0.05" in lines

    def test_brdf_hapke(self):
        from pyradtran.models.surface import SurfaceConfig
        s = SurfaceConfig(brdf_hapke={"w": 0.4, "b0": 1.0, "h": 0.06})
        lines = s.to_uvspec_lines()
        assert "brdf_hapke w 0.4" in lines

    def test_brdf_rpv(self):
        from pyradtran.models.surface import SurfaceConfig
        s = SurfaceConfig(brdf_rpv={"rho0": 0.076, "k": 0.9, "theta": -0.1, "scale": 0.1})
        lines = s.to_uvspec_lines()
        assert "brdf_rpv rho0 0.076" in lines

    def test_brdf_cam(self):
        from pyradtran.models.surface import SurfaceConfig
        s = SurfaceConfig(brdf_cam={"pcl": 0.1, "sal": 0.05, "u10": 7.0})
        lines = s.to_uvspec_lines()
        assert "brdf_cam u10 7.0" in lines

    def test_bpdf_litvinov(self):
        from pyradtran.models.surface import SurfaceConfig
        s = SurfaceConfig(bpdf_litvinov={"albedo": 0.6, "rms_slope": 0.3})
        lines = s.to_uvspec_lines()
        assert any("bpdf_litvinov" in line for line in lines)

    def test_bpdf_maignan(self):
        from pyradtran.models.surface import SurfaceConfig
        s = SurfaceConfig(bpdf_maignan={"c_maign": 0.18})
        lines = s.to_uvspec_lines()
        assert any("bpdf_maignan" in line for line in lines)

    def test_bpdf_tsang_u10(self):
        from pyradtran.models.surface import SurfaceConfig
        s = SurfaceConfig(bpdf_tsang_u10=5.0)
        lines = s.to_uvspec_lines()
        assert "bpdf_tsang_u10 5.0" in lines

    def test_albedo_map(self):
        from pyradtran.models.surface import SurfaceConfig
        s = SurfaceConfig(albedo_map="/data/albedo.nc")
        lines = s.to_uvspec_lines()
        assert "albedo_map /data/albedo.nc" in lines

    def test_albedo_map_with_variable(self):
        from pyradtran.models.surface import SurfaceConfig
        s = SurfaceConfig(albedo_map=("/data/albedo.nc", "ALBEDO"))
        lines = s.to_uvspec_lines()
        assert "albedo_map /data/albedo.nc ALBEDO" in lines

    def test_albedo_library(self):
        from pyradtran.models.surface import SurfaceConfig
        s = SurfaceConfig(albedo_library="IGBP")
        lines = s.to_uvspec_lines()
        assert "albedo_library IGBP" in lines

    def test_albedo_with_brdf_raises(self):
        from pyradtran.models.surface import SurfaceConfig
        with pytest.raises(Exception):
            SurfaceConfig(albedo=0.2, brdf_ambrals={"iso": 0.3})

    def test_albedo_with_bpdf_raises(self):
        from pyradtran.models.surface import SurfaceConfig
        with pytest.raises(Exception):
            SurfaceConfig(albedo=0.2, bpdf_tsang_u10=5.0)


# ---------------------------------------------------------------------------
# Task 4: AerosolConfig tests
# ---------------------------------------------------------------------------


class TestAerosolConfig:
    def test_default(self):
        from pyradtran.models.aerosol import AerosolConfig

        a = AerosolConfig(default=True)
        lines = a.to_uvspec_lines()
        assert "aerosol_default" in lines

    def test_angstrom(self):
        from pyradtran.models.aerosol import AerosolConfig

        a = AerosolConfig(default=True, angstrom_alpha=1.3, angstrom_beta=0.08)
        lines = a.to_uvspec_lines()
        assert "aerosol_angstrom 1.3 0.08" in lines

    def test_angstrom_requires_default(self):
        from pyradtran.models.aerosol import AerosolConfig

        with pytest.raises(Exception):
            AerosolConfig(angstrom_alpha=1.3, angstrom_beta=0.08)

    def test_angstrom_requires_both_alpha_and_beta(self):
        from pyradtran.models.aerosol import AerosolConfig

        with pytest.raises(Exception, match="angstrom_beta must be set"):
            AerosolConfig(default=True, angstrom_alpha=1.3)
        with pytest.raises(Exception, match="angstrom_alpha must be set"):
            AerosolConfig(default=True, angstrom_beta=0.08)

    def test_set_tau_at_wvl(self):
        from pyradtran.models.aerosol import AerosolConfig

        a = AerosolConfig(set_tau_at_wvl=(550.0, 0.3))
        lines = a.to_uvspec_lines()
        assert "aerosol_set_tau_at_wvl 550.0 0.3" in lines

    def test_haze_vulcan(self):
        from pyradtran.models.aerosol import AerosolConfig

        a = AerosolConfig(haze=1, vulcan=1, season=1, visibility=23.0)
        lines = a.to_uvspec_lines()
        assert "aerosol_haze 1" in lines
        assert "aerosol_vulcan 1" in lines

    def test_file(self):
        from pyradtran.models.aerosol import AerosolConfig

        a = AerosolConfig(file=("explicit", "/data/profile.dat"))
        lines = a.to_uvspec_lines()
        assert "aerosol_file explicit /data/profile.dat" in lines

    def test_invalid_file_type(self):
        from pyradtran.models.aerosol import AerosolConfig

        with pytest.raises(Exception):
            AerosolConfig(file=("invalid", "/data/profile.dat"))

    def test_visibility_valid_range(self):
        from pyradtran.models.aerosol import AerosolConfig

        with pytest.raises(Exception):
            AerosolConfig(visibility=-1.0)

    # --- Phase 2 tests ---

    def test_aerosol_modify_scale(self):
        from pyradtran.models.aerosol import AerosolConfig
        a = AerosolConfig(
            default=True,
            modify=[{"variable": "ssa", "action": "scale", "value": 0.85}],
        )
        lines = a.to_uvspec_lines()
        assert "aerosol_modify ssa scale 0.85" in lines

    def test_aerosol_modify_set(self):
        from pyradtran.models.aerosol import AerosolConfig
        a = AerosolConfig(
            default=True,
            modify=[{"variable": "gg", "action": "set", "value": 0.7}],
        )
        lines = a.to_uvspec_lines()
        assert "aerosol_modify gg set 0.7" in lines

    def test_aerosol_refrac_index(self):
        from pyradtran.models.aerosol import AerosolConfig
        a = AerosolConfig(refrac_index=(1.75, 0.45))
        lines = a.to_uvspec_lines()
        assert "aerosol_refrac_index 1.75 0.45" in lines

    def test_aerosol_refrac_file(self):
        from pyradtran.models.aerosol import AerosolConfig
        a = AerosolConfig(refrac_file="/data/soot_refr.dat")
        lines = a.to_uvspec_lines()
        assert "aerosol_refrac_file /data/soot_refr.dat" in lines

    def test_aerosol_sizedist_file(self):
        from pyradtran.models.aerosol import AerosolConfig
        a = AerosolConfig(sizedist_file="/data/lognormal.dat")
        lines = a.to_uvspec_lines()
        assert "aerosol_sizedist_file /data/lognormal.dat" in lines

    def test_aerosol_species_file(self):
        from pyradtran.models.aerosol import AerosolConfig
        a = AerosolConfig(species_file="/data/continental_average.dat")
        lines = a.to_uvspec_lines()
        assert "aerosol_species_file /data/continental_average.dat" in lines

    def test_aerosol_species_file_with_list(self):
        from pyradtran.models.aerosol import AerosolConfig
        a = AerosolConfig(
            species_file="/data/continental_average.dat",
            species_names=["inso", "soot"],
        )
        lines = a.to_uvspec_lines()
        assert "aerosol_species_file /data/continental_average.dat inso soot" in lines

    def test_aerosol_species_library(self):
        from pyradtran.models.aerosol import AerosolConfig
        a = AerosolConfig(species_library="/data/opac_optprop/")
        lines = a.to_uvspec_lines()
        assert "aerosol_species_library /data/opac_optprop/" in lines

    def test_aerosol_modify_invalid_variable(self):
        from pyradtran.models.aerosol import AerosolConfig
        with pytest.raises(Exception):
            AerosolConfig(
                default=True,
                modify=[{"variable": "bogus", "action": "scale", "value": 1.0}],
            )

    def test_aerosol_modify_invalid_action(self):
        from pyradtran.models.aerosol import AerosolConfig
        with pytest.raises(Exception):
            AerosolConfig(
                default=True,
                modify=[{"variable": "ssa", "action": "bogus", "value": 1.0}],
            )


# ---------------------------------------------------------------------------
# Task 2: CloudConfig tests
# ---------------------------------------------------------------------------


class TestCloudConfig:
    def test_water_cloud_hu(self):
        from pyradtran.models.cloud import CloudConfig
        c = CloudConfig(wc_properties="hu")
        lines = c.to_uvspec_lines()
        assert "wc_properties hu" in lines

    def test_water_cloud_echam4(self):
        from pyradtran.models.cloud import CloudConfig
        c = CloudConfig(wc_properties="echam4")
        lines = c.to_uvspec_lines()
        assert "wc_properties echam4" in lines

    def test_ice_cloud_fu(self):
        from pyradtran.models.cloud import CloudConfig
        c = CloudConfig(ic_properties="fu")
        lines = c.to_uvspec_lines()
        assert "ic_properties fu" in lines

    def test_ice_cloud_baum(self):
        from pyradtran.models.cloud import CloudConfig
        c = CloudConfig(ic_properties="baum")
        lines = c.to_uvspec_lines()
        assert "ic_properties baum" in lines

    def test_ice_cloud_yang2013(self):
        from pyradtran.models.cloud import CloudConfig
        c = CloudConfig(ic_properties="yang2013")
        lines = c.to_uvspec_lines()
        assert "ic_properties yang2013" in lines

    def test_ic_habit(self):
        from pyradtran.models.cloud import CloudConfig
        c = CloudConfig(ic_habit="rosette-6", ic_habit_roughness="moderate")
        lines = c.to_uvspec_lines()
        assert "ic_habit rosette-6" in lines
        assert "ic_habit_yang2013 rosette-6 moderate" in lines

    def test_ic_file(self):
        from pyradtran.models.cloud import CloudConfig
        c = CloudConfig(ic_file=("1d", "/data/cloud3d.dat"))
        lines = c.to_uvspec_lines()
        assert "ic_file 1d /data/cloud3d.dat" in lines

    def test_wc_file(self):
        from pyradtran.models.cloud import CloudConfig
        c = CloudConfig(wc_file=("1d", "/data/wc.dat"))
        lines = c.to_uvspec_lines()
        assert "wc_file 1d /data/wc.dat" in lines

    def test_cloud_modify(self):
        from pyradtran.models.cloud import CloudConfig
        c = CloudConfig(
            wc_properties="hu",
            modify=[{"variable": "tau", "action": "set", "value": 10.0}],
        )
        lines = c.to_uvspec_lines()
        assert "wc_modify tau set 10.0" in lines

    def test_ic_modify(self):
        from pyradtran.models.cloud import CloudConfig
        c = CloudConfig(
            ic_properties="fu",
            ic_modify=[{"variable": "gg", "action": "scale", "value": 0.85}],
        )
        lines = c.to_uvspec_lines()
        assert "ic_modify gg scale 0.85" in lines

    def test_cloud_cover(self):
        from pyradtran.models.cloud import CloudConfig
        c = CloudConfig(cloud_cover=0.7)
        lines = c.to_uvspec_lines()
        assert "cloudcover wc 0.7" in lines

    def test_cloud_cover_with_type(self):
        from pyradtran.models.cloud import CloudConfig
        c = CloudConfig(cloud_cover=0.7, cloud_cover_type="ic")
        lines = c.to_uvspec_lines()
        assert "cloudcover ic 0.7" in lines

    def test_cloud_overlap(self):
        from pyradtran.models.cloud import CloudConfig
        c = CloudConfig(cloud_overlap="maxrand")
        lines = c.to_uvspec_lines()
        assert "cloud_overlap maxrand" in lines

    def test_invalid_ic_properties(self):
        from pyradtran.models.cloud import CloudConfig
        with pytest.raises(Exception):
            CloudConfig(ic_properties="bogus_scheme")

    def test_invalid_wc_properties(self):
        from pyradtran.models.cloud import CloudConfig
        with pytest.raises(Exception):
            CloudConfig(wc_properties="bogus_scheme")

    def test_modify_invalid_variable(self):
        from pyradtran.models.cloud import CloudConfig
        with pytest.raises(Exception):
            CloudConfig(
                wc_properties="hu",
                modify=[{"variable": "bogus", "action": "set", "value": 1.0}],
            )

    def test_interpolate_flag(self):
        from pyradtran.models.cloud import CloudConfig
        c = CloudConfig(ic_properties="fu", interpolate=True)
        lines = c.to_uvspec_lines()
        assert "ic_properties fu interpolate" in lines

    def test_empty_cloud(self):
        from pyradtran.models.cloud import CloudConfig
        c = CloudConfig()
        lines = c.to_uvspec_lines()
        assert lines == []


# ---------------------------------------------------------------------------
# Phase 3: McConfig tests
# ---------------------------------------------------------------------------


class TestMcConfig:
    def test_photons(self):
        mc = McConfig(photons=100000)
        lines = mc.to_uvspec_lines()
        assert "mc_photons 100000" in lines

    def test_backward(self):
        mc = McConfig(backward=True)
        lines = mc.to_uvspec_lines()
        assert "mc_backward" in lines

    def test_backward_pixel_range(self):
        mc = McConfig(backward=True, backward_pixel_range=(0, 0, 10, 10))
        lines = mc.to_uvspec_lines()
        assert "mc_backward 0 0 10 10" in lines

    def test_escape(self):
        mc = McConfig(escape="on")
        lines = mc.to_uvspec_lines()
        assert "mc_escape on" in lines

    def test_escape_off(self):
        mc = McConfig(escape="off")
        lines = mc.to_uvspec_lines()
        assert "mc_escape off" in lines

    def test_vroom(self):
        mc = McConfig(vroom="on")
        lines = mc.to_uvspec_lines()
        assert "mc_vroom on" in lines

    def test_polarisation(self):
        mc = McConfig(polarisation=True)
        lines = mc.to_uvspec_lines()
        assert "mc_polarisation" in lines

    def test_polarisation_with_state(self):
        mc = McConfig(polarisation=True, polarisation_state=2)
        lines = mc.to_uvspec_lines()
        assert "mc_polarisation 2" in lines

    def test_randomseed(self):
        mc = McConfig(random_seed=42)
        lines = mc.to_uvspec_lines()
        assert "mc_randomseed 42" in lines

    def test_minphotons(self):
        mc = McConfig(min_photons=100)
        lines = mc.to_uvspec_lines()
        assert "mc_minphotons 100" in lines

    def test_maxscatters(self):
        mc = McConfig(max_scatters=50)
        lines = mc.to_uvspec_lines()
        assert "mc_maxscatters 50" in lines

    def test_spectral_is(self):
        mc = McConfig(spectral_is=550.0)
        lines = mc.to_uvspec_lines()
        assert "mc_spectral_is 550.0" in lines

    def test_delta_scaling(self):
        mc = McConfig(delta_scaling_mucut=0.99, delta_scaling_n_start=0)
        lines = mc.to_uvspec_lines()
        assert "mc_delta_scaling 0.99 0" in lines

    def test_rad_alpha(self):
        mc = McConfig(rad_alpha=10.0)
        lines = mc.to_uvspec_lines()
        assert "mc_rad_alpha 10.0" in lines

    def test_backward_output(self):
        mc = McConfig(backward=True, backward_output="edn")
        lines = mc.to_uvspec_lines()
        assert "mc_backward_output edn" in lines

    def test_backward_output_with_unit(self):
        mc = McConfig(backward=True, backward_output="heat", backward_output_unit="K_per_day")
        lines = mc.to_uvspec_lines()
        assert "mc_backward_output heat K_per_day" in lines

    def test_forward_output(self):
        mc = McConfig(forward_output="heating")
        lines = mc.to_uvspec_lines()
        assert "mc_forward_output heating" in lines

    def test_backward_heat(self):
        mc = McConfig(backward=True, backward_heat="EMABS")
        lines = mc.to_uvspec_lines()
        assert "mc_backward_heat EMABS" in lines

    def test_surface_albedo(self):
        mc = McConfig(surface_reflect_always=True)
        lines = mc.to_uvspec_lines()
        assert "mc_surface_reflectalways" in lines

    def test_std(self):
        mc = McConfig(std=0.01)
        lines = mc.to_uvspec_lines()
        assert "mc_std 0.01" in lines

    def test_progressbar(self):
        mc = McConfig(progressbar=2)
        lines = mc.to_uvspec_lines()
        assert "mc_progressbar 2" in lines

    def test_jacobian(self):
        mc = McConfig(backward=True, jacobian="1D")
        lines = mc.to_uvspec_lines()
        assert "mc_jacobian 1D" in lines

    def test_jacobian_std(self):
        mc = McConfig(backward=True, jacobian="1D", jacobian_std=True)
        lines = mc.to_uvspec_lines()
        assert "mc_jacobian_std" in lines

    def test_escape_invalid_value(self):
        with pytest.raises(Exception):
            McConfig(escape="invalid")

    def test_backward_output_invalid(self):
        with pytest.raises(Exception):
            McConfig(backward=True, backward_output="invalid_quantity")

    def test_backward_heat_invalid(self):
        with pytest.raises(Exception):
            McConfig(backward=True, backward_heat="invalid")

    def test_forward_output_requires_mystic_not_backward(self):
        mc = McConfig(backward=True, forward_output="heating")
        lines = mc.to_uvspec_lines()
        assert "mc_forward_output heating" in lines

    def test_empty_mc(self):
        mc = McConfig()
        lines = mc.to_uvspec_lines()
        assert lines == []

    def test_extra_field_forbidden(self):
        with pytest.raises(Exception):
            McConfig(nonexistent_option=1)


# ---------------------------------------------------------------------------
# Phase 3: McConfig tests - 3D Geometry Options (Phase 4)
# ---------------------------------------------------------------------------


class TestMcConfig3D:
    """Tests for MC 3D geometry and sensor options (Phase 4)."""

    def test_spherical_1d(self):
        mc = McConfig(spherical="1D")
        lines = mc.to_uvspec_lines()
        assert "mc_spherical 1D" in lines

    def test_spherical_3d(self):
        mc = McConfig(spherical="3D")
        lines = mc.to_uvspec_lines()
        assert "mc_spherical 3D" in lines

    def test_spherical_invalid(self):
        with pytest.raises(Exception):
            McConfig(spherical="2D")

    def test_tenstream(self):
        mc = McConfig(tenstream=True)
        lines = mc.to_uvspec_lines()
        assert "mc_tenstream" in lines

    def test_ipa(self):
        mc = McConfig(ipa=True)
        lines = mc.to_uvspec_lines()
        assert "mc_ipa" in lines

    def test_tipa_dir(self):
        mc = McConfig(tipa="dir")
        lines = mc.to_uvspec_lines()
        assert "mc_tipa dir" in lines

    def test_tipa_dir3d(self):
        mc = McConfig(tipa="dir3d")
        lines = mc.to_uvspec_lines()
        assert "mc_tipa dir3d" in lines

    def test_tipa_invalid(self):
        with pytest.raises(Exception):
            McConfig(tipa="invalid")

    def test_sensor_direction(self):
        mc = McConfig(sensor_direction=(1.0, 0.0, -1.0))
        lines = mc.to_uvspec_lines()
        assert "mc_sensordirection 1.0 0.0 -1.0" in lines

    def test_sensor_position(self):
        mc = McConfig(sensor_position=(0.0, 0.0, 10.0))
        lines = mc.to_uvspec_lines()
        assert "mc_sensorposition 0.0 0.0 10.0" in lines

    def test_spherical3d_scene(self):
        mc = McConfig(spherical3d_scene=(-10.0, 30.0, 10.0, 50.0))
        lines = mc.to_uvspec_lines()
        assert "mc_spherical3D_scene -10.0 30.0 10.0 50.0" in lines

    def test_cloud_grid(self):
        mc = McConfig(cloud_grid=(100, 100, 50))
        lines = mc.to_uvspec_lines()
        assert "mc_cloud_grid 100 100 50" in lines

    def test_basename(self):
        mc = McConfig(basename="my_sim")
        lines = mc.to_uvspec_lines()
        assert "mc_basename my_sim" in lines

    def test_minscatters(self):
        mc = McConfig(min_scatters=5)
        lines = mc.to_uvspec_lines()
        assert "mc_minscatters 5" in lines

    def test_sun_angular_size(self):
        mc = McConfig(sun_angular_size=0.53)
        lines = mc.to_uvspec_lines()
        assert "mc_sun_angular_size 0.53" in lines


# ---------------------------------------------------------------------------
# Phase 3: McConfig tests - Advanced Surface Files (Phase 4)
# ---------------------------------------------------------------------------


class TestMcConfigSurface:
    """Tests for MC advanced surface file options (Phase 4)."""

    def test_albedo_file(self):
        mc = McConfig(albedo_file="/data/albedo.nc")
        lines = mc.to_uvspec_lines()
        assert "mc_albedo_file /data/albedo.nc" in lines

    def test_albedo_spectral_file(self):
        mc = McConfig(albedo_spectral_file="/data/alb_spec.dat")
        lines = mc.to_uvspec_lines()
        assert "mc_albedo_spectral_file /data/alb_spec.dat" in lines

    def test_rossli_file(self):
        mc = McConfig(rossli_file="/data/rossli.dat")
        lines = mc.to_uvspec_lines()
        assert "mc_rossli_file /data/rossli.dat" in lines

    def test_ambrals_spectral_file(self):
        mc = McConfig(ambrals_spectral_file="/data/ambrals.dat")
        lines = mc.to_uvspec_lines()
        assert "mc_ambrals_spectral_file /data/ambrals.dat" in lines

    def test_rpv_file(self):
        mc = McConfig(rpv_file="/data/rpv.dat")
        lines = mc.to_uvspec_lines()
        assert "mc_rpv_file /data/rpv.dat" in lines

    def test_bpdf(self):
        mc = McConfig(bpdf="maignan")
        lines = mc.to_uvspec_lines()
        assert "mc_bpdf maignan" in lines

    def test_surface_parallel(self):
        mc = McConfig(surface_parallel=True)
        lines = mc.to_uvspec_lines()
        assert "mc_surfaceparallel" in lines

    def test_elevation_file(self):
        mc = McConfig(elevation_file="/data/dem.nc")
        lines = mc.to_uvspec_lines()
        assert "mc_elevation_file /data/dem.nc" in lines

    def test_lidar_file(self):
        mc = McConfig(lidar_file="/data/lidar_input.dat")
        lines = mc.to_uvspec_lines()
        assert "mc_lidar_file /data/lidar_input.dat" in lines

    def test_triangular_surface_file(self):
        mc = McConfig(triangular_surface_file="/data/mesh.dat")
        lines = mc.to_uvspec_lines()
        assert "mc_triangular_surface_file /data/mesh.dat" in lines

    def test_bpdf_invalid(self):
        with pytest.raises(Exception):
            McConfig(bpdf="invalid_model")


# ---------------------------------------------------------------------------
# Phase 3: AdvancedConfig tests
# ---------------------------------------------------------------------------


class TestSolverConfigDynamic:
    """Tests for dynamic solver options (Phase 4)."""

    def test_dynamic_twostream(self):
        from pyradtran.models.solver import SolverConfig
        s = SolverConfig(method="dynamic_twostream")
        lines = s.to_uvspec_lines()
        assert "rte_solver dynamic_twostream" in lines

    def test_dynamic_tenstream(self):
        from pyradtran.models.solver import SolverConfig
        s = SolverConfig(method="dynamic_tenstream")
        lines = s.to_uvspec_lines()
        assert "rte_solver dynamic_tenstream" in lines

    def test_dynamic_iterations(self):
        from pyradtran.models.solver import SolverConfig
        s = SolverConfig(method="dynamic_tenstream", dynamic_iterations=100)
        lines = s.to_uvspec_lines()
        assert "dynamic_tenstream_iterations 100" in lines

    def test_dynamic_history(self):
        from pyradtran.models.solver import SolverConfig
        s = SolverConfig(method="dynamic_tenstream", dynamic_history=True)
        lines = s.to_uvspec_lines()
        assert "dynamic_tenstream_history" in lines

    def test_dynamic_heat_unit(self):
        from pyradtran.models.solver import SolverConfig
        s = SolverConfig(method="dynamic_tenstream", dynamic_heat_unit="K_per_day")
        lines = s.to_uvspec_lines()
        assert "dynamic_tenstream_heat_unit K_per_day" in lines

    def test_dynamic_heat_unit_w_per_m3(self):
        from pyradtran.models.solver import SolverConfig
        s = SolverConfig(method="dynamic_tenstream", dynamic_heat_unit="W_per_m3")
        lines = s.to_uvspec_lines()
        assert "dynamic_tenstream_heat_unit W_per_m3" in lines

    def test_dynamic_heat_unit_invalid(self):
        with pytest.raises(Exception):
            SolverConfig(method="dynamic_tenstream", dynamic_heat_unit="invalid")

    def test_dynamic_iterations_negative(self):
        with pytest.raises(Exception):
            SolverConfig(method="dynamic_tenstream", dynamic_iterations=-1)


class TestAdvancedConfig:
    def test_fluorescence(self):
        a = AdvancedConfig(fluorescence=0.5)
        lines = a.to_uvspec_lines()
        assert "fluorescence 0.5" in lines

    def test_fluorescence_file(self):
        a = AdvancedConfig(fluorescence_file="/data/fluor.dat")
        lines = a.to_uvspec_lines()
        assert "fluorescence_file /data/fluor.dat" in lines

    def test_raman(self):
        a = AdvancedConfig(raman=True)
        lines = a.to_uvspec_lines()
        assert "raman" in lines

    def test_raman_original(self):
        a = AdvancedConfig(raman=True, raman_variant="original")
        lines = a.to_uvspec_lines()
        assert "raman original" in lines

    def test_fluorescence_valid_range(self):
        with pytest.raises(Exception):
            AdvancedConfig(fluorescence=-1.0)

    def test_fluorescence_mutual_exclusion(self):
        with pytest.raises(Exception):
            AdvancedConfig(fluorescence=0.5, fluorescence_file="/data/f.dat")

    def test_extra_field_forbidden(self):
        with pytest.raises(Exception):
            AdvancedConfig(invalid=1)
