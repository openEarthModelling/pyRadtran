"""Tests for Pydantic model validation and serialization.

Covers Tasks 2, 3, and 4 of Phase 1:
  - UvspecOption base class
  - AtmosphereConfig
  - SourceConfig, WavelengthConfig, SolverConfig, OutputConfig
  - SurfaceConfig, AerosolConfig
  - Placeholder models (CloudConfig, McConfig, AdvancedConfig)
"""

import pytest

from pyradtran.models.base import UvspecOption

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

        s = SolverConfig(method="disort", streams=16)
        lines = s.to_uvspec_lines()
        assert "rte_solver disort" in lines
        assert "number_of_streams 16" in lines

    def test_twostr(self):
        from pyradtran.models.solver import SolverConfig

        s = SolverConfig(method="twostr")
        lines = s.to_uvspec_lines()
        assert "rte_solver twostr" in lines

    def test_pseudospherical(self):
        from pyradtran.models.solver import SolverConfig

        s = SolverConfig(method="disort", streams=8, pseudospherical=True)
        lines = s.to_uvspec_lines()
        assert "pseudospherical" in lines

    def test_deltam(self):
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
            modify=[{"variable": "gg", "action": "set", "value": 0.70}],
        )
        lines = a.to_uvspec_lines()
        assert "aerosol_modify gg set 0.70" in lines

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
        c = CloudConfig(ic_habit="rosette-6", ic_habit_roughness=0.5)
        lines = c.to_uvspec_lines()
        assert "ic_habit rosette-6" in lines
        assert "ic_habit_yang2013 rosette-6 0.5" in lines

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
        assert "cloudcover 0.7" in lines

    def test_cloud_overlap(self):
        from pyradtran.models.cloud import CloudConfig
        c = CloudConfig(cloud_overlap="maxrnd")
        lines = c.to_uvspec_lines()
        assert "cloud_overlap maxrnd" in lines

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
# Placeholder models
# ---------------------------------------------------------------------------


class TestPlaceholderModels:
    def test_mc_config(self):
        from pyradtran.models.mc import McConfig

        m = McConfig()
        assert m.to_uvspec_lines() == []

    def test_advanced_config(self):
        from pyradtran.models.advanced import AdvancedConfig

        a = AdvancedConfig()
        assert a.to_uvspec_lines() == []

    def test_mc_config(self):
        from pyradtran.models.mc import McConfig

        m = McConfig()
        assert m.to_uvspec_lines() == []

    def test_advanced_config(self):
        from pyradtran.models.advanced import AdvancedConfig

        a = AdvancedConfig()
        assert a.to_uvspec_lines() == []
