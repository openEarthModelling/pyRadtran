"""Tests for Pydantic model validation and serialization.

Covers:
  - UvspecOption base class
  - AtmosphereConfig, SourceConfig, WavelengthConfig, SolverConfig, OutputConfig
  - SurfaceConfig
  - AerosolModel, OpacPreset, OpacCustom, ExternalAerosol, AerosolModifyEntry
  - CloudConfig, McConfig, AdvancedConfig
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
            "reptran",
            "reptran fine",
            "reptran coarse",
            "reptran medium",
            "lowtran",
            "kato",
            "kato2",
            "fu",
            "crs",
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

    def test_mol_modify_precip_cm(self):
        from pyradtran.models.atmosphere import AtmosphereConfig

        a = AtmosphereConfig(profile="us", mol_modify=[("H2O", 2.5, "precip_cm")])
        lines = a.to_uvspec_lines()
        assert "mol_modify H2O 2.5 precip_cm" in lines

    def test_mol_modify_ppmv(self):
        from pyradtran.models.atmosphere import AtmosphereConfig

        a = AtmosphereConfig(profile="us", mol_modify=[("CO2", 410.0, "ppmv")])
        lines = a.to_uvspec_lines()
        assert "mol_modify CO2 410.0 ppmv" in lines

    def test_atm_z_grid(self):
        from pyradtran.models.atmosphere import AtmosphereConfig

        a = AtmosphereConfig(profile="us", atm_z_grid=[0, 1, 2, 5, 10, 20, 35, 50])
        lines = a.to_uvspec_lines()
        assert "atm_z_grid 0.0 1.0 2.0 5.0 10.0 20.0 35.0 50.0" in lines

    def test_radiosonde(self):
        from pyradtran.models.atmosphere import AtmosphereConfig

        a = AtmosphereConfig(profile="/data/sonde.dat", radiosonde=True)
        lines = a.to_uvspec_lines()
        assert "radiosonde" in lines

    def test_radiosonde_levels_only_requires_radiosonde(self):
        from pyradtran.models.atmosphere import AtmosphereConfig

        with pytest.raises(Exception, match="radiosonde_levels_only"):
            AtmosphereConfig(profile="us", radiosonde_levels_only=True)

    def test_mol_file(self):
        from pyradtran.models.atmosphere import AtmosphereConfig

        a = AtmosphereConfig(
            profile="us", mol_file=[{"species": "CH4", "file": "/data/ch4_vmr.dat"}]
        )
        lines = a.to_uvspec_lines()
        assert "mol_file CH4 /data/ch4_vmr.dat" in lines

    def test_mol_file_with_unit(self):
        from pyradtran.models.atmosphere import AtmosphereConfig

        a = AtmosphereConfig(
            profile="us",
            mol_file=[{"species": "CH4", "file": "/data/ch4_vmr.dat", "unit": "ppmv"}],
        )
        lines = a.to_uvspec_lines()
        assert "mol_file CH4 /data/ch4_vmr.dat ppmv" in lines

    def test_mol_tau_file(self):
        from pyradtran.models.atmosphere import AtmosphereConfig

        a = AtmosphereConfig(profile="us", mol_tau_file=("abs", "/data/tau.dat"))
        lines = a.to_uvspec_lines()
        assert "mol_tau_file abs /data/tau.dat" in lines

    def test_rayleigh_depol(self):
        from pyradtran.models.atmosphere import AtmosphereConfig

        a = AtmosphereConfig(profile="us", rayleigh_depol=0.0279)
        lines = a.to_uvspec_lines()
        assert "rayleigh_depol 0.0279" in lines

    def test_raman(self):
        from pyradtran.models.atmosphere import AtmosphereConfig

        a = AtmosphereConfig(profile="us", raman=True)
        lines = a.to_uvspec_lines()
        assert "raman" in lines


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

    def test_latitude_longitude(self):
        from pyradtran.models.source import SourceConfig

        s = SourceConfig(
            source="solar",
            sza=60.0,
            latitude=("N", 30, 0, 0),
            longitude=("E", 120, 0, 0),
        )
        lines = s.to_uvspec_lines()
        assert "latitude N 30 0 0" in lines
        assert "longitude E 120 0 0" in lines

    def test_time(self):
        from pyradtran.models.source import SourceConfig

        s = SourceConfig(source="solar", sza=60.0, time="10:30:00")
        lines = s.to_uvspec_lines()
        assert "time 10:30:00" in lines

    def test_time_interpolate(self):
        from pyradtran.models.source import SourceConfig

        s = SourceConfig(source="solar", sza=60.0, time="10:30:00", time_interpolate=True)
        lines = s.to_uvspec_lines()
        assert "time_interpolate" in lines

    def test_earth_radius(self):
        from pyradtran.models.source import SourceConfig

        s = SourceConfig(source="solar", sza=60.0, earth_radius=6371.0)
        lines = s.to_uvspec_lines()
        assert "earth_radius 6371.0" in lines

    def test_sza_file(self):
        from pyradtran.models.source import SourceConfig

        s = SourceConfig(source="solar", sza_file="/data/sza.dat")
        lines = s.to_uvspec_lines()
        assert "sza_file /data/sza.dat" in lines

    def test_isotropic_source_toa(self):
        from pyradtran.models.source import SourceConfig

        s = SourceConfig(source="solar", isotropic_source_toa=True)
        lines = s.to_uvspec_lines()
        assert "isotropic_source_toa" in lines

    def test_sza_mutual_exclusion(self):
        from pyradtran.models.source import SourceConfig

        with pytest.raises(Exception):
            SourceConfig(source="solar", sza=30.0, sza_file="/data/sza.dat")


# ---------------------------------------------------------------------------
# Task 5: SourceConfig satellite geometry tests
# ---------------------------------------------------------------------------


class TestSourceConfigSatellite:
    """Tests for satellite geometry options (Phase 4)."""

    def test_satellite_geometry(self):
        from pyradtran.models.source import SourceConfig

        s = SourceConfig(
            source="solar", sza=60.0, satellite_geometry="SENTINEL2A", satellite_pixel=(10, 20)
        )
        lines = s.to_uvspec_lines()
        assert "satellite_geometry SENTINEL2A" in lines

    def test_satellite_pixel(self):
        from pyradtran.models.source import SourceConfig

        s = SourceConfig(
            source="solar", sza=60.0, satellite_geometry="MPS", satellite_pixel=(100, 200)
        )
        lines = s.to_uvspec_lines()
        assert "satellite_pixel 100 200" in lines

    def test_satellite_pixel_negative_coords(self):
        from pyradtran.models.source import SourceConfig

        s = SourceConfig(
            source="solar", sza=0.0, satellite_geometry="MPS", satellite_pixel=(-50, 100)
        )
        lines = s.to_uvspec_lines()
        assert "satellite_pixel -50 100" in lines

    def test_satellite_geometry_without_pixel_raises(self):
        from pyradtran.models.source import SourceConfig

        with pytest.raises(ValueError, match="satellite_pixel"):
            SourceConfig(source="solar", sza=60.0, satellite_geometry="MPS")

    def test_satellite_pixel_without_geometry_raises(self):
        from pyradtran.models.source import SourceConfig

        with pytest.raises(ValueError, match="satellite_geometry"):
            SourceConfig(source="solar", sza=60.0, satellite_pixel=(10, 20))

    def test_satellite_geometry_and_pixel(self):
        from pyradtran.models.source import SourceConfig

        s = SourceConfig(
            source="solar", sza=60.0, satellite_geometry="MPS", satellite_pixel=(10, 20)
        )
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

    def test_single_wavelength(self):
        from pyradtran.models.wavelength import WavelengthConfig

        w = WavelengthConfig(wavelength_min=500.0)
        lines = w.to_uvspec_lines()
        assert "wavelength 500.0" in lines
        assert "500.0 500.0" not in "".join(lines)

    def test_wavelength_grid_file(self):
        from pyradtran.models.wavelength import WavelengthConfig

        w = WavelengthConfig(wavelength_grid_file="/data/wl_grid.dat")
        lines = w.to_uvspec_lines()
        assert "wavelength_grid_file /data/wl_grid.dat" in lines

    def test_wavelength_index(self):
        from pyradtran.models.wavelength import WavelengthConfig

        w = WavelengthConfig(wavelength_min=300.0, wavelength_max=800.0, wavelength_index=(10, 50))
        lines = w.to_uvspec_lines()
        assert "wavelength_index 10 50" in lines

    def test_slit_function_file(self):
        from pyradtran.models.wavelength import WavelengthConfig

        w = WavelengthConfig(
            wavelength_min=300.0,
            wavelength_max=800.0,
            slit_function_file="/data/slit.dat",
        )
        lines = w.to_uvspec_lines()
        assert "slit_function_file /data/slit.dat" in lines

    def test_thermal_bands_file(self):
        from pyradtran.models.wavelength import WavelengthConfig

        w = WavelengthConfig(
            wavelength_min=8000.0,
            wavelength_max=12000.0,
            thermal_bands_file="/data/bands.dat",
        )
        lines = w.to_uvspec_lines()
        assert "thermal_bands_file /data/bands.dat" in lines


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

    def test_disort_intcor(self):
        from pyradtran.models.solver import SolverConfig

        s = SolverConfig(method="disort", streams=16, disort_intcor="moments")
        lines = s.to_uvspec_lines()
        assert "disort_intcor moments" in lines

    def test_disort_spherical_albedo(self):
        from pyradtran.models.solver import SolverConfig

        s = SolverConfig(method="disort", disort_spherical_albedo=True)
        lines = s.to_uvspec_lines()
        assert "disort_spherical_albedo" in lines

    def test_schwarzschild_streams(self):
        from pyradtran.models.solver import SolverConfig

        s = SolverConfig(method="schwarzschild", schwarzschild_streams=8)
        lines = s.to_uvspec_lines()
        assert "schwarzschild_streams 8" in lines

    def test_disort_intcor_invalid(self):
        from pyradtran.models.solver import SolverConfig

        with pytest.raises(Exception):
            SolverConfig(method="disort", disort_intcor="invalid")


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

    def test_zout_toa_boa(self):
        o = OutputConfig(zout=["toa", 5.0, "boa"])
        lines = o.to_uvspec_lines()
        zout_lines = [ln for ln in lines if ln.startswith("zout")]
        assert len(zout_lines) == 1
        assert "toa" in zout_lines[0]
        assert "boa" in zout_lines[0]

    def test_write_optical_properties(self):
        o = OutputConfig(write_optical_properties=True)
        lines = o.to_uvspec_lines()
        assert "write_optical_properties" in lines


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

    def test_brdf_rossli_file(self):
        from pyradtran.models.surface import SurfaceConfig

        s = SurfaceConfig(brdf_rossli_file="/data/rossli.dat")
        lines = s.to_uvspec_lines()
        assert "brdf_rossli_file /data/rossli.dat" in lines

    def test_brdf_rossli_hotspot(self):
        from pyradtran.models.surface import SurfaceConfig

        s = SurfaceConfig(brdf_ambrals={"iso": 0.3}, brdf_rossli_hotspot=True)
        lines = s.to_uvspec_lines()
        assert "brdf_rossli_hotspot" in lines

    def test_brdf_rpv_file(self):
        from pyradtran.models.surface import SurfaceConfig

        s = SurfaceConfig(brdf_rpv_file="/data/rpv.dat")
        lines = s.to_uvspec_lines()
        assert "brdf_rpv_file /data/rpv.dat" in lines

    def test_surface_type_map(self):
        from pyradtran.models.surface import SurfaceConfig

        s = SurfaceConfig(surface_type_map="/data/surface_type.nc")
        lines = s.to_uvspec_lines()
        assert "surface_type_map /data/surface_type.nc" in lines

    def test_surface_temperature_map(self):
        from pyradtran.models.surface import SurfaceConfig

        s = SurfaceConfig(surface_temperature_map=("/data/temp.nc", "TS", 1.0))
        lines = s.to_uvspec_lines()
        assert "surface_temperature_map /data/temp.nc TS 1.0" in lines

    def test_surface_temperature_map_file_only(self):
        from pyradtran.models.surface import SurfaceConfig

        s = SurfaceConfig(surface_temperature_map="/data/temp.nc")
        lines = s.to_uvspec_lines()
        assert "surface_temperature_map /data/temp.nc" in lines


# ---------------------------------------------------------------------------
# OPAC aerosol model tests
# ---------------------------------------------------------------------------


class TestOpacPreset:
    def test_invalid_species_names(self):
        from pyradtran.models.aerosol import OpacPreset, OpacPresetName

        with pytest.raises(Exception):
            OpacPreset(
                name=OpacPresetName.CONTINENTAL_AVERAGE,
                species_names=["bogus"],
            )

    def test_enum_has_ten_values(self):
        from pyradtran.models.aerosol import OpacPresetName

        assert len(OpacPresetName) == 10

    def test_string_preset_name(self):
        from pyradtran.models.aerosol import OpacPreset, OpacPresetName

        a = OpacPreset(name=OpacPresetName("maritime_clean"))
        assert a.name == OpacPresetName.MARITIME_CLEAN


class TestOpacCustom:
    def test_invalid_species_names(self):
        from pyradtran.models.aerosol import OpacCustom

        with pytest.raises(Exception):
            OpacCustom(species_file="/data/my_profile.dat", species_names=["bogus"])


class TestExternalAerosol:
    def test_single_file(self):
        from pyradtran.models.aerosol import ExternalFile

        a = ExternalFile(files=[("explicit", "/data/profile.dat")])
        lines = a.to_uvspec_lines()
        assert "aerosol_file explicit /data/profile.dat" in lines

    def test_multiple_files(self):
        from pyradtran.models.aerosol import ExternalFile

        a = ExternalFile(
            files=[
                ("tau", "/data/tau.dat"),
                ("ssa", "/data/ssa.dat"),
                ("moments", "/data/mom.dat"),
            ],
        )
        lines = a.to_uvspec_lines()
        assert "aerosol_file tau /data/tau.dat" in lines
        assert "aerosol_file ssa /data/ssa.dat" in lines
        assert "aerosol_file moments /data/mom.dat" in lines

    def test_ref_file_type(self):
        from pyradtran.models.aerosol import ExternalFile

        a = ExternalFile(files=[("ref", "/data/refr.dat")])
        lines = a.to_uvspec_lines()
        assert "aerosol_file ref /data/refr.dat" in lines

    def test_siz_file_type(self):
        from pyradtran.models.aerosol import ExternalFile

        a = ExternalFile(files=[("siz", "/data/sizedist.dat")])
        lines = a.to_uvspec_lines()
        assert "aerosol_file siz /data/sizedist.dat" in lines

    def test_invalid_file_type(self):
        from pyradtran.models.aerosol import ExternalFile

        with pytest.raises(Exception):
            ExternalFile(files=[("invalid", "/data/profile.dat")])


class TestAerosolModel:
    """Tests for AerosolModel base class capabilities (via ExternalFile)."""

    def test_set_tau_at_wvl(self):
        from pyradtran.models.aerosol import ExternalFile

        a = ExternalFile(files=[("explicit", "/data/x.dat")], set_tau_at_wvl=(550.0, 0.3))
        items = a.to_uvspec_items()
        lines = [line for _, line in items]
        assert "aerosol_file explicit /data/x.dat" in lines
        assert "aerosol_set_tau_at_wvl 550.0 0.3" in lines

    def test_king_byrne(self):
        from pyradtran.models.aerosol import ExternalFile

        a = ExternalFile(files=[("explicit", "/data/x.dat")], king_byrne=(1.027, -0.25, 0.03))
        items = a.to_uvspec_items()
        lines = [line for _, line in items]
        assert "aerosol_king_byrne 1.027 -0.25 0.03" in lines

    def test_modify_scale(self):
        from pyradtran.models.aerosol import ExternalFile

        a = ExternalFile(
            files=[("explicit", "/data/x.dat")],
            modify=[{"variable": "ssa", "action": "scale", "value": 0.85}],
        )
        items = a.to_uvspec_items()
        lines = [line for _, line in items]
        assert "aerosol_modify ssa scale 0.85" in lines

    def test_modify_set(self):
        from pyradtran.models.aerosol import ExternalFile

        a = ExternalFile(
            files=[("explicit", "/data/x.dat")],
            modify=[{"variable": "gg", "action": "set", "value": 0.7}],
        )
        items = a.to_uvspec_items()
        lines = [line for _, line in items]
        assert "aerosol_modify gg set 0.7" in lines

    def test_modify_invalid_variable(self):
        from pyradtran.models.aerosol import OpacPreset, OpacPresetName

        with pytest.raises(Exception):
            OpacPreset(
                name=OpacPresetName.CONTINENTAL_AVERAGE,
                modify=[{"variable": "bogus", "action": "scale", "value": 1.0}],
            )

    def test_modify_invalid_action(self):
        from pyradtran.models.aerosol import OpacPreset, OpacPresetName

        with pytest.raises(Exception):
            OpacPreset(
                name=OpacPresetName.CONTINENTAL_AVERAGE,
                modify=[{"variable": "ssa", "action": "bogus", "value": 1.0}],
            )

    def test_phase_is_5(self):
        from pyradtran.models.aerosol import ExternalFile

        a = ExternalFile(files=[("explicit", "/data/x.dat")])
        items = a.to_uvspec_items()
        assert all(phase == 5 for phase, _ in items)

    def test_external_aerosol_set_tau_at_wvl(self):
        from pyradtran.models.aerosol import ExternalFile

        a = ExternalFile(
            files=[("explicit", "/data/aerosol.dat")],
            set_tau_at_wvl=(550.0, 0.2),
        )
        items = a.to_uvspec_items()
        lines = [line for _, line in items]
        assert "aerosol_file explicit /data/aerosol.dat" in lines
        assert "aerosol_set_tau_at_wvl 550.0 0.2" in lines


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

    def test_cloud_fraction_file(self):
        from pyradtran.models.cloud import CloudConfig

        c = CloudConfig(cloud_fraction_file="/data/cfrac.dat")
        lines = c.to_uvspec_lines()
        assert "cloud_fraction_file /data/cfrac.dat" in lines

    def test_cloud_fraction_map(self):
        from pyradtran.models.cloud import CloudConfig

        c = CloudConfig(cloud_fraction_map=("/data/cfrac.nc", "CF", 1.0))
        lines = c.to_uvspec_lines()
        assert "cloud_fraction_map /data/cfrac.nc CF 1.0" in lines

    def test_wc_saturate(self):
        from pyradtran.models.cloud import CloudConfig

        c = CloudConfig(wc_properties="hu", wc_saturate=True)
        lines = c.to_uvspec_lines()
        assert "wc_saturate" in lines

    def test_ic_saturate(self):
        from pyradtran.models.cloud import CloudConfig

        c = CloudConfig(ic_properties="fu", ic_saturate=True)
        lines = c.to_uvspec_lines()
        assert "ic_saturate" in lines

    def test_wc_ipa(self):
        from pyradtran.models.cloud import CloudConfig

        c = CloudConfig(wc_properties="hu", wc_ipa=True)
        lines = c.to_uvspec_lines()
        assert "wc_ipa" in lines

    def test_wc_layer(self):
        from pyradtran.models.cloud import CloudConfig

        c = CloudConfig(wc_properties="hu", wc_layer=5)
        lines = c.to_uvspec_lines()
        assert "wc_layer 5" in lines


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

    def test_relerr(self):
        mc = McConfig(relerr=0.01)
        lines = mc.to_uvspec_lines()
        assert "mc_relerr 0.01" in lines

    def test_coherent_backscatter(self):
        mc = McConfig(coherent_backscatter=True)
        lines = mc.to_uvspec_lines()
        assert "mc_coherent_backscatter" in lines

    def test_nca(self):
        mc = McConfig(nca=True)
        lines = mc.to_uvspec_lines()
        assert "mc_nca" in lines

    def test_bcond(self):
        mc = McConfig(bcond="periodic")
        lines = mc.to_uvspec_lines()
        assert "mc_bcond periodic" in lines

    def test_bcond_invalid(self):
        with pytest.raises(Exception):
            McConfig(bcond="invalid")

    def test_aerosol_is(self):
        mc = McConfig(aerosol_is=True)
        lines = mc.to_uvspec_lines()
        assert "mc_aerosol_is" in lines


# ---------------------------------------------------------------------------
# Phase 4: McConfig 3D geometry options
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
        with pytest.raises(ValueError):
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
        with pytest.raises(ValueError):
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
# Phase 4: McConfig advanced surface files
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

    def test_rpv_spectral_file(self):
        mc = McConfig(rpv_spectral_file="/data/rpv.dat")
        lines = mc.to_uvspec_lines()
        assert "mc_rpv_spectral_file /data/rpv.dat" in lines

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
        with pytest.raises(ValueError):
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

    def test_dynamic_twostream_iterations(self):
        from pyradtran.models.solver import SolverConfig

        s = SolverConfig(method="dynamic_twostream", dynamic_iterations=50)
        lines = s.to_uvspec_lines()
        assert "dynamic_twostream_iterations 50" in lines

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
        from pyradtran.models.solver import SolverConfig

        with pytest.raises(ValueError):
            SolverConfig(method="dynamic_tenstream", dynamic_heat_unit="invalid")

    def test_dynamic_iterations_negative(self):
        from pyradtran.models.solver import SolverConfig

        with pytest.raises(ValueError):
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


# ---------------------------------------------------------------------------
# to_uvspec_items() phase tests
# ---------------------------------------------------------------------------


def test_to_uvspec_items_default_phase():
    items = FakeOption(wavelength=550.0, sza=30.0).to_uvspec_items()
    assert len(items) == 2
    assert items[0] == (9, "wavelength 550.0")
    assert items[1] == (9, "sza 30.0")


def test_atmosphere_phase_1():
    from pyradtran.models.atmosphere import AtmosphereConfig

    items = AtmosphereConfig(profile="us", altitude=2.5).to_uvspec_items()
    assert all(phase == 1 for phase, _ in items)


def test_source_phase_2():
    from pyradtran.models.source import SourceConfig

    items = SourceConfig(source="solar", sza=30.0).to_uvspec_items()
    assert all(phase == 2 for phase, _ in items)


def test_solver_phase_8():
    from pyradtran.models.solver import SolverConfig

    items = SolverConfig(method="disort", streams=16).to_uvspec_items()
    assert all(phase == 8 for phase, _ in items)


def test_output_phase_9():
    from pyradtran.models.output import OutputConfig

    items = OutputConfig(quiet=True).to_uvspec_items()
    assert all(phase == 9 for phase, _ in items)


class TestSpecialConfig:
    def test_no_absorption(self):
        from pyradtran.models.special import SpecialConfig

        s = SpecialConfig(no_absorption=True)
        items = s._scattering_items()
        assert (4, "no_absorption") in items

    def test_no_scattering(self):
        from pyradtran.models.special import SpecialConfig

        s = SpecialConfig(no_scattering=True)
        items = s._scattering_items()
        assert (4, "no_scattering") in items

    def test_no_scattering_mol(self):
        from pyradtran.models.special import SpecialConfig

        s = SpecialConfig(no_scattering_mol=True)
        items = s._scattering_items()
        assert (4, "no_scattering mol") in items

    def test_include_files(self):
        from pyradtran.models.special import SpecialConfig

        s = SpecialConfig(include_files=["extra_options.inp"])
        assert s.include_files == ["extra_options.inp"]

    def test_mutual_exclusion(self):
        from pyradtran.models.special import SpecialConfig

        with pytest.raises(Exception):
            SpecialConfig(no_scattering=True, no_scattering_mol=True)

    def test_empty(self):
        from pyradtran.models.special import SpecialConfig

        s = SpecialConfig()
        assert s._scattering_items() == []
