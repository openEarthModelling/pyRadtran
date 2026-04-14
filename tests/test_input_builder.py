"""Tests for uvspec input file generation."""

from pyradtran.core.input_builder import build_input_text
from pyradtran.models.aerosol import AerosolConfig
from pyradtran.models.atmosphere import AtmosphereConfig
from pyradtran.models.output import OutputConfig
from pyradtran.models.solver import SolverConfig
from pyradtran.models.source import SourceConfig
from pyradtran.models.surface import SurfaceConfig
from pyradtran.models.wavelength import WavelengthConfig


def test_basic_solar_scene():
    atmosphere = AtmosphereConfig(profile="us")
    source = SourceConfig(source="solar", sza=30.0)
    wavelength = WavelengthConfig(wavelength_min=250.0, wavelength_max=1200.0)
    solver = SolverConfig(method="disort", streams=16)
    output = OutputConfig(quantities=["lambda", "edir"])
    surface = SurfaceConfig(albedo=0.2)
    text = build_input_text(
        atmosphere=atmosphere, source=source, wavelength=wavelength,
        solver=solver, output=output, surface=surface,
    )
    lines = text.strip().split("\n")
    assert "atmosphere_file US-standard" in lines
    assert "source solar" in lines
    assert "sza 30.0" in lines
    assert "wavelength 250.0 1200.0" in lines
    assert "rte_solver disort" in lines
    assert "number_of_streams 16" in lines
    assert "output_user lambda edir" in lines
    assert "albedo 0.2" in lines


def test_with_raw_keywords():
    text = build_input_text(
        atmosphere=AtmosphereConfig(profile="ms"),
        source=SourceConfig(source="solar", sza=60.0),
        wavelength=WavelengthConfig(wavelength_min=300.0, wavelength_max=2500.0),
        solver=SolverConfig(method="disort", streams=8),
        output=OutputConfig(quiet=True),
        raw_keywords=[("verbose", ""), ("mc_backward", "")],
    )
    lines = text.strip().split("\n")
    assert "verbose" in lines
    assert "mc_backward" in lines


def test_with_aerosol():
    text = build_input_text(
        atmosphere=AtmosphereConfig(profile="us"),
        source=SourceConfig(source="solar", sza=45.0),
        wavelength=WavelengthConfig(wavelength_min=300.0, wavelength_max=2500.0),
        solver=SolverConfig(method="disort", streams=16),
        output=OutputConfig(quiet=True),
        aerosol=AerosolConfig(default=True, angstrom_alpha=1.3, angstrom_beta=0.08),
    )
    lines = text.strip().split("\n")
    assert "aerosol_default" in lines
    assert "aerosol_angstrom 1.3 0.08" in lines


def test_with_data_files_path():
    text = build_input_text(
        atmosphere=AtmosphereConfig(profile="us"),
        source=SourceConfig(source="solar", sza=30.0),
        wavelength=WavelengthConfig(wavelength_min=250.0, wavelength_max=1200.0),
        solver=SolverConfig(method="disort", streams=16),
        output=OutputConfig(quiet=True),
        data_files_path="/usr/local/share/libRadtran/data",
    )
    lines = text.strip().split("\n")
    assert "data_files_path /usr/local/share/libRadtran/data" in lines
