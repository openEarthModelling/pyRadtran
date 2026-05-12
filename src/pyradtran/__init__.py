"""pyRadtran — A complete Python wrapper for libRadtran radiative transfer."""

from importlib.metadata import version

__version__ = version("pyradtran")

from pyradtran.convenience import (
    run_3d,
    run_cloudy_scene,
    run_lidar,
    run_polarized,
    run_satellite,
    run_solar_radiance,
    run_solar_transmittance,
    run_thermal_brightness,
    run_with_aerosol,
    run_with_opac_custom,
    run_with_opac_preset,
)
from pyradtran.core.runner import Runner, RunnerConfig
from pyradtran.models import ThreeDConfig
from pyradtran.models.aerosol import (
    AerosolModel,
    AerosolModifyEntry,
    ExternalAerosol,
    ExternalFile,
    OpacCustom,
    OpacPreset,
    OpacPresetName,
)
from pyradtran.models.aerosol_composite import (
    CompositeAerosol,
    IntegrationConfig,
    LayerOptics,
    LoadedSpecies,
    MieSpecies,
    OPACSpecies,
    ParticleOptics,
    PrecomputedSpecies,
    RefractiveIndex,
    SizeDistribution,
    SpeciesOptics,
)
from pyradtran.scene import Scene

__all__ = [
    "Scene",
    "ThreeDConfig",
    "Runner",
    "RunnerConfig",
    # Aerosol classes
    "AerosolModel",
    "AerosolModifyEntry",
    "OpacPreset",
    "OpacPresetName",
    "OpacCustom",
    "ExternalAerosol",
    "ExternalFile",
    # Composite aerosol classes
    "RefractiveIndex",
    "ParticleOptics",
    "SizeDistribution",
    "IntegrationConfig",
    "MieSpecies",
    "PrecomputedSpecies",
    "OPACSpecies",
    "LoadedSpecies",
    "LayerOptics",
    "SpeciesOptics",
    "CompositeAerosol",
    # Convenience functions
    "run_3d",
    "run_solar_transmittance",
    "run_thermal_brightness",
    "run_solar_radiance",
    "run_satellite",
    "run_with_aerosol",
    "run_with_opac_preset",
    "run_with_opac_custom",
    "run_cloudy_scene",
    "run_lidar",
    "run_polarized",
]
