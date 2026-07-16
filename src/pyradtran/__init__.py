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
    run_with_opac_custom,
    run_with_opac_preset,
)
from pyradtran.core.postprocess import (
    BudgetResult,
    add_budget_vars,
    compute_budget,
    evaluate_blocks_on_grid,
    evaluate_composite_on_grid,
)
from pyradtran.core.runner import Runner, RunnerConfig
from pyradtran.data import DataResolver
from pyradtran.models import ThreeDConfig
from pyradtran.models.aerosol import (
    AerosolModel,
    AerosolModifyEntry,
    OpacCustom,
    OpacPreset,
    OpacPresetName,
)
from pyradtran.models.aerosol_composite import (
    BulkSpecies,
    CompositeAerosol,
    IntegrationConfig,
    LayerOptics,
    MieSpecies,
    RefractiveIndex,
    SizeDistribution,
    SpeciesOptics,
)
from pyradtran.models.blocks import (
    AerosolBlock,
    DirectLayerOpticsBlock,
    ExponentialProfile,
    MassProfile,
    Piece,
    PlacedBlock,
    TabulatedProfile,
    VerticalProfile,
    od_to_mass_profile,
)
from pyradtran.scene import Scene
from pyradtran.workflow import AttributionResult, compute_component_attribution

__all__ = [
    "Scene",
    "ThreeDConfig",
    "Runner",
    "RunnerConfig",
    "DataResolver",
    # Aerosol classes
    "AerosolModel",
    "AerosolModifyEntry",
    "OpacPreset",
    "OpacPresetName",
    "OpacCustom",
    # Composite aerosol classes
    "RefractiveIndex",
    "SizeDistribution",
    "IntegrationConfig",
    "MieSpecies",
    "BulkSpecies",
    "LayerOptics",
    "SpeciesOptics",
    "CompositeAerosol",
    # LEGO block interface
    "AerosolBlock",
    "Piece",
    "PlacedBlock",
    "DirectLayerOpticsBlock",
    "VerticalProfile",
    "MassProfile",
    "ExponentialProfile",
    "TabulatedProfile",
    "od_to_mass_profile",
    # Post-processing + workflow entry points
    "BudgetResult",
    "add_budget_vars",
    "compute_budget",
    "evaluate_composite_on_grid",
    "evaluate_blocks_on_grid",
    "compute_component_attribution",
    "AttributionResult",
    # Convenience functions
    "run_3d",
    "run_solar_transmittance",
    "run_thermal_brightness",
    "run_solar_radiance",
    "run_satellite",
    "run_with_opac_preset",
    "run_with_opac_custom",
    "run_cloudy_scene",
    "run_lidar",
    "run_polarized",
]
