"""Serialize Scene configuration to uvspec input text."""

from __future__ import annotations

from pyradtran.models.advanced import AdvancedConfig
from pyradtran.models.aerosol import AerosolModel
from pyradtran.models.atmosphere import AtmosphereConfig
from pyradtran.models.cloud import CloudConfig
from pyradtran.models.mc import McConfig
from pyradtran.models.output import OutputConfig
from pyradtran.models.solver import SolverConfig
from pyradtran.models.source import SourceConfig
from pyradtran.models.special import SpecialConfig
from pyradtran.models.sslidar import SslidarConfig
from pyradtran.models.surface import SurfaceConfig
from pyradtran.models.three_d import ThreeDConfig
from pyradtran.models.wavelength import WavelengthConfig


def build_input_text(
    atmosphere: AtmosphereConfig,
    source: SourceConfig,
    wavelength: WavelengthConfig,
    solver: SolverConfig,
    output: OutputConfig,
    surface: SurfaceConfig | None = None,
    aerosol: AerosolModel | None = None,
    cloud: CloudConfig | None = None,
    mc: McConfig | None = None,
    sslidar: SslidarConfig | None = None,
    advanced: AdvancedConfig | None = None,
    three_d: ThreeDConfig | None = None,
    raw_keywords: list[tuple[str, str]] | None = None,
    data_files_path: str | None = None,
    special: SpecialConfig | None = None,
) -> str:
    """Build a complete uvspec input string from configuration models.

    Keywords are collected as (phase, line) pairs from each model's
    ``to_uvspec_items()`` method, then sorted by phase to guarantee
    correct uvspec keyword ordering.

    Phase assignments:
        0  - data_files_path, include directives
        1  - atmosphere
        2  - source
        3  - wavelength
        4  - scattering/absorption switches (no_absorption, no_scattering)
        5  - aerosol
        6  - cloud
        7  - surface
        8  - solver
        9  - output
        10 - mc, sslidar
        11 - three_d
        12 - advanced
        13 - raw_keywords (user-supplied, always last)
    """
    items: list[tuple[int, str]] = []

    # Phase 0: data_files_path and include directives
    if data_files_path is not None:
        items.append((0, f"data_files_path {data_files_path}"))
    if special is not None:
        for f in special.include_files:
            items.append((0, f"include {f}"))
        items.extend(special._scattering_items())

    # Collect items from all models (each assigns its own phase)
    items.extend(atmosphere.to_uvspec_items())
    items.extend(source.to_uvspec_items())
    if aerosol is not None:
        items.extend(aerosol.to_uvspec_items())
    if cloud is not None:
        items.extend(cloud.to_uvspec_items())
    if surface is not None:
        items.extend(surface.to_uvspec_items())
    items.extend(solver.to_uvspec_items())
    items.extend(wavelength.to_uvspec_items())
    items.extend(output.to_uvspec_items())
    if mc is not None:
        items.extend(mc.to_uvspec_items())
    if sslidar is not None:
        items.extend(sslidar.to_uvspec_items())
    if advanced is not None:
        items.extend(advanced.to_uvspec_items())
    if three_d is not None:
        items.extend(three_d.to_uvspec_items())

    # Phase 13: raw_keywords always last
    if raw_keywords:
        for key, value in raw_keywords:
            if value:
                items.append((13, f"{key} {value}"))
            else:
                items.append((13, key))

    # Sort by phase, preserving relative order within the same phase
    items.sort(key=lambda x: x[0])

    return "\n".join(line for _, line in items) + "\n"
