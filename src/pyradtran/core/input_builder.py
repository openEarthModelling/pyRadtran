"""Serialize Scene configuration to uvspec input text."""

from __future__ import annotations

from pyradtran.models.advanced import AdvancedConfig
from pyradtran.models.aerosol import AerosolConfig
from pyradtran.models.atmosphere import AtmosphereConfig
from pyradtran.models.cloud import CloudConfig
from pyradtran.models.mc import McConfig
from pyradtran.models.output import OutputConfig
from pyradtran.models.solver import SolverConfig
from pyradtran.models.source import SourceConfig
from pyradtran.models.surface import SurfaceConfig
from pyradtran.models.wavelength import WavelengthConfig


def build_input_text(
    atmosphere: AtmosphereConfig,
    source: SourceConfig,
    wavelength: WavelengthConfig,
    solver: SolverConfig,
    output: OutputConfig,
    surface: SurfaceConfig | None = None,
    aerosol: AerosolConfig | None = None,
    cloud: CloudConfig | None = None,
    mc: McConfig | None = None,
    advanced: AdvancedConfig | None = None,
    raw_keywords: list[tuple[str, str]] | None = None,
    data_files_path: str | None = None,
) -> str:
    """Build a complete uvspec input string from configuration models."""
    lines: list[str] = []
    lines.extend(atmosphere.to_uvspec_lines())
    lines.extend(source.to_uvspec_lines())
    if aerosol is not None:
        lines.extend(aerosol.to_uvspec_lines())
    if cloud is not None:
        lines.extend(cloud.to_uvspec_lines())
    if surface is not None:
        lines.extend(surface.to_uvspec_lines())
    lines.extend(solver.to_uvspec_lines())
    lines.extend(wavelength.to_uvspec_lines())
    lines.extend(output.to_uvspec_lines())
    if mc is not None:
        lines.extend(mc.to_uvspec_lines())
    if advanced is not None:
        lines.extend(advanced.to_uvspec_lines())
    if data_files_path is not None:
        lines.append(f"data_files_path {data_files_path}")
    if raw_keywords:
        for key, value in raw_keywords:
            if value:
                lines.append(f"{key} {value}")
            else:
                lines.append(key)
    return "\n".join(lines) + "\n"
