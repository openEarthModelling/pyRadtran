"""Surface configuration model (Phase 1: basic albedo only).

Maps to uvspec keywords: albedo, albedo_file, sur_temperature.

Reference: libRadtran src_py/surface_options.py
"""

from __future__ import annotations

from pyradtran.models.base import UvspecOption
from pydantic import Field, model_validator


class SurfaceConfig(UvspecOption):
    """Surface reflection and temperature configuration.

    Attributes:
        albedo: Constant Lambertian albedo [0, 1]. Mutually exclusive with albedo_file.
        albedo_file: Path to wavelength-dependent albedo file.
        sur_temperature: Surface temperature in Kelvin (for thermal IR).
    """

    albedo: float | None = Field(default=None, ge=0.0, le=1.0)
    albedo_file: str | None = None
    sur_temperature: float | None = None

    @model_validator(mode="after")
    def check_mutual_exclusion(self) -> SurfaceConfig:
        if self.albedo is not None and self.albedo_file is not None:
            raise ValueError("Cannot set both albedo and albedo_file -- choose one")
        return self

    def to_uvspec_lines(self) -> list[str]:
        lines: list[str] = []
        if self.albedo is not None:
            lines.append(f"albedo {self.albedo}")
        elif self.albedo_file is not None:
            lines.append(f"albedo_file {self.albedo_file}")
        if self.sur_temperature is not None:
            lines.append(f"sur_temperature {self.sur_temperature}")
        return lines
