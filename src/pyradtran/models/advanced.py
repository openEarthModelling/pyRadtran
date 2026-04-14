"""Advanced options model.

Maps to uvspec keywords: fluorescence, fluorescence_file, raman.

Phase 3 scope. Phase 4 may add 3D, satellite geometry, dynamic solvers, etc.

Reference: libRadtran src_py/surface_options.py, src_py/solver_options.py
"""

from __future__ import annotations

from pydantic import Field, model_validator

from pyradtran.models.base import UvspecOption


class AdvancedConfig(UvspecOption):
    """Advanced radiative transfer options.

    Attributes:
        fluorescence: Isotropic surface fluorescence emission (W/m^2/nm).
        fluorescence_file: Path to wavelength-dependent fluorescence file.
            Mutually exclusive with fluorescence.
        raman: Enable first-order rotational Raman scattering.
            Requires rte_solver disort.
        raman_variant: Raman variant -- "original" (default: standard).
    """

    fluorescence: float | None = Field(default=None, ge=0.0)
    fluorescence_file: str | None = None
    raman: bool = False
    raman_variant: str | None = None

    @model_validator(mode="after")
    def validate_advanced(self) -> AdvancedConfig:
        if self.fluorescence is not None and self.fluorescence_file is not None:
            raise ValueError("Cannot set both fluorescence and fluorescence_file")
        return self

    def to_uvspec_lines(self) -> list[str]:
        lines: list[str] = []
        if self.fluorescence is not None:
            lines.append(f"fluorescence {self.fluorescence}")
        elif self.fluorescence_file is not None:
            lines.append(f"fluorescence_file {self.fluorescence_file}")
        if self.raman:
            line = "raman"
            if self.raman_variant is not None:
                line += f" {self.raman_variant}"
            lines.append(line)
        return lines
