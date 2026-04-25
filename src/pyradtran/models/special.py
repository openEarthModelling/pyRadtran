"""Special options model for scattering/absorption control and file includes.

Maps to uvspec keywords: no_absorption, no_scattering, no_scattering mol, include.

Reference: libRadtran src/uvspec_lex.l
"""

from __future__ import annotations

from pydantic import Field, model_validator

from pyradtran.models.base import UvspecOption


class SpecialConfig(UvspecOption):
    """Special options: scattering/absorption toggles and include files.

    Attributes:
        no_absorption: Disable all absorption.
        no_scattering: Disable all scattering.
        no_scattering_mol: Disable molecular scattering only.
        include_files: Paths to additional input files to include (Phase 0).
    """

    no_absorption: bool = False
    no_scattering: bool = False
    no_scattering_mol: bool = False
    include_files: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_scattering(self) -> SpecialConfig:
        if self.no_scattering and self.no_scattering_mol:
            raise ValueError("no_scattering and no_scattering_mol are mutually exclusive")
        return self

    def to_uvspec_lines(self) -> list[str]:
        lines: list[str] = []
        if self.no_absorption:
            lines.append("no_absorption")
        if self.no_scattering:
            lines.append("no_scattering")
        if self.no_scattering_mol:
            lines.append("no_scattering mol")
        return lines

    def _scattering_items(self) -> list[tuple[int, str]]:
        """Return (phase, line) for scattering/absorption options (Phase 4)."""
        return [(4, line) for line in self.to_uvspec_lines()]
