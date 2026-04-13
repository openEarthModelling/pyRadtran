"""Aerosol configuration model (Phase 1: basic options).

Maps to uvspec keywords: aerosol_default, aerosol_angstrom,
aerosol_set_tau_at_wvl, aerosol_haze, aerosol_vulcan,
aerosol_season, aerosol_visibility, aerosol_file.

Reference: libRadtran src_py/aerosol_options.py
"""

from __future__ import annotations

from pyradtran.models.base import UvspecOption
from pydantic import Field, model_validator


class AerosolConfig(UvspecOption):
    """Basic aerosol configuration.

    Attributes:
        default: Enable default Shettle (1989) aerosol.
        angstrom_alpha: Angstrom alpha exponent (requires default=True).
        angstrom_beta: Angstrom beta coefficient in um^-1 (requires default=True).
        set_tau_at_wvl: Tuple of (wavelength_nm, tau) to set AOD.
        haze: Aerosol type in lower 2 km [1, 6].
        vulcan: Aerosol situation above 2 km [1, 4].
        season: Season [1, 2]. 1=spring-summer, 2=fall-winter.
        visibility: Horizontal visibility in km.
        file: Tuple of (file_type, file_path). Type: "gg", "ssa", "tau", "explicit", "moments".
    """

    default: bool = False
    angstrom_alpha: float | None = None
    angstrom_beta: float | None = None
    set_tau_at_wvl: tuple[float, float] | None = None
    haze: int | None = Field(default=None, ge=1, le=6)
    vulcan: int | None = Field(default=None, ge=1, le=4)
    season: int | None = Field(default=None, ge=1, le=2)
    visibility: float | None = Field(default=None, ge=0.0, le=1e6)
    file: tuple[str, str] | None = None

    _VALID_FILE_TYPES = frozenset({"gg", "ssa", "tau", "explicit", "moments"})

    @model_validator(mode="after")
    def validate_aerosol(self) -> AerosolConfig:
        if self.angstrom_alpha is not None and not self.default:
            raise ValueError("angstrom_alpha requires default=True")
        if self.angstrom_beta is not None and not self.default:
            raise ValueError("angstrom_beta requires default=True")
        if self.file is not None:
            file_type = self.file[0]
            if file_type not in self._VALID_FILE_TYPES:
                raise ValueError(
                    f"Unknown aerosol file type '{file_type}'. "
                    f"Valid: {sorted(self._VALID_FILE_TYPES)}"
                )
        return self

    def to_uvspec_lines(self) -> list[str]:
        lines: list[str] = []
        if self.default:
            lines.append("aerosol_default")
        if self.file is not None:
            lines.append(f"aerosol_file {self.file[0]} {self.file[1]}")
        elif self.haze is not None or self.vulcan is not None:
            if self.haze is not None:
                lines.append(f"aerosol_haze {self.haze}")
            if self.vulcan is not None:
                lines.append(f"aerosol_vulcan {self.vulcan}")
            if self.season is not None:
                lines.append(f"aerosol_season {self.season}")
            if self.visibility is not None:
                lines.append(f"aerosol_visibility {self.visibility}")
        if self.default and self.angstrom_alpha is not None:
            lines.append(f"aerosol_angstrom {self.angstrom_alpha} {self.angstrom_beta}")
        if self.set_tau_at_wvl is not None:
            wl, tau = self.set_tau_at_wvl
            lines.append(f"aerosol_set_tau_at_wvl {wl} {tau}")
        return lines
