"""Aerosol configuration model.

Maps to uvspec keywords: aerosol_default, aerosol_angstrom,
aerosol_set_tau_at_wvl, aerosol_haze, aerosol_vulcan,
aerosol_season, aerosol_visibility, aerosol_file,
aerosol_modify, aerosol_refrac_index, aerosol_refrac_file,
aerosol_sizedist_file, aerosol_species_file, aerosol_species_library.

Reference: libRadtran src/uvspec_lex.l (aerosol options)
"""

from __future__ import annotations

from pydantic import Field, model_validator

from pyradtran.models.base import UvspecOption

_VALID_FILE_TYPES = frozenset({"gg", "ssa", "tau", "explicit", "moments"})
_VALID_MODIFY_VARIABLES = frozenset({"gg", "ssa", "tau", "tau550"})
_VALID_MODIFY_ACTIONS = frozenset({"scale", "set"})


class AerosolModifyEntry(UvspecOption):
    """A single aerosol_modify directive.

    Attributes:
        variable: Property to modify (gg, ssa, tau, tau550).
        action: How to modify (scale or set).
        value: Numeric value.
    """

    model_config = {"extra": "forbid", "frozen": True, "populate_by_name": True}

    variable: str
    action: str
    value: float

    @model_validator(mode="after")
    def validate_entry(self) -> AerosolModifyEntry:
        if self.variable not in _VALID_MODIFY_VARIABLES:
            raise ValueError(
                f"Invalid aerosol_modify variable '{self.variable}'. "
                f"Valid: {sorted(_VALID_MODIFY_VARIABLES)}"
            )
        if self.action not in _VALID_MODIFY_ACTIONS:
            raise ValueError(
                f"Invalid aerosol_modify action '{self.action}'. "
                f"Valid: {sorted(_VALID_MODIFY_ACTIONS)}"
            )
        return self

    def to_uvspec_line(self) -> str:
        return f"aerosol_modify {self.variable} {self.action} {self.value}"


class AerosolConfig(UvspecOption):
    """Aerosol configuration.

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
        modify: List of aerosol modification directives.
        refrac_index: Tuple of (real, imaginary) refractive index parts.
        refrac_file: Path to refractive index file (wavelength-dependent).
        sizedist_file: Path to size distribution file.
        species_file: Path to aerosol species profile file.
        species_names: Optional list of species names for species_file.
        species_library: Path to aerosol species library directory.
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
    modify: list[AerosolModifyEntry] = Field(default_factory=list)
    refrac_index: tuple[float, float] | None = None
    refrac_file: str | None = None
    sizedist_file: str | None = None
    species_file: str | None = None
    species_names: list[str] | None = None
    species_library: str | None = None

    @model_validator(mode="after")
    def validate_aerosol(self) -> AerosolConfig:
        if self.angstrom_alpha is not None and not self.default:
            raise ValueError("angstrom_alpha requires default=True")
        if self.angstrom_beta is not None and not self.default:
            raise ValueError("angstrom_beta requires default=True")
        if self.angstrom_alpha is not None and self.angstrom_beta is None:
            raise ValueError("angstrom_beta must be set when angstrom_alpha is set")
        if self.angstrom_beta is not None and self.angstrom_alpha is None:
            raise ValueError("angstrom_alpha must be set when angstrom_beta is set")
        if self.file is not None:
            file_type = self.file[0]
            if file_type not in _VALID_FILE_TYPES:
                raise ValueError(
                    f"Unknown aerosol file type '{file_type}'. "
                    f"Valid: {sorted(_VALID_FILE_TYPES)}"
                )
        if self.refrac_index is not None and self.refrac_file is not None:
            raise ValueError("Cannot set both refrac_index and refrac_file")
        if self.species_names is not None and self.species_file is None:
            raise ValueError("species_names requires species_file")
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
        if self.refrac_index is not None:
            real, imag = self.refrac_index
            lines.append(f"aerosol_refrac_index {real} {imag}")
        if self.refrac_file is not None:
            lines.append(f"aerosol_refrac_file {self.refrac_file}")
        if self.sizedist_file is not None:
            lines.append(f"aerosol_sizedist_file {self.sizedist_file}")
        if self.species_file is not None:
            if self.species_names:
                names = " ".join(self.species_names)
                lines.append(f"aerosol_species_file {self.species_file} {names}")
            else:
                lines.append(f"aerosol_species_file {self.species_file}")
        if self.species_library is not None:
            lines.append(f"aerosol_species_library {self.species_library}")
        for entry in self.modify:
            lines.append(entry.to_uvspec_line())
        return lines