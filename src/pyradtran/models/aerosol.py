"""Aerosol configuration models.

Provides a strict class hierarchy for aerosol configuration:

- OpacPreset: OPAC preset mixture profiles (continental_average, maritime_clean, etc.)
- OpacCustom: OPAC custom species profile files
- ExternalAerosol: External optical property files (explicit, gg, ssa, tau, moments)

Reference: libRadtran src/uvspec_lex.l (aerosol options)
Reference: Hess et al. (1998), Bull. Amer. Meteor. Soc., 79, 831-844
"""

from __future__ import annotations

from abc import abstractmethod
from enum import Enum

from pydantic import Field, model_validator

from pyradtran.models.base import UvspecOption

_VALID_FILE_TYPES = frozenset({"gg", "ssa", "tau", "explicit", "moments", "ref", "siz"})
_VALID_MODIFY_VARIABLES = frozenset({"gg", "ssa", "tau", "tau550"})
_VALID_MODIFY_ACTIONS = frozenset({"scale", "set"})
_VALID_OPAC_SPECIES = frozenset(
    {
        "inso",
        "waso",
        "soot",
        "ssam",
        "sscm",
        "minm",
        "miam",
        "micm",
        "mitr",
        "suso",
    }
)


def _validate_opac_species_names(names: list[str]) -> None:
    """Validate that species names are valid OPAC species."""
    invalid = set(names) - _VALID_OPAC_SPECIES
    if invalid:
        raise ValueError(
            f"Invalid OPAC species: {sorted(invalid)}. Valid: {sorted(_VALID_OPAC_SPECIES)}"
        )


class OpacPresetName(str, Enum):
    """OPAC preset mixture profile names.

    These correspond to files in data/aerosol/OPAC/standard_aerosol_files/.
    """

    CONTINENTAL_AVERAGE = "continental_average"
    CONTINENTAL_CLEAN = "continental_clean"
    CONTINENTAL_POLLUTED = "continental_polluted"
    URBAN = "urban"
    MARITIME_CLEAN = "maritime_clean"
    MARITIME_POLLUTED = "maritime_polluted"
    MARITIME_TROPICAL = "maritime_tropical"
    DESERT = "desert"
    DESERT_SPHEROIDS = "desert_spheroids"
    ANTARCTIC = "antarctic"


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

    def _format_line(self) -> str:
        return f"aerosol_modify {self.variable} {self.action} {self.value}"


class AerosolModel(UvspecOption):
    """Abstract base class for all aerosol configurations.

    Subclasses implement mode-specific ``to_uvspec_lines()``.
    Common capabilities (set_tau_at_wvl, king_byrne, modify) are handled here.
    """

    set_tau_at_wvl: tuple[float, float] | None = None
    king_byrne: tuple[float, float, float] | None = None
    modify: list[AerosolModifyEntry] = Field(default_factory=list)

    @abstractmethod
    def to_uvspec_lines(self) -> list[str]: ...

    def to_uvspec_items(self) -> list[tuple[int, str]]:
        phase = 5
        items = [(phase, line) for line in self.to_uvspec_lines()]
        if self.set_tau_at_wvl is not None:
            wl, tau = self.set_tau_at_wvl
            items.append((phase, f"aerosol_set_tau_at_wvl {wl} {tau}"))
        if self.king_byrne is not None:
            a0, a1, a2 = self.king_byrne
            items.append((phase, f"aerosol_king_byrne {a0} {a1} {a2}"))
        for entry in self.modify:
            items.append((phase, entry._format_line()))
        return items


class OpacPreset(AerosolModel):
    """OPAC preset mixture profile aerosol.

    Uses pre-defined aerosol species mixture profiles from the OPAC library.
    The ``name`` selects from 10 predefined mixture profiles.
    Optionally filter to specific species via ``species_names``.

    Attributes:
        name: Preset mixture profile name.
        library: OPAC library path or "OPAC" for uvspec default resolution.
        species_names: Optional species filter (e.g. ["inso", "soot"]).
    """

    name: OpacPresetName
    library: str = "OPAC"
    species_names: list[str] | None = None

    @model_validator(mode="after")
    def validate_species(self) -> OpacPreset:
        if self.species_names:
            _validate_opac_species_names(self.species_names)
        return self

    def to_uvspec_lines(self) -> list[str]:
        lines = [f"aerosol_species_library {self.library}"]
        if self.species_names:
            names = " ".join(self.species_names)
            lines.append(f"aerosol_species_file {self.name.value} {names}")
        else:
            lines.append(f"aerosol_species_file {self.name.value}")
        return lines


class OpacCustom(AerosolModel):
    """OPAC custom species profile aerosol.

    Uses a user-provided mass concentration profile file with the OPAC library.

    Attributes:
        species_file: Path to an ASCII profile file.
        library: OPAC library path or "OPAC" for default resolution.
        species_names: Optional species filter.
    """

    species_file: str = Field(min_length=1)
    species_names: list[str] | None = None
    library: str = "OPAC"

    @model_validator(mode="after")
    def validate_species(self) -> OpacCustom:
        if self.species_names:
            _validate_opac_species_names(self.species_names)
        return self

    def to_uvspec_lines(self) -> list[str]:
        lines = [f"aerosol_species_library {self.library}"]
        if self.species_names:
            names = " ".join(self.species_names)
            lines.append(f"aerosol_species_file {self.species_file} {names}")
        else:
            lines.append(f"aerosol_species_file {self.species_file}")
        return lines


class ExternalFile(AerosolModel):
    """External aerosol optical property files.

    Attributes:
        files: List of (file_type, file_path) tuples.
            Types: "gg", "ssa", "tau", "explicit", "moments", "ref", "siz".
    """

    files: list[tuple[str, str]] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_files(self) -> ExternalFile:
        for file_type, _ in self.files:
            if file_type not in _VALID_FILE_TYPES:
                raise ValueError(
                    f"Unknown aerosol file type '{file_type}'. Valid: {sorted(_VALID_FILE_TYPES)}"
                )
        return self

    def to_uvspec_lines(self) -> list[str]:
        return [f"aerosol_file {ft} {fp}" for ft, fp in self.files]


# Backwards-compatibility alias — remove after one release.
ExternalAerosol = ExternalFile
