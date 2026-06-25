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

from pydantic import BaseModel, Field, model_validator

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


class OpacPreset(BaseModel):
    """Factory: fold an OPAC preset mixture into LEGO :class:`PlacedBlock` pieces.

    Each species with nonzero mass in the preset profile becomes one
    ``MieSpecies`` (OPAC refractive index + OPAC lognormal at ``rh_pct``) using
    the real Mie phase function, placed via its preset mass column. The pieces
    drop into ``CompositeAerosol(pieces=...)`` -- the same path as bulk/Mie
    blocks. No precomputed OPAC tables are read; libRadtran ships only the
    ingredients (refractive index, size distribution, mass profile).
    """

    model_config = {"extra": "forbid", "frozen": True, "populate_by_name": True}

    name: OpacPresetName
    rh_pct: float = 50.0
    species_names: list[str] | None = None
    data_path: str | None = None
    n_legendre: int = 32

    @model_validator(mode="after")
    def validate_species(self) -> OpacPreset:
        if self.species_names:
            _validate_opac_species_names(self.species_names)
        return self

    def to_placed_blocks(self) -> list:
        """Return one :class:`~pyradtran.models.blocks.PlacedBlock` per preset
        species with nonzero mass."""
        import numpy as np

        from pyradtran.models.aerosol_composite import MieSpecies
        from pyradtran.models.blocks import PlacedBlock, TabulatedProfile
        from pyradtran.optics.opac import (
            read_opac_preset_profile,
            read_opac_refractive_index,
            read_opac_size_distribution,
        )

        profile = read_opac_preset_profile(self.name.value, data_path=self.data_path)
        chosen = self.species_names if self.species_names is not None else list(profile)
        blocks: list[PlacedBlock] = []
        for sp in chosen:
            if sp not in profile:
                continue
            z_km, mass_g_m3 = profile[sp]
            mass = np.asarray(mass_g_m3, dtype=float)
            if not np.any(mass > 0):
                continue
            ri = read_opac_refractive_index(sp, self.rh_pct, data_path=self.data_path)
            sd, rho_kg_m3 = read_opac_size_distribution(sp, self.rh_pct, data_path=self.data_path)
            mie = MieSpecies(
                refractive_index=ri,
                size_distribution=sd,
                particle_density_kg_m3=rho_kg_m3,
                name=f"OPAC:{sp}",
                phase_function="mie",
            )
            blocks.append(
                PlacedBlock(
                    block=mie,
                    profile=TabulatedProfile(
                        z_km=tuple(float(z) for z in np.asarray(z_km).tolist()),
                        kg_m3=tuple(float(v) for v in (mass * 1e-3).tolist()),  # g/m^3 -> kg/m^3
                    ),
                )
            )
        if not blocks:
            raise ValueError(f"OPAC preset {self.name.value!r} has no species with nonzero mass")
        return blocks

    def to_composite(self, wavelength_grid_um, output_dir=None):
        """Wrap ``to_placed_blocks()`` in a
        :class:`~pyradtran.models.aerosol_composite.CompositeAerosol` on the
        preset grid."""
        import numpy as np

        from pyradtran.models.aerosol_composite import CompositeAerosol
        from pyradtran.optics.opac import read_opac_preset_profile

        blocks = self.to_placed_blocks()
        profile = read_opac_preset_profile(self.name.value, data_path=self.data_path)
        any_z = next(iter(profile.values()))[0]
        altitude_desc = sorted((float(z) for z in np.asarray(any_z).tolist()), reverse=True)
        return CompositeAerosol(
            pieces=blocks,
            wavelength_grid_um=list(wavelength_grid_um),
            altitude_grid_km=altitude_desc,
            n_legendre=self.n_legendre,
            output_dir=output_dir,
        )


class OpacCustom(BaseModel):
    """Factory: fold a user OPAC species profile file into :class:`PlacedBlock` pieces.

    Like :class:`OpacPreset` but the mass-concentration profile is a user-supplied
    ASCII file (same format as ``standard_aerosol_files``).
    """

    model_config = {"extra": "forbid", "frozen": True, "populate_by_name": True}

    species_file: str = Field(min_length=1)
    rh_pct: float = 50.0
    species_names: list[str] | None = None
    data_path: str | None = None
    n_legendre: int = 32

    @model_validator(mode="after")
    def validate_species(self) -> OpacCustom:
        if self.species_names:
            _validate_opac_species_names(self.species_names)
        return self

    def to_placed_blocks(self) -> list:
        """Return one :class:`~pyradtran.models.blocks.PlacedBlock` per profile
        species with nonzero mass."""
        import numpy as np

        from pyradtran.models.aerosol_composite import MieSpecies
        from pyradtran.models.blocks import PlacedBlock, TabulatedProfile
        from pyradtran.optics.opac import (
            read_opac_profile_file,
            read_opac_refractive_index,
            read_opac_size_distribution,
        )

        profile = read_opac_profile_file(self.species_file)
        chosen = self.species_names if self.species_names is not None else list(profile)
        blocks: list[PlacedBlock] = []
        for sp in chosen:
            if sp not in profile:
                continue
            z_km, mass_g_m3 = profile[sp]
            mass = np.asarray(mass_g_m3, dtype=float)
            if not np.any(mass > 0):
                continue
            ri = read_opac_refractive_index(sp, self.rh_pct, data_path=self.data_path)
            sd, rho_kg_m3 = read_opac_size_distribution(sp, self.rh_pct, data_path=self.data_path)
            mie = MieSpecies(
                refractive_index=ri,
                size_distribution=sd,
                particle_density_kg_m3=rho_kg_m3,
                name=f"OPAC:{sp}",
                phase_function="mie",
            )
            blocks.append(
                PlacedBlock(
                    block=mie,
                    profile=TabulatedProfile(
                        z_km=tuple(float(z) for z in np.asarray(z_km).tolist()),
                        kg_m3=tuple(float(v) for v in (mass * 1e-3).tolist()),
                    ),
                )
            )
        if not blocks:
            raise ValueError("OPAC custom profile has no species with nonzero mass")
        return blocks

    def to_composite(self, wavelength_grid_um, output_dir=None):
        """Wrap ``to_placed_blocks()`` in a
        :class:`~pyradtran.models.aerosol_composite.CompositeAerosol` on the
        profile grid."""
        import numpy as np

        from pyradtran.models.aerosol_composite import CompositeAerosol
        from pyradtran.optics.opac import read_opac_profile_file

        blocks = self.to_placed_blocks()
        profile = read_opac_profile_file(self.species_file)
        any_z = next(iter(profile.values()))[0]
        altitude_desc = sorted((float(z) for z in np.asarray(any_z).tolist()), reverse=True)
        return CompositeAerosol(
            pieces=blocks,
            wavelength_grid_um=list(wavelength_grid_um),
            altitude_grid_km=altitude_desc,
            n_legendre=self.n_legendre,
            output_dir=output_dir,
        )


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
