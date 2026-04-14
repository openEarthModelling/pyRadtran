"""Cloud configuration model (Phase 2).

Maps to uvspec keywords: ic_file, ic_properties, ic_habit, ic_habit_yang2013,
ic_modify, wc_file, wc_properties, wc_modify, cloudcover, cloud_overlap.

Reference: libRadtran src/uvspec_lex.l (cloud options)
"""

from __future__ import annotations

from pydantic import Field, model_validator

from pyradtran.models.base import UvspecOption

_VALID_IC_PROPERTIES = frozenset({
    "fu", "echam4", "yang", "key", "baum", "baum_v36",
    "hey", "yang2013", "raytracing", "mie",
})

_VALID_IC_HABITS = frozenset({
    "solid-column", "hollow-column", "rough-aggregate",
    "rosette-4", "rosette-6", "plate", "droxtal", "dendrite", "ghm",
})

_VALID_YANG2013_HABITS = frozenset({
    "column_8elements", "droxtal", "hollow_bullet_rosette",
    "hollow_column", "plate", "plate_10elements", "plate_5elements",
    "solid_bullet_rosette", "solid_column",
})

_VALID_WC_PROPERTIES = frozenset({"hu", "echam4", "mie"})

_VALID_CLOUD_OVERLAP = frozenset({
    "max", "maxrand", "off", "rand",
})

_VALID_MODIFY_VARIABLES = frozenset({"gg", "ssa", "tau", "tau550"})
_VALID_MODIFY_ACTIONS = frozenset({"scale", "set"})


class CloudModifyEntry(UvspecOption):
    """A single cloud modify directive (wc_modify or ic_modify)."""

    model_config = {"extra": "forbid", "frozen": True, "populate_by_name": True}

    variable: str
    action: str
    value: float

    @model_validator(mode="after")
    def validate_entry(self) -> CloudModifyEntry:
        if self.variable not in _VALID_MODIFY_VARIABLES:
            raise ValueError(
                f"Invalid cloud modify variable '{self.variable}'. "
                f"Valid: {sorted(_VALID_MODIFY_VARIABLES)}"
            )
        if self.action not in _VALID_MODIFY_ACTIONS:
            raise ValueError(
                f"Invalid cloud modify action '{self.action}'. "
                f"Valid: {sorted(_VALID_MODIFY_ACTIONS)}"
            )
        return self


class CloudConfig(UvspecOption):
    """Cloud configuration for water and ice clouds.

    Attributes:
        ic_properties: Ice cloud optical property parameterization.
        ic_file: Tuple of (dimension, path) for ice cloud external file.
        ic_habit: Ice crystal habit type.
        ic_habit_roughness: Roughness for yang2013 ("smooth", "moderate", "severe").
        ic_modify: List of ice cloud modification directives.
        wc_properties: Water cloud optical property parameterization.
        wc_file: Tuple of (dimension, path) for water cloud external file.
        wc_modify: List of water cloud modification directives.
        cloud_cover_type: Cloud type for cloud cover ("ic" or "wc").
        cloud_cover: Cloud cover fraction [0, 1].
        cloud_overlap: Cloud overlap method.
        interpolate: Append 'interpolate' to ic/wc_properties for spectral mode.
    """

    ic_properties: str | None = None
    ic_file: tuple[str, str] | None = None
    ic_habit: str | None = None
    ic_habit_roughness: str | None = None
    ic_modify: list[CloudModifyEntry] = Field(default_factory=list)
    wc_properties: str | None = None
    wc_file: tuple[str, str] | None = None
    wc_modify: list[CloudModifyEntry] = Field(default_factory=list)
    modify: list[CloudModifyEntry] = Field(default_factory=list)
    cloud_cover_type: str | None = None
    cloud_cover: float | None = Field(default=None, ge=0.0, le=1.0)
    cloud_overlap: str | None = None
    interpolate: bool = False

    @model_validator(mode="after")
    def validate_cloud(self) -> CloudConfig:
        if self.ic_properties is not None and self.ic_properties not in _VALID_IC_PROPERTIES:
            raise ValueError(
                f"Invalid ic_properties '{self.ic_properties}'. "
                f"Valid: {sorted(_VALID_IC_PROPERTIES)}"
            )
        if self.ic_habit is not None and self.ic_habit not in _VALID_IC_HABITS:
            if self.ic_habit not in _VALID_YANG2013_HABITS:
                raise ValueError(
                    f"Invalid ic_habit '{self.ic_habit}'. "
                    f"Valid standard: {sorted(_VALID_IC_HABITS)}. "
                    f"Valid yang2013: {sorted(_VALID_YANG2013_HABITS)}."
                )
        if self.wc_properties is not None and self.wc_properties not in _VALID_WC_PROPERTIES:
            raise ValueError(
                f"Invalid wc_properties '{self.wc_properties}'. "
                f"Valid: {sorted(_VALID_WC_PROPERTIES)}"
            )
        if self.cloud_overlap is not None and self.cloud_overlap not in _VALID_CLOUD_OVERLAP:
            raise ValueError(
                f"Invalid cloud_overlap '{self.cloud_overlap}'. "
                f"Valid: {sorted(_VALID_CLOUD_OVERLAP)}"
            )
        if self.cloud_cover is not None and self.cloud_cover_type is not None:
            if self.cloud_cover_type not in ("ic", "wc"):
                raise ValueError(
                    f"Invalid cloud_cover_type '{self.cloud_cover_type}'. Valid: ic, wc"
                )
        return self

    def to_uvspec_lines(self) -> list[str]:
        lines: list[str] = []
        if self.ic_file is not None:
            dim, path = self.ic_file
            lines.append(f"ic_file {dim} {path}")
        if self.ic_properties is not None:
            suffix = " interpolate" if self.interpolate else ""
            lines.append(f"ic_properties {self.ic_properties}{suffix}")
        if self.ic_habit is not None:
            lines.append(f"ic_habit {self.ic_habit}")
            if self.ic_habit_roughness is not None:
                lines.append(
                    f"ic_habit_yang2013 {self.ic_habit} {self.ic_habit_roughness}"
                )
        for entry in self.ic_modify:
            lines.append(f"ic_modify {entry.variable} {entry.action} {entry.value}")
        if self.wc_file is not None:
            dim, path = self.wc_file
            lines.append(f"wc_file {dim} {path}")
        if self.wc_properties is not None:
            suffix = " interpolate" if self.interpolate else ""
            lines.append(f"wc_properties {self.wc_properties}{suffix}")
        for entry in self.wc_modify + self.modify:
            lines.append(f"wc_modify {entry.variable} {entry.action} {entry.value}")
        if self.cloud_cover is not None:
            if self.cloud_cover_type is not None:
                lines.append(f"cloudcover {self.cloud_cover_type} {self.cloud_cover}")
            else:
                lines.append(f"cloudcover wc {self.cloud_cover}")
        if self.cloud_overlap is not None:
            lines.append(f"cloud_overlap {self.cloud_overlap}")
        return lines