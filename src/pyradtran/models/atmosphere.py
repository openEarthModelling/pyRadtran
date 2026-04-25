"""Atmosphere configuration model.

Maps to uvspec keywords: atmosphere_file, altitude, pressure,
mol_modify, mol_abs_param.

Reference: libRadtran src_py/molecular_options.py, src_py/surface_options.py
"""

from __future__ import annotations

from pydantic import Field, model_validator

from pyradtran.models.base import UvspecOption

# Shorthand aliases
PROFILE_ALIASES: dict[str, str] = {
    "sw": "subarctic_winter",
    "ss": "subarctic_summer",
    "ms": "midlatitude_summer",
    "mw": "midlatitude_winter",
    "tp": "tropics",
    "us": "US-standard",
}


class AtmosphereConfig(UvspecOption):
    """Atmospheric profile and molecular absorption configuration.

    Attributes:
        profile: Named AFGL atmosphere or path to custom atmosphere file.
            Shorthands: us, ms, mw, tp, ss, sw.
            Full names: US-standard, midlatitude_summer, midlatitude_winter,
            tropics, subarctic_summer, subarctic_winter.
        altitude: Surface altitude above sea level in km. Default: 0.0.
        pressure: Surface pressure in hPa. Scales pressure, air, O2, CO2 profiles.
        mol_modify: List of (species, column_value, unit) tuples.
            Species: O3, O2, H2O, CO2, NO2, BRO, OCLO, HCHO, O4, SO2, CH4, N2O, CO, N2.
            Units: DU, CM_2, MM.
        mol_abs_param: Molecular absorption parameterization scheme string.
            Default is reptran coarse (handled by uvspec internally).
        atm_z_grid: Custom altitude grid in km.
        radiosonde: Use radiosonde profile data.
        radiosonde_levels_only: Use only radiosonde levels (requires radiosonde=True).
        mol_file: List of molecular VMR profile files, each a dict with
            keys "species", "file", and optional "unit".
        mol_tau_file: Molecular optical thickness file as (kind, filepath) tuple.
        rayleigh_depol: Rayleigh scattering depolarization factor.
        raman: Enable Raman scattering.
    """

    profile: str = Field(min_length=1)
    altitude: float = Field(default=0.0, ge=-1e6, le=1e6)
    pressure: float | None = Field(default=None, ge=0, le=1e6)
    mol_modify: list[tuple[str, float, str]] = Field(default_factory=list)
    mol_abs_param: str | None = None
    crs_model: dict[str, str] | list[dict[str, str]] | None = None
    atm_z_grid: list[float] | None = Field(default=None)
    radiosonde: bool = False
    radiosonde_levels_only: bool = False
    mol_file: list[dict[str, str]] | None = None
    mol_tau_file: tuple[str, str] | None = None
    rayleigh_depol: float | None = Field(default=None, ge=0.0, le=1.0)
    raman: bool = False

    _VALID_MOL_SPECIES = frozenset({
        "O3", "O2", "H2O", "CO2", "NO2", "BRO", "OCLO",
        "HCHO", "O4", "SO2", "CH4", "N2O", "CO", "N2",
    })
    _VALID_MOL_UNITS = frozenset({"DU", "CM_2", "MM", "precip_cm", "ppmv"})
    _VALID_CRS_SPECIES = frozenset({
        "rayleigh", "o3", "no2", "o4",
    })
    _VALID_CRS_MODELS = frozenset({
        "Bass_and_Paur", "Bodhaine", "Bodhaine29", "Bogumil", "Burrows",
        "Daumont", "Greenblatt", "Molina", "Nicolet", "Penndorf",
        "Serdyuchenko", "Thalman", "Vandaele",
    })

    @model_validator(mode="after")
    def validate_mol_modify_entries(self) -> AtmosphereConfig:
        for species, value, unit in self.mol_modify:
            if species not in self._VALID_MOL_SPECIES:
                raise ValueError(
                    f"Invalid mol_modify species '{species}'. "
                    f"Valid: {sorted(self._VALID_MOL_SPECIES)}"
                )
            if unit not in self._VALID_MOL_UNITS:
                raise ValueError(
                    f"Invalid mol_modify unit '{unit}'. Valid: {sorted(self._VALID_MOL_UNITS)}"
                )
            if value < 0 or value > 1e6:
                raise ValueError(
                    f"mol_modify column value {value} out of range [0, 1e6]"
                )
        return self

    @model_validator(mode="after")
    def validate_crs_model(self) -> AtmosphereConfig:
        if self.crs_model is not None:
            entries = self.crs_model if isinstance(self.crs_model, list) else [self.crs_model]
            for entry in entries:
                sp = entry.get("species", "")
                md = entry.get("model", "")
                if sp not in self._VALID_CRS_SPECIES:
                    raise ValueError(
                        f"Invalid crs_model species '{sp}'. "
                        f"Valid: {sorted(self._VALID_CRS_SPECIES)}"
                    )
                if md not in self._VALID_CRS_MODELS:
                    raise ValueError(
                        f"Invalid crs_model model '{md}'. "
                        f"Valid: {sorted(self._VALID_CRS_MODELS)}"
                    )
        return self

    @model_validator(mode="after")
    def validate_radiosonde(self) -> AtmosphereConfig:
        if self.radiosonde_levels_only and not self.radiosonde:
            raise ValueError("radiosonde_levels_only requires radiosonde=True")
        return self

    def _resolve_profile(self) -> str:
        name = self.profile.strip()
        return PROFILE_ALIASES.get(name, name)

    def to_uvspec_lines(self) -> list[str]:
        lines: list[str] = []
        profile = self._resolve_profile()
        lines.append(f"atmosphere_file {profile}")
        if self.altitude != 0.0:
            lines.append(f"altitude {self.altitude}")
        if self.pressure is not None:
            lines.append(f"pressure {self.pressure}")
        for species, value, unit in self.mol_modify:
            lines.append(f"mol_modify {species} {value} {unit}")
        if self.mol_abs_param is not None:
            lines.append(f"mol_abs_param {self.mol_abs_param}")
        if self.crs_model is not None:
            entries = self.crs_model if isinstance(self.crs_model, list) else [self.crs_model]
            for entry in entries:
                lines.append(f"crs_model {entry['species']} {entry['model']}")
        if self.atm_z_grid is not None:
            vals = " ".join(str(v) for v in self.atm_z_grid)
            lines.append(f"atm_z_grid {vals}")
        if self.radiosonde:
            lines.append("radiosonde")
        if self.radiosonde_levels_only:
            lines.append("radiosonde_levels_only")
        if self.mol_file is not None:
            for entry in self.mol_file:
                species = entry["species"]
                filepath = entry["file"]
                unit = entry.get("unit", "")
                if unit:
                    lines.append(f"mol_file {species} {filepath} {unit}")
                else:
                    lines.append(f"mol_file {species} {filepath}")
        if self.mol_tau_file is not None:
            kind, filepath = self.mol_tau_file
            lines.append(f"mol_tau_file {kind} {filepath}")
        if self.rayleigh_depol is not None:
            lines.append(f"rayleigh_depol {self.rayleigh_depol}")
        if self.raman:
            lines.append("raman")
        return lines

    def to_uvspec_items(self) -> list[tuple[int, str]]:
        phase = 1
        return [(phase, line) for line in self.to_uvspec_lines()]
