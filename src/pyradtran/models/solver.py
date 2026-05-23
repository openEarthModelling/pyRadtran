"""Solver configuration model.

Maps to uvspec keywords: rte_solver, number_of_streams, pseudospherical, deltam.

Reference: libRadtran src_py/solver_options.py
"""

from __future__ import annotations

from pydantic import Field, model_validator

from pyradtran.models.base import UvspecOption

VALID_SOLVERS = frozenset(
    {
        "disort",
        "twostr",
        "mystic",
        "rodents",
        "sslidar",
        "null",
        "sdisort",
        "fdisort1",
        "fdisort2",
        "sos",
        "ftwostr",
        "montecarlo",
        "tzs",
        "sssi",
        "sss",
        "twostrebe",
        "schwarzschild",
        "twomaxrnd",
        "twomaxrnd3C",
        "dynamic_twostream",
        "dynamic_tenstream",
    }
)

VALID_HEAT_UNITS = frozenset({"K_per_day", "W_per_m2_and_dz", "W_per_m3"})

_VALID_DISORT_INTCOR = frozenset({"phase", "moments", "off"})


class SolverConfig(UvspecOption):
    """Radiative transfer solver configuration.

    Attributes:
        method: Solver name (disort, twostr, mystic, etc.).
        streams: Number of streams for discrete ordinates solvers. Default: 6.
        pseudospherical: Enable pseudo-spherical correction (disort/twostr only).
        deltam: Enable delta-M scaling.
        dynamic_iterations: Number of iterations for dynamic solvers.
        dynamic_history: Enable history tracking for dynamic solvers.
        dynamic_heat_unit: Heat unit for dynamic solvers.
    """

    method: str
    streams: int = Field(default=6, ge=1)
    pseudospherical: bool = False
    deltam: bool = False
    dynamic_iterations: int | None = Field(default=None, ge=0)
    dynamic_history: bool = False
    dynamic_heat_unit: str | None = None
    disort_intcor: str | None = None
    disort_spherical_albedo: bool = False
    schwarzschild_streams: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_solver(self) -> SolverConfig:
        if self.method not in VALID_SOLVERS:
            raise ValueError(f"Unknown solver '{self.method}'. Valid: {sorted(VALID_SOLVERS)}")
        if self.dynamic_heat_unit is not None and self.dynamic_heat_unit not in VALID_HEAT_UNITS:
            raise ValueError(
                f"Invalid dynamic_heat_unit '{self.dynamic_heat_unit}'. "
                f"Valid: {sorted(VALID_HEAT_UNITS)}"
            )
        if self.disort_intcor is not None and self.disort_intcor not in _VALID_DISORT_INTCOR:
            raise ValueError(
                f"Invalid disort_intcor '{self.disort_intcor}'. "
                f"Valid: {sorted(_VALID_DISORT_INTCOR)}"
            )
        return self

    def to_uvspec_lines(self) -> list[str]:
        lines: list[str] = []
        lines.append(f"rte_solver {self.method}")
        lines.append(f"number_of_streams {self.streams}")
        if self.pseudospherical:
            lines.append("pseudospherical")
        if self.deltam:
            lines.append("deltam")
        if self.disort_intcor is not None:
            lines.append(f"disort_intcor {self.disort_intcor}")
        if self.disort_spherical_albedo:
            lines.append("disort_spherical_albedo")
        if self.schwarzschild_streams is not None:
            lines.append(f"schwarzschild_streams {self.schwarzschild_streams}")
        if self.method.startswith("dynamic_"):
            prefix = self.method  # e.g. "dynamic_twostream" or "dynamic_tenstream"
            if self.dynamic_iterations is not None:
                lines.append(f"{prefix}_iterations {self.dynamic_iterations}")
            if self.dynamic_history:
                lines.append(f"{prefix}_history")
            if self.dynamic_heat_unit is not None:
                lines.append(f"{prefix}_heat_unit {self.dynamic_heat_unit}")
        return lines

    def to_uvspec_items(self) -> list[tuple[int, str]]:
        phase = 8
        return [(phase, line) for line in self.to_uvspec_lines()]
