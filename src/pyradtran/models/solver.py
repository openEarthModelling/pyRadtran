"""Solver configuration model.

Maps to uvspec keywords: rte_solver, number_of_streams, pseudospherical, deltam.

Reference: libRadtran src_py/solver_options.py
"""

from __future__ import annotations

from pydantic import Field, model_validator

from pyradtran.models.base import UvspecOption

VALID_SOLVERS = frozenset({
    "disort", "twostr", "mystic", "rodents", "sslidar",
    "null", "sdisort", "fdisort1", "fdisort2", "sos",
    "ftwostr", "montecarlo", "tzs", "sssi", "sss",
    "twostrebe", "schwarzschild", "twomaxrnd", "twomaxrnd3C",
})


class SolverConfig(UvspecOption):
    """Radiative transfer solver configuration.

    Attributes:
        method: Solver name (disort, twostr, mystic, etc.).
        streams: Number of streams for discrete ordinates solvers. Default: 6.
        pseudospherical: Enable pseudo-spherical correction (disort/twostr only).
        deltam: Enable delta-M scaling.
    """

    method: str
    streams: int = Field(default=6, ge=1)
    pseudospherical: bool = False
    deltam: bool = False

    @model_validator(mode="after")
    def validate_solver(self) -> SolverConfig:
        if self.method not in VALID_SOLVERS:
            raise ValueError(
                f"Unknown solver '{self.method}'. Valid: {sorted(VALID_SOLVERS)}"
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
        return lines
