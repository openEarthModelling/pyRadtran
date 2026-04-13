"""Output configuration model.

Maps to uvspec keywords: output_user, output_quantity, output_process,
output_format, quiet, verbose, zout, output_file.

Reference: libRadtran src_py/output_options.py
"""

from __future__ import annotations

from pyradtran.models.base import UvspecOption
from pydantic import Field, model_validator

VALID_OUTPUT_FORMATS = frozenset({"ascii", "flexstor", "netcdf", "sat_picture"})


class OutputConfig(UvspecOption):
    """Output format and content configuration.

    Attributes:
        quantities: Output column quantities (e.g. ["lambda", "edir", "edn", "eup"]).
        quantity: Output quantity type (transmittance, reflectivity, brightness).
        process: Output processing (e.g. "pseudoplanar", "planar").
        format: Output file format -- "netcdf" (default), "ascii", "flexstor".
        quiet: Suppress uvspec stdout messages. Default: True.
        verbose: Enable verbose uvspec output.
        zout: Output altitudes in km above ground level.
        output_file: Path for output file (overrides auto-generated name).
    """

    quantities: list[str] = Field(default_factory=list)
    quantity: str | None = None
    process: str | None = None
    format: str = Field(default="netcdf")
    quiet: bool = True
    verbose: bool = False
    zout: list[float] = Field(default_factory=list)
    output_file: str | None = None

    @model_validator(mode="after")
    def validate_output(self) -> OutputConfig:
        if self.format not in VALID_OUTPUT_FORMATS:
            raise ValueError(
                f"Unknown output format '{self.format}'. Valid: {sorted(VALID_OUTPUT_FORMATS)}"
            )
        if self.quiet and self.verbose:
            raise ValueError("Cannot set both quiet=True and verbose=True")
        return self

    def to_uvspec_lines(self) -> list[str]:
        lines: list[str] = []
        if self.quantities:
            lines.append(f"output_user {' '.join(self.quantities)}")
        if self.quantity is not None:
            lines.append(f"output_quantity {self.quantity}")
        if self.process is not None:
            lines.append(f"output_process {self.process}")
        lines.append(f"output_format {self.format}")
        if self.quiet:
            lines.append("quiet")
        elif self.verbose:
            lines.append("verbose")
        if self.zout:
            zout_str = " ".join(str(z) for z in self.zout)
            lines.append(f"zout {zout_str}")
        if self.output_file is not None:
            lines.append(f"output_file {self.output_file}")
        return lines
