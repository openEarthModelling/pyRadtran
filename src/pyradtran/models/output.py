"""Output configuration model.

Maps to uvspec keywords: output_user, output_quantity, output_process,
output_format, quiet, verbose, zout, output_file.

Reference: libRadtran src_py/output_options.py
"""

from __future__ import annotations

from pydantic import Field, model_validator

from pyradtran.models.base import UvspecOption

VALID_OUTPUT_FORMATS = frozenset({"ascii", "flexstor", "netcdf", "sat_picture"})
VALID_HEATING_RATE_MODES = frozenset({"none", "local", "layer_fd", "layer_cd", "ipa3d"})
VALID_OUTPUT_PROCESSES = frozenset(
    {
        "integrate",
        "sum",
        "rgbraw",
        "rgb_raw",
        "rgb",
        "per_nm",
        "per_cm-1",
        "per_band",
    }
)


class OutputConfig(UvspecOption):
    """Output format and content configuration.

    Attributes:
        quantities: Output column quantities (e.g. ["lambda", "edir", "edn", "eup"]).
        quantity: Output quantity type (transmittance, reflectivity, brightness).
        process: Output processing (e.g. 'integrate', 'per_nm', 'sum').
        format: Output file format -- "netcdf" (default), "ascii", "flexstor".
        quiet: Suppress uvspec stdout messages. Default: True.
        verbose: Enable verbose uvspec output.
        zout: Output altitudes in km above ground level. Supports float values
            and special strings like "toa" and "boa".
        output_file: Path for output file (overrides auto-generated name).
        heating_rate: Heating rate calculation mode (e.g. 'local', 'layer_fd').
        write_optical_properties: Write optical properties to output file.
    """

    quantities: list[str] = Field(default_factory=list)
    quantity: str | None = None
    process: str | None = None
    format: str = Field(default="ascii")
    quiet: bool = True
    verbose: bool = False
    zout: list[float | str] = Field(default_factory=list)
    output_file: str | None = None
    heating_rate: str | None = None
    write_optical_properties: bool = False

    @model_validator(mode="after")
    def validate_output(self) -> OutputConfig:
        if self.format not in VALID_OUTPUT_FORMATS:
            raise ValueError(
                f"Unknown output format '{self.format}'. Valid: {sorted(VALID_OUTPUT_FORMATS)}"
            )
        if self.quiet and self.verbose:
            raise ValueError("Cannot set both quiet=True and verbose=True")
        if self.heating_rate is not None and self.heating_rate not in VALID_HEATING_RATE_MODES:
            raise ValueError(
                f"Unknown heating_rate "
                f"'{self.heating_rate}'. Valid: {sorted(VALID_HEATING_RATE_MODES)}"
            )
        if self.process is not None and self.process not in VALID_OUTPUT_PROCESSES:
            raise ValueError(
                f"Unknown output_process '{self.process}'. Valid: {sorted(VALID_OUTPUT_PROCESSES)}"
            )
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
        if self.heating_rate is not None:
            lines.append(f"heating_rate {self.heating_rate}")
        if self.zout:
            zout_str = " ".join(str(z) for z in self.zout)
            lines.append(f"zout {zout_str}")
        if self.output_file is not None:
            lines.append(f"output_file {self.output_file}")
        if self.write_optical_properties:
            lines.append("write_optical_properties")
        return lines

    def to_uvspec_items(self) -> list[tuple[int, str]]:
        phase = 9
        return [(phase, line) for line in self.to_uvspec_lines()]
