"""Wavelength configuration model.

Maps to uvspec keywords: wavelength, spline, filter_function_file,
wavelength_grid_file, wavelength_index, slit_function_file, spline_file,
fluorescence, fluorescence_file, thermal_bands_file, thermal_bandwidth.

Reference: libRadtran src_py/spectral_options.py
"""

from __future__ import annotations

from pydantic import Field, model_validator

from pyradtran.models.base import UvspecOption


class WavelengthConfig(UvspecOption):
    """Spectral range configuration.

    Attributes:
        wavelength_min: Shortest wavelength (nm or cm-1), or single wavelength
            when wavelength_max is not set.
        wavelength_max: Longest wavelength (nm or cm-1). Optional; when None,
            wavelength_min is treated as a single wavelength value.
        unit: Wavelength unit -- "nm" (default) or "cm-1" for wavenumbers.
        spline: Spline smoothing arguments string (3 floats).
        filter_function_file: Path to filter function file.
        wavelength_grid_file: Path to wavelength grid file (alternative to wavelength_min).
        wavelength_index: Tuple of (start_index, end_index) for wavelength subset.
        slit_function_file: Path to slit function file.
        spline_file: Path to spline file.
        fluorescence: Enable fluorescence calculation.
        fluorescence_file: Path to fluorescence data file.
        thermal_bands_file: Path to thermal bands file.
        thermal_bandwidth: Tuple of (width, unit) for thermal bandwidth.
    """

    wavelength_min: float | None = Field(default=None, ge=0.0, le=1e6)
    wavelength_max: float | None = Field(default=None, ge=0.0, le=1e6)
    unit: str = Field(default="nm", pattern=r"^(nm|cm-1)$")
    spline: str | None = None
    filter_function_file: str | None = None
    wavelength_grid_file: str | None = None
    wavelength_index: tuple[int, int] | None = None
    slit_function_file: str | None = None
    spline_file: str | None = None
    fluorescence: bool = False
    fluorescence_file: str | None = None
    thermal_bands_file: str | None = None
    thermal_bandwidth: tuple[float, str] | None = None

    @model_validator(mode="after")
    def validate_wavelength_set(self) -> WavelengthConfig:
        if self.wavelength_min is None and self.wavelength_grid_file is None:
            raise ValueError("At least one of wavelength_min or wavelength_grid_file must be set")
        return self

    def to_uvspec_lines(self) -> list[str]:
        lines: list[str] = []
        if self.wavelength_grid_file is not None:
            lines.append(f"wavelength_grid_file {self.wavelength_grid_file}")
        if self.wavelength_min is not None:
            if self.wavelength_max is not None:
                # Range mode: min and max
                if self.unit == "cm-1":
                    lines.append(f"wavelength {self.wavelength_min} {self.wavelength_max} cm-1")
                else:
                    lines.append(f"wavelength {self.wavelength_min} {self.wavelength_max}")
            else:
                # Single wavelength mode
                if self.unit == "cm-1":
                    lines.append(f"wavelength {self.wavelength_min} cm-1")
                else:
                    lines.append(f"wavelength {self.wavelength_min}")
        if self.wavelength_index is not None:
            i0, i1 = self.wavelength_index
            lines.append(f"wavelength_index {i0} {i1}")
        if self.spline is not None:
            lines.append(f"spline {self.spline}")
        if self.spline_file is not None:
            lines.append(f"spline_file {self.spline_file}")
        if self.slit_function_file is not None:
            lines.append(f"slit_function_file {self.slit_function_file}")
        if self.filter_function_file is not None:
            lines.append(f"filter_function_file {self.filter_function_file}")
        if self.fluorescence:
            lines.append("fluorescence")
        if self.fluorescence_file is not None:
            lines.append(f"fluorescence_file {self.fluorescence_file}")
        if self.thermal_bands_file is not None:
            lines.append(f"thermal_bands_file {self.thermal_bands_file}")
        if self.thermal_bandwidth is not None:
            width, unit = self.thermal_bandwidth
            lines.append(f"thermal_bandwidth {width} {unit}")
        return lines

    def to_uvspec_items(self) -> list[tuple[int, str]]:
        phase = 3
        return [(phase, line) for line in self.to_uvspec_lines()]
