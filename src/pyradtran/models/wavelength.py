"""Wavelength configuration model.

Maps to uvspec keywords: wavelength, spline, filter_function_file.

Reference: libRadtran src_py/spectral_options.py
"""

from pydantic import Field

from pyradtran.models.base import UvspecOption


class WavelengthConfig(UvspecOption):
    """Spectral range configuration.

    Attributes:
        wavelength_min: Shortest wavelength (nm or cm-1).
        wavelength_max: Longest wavelength (nm or cm-1).
        unit: Wavelength unit -- "nm" (default) or "cm-1" for wavenumbers.
        spline: Spline smoothing arguments string (3 floats).
        filter_function_file: Path to filter function file.
    """

    wavelength_min: float = Field(ge=0.0, le=1e6)
    wavelength_max: float = Field(ge=0.0, le=1e6)
    unit: str = Field(default="nm", pattern=r"^(nm|cm-1)$")
    spline: str | None = None
    filter_function_file: str | None = None

    def to_uvspec_lines(self) -> list[str]:
        lines: list[str] = []
        if self.unit == "cm-1":
            lines.append(f"wavelength {self.wavelength_min} {self.wavelength_max} cm-1")
        else:
            lines.append(f"wavelength {self.wavelength_min} {self.wavelength_max}")
        if self.spline is not None:
            lines.append(f"spline {self.spline}")
        if self.filter_function_file is not None:
            lines.append(f"filter_function_file {self.filter_function_file}")
        return lines
