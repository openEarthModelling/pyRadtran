"""Source configuration model.

Maps to uvspec keywords: source, sza, phi0, day_of_year, solar_flux_file, umu, phi.

Reference: libRadtran src_py/geometry_options.py, src_py/spectral_options.py
"""

from __future__ import annotations

from pydantic import Field, model_validator

from pyradtran.models.base import UvspecOption


class SourceConfig(UvspecOption):
    """Solar or thermal radiation source configuration.

    Attributes:
        source: Radiation source type -- "solar" or "thermal".
        sza: Solar zenith angle in degrees [0, 180]. Required for solar source.
        phi0: Solar azimuth angle in degrees [-360, 360].
        day_of_year: Day of year [1, 366].
        solar_flux_file: Path to solar flux file (e.g. kurudz_0.1nm.dat).
        umu: Viewing zenith angles (cosines). Positive = upward, negative = downward.
        phi: Viewing azimuth angles in degrees.
        satellite_geometry: Satellite geometry specification (e.g., SENTINEL2A, MPS).
        satellite_pixel: Pixel coordinates (x, y) for satellite pixel-based geometry.
    """

    source: str = Field(pattern=r"^(solar|thermal)$")
    sza: float | None = Field(default=None, ge=0.0, le=180.0)
    phi0: float | None = Field(default=None, ge=-360.0, le=360.0)
    day_of_year: int | None = Field(default=None, ge=1, le=366)
    solar_flux_file: str | None = None
    umu: list[float] = Field(default_factory=list)
    phi: list[float] = Field(default_factory=list)
    satellite_geometry: str | None = None
    satellite_pixel: tuple[int, int] | None = None

    @model_validator(mode="after")
    def check_sza_for_solar(self) -> SourceConfig:
        if self.source == "solar" and self.sza is None:
            raise ValueError("sza is required when source='solar'")
        return self

    @model_validator(mode="after")
    def check_satellite_consistency(self) -> SourceConfig:
        if self.satellite_pixel is not None and self.satellite_geometry is None:
            raise ValueError(
                "satellite_pixel requires satellite_geometry to be set"
            )
        if self.satellite_geometry is not None and self.satellite_pixel is None:
            raise ValueError(
                "satellite_geometry requires satellite_pixel to be set"
            )
        return self

    def to_uvspec_lines(self) -> list[str]:
        lines: list[str] = []
        if self.solar_flux_file:
            lines.append(f"source {self.source} {self.solar_flux_file}")
        else:
            lines.append(f"source {self.source}")
        if self.sza is not None:
            lines.append(f"sza {self.sza}")
        if self.phi0 is not None:
            lines.append(f"phi0 {self.phi0}")
        if self.day_of_year is not None:
            lines.append(f"day_of_year {self.day_of_year}")
        if self.umu:
            lines.append(f"umu {' '.join(str(v) for v in self.umu)}")
        if self.phi:
            lines.append(f"phi {' '.join(str(v) for v in self.phi)}")
        if self.satellite_geometry is not None:
            lines.append(f"satellite_geometry {self.satellite_geometry}")
        if self.satellite_pixel is not None:
            x, y = self.satellite_pixel
            lines.append(f"satellite_pixel {x} {y}")
        return lines
