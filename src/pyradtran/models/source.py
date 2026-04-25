"""Source configuration model.

Maps to uvspec keywords: source, sza, phi0, day_of_year, solar_flux_file, umu, phi,
latitude, longitude, time, time_interpolate, time_interval, earth_radius, sza_file,
isotropic_source_toa.

Reference: libRadtran src_py/geometry_options.py, src_py/spectral_options.py
"""

from __future__ import annotations

from pydantic import Field, model_validator

from pyradtran.models.base import UvspecOption


class SourceConfig(UvspecOption):
    """Solar or thermal radiation source configuration.

    Attributes:
        source: Radiation source type -- "solar" or "thermal".
        sza: Solar zenith angle in degrees [0, 180]. Required for solar source
            (unless sza_file or isotropic_source_toa is set).
        phi0: Solar azimuth angle in degrees [-360, 360].
        day_of_year: Day of year [1, 366].
        solar_flux_file: Path to solar flux file (e.g. kurudz_0.1nm.dat).
        umu: Viewing zenith angles (cosines). Positive = upward, negative = downward.
        phi: Viewing azimuth angles in degrees.
        satellite_geometry: Satellite geometry specification (e.g., SENTINEL2A, MPS).
        satellite_pixel: Pixel coordinates (x, y) for satellite pixel-based geometry.
        latitude: Geographic latitude as (hemisphere, degrees, minutes, seconds).
        longitude: Geographic longitude as (hemisphere, degrees, minutes, seconds).
        time: Time of day string (e.g. "10:30:00").
        time_interpolate: Enable time interpolation.
        time_interval: Tuple of (start_time, end_time) for time range.
        earth_radius: Earth radius in km.
        sza_file: Path to file containing solar zenith angle data.
        isotropic_source_toa: Use isotropic source at top of atmosphere.
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
    latitude: tuple[str, int, int, int] | None = None
    longitude: tuple[str, int, int, int] | None = None
    time: str | None = None
    time_interpolate: bool = False
    time_interval: tuple[str, str] | None = None
    earth_radius: float | None = Field(default=None, ge=0.0)
    sza_file: str | None = None
    isotropic_source_toa: bool = False

    @model_validator(mode="after")
    def check_sza_for_solar(self) -> SourceConfig:
        if (
            self.source == "solar"
            and self.sza is None
            and self.sza_file is None
            and not self.isotropic_source_toa
        ):
            msg = "sza, sza_file, or isotropic_source_toa is required when source='solar'"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def check_sza_mutual_exclusion(self) -> SourceConfig:
        count = sum(1 for x in (self.sza, self.sza_file) if x is not None)
        count += int(self.isotropic_source_toa)
        if count > 1:
            raise ValueError("sza, sza_file, and isotropic_source_toa are mutually exclusive")
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
        if self.latitude is not None:
            h, d, m, s = self.latitude
            lines.append(f"latitude {h} {d} {m} {s}")
        if self.longitude is not None:
            h, d, m, s = self.longitude
            lines.append(f"longitude {h} {d} {m} {s}")
        if self.time is not None:
            lines.append(f"time {self.time}")
        if self.time_interpolate:
            lines.append("time_interpolate")
        if self.time_interval is not None:
            start, end = self.time_interval
            lines.append(f"time_interval {start} {end}")
        if self.earth_radius is not None:
            lines.append(f"earth_radius {self.earth_radius}")
        if self.sza_file is not None:
            lines.append(f"sza_file {self.sza_file}")
        if self.isotropic_source_toa:
            lines.append("isotropic_source_toa")
        return lines

    def to_uvspec_items(self) -> list[tuple[int, str]]:
        phase = 2
        return [(phase, line) for line in self.to_uvspec_lines()]
