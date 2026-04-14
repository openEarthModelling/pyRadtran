"""SSLidar (Single Scattering Lidar) configuration model.

Maps to uvspec keywords: sslidar, sslidar_nranges, sslidar_polarisation.

Reference: libRadtran src_py/geometry_options.py, src_py/solver_options.py
"""

from pydantic import Field

from pyradtran.models.base import UvspecOption


class SslidarConfig(UvspecOption):
    """Single scattering lidar configuration.

    Requires rte_solver sslidar.

    Attributes:
        area: Telescope area in m^2.
        E0: Pulse energy in Joules.
        efficiency: Detector efficiency.
        position: Lidar position above ground in km.
        range_bin: Range bin width in km.
        n_ranges: Number of range bins.
        polarisation: Enable polarisation measurement.
    """

    area: float | None = Field(default=None, ge=0.0, le=1e6)
    E0: float | None = Field(default=None, ge=0.0, le=1e6)
    efficiency: float | None = Field(default=None, ge=0.0, le=1.0)
    position: float | None = Field(default=None, ge=0.0, le=1e6)
    range_bin: float | None = Field(default=None, ge=0.0, le=1e6)
    n_ranges: int | None = Field(default=None, ge=1)
    polarisation: bool = False

    def to_uvspec_lines(self) -> list[str]:
        lines: list[str] = []
        if self.area is not None:
            lines.append(f"sslidar area {self.area}")
        if self.E0 is not None:
            lines.append(f"sslidar E0 {self.E0}")
        if self.efficiency is not None:
            lines.append(f"sslidar eff {self.efficiency}")
        if self.position is not None:
            lines.append(f"sslidar position {self.position}")
        if self.range_bin is not None:
            lines.append(f"sslidar range {self.range_bin}")
        if self.n_ranges is not None:
            lines.append(f"sslidar_nranges {self.n_ranges}")
        if self.polarisation:
            lines.append("sslidar_polarisation")
        return lines
