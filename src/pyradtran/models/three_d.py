"""3D atmosphere and cloud configuration model.

Maps to uvspec keywords: atmosphere_file_3D, cloud_file_3D,
output3D, processing3D, ipa_3d.

Reference: libRadtran uvspec_lex.l — 3D-related keywords.
"""

from __future__ import annotations

from pyradtran.models.base import UvspecOption


class ThreeDConfig(UvspecOption):
    """3D atmospheric/cloud field configuration.

    Enables 3D radiative transfer simulations with gridded
    atmosphere and cloud fields.

    Attributes:
        atmosphere_file: Path to 3D atmospheric field file (NetCDF).
        cloud_file: Path to 3D cloud optical property file (NetCDF).
        output_3d: Enable 3D output processing.
        processing_3d: Enable 3D post-processing.
        ipa_3d: Use independent pixel approximation for 3D.
    """

    atmosphere_file: str | None = None
    cloud_file: str | None = None
    output_3d: bool = False
    processing_3d: bool = False
    ipa_3d: bool = False

    def to_uvspec_lines(self) -> list[str]:
        lines: list[str] = []
        if self.atmosphere_file is not None:
            lines.append(f"atmosphere_file_3D {self.atmosphere_file}")
        if self.cloud_file is not None:
            lines.append(f"cloud_file_3D {self.cloud_file}")
        if self.output_3d:
            lines.append("output3D")
        if self.processing_3d:
            lines.append("processing3D")
        if self.ipa_3d:
            lines.append("ipa_3d")
        return lines