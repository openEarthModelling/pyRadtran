"""3D atmosphere and cloud configuration model.

Maps to uvspec keywords: atmosphere_file (3D NetCDF fields).

Note: 3D cloud fields are specified via CloudConfig using
wc_file=("3D", path) or ic_file=("3D", path), not through
a separate keyword. The ipa_3d and output3D flags are set
automatically by uvspec when tipa or MYSTIC is enabled.

Reference: libRadtran src/uvspec_lex.l (MYSTIC 3D options)
"""

from __future__ import annotations

from pyradtran.models.base import UvspecOption


class ThreeDConfig(UvspecOption):
    """3D atmospheric field configuration.

    Configures 3D atmospheric input fields for MYSTIC and
    dynamic tenstream solvers.

    Attributes:
        atmosphere_file: Path to 3D atmospheric field NetCDF file.

    Note:
        3D cloud fields are specified through CloudConfig:
        scene.set_cloud(wc_file=("3D", "/path/to/cloud3d.nc"))
        or scene.set_cloud(ic_file=("3D", "/path/to/ic3d.nc")).

        The ipa (independent pixel approximation) for 3D is enabled
        via McConfig.ipa=True or McConfig.tipa="dir"/"dir3d".
    """

    atmosphere_file: str | None = None

    def to_uvspec_lines(self) -> list[str]:
        lines: list[str] = []
        if self.atmosphere_file is not None:
            lines.append(f"atmosphere_file {self.atmosphere_file}")
        return lines

    def to_uvspec_items(self) -> list[tuple[int, str]]:
        phase = 11
        return [(phase, line) for line in self.to_uvspec_lines()]
