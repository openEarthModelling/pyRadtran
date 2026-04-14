"""Surface configuration model.

Maps to uvspec keywords: albedo, albedo_file, albedo_map, albedo_library,
sur_temperature, brdf_ambrals, brdf_hapke, brdf_rpv, brdf_cam,
bpdf_litvinov, bpdf_maignan, bpdf_tsang_u10.

Reference: libRadtran src/uvspec_lex.l (surface/BRDF options)
"""

from __future__ import annotations

from pydantic import Field, model_validator

from pyradtran.models.base import UvspecOption

_VALID_AMBRALS_PARAMS = frozenset({"geo", "iso", "vol"})
_VALID_HAPKE_PARAMS = frozenset({"b0", "h", "w"})
_VALID_RPV_PARAMS = frozenset({"rho0", "k", "theta", "scale"})
_VALID_CAM_PARAMS = frozenset({"pcl", "sal", "u10"})
_VALID_LITVINOV_PARAMS = frozenset({
    "albedo", "rms_slope", "fresnel", "shadowing",
    "refr_index", "refr_index_i", "rms_slope2",
})
_VALID_MAIGNAN_PARAMS = frozenset({"c_maign", "arvi", "refr_index", "refr_index_i"})


class SurfaceConfig(UvspecOption):
    """Surface reflection and temperature configuration.

    Attributes:
        albedo: Constant Lambertian albedo [0, 1]. Mutually exclusive with BRDF/BPDF.
        albedo_file: Path to wavelength-dependent albedo file.
        albedo_map: Path to spatial albedo map NetCDF file, or tuple (path, variable).
        albedo_library: Albedo library name or path ("IGBP").
        sur_temperature: Surface temperature in Kelvin (for thermal IR).
        brdf_ambrals: Dict of AMBRALS BRDF parameters (iso, vol, geo).
        brdf_hapke: Dict of Hapke BRDF parameters (w, b0, h).
        brdf_rpv: Dict of RPV BRDF parameters (rho0, k, theta, scale).
        brdf_cam: Dict of CAM BRDF parameters (pcl, sal, u10).
        bpdf_litvinov: Dict of Litvinov BPDF parameters.
        bpdf_maignan: Dict of Maignan BPDF parameters.
        bpdf_tsang_u10: Wind speed for Tsang BPDF (m/s).
    """

    albedo: float | None = Field(default=None, ge=0.0, le=1.0)
    albedo_file: str | None = None
    albedo_map: str | tuple[str, str] | None = None
    albedo_library: str | None = None
    sur_temperature: float | None = None
    brdf_ambrals: dict[str, float] | None = None
    brdf_hapke: dict[str, float] | None = None
    brdf_rpv: dict[str, float] | None = None
    brdf_cam: dict[str, float] | None = None
    bpdf_litvinov: dict[str, float] | None = None
    bpdf_maignan: dict[str, float] | None = None
    bpdf_tsang_u10: float | None = Field(default=None, ge=0.0, le=100.0)

    @model_validator(mode="after")
    def check_mutual_exclusion(self) -> SurfaceConfig:
        brdf_or_bpdf = (
            self.brdf_ambrals or self.brdf_hapke or self.brdf_rpv
            or self.brdf_cam or self.bpdf_litvinov or self.bpdf_maignan
            or self.bpdf_tsang_u10 is not None
        )
        if self.albedo is not None and brdf_or_bpdf:
            raise ValueError(
                "Cannot set albedo together with BRDF/BPDF options"
            )
        if self.albedo_file is not None and brdf_or_bpdf:
            raise ValueError(
                "Cannot set albedo_file together with BRDF/BPDF options"
            )
        if self.albedo is not None and self.albedo_file is not None:
            raise ValueError("Cannot set both albedo and albedo_file")
        if self.albedo is not None and self.albedo_map is not None:
            raise ValueError("Cannot set both albedo and albedo_map")
        if self.albedo_file is not None and self.albedo_map is not None:
            raise ValueError("Cannot set both albedo_file and albedo_map")
        if self.brdf_ambrals is not None:
            for key in self.brdf_ambrals:
                if key not in _VALID_AMBRALS_PARAMS:
                    raise ValueError(
                        f"Invalid brdf_ambrals parameter '{key}'. "
                        f"Valid: {sorted(_VALID_AMBRALS_PARAMS)}"
                    )
        if self.brdf_hapke is not None:
            for key in self.brdf_hapke:
                if key not in _VALID_HAPKE_PARAMS:
                    raise ValueError(
                        f"Invalid brdf_hapke parameter '{key}'. "
                        f"Valid: {sorted(_VALID_HAPKE_PARAMS)}"
                    )
        if self.brdf_rpv is not None:
            for key in self.brdf_rpv:
                if key not in _VALID_RPV_PARAMS:
                    raise ValueError(
                        f"Invalid brdf_rpv parameter '{key}'. "
                        f"Valid: {sorted(_VALID_RPV_PARAMS)}"
                    )
        if self.brdf_cam is not None:
            for key in self.brdf_cam:
                if key not in _VALID_CAM_PARAMS:
                    raise ValueError(
                        f"Invalid brdf_cam parameter '{key}'. "
                        f"Valid: {sorted(_VALID_CAM_PARAMS)}"
                    )
        return self

    def to_uvspec_lines(self) -> list[str]:
        lines: list[str] = []
        if self.albedo is not None:
            lines.append(f"albedo {self.albedo}")
        elif self.albedo_file is not None:
            lines.append(f"albedo_file {self.albedo_file}")
        elif self.albedo_map is not None:
            if isinstance(self.albedo_map, tuple):
                lines.append(f"albedo_map {self.albedo_map[0]} {self.albedo_map[1]}")
            else:
                lines.append(f"albedo_map {self.albedo_map}")
        if self.albedo_library is not None:
            lines.append(f"albedo_library {self.albedo_library}")
        if self.brdf_ambrals is not None:
            for param, value in self.brdf_ambrals.items():
                lines.append(f"brdf_ambrals {param} {value}")
        if self.brdf_hapke is not None:
            for param, value in self.brdf_hapke.items():
                lines.append(f"brdf_hapke {param} {value}")
        if self.brdf_rpv is not None:
            for param, value in self.brdf_rpv.items():
                lines.append(f"brdf_rpv {param} {value}")
        if self.brdf_cam is not None:
            for param, value in self.brdf_cam.items():
                lines.append(f"brdf_cam {param} {value}")
        if self.bpdf_litvinov is not None:
            vals = " ".join(str(v) for v in self.bpdf_litvinov.values())
            lines.append(f"bpdf_litvinov {vals}")
        if self.bpdf_maignan is not None:
            vals = " ".join(str(v) for v in self.bpdf_maignan.values())
            lines.append(f"bpdf_maignan {vals}")
        if self.bpdf_tsang_u10 is not None:
            lines.append(f"bpdf_tsang_u10 {self.bpdf_tsang_u10}")
        if self.sur_temperature is not None:
            lines.append(f"sur_temperature {self.sur_temperature}")
        return lines
