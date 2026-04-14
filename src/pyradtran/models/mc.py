"""Monte Carlo (MYSTIC) configuration model.

Maps to uvspec keywords: mc_photons, mc_backward, mc_escape, mc_vroom,
mc_polarisation, mc_randomseed, mc_minphotons, mc_maxscatters, mc_spectral_is,
mc_delta_scaling, mc_rad_alpha, mc_backward_output,
mc_forward_output, mc_backward_heat, mc_std, mc_jacobian, mc_progressbar,
mc_surface_reflectalways, mc_spherical, mc_tenstream, mc_ipa, mc_tipa,
mc_sensordirection, mc_sensorposition, mc_spherical3D_scene, mc_cloud_grid,
mc_basename, mc_minscatters, mc_sun_angular_size.

Reference: libRadtran src_py/mc_options.py
"""

from __future__ import annotations

from pydantic import Field, model_validator

from pyradtran.models.base import UvspecOption

VALID_BACKWARD_OUTPUTS = frozenset({
    "edir", "edn", "eup", "fdir", "fdn", "fup",
    "act", "abs", "emis", "heat", "exp", "exn", "eyp", "eyn", "ednpv",
})

VALID_BACKWARD_HEAT = frozenset({"HYBRID", "EMABS", "EMABSOPT", "DENET"})

VALID_FORWARD_OUTPUTS = frozenset({"heating", "actinic"})

VALID_OUTPUT_UNITS = frozenset({
    "W_per_m2_and_dz",
    "W_per_m3",
    "K_per_day",
})

VALID_BPDF_MODELS = frozenset({
    "litvinov", "maignan", "tsang_u10",
})


class McConfig(UvspecOption):
    """Monte Carlo (MYSTIC) solver configuration.

    Attributes:
        photons: Total number of photons to trace. Required for MYSTIC.
        min_photons: Minimum photons per spectral band (correlated-k).
        backward: Enable backward photon tracing. Required for most MC output.
        backward_pixel_range: Optional (ix_start, iy_start, ix_end, iy_end).
        escape: Calculate radiances via escape probabilities -- "on" or "off".
        vroom: Variance Reduction Optimal Options Method -- "on" or "off".
        polarisation: Enable polarisation calculations.
        polarisation_state: Initial Stokes vector (-3 to 4, default 0).
        random_seed: Random seed for reproducibility.
        max_scatters: Maximum scatters before photon destruction (testing).
        spectral_is: Wavelength for spectral importance sampling.
        delta_scaling_mucut: Truncation threshold for delta-M scaling.
        delta_scaling_n_start: Stream number for delta-M scaling start.
        rad_alpha: Opening angle for all-sky radiance (degrees).
        backward_output: Output quantity for backward MC (e.g. "edn").
        backward_output_unit: Unit for backward MC output.
        forward_output: Output quantity for forward MC (e.g. "heating").
        backward_heat: Thermal heating method -- "HYBRID", "EMABS", etc.
        std: Target standard deviation (stop when reached).
        jacobian: Jacobian calculation mode -- "1D" or "3D".
        jacobian_std: Calculate Jacobian standard deviation.
        progressbar: MC progress bar mode (0=off, 2, 3).
        surface_reflect_always: Always reflect at surface, weight via albedo.
        photons_file: Path to photon distribution file.
        spherical: 3D geometry mode - "1D" or "3D".
        tenstream: Enable tenstream approximation.
        ipa: Importance photon acceleration.
        tipa: Tracked importance photon acceleration - "dir" or "dir3d".
        sensor_direction: Sensor viewing direction (dx, dy, dz).
        sensor_position: Sensor position (x, y, z).
        spherical3d_scene: Spherical 3D scene bounds (lon_min, lat_min, lon_max, lat_max).
        cloud_grid: Cloud grid dimensions (nx, ny, nz).
        basename: Basename for output files.
        min_scatters: Minimum number of scatters before stopping.
        sun_angular_size: Sun angular size in degrees.
    """

    photons: int | None = Field(default=None, ge=0)
    min_photons: int | None = Field(default=None, ge=0)
    backward: bool = False
    backward_pixel_range: tuple[int, int, int, int] | None = None
    escape: str | None = None
    vroom: str | None = None
    polarisation: bool = False
    polarisation_state: int | None = Field(default=None, ge=-3, le=4)
    random_seed: int | None = Field(default=None, ge=0, le=1_000_000_000_000_000)
    max_scatters: int | None = Field(default=None, ge=0)
    spectral_is: float | None = Field(default=None, ge=0.0, le=1_000_000.0)
    delta_scaling_mucut: float | None = Field(default=None, ge=0.0, le=1.0)
    delta_scaling_n_start: int | None = Field(default=None, ge=0)
    rad_alpha: float | None = Field(default=None, ge=0.0, le=90.0)
    backward_output: str | None = None
    backward_output_unit: str | None = None
    forward_output: str | None = None
    backward_heat: str | None = None
    std: float | None = Field(default=None, ge=0.0)
    jacobian: str | None = None
    jacobian_std: bool = False
    progressbar: int | None = Field(default=None, ge=0, le=3)
    surface_reflect_always: bool = False
    photons_file: str | None = None

    # --- Surface files (Phase 4) ---
    albedo_file: str | None = None
    albedo_spectral_file: str | None = None
    rossli_file: str | None = None
    ambrals_spectral_file: str | None = None
    rpv_file: str | None = None
    bpdf: str | None = None
    surface_parallel: bool = False
    elevation_file: str | None = None
    lidar_file: str | None = None
    triangular_surface_file: str | None = None

    # --- 3D geometry (Phase 4) ---
    spherical: str | None = None  # "1D" or "3D"
    tenstream: bool = False
    ipa: bool = False
    tipa: str | None = None  # "dir" or "dir3d"
    sensor_direction: tuple[float, float, float] | None = None  # (dx, dy, dz)
    sensor_position: tuple[float, float, float] | None = None  # (x, y, z)
    spherical3d_scene: tuple[float, float, float, float] | None = None  # (lon_min, lat_min, lon_max, lat_max)
    cloud_grid: tuple[int, int, int] | None = None  # (nx, ny, nz)
    basename: str | None = None
    min_scatters: int | None = Field(default=None, ge=0)
    sun_angular_size: float | None = Field(default=None, ge=0.0, le=10.0)

    @model_validator(mode="after")
    def validate_mc(self) -> McConfig:
        if self.escape is not None and self.escape not in ("on", "off"):
            raise ValueError(f"mc escape must be 'on' or 'off', got '{self.escape}'")
        if self.vroom is not None and self.vroom not in ("on", "off"):
            raise ValueError(f"mc_vroom must be 'on' or 'off', got '{self.vroom}'")
        if self.backward_output is not None and self.backward_output not in VALID_BACKWARD_OUTPUTS:
            raise ValueError(
                f"Invalid backward_output '{self.backward_output}'. "
                f"Valid: {sorted(VALID_BACKWARD_OUTPUTS)}"
            )
        if (
            self.backward_output_unit is not None
            and self.backward_output_unit not in VALID_OUTPUT_UNITS
        ):
            raise ValueError(
                f"Invalid backward_output_unit '{self.backward_output_unit}'. "
                f"Valid: {sorted(VALID_OUTPUT_UNITS)}"
            )
        if self.forward_output is not None and self.forward_output not in VALID_FORWARD_OUTPUTS:
            raise ValueError(
                f"Invalid forward_output '{self.forward_output}'. "
                f"Valid: {sorted(VALID_FORWARD_OUTPUTS)}"
            )
        if self.backward_heat is not None and self.backward_heat not in VALID_BACKWARD_HEAT:
            raise ValueError(
                f"Invalid backward_heat '{self.backward_heat}'. "
                f"Valid: {sorted(VALID_BACKWARD_HEAT)}"
            )
        if self.backward_output_unit is not None and self.backward_output is None:
            raise ValueError("backward_output_unit requires backward_output to be set")
        if self.jacobian is not None and self.jacobian not in ("1D", "3D"):
            raise ValueError("mc_jacobian must be '1D' or '3D'")
        if self.jacobian_std and not self.jacobian:
            raise ValueError("jacobian_std requires jacobian to be set")
        if self.jacobian_std and not self.backward:
            raise ValueError("jacobian_std requires backward=True")

        # 3D geometry validation
        if self.spherical is not None and self.spherical not in ("1D", "3D"):
            raise ValueError(f"mc spherical must be '1D' or '3D', got '{self.spherical}'")
        if self.tipa is not None and self.tipa not in ("dir", "dir3d"):
            raise ValueError(f"mc_tipa must be 'dir' or 'dir3d', got '{self.tipa}'")

        if self.bpdf is not None and self.bpdf not in VALID_BPDF_MODELS:
            raise ValueError(
                f"Invalid bpdf '{self.bpdf}'. Valid: {sorted(VALID_BPDF_MODELS)}"
            )

        return self

    def to_uvspec_lines(self) -> list[str]:
        lines: list[str] = []

        if self.photons is not None:
            lines.append(f"mc_photons {self.photons}")

        if self.min_photons is not None:
            lines.append(f"mc_minphotons {self.min_photons}")

        if self.photons_file is not None:
            lines.append(f"mc_photons_file {self.photons_file}")

        if self.backward:
            if self.backward_pixel_range is not None:
                ix0, iy0, ix1, iy1 = self.backward_pixel_range
                lines.append(f"mc_backward {ix0} {iy0} {ix1} {iy1}")
            else:
                lines.append("mc_backward")

        if self.escape is not None:
            lines.append(f"mc_escape {self.escape}")

        if self.vroom is not None:
            lines.append(f"mc_vroom {self.vroom}")

        if self.polarisation:
            if self.polarisation_state is not None:
                lines.append(f"mc_polarisation {self.polarisation_state}")
            else:
                lines.append("mc_polarisation")

        if self.random_seed is not None:
            lines.append(f"mc_randomseed {self.random_seed}")

        if self.max_scatters is not None:
            lines.append(f"mc_maxscatters {self.max_scatters}")

        if self.spectral_is is not None:
            lines.append(f"mc_spectral_is {self.spectral_is}")

        if self.delta_scaling_mucut is not None:
            val = self.delta_scaling_mucut
            n = self.delta_scaling_n_start if self.delta_scaling_n_start is not None else 0
            lines.append(f"mc_delta_scaling {val} {n}")

        if self.rad_alpha is not None:
            lines.append(f"mc_rad_alpha {self.rad_alpha}")

        if self.backward_output is not None:
            line = f"mc_backward_output {self.backward_output}"
            if self.backward_output_unit is not None:
                line += f" {self.backward_output_unit}"
            lines.append(line)

        if self.forward_output is not None:
            lines.append(f"mc_forward_output {self.forward_output}")

        if self.backward_heat is not None:
            lines.append(f"mc_backward_heat {self.backward_heat}")

        if self.std is not None:
            lines.append(f"mc_std {self.std}")

        if self.jacobian is not None:
            lines.append(f"mc_jacobian {self.jacobian}")
            if self.jacobian_std:
                lines.append("mc_jacobian_std")

        if self.progressbar is not None:
            lines.append(f"mc_progressbar {self.progressbar}")

        if self.surface_reflect_always:
            lines.append("mc_surface_reflectalways")

        # --- 3D geometry (Phase 4) ---
        if self.spherical is not None:
            lines.append(f"mc_spherical {self.spherical}")

        if self.tenstream:
            lines.append("mc_tenstream")

        if self.ipa:
            lines.append("mc_ipa")

        if self.tipa is not None:
            lines.append(f"mc_tipa {self.tipa}")

        if self.sensor_direction is not None:
            dx, dy, dz = self.sensor_direction
            lines.append(f"mc_sensordirection {dx} {dy} {dz}")

        if self.sensor_position is not None:
            x, y, z = self.sensor_position
            lines.append(f"mc_sensorposition {x} {y} {z}")

        if self.spherical3d_scene is not None:
            lon0, lat0, lon1, lat1 = self.spherical3d_scene
            lines.append(f"mc_spherical3D_scene {lon0} {lat0} {lon1} {lat1}")

        if self.cloud_grid is not None:
            nx, ny, nz = self.cloud_grid
            lines.append(f"mc_cloud_grid {nx} {ny} {nz}")

        if self.basename is not None:
            lines.append(f"mc_basename {self.basename}")

        if self.min_scatters is not None:
            lines.append(f"mc_minscatters {self.min_scatters}")

        if self.sun_angular_size is not None:
            lines.append(f"mc_sun_angular_size {self.sun_angular_size}")

        # --- Surface files (Phase 4) ---
        if self.albedo_file is not None:
            lines.append(f"mc_albedo_file {self.albedo_file}")
        if self.albedo_spectral_file is not None:
            lines.append(f"mc_albedo_spectral_file {self.albedo_spectral_file}")
        if self.rossli_file is not None:
            lines.append(f"mc_rossli_file {self.rossli_file}")
        if self.ambrals_spectral_file is not None:
            lines.append(f"mc_ambrals_spectral_file {self.ambrals_spectral_file}")
        if self.rpv_file is not None:
            lines.append(f"mc_rpv_file {self.rpv_file}")
        if self.bpdf is not None:
            lines.append(f"mc_bpdf {self.bpdf}")
        if self.surface_parallel:
            lines.append("mc_surfaceparallel")
        if self.elevation_file is not None:
            lines.append(f"mc_elevation_file {self.elevation_file}")
        if self.lidar_file is not None:
            lines.append(f"mc_lidar_file {self.lidar_file}")
        if self.triangular_surface_file is not None:
            lines.append(f"mc_triangular_surface_file {self.triangular_surface_file}")

        return lines
