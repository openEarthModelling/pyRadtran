"""Monte Carlo (MYSTIC) configuration model.

Maps to uvspec keywords: mc_photons, mc_backward, mc_escape, mc_vroom,
mc_polarisation, mc_randomseed, mc_minphotons, mc_maxscatters, mc_spectral_is,
mc_delta_scaling, mc_rad_alpha, mc_backward_output,
mc_forward_output, mc_backward_heat, mc_std, mc_jacobian, mc_progressbar,
mc_surface_reflectalways.

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
        if self.backward_output_unit is not None and self.backward_output_unit not in VALID_OUTPUT_UNITS:
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

        return lines
