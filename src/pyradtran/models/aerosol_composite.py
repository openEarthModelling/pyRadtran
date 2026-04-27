"""Composable aerosol optical-property models (Tier 1–4).

See design spec: docs/superpowers/specs/2026-04-27-composite-aerosol-design.md
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, model_validator
from typing import Literal


class RefractiveIndex(BaseModel):
    """Wavelength-dependent complex refractive index.

    Interpolation is log-linear in wavelength.
    """

    model_config = {"extra": "forbid", "frozen": True, "populate_by_name": True}

    wavelength_um: list[float] = Field(min_length=2)
    n_real: list[float]
    k_imag: list[float] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_lengths_and_sorted(self) -> "RefractiveIndex":
        n = len(self.wavelength_um)
        if len(self.n_real) != n or len(self.k_imag) != n:
            raise ValueError(
                f"Length mismatch: wavelength_um={n}, n_real={len(self.n_real)}, "
                f"k_imag={len(self.k_imag)}"
            )
        if self.wavelength_um != sorted(self.wavelength_um):
            raise ValueError("wavelength_um must be strictly ascending")
        if any(k < 0 for k in self.k_imag):
            raise ValueError("k_imag must be >= 0")
        return self

    def at(self, wl_um: np.ndarray) -> np.ndarray:
        """Interpolate refractive index to requested wavelengths."""
        wl_tab = np.asarray(self.wavelength_um)
        n_tab = np.asarray(self.n_real)
        k_tab = np.asarray(self.k_imag)

        wl = np.asarray(wl_um)
        if np.any(wl < wl_tab[0]) or np.any(wl > wl_tab[-1]):
            raise ValueError(
                f"Wavelength {wl.min():.4f}–{wl.max():.4f} um outside "
                f"tabulated range {wl_tab[0]:.4f}–{wl_tab[-1]:.4f} um"
            )

        n_interp = np.interp(wl, wl_tab, n_tab)
        logk_interp = np.interp(np.log(wl), np.log(wl_tab), np.log(k_tab + 1e-20))
        k_interp = np.exp(logk_interp)
        return n_interp + 1j * k_interp


class ParticleOptics(BaseModel):
    """Single-particle optical properties vs wavelength and radius."""

    model_config = {
        "extra": "forbid",
        "frozen": True,
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
    }

    wavelength_um: list[float] = Field(min_length=1)
    radius_um: list[float] = Field(min_length=1)
    Qext: NDArray
    Qsca: NDArray
    g: NDArray
    legendre_moments: NDArray | None = None

    @model_validator(mode="after")
    def validate_shapes(self) -> "ParticleOptics":
        n_wl = len(self.wavelength_um)
        n_r = len(self.radius_um)
        expected = (n_wl, n_r)

        if self.Qext.shape != expected:
            raise ValueError(f"Qext shape {self.Qext.shape} != expected {expected}")
        if self.Qsca.shape != expected:
            raise ValueError(f"Qsca shape {self.Qsca.shape} != expected {expected}")
        if self.g.shape != expected:
            raise ValueError(f"g shape {self.g.shape} != expected {expected}")
        if self.wavelength_um != sorted(self.wavelength_um):
            raise ValueError("wavelength_um must be strictly ascending")
        if self.radius_um != sorted(self.radius_um):
            raise ValueError("radius_um must be strictly ascending")
        if np.any(self.Qsca > self.Qext + 1e-12):
            raise ValueError("Qsca must be <= Qext")
        if np.any(np.abs(self.g) > 1.0 + 1e-12):
            raise ValueError("|g| must be <= 1")

        if self.legendre_moments is not None:
            if self.legendre_moments.shape[:2] != expected:
                raise ValueError(
                    f"legendre_moments leading dims {self.legendre_moments.shape[:2]} "
                    f"!= expected {expected}"
                )
        return self

    @classmethod
    def from_cross_sections(
        cls,
        *,
        wavelength_um: list[float],
        radius_um: list[float],
        Cext_um2: NDArray,
        Csca_um2: NDArray,
        g: NDArray,
        legendre_moments: NDArray | None = None,
    ) -> "ParticleOptics":
        """Build from cross-sections (convert to Q-factors)."""
        wl = np.asarray(wavelength_um).reshape(-1, 1)
        r = np.asarray(radius_um).reshape(1, -1)
        area = np.pi * r**2
        Qext = np.asarray(Cext_um2) / area
        Qsca = np.asarray(Csca_um2) / area
        return cls(
            wavelength_um=wavelength_um,
            radius_um=radius_um,
            Qext=Qext,
            Qsca=Qsca,
            g=g,
            legendre_moments=legendre_moments,
        )


class SizeDistribution(BaseModel):
    """Aerosol particle size distribution."""

    model_config = {"extra": "forbid", "frozen": True, "populate_by_name": True}

    kind: Literal["lognormal", "modified_gamma", "discrete", "monodisperse"]
    params: dict
    number_density_per_m3: float = 1.0

    @model_validator(mode="after")
    def validate_params(self) -> "SizeDistribution":
        if self.kind == "lognormal":
            if "r_g_um" not in self.params or "sigma_g" not in self.params:
                raise ValueError("lognormal requires r_g_um and sigma_g")
            if self.params["sigma_g"] <= 1.0:
                raise ValueError("sigma_g must be > 1")
        elif self.kind == "modified_gamma":
            for key in ("alpha", "gamma", "r_c_um"):
                if key not in self.params:
                    raise ValueError(f"modified_gamma requires {key}")
        elif self.kind == "discrete":
            if "radius_um" not in self.params or "weights" not in self.params:
                raise ValueError("discrete requires radius_um and weights")
            if len(self.params["radius_um"]) != len(self.params["weights"]):
                raise ValueError("radius_um and weights must have same length")
        elif self.kind == "monodisperse":
            if "radius_um" not in self.params:
                raise ValueError("monodisperse requires radius_um")
        return self

    def evaluate(self, r_grid_um: np.ndarray) -> np.ndarray:
        r = np.asarray(r_grid_um)
        if self.kind == "lognormal":
            rg = self.params["r_g_um"]
            sg = self.params["sigma_g"]
            ln_s = np.log(sg)
            dn_dlnr = (
                self.number_density_per_m3
                / (np.sqrt(2.0 * np.pi) * ln_s)
                * np.exp(-0.5 * (np.log(r / rg) / ln_s) ** 2)
            )
            dn_dr = dn_dlnr / r
        elif self.kind == "monodisperse":
            r0 = self.params["radius_um"]
            sigma = r0 * 0.01
            dn_dr = (
                self.number_density_per_m3
                / (sigma * np.sqrt(2.0 * np.pi))
                * np.exp(-0.5 * ((r - r0) / sigma) ** 2)
            )
        elif self.kind == "discrete":
            radii = np.asarray(self.params["radius_um"])
            weights = np.asarray(self.params["weights"])
            weights = weights / weights.sum()
            dn_dr = np.zeros_like(r, dtype=float)
            for rad, w in zip(radii, weights):
                sigma = rad * 0.01
                dn_dr += (
                    w
                    * self.number_density_per_m3
                    / (sigma * np.sqrt(2.0 * np.pi))
                    * np.exp(-0.5 * ((r - rad) / sigma) ** 2)
                )
        elif self.kind == "modified_gamma":
            alpha = self.params["alpha"]
            gamma = self.params["gamma"]
            rc = self.params["r_c_um"]
            b = alpha / (gamma * rc**gamma)
            # Use math.gamma instead of scipy.special.gamma to avoid scipy dependency
            import math
            A = (
                self.number_density_per_m3
                * gamma
                * b ** ((alpha + 1) / gamma)
                / math.gamma((alpha + 1) / gamma)
            )
            dn_dr = A * r**alpha * np.exp(-b * r**gamma)
        else:
            raise ValueError(f"Unknown kind: {self.kind}")
        return dn_dr


class IntegrationConfig(BaseModel):
    """Configuration for size-distribution numerical integration."""

    model_config = {"extra": "forbid", "frozen": True, "populate_by_name": True}

    n_radius_grid: int = 200
    radius_min_um: float = 0.001
    radius_max_um: float = 100.0
