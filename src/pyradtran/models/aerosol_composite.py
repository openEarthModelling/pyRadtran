"""Composable aerosol optical-property models (Tier 1–4).

See design spec: docs/superpowers/specs/2026-04-27-composite-aerosol-design.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, model_validator


class RefractiveIndex(BaseModel):
    """Wavelength-dependent complex refractive index.

    Interpolation is log-linear in wavelength.
    """

    model_config = {"extra": "forbid", "frozen": True, "populate_by_name": True}

    wavelength_um: list[float] = Field(min_length=2)
    n_real: list[float]
    k_imag: list[float] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_lengths_and_sorted(self) -> RefractiveIndex:
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


class SizeDistribution(BaseModel):
    """Aerosol particle size distribution."""

    model_config = {"extra": "forbid", "frozen": True, "populate_by_name": True}

    kind: Literal["lognormal", "modified_gamma", "discrete", "monodisperse"]
    params: dict
    number_density_per_m3: float = 1.0

    @model_validator(mode="after")
    def validate_params(self) -> SizeDistribution:
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
            for rad, w in zip(radii, weights, strict=False):
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


@dataclass
class SpeciesOptics:
    """Mass-normalized intensive optical properties of an aerosol species."""

    beta_ext_per_mass: NDArray
    ssa: NDArray
    g: NDArray
    legendre_moments: NDArray | None = None


class BulkSpecies(BaseModel):
    """Species backed by an aerosol3D BulkAerosolOpticsData.

    The size-distribution integration is already done in aerosol3D; this class
    only rescales cross-sections to per-mass, resamples wavelength, and selects
    the Legendre convention. Duck-typed: accepts any object exposing the
    BulkAerosolOpticsData attributes (no hard aerosol3D import).
    """

    model_config = {"extra": "forbid", "frozen": True, "arbitrary_types_allowed": True}

    bulk: Any  # BulkAerosolOpticsData-like
    name: str = "BulkSpecies"

    @property
    def mass_per_particle_kg(self) -> float:
        """Volume-weighted per-particle mass from the bulk size distribution."""
        rho = self.bulk.effective_density_kg_m3
        if rho is None or getattr(self.bulk, "size_distribution", None) is None:
            raise ValueError(
                "BulkAerosolOpticsData needs effective_density_kg_m3 and size_distribution"
            )
        r3_nm3 = float(self.bulk.size_distribution.moment(3.0))
        vol_m3 = (4.0 / 3.0) * np.pi * r3_nm3 * 1e-27  # nm^3 -> m^3
        return rho * vol_m3

    def intensive(self, wl_um: np.ndarray, n_legendre: int = 32) -> SpeciesOptics:
        wl = np.asarray(wl_um, dtype=float)
        wl_tab_um = np.asarray(self.bulk.wavelength_nm, dtype=float) / 1000.0

        def _loglog(col):
            return np.exp(
                np.interp(wl, wl_tab_um, np.log(np.clip(np.asarray(col) * 1e-6, 1e-30, None)))
            )

        def _lin(col):
            return np.interp(wl, wl_tab_um, np.asarray(col, dtype=float))

        C_ext_m2 = _loglog(self.bulk.C_ext)  # nm^2 -> um^2 -> m^2 (below)
        ssa = _lin(self.bulk.SSA)
        g = np.clip(_lin(self.bulk.g), -1.0, 1.0)

        rho = self.bulk.effective_density_kg_m3
        if rho is None:
            raise ValueError("BulkAerosolOpticsData.effective_density_kg_m3 is required")
        if self.bulk.size_distribution is None:
            raise ValueError("BulkAerosolOpticsData.size_distribution is required")
        r3_nm3 = self.bulk.size_distribution.moment(3)
        vol_m3 = (4.0 / 3.0) * np.pi * r3_nm3 * 1e-27  # nm^3 -> m^3
        mass_per_particle = rho * vol_m3
        beta_ext_per_mass = C_ext_m2 * 1e-12 / mass_per_particle  # um^2 -> m^2

        moments = self._select_and_resample_moments(wl, wl_tab_um, n_legendre)
        return SpeciesOptics(
            beta_ext_per_mass=beta_ext_per_mass,
            ssa=ssa,
            g=g,
            legendre_moments=moments,
        )

    def _select_and_resample_moments(self, wl, wl_tab_um, n_legendre):
        # Hypothesis (spec §4.5): libRadtran explicit .LAYER wants the g_l form.
        src = getattr(self.bulk, "legendre_moments_beta", None)
        if src is None:
            src = getattr(self.bulk, "beta", None)
            if src is not None:
                l_vals = np.arange(src.shape[-1])
                src = src / (2 * l_vals + 1)
        if src is None:
            return None
        n_have = src.shape[-1]
        n_use = min(n_have, n_legendre)
        out = np.zeros((len(wl), n_legendre))
        for l in range(n_use):
            out[:, l] = np.interp(wl, wl_tab_um, src[:, l])
        return out


class MieSpecies(BaseModel):
    """Mie-computed species from refractive index + size distribution."""

    model_config = {"extra": "forbid", "frozen": True, "populate_by_name": True}

    refractive_index: RefractiveIndex
    size_distribution: SizeDistribution
    particle_density_kg_m3: float = Field(gt=0)
    integration_config: IntegrationConfig = Field(default_factory=IntegrationConfig)
    phase_function: Literal["hg", "mie"] = "hg"
    name: str = "MieSpecies"

    @property
    def mass_per_particle_kg(self) -> float:
        """Average particle mass = density * mean volume over the size distribution."""
        from pyradtran.optics.mie import _mass_per_particle_avg

        cfg = self.integration_config
        r = np.logspace(np.log10(cfg.radius_min_um), np.log10(cfg.radius_max_um), cfg.n_radius_grid)
        dn = self.size_distribution.evaluate(r)  # normalized PDF (~∫ dn dr = 1)
        return _mass_per_particle_avg(r, dn, self.particle_density_kg_m3)

    def intensive(self, wl_um: np.ndarray, n_legendre: int = 32) -> SpeciesOptics:
        """Compute mass-normalized intensive optical properties.

        Args:
            wl_um: Wavelengths in micrometers.
            n_legendre: Number of Legendre moments to generate. When
                ``phase_function='mie'``, moments are projected from the real
                Mie phase function (S1/S2 -> Legendre); the default ``'hg'``
                derives them from the Henyey-Greenstein approximation ``g**l``.

        Returns:
            SpeciesOptics with beta_ext_per_mass, ssa, g, and legendre_moments.
        """
        wl = np.asarray(wl_um)
        n_wl = len(wl)
        config = self.integration_config

        r_dense = np.logspace(
            np.log10(config.radius_min_um),
            np.log10(config.radius_max_um),
            config.n_radius_grid,
        )
        self.size_distribution.evaluate(r_dense)

        # Deferred imports avoid circular dependency:
        # mie.py imports SizeDistribution from this module.
        from pyradtran.optics.mie import bhmie, phase_function_to_legendre

        Qext = np.zeros((n_wl, config.n_radius_grid))
        Qsca = np.zeros((n_wl, config.n_radius_grid))
        g = np.zeros((n_wl, config.n_radius_grid))
        moments_grid = (
            np.zeros((n_wl, config.n_radius_grid, n_legendre))
            if self.phase_function == "mie"
            else None
        )
        n_angles = 181

        m_vals = self.refractive_index.at(wl)

        for i_wl in range(n_wl):
            for i_r in range(config.n_radius_grid):
                x = 2.0 * np.pi * r_dense[i_r] / wl[i_wl]
                if self.phase_function == "mie":
                    result = bhmie(x, m_vals[i_wl], n_angles=n_angles)
                    assert moments_grid is not None  # only None when phase_function != "mie"
                    moments_grid[i_wl, i_r, :] = phase_function_to_legendre(
                        result["S1"], result["S2"], result["angles_deg"], n_legendre
                    )
                else:
                    result = bhmie(x, m_vals[i_wl])
                Qext[i_wl, i_r] = result["Qext"]
                Qsca[i_wl, i_r] = result["Qsca"]
                g[i_wl, i_r] = result["g"]

        from pyradtran.optics.mie import integrate_size_distribution

        internal = integrate_size_distribution(
            wavelength_um=wl.tolist(),
            radius_um=r_dense.tolist(),
            Qext=Qext,
            Qsca=Qsca,
            g=g,
            legendre_moments=moments_grid,
            size_distribution=self.size_distribution,
            particle_density_kg_m3=self.particle_density_kg_m3,
            config=config,
            n_legendre=n_legendre,
        )

        return SpeciesOptics(
            beta_ext_per_mass=internal.beta_ext_per_mass,
            ssa=internal.ssa,
            g=internal.g,
            legendre_moments=internal.legendre_moments,
        )


@dataclass
class LayerOptics:
    """Extensive optical properties per layer."""

    tau: NDArray  # (n_wl, n_layer)
    ssa: NDArray  # (n_wl, n_layer)
    g: NDArray  # (n_wl, n_layer)
    legendre_moments: NDArray  # (n_wl, n_layer, n_legendre)


from pyradtran.models.aerosol import AerosolModel


class CompositeAerosol(AerosolModel):
    """Externally mix any number of Piece blocks via a single explicit-file path.

    Each item in ``pieces`` is a :class:`~pyradtran.models.blocks.Piece` (e.g.
    ``PlacedBlock`` or ``DirectLayerOpticsBlock``). They are combined with
    scattering-optical-depth weighting and written as one explicit
    ``.master``/``.LAYER`` set. The old mutual-exclusion rules and the
    single-source shortcut are gone: one path regardless of piece count or type.
    """

    model_config = {
        "extra": "forbid",
        "frozen": True,
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
    }

    pieces: list = Field(min_length=1)  # list[Piece] — PlacedBlock / DirectLayerOpticsBlock
    wavelength_grid_um: list[float] = Field(min_length=1)
    altitude_grid_km: list[float] = Field(min_length=2)
    n_legendre: int = 32
    output_dir: Path | None = None

    @model_validator(mode="after")
    def validate_grids(self) -> CompositeAerosol:
        if self.wavelength_grid_um != sorted(self.wavelength_grid_um):
            raise ValueError("wavelength_grid_um must be strictly ascending")
        if self.altitude_grid_km != sorted(self.altitude_grid_km, reverse=True):
            raise ValueError("altitude_grid_km must be strictly descending")
        if not self.pieces:
            raise ValueError("at least one Piece is required")
        for piece in self.pieces:
            if not hasattr(piece, "to_layer_optics"):
                raise ValueError(
                    f"Each piece must implement to_layer_optics (be a Piece); "
                    f"got {type(piece).__name__}"
                )
        return self

    def evaluate(self, wl_um=None, z_km=None, n_legendre=None) -> LayerOptics:
        """Return the mixed (externally combined) LayerOptics without writing files."""
        from pyradtran.optics.mixing import combine_sources

        wl = np.asarray(self.wavelength_grid_um if wl_um is None else wl_um, dtype=float)
        z = np.asarray(self.altitude_grid_km if z_km is None else z_km, dtype=float)
        nleg = self.n_legendre if n_legendre is None else n_legendre
        layer_optics = [p.to_layer_optics(wl, z, n_legendre=nleg) for p in self.pieces]
        mixed = combine_sources(layer_optics, n_legendre=nleg)
        return LayerOptics(
            tau=mixed["tau"],
            ssa=mixed["ssa"],
            g=mixed["g"],
            legendre_moments=mixed["legendre_moments"],
        )

    def to_uvspec_lines(self) -> list[str]:
        """Single execution path: every Piece -> LayerOptics -> combine -> explicit file."""
        from pyradtran.optics.layer_writer import write_explicit_aerosol
        from pyradtran.optics.mixing import combine_sources

        wl = np.asarray(self.wavelength_grid_um, dtype=float)
        z = np.asarray(self.altitude_grid_km, dtype=float)
        layer_optics = [p.to_layer_optics(wl, z, n_legendre=self.n_legendre) for p in self.pieces]
        mixed = combine_sources(layer_optics, n_legendre=self.n_legendre)
        outdir = self.output_dir if self.output_dir is not None else Path.cwd() / "aerosol"
        source_sigs = [getattr(p, "name", type(p).__name__) for p in self.pieces]
        master_path = write_explicit_aerosol(
            tau=mixed["tau"],
            ssa=mixed["ssa"],
            g=mixed["g"],
            legendre_moments=mixed["legendre_moments"],
            wavelength_um=wl,
            altitude_km=z,
            output_dir=outdir,
            source_signatures=source_sigs,
        )
        return ["aerosol_default", f"aerosol_file explicit {master_path}"]
