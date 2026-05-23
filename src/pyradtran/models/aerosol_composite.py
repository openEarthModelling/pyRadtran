"""Composable aerosol optical-property models (Tier 1–4).

See design spec: docs/superpowers/specs/2026-04-27-composite-aerosol-design.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

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
    def validate_shapes(self) -> ParticleOptics:
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

        if self.legendre_moments is not None and self.legendre_moments.shape[:2] != expected:
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
    ) -> ParticleOptics:
        """Build from cross-sections (convert to Q-factors)."""
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

    @classmethod
    def from_aerosol3d(cls, data) -> ParticleOptics:
        """Create ParticleOptics from Aerosol3D AerosolOpticsData.

        Handles unit conversion (nm -> um, nm^2 -> um^2) and Legendre
        convention (prefers beta_l, falls back to k_l conversion).

        Args:
            data: Object with wavelength_nm, C_ext, C_sca, g, r_eff_nm,
                  n_legendre, legendre_moments_beta, legendre_moments fields.

        Returns:
            ParticleOptics ready for use in PrecomputedSpecies.
        """
        wavelengths_um = (data.wavelength_nm / 1000.0).tolist()
        r_eff_um = data.r_eff_nm / 1000.0

        legendre = None
        if data.legendre_moments_beta is not None:
            legendre = data.legendre_moments_beta.reshape((-1, 1, data.n_legendre))
        elif data.legendre_moments is not None:
            l = np.arange(data.n_legendre)
            legendre = (data.legendre_moments / (2 * l + 1)).reshape((-1, 1, data.n_legendre))

        return cls.from_cross_sections(
            wavelength_um=wavelengths_um,
            radius_um=[r_eff_um],
            Cext_um2=(data.C_ext * 1e-6).reshape((-1, 1)),
            Csca_um2=(data.C_sca * 1e-6).reshape((-1, 1)),
            g=data.g.reshape((-1, 1)),
            legendre_moments=legendre,
        )


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


class Species(Protocol):
    """Protocol for Tier 2 species classes."""

    def intensive(self, wl_um: np.ndarray, n_legendre: int = 32) -> SpeciesOptics: ...


class MieSpecies(BaseModel):
    """Mie-computed species from refractive index + size distribution."""

    model_config = {"extra": "forbid", "frozen": True, "populate_by_name": True}

    refractive_index: RefractiveIndex
    size_distribution: SizeDistribution
    particle_density_kg_m3: float = Field(gt=0)
    integration_config: IntegrationConfig = Field(default_factory=IntegrationConfig)

    def intensive(self, wl_um: np.ndarray, n_legendre: int = 32) -> SpeciesOptics:
        """Compute mass-normalized intensive optical properties.

        Args:
            wl_um: Wavelengths in micrometers.
            n_legendre: Number of Legendre moments to generate. Because the
                bhmie() path does not compute angular scattering by default,
                moments are derived from the Henyey-Greenstein phase function
                rather than a full Mie phase function.

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
        # mie.py imports ParticleOptics/SizeDistribution from this module.
        from pyradtran.optics.mie import bhmie

        Qext = np.zeros((n_wl, config.n_radius_grid))
        Qsca = np.zeros((n_wl, config.n_radius_grid))
        g = np.zeros((n_wl, config.n_radius_grid))

        m_vals = self.refractive_index.at(wl)

        for i_wl in range(n_wl):
            for i_r in range(config.n_radius_grid):
                x = 2.0 * np.pi * r_dense[i_r] / wl[i_wl]
                result = bhmie(x, m_vals[i_wl])
                Qext[i_wl, i_r] = result["Qext"]
                Qsca[i_wl, i_r] = result["Qsca"]
                g[i_wl, i_r] = result["g"]

        from pyradtran.optics.mie import integrate_size_distribution

        particle_optics = ParticleOptics(
            wavelength_um=wl.tolist(),
            radius_um=r_dense.tolist(),
            Qext=Qext,
            Qsca=Qsca,
            g=g,
            legendre_moments=None,
        )

        internal = integrate_size_distribution(
            particle_optics=particle_optics,
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


class PrecomputedSpecies(BaseModel):
    """Species from precomputed ParticleOptics + size distribution."""

    model_config = {"extra": "forbid", "frozen": True, "populate_by_name": True}

    particle_optics: ParticleOptics
    size_distribution: SizeDistribution
    particle_density_kg_m3: float = Field(gt=0)
    integration_config: IntegrationConfig = Field(default_factory=IntegrationConfig)

    def intensive(self, wl_um: np.ndarray, n_legendre: int = 32) -> SpeciesOptics:
        """Integrate precomputed Q-factors over size distribution."""
        from pyradtran.optics.mie import integrate_size_distribution

        # Verify wavelength coverage
        wl = np.asarray(wl_um)
        wl_tab = np.asarray(self.particle_optics.wavelength_um)
        if np.any(wl < wl_tab[0]) or np.any(wl > wl_tab[-1]):
            raise ValueError(
                f"Wavelength {wl.min():.4f}–{wl.max():.4f} um outside "
                f"particle optics range {wl_tab[0]:.4f}–{wl_tab[-1]:.4f} um"
            )

        # Interpolate ParticleOptics to requested wavelengths (log-linear in Q)
        n_r = len(self.particle_optics.radius_um)
        n_req = len(wl)
        Qext_interp = np.zeros((n_req, n_r))
        Qsca_interp = np.zeros((n_req, n_r))
        g_interp = np.zeros((n_req, n_r))

        for i_r in range(n_r):
            log_Qext = np.log(np.clip(self.particle_optics.Qext[:, i_r], 1e-30, None))
            Qext_interp[:, i_r] = np.exp(np.interp(wl, wl_tab, log_Qext))

            log_Qsca = np.log(np.clip(self.particle_optics.Qsca[:, i_r], 1e-30, None))
            Qsca_interp[:, i_r] = np.exp(np.interp(wl, wl_tab, log_Qsca))

            g_interp[:, i_r] = np.clip(
                np.interp(wl, wl_tab, self.particle_optics.g[:, i_r]), -1.0, 1.0
            )

        legendre_interp = None
        if self.particle_optics.legendre_moments is not None:
            n_mom = self.particle_optics.legendre_moments.shape[2]
            legendre_interp = np.zeros((n_req, n_r, n_mom))
            for i_r in range(n_r):
                for l in range(n_mom):
                    legendre_interp[:, i_r, l] = np.interp(
                        wl, wl_tab, self.particle_optics.legendre_moments[:, i_r, l]
                    )

        particle_optics_interp = ParticleOptics(
            wavelength_um=wl.tolist(),
            radius_um=self.particle_optics.radius_um,
            Qext=Qext_interp,
            Qsca=Qsca_interp,
            g=g_interp,
            legendre_moments=legendre_interp,
        )

        internal = integrate_size_distribution(
            particle_optics=particle_optics_interp,
            size_distribution=self.size_distribution,
            particle_density_kg_m3=self.particle_density_kg_m3,
            config=self.integration_config,
            n_legendre=n_legendre,
        )

        return SpeciesOptics(
            beta_ext_per_mass=internal.beta_ext_per_mass,
            ssa=internal.ssa,
            g=internal.g,
            legendre_moments=internal.legendre_moments,
        )


class OPACSpecies(BaseModel):
    """OPAC species from a pre-computed libRadtran netCDF optical-property file.

    The netCDF file is expected to contain ``output_dtauc``, ``output_ssalb``,
    and ``output_pmom`` variables (libRadtran ``write_optical_properties`` format).
    """

    model_config = {"extra": "forbid", "frozen": True, "populate_by_name": True}

    netcdf_path: str = Field(min_length=1)
    # Optional: if the file contains multiple wavelengths, this selects one
    wavelength_nm: float | None = None

    def intensive(self, wl_um: np.ndarray, n_legendre: int = 32) -> SpeciesOptics:
        """Read netCDF and return intensive properties.

        For a netCDF with ``nlyr`` layers, the returned ``beta_ext_per_mass``
        assumes the optical depth is distributed uniformly across layers for
        the purpose of SpeciesOptics (the actual layer geometry is applied
        later in LoadedSpecies).
        """
        from pyradtran.optics.opac_tables import read_opac_netcdf

        data = read_opac_netcdf(self.netcdf_path)
        dtauc = data["dtauc"]  # (nlyr,)
        ssalb = data["ssalb"]  # (nlyr,)
        pmom = data["pmom"]  # (nlyr, nmom)

        # For SpeciesOptics we need per-wavelength values.
        # If the netCDF has a single wavelength, average over layers.
        # If it has per-layer data at one wavelength, average.
        # The spec assumes one wavelength per netCDF file.
        beta_ext = np.mean(dtauc)  # placeholder -- actual scaling done in LoadedSpecies
        ssa_mean = np.mean(ssalb)
        g_mean = np.mean(pmom[:, 1]) if pmom.shape[1] > 1 else 0.0

        # Build Legendre moments: k_l = (2l+1) * g_l where g_l are expansion coefficients
        # pmom[:, l] corresponds to moment l (0-indexed)
        n_mom_available = pmom.shape[1]
        n_mom = min(n_mom_available, n_legendre)
        legendre_moments = np.zeros((1, n_legendre))
        for l in range(n_mom):
            legendre_moments[0, l] = np.mean(pmom[:, l])

        return SpeciesOptics(
            beta_ext_per_mass=np.array([beta_ext]),
            ssa=np.array([ssa_mean]),
            g=np.array([g_mean]),
            legendre_moments=legendre_moments,
        )


@dataclass
class LayerOptics:
    """Extensive optical properties per layer."""

    tau: NDArray  # (n_wl, n_layer)
    ssa: NDArray  # (n_wl, n_layer)
    g: NDArray  # (n_wl, n_layer)
    legendre_moments: NDArray  # (n_wl, n_layer, n_legendre)


class LoadedSpecies(BaseModel):
    """Tier 3: extensive in-atmosphere optical properties."""

    model_config = {"extra": "forbid", "frozen": True, "populate_by_name": True}

    species: MieSpecies | PrecomputedSpecies | OPACSpecies
    mass_profile_kg_m3: list[float] = Field(min_length=1)
    altitude_km: list[float] = Field(min_length=2)  # layer boundaries, descending
    rh_profile: list[float] | None = None  # required for OPACSpecies

    @model_validator(mode="after")
    def validate_profiles(self) -> LoadedSpecies:
        n_layers = len(self.altitude_km) - 1
        if len(self.mass_profile_kg_m3) != n_layers:
            raise ValueError(
                f"mass_profile_kg_m3 length ({len(self.mass_profile_kg_m3)}) must equal "
                f"number of layers ({n_layers}) from altitude_km"
            )
        if self.altitude_km != sorted(self.altitude_km, reverse=True):
            raise ValueError("altitude_km must be strictly descending")
        if isinstance(self.species, OPACSpecies) and self.rh_profile is None:
            raise ValueError("rh_profile is required when species is OPACSpecies")
        if self.rh_profile is not None and len(self.rh_profile) != n_layers:
            raise ValueError(
                f"rh_profile length ({len(self.rh_profile)}) "
                f"must equal number of layers ({n_layers})"
            )
        return self

    def evaluate(
        self,
        wl_um: np.ndarray,
        z_km: np.ndarray,
        n_legendre: int = 32,
    ) -> LayerOptics:
        """Evaluate optical properties on (wavelength, altitude) grid.

        Args:
            wl_um: Wavelength grid in um, strictly ascending.
            z_km: Altitude grid in km, strictly descending (layer centers or boundaries).
            n_legendre: Number of Legendre moments.

        Returns:
            LayerOptics with shape (n_wl, n_layer).
        """
        wl = np.asarray(wl_um)
        z = np.asarray(z_km)
        n_wl = len(wl)

        # Get intensive properties from species
        intensive = self.species.intensive(wl, n_legendre=n_legendre)
        beta_ext = intensive.beta_ext_per_mass  # (n_wl,)
        ssa = intensive.ssa  # (n_wl,)
        g = intensive.g  # (n_wl,)

        # Compute layer thicknesses from altitude grid
        # z is descending: z[0] > z[1] > ...; thickness = z[i] - z[i+1]
        dz_m = -np.diff(z) * 1000.0  # km -> m (np.diff gives z[i+1]-z[i], negate for descending)
        n_layer = len(dz_m)

        # Interpolate mass profile to layer centers
        alt_centers = 0.5 * (np.array(self.altitude_km[:-1]) + np.array(self.altitude_km[1:]))
        layer_centers = 0.5 * (z[:-1] + z[1:])
        mass_interp = np.interp(
            layer_centers,
            alt_centers[::-1],  # ascending for np.interp
            np.array(self.mass_profile_kg_m3)[::-1],
        )
        mass_interp = np.clip(mass_interp, 0.0, None)  # mass cannot be negative

        # Compute optical depth per layer
        tau = np.zeros((n_wl, n_layer))
        if isinstance(self.species, OPACSpecies):
            # OPACSpecies returns per-layer optical depth directly in
            # beta_ext_per_mass (netCDF dtauc is already dimensionless).
            # Interpolate to the output layer grid using the species altitude grid.
            species_alt_centers = alt_centers
            output_alt_centers = layer_centers
            for i_wl in range(n_wl):
                tau[i_wl, :] = np.interp(
                    output_alt_centers,
                    species_alt_centers[::-1],
                    np.full(n_layer, beta_ext[i_wl])[::-1],
                )
        else:
            # beta_ext [m^2/kg] * mass [kg/m^3] * dz [m] = dimensionless
            for i_wl in range(n_wl):
                tau[i_wl, :] = beta_ext[i_wl] * mass_interp * dz_m

        # Broadcast ssa and g to (n_wl, n_layer)
        ssa_layer = np.broadcast_to(ssa[:, np.newaxis], (n_wl, n_layer))
        g_layer = np.broadcast_to(g[:, np.newaxis], (n_wl, n_layer))

        # Legendre moments
        if intensive.legendre_moments is not None:
            n_mom = intensive.legendre_moments.shape[1]
            legendre_layer = np.zeros((n_wl, n_layer, n_legendre))
            for l in range(min(n_mom, n_legendre)):
                legendre_layer[:, :, l] = np.broadcast_to(
                    intensive.legendre_moments[:, l][:, np.newaxis],
                    (n_wl, n_layer),
                )
            # Zero-pad remaining moments
            for l in range(n_mom, n_legendre):
                legendre_layer[:, :, l] = 0.0
        else:
            # Henyey-Greenstein fill: k_l = (2l+1) g^l
            legendre_layer = np.zeros((n_wl, n_layer, n_legendre))
            for l in range(n_legendre):
                legendre_layer[:, :, l] = (2 * l + 1) * g_layer**l

        return LayerOptics(
            tau=tau,
            ssa=ssa_layer,
            g=g_layer,
            legendre_moments=legendre_layer,
        )


from pyradtran.models.aerosol import AerosolModel, ExternalFile, OpacCustom, OpacPreset


class CompositeAerosol(AerosolModel):
    """Tier 4: composite aerosol scene mixing multiple sources."""

    model_config = {"extra": "forbid", "frozen": True, "populate_by_name": True}

    sources: list = Field(min_length=1)
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

        # Validate source types
        for src in self.sources:
            if not isinstance(src, LoadedSpecies | OpacPreset | OpacCustom | ExternalFile):
                raise ValueError(
                    f"Invalid source type: {type(src).__name__}. "
                    f"Expected LoadedSpecies, OpacPreset, OpacCustom, or ExternalFile."
                )

        # Disallow mixing incompatible source types
        preset_sources = [s for s in self.sources if isinstance(s, OpacPreset | OpacCustom)]
        loaded_sources = [s for s in self.sources if isinstance(s, LoadedSpecies)]
        external_sources = [s for s in self.sources if isinstance(s, ExternalFile)]

        if len(preset_sources) > 1:
            raise ValueError("Only one OpacPreset/OpacCustom source allowed in CompositeAerosol")
        if len(loaded_sources) > 0 and len(preset_sources) > 0:
            raise ValueError(
                "Cannot mix LoadedSpecies with OpacPreset/OpacCustom in CompositeAerosol"
            )
        if len(loaded_sources) > 0 and len(external_sources) > 0:
            raise ValueError("Cannot mix LoadedSpecies with ExternalFile in CompositeAerosol")
        if len(preset_sources) > 0 and len(external_sources) > 0:
            raise ValueError(
                "Cannot mix OpacPreset/OpacCustom with ExternalFile in CompositeAerosol"
            )

        return self

    def to_uvspec_lines(self) -> list[str]:
        """Generate ``aerosol_file explicit <master>`` line.

        For single preset/external sources, delegates directly to their
        ``to_uvspec_lines()`` to avoid the explicit-file overhead.
        """
        from pyradtran.optics.layer_writer import write_explicit_aerosol
        from pyradtran.optics.mixing import combine_sources

        # Shortcut: single non-composite source
        if len(self.sources) == 1 and not isinstance(self.sources[0], LoadedSpecies):
            return self.sources[0].to_uvspec_lines()

        wl = np.asarray(self.wavelength_grid_um)
        z = np.asarray(self.altitude_grid_km)

        # Evaluate each LoadedSpecies
        layer_optics_list = []
        for src in self.sources:
            if isinstance(src, LoadedSpecies):
                layer_optics_list.append(src.evaluate(wl, z, n_legendre=self.n_legendre))

        # Mix
        mixed = combine_sources(layer_optics_list, n_legendre=self.n_legendre)

        # Write files
        outdir = self.output_dir
        if outdir is None:
            outdir = Path.cwd() / "aerosol"

        source_sigs = [str(type(s).__name__) for s in self.sources]
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
