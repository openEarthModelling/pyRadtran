"""LEGO aerosol blocks: unified interface for mixing any aerosol source.

Three-layer protocol (see spec ``2026-06-18-lego-aerosol-blocks-design.md`` §4.2):

  - :class:`AerosolBlock` — ``intensive()``: the mass-normalized optical identity.
  - :class:`VerticalProfile` — column placement (per-layer kg/m^3).
  - :class:`Piece` — ``to_layer_optics()``: the mixing contract (added in Task 3).

This module imports :class:`SpeciesOptics` (and, from Task 3, :class:`LayerOptics`)
from ``aerosol_composite``; the reverse dependency is deferred (inside
``CompositeAerosol`` methods) to avoid a circular import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from pyradtran.models.aerosol import AerosolModifyEntry
from pyradtran.models.aerosol_composite import LayerOptics, SpeciesOptics


@runtime_checkable
class AerosolBlock(Protocol):
    """A mass-normalized aerosol species that can report its intensive optics."""

    name: str

    def intensive(self, wl_um: np.ndarray, n_legendre: int = 32) -> SpeciesOptics: ...

    @property
    def mass_per_particle_kg(self) -> float: ...


@runtime_checkable
class VerticalProfile(Protocol):
    """Column placement: returns mass concentration (kg/m^3) at each altitude."""

    def evaluate(self, altitude_km) -> np.ndarray: ...


@dataclass(frozen=True)
class MassProfile:
    """Explicit per-layer mass concentration (kg/m^3).

    Values are stored in descending-altitude layer order (matching the
    ``altitude_grid_km`` convention used by :class:`CompositeAerosol`).
    ``evaluate`` returns them verbatim; the altitude argument is accepted only
    for interface compatibility with :class:`ExponentialProfile`.
    """

    kg_m3_per_layer: tuple[float, ...]

    def evaluate(self, altitude_km) -> np.ndarray:
        return np.asarray(self.kg_m3_per_layer, dtype=float)


@dataclass(frozen=True)
class ExponentialProfile:
    """``rho(z) = rho0 * exp(-z / H)``, evaluated at the requested altitudes (km)."""

    rho0_kg_m3: float
    scale_height_km: float

    def evaluate(self, altitude_km) -> np.ndarray:
        z = np.asarray(altitude_km, dtype=float)
        return self.rho0_kg_m3 * np.exp(-z / self.scale_height_km)


@dataclass(frozen=True)
class TabulatedProfile:
    """Mass concentration (kg/m^3) tabulated vs altitude (km).

    Linear interpolation in altitude; outside ``[z_min, z_max]`` the value is
    clipped to the nearest table entry. ``z_km`` may be ascending or descending.
    Used to place OPAC preset mass columns, which come on the preset's own grid.
    """

    z_km: tuple[float, ...]
    kg_m3: tuple[float, ...]

    def evaluate(self, altitude_km) -> np.ndarray:
        z = np.asarray(self.z_km, dtype=float)
        rho = np.asarray(self.kg_m3, dtype=float)
        if z[0] > z[-1]:  # np.interp requires ascending x
            z = z[::-1]
            rho = rho[::-1]
        x = np.asarray(altitude_km, dtype=float)
        return np.interp(x, z, rho)


def od_to_mass_profile(
    block: AerosolBlock,
    tau_ref: float,
    ref_nm: float,
    altitude_km,
    scale_height_km: float,
) -> MassProfile:
    """Invert a target column optical depth into an exponential :class:`MassProfile`.

    The returned per-layer masses are chosen so that, evaluated on ``altitude_km``
    with the block's ``beta_ext_per_mass`` at ``ref_nm``, they sum to exactly
    ``tau_ref``::

        tau_ref = beta_ext_per_mass(ref) * sum_layers( rho_layer * dz )

    This is the discrete, grid-exact inversion (the column OD the RT solver will
    actually see), replacing the example-side ``compute_mass_profile`` glue.

    Args:
        block: An :class:`AerosolBlock` with a positive ``beta_ext_per_mass`` at ref.
        tau_ref: Target column optical depth at ``ref_nm`` (dimensionless).
        ref_nm: Reference wavelength in nm.
        altitude_km: Layer-boundary altitudes in km, strictly descending.
        scale_height_km: Exponential scale height H (km).

    Returns:
        A :class:`MassProfile` whose layers follow the exponential shape and
        whose column OD equals ``tau_ref``.
    """
    alt = np.asarray(altitude_km, dtype=float)
    ref_um = ref_nm / 1000.0
    beta_ext = float(block.intensive(np.array([ref_um])).beta_ext_per_mass[0])  # m^2/kg
    if beta_ext <= 0.0:
        raise ValueError(
            "block has non-positive beta_ext_per_mass at the reference wavelength; "
            "cannot invert a target optical depth"
        )
    centers = 0.5 * (alt[:-1] + alt[1:])
    dz_m = -np.diff(alt) * 1000.0  # km -> m (descending grid -> positive dz)
    shape = np.exp(-centers / scale_height_km)
    rho0 = tau_ref / (beta_ext * float(np.sum(shape * dz_m)))
    rho_per_layer = rho0 * shape
    return MassProfile(kg_m3_per_layer=tuple(rho_per_layer.tolist()))


# Piece / PlacedBlock / DirectLayerOpticsBlock are added in Tasks 3 & 4.


@runtime_checkable
class Piece(Protocol):
    """The mixing contract: anything :class:`CompositeAerosol` accepts produces
    per-layer :class:`LayerOptics` on the shared (wavelength, altitude) grid."""

    name: str

    def to_layer_optics(
        self, wl_um: np.ndarray, altitude_km, n_legendre: int = 32
    ) -> LayerOptics: ...


@dataclass(frozen=True)
class PlacedBlock:
    """Intensity-route piece: a mass-normalized species placed in the column.

    Ports the non-OPAC path of ``LoadedSpecies.evaluate``: the block's intensive
    optics are weighted by the profile's per-layer mass and the layer thickness
    to give extensive per-layer optical properties.
    """

    block: AerosolBlock
    profile: VerticalProfile
    modify: tuple[AerosolModifyEntry, ...] = ()

    @property
    def name(self) -> str:
        return self.block.name

    def to_layer_optics(
        self, wl_um: np.ndarray, altitude_km, n_legendre: int = 32
    ) -> LayerOptics:
        wl = np.asarray(wl_um, dtype=float)
        z = np.asarray(altitude_km, dtype=float)
        n_wl = wl.shape[0]

        intensive = self.block.intensive(wl, n_legendre=n_legendre)
        beta_ext = intensive.beta_ext_per_mass  # (n_wl,)
        ssa = intensive.ssa
        g = intensive.g

        dz_m = -np.diff(z) * 1000.0  # km -> m (descending grid -> positive dz)
        n_layer = dz_m.shape[0]
        centers = 0.5 * (z[:-1] + z[1:])
        mass = np.clip(self.profile.evaluate(centers), 0.0, None)  # (n_layer,)

        tau = np.zeros((n_wl, n_layer))
        for i in range(n_wl):
            tau[i, :] = beta_ext[i] * mass * dz_m

        ssa_layer = np.asarray(ssa)[:, None] * np.ones((n_wl, n_layer))
        g_layer = np.asarray(g)[:, None] * np.ones((n_wl, n_layer))

        if intensive.legendre_moments is not None:
            n_mom = intensive.legendre_moments.shape[1]
            moments_layer = np.zeros((n_wl, n_layer, n_legendre))
            for l in range(min(n_mom, n_legendre)):
                moments_layer[:, :, l] = intensive.legendre_moments[:, l][:, None]
        else:
            # Henyey-Greenstein fill: g_l = g^l (matches LoadedSpecies / _fill_hg_moments)
            moments_layer = np.zeros((n_wl, n_layer, n_legendre))
            for l in range(n_legendre):
                moments_layer[:, :, l] = g_layer ** l

        # Apply per-block modify (tau / ssa / gg scale or set).
        for entry in self.modify:
            if entry.variable == "tau":
                if entry.action == "scale":
                    tau = tau * entry.value
                else:
                    tau = np.full_like(tau, entry.value)
            elif entry.variable == "ssa":
                if entry.action == "scale":
                    ssa_layer = np.clip(ssa_layer * entry.value, 0.0, 1.0)
                else:
                    ssa_layer = np.full_like(ssa_layer, entry.value)
            elif entry.variable == "gg":
                if entry.action == "scale":
                    g_layer = np.clip(g_layer * entry.value, -1.0, 1.0)
                else:
                    g_layer = np.full_like(g_layer, entry.value)

        return LayerOptics(
            tau=tau, ssa=ssa_layer, g=g_layer, legendre_moments=moments_layer
        )


@dataclass(frozen=True)
class DirectLayerOpticsBlock:
    """Direct-route piece: a pre-computed explicit aerosol file set (.master + .LAYER).

    The file already contains per-layer tau/ssa/pmom, so no :class:`VerticalProfile`
    is needed — ``to_layer_optics`` parses the file directly. Wavelength resampling
    is not performed (v1): the requested grid must match the file's grid.
    """

    master_path: str
    name: str = "explicit_file"

    def to_layer_optics(
        self, wl_um: np.ndarray, altitude_km, n_legendre: int = 32
    ) -> LayerOptics:
        from pyradtran.optics.layer_writer import read_explicit_aerosol

        tau, ssa, g, moments, _wl_file, _alt_file = read_explicit_aerosol(self.master_path)
        req_wl = np.asarray(wl_um)
        if tau.shape[0] != req_wl.shape[0]:
            raise ValueError(
                f"DirectLayerOpticsBlock file has {tau.shape[0]} wavelengths but "
                f"{req_wl.shape[0]} requested; explicit files are not auto-resampled in v1."
            )
        n_have = moments.shape[2]
        out = np.zeros((tau.shape[0], tau.shape[1], n_legendre))
        for l in range(min(n_have, n_legendre)):
            out[:, :, l] = moments[:, :, l]
        return LayerOptics(tau=tau, ssa=ssa, g=g, legendre_moments=out)
