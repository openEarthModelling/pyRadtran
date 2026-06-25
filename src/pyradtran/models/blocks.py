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

from pyradtran.models.aerosol_composite import SpeciesOptics


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
