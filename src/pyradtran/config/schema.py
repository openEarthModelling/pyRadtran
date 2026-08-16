"""YAML configuration schema (``config_version: 1``).

A config file describes one radiative-transfer experiment:

- ``scene`` mirrors the :class:`~pyradtran.scene.Scene` builder kwargs;
- ``aerosol`` assembles LEGO blocks (discriminated on ``kind``) with a
  column ``placement``;
- ``analysis`` declares post-processing intents (orchestrated by
  :mod:`pyradtran.config.orchestrator`).

The loader (:mod:`pyradtran.config.loader`) maps a validated config onto the
same builder calls the Python API uses, so a YAML config and an equivalent
API-built scene produce identical uvspec input text.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from pyradtran.models.aerosol import OpacPresetName
from pyradtran.models.aerosol_composite import (
    IntegrationConfig,
    RefractiveIndex,
    SizeDistribution,
)

CONFIG_VERSION = 1


class _Strict(BaseModel):
    """Base for config models: reject unknown keys (typo protection)."""

    model_config = ConfigDict(extra="forbid")


# --- Scene section (builder-kwargs pass-through) ---


class SceneSection(_Strict):
    """Mirror of the Scene builder kwargs.

    Each sub-dict is passed verbatim to the corresponding ``Scene.set_*()``
    call. Unknown keys are rejected downstream by the frozen, extra=forbid
    uvspec option models at load time, so typos fail loudly.

    Keys:

    - ``atmosphere``: ``profile`` (+ ``altitude``, ``mol_modify``, ...)
    - ``source``: ``sza`` (+ ``phi0``, ...); ``source: solar|thermal`` optional
    - ``wavelength``: ``min_nm``, ``max_nm`` (+ ``unit``, ...)
    - ``solver``: ``method``, ``streams`` (+ ``disort_intcor``, ``pseudospherical``, ...)
    - ``surface``: ``albedo`` (+ BRDF/BPDF alternatives)
    - ``output``: ``quantities``, ``format``, ``zout``, ``heating_rate``,
      ``process`` (e.g. ``integrate`` for band-integrated fluxes), ...
    """

    atmosphere: dict[str, Any]
    source: dict[str, Any]
    wavelength: dict[str, Any]
    solver: dict[str, Any] = Field(default_factory=dict)
    surface: dict[str, Any] | None = None
    output: dict[str, Any] = Field(default_factory=dict)


# --- Column placements ---


class OdInversionPlacement(_Strict):
    """Invert a target column optical depth into an exponential mass profile.

    The placed block's column OD at ``ref_nm`` equals ``tau_ref`` exactly on
    the composite's altitude grid (``od_to_mass_profile``).
    """

    kind: Literal["od_inversion"]
    tau_ref: float = Field(gt=0)
    ref_nm: float = Field(default=550.0, gt=0)
    scale_height_km: float = Field(gt=0)


class MassPlacement(_Strict):
    """Explicit per-layer mass concentration (kg/m^3), descending-altitude order."""

    kind: Literal["mass"]
    kg_m3_per_layer: list[float] = Field(min_length=1)


class ExponentialPlacement(_Strict):
    """``rho(z) = rho0 * exp(-z / H)``."""

    kind: Literal["exponential"]
    rho0_kg_m3: float = Field(ge=0)
    scale_height_km: float = Field(gt=0)


class TabulatedPlacement(_Strict):
    """Mass concentration tabulated vs altitude (linear interpolation)."""

    kind: Literal["tabulated"]
    z_km: list[float] = Field(min_length=1)
    kg_m3: list[float] = Field(min_length=1)


PlacementSpec = Annotated[
    OdInversionPlacement | MassPlacement | ExponentialPlacement | TabulatedPlacement,
    Field(discriminator="kind"),
]


# --- Blocks ---


class MieBlockSpec(_Strict):
    """Mie-computed species: refractive index + size distribution + placement."""

    kind: Literal["mie"]
    name: str = "MieSpecies"
    refractive_index: RefractiveIndex
    size_distribution: SizeDistribution
    particle_density_kg_m3: float = Field(gt=0)
    phase_function: Literal["hg", "mie"] = "hg"
    integration: IntegrationConfig | None = None
    placement: PlacementSpec


class BulkBlockSpec(_Strict):
    """Bulk optics NetCDF written by aerosol3D ``BulkAerosolOpticsData``.

    Requires the aerosol3D package at load time (only for the reader).
    """

    kind: Literal["bulk"]
    name: str = "BulkSpecies"
    file: str
    placement: PlacementSpec


class OpacPresetBlockSpec(_Strict):
    """OPAC preset mixture; self-placed via its own preset mass column."""

    kind: Literal["opac_preset"]
    name: str = "OPAC preset"
    preset: OpacPresetName
    rh_pct: float = Field(default=50.0, ge=0, le=100)
    species_names: list[str] | None = None
    data_path: str | None = None
    n_legendre: int = Field(default=32, ge=1)


class ExplicitLayerBlockSpec(_Strict):
    """Pre-computed explicit aerosol file set (``.master`` + ``.LAYER``)."""

    kind: Literal["explicit_layer"]
    name: str = "explicit_file"
    master_path: str


BlockSpec = Annotated[
    MieBlockSpec | BulkBlockSpec | OpacPresetBlockSpec | ExplicitLayerBlockSpec,
    Field(discriminator="kind"),
]


class AerosolSection(_Strict):
    """Composite-aerosol assembly: shared grids + LEGO blocks."""

    wavelength_grid_um: list[float] = Field(min_length=1)
    altitude_grid_km: list[float] = Field(min_length=2)
    n_legendre: int = Field(default=32, ge=1)
    output_dir: str | None = None
    blocks: list[BlockSpec] = Field(min_length=1)


# --- Analysis section ---


class EnergyConservationSpec(_Strict):
    """Column energy identity assertion (``assert_energy_conservation``)."""

    tol: float = Field(default=0.05, gt=0)
    albedo: float | None = Field(default=None, ge=0, le=1)  # None -> scene surface


class AnalysisSection(_Strict):
    """Post-processing intents executed after the main run.

    - ``energy_conservation``: assert the column energy budget identity.
    - ``drf``: direct radiative forcing vs an auto-built no-aerosol baseline.
    - ``heating``: second uvspec invocation in heating-rate mode (libRadtran
      cannot emit fluxes and heating in one run) and merge into the result.
    - ``attribution``: leave-one-out component attribution (N+1 runs).
    - ``plots``: names resolved against the viz plot registry (T3).
    - ``save_netcdf``: filename for the merged result dataset.
    """

    energy_conservation: EnergyConservationSpec | None = None
    drf: bool = False
    heating: bool = False
    attribution: bool = False
    plots: list[str] = Field(default_factory=list)
    save_netcdf: str | None = None


class PyRadtranConfig(_Strict):
    """Top-level config document (``config_version: 1``)."""

    config_version: Literal[1]
    name: str = "unnamed"
    scene: SceneSection
    aerosol: AerosolSection | None = None
    analysis: AnalysisSection | None = None
