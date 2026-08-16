"""Config loader: YAML document → validated config → Scene + CompositeAerosol.

The loader calls the same builder chain as an API user would, so a YAML
config and the equivalent Python code produce identical uvspec input text.
This equivalence is enforced by the round-trip regression test
(``tests/test_config_roundtrip.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from pyradtran.config.schema import (
    CONFIG_VERSION,
    AerosolSection,
    BlockSpec,
    BulkBlockSpec,
    ExponentialPlacement,
    MassPlacement,
    MieBlockSpec,
    OdInversionPlacement,
    OpacPresetBlockSpec,
    PyRadtranConfig,
    SceneSection,
)
from pyradtran.models.aerosol import OpacPreset
from pyradtran.models.aerosol_composite import (
    BulkSpecies,
    CompositeAerosol,
    IntegrationConfig,
    MieSpecies,
)
from pyradtran.models.blocks import (
    DirectLayerOpticsBlock,
    ExponentialProfile,
    MassProfile,
    PlacedBlock,
    TabulatedProfile,
    od_to_mass_profile,
)
from pyradtran.scene import Scene


@dataclass
class LoadedConfig:
    """A validated config plus its built runtime objects."""

    config: PyRadtranConfig
    scene: Scene
    aerosol: CompositeAerosol | None


def load_config(source: str | Path | dict | PyRadtranConfig) -> LoadedConfig:
    """Load a config from a YAML file path, a raw dict, or a built config.

    Raises :class:`pydantic.ValidationError` / ``ValueError`` on invalid input
    (unknown ``config_version``, unknown block ``kind``, typo'd keys, ...).
    """
    if isinstance(source, PyRadtranConfig):
        cfg = source
    else:
        if isinstance(source, str | Path):
            path = Path(source)
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError(f"{path}: config must be a YAML mapping")
        elif isinstance(source, dict):
            raw = source
        else:
            raise TypeError(f"unsupported config source: {type(source)!r}")
        cfg = PyRadtranConfig.model_validate(raw)
    aerosol = build_aerosol(cfg.aerosol) if cfg.aerosol is not None else None
    scene = build_scene(cfg.scene, aerosol=aerosol)
    return LoadedConfig(config=cfg, scene=scene, aerosol=aerosol)


def export_config(source: dict | PyRadtranConfig, path: str | Path) -> Path:
    """Validate and serialize a config as canonical YAML.

    Accepts a raw dict (validated on the way out) or a built config; returns
    the written path. The output re-loads identically
    (``load_config(export_config(cfg, p))`` builds the same scene).
    """
    cfg = source if isinstance(source, PyRadtranConfig) else PyRadtranConfig.model_validate(source)
    out = Path(path)
    out.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
    return out


# --- Scene assembly (mirrors examples/multicomponent_viz/canonical.py) ---


def build_scene(section: SceneSection, aerosol: CompositeAerosol | None = None) -> Scene:
    """Map a validated ``scene`` section onto the Scene builder.

    Calls mirror the canonical example's builder chain exactly so the
    generated uvspec text is byte-identical for equivalent inputs.
    """
    scene = Scene().set_atmosphere(**section.atmosphere)

    src = dict(section.source)
    src_kind = src.pop("source", "solar")
    if src_kind == "solar":
        if "sza" not in src:
            raise ValueError("scene.source: 'sza' is required for a solar source")
        scene = scene.set_source_solar(sza=src.pop("sza"), **src)
    elif src_kind == "thermal":
        scene = scene.set_source_thermal(**src)
    else:
        raise ValueError(f"scene.source: unknown source {src_kind!r}")

    wl = dict(section.wavelength)
    if "min_nm" not in wl:
        raise ValueError("scene.wavelength: 'min_nm' is required")
    scene = scene.set_wavelength(wl.pop("min_nm"), wl.pop("max_nm", None), **wl)

    sol = dict(section.solver)
    scene = scene.set_solver(
        method=sol.pop("method", "disort"),
        streams=sol.pop("streams", 16),
        disort_intcor=sol.pop("disort_intcor", None),
        pseudospherical=sol.pop("pseudospherical", False),
        **sol,
    )

    if section.surface is not None:
        scene = scene.set_surface(**section.surface)
    if section.output:
        scene = scene.set_output(**section.output)
    if aerosol is not None:
        scene = scene.set_aerosol(aerosol)
    return scene


# --- Aerosol assembly (mirrors canonical.build_composite) ---


def build_aerosol(section: AerosolSection) -> CompositeAerosol:
    """Assemble the LEGO blocks described by the ``aerosol`` section."""
    pieces = [piece for spec in section.blocks for piece in _pieces_for(spec, section)]
    return CompositeAerosol(
        pieces=pieces,
        wavelength_grid_um=list(section.wavelength_grid_um),
        altitude_grid_km=list(section.altitude_grid_km),
        n_legendre=section.n_legendre,
        output_dir=Path(section.output_dir) if section.output_dir else None,
    )


def _pieces_for(spec: BlockSpec, section: AerosolSection) -> list:
    if isinstance(spec, MieBlockSpec):
        species = MieSpecies(
            refractive_index=spec.refractive_index,
            size_distribution=spec.size_distribution,
            particle_density_kg_m3=spec.particle_density_kg_m3,
            integration_config=(
                spec.integration if spec.integration is not None else IntegrationConfig()
            ),
            phase_function=spec.phase_function,
            name=spec.name,
        )
        return [PlacedBlock(block=species, profile=_placement(spec.placement, species, section))]

    if isinstance(spec, BulkBlockSpec):
        species = BulkSpecies(bulk=_read_bulk(spec.file), name=spec.name)
        return [PlacedBlock(block=species, profile=_placement(spec.placement, species, section))]

    if isinstance(spec, OpacPresetBlockSpec):
        preset = OpacPreset(
            name=spec.preset,
            rh_pct=spec.rh_pct,
            species_names=spec.species_names,
            data_path=spec.data_path,
            n_legendre=spec.n_legendre,
        )
        return preset.to_placed_blocks()

    # ExplicitLayerBlockSpec: already a Piece (no placement needed)
    return [DirectLayerOpticsBlock(master_path=spec.master_path, name=spec.name)]


def _read_bulk(file: str):
    """Read an aerosol3D bulk-optics NetCDF (lazy import; optional dependency)."""
    try:
        from Aerosol3D.bulk.datastructs import BulkAerosolOpticsData
    except ImportError as e:  # pragma: no cover - depends on optional install
        raise ImportError(
            "config block kind 'bulk' requires the aerosol3D package "
            "(Aerosol3D.bulk.datastructs.BulkAerosolOpticsData)"
        ) from e
    return BulkAerosolOpticsData.from_netcdf(file)


def _placement(placement, species, section: AerosolSection):
    if isinstance(placement, OdInversionPlacement):
        return od_to_mass_profile(
            species,
            tau_ref=placement.tau_ref,
            ref_nm=placement.ref_nm,
            altitude_km=list(section.altitude_grid_km),
            scale_height_km=placement.scale_height_km,
        )
    if isinstance(placement, MassPlacement):
        return MassProfile(kg_m3_per_layer=tuple(placement.kg_m3_per_layer))
    if isinstance(placement, ExponentialPlacement):
        return ExponentialProfile(
            rho0_kg_m3=placement.rho0_kg_m3, scale_height_km=placement.scale_height_km
        )
    # TabulatedPlacement
    return TabulatedProfile(z_km=tuple(placement.z_km), kg_m3=tuple(placement.kg_m3))


__all__ = [
    "CONFIG_VERSION",
    "LoadedConfig",
    "build_aerosol",
    "build_scene",
    "export_config",
    "load_config",
]
