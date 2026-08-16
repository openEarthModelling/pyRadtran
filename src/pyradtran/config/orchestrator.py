"""Orchestrator: execute a config's main run plus its analysis intents.

:func:`run_config` drives the full pipeline a config describes:

1. main uvspec run (fluxes) of the loaded scene;
2. optional analyses — energy conservation, heating (second uvspec
   invocation, merged), DRF vs a no-aerosol baseline, leave-one-out
   attribution (N+1 runs);
3. optional plots (names resolved against a fixed registry) and NetCDF
   persistence.

The multi-run analyses mirror the proven flow of
``examples/multicomponent_viz/run_demo.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xarray as xr

from pyradtran.config.loader import LoadedConfig, build_scene, load_config
from pyradtran.config.schema import AnalysisSection, SceneSection
from pyradtran.core.postprocess import EnergyBudget, assert_energy_conservation
from pyradtran.core.runner import Runner
from pyradtran.workflow import AttributionResult, compute_component_attribution

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DrfResult:
    """Spectral direct radiative forcing (aerosol minus no-aerosol), W/m².

    Sign convention: negative TOA/surface = cooling (IPCC).
    """

    wavelength_nm: np.ndarray
    toa: np.ndarray
    surface: np.ndarray

    @property
    def atmosphere(self) -> np.ndarray:
        """TOA minus surface forcing (energy deposited in the column)."""
        return self.toa - self.surface


@dataclass
class RunResult:
    """Everything a config run produced."""

    rt: xr.Dataset  # main flux run (+ merged heating_rate if requested)
    budget: EnergyBudget | None = None
    drf: DrfResult | None = None
    attribution: AttributionResult | None = None
    figures: list[Path] = field(default_factory=list)
    netcdf_path: Path | None = None


def run_config(
    source: str | Path | dict | LoadedConfig,
    *,
    uvspec_exe: str | None = None,
    data_path: str | None = None,
    plot_dir: str | Path | None = None,
) -> RunResult:
    """Load (if needed) and run a config end-to-end.

    Args:
        source: YAML path, raw dict, or an already-loaded config.
        uvspec_exe / data_path: forwarded to :meth:`Runner.execute`.
        plot_dir: directory for requested plots (defaults to the config's
            aerosol ``output_dir``, else the current directory).

    Returns:
        :class:`RunResult`. Raises ``AssertionError`` if an energy-budget
        intent is violated, or ``ValueError`` for unsatisfiable intents.
    """
    loaded = source if isinstance(source, LoadedConfig) else load_config(source)
    section = loaded.config.scene
    analysis = loaded.config.analysis or AnalysisSection()
    run = dict(uvspec_exe=uvspec_exe, data_path=data_path)

    rt = Runner.execute(loaded.scene, **run)

    budget = None
    if analysis.energy_conservation is not None:
        spec = analysis.energy_conservation
        albedo = spec.albedo if spec.albedo is not None else _scene_albedo(section)
        budget = assert_energy_conservation(rt, albedo=albedo, tol=spec.tol)

    if analysis.heating:
        # libRadtran's heating_rate mode replaces flux output — second run.
        heat_scene = build_scene(_heating_section(section), aerosol=loaded.aerosol)
        rt_heat = Runner.execute(heat_scene, **run)
        rt = rt.assign(heating_rate=rt_heat["heating_rate"])

    drf = None
    if analysis.drf:
        _require_aerosol(loaded, "analysis.drf")
        rt_noaer = Runner.execute(build_scene(section), **run)
        drf = _compute_drf(rt, rt_noaer)

    attribution = None
    if analysis.attribution:
        _require_aerosol(loaded, "analysis.attribution")

        def execute_many(scenes):
            # Sequential: Runner.execute_many swallows exceptions and cannot
            # pickle CompositeAerosol scenes (fixed separately in T7).
            return [Runner.execute(s, **run) for s in scenes]

        attribution = compute_component_attribution(
            lambda comp: build_scene(section, aerosol=comp), loaded.aerosol, execute_many
        )

    figures = []
    if analysis.plots:
        target = Path(plot_dir) if plot_dir is not None else _default_plot_dir(loaded)
        target.mkdir(parents=True, exist_ok=True)
        figures = _make_plots(
            analysis.plots,
            RunResult(rt=rt, budget=budget, drf=drf, attribution=attribution),
            target,
        )

    netcdf_path = None
    if analysis.save_netcdf:
        netcdf_path = Path(analysis.save_netcdf)
        rt.to_netcdf(netcdf_path)

    return RunResult(
        rt=rt,
        budget=budget,
        drf=drf,
        attribution=attribution,
        figures=figures,
        netcdf_path=netcdf_path,
    )


# --- helpers ---


def _scene_albedo(section: SceneSection) -> float:
    surface = section.surface or {}
    if "albedo" not in surface:
        raise ValueError(
            "analysis.energy_conservation: no 'albedo' given and the scene "
            "surface defines none — set analysis.energy_conservation.albedo"
        )
    return float(surface["albedo"])


def _heating_section(section: SceneSection) -> SceneSection:
    """Scene variant whose output drops quantities (switches uvspec to
    heating-rate mode) — mirrors canonical.build_scene_heating."""
    return section.model_copy(
        update={"output": {k: v for k, v in section.output.items() if k != "quantities"}}
    )


def _require_aerosol(loaded: LoadedConfig, intent: str) -> None:
    if loaded.aerosol is None:
        raise ValueError(f"{intent}: requires an 'aerosol' section in the config")


def _compute_drf(rt: xr.Dataset, rt_noaer: xr.Dataset) -> DrfResult:
    surf = int(np.argmin(rt["zout"].values))
    toa = int(np.argmax(rt["zout"].values))

    def net_toa(ds):
        return (ds["edir"] - ds["eup"]).isel(zout=toa).values

    def net_sfc(ds):
        return (ds["edir"] + ds["edn"] - ds["eup"]).isel(zout=surf).values

    return DrfResult(
        wavelength_nm=np.asarray(rt["wavelength"].values, dtype=float),
        toa=net_toa(rt) - net_toa(rt_noaer),
        surface=net_sfc(rt) - net_sfc(rt_noaer),
    )


def _default_plot_dir(loaded: LoadedConfig) -> Path:
    if loaded.config.aerosol is not None and loaded.config.aerosol.output_dir:
        return Path(loaded.config.aerosol.output_dir)
    return Path.cwd()


def _make_plots(names: list[str], result: RunResult, target: Path) -> list[Path]:
    """Render the requested plots by name; returns the written PNG paths."""
    import matplotlib.pyplot as plt

    from pyradtran import viz
    from pyradtran.core.postprocess import add_budget_vars

    registry = {
        "rt_spectral": lambda: viz.plot_spectral(result.rt),
        "rt_flux_profile_edir": lambda: viz.plot_flux_profile(
            result.rt, variable="edir", wavelength_nm=550.0
        ),
        "rt_flux_profile_edn": lambda: viz.plot_flux_profile(
            result.rt, variable="edn", wavelength_nm=550.0
        ),
        "rt_flux_profile_eup": lambda: viz.plot_flux_profile(
            result.rt, variable="eup", wavelength_nm=550.0
        ),
        "rt_budget": lambda: viz.plot_budget(add_budget_vars(result.rt)),
        "rt_heating_rate": lambda: viz.plot_heating_rate(result.rt, wavelength_nm=550.0),
        "rt_overview": lambda: viz.plot_rt_overview(result.rt, wavelength_nm=550.0),
        "drf_spectral": lambda: viz.plot_drf_spectral(
            result.drf.wavelength_nm, result.drf.toa, result.drf.surface
        ),
        "attribution_edir": lambda: viz.plot_component_attribution(
            result.attribution, variable="edir", level="surface"
        ),
        "attribution_spectral": lambda: viz.plot_spectral_attribution(
            result.attribution, variable="edir", level="surface"
        ),
    }
    figures = []
    for name in names:
        if name not in registry:
            raise ValueError(f"unknown plot {name!r}; valid: {sorted(registry)}")
        if result.drf is None and name.startswith("drf_"):
            raise ValueError(f"plot {name!r} requires analysis.drf: true")
        if result.attribution is None and name.startswith("attribution_"):
            raise ValueError(f"plot {name!r} requires analysis.attribution: true")
        if name == "rt_heating_rate" and "heating_rate" not in result.rt:
            raise ValueError(f"plot {name!r} requires analysis.heating: true")
        fig, _ = registry[name]()
        path = target / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        figures.append(path)
        logger.info("  plot -> %s", path)
    return figures


__all__ = ["DrfResult", "RunResult", "run_config"]
