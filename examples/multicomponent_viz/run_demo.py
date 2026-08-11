"""Comprehensive pyRadtran demo: LEGO blocks -> DISORT -> full viz + workflow.

Self-contained (pyRadtran APIs only, no aerosol3d dependency). Scene config +
composite builders live in ``canonical.py`` (shared with the regression test).

Requires libRadtran. Set PYRADTRAN_DATA_PATH to its data/ dir (or rely on the
bundled resolver).

Usage:
    python run_demo.py
"""

import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from canonical import (
    ALTITUDE_GRID_KM,
    N_LEGENDRE,
    OUTPUT_DIR,
    WAVELENGTHS_UM,
    build_composite,
    build_scene,
)
from pyradtran import Runner
from pyradtran.core.output_parser import HEATING_RATE_COLUMN
from pyradtran.core.postprocess import (
    add_budget_vars,
    evaluate_blocks_on_grid,
    evaluate_composite_on_grid,
)
from pyradtran.viz import (
    plot_block_profiles,
    plot_budget,
    plot_component_attribution,
    plot_composite_optics,
    plot_flux_profile,
    plot_heating_rate,
    plot_rt_overview,
    plot_spectral,
    save,
    set_theme,
)
from pyradtran.workflow import compute_component_attribution

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR.mkdir(exist_ok=True)

_FLUX_VARS = {"edir", "edn", "eup", "udir", "udn", "uup"}


def _save(fig, name):
    path = OUTPUT_DIR / name
    save(fig, str(path), formats=("png",))
    plt.close(fig)
    logger.info("  saved %s", path.name)


def _resolve_heating_var(ds):
    if HEATING_RATE_COLUMN in ds.data_vars:
        return HEATING_RATE_COLUMN
    extras = [v for v in ds.data_vars if v not in _FLUX_VARS and v != "wavelength"]
    if len(extras) == 1:
        return extras[0]
    return None


def main():
    set_theme("publication")
    data_path = os.environ.get("PYRADTRAN_DATA_PATH")  # None -> bundled resolver
    if data_path is None:
        logger.info("PYRADTRAN_DATA_PATH unset; using bundled DataResolver")
    else:
        logger.info("using libRadtran data: %s", data_path)

    logger.info("=== Building 3-block composite (pyRadtran MieSpecies) ===")
    composite = build_composite()

    # --- Composite diagnostics (analytic) ---
    logger.info("=== Composite diagnostics (analytic) ===")
    grid_ds = evaluate_composite_on_grid(
        composite, WAVELENGTHS_UM, ALTITUDE_GRID_KM, n_legendre=N_LEGENDRE
    )
    for q in ("tau", "ssa", "g"):
        fig, _ = plot_composite_optics(grid_ds, quantity=q)
        _save(fig, f"composite_{q}.png")
    block_dict = evaluate_blocks_on_grid(
        composite, WAVELENGTHS_UM, ALTITUDE_GRID_KM, n_legendre=N_LEGENDRE
    )
    fig, _ = plot_block_profiles(block_dict, quantity="tau")
    _save(fig, "block_tau_profiles.png")
    fig, _ = plot_block_profiles(block_dict, quantity="rho")
    _save(fig, "block_rho_profiles.png")

    # --- RT (full composite) ---
    logger.info("=== RT run (full composite, DISORT) ===")
    rt = Runner.execute(build_scene(composite), data_path=data_path)
    logger.info("  RT data_vars=%s dims=%s", list(rt.data_vars), dict(rt.sizes))

    fig, _ = plot_spectral(rt)
    _save(fig, "rt_spectral.png")
    fig, _ = plot_flux_profile(rt, variable="edir", wavelength_nm=550.0)
    _save(fig, "rt_flux_profile_edir.png")
    rt_budget = add_budget_vars(rt)
    fig, _ = plot_budget(rt_budget)
    _save(fig, "rt_budget.png")

    hvar = _resolve_heating_var(rt)
    if hvar is not None:
        if hvar != HEATING_RATE_COLUMN:
            rt = rt.rename({hvar: HEATING_RATE_COLUMN})
        fig, _ = plot_heating_rate(rt, wavelength_nm=550.0)
        _save(fig, "rt_heating_rate.png")
    else:
        logger.warning("  no heating-rate column in RT output; skipping plot_heating_rate")

    rt.to_netcdf(str(OUTPUT_DIR / "rt_full.nc"))
    fig, _ = plot_rt_overview(rt, wavelength_nm=550.0)
    _save(fig, "rt_overview.png")

    # --- Component attribution (leave-one-out) ---
    logger.info("=== Component attribution (N+1 DISORT runs) ===")

    def execute_many(scenes):
        # Sequential: Runner.execute_many (ProcessPoolExecutor) swallows exceptions
        # and fails on CompositeAerosol scenes (pickling). Sequential surfaces errors.
        return [Runner.execute(s, data_path=data_path) for s in scenes]

    result = compute_component_attribution(build_scene, composite, execute_many)
    logger.info("  contributions: %s", list(result.contributions))
    fig, _ = plot_component_attribution(result, variable="edir", level="surface")
    _save(fig, "attribution_edir.png")

    logger.info("=== Comprehensive demo complete. Figures in %s ===", OUTPUT_DIR)


if __name__ == "__main__":
    main()
