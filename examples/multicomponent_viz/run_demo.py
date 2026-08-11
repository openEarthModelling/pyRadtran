"""Comprehensive pyRadtran demo: LEGO blocks -> DISORT -> full viz + workflow.

Self-contained (pyRadtran APIs only, no aerosol3d dependency). Scene config +
composite builders live in ``canonical.py`` (shared with the regression test).

Scientific self-validation wired in:
  - B1: column tau@550 logged + checked against sum of block targets.
  - B2: column energy conservation asserted (F_inc = eup_TOA + (1-a)(edir+edn)_surf
    + F_abs_atm); script exits non-zero if the physics is violated.
  - T3 DRF: no-aerosol baseline run -> direct radiative forcing spectrum.
  - B3: ``--dump-baseline PATH`` writes the scalar regression fixture.

Requires libRadtran. Set PYRADTRAN_DATA_PATH to its data/ dir (or rely on the
bundled resolver).

Usage:
    python run_demo.py
    python run_demo.py --dump-baseline tests/fixtures/multicomponent_baseline.json
"""

import argparse
import json
import logging
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from canonical import (
    ALBEDO,
    ALTITUDE_GRID_KM,
    BLOCKS,
    N_LEGENDRE,
    OUTPUT_DIR,
    WAVELENGTHS_UM,
    _WL_RI,
    build_composite,
    build_scene,
    build_scene_no_aerosol,
)
from pyradtran import Runner
from pyradtran.core.output_parser import HEATING_RATE_COLUMN
from pyradtran.core.postprocess import (
    add_budget_vars,
    assert_energy_conservation,
    evaluate_blocks_on_grid,
    evaluate_composite_on_grid,
)
from pyradtran.models.aerosol_composite import RefractiveIndex, SizeDistribution
from pyradtran.viz import (
    plot_block_profiles,
    plot_block_spectral_optics,
    plot_budget,
    plot_component_attribution,
    plot_composite_optics,
    plot_drf_spectral,
    plot_flux_profile,
    plot_heating_rate,
    plot_legendre_decay,
    plot_phase_functions,
    plot_rt_overview,
    plot_size_distributions,
    plot_spectral,
    plot_spectral_attribution,
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dump-baseline",
        type=str,
        default=None,
        help="Write B3 scalar baseline JSON to PATH and continue.",
    )
    args, _ = parser.parse_known_args()

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
    # B1: column tau@550 must stack to the sum of block targets.
    col_tau_550 = float(np.sum(grid_ds["tau"].sel(wavelength=0.55).values))
    target_sum = sum(b["tau_550"] for b in BLOCKS)
    logger.info("  column tau@550 = %.4f (target sum %.2f)", col_tau_550, target_sum)
    assert np.isclose(col_tau_550, target_sum, rtol=0.01), (
        f"B1 FAIL: column tau@550={col_tau_550:.4f}, expected {target_sum:.4f}"
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

    # T2: input diagnostics — size distributions, per-block spectral optics,
    # phase functions, Legendre decay (the Mie->DISORT scattering physics).
    blocks_ri_sd = {
        b["name"]: (
            RefractiveIndex(wavelength_um=_WL_RI, n_real=b["n_real"], k_imag=b["k_imag"]),
            SizeDistribution(
                kind="lognormal", params={"r_g_um": b["r_g_um"], "sigma_g": b["sigma_g"]}
            ),
        )
        for b in BLOCKS
    }
    fig, _ = plot_size_distributions({n: sd for n, (_, sd) in blocks_ri_sd.items()})
    _save(fig, "size_distributions.png")
    for q in ("tau", "ssa", "g"):
        fig, _ = plot_block_spectral_optics(block_dict, quantity=q)
        _save(fig, f"block_spectral_{q}.png")
    fig, _ = plot_phase_functions(blocks_ri_sd, wavelength_um=0.55)
    _save(fig, "phase_functions_550.png")
    fig, _ = plot_legendre_decay(
        composite, wavelength_um=0.55, n_legendre=N_LEGENDRE, altitude_grid_km=ALTITUDE_GRID_KM
    )
    _save(fig, "legendre_decay_550.png")

    # --- RT (full composite) ---
    logger.info("=== RT run (full composite, DISORT) ===")
    rt = Runner.execute(build_scene(composite), data_path=data_path)
    logger.info("  RT data_vars=%s dims=%s", list(rt.data_vars), dict(rt.sizes))

    # B2: column energy conservation (hard assertion).
    budget = assert_energy_conservation(rt, albedo=ALBEDO, tol=0.05)
    i550 = int(np.argmin(np.abs(budget.wavelength - 0.55)))
    logger.info(
        "  energy@550: F_inc=%.1f  up_TOA=%.1f  abs_surf=%.1f  abs_atm=%.1f  (W/m²)",
        budget.f_incident[i550],
        budget.f_up_toa[i550],
        budget.f_abs_surface[i550],
        budget.f_abs_atm[i550],
    )

    fig, _ = plot_spectral(rt)
    _save(fig, "rt_spectral.png")
    fig, _ = plot_flux_profile(rt, variable="edir", wavelength_nm=550.0)
    _save(fig, "rt_flux_profile_edir.png")
    # T1: exercise edn + eup flux profiles too (was edir-only).
    for var in ("edn", "eup"):
        fig, _ = plot_flux_profile(rt, variable=var, wavelength_nm=550.0)
        _save(fig, f"rt_flux_profile_{var}.png")
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

    # --- T3: direct radiative forcing (with aerosol - without aerosol) ---
    logger.info("=== DRF baseline (no aerosol) ===")
    rt_noaer = Runner.execute(build_scene_no_aerosol(), data_path=data_path)
    wl_nm = rt["wavelength"].values * 1000.0
    surf_idx = int(np.argmin(rt["zout"].values))
    toa_idx = int(np.argmax(rt["zout"].values))
    net_toa_aer = rt["edir"].isel(zout=toa_idx).values - rt["eup"].isel(zout=toa_idx).values
    net_toa_noaer = (
        rt_noaer["edir"].isel(zout=toa_idx).values - rt_noaer["eup"].isel(zout=toa_idx).values
    )
    net_sfc_aer = (rt["edir"] + rt["edn"] - rt["eup"]).isel(zout=surf_idx).values
    net_sfc_noaer = (
        (rt_noaer["edir"] + rt_noaer["edn"] - rt_noaer["eup"]).isel(zout=surf_idx).values
    )
    drf_toa = net_toa_aer - net_toa_noaer  # <0 = cooling (IPCC convention)
    drf_surf = net_sfc_aer - net_sfc_noaer
    fig, _ = plot_drf_spectral(wl_nm, drf_toa, drf_surf)
    _save(fig, "drf_spectral.png")
    logger.info(
        "  DRF@550: TOA=%.2f  surf=%.2f  atm=%.2f W/m²",
        float(np.interp(550.0, wl_nm, drf_toa)),
        float(np.interp(550.0, wl_nm, drf_surf)),
        float(np.interp(550.0, wl_nm, drf_toa - drf_surf)),
    )

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
    # T1: attribution at additional variable/level. plot_component_attribution
    # treats non-"surface" level as an integer zout index, so resolve "toa".
    toa_zout_idx = int(np.argmax(result.full["zout"].values))
    fig, _ = plot_component_attribution(result, variable="eup", level=toa_zout_idx)
    _save(fig, "attribution_eup_toa.png")
    fig, _ = plot_component_attribution(result, variable="edn", level="surface")
    _save(fig, "attribution_edn_surf.png")
    # T3: spectral attribution (per-block contribution across the band).
    fig, _ = plot_spectral_attribution(result, variable="edir", level="surface")
    _save(fig, "attribution_edir_spectral.png")

    # --- B3: optional scalar baseline dump ---
    if args.dump_baseline:
        i550_rt = int(np.argmin(np.abs(rt["wavelength"].values - 0.55)))
        scalars = {
            "_meta": {
                "scene": "multicomponent_viz",
                "note": "regenerate via run_demo.py --dump-baseline",
            },
            "edir_surf_550nm": float(rt["edir"].isel(wavelength=i550_rt, zout=surf_idx).values),
            "edn_surf_550nm": float(rt["edn"].isel(wavelength=i550_rt, zout=surf_idx).values),
            "eup_surf_550nm": float(rt["eup"].isel(wavelength=i550_rt, zout=surf_idx).values),
            "eup_toa_550nm": float(rt["eup"].isel(wavelength=i550_rt, zout=toa_idx).values),
            "column_tau_550nm": col_tau_550,
            "F_abs_atm_550nm": float(budget.f_abs_atm[i550]),
        }
        Path(args.dump_baseline).write_text(json.dumps(scalars, indent=2))
        logger.info("wrote baseline -> %s", args.dump_baseline)

    logger.info("=== Comprehensive demo complete. Figures in %s ===", OUTPUT_DIR)


if __name__ == "__main__":
    main()
