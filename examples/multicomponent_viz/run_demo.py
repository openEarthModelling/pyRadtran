"""Comprehensive pyRadtran demo: LEGO blocks -> DISORT -> full viz + workflow.

Self-contained (pyRadtran APIs only, no aerosol3d dependency). Three aerosol
blocks with contrasting optics are built from refractive indices + lognormal
size distributions via ``MieSpecies``, externally mixed into one
``CompositeAerosol``, and run through DISORT. The script then exercises the
FULL ``pyradtran.viz`` plot surface and the component-attribution workflow.

  Composite diagnostics (analytic mixing, no RT):
    - evaluate_composite_on_grid -> plot_composite_optics (tau / ssa / g)
    - evaluate_blocks_on_grid    -> plot_block_profiles (per-block tau(z), rho(z))

  RT result plots (full composite, real DISORT):
    - plot_spectral, plot_flux_profile, plot_budget (via add_budget_vars),
      plot_heating_rate (if libRadtran emits it), plot_rt_overview

  Workflow (component attribution, leave-one-out):
    - compute_component_attribution -> plot_component_attribution

Requires libRadtran. Set PYRADTRAN_DATA_PATH to its data/ dir (or rely on the
bundled resolver).

Usage:
    python run_demo.py
"""

import logging
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyradtran import Runner, Scene
from pyradtran.core.output_parser import HEATING_RATE_COLUMN
from pyradtran.core.postprocess import (
    add_budget_vars,
    evaluate_blocks_on_grid,
    evaluate_composite_on_grid,
)
from pyradtran.models.aerosol_composite import (
    CompositeAerosol,
    IntegrationConfig,
    MieSpecies,
    RefractiveIndex,
    SizeDistribution,
)
from pyradtran.models.blocks import PlacedBlock, od_to_mass_profile
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

EXAMPLE_DIR = Path(__file__).parent
OUTPUT_DIR = EXAMPLE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Spectral / grid (composite optics on a coarse grid; DISORT resamples) ---
WAVELENGTHS_UM = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
N_LEGENDRE = 32
ALTITUDE_GRID_KM = [10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.0]  # descending (TOA -> surface)
REF_NM = 550.0
ZOUT_LEVELS = [0, 1, 2, 4, 6, 8, 10, "toa"]

# --- RT scene ---
SZA_DEG = 30.0
ALBEDO = 0.1
N_STREAMS = 16
SCENE_KW = {
    "atmosphere": {"profile": "us", "altitude": 0.0},
    "source": {"sza": SZA_DEG},
    "wavelength": {"min_nm": 401.0, "max_nm": 699.0},
    "solver": {
        "method": "disort",
        "streams": N_STREAMS,
        "disort_intcor": "moments",
        "pseudospherical": True,
    },
    "surface": {"albedo": ALBEDO},
    "output": {
        "quantities": ["lambda", "edir", "edn", "eup"],
        "format": "ascii",
        "zout": ZOUT_LEVELS,
        "heating_rate": "local",
    },
}

# --- Three LEGO blocks: refractive index, lognormal size dist, placement ---
_WL_RI = [0.30, 0.40, 0.55, 0.70, 1.00]  # µm; constant n/k across the band
BLOCKS = [
    {
        "name": "black_carbon",
        "n_real": [1.95] * 5,
        "k_imag": [0.79] * 5,
        "density": 1800.0,
        "r_g_um": 0.10,
        "sigma_g": 2.0,
        "tau_550": 0.15,
        "scale_height_km": 1.5,
    },
    {
        "name": "sulfate",
        "n_real": [1.53] * 5,
        "k_imag": [0.0] * 5,
        "density": 1770.0,
        "r_g_um": 0.15,
        "sigma_g": 1.7,
        "tau_550": 0.15,
        "scale_height_km": 2.0,
    },
    {
        "name": "mineral_dust",
        "n_real": [1.53] * 5,
        "k_imag": [0.008] * 5,
        "density": 2600.0,
        "r_g_um": 0.50,
        "sigma_g": 2.2,
        "tau_550": 0.20,
        "scale_height_km": 3.0,
    },
]

_FLUX_VARS = {"edir", "edn", "eup", "udir", "udn", "uup"}


def _save(fig, name):
    path = OUTPUT_DIR / name
    save(fig, str(path), formats=("png",))
    plt.close(fig)
    logger.info("  saved %s", path.name)


def build_composite() -> CompositeAerosol:
    """Build MieSpecies for each block, invert its target OD@550 into a mass
    profile via the API (od_to_mass_profile), and assemble the composite."""
    pieces = []
    for b in BLOCKS:
        ri = RefractiveIndex(wavelength_um=_WL_RI, n_real=b["n_real"], k_imag=b["k_imag"])
        sd = SizeDistribution(kind="lognormal", params={"r_g_um": b["r_g_um"], "sigma_g": b["sigma_g"]})
        species = MieSpecies(
            refractive_index=ri,
            size_distribution=sd,
            particle_density_kg_m3=b["density"],
            integration_config=IntegrationConfig(),
            name=b["name"],
        )
        profile = od_to_mass_profile(
            species,
            tau_ref=b["tau_550"],
            ref_nm=REF_NM,
            altitude_km=ALTITUDE_GRID_KM,
            scale_height_km=b["scale_height_km"],
        )
        pieces.append(PlacedBlock(block=species, profile=profile))
        logger.info(
            "  block %-14s m=%.2f%+.3fi r_g=%.2fµm tau@550=%.2f H=%.1fkm",
            b["name"], b["n_real"][0], b["k_imag"][0], b["r_g_um"], b["tau_550"], b["scale_height_km"],
        )
    return CompositeAerosol(
        pieces=pieces,
        wavelength_grid_um=list(WAVELENGTHS_UM),
        altitude_grid_km=list(ALTITUDE_GRID_KM),
        n_legendre=N_LEGENDRE,
        output_dir=OUTPUT_DIR,
    )


def build_scene(aerosol: CompositeAerosol) -> Scene:
    c = SCENE_KW
    return (
        Scene()
        .set_atmosphere(profile=c["atmosphere"]["profile"], altitude=c["atmosphere"]["altitude"])
        .set_source_solar(sza=c["source"]["sza"])
        .set_wavelength(c["wavelength"]["min_nm"], c["wavelength"]["max_nm"])
        .set_solver(
            method=c["solver"]["method"],
            streams=c["solver"]["streams"],
            disort_intcor=c["solver"].get("disort_intcor"),
            pseudospherical=c["solver"].get("pseudospherical", False),
        )
        .set_surface(albedo=c["surface"]["albedo"])
        .set_output(**c["output"])
        .set_aerosol(aerosol)
    )


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
