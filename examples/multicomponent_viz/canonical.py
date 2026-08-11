"""Canonical scene configuration for the multicomponent_viz example.

Shared by run_demo.py and tests/test_multicomponent_regression.py so the
regression baseline and the demo cannot drift apart.
"""

import logging
from pathlib import Path

from pyradtran.models.aerosol_composite import (
    CompositeAerosol,
    IntegrationConfig,
    MieSpecies,
    RefractiveIndex,
    SizeDistribution,
)
from pyradtran.models.blocks import PlacedBlock, od_to_mass_profile
from pyradtran.scene import Scene

logger = logging.getLogger(__name__)

EXAMPLE_DIR = Path(__file__).parent
OUTPUT_DIR = EXAMPLE_DIR / "output"

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
    """Build the RT scene with the given aerosol composite attached."""
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


def build_scene_no_aerosol() -> Scene:
    """Same scene as build_scene() minus the aerosol — DRF baseline."""
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
    )


def build_scene_heating(aerosol: CompositeAerosol) -> Scene:
    """Scene variant for heating-rate output (libRadtran heating_rate mode).

    Identical to build_scene() but drops ``output_user`` quantities. With
    ``heating_rate`` set and no ``output_user``, uvspec emits heating rates
    (K/day, wide format) instead of fluxes — a second run is required because
    libRadtran cannot emit both in one invocation.
    """
    c = SCENE_KW
    heating_output = {k: v for k, v in c["output"].items() if k != "quantities"}
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
        .set_output(**heating_output)
        .set_aerosol(aerosol)
    )
