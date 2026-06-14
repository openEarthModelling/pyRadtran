"""Gated end-to-end Legendre convention validation test.

This test empirically determines which Legendre coefficient convention
libRadtran's ``aerosol_file explicit`` wants: the PMOM g_l form
(g_0=1, g_1=g, g_2=g^2, ...) or the integrated k_l form
(k_l = (2l+1)*g_l, i.e. k_1=3g).

Mechanism:
  - ``BulkSpecies`` emits ``legendre_moments_beta`` raw to the .LAYER writer
    (no scaling). By putting the g_l array or the k_l array into that field,
    we force two genuinely different .LAYER files through uvspec.
  - The tie is broken by comparing each form's diffuse downwelling (edn)
    against an independent reference: a scene that uses libRadtran's
    INTERNAL Henyey-Greenstein phase function via ``aerosol_modify gg set 0.5``
    (which does not depend on the file convention at all) with matching
    optical depth.

Expected: the g_l form reproduces the internal-HG reference; the k_l form
does not. This confirms libRadtran wants the g_l (PMOM k_m = g^m) form —
consistent with the authoritative example files AERO_050.LAYER and
AERO_MOMENTS.DAT in the libRadtran distribution.

Skipped cleanly when uvspec or libRadtran data is unavailable (CI without
libRadtran does not run this).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyradtran.core.runner import Runner
from pyradtran.models.aerosol_composite import (
    BulkSpecies,
    CompositeAerosol,
    LoadedSpecies,
)
from pyradtran.scene import Scene

# --- Gating: skip cleanly when uvspec/data unavailable ---

_UVSPEC_EXE = shutil.which("uvspec")


def _resolve_data_path() -> str | None:
    """Resolve libRadtran data dir from env or install fallback."""
    for env_var in ("LIBRADTRAN_DATA_FILES", "LIBRADTRANDIR", "PYRADTRAN_DATA_PATH"):
        val = os.environ.get(env_var)
        if val:
            if env_var == "LIBRADTRANDIR":
                val = os.path.join(val, "data")
            if os.path.isdir(val):
                return val
    # Install fallback: sibling Radiation/libRadtran-2.0.6/data
    candidate = Path(__file__).resolve().parents[2] / "libRadtran-2.0.6" / "data"
    if candidate.is_dir():
        return str(candidate)
    return None


DATA_PATH = _resolve_data_path()
_UVSPEC_AVAILABLE = _UVSPEC_EXE is not None and DATA_PATH is not None

pytestmark = pytest.mark.skipif(
    not _UVSPEC_AVAILABLE,
    reason="uvspec binary and/or libRadtran data directory required for this test",
)

# Henyey-Greenstein asymmetry parameter and aerosol load used in all runs.
_G = 0.5
_MASS_PROFILE = [0.0001, 0.0001]  # kg/m^3 per layer; total tau ~ 0.032
# Fractional tolerance for matching the internal-HG reference. The explicit
# file uses a 2-layer profile while the reference uses aerosol_default's
# vertical distribution, so a loose tolerance is expected; the wrong
# convention (k_l, interpreted as g=1.5) produces >20% error in edn.
_EDN_REL_TOL = 0.10


def _hg_bulk(moment_form: str, g: float = _G, n_wl: int = 2, n_l: int = 32):
    """Construct a minimal BulkAerosolOpticsData-like object with HG moments.

    Two genuinely different forms are produced:
      - ``g_l``: legendre_moments_beta = g^l  (PMOM form, g_0=1, g_1=g)
      - ``k_l``: legendre_moments_beta = (2l+1)*g^l  (integrated form, k_1=3g)

    BulkSpecies reads ``legendre_moments_beta`` raw (no conversion), so the
    two forms write different .LAYER files.

    Two wavelengths are used because libRadtran's explicit-aerosol spline
    interpolation requires >=2 data points (a single point triggers
    ``calc_splined_value`` Error -7).
    """
    from Aerosol3D.bulk.datastructs import SizeDistribution

    l_vals = np.arange(n_l)
    sd = SizeDistribution.lognormal(rg_nm=100.0, sigma_ln=0.4)
    # g_l form: PMOM coefficient k_m = g^m. k_l form: feed (2m+1)*g^m raw
    # (BulkSpecies reads legendre_moments_beta without scaling).
    lmb = g**l_vals if moment_form == "g_l" else (2 * l_vals + 1) * g**l_vals
    # Aerosol grid spans wider than the RT wavelength range so libRadtran's
    # spline interpolation stays in-bounds (its internal grid is offset from
    # the requested wavelength_min/max boundaries).
    return SimpleNamespace(
        wavelength_nm=np.array([500.0, 600.0]),
        C_ext=np.full(n_wl, 1.0),
        C_sca=np.full(n_wl, 1.0),
        SSA=np.full(n_wl, 1.0 - 1e-6),
        g=np.full(n_wl, g),
        beta=np.tile((2 * l_vals + 1) * g**l_vals, (n_wl, 1)),
        legendre_moments_beta=np.tile(lmb, (n_wl, 1)),
        n_legendre=n_l,
        size_distribution=sd,
        effective_density_kg_m3=1800.0,
    )


def _explicit_scene(moment_form: str, output_dir: Path) -> Scene:
    """Build a Scene whose explicit-aerosol .LAYER files use ``moment_form``."""
    bulk = _hg_bulk(moment_form=moment_form)
    species = BulkSpecies(bulk=bulk)
    loaded = LoadedSpecies(
        species=species,
        mass_profile_kg_m3=list(_MASS_PROFILE),
        altitude_km=[5.0, 2.5, 0.0],  # 2 layers, descending
    )
    comp = CompositeAerosol(
        sources=[loaded],
        wavelength_grid_um=[0.500, 0.600],
        altitude_grid_km=[5.0, 2.5, 0.0],
        n_legendre=32,
        output_dir=output_dir,
    )
    return (
        Scene()
        .set_atmosphere(profile="us", altitude=0.0)
        .set_source_solar(sza=30.0)
        .set_wavelength(545.0, 555.0)
        .set_solver(method="disort", streams=16, disort_intcor="moments")
        .set_surface(albedo=0.0)
        .set_output(quantities=["lambda", "edir", "edn"], format="ascii", quiet=True)
        .set_aerosol(comp)
    )


def _reference_tau() -> float:
    """Compute the total aerosol optical depth the explicit file produces.

    This is the tau value passed to the internal-HG reference scene via
    ``aerosol_modify tau550 set <tau>``.
    """
    bulk = _hg_bulk(moment_form="g_l")
    species = BulkSpecies(bulk=bulk)
    loaded = LoadedSpecies(
        species=species,
        mass_profile_kg_m3=list(_MASS_PROFILE),
        altitude_km=[5.0, 2.5, 0.0],
    )
    layer_optics = loaded.evaluate(np.array([0.55]), np.array([5.0, 2.5, 0.0]), n_legendre=32)
    return float(np.sum(layer_optics.tau))


def _reference_scene(tau: float) -> Scene:
    """Build a Scene using libRadtran's INTERNAL HG phase function.

    ``aerosol_modify gg set 0.5`` makes libRadtran compute the HG phase
    function internally from g=0.5 — completely independent of the
    explicit-file convention. Combined with ``tau550 set <tau>`` this
    produces the ground-truth edn for HG g=0.5 at the same optical depth.
    """
    return (
        Scene()
        .set_atmosphere(profile="us", altitude=0.0)
        .set_source_solar(sza=30.0)
        .set_wavelength(545.0, 555.0)
        .set_solver(method="disort", streams=16, disort_intcor="moments")
        .set_surface(albedo=0.0)
        .set_output(quantities=["lambda", "edir", "edn"], format="ascii", quiet=True)
        .add_raw_keyword("aerosol_default")
        .add_raw_keyword("aerosol_modify", f"tau550 set {tau:.6f}")
        .add_raw_keyword("aerosol_modify", f"gg set {_G}")
    )


def _run_edn(scene: Scene) -> np.ndarray:
    """Run uvspec and return the edn array (or raise on failure)."""
    result = Runner.execute(scene, data_path=DATA_PATH)
    return np.asarray(result["edn"].values, dtype=float)


def _matches_reference(edn: np.ndarray, edn_ref: np.ndarray) -> bool:
    """True iff edn is within ``_EDN_REL_TOL`` of edn_ref on average."""
    rel_err = float(np.mean(np.abs(edn - edn_ref) / np.abs(edn_ref)))
    return rel_err < _EDN_REL_TOL


def test_g_l_form_matches_internal_hg_reference(tmp_path):
    """Validate the g_l convention by matching libRadtran's internal HG.

    Asserts:
      - The g_l form reproduces the internal-HG reference (within tolerance).
      - The k_l form does NOT reproduce the reference (it is interpreted as
        g=1.5, producing a large error in diffuse downwelling).

    If the g_l form fails to match, the convention hypothesis is REFUTED
    and BulkSpecies should be flipped to emit k_l. If both match, the test
    is inconclusive and the tolerance must be tightened.
    """
    tau = _reference_tau()
    print(f"\nTotal aerosol tau: {tau:.6f}")

    edn_ref = _run_edn(_reference_scene(tau))
    print(f"[reference] internal HG g={_G}, edn mean = {np.mean(edn_ref):.4f}")

    results: dict[str, dict] = {}
    for form in ("g_l", "k_l"):
        outdir = tmp_path / form
        outdir.mkdir(parents=True, exist_ok=True)
        scene = _explicit_scene(form, outdir)
        try:
            edn = _run_edn(scene)
        except Exception as exc:  # noqa: BLE001
            results[form] = {
                "ok": False,
                "matched": False,
                "diag": f"uvspec raised: {type(exc).__name__}: {exc}",
            }
            print(f"[{form} form] FAILED: {results[form]['diag']}")
            continue
        matched = _matches_reference(edn, edn_ref)
        rel_err = float(np.mean(np.abs(edn - edn_ref) / np.abs(edn_ref)))
        results[form] = {
            "ok": True,
            "matched": matched,
            "edn_mean": float(np.mean(edn)),
            "rel_err": rel_err,
        }
        print(
            f"[{form} form] edn mean = {np.mean(edn):.4f}, "
            f"rel_err vs ref = {rel_err:.4f}, matches = {matched}"
        )

    g_l = results["g_l"]
    k_l = results["k_l"]

    # Hypothesis: g_l is the correct convention (libRadtran wants PMOM g_l).
    assert g_l["ok"] and g_l["matched"], (
        "g_l form did NOT match the internal-HG reference — convention "
        f"hypothesis REFUTED (g_l: {g_l}; k_l: {k_l})"
    )
    assert not (k_l["ok"] and k_l["matched"]), (
        "both forms match the reference — convention test INCONCLUSIVE, "
        f"tolerance must be tightened (g_l: {g_l}; k_l: {k_l})"
    )
