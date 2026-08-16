"""Randles et al. (2013) shortwave benchmark: case matrix + runner.

Reproduces the shortwave radiative-transfer intercomparison protocol of
Randles, C. A. et al. (2013), "Intercomparison of shortwave radiative
transfer schemes to improve the aerosol and surface albedo treatment in
climate models", Atmos. Chem. Phys. 13, 2347-2362 (Tables 2/3/5/A3/A4).

Protocol
--------
Case 1  Rayleigh-only (no aerosol).
Case 2a AOD550 = 0.2, Angstroem exponent 1, SSA = 1.0, g = 0.7.
Case 2b As 2a but SSA = 0.8.
Surface albedo 0.2 Lambert; SZA {30, 75} deg; atmospheres AFGL subarctic
winter ("saw") and tropics ("trop"); no clouds. Aerosol confined to the
lowest 2 km, linearly decreasing to zero at 2 km. Bands: broadband
0.2-4.0 um ("bb") and UV-VIS 0.2-0.7 um ("uvvis").

Normalization
-------------
Each model divides its own band TOA incident flux and multiplies by the
LBL median constants (W/m2): BB {SZA30: 1189.28, SZA75: 355.43},
UVVIS {SZA30: 563.38, SZA75: 168.37}. This removes the solar-constant
choice. Formally ``F_norm = F_raw / edir_toa_raw(same run) * C[band][sza]``.

Replication assumptions
-----------------------
The paper leaves details to each participating model; we fix them as:

1. Built-in AFGL profiles are used (the paper's participants used their
   individual-model profiles; the comparison tolerance absorbs this).
2. "Aerosol linear in the lowest 2 km" = a linear taper from the surface
   to zero at 2 km, ``w(z) = max(0, 1 - z / 2 km)``.
3. g = 0.7 Henyey-Greenstein phase function represented through PMOM
   moments ``beta_l = g**l`` (the DISORT/libRadtran coefficient form).
4. Angstroem exponent 1 means ``tau(lambda) = 0.2 * (lambda / 0.55 um)**-1``.

Comparison thresholds live in the reference file ``_meta`` block
(fluxes: PASS within +/-8%, WARN within +/-12%, else FAIL; RF: PASS if
|diff| <= 15% OR <= 1.5 W/m2).
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from pyradtran.core.runner import Runner
from pyradtran.models.aerosol_composite import CompositeAerosol, LayerOptics
from pyradtran.scene import Scene

#: Benchmark cases: Rayleigh-only, and the two aerosol loadings.
CASES: tuple[str, ...] = ("case1", "case2a", "case2b")

#: Atmosphere keys -> pyRadtran AFGL profile aliases (sw/trop).
ATMOSPHERES: dict[str, str] = {"saw": "sw", "trop": "tp"}

#: Solar zenith angles in degrees.
SZAS: tuple[int, ...] = (30, 75)

#: Spectral bands: key -> (min_nm, max_nm).
BANDS: dict[str, tuple[float, float]] = {
    "bb": (200.0, 4000.0),
    "uvvis": (200.0, 700.0),
}

#: LBL median band TOA incident flux constants (W/m2) per band and SZA.
NORMALIZATION_CONSTANTS: dict[str, dict[int, float]] = {
    "bb": {30: 1189.28, 75: 355.43},
    "uvvis": {30: 563.38, 75: 168.37},
}

#: Lambertian surface albedo (protocol Table 2).
SURFACE_ALBEDO = 0.2

#: Aerosol optical depth at the reference wavelength.
AOD_550 = 0.2
#: Reference wavelength for the Angstroem-1 power law (um).
WL_REF_UM = 0.55
#: Aerosol top altitude (km): linear taper reaches zero here.
AEROSOL_TOP_KM = 2.0

#: Vertical grid (km, strictly descending): 3 layers, lowest 2 aerosol-bearing.
ALTITUDE_GRID_KM: list[float] = [3.0, 2.0, 1.0, 0.0]

#: Number of Legendre moments written for the aerosol phase function.
N_LEGENDRE = 32

#: Wavelength grid (um) for the explicit aerosol layer files. Geometrically
#: spaced so uvspec's linear-in-wavelength interpolation of the Angstroem-1
#: power law stays accurate (max relative error ~3e-4) across both bands.
AEROSOL_WAVELENGTH_GRID_UM: list[float] = np.geomspace(0.2, 4.0, 65).tolist()

#: Path of the bundled LBL reference file.
_REFERENCE_PATH = Path(__file__).resolve().parent / "reference" / "randles2013_lbl.json"


def load_reference() -> dict:
    """Load the bundled LBL reference values (Tables 3/5/A3/A4).

    Returns:
        Dict with keys ``case1``/``case2a``/``case2b``, each mapping a
        quantity name to ``{"saw30": v, "saw75": v, "trop30": v, "trop75": v}``,
        plus a ``_meta`` block (citation, normalization constants, thresholds).
    """
    with open(_REFERENCE_PATH) as f:
        return json.load(f)


def taper_layer_weights(altitude_km) -> np.ndarray:
    """Per-layer column weights of the linear taper ``w(z) = max(0, 1 - z/2)``.

    The weight of layer *i* (between consecutive descending grid levels) is
    ``integral_layer w(z) dz / integral_0^2km w(z) dz``. ``w`` is piecewise
    linear, so the trapezoid over the layer endpoints is exact.

    Returns:
        Array of shape ``(n_layer,)`` summing to 1 on a grid covering 0-2 km.
    """
    z = np.asarray(altitude_km, dtype=float)
    w = np.clip(1.0 - z / AEROSOL_TOP_KM, 0.0, None)
    dz = z[:-1] - z[1:]  # positive: grid is strictly descending
    integrals = 0.5 * (w[:-1] + w[1:]) * dz
    total = float(integrals.sum())
    if total <= 0.0:
        raise ValueError("altitude grid does not intersect the 0-2 km aerosol layer")
    return integrals / total


@dataclass(frozen=True)
class RandlesAerosol:
    """Piece implementing the Randles 2013 case-2 aerosol (duck-typed Piece).

    Column optical depth follows the Angstroem-1 law
    ``tau_col(lambda) = AOD_550 * (lambda / 0.55 um)**-1`` and is distributed
    over layers with the linear-taper weights (assumption 2). SSA is constant
    per case (2a: 1.0, 2b: 0.8), g is constant 0.7, and the phase function is
    Henyey-Greenstein via PMOM moments ``beta_l = g**l`` (assumption 3).
    """

    ssa: float
    g: float = 0.7

    @property
    def name(self) -> str:
        """Signature used for explicit-aerosol file naming."""
        return f"RandlesAerosol(ssa={self.ssa:g})"

    def to_layer_optics(self, wl_um, altitude_km, n_legendre: int = 32) -> LayerOptics:
        """Return LayerOptics for this piece on the given grids.

        Args:
            wl_um: Wavelengths (um), shape (n_wl,).
            altitude_km: Descending layer-boundary altitudes (km), len n_layer+1.
            n_legendre: Number of Legendre moments.

        Returns:
            LayerOptics with tau/ssa/g of shape (n_wl, n_layer) and
            legendre_moments of shape (n_wl, n_layer, n_legendre).
        """
        wl = np.atleast_1d(np.asarray(wl_um, dtype=float))
        weights = taper_layer_weights(altitude_km)
        n_layer = len(weights)

        tau_col = AOD_550 * (wl / WL_REF_UM) ** -1.0
        tau = np.outer(tau_col, weights)
        ssa = np.full((len(wl), n_layer), float(self.ssa))
        g = np.full((len(wl), n_layer), float(self.g))
        beta = self.g ** np.arange(n_legendre, dtype=float)
        moments = np.broadcast_to(beta, (len(wl), n_layer, n_legendre)).copy()
        return LayerOptics(tau=tau, ssa=ssa, g=g, legendre_moments=moments)


def build_scene(
    case: str,
    atm: str,
    sza: int,
    band: str,
    *,
    aerosol_output_dir: str | os.PathLike | None = None,
) -> Scene:
    """Build one benchmark Scene.

    Args:
        case: One of :data:`CASES`. ``case1`` omits the aerosol entirely.
        atm: Key of :data:`ATMOSPHERES` ("saw" or "trop").
        sza: Solar zenith angle (30 or 75).
        band: Key of :data:`BANDS` ("bb" or "uvvis").
        aerosol_output_dir: Directory for the explicit aerosol layer files
            (aerosol cases only). Defaults to ``./aerosol`` (CompositeAerosol).

    Returns:
        A Scene with AFGL profile + reptran, DISORT 16 streams +
        ``disort_intcor moments`` + pseudospherical, albedo 0.2, and ASCII
        band-integrated (``output_process integrate``) output at surface and
        TOA with quantities [lambda, edir, edn, eup].
    """
    if case not in CASES:
        raise ValueError(f"Unknown case '{case}'. Valid: {list(CASES)}")
    if atm not in ATMOSPHERES:
        raise ValueError(f"Unknown atmosphere '{atm}'. Valid: {list(ATMOSPHERES)}")
    if sza not in SZAS:
        raise ValueError(f"Unknown sza {sza}. Valid: {list(SZAS)}")
    if band not in BANDS:
        raise ValueError(f"Unknown band '{band}'. Valid: {list(BANDS)}")

    scene = (
        Scene()
        .set_atmosphere(profile=ATMOSPHERES[atm], mol_abs_param="reptran")
        .set_source_solar(sza=float(sza))
        .set_wavelength(*BANDS[band])
        .set_solver(method="disort", streams=16, disort_intcor="moments", pseudospherical=True)
        .set_surface(albedo=SURFACE_ALBEDO)
        .set_output(
            quantities=["lambda", "edir", "edn", "eup"],
            format="ascii",
            zout=[0, "toa"],
            process="integrate",
        )
    )
    if case != "case1":
        ssa = 1.0 if case == "case2a" else 0.8
        scene = scene.set_aerosol(
            CompositeAerosol(
                pieces=[RandlesAerosol(ssa=ssa)],
                wavelength_grid_um=AEROSOL_WAVELENGTH_GRID_UM,
                altitude_grid_km=ALTITUDE_GRID_KM,
                n_legendre=N_LEGENDRE,
                output_dir=Path(aerosol_output_dir) if aerosol_output_dir else None,
            )
        )
    return scene


def _extract_fluxes(ds: xr.Dataset) -> dict[str, float]:
    """Pull surface/TOA band fluxes from one parsed ``integrate`` dataset.

    The parsed dataset has dims (wavelength: 1, zout: 2); wavelength is the
    collapsed band dimension of ``output_process integrate`` output.
    """
    zout = np.asarray(ds["zout"].values, dtype=float)
    i_sfc = int(np.argmin(zout))
    i_toa = int(np.argmax(zout))

    def flux(quantity: str, i_level: int) -> float:
        arr = np.asarray(ds[quantity].values, dtype=float)
        if arr.ndim != 2 or arr.shape[0] != 1:
            raise ValueError(
                f"Expected band-integrated output (single wavelength) for "
                f"{quantity}; got shape {arr.shape}"
            )
        return float(arr[0, i_level])

    return {
        "edir_sfc": flux("edir", i_sfc),
        "edn_sfc": flux("edn", i_sfc),
        "eup_sfc": flux("eup", i_sfc),
        "edir_toa": flux("edir", i_toa),
        "eup_toa": flux("eup", i_toa),
    }


def _prewrite_aerosol_files(scenes: list[Scene]) -> None:
    """Write explicit aerosol files serially before threaded execution.

    All runs of one aerosol case share the same content hash, hence the same
    layer files. Writing them here (in the main thread) means the worker
    threads only hit the on-disk cache — no concurrent writers on one file.
    """
    seen: set[str] = set()
    for scene in scenes:
        aerosol = scene.aerosol
        if aerosol is None:
            continue
        sig = str(aerosol)
        if sig in seen:
            continue
        aerosol.to_uvspec_lines()
        seen.add(sig)


def run_randles2013(
    output_dir: str | os.PathLike,
    *,
    uvspec_exe: str | None = None,
    data_path: str | None = None,
    cases: Sequence[str] | None = None,
) -> dict:
    """Run the Randles 2013 case matrix and return normalized fluxes.

    Executes ``len(cases) * 2 atmospheres * 2 SZAs * 2 bands`` uvspec
    invocations (24 with all cases) via :meth:`Runner.execute_many`, then
    normalizes every flux ``F_norm = F_raw / edir_toa_raw(same run) *
    C[band][sza]`` with the LBL median constants.

    Args:
        output_dir: Directory for the results file and aerosol layer files.
        uvspec_exe: Path to the uvspec binary (auto-detected if None).
        data_path: libRadtran data directory (auto-detected if None).
        cases: Subset of :data:`CASES` to run (default: all). RF entries for
            the aerosol cases are only produced when ``case1`` is included,
            since RF is the case-2 minus case-1 difference.

    Returns:
        Nested dict ``{case: {band: {atm: {sza: {quantity: value}}}}}`` with
        quantities ``edir_sfc``, ``edn_sfc``, ``eup_sfc``, ``eup_toa``,
        ``total_sfc_down`` (all normalized W/m2); case1 broadband additionally
        carries ``absorptance`` (dimensionless, same-run ratio) and
        ``nir_sfc_down`` (bb total minus uvvis total); the aerosol cases
        additionally carry ``rf_toa``/``rf_sfc`` (net case-2 minus case-1;
        net_toa = edir - eup at TOA, net_sfc = edir + edn - eup at surface).
        The whole dict is saved as ``<output_dir>/randles2013_results.json``
        and its path is returned under key ``results_path``.

    Raises:
        ValueError: If ``cases`` contains an unknown case name.
        RuntimeError: If any uvspec invocation fails (propagated from
            :meth:`Runner.execute_many`).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = CASES if cases is None else tuple(dict.fromkeys(cases))
    unknown = set(selected) - set(CASES)
    if unknown:
        raise ValueError(f"Unknown case(s) {sorted(unknown)}. Valid: {list(CASES)}")

    runs = [
        (case, atm, sza, band)
        for case in selected
        for atm in ATMOSPHERES
        for sza in SZAS
        for band in BANDS
    ]
    scenes = [
        build_scene(case, atm, sza, band, aerosol_output_dir=out_dir / "aerosol")
        for case, atm, sza, band in runs
    ]
    _prewrite_aerosol_files(scenes)

    datasets = Runner.execute_many(scenes, uvspec_exe=uvspec_exe, data_path=data_path)

    results: dict = {}
    raw_by_run: dict[tuple[str, str, str, int], dict[str, float]] = {}
    for (case, atm, sza, band), ds in zip(runs, datasets, strict=True):
        raw = _extract_fluxes(ds)
        raw_by_run[(case, band, atm, sza)] = raw
        scale = NORMALIZATION_CONSTANTS[band][sza] / raw["edir_toa"]
        entry = {
            "edir_sfc": raw["edir_sfc"] * scale,
            "edn_sfc": raw["edn_sfc"] * scale,
            "eup_sfc": raw["eup_sfc"] * scale,
            "eup_toa": raw["eup_toa"] * scale,
            "total_sfc_down": (raw["edir_sfc"] + raw["edn_sfc"]) * scale,
        }
        results.setdefault(case, {}).setdefault(band, {}).setdefault(atm, {})[sza] = entry

    if "case1" in results:
        for atm in ATMOSPHERES:
            for sza in SZAS:
                raw_bb = raw_by_run[("case1", "bb", atm, sza)]
                # Dimensionless same-run ratio: C cancels, so compute from raw.
                absorbed_surf = (1.0 - SURFACE_ALBEDO) * (raw_bb["edir_sfc"] + raw_bb["edn_sfc"])
                absorptance = 1.0 - (raw_bb["eup_toa"] + absorbed_surf) / raw_bb["edir_toa"]
                bb = results["case1"]["bb"][atm][sza]
                uvvis = results["case1"]["uvvis"][atm][sza]
                bb["absorptance"] = absorptance
                bb["nir_sfc_down"] = bb["total_sfc_down"] - uvvis["total_sfc_down"]

        for case in ("case2a", "case2b"):
            if case not in results:
                continue
            for band in BANDS:
                for atm in ATMOSPHERES:
                    for sza in SZAS:
                        base = results["case1"][band][atm][sza]
                        pert = results[case][band][atm][sza]
                        # net_toa = edir_toa - eup_toa = C - eup_toa: the C
                        # constants cancel in the case-2 minus case-1 difference.
                        pert["rf_toa"] = base["eup_toa"] - pert["eup_toa"]
                        pert["rf_sfc"] = (pert["total_sfc_down"] - pert["eup_sfc"]) - (
                            base["total_sfc_down"] - base["eup_sfc"]
                        )

    results_path = out_dir / "randles2013_results.json"
    results["results_path"] = str(results_path)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    return results
