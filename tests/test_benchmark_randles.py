"""Randles et al. (2013) benchmark regression — real uvspec runs (slow + gated).

Runs the full benchmark matrix — 3 cases x saw/trop x SZA 30/75 x bb/uvvis,
24 uvspec invocations, ~20-60 s — once per session and asserts the paper's
headline results against the bundled LBL reference (loaded from
``load_reference()`` at test time, never hardcoded here):

1. Case 1 (Rayleigh-only): direct/diffuse broadband SFC-down, diffuse
   broadband TOA-up, UV-VIS and NIR SFC-down fluxes within +/-8% (the
   reference PASS band, :data:`FLUX_PASS_PCT`) of LBL for all four configs
   (saw30/saw75/trop30/trop75) — 20 assertions, supersets the brief's
   "8 minimum" (the brief only demands SAW30 + SZA75).
2. Case 2b @ SZA30: TOA radiative forcing POSITIVE for both atmospheres —
   the paper's central sign result (SSA-0.8 absorbing aerosol over the
   bright Rayleigh atmosphere / 0.2-albedo surface warms at TOA).
3. Case 2a: SFC radiative forcing NEGATIVE for all 4 configs.
4. :func:`compare_benchmark` over the full matrix: zero FAIL rows
   (WARN tolerated — see below); row summary printed (visible with ``-s``)
   and the Markdown/CSV report + overlay PNG written into the fixture
   tmp dir (asserted to exist and render with real data).

Expected WARN-tolerant quantities
---------------------------------
- ``case1/absorptance``: reptran-coarse band integration shifts broadband
  absorptance ~-4% to -7% relative to the LBL median. On the current
  libRadtran 2.0.6 run it stays inside the 8% PASS band, but it sits
  closest to the edge of all quantities; version drift is expected to
  push it into WARN (<=12%), not FAIL. It is therefore deliberately NOT
  part of the +/-8% test above and is only covered by the zero-FAIL gate.
- Small-flux SZA-75 rows (e.g. ``case1/nir_sfc_down`` trop75, ~+3.8%) may
  similarly drift between PASS and WARN without indicating a regression.
- ``case2a/atmospheric_rf`` rows are report-only (``"n/a"``: <1 W/m2, RSD
  invalid per the paper) and are excluded from the FAIL gate.

Gating: requests the conftest ``uvspec_exe``/``data_path`` fixtures, so the
whole module skips cleanly without libRadtran. Marked slow — deselect with
``-m "not slow"``.
"""

import csv
from pathlib import Path

import pytest

from pyradtran.benchmarks import (
    compare_benchmark,
    load_reference,
    plot_benchmark_overlay,
    run_randles2013,
    write_report,
)
from pyradtran.benchmarks.compare import CONFIGS, FLUX_PASS_PCT, NA_STATUS

pytestmark = pytest.mark.slow

#: Case-1 flux quantities asserted against the +/-8% band: reference
#: quantity -> (results band, results key). Covers the brief's "SFC-down BB
#: and TOA-up BB quantities" plus the UV-VIS/NIR split. ``absorptance`` is
#: excluded on purpose (reptran-coarse shift — see module docstring).
_CASE1_FLUX_QUANTITIES: dict[str, tuple[str, str]] = {
    "direct_bb_sfc_down": ("bb", "edir_sfc"),
    "diffuse_bb_sfc_down": ("bb", "edn_sfc"),
    "diffuse_bb_toa_up": ("bb", "eup_toa"),
    "uvvis_sfc_down": ("uvvis", "total_sfc_down"),
    "nir_sfc_down": ("bb", "nir_sfc_down"),
}

#: Memoization cache: the conftest gate fixtures are function-scoped and
#: pytest forbids a module-scoped fixture from requesting them (ScopeMismatch),
#: so the expensive full-matrix run is cached here instead — same
#: once-per-session semantics.
_MATRIX_CACHE: dict = {}


@pytest.fixture
def randles_results(uvspec_exe, data_path, tmp_path_factory):
    """Run the full 24-invocation Randles matrix once per session.

    Skips (via the gate fixtures) when uvspec or the libRadtran data
    directory is absent. The results dict (plus its saved
    ``randles2013_results.json`` under ``results_path``) is shared by every
    test in this module.
    """
    if "results" not in _MATRIX_CACHE:
        out_dir = tmp_path_factory.mktemp("randles2013")
        _MATRIX_CACHE["results"] = run_randles2013(
            out_dir, uvspec_exe=uvspec_exe, data_path=data_path
        )
    return _MATRIX_CACHE["results"]


def _split_config(config: str) -> tuple[str, int]:
    """Split a config key ("saw30"/"trop75") into (atmosphere, sza)."""
    return config[:-2], int(config[-2:])


@pytest.mark.parametrize("quantity", list(_CASE1_FLUX_QUANTITIES))
@pytest.mark.parametrize("config", CONFIGS)
def test_case1_fluxes_within_pass_band(randles_results, quantity, config):
    """Brief item 1: Case-1 SFC-down/TOA-up fluxes within +/-8% of LBL."""
    reference = load_reference()
    band, key = _CASE1_FLUX_QUANTITIES[quantity]
    atm, sza = _split_config(config)
    lbl = float(reference["case1"][quantity][config])
    ours = randles_results["case1"][band][atm][sza][key]
    rel_pct = (ours - lbl) / abs(lbl) * 100.0
    assert abs(rel_pct) <= FLUX_PASS_PCT, (
        f"case1 {quantity} {config}: ours {ours:.2f} vs LBL {lbl:.2f} W/m2 "
        f"({rel_pct:+.2f}% exceeds the +/-{FLUX_PASS_PCT:g}% PASS band)"
    )


@pytest.mark.parametrize("atm", ["saw", "trop"])
def test_case2b_toa_rf_positive_at_sza30(randles_results, atm):
    """Brief item 2: Case-2b TOA RF is positive at SZA30 (paper's sign result)."""
    rf = randles_results["case2b"]["bb"][atm][30]["rf_toa"]
    lbl = float(load_reference()["case2b"]["toa_rf"][f"{atm}30"])
    assert rf > 0.0, (
        f"case2b TOA RF @ {atm} SZA30 = {rf:.2f} W/m2, expected positive "
        f"(LBL reference {lbl:+.1f} W/m2)"
    )


@pytest.mark.parametrize("config", CONFIGS)
def test_case2a_sfc_rf_negative(randles_results, config):
    """Brief item 3: Case-2a SFC RF is negative in all 4 configs (less light down)."""
    atm, sza = _split_config(config)
    rf = randles_results["case2a"]["bb"][atm][sza]["rf_sfc"]
    lbl = float(load_reference()["case2a"]["sfc_rf"][config])
    assert rf < 0.0, (
        f"case2a SFC RF @ {config} = {rf:.2f} W/m2, expected negative "
        f"(LBL reference {lbl:+.1f} W/m2)"
    )


def test_compare_benchmark_no_fail_rows(randles_results):
    """Full-matrix comparison: zero FAIL rows (WARN tolerated, n/a report-only).

    Prints the status counts plus every non-PASS row (visible with ``-s``)
    so a WARN/FAIL breakdown is directly readable from the test log.
    """
    rows = compare_benchmark(randles_results)
    assert rows, "compare_benchmark produced no rows from the full matrix"

    fails = [r for r in rows if r["status"] == "FAIL"]
    warns = [r for r in rows if r["status"] == "WARN"]
    passes = [r for r in rows if r["status"] == "PASS"]
    report_only = [r for r in rows if r["status"] == NA_STATUS]

    print(
        f"\nRandles benchmark comparison: {len(rows)} rows — "
        f"{len(passes)} pass / {len(warns)} warn / {len(fails)} fail "
        f"(+{len(report_only)} report-only {NA_STATUS})"
    )
    for row in rows:
        if row["status"] != "PASS":
            print(
                f"  {row['status']:>4s} {row['case']}/{row['quantity']}/"
                f"{row['atm']}{row['sza']}: LBL {row['lbl']:g} vs ours "
                f"{row['ours']:g} ({row['rel_diff']:+.2f}%)"
            )

    assert not fails, f"{len(fails)} FAIL row(s) — see printed comparison above"


def test_report_and_overlay_render_from_real_data(randles_results):
    """Markdown/CSV report + overlay PNG render from the real matrix output.

    The artifacts are written into the fixture tmp dir (next to the saved
    results JSON) once per session as a sanity check that the report and
    figure tooling works with real uvspec data.
    """
    rows = compare_benchmark(randles_results)
    out_dir = Path(randles_results["results_path"]).parent

    md_path, csv_path = write_report(rows, out_dir / "randles2013_report.md")
    png_path = plot_benchmark_overlay(rows, out_dir / "randles2013_overlay.png")

    assert md_path.is_file() and md_path.stat().st_size > 0
    assert csv_path.is_file()
    assert png_path.is_file()

    with open(csv_path, newline="", encoding="utf-8") as f:
        n_csv_rows = sum(1 for _ in csv.DictReader(f))
    assert n_csv_rows == len(rows), f"CSV has {n_csv_rows} rows, expected {len(rows)}"

    with open(png_path, "rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n"  # PNG magic
    assert png_path.stat().st_size > 10_000
