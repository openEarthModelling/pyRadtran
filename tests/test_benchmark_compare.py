"""Offline unit tests for benchmark comparison, reporting and overlay plots.

Synthetic results dicts are built from the bundled LBL reference itself
(ours = lbl x multiplier, plus fixed RF offsets), so no uvspec is needed.
"""

import csv
from collections import Counter
from pathlib import Path

import pytest

from pyradtran.benchmarks import (
    compare_benchmark,
    format_report,
    load_reference,
    plot_benchmark_overlay,
    write_report,
)
from pyradtran.benchmarks.compare import CONFIGS, _status_for
from pyradtran.benchmarks.randles2013 import CASES

REF = load_reference()

#: case1 carries 6 reference quantities, the aerosol cases 5 each; case2a
#: additionally gets 4 report-only atmospheric_rf rows.
EXPECTED_ROWS = {"case1": 24, "case2a": 24, "case2b": 20}


def build_results(flux_mult=1.0, rf_offset=0.0, *, string_sza_keys=False, with_path=False):
    """Synthetic run results whose entries sit at known offsets from the reference.

    Flux-like entries are scaled by ``flux_mult``; RF entries are shifted by
    ``rf_offset`` (W/m2). ``uvvis`` bands carry bogus ``rf_*`` entries (99.9)
    that the comparison must ignore (reference RF is broadband-only).
    """
    results: dict = {}

    def put(case, band, atm, sza, entry):
        target = results.setdefault(case, {}).setdefault(band, {}).setdefault(atm, {})
        target[str(sza) if string_sza_keys else sza] = entry

    for config in CONFIGS:
        atm, sza = config[:-2], int(config[-2:])
        c1 = REF["case1"]
        put(
            "case1",
            "bb",
            atm,
            sza,
            {
                "edir_sfc": c1["direct_bb_sfc_down"][config] * flux_mult,
                "edn_sfc": c1["diffuse_bb_sfc_down"][config] * flux_mult,
                "eup_sfc": 100.0,
                "eup_toa": c1["diffuse_bb_toa_up"][config] * flux_mult,
                "total_sfc_down": 0.0,
                "nir_sfc_down": c1["nir_sfc_down"][config] * flux_mult,
                "absorptance": c1["absorptance"][config] * flux_mult,
            },
        )
        put(
            "case1",
            "uvvis",
            atm,
            sza,
            {
                "edir_sfc": 1.0,
                "edn_sfc": 1.0,
                "eup_sfc": 1.0,
                "eup_toa": 1.0,
                "total_sfc_down": c1["uvvis_sfc_down"][config] * flux_mult,
            },
        )
        for case in ("case2a", "case2b"):
            rc = REF[case]
            put(
                case,
                "bb",
                atm,
                sza,
                {
                    "edir_sfc": 1.0,
                    "edn_sfc": 1.0,
                    "eup_sfc": 1.0,
                    "eup_toa": rc["diffuse_bb_toa_up"][config] * flux_mult,
                    "total_sfc_down": rc["total_bb_sfc_down"][config] * flux_mult,
                    "rf_toa": rc["toa_rf"][config] + rf_offset,
                    "rf_sfc": rc["sfc_rf"][config] + rf_offset,
                },
            )
            put(
                case,
                "uvvis",
                atm,
                sza,
                {
                    "edir_sfc": 1.0,
                    "edn_sfc": 1.0,
                    "eup_sfc": 1.0,
                    "eup_toa": 1.0,
                    "total_sfc_down": rc["uvvis_sfc_down"][config] * flux_mult,
                    "rf_toa": 99.9,
                    "rf_sfc": 99.9,
                },
            )
    if with_path:
        results["results_path"] = "/tmp/randles2013_results.json"
    return results


class TestCompareBenchmark:
    @pytest.mark.parametrize(
        ("mult", "expected"),
        [(1.0, "PASS"), (1.10, "WARN"), (1.20, "FAIL")],
    )
    def test_flux_status_at_0_10_20_percent(self, mult, expected):
        rows = compare_benchmark(build_results(flux_mult=mult))
        flux_rows = [r for r in rows if not r["quantity"].endswith("_rf")]
        assert len(flux_rows) == 48
        statuses = {r["status"] for r in flux_rows}
        assert statuses == {expected}
        for row in flux_rows:
            assert row["rel_diff"] == pytest.approx((mult - 1.0) * 100.0, abs=1e-9)
            assert row["abs_diff"] == pytest.approx(row["lbl"] * (mult - 1.0), abs=1e-9)

    def test_exact_results_all_pass_except_report_only(self):
        rows = compare_benchmark(build_results())
        assert Counter(r["status"] for r in rows) == Counter({"PASS": 64, "n/a": 4})

    def test_row_shape_and_counts_per_case(self):
        rows = compare_benchmark(build_results())
        required = {
            "case",
            "quantity",
            "atm",
            "sza",
            "lbl",
            "ours",
            "abs_diff",
            "rel_diff",
            "status",
        }
        assert all(required <= set(row) for row in rows)
        assert Counter(r["case"] for r in rows) == Counter(EXPECTED_ROWS)
        assert len(rows) == sum(EXPECTED_ROWS.values()) == 68
        for case in CASES:
            assert {r["atm"] for r in rows if r["case"] == case} == {"saw", "trop"}
            assert {r["sza"] for r in rows if r["case"] == case} == {30, 75}

    def test_rf_rows_broadband_only_and_report_only_atmospheric(self):
        rows = compare_benchmark(build_results())
        rf_rows = [r for r in rows if r["quantity"].endswith("_rf")]
        assert {r["quantity"] for r in rf_rows} == {"toa_rf", "sfc_rf", "atmospheric_rf"}
        # The uvvis rf_* entries (99.9) never leak: RF rows read band "bb" only.
        assert all(r["band"] == "bb" for r in rf_rows)
        assert all(r["status"] == "PASS" for r in rf_rows if r["quantity"] != "atmospheric_rf")
        atm_rows = [r for r in rf_rows if r["quantity"] == "atmospheric_rf"]
        assert len(atm_rows) == 4
        assert all(r["status"] == "n/a" for r in atm_rows)
        assert all(r["case"] == "case2a" for r in atm_rows)
        saw30 = next(r for r in atm_rows if r["atm"] == "saw" and r["sza"] == 30)
        assert saw30["lbl"] == pytest.approx(-8.6 - (-9.7))  # toa_rf - sfc_rf
        assert saw30["ours"] == pytest.approx(saw30["lbl"])

    def test_rf_abs_clause_fires_beyond_the_rel_band(self):
        results = build_results()
        # case2b toa_rf trop75: smallest-magnitude RF reference (-6.5 W/m2),
        # so a +1.4 offset is >15% relative but <=1.5 W/m2 absolute.
        results["case2b"]["bb"]["trop"][75]["rf_toa"] = -6.5 + 1.4
        row = _find(compare_benchmark(results), "case2b", "toa_rf", "trop", 75)
        assert row["abs_diff"] == pytest.approx(1.4)
        assert abs(row["rel_diff"]) > 15.0  # rel clause cannot fire
        assert row["status"] == "PASS"  # via the 1.5 W/m2 clause
        # Past 1.5 but within 3 W/m2 (and >25% relative) -> WARN.
        results["case2b"]["bb"]["trop"][75]["rf_toa"] = -6.5 + 2.0
        row = _find(compare_benchmark(results), "case2b", "toa_rf", "trop", 75)
        assert row["status"] == "WARN"
        # Beyond both bands -> FAIL.
        results["case2b"]["bb"]["trop"][75]["rf_toa"] = -6.5 + 5.0
        row = _find(compare_benchmark(results), "case2b", "toa_rf", "trop", 75)
        assert row["status"] == "FAIL"

    def test_rf_rule_boundary_direct(self):
        # 0.5 W/m2 abs at 100% rel -> PASS via the 1.5 W/m2 clause.
        assert _status_for("toa_rf", abs_diff=0.5, rel_diff=100.0) == "PASS"
        # 2.0 W/m2 at 100% rel -> not PASS (WARN via the 3 W/m2 clause).
        assert _status_for("toa_rf", abs_diff=2.0, rel_diff=100.0) == "WARN"
        # 3.5 W/m2 at 200% rel -> beyond both bands.
        assert _status_for("toa_rf", abs_diff=3.5, rel_diff=200.0) == "FAIL"
        # Relative clause alone.
        assert _status_for("sfc_rf", abs_diff=10.0, rel_diff=10.0) == "PASS"

    def test_results_path_key_ignored_and_string_sza_keys_accepted(self):
        rows = compare_benchmark(build_results(string_sza_keys=True, with_path=True))
        assert len(rows) == 68
        assert Counter(r["case"] for r in rows) == Counter(EXPECTED_ROWS)
        assert all(r["status"] in ("PASS", "n/a") for r in rows)

    def test_missing_quantity_rows_skipped(self):
        results = build_results()
        for atm in ("saw", "trop"):
            for sza in (30, 75):
                del results["case1"]["bb"][atm][sza]["nir_sfc_down"]
        rows = compare_benchmark(results)
        assert Counter(r["case"] for r in rows) == Counter(
            {"case1": 20, "case2a": 24, "case2b": 20}
        )
        assert "nir_sfc_down" not in {r["quantity"] for r in rows}

    def test_empty_results_give_no_rows(self):
        assert compare_benchmark({}) == []

    def test_unknown_benchmark_raises(self):
        with pytest.raises(ValueError, match="benchmark"):
            compare_benchmark(build_results(), benchmark="ipcc_ar6")


class TestReport:
    def test_markdown_contains_counts_and_statuses(self):
        rows = compare_benchmark(build_results(flux_mult=1.20))
        md = format_report(rows)
        assert "**16 pass / 0 warn / 48 fail** (+4 report-only n/a)" in md
        assert "❌" in md and "✅" in md
        # case1 carries only flux quantities -> all 24 fail at +20%.
        assert "## case1 — 0 pass / 0 warn / 24 fail" in md

        rows = compare_benchmark(build_results(flux_mult=1.10))
        md = format_report(rows)
        assert "**16 pass / 48 warn / 0 fail**" in md
        assert "⚠️" in md

    def test_markdown_one_table_per_case_with_all_configs(self):
        md = format_report(compare_benchmark(build_results()))
        for case in CASES:
            assert f"## {case} —" in md
        assert "| quantity | saw30 | saw75 | trop30 | trop75 |" in md
        assert "942.4 / 942.4 (+0.0%) ✅" in md  # direct_bb_sfc_down saw30
        assert "atmospheric_rf (W/m2)" in md  # report-only row present

    def test_empty_rows_report(self):
        md = format_report([])
        assert "no rows" in md

    def test_write_report_markdown_and_csv_roundtrip(self, tmp_path):
        rows = compare_benchmark(build_results(flux_mult=1.10))
        md_path, csv_path = write_report(rows, tmp_path / "report.md")
        assert isinstance(md_path, Path) and isinstance(csv_path, Path)
        assert md_path == tmp_path / "report.md"
        assert csv_path == tmp_path / "report.csv"
        assert md_path.is_file() and csv_path.is_file()
        assert "48 warn" in md_path.read_text(encoding="utf-8")

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            csv_rows = list(reader)
        assert len(csv_rows) == len(rows) == 68
        required = {
            "case",
            "quantity",
            "atm",
            "sza",
            "lbl",
            "ours",
            "abs_diff",
            "rel_diff",
            "status",
        }
        assert required <= set(reader.fieldnames)
        assert len(reader.fieldnames) == 12
        assert {r["status"] for r in csv_rows} <= {"PASS", "WARN", "FAIL", "n/a"}
        assert csv_rows[0]["case"] == rows[0]["case"]
        assert csv_rows[0]["quantity"] == rows[0]["quantity"]
        assert float(csv_rows[0]["lbl"]) == pytest.approx(rows[0]["lbl"])
        assert float(csv_rows[0]["rel_diff"]) == pytest.approx(rows[0]["rel_diff"])


class TestOverlayPlot:
    def test_writes_nonempty_png(self, tmp_path):
        rows = compare_benchmark(build_results(flux_mult=1.10))
        out = tmp_path / "overlay.png"
        returned = plot_benchmark_overlay(rows, out)
        assert Path(returned) == out
        assert out.is_file()
        with open(out, "rb") as f:
            assert f.read(8) == b"\x89PNG\r\n\x1a\n"  # PNG magic
        assert out.stat().st_size > 10_000

    def test_case1_only_rows_render(self, tmp_path):
        results = {"case1": build_results()["case1"]}
        out = tmp_path / "case1.png"
        plot_benchmark_overlay(compare_benchmark(results), out)
        assert out.stat().st_size > 10_000

    def test_empty_rows_raise(self, tmp_path):
        with pytest.raises(ValueError, match="row"):
            plot_benchmark_overlay([], tmp_path / "empty.png")


def _find(rows, case, quantity, atm, sza):
    return next(
        r
        for r in rows
        if r["case"] == case and r["quantity"] == quantity and r["atm"] == atm and r["sza"] == sza
    )
