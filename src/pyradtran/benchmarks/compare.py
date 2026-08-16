"""Compare benchmark results against the bundled LBL reference.

Turns a results dict from :func:`pyradtran.benchmarks.run_randles2013`
(shape ``{case: {band: {atm: {sza: {quantity: value}}}}}``, optionally plus a
stray ``results_path`` string key) into flat comparison rows, a Markdown/CSV
report, and a LBL-vs-pyRadtran overlay figure.

Status classification
---------------------
Fluxes (every quantity except ``*_rf``): PASS if ``|rel| <= 8%``, WARN if
``|rel| <= 12%``, else FAIL (reference ``_meta`` thresholds).

Radiative forcing (``toa_rf``/``sfc_rf``): PASS if ``|rel| <= 15%`` OR
``|abs| <= 1.5 W/m2`` (reference ``_meta``). The paper only fixes the PASS
band; by analogy we set WARN if ``|rel| <= 25%`` OR ``|abs| <= 3 W/m2``, and
FAIL beyond both. ``rel_diff`` is in percent; when the reference value is 0
the absolute difference (quantity units) is used as the fallback so the
thresholds remain well defined.

Case-2a atmospheric RF (``toa_rf - sfc_rf``, <1 W/m2 and RSD-invalid per the
paper) is emitted as report-only rows with status ``"n/a"``.

Robustness
----------
- Non-case keys in ``results`` (e.g. ``results_path``) are ignored: only the
  known case names are read.
- SZA keys may be ints (in-memory) or strings ("30"; results loaded back from
  JSON); both are accepted.
- RF reference values are broadband-only, so RF rows are emitted exclusively
  from the ``bb`` band; any ``rf_*`` entries under ``uvvis`` are ignored.
"""

from __future__ import annotations

import csv
from pathlib import Path

from pyradtran.benchmarks.randles2013 import CASES, load_reference

#: Config keys of the reference file, in canonical report order.
CONFIGS: tuple[str, ...] = ("saw30", "saw75", "trop30", "trop75")

# Thresholds (kept in sync with reference/_meta "thresholds"; the RF WARN band
# is our documented extension of the paper's PASS-only rule).
FLUX_PASS_PCT = 8.0
FLUX_WARN_PCT = 12.0
RF_PASS_REL_PCT = 15.0
RF_PASS_ABS_W_M2 = 1.5
RF_WARN_REL_PCT = 25.0
RF_WARN_ABS_W_M2 = 3.0

#: Status -> Markdown emoji (plain words are used in the CSV).
STATUS_EMOJI: dict[str, str] = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}

#: Report-only status for case-2a atmospheric RF rows.
NA_STATUS = "n/a"

#: Reference quantity -> (results band, results quantity key), per case.
_CASE2_QUANTITIES: dict[str, tuple[str, str]] = {
    "total_bb_sfc_down": ("bb", "total_sfc_down"),
    "diffuse_bb_toa_up": ("bb", "eup_toa"),
    "uvvis_sfc_down": ("uvvis", "total_sfc_down"),
    # Broadband only: the reference RF is a bb difference; uvvis rf_* ignored.
    "toa_rf": ("bb", "rf_toa"),
    "sfc_rf": ("bb", "rf_sfc"),
}
_QUANTITY_MAP: dict[str, dict[str, tuple[str, str]]] = {
    "case1": {
        "direct_bb_sfc_down": ("bb", "edir_sfc"),
        "diffuse_bb_sfc_down": ("bb", "edn_sfc"),
        "diffuse_bb_toa_up": ("bb", "eup_toa"),
        "uvvis_sfc_down": ("uvvis", "total_sfc_down"),
        "nir_sfc_down": ("bb", "nir_sfc_down"),
        "absorptance": ("bb", "absorptance"),
    },
    "case2a": _CASE2_QUANTITIES,
    "case2b": _CASE2_QUANTITIES,
}

#: Column order of the CSV side of :func:`write_report`.
CSV_COLUMNS: tuple[str, ...] = (
    "benchmark",
    "case",
    "quantity",
    "band",
    "atm",
    "sza",
    "units",
    "lbl",
    "ours",
    "abs_diff",
    "rel_diff",
    "status",
)


def _split_config(config: str) -> tuple[str, int]:
    """Split a config key ("saw30"/"trop75") into (atmosphere, sza)."""
    return config[:-2], int(config[-2:])


def _lookup_ours(results: dict, case: str, quantity: str, config: str) -> float | None:
    """Fetch our value for (case, reference quantity, config).

    Returns None when the results dict does not carry the entry (subset run).
    SZA keys are accepted as int or str (JSON round trip).
    """
    band, key = _QUANTITY_MAP[case][quantity]
    atm, sza = _split_config(config)
    entry = results.get(case, {}).get(band, {}).get(atm)
    if not isinstance(entry, dict):
        return None
    for sza_key in (sza, str(sza)):
        run = entry.get(sza_key)
        if isinstance(run, dict):
            value = run.get(key)
            if value is not None:
                return float(value)
    return None


def _status_for(quantity: str, abs_diff: float, rel_diff: float) -> str:
    """Classify one comparison using the flux or RF rule (see module docstring)."""
    if quantity.endswith("_rf"):
        if abs(rel_diff) <= RF_PASS_REL_PCT or abs(abs_diff) <= RF_PASS_ABS_W_M2:
            return "PASS"
        if abs(rel_diff) <= RF_WARN_REL_PCT or abs(abs_diff) <= RF_WARN_ABS_W_M2:
            return "WARN"
        return "FAIL"
    if abs(rel_diff) <= FLUX_PASS_PCT:
        return "PASS"
    if abs(rel_diff) <= FLUX_WARN_PCT:
        return "WARN"
    return "FAIL"


def compare_benchmark(results: dict, benchmark: str = "randles2013") -> list[dict]:
    """Compare run results against the bundled LBL reference.

    Args:
        results: Output of :func:`run_randles2013` (possibly loaded back from
            its saved JSON, in which case SZA keys are strings and a
            ``results_path`` key is present; both are handled).
        benchmark: Benchmark whose reference to compare against. Currently
            only ``"randles2013"``.

    Returns:
        One row per (case, quantity, atm, sza) with fields ``case``,
        ``quantity``, ``band``, ``atm``, ``sza``, ``units``, ``lbl``, ``ours``,
        ``abs_diff``, ``rel_diff`` (percent; absolute fallback when ``lbl == 0``)
        and ``status`` (PASS/WARN/FAIL; case-2a ``atmospheric_rf`` rows are
        report-only with status ``"n/a"``). Quantities missing from the results
        dict are skipped, so subset runs yield fewer rows.

    Raises:
        ValueError: If ``benchmark`` names no known benchmark.
    """
    if benchmark != "randles2013":
        raise ValueError(f"Unknown benchmark '{benchmark}'. Valid: ['randles2013']")
    reference = load_reference()

    rows: list[dict] = []
    # Iterate only known case keys: stray keys (results_path, ...) are skipped.
    for case in CASES:
        if case not in results:
            continue
        for quantity, config_values in reference[case].items():
            band = _QUANTITY_MAP[case][quantity][0]
            for config in CONFIGS:
                atm, sza = _split_config(config)
                lbl = float(config_values[config])
                ours = _lookup_ours(results, case, quantity, config)
                if ours is None:
                    continue
                abs_diff = ours - lbl
                rel_diff = abs_diff if lbl == 0.0 else abs_diff / abs(lbl) * 100.0
                rows.append(
                    {
                        "benchmark": benchmark,
                        "case": case,
                        "quantity": quantity,
                        "band": band,
                        "atm": atm,
                        "sza": sza,
                        "units": "1" if quantity == "absorptance" else "W/m2",
                        "lbl": lbl,
                        "ours": ours,
                        "abs_diff": abs_diff,
                        "rel_diff": rel_diff,
                        "status": _status_for(quantity, abs_diff, rel_diff),
                    }
                )
        if case == "case2a":
            rows.extend(_atmospheric_rf_rows(results, reference))
    return rows


def _atmospheric_rf_rows(results: dict, reference: dict) -> list[dict]:
    """Report-only rows for the case-2a atmospheric RF (``toa_rf - sfc_rf``).

    The paper flags case-2a atmospheric RF as <1 W/m2 with invalid RSD, so it
    is reported (LBL value derived as the difference of the tabulated toa/sfc
    reference RFs) but not classified; ``status`` is ``"n/a"``.
    """
    rows = []
    for config in CONFIGS:
        atm, sza = _split_config(config)
        toa = _lookup_ours(results, "case2a", "toa_rf", config)
        sfc = _lookup_ours(results, "case2a", "sfc_rf", config)
        if toa is None or sfc is None:
            continue
        ours = toa - sfc
        lbl = reference["case2a"]["toa_rf"][config] - reference["case2a"]["sfc_rf"][config]
        abs_diff = ours - lbl
        rel_diff = abs_diff if lbl == 0.0 else abs_diff / abs(lbl) * 100.0
        rows.append(
            {
                "benchmark": "randles2013",
                "case": "case2a",
                "quantity": "atmospheric_rf",
                "band": "bb",
                "atm": atm,
                "sza": sza,
                "units": "W/m2",
                "lbl": float(lbl),
                "ours": ours,
                "abs_diff": abs_diff,
                "rel_diff": rel_diff,
                "status": NA_STATUS,
            }
        )
    return rows


def _cell(row: dict) -> str:
    """Render one row as a Markdown table cell: ``lbl / ours (rel) status``."""
    rel = (
        f"{row['rel_diff']:+.1f}%"
        if row["lbl"] != 0.0
        else f"{row['rel_diff']:+.2f} (abs)"  # percent fallback not defined
    )
    status = row["status"]
    tag = STATUS_EMOJI.get(status, status)
    return f"{row['lbl']:g} / {row['ours']:g} ({rel}) {tag}"


def format_report(rows: list[dict]) -> str:
    """Render comparison rows as Markdown.

    One table per case with quantities as rows and the four configs
    (saw30/saw75/trop30/trop75) as columns; each cell shows
    ``LBL / ours (rel diff) status-emoji``. A header line carries the global
    counts (``N pass / N warn / N fail``) and each case heading repeats its
    own counts.
    """
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0, NA_STATUS: 0}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1

    lines: list[str] = []
    benchmark = rows[0]["benchmark"] if rows else "benchmark"
    lines.append(f"# {benchmark} — pyRadtran vs LBL reference")
    lines.append("")
    summary = f"**{counts['PASS']} pass / {counts['WARN']} warn / {counts['FAIL']} fail**"
    if counts[NA_STATUS]:
        summary += f" (+{counts[NA_STATUS]} report-only n/a)"
    lines.append(f"{summary} — {len(rows)} rows")
    lines.append("")
    lines.append(
        f"Thresholds — fluxes: PASS |rel| ≤ {FLUX_PASS_PCT:g}% / WARN ≤ {FLUX_WARN_PCT:g}% / "
        f"else FAIL; RF: PASS |rel| ≤ {RF_PASS_REL_PCT:g}% or |abs| ≤ {RF_PASS_ABS_W_M2:g} W/m2, "
        f"WARN ≤ {RF_WARN_REL_PCT:g}% or ≤ {RF_WARN_ABS_W_M2:g} W/m2, else FAIL. "
        f"Cells: LBL / ours (rel diff)."
    )

    if not rows:
        lines.append("")
        lines.append("(no rows)")
        return "\n".join(lines) + "\n"

    case_order = [c for c in CASES if any(row["case"] == c for row in rows)]
    for case in case_order:
        case_rows = [row for row in rows if row["case"] == case]
        cc = {"PASS": 0, "WARN": 0, "FAIL": 0, NA_STATUS: 0}
        for row in case_rows:
            cc[row["status"]] = cc.get(row["status"], 0) + 1
        heading = f"## {case} — {cc['PASS']} pass / {cc['WARN']} warn / {cc['FAIL']} fail"
        if cc[NA_STATUS]:
            heading += f" (+{cc[NA_STATUS]} n/a)"
        lines.append("")
        lines.append(heading)
        lines.append("")
        lines.append("| quantity | " + " | ".join(CONFIGS) + " |")
        lines.append("| --- | " + " | ".join(["---"] * len(CONFIGS)) + " |")
        by_key: dict[tuple[str, str], dict] = {
            (row["quantity"], f"{row['atm']}{row['sza']}"): row for row in case_rows
        }
        quantities = list(dict.fromkeys(row["quantity"] for row in case_rows))
        for quantity in quantities:
            units = next(r["units"] for r in case_rows if r["quantity"] == quantity)
            cells = []
            for config in CONFIGS:
                row = by_key.get((quantity, config))
                cells.append(_cell(row) if row else "—")
            lines.append(f"| {quantity} ({units}) | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_report(rows: list[dict], path: str | Path) -> tuple[Path, Path]:
    """Write the Markdown report to ``path`` and a CSV to the same stem.

    The CSV uses :data:`CSV_COLUMNS` with plain status words (PASS/WARN/FAIL,
    ``n/a``) instead of emojis.

    Returns:
        ``(markdown_path, csv_path)``.
    """
    md_path = Path(path)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(format_report(rows), encoding="utf-8")

    csv_path = md_path.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: _csv_cell(row.get(col)) for col in CSV_COLUMNS})
    return md_path, csv_path


def _csv_cell(value) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


# Short tick labels for the overlay panels (full names stay in the report).
_QUANTITY_SHORT: dict[str, str] = {
    "direct_bb_sfc_down": "direct",
    "diffuse_bb_sfc_down": "diffuse",
    "diffuse_bb_toa_up": "diffuse",
    "uvvis_sfc_down": "uvvis",
    "nir_sfc_down": "nir",
    "total_bb_sfc_down": "total",
    "toa_rf": "toa",
    "sfc_rf": "sfc",
    "atmospheric_rf": "atm",
}


def plot_benchmark_overlay(rows: list[dict], path: str | Path) -> Path:
    """Grouped LBL-vs-pyRadtran bar chart per case and save it as PNG.

    One row of panels per case: surface-downwelling fluxes, TOA-upwelling
    fluxes, and — for the aerosol cases 2a/2b — radiative forcing. Each group
    is one (config, quantity) pair with two adjacent bars (LBL reference in
    gray, pyRadtran in blue). The dimensionless ``absorptance`` is not plotted
    (it would break the shared W/m2 flux axis); it stays in the report.

    matplotlib is imported lazily (Agg-safe headless rendering).

    Args:
        rows: Rows from :func:`compare_benchmark`.
        path: Destination PNG path.

    Returns:
        The saved path.

    Raises:
        ValueError: If ``rows`` is empty.
    """
    if not rows:
        raise ValueError("plot_benchmark_overlay requires at least one comparison row")

    from pyradtran.viz._style import get_palette, require_mpl, set_theme

    require_mpl()
    import matplotlib.pyplot as plt

    set_theme()
    ours_color = get_palette(1)[0]
    lbl_color = "#7f7f7f"

    by_key = {(row["case"], row["quantity"], f"{row['atm']}{row['sza']}"): row for row in rows}
    case_order = [c for c in CASES if any(row["case"] == c for row in rows)]

    panels = (
        ("sfc_down", "SFC downwelling fluxes"),
        ("toa_up", "TOA upwelling fluxes"),
        ("rf", "Radiative forcing"),
    )

    def panel_quantities(case: str) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {"sfc_down": [], "toa_up": [], "rf": []}
        for quantity in dict.fromkeys(r["quantity"] for r in rows if r["case"] == case):
            if quantity.endswith("_rf"):
                groups["rf"].append(quantity)
            elif quantity.endswith("_sfc_down"):
                groups["sfc_down"].append(quantity)
            elif quantity.endswith("_toa_up"):
                groups["toa_up"].append(quantity)
        return groups

    fig, axes = plt.subplots(
        nrows=len(case_order),
        ncols=3,
        figsize=(15, 3.6 * len(case_order) + 0.8),
        squeeze=False,
    )
    fig.suptitle("Randles 2013 benchmark — LBL vs pyRadtran (W/m2)", fontsize=13)

    for i, case in enumerate(case_order):
        groups = panel_quantities(case)
        for j, (panel, title) in enumerate(panels):
            ax = axes[i][j]
            quantities = groups[panel]
            if not quantities:
                ax.set_visible(False)
                continue
            # Subset runs may cover only part of CONFIGS: skip absent
            # (config, quantity) pairs — like format_report's "—" cells —
            # so x positions stay contiguous instead of raising KeyError.
            positions = [
                (config, quantity)
                for config in CONFIGS
                for quantity in quantities
                if (case, quantity, config) in by_key
            ]
            left = [by_key[(case, quantity, config)]["lbl"] for config, quantity in positions]
            right = [by_key[(case, quantity, config)]["ours"] for config, quantity in positions]
            x = range(len(positions))
            width = 0.38
            ax.bar(
                [p - width / 2 for p in x],
                left,
                width=width,
                color=lbl_color,
                label="LBL reference",
            )
            ax.bar(
                [p + width / 2 for p in x], right, width=width, color=ours_color, label="pyRadtran"
            )
            ax.set_xticks(
                list(x),
                [
                    f"{config}\n{_QUANTITY_SHORT.get(quantity, quantity)}"
                    for config, quantity in positions
                ],
                fontsize=7,
            )
            if panel == "rf":
                ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_title(f"{case} — {title}", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{case}\nW/m2")
            if i == 0 and j == 0:
                ax.legend(loc="best", fontsize=8)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
