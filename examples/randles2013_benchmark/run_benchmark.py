#!/usr/bin/env python3
"""Randles et al. (2013) AeroCom shortwave benchmark — end-to-end example.

Runs the full case matrix (3 cases x 2 AFGL atmospheres x 2 SZAs x 2 bands
= 24 uvspec invocations, ~20-60 s), normalizes against the LBL median
constants, compares against the bundled reference
(``pyradtran.benchmarks.reference.randles2013_lbl.json``) and writes

- ``output/randles2013_results.json`` — raw + normalized results
- ``output/randles2013_report.md`` / ``.csv`` — PASS/WARN/FAIL comparison
- ``output/randles2013_overlay.png`` — LBL-vs-pyRadtran bar overlay

``--from-json PATH`` regenerates the report and plot from an earlier
results JSON without re-running uvspec. The exit code is non-zero if any
row FAILs, so the driver doubles as a CI gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering; mirrors examples/multicomponent_viz

from pyradtran.benchmarks import (  # noqa: E402
    compare_benchmark,
    format_report,
    plot_benchmark_overlay,
    run_randles2013,
    write_report,
)

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "output"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Randles 2013 benchmark driver")
    parser.add_argument("--uvspec", default=None, help="uvspec binary (auto-detected if omitted)")
    parser.add_argument(
        "--data-path", default=None, help="libRadtran data directory (auto-detected)"
    )
    parser.add_argument(
        "--cases", nargs="*", default=None, help="subset of case1 case2a case2b (default: all)"
    )
    parser.add_argument(
        "--from-json", default=None, help="reuse an existing results JSON; skips uvspec"
    )
    parser.add_argument(
        "--out-dir", default=str(DEFAULT_OUT), help=f"output directory (default: {DEFAULT_OUT})"
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_json:
        results = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
        print(f"[randles2013] reusing results from {args.from_json}")
    else:
        print(f"[randles2013] running the case matrix -> {out_dir}")
        results = run_randles2013(
            out_dir, uvspec_exe=args.uvspec, data_path=args.data_path, cases=args.cases or None
        )

    rows = compare_benchmark(results)
    md_path, csv_path = write_report(rows, out_dir / "randles2013_report.md")
    png_path = plot_benchmark_overlay(rows, out_dir / "randles2013_overlay.png")

    print(format_report(rows))
    print(f"[randles2013] results: {results.get('results_path', '(in-memory)')}")
    print(f"[randles2013] report:  {md_path}")
    print(f"[randles2013] csv:     {csv_path}")
    print(f"[randles2013] plot:    {png_path}")

    n_fail = sum(row["status"] == "FAIL" for row in rows)
    n_warn = sum(row["status"] == "WARN" for row in rows)
    print(f"[randles2013] {len(rows)} rows: {n_fail} FAIL, {n_warn} WARN")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
