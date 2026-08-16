Benchmarks
==========

``pyradtran.benchmarks`` replicates published radiative-transfer
intercomparisons end-to-end with real uvspec runs and compares the results
against the published line-by-line (LBL) reference values. It currently ships
one benchmark: the shortwave aerosol experiment of Randles et al. (2013).

Randles et al. (2013) protocol
------------------------------

Randles, C. A. et al., "Intercomparison of shortwave radiative transfer
schemes to improve the aerosol and surface albedo treatment in climate
models", Atmos. Chem. Phys. 13, 2347-2362 (2013) — the AeroCom RT experiment,
Tables 3, 5, A3 and A4. Three cases:

.. list-table::
   :header-rows: 1

   * - Case
     - Aerosol
     - AOD550
     - Ångström
     - SSA
     - g
   * - ``case1``
     - none (Rayleigh only)
     - —
     - —
     - —
     - —
   * - ``case2a``
     - lowest 2 km
     - 0.2
     - 1
     - 1.0
     - 0.7
   * - ``case2b``
     - as 2a
     - 0.2
     - 1
     - 0.8
     - 0.7

Fixed conditions: surface albedo 0.2 (Lambert), SZA 30° and 75°, AFGL subarctic
winter (``saw``) and tropics (``trop``) atmospheres, no clouds, and two bands —
broadband 0.2–4.0 µm (``bb``) and UV-VIS 0.2–0.7 µm (``uvvis``). The full
matrix, 3 cases × 2 atmospheres × 2 SZAs × 2 bands, is 24 uvspec invocations
(~20–60 s). The solver setup is DISORT with 16 streams, ``disort_intcor
moments``, pseudospherical geometry and reptran correlated-k with
band-integrated output; the phase function is supplied as 32 Legendre moments.

Replication assumptions
-----------------------

The paper leaves details to each participating model; this replication fixes
them as:

1. Built-in AFGL profiles are used (the participants used their
   individual-model profiles; the comparison tolerance absorbs this).
2. "Aerosol linear in the lowest 2 km" is a linear taper from the surface to
   zero at 2 km, ``w(z) = max(0, 1 - z / 2 km)``.
3. ``g = 0.7`` Henyey–Greenstein phase function represented through PMOM
   moments ``beta_l = g**l`` (the DISORT/libRadtran coefficient form).
4. Ångström exponent 1 means ``tau(lambda) = 0.2 * (lambda / 0.55 um)**-1``.

Normalization
-------------

Each result is divided by the same run's band TOA incident flux and multiplied
by the LBL median constants, which removes the solar-constant choice.
Formally ``F_norm = F_raw / edir_toa_raw(same run) * C[band][sza]`` with:

.. list-table::
   :header-rows: 1

   * - Constant (W/m²)
     - SZA 30°
     - SZA 75°
   * - Broadband ``bb``
     - 1189.28
     - 355.43
   * - UV-VIS ``uvvis``
     - 563.38
     - 168.37

Reference and thresholds
------------------------

``pyradtran/benchmarks/reference/randles2013_lbl.json`` bundles the 64
tabulated LBL values (per case and quantity, for the four configs
``saw30`` / ``saw75`` / ``trop30`` / ``trop75``) plus a ``_meta`` block holding
the citation and the comparison thresholds:

.. list-table::
   :header-rows: 1

   * - Quantity
     - PASS
     - WARN
     - FAIL
   * - Fluxes (everything except ``*_rf``)
     - ``|rel| <= 8%``
     - ``|rel| <= 12%``
     - beyond
   * - Radiative forcing (``toa_rf`` / ``sfc_rf``)
     - ``|rel| <= 15%`` **or** ``|abs| <= 1.5 W/m²``
     - ``|rel| <= 25%`` or ``|abs| <= 3 W/m²``
     - beyond both

Case-2a atmospheric forcing (``toa_rf - sfc_rf``) is below 1 W/m² and
RSD-invalid per the paper; those four rows are classified ``n/a`` and are
report-only.

Running the benchmark
---------------------

.. code-block:: python

   from pyradtran.benchmarks import (
       CASES,
       compare_benchmark,
       format_report,
       load_reference,
       plot_benchmark_overlay,
       run_randles2013,
       write_report,
   )

   reference = load_reference()  # bundled LBL values + _meta
   print(CASES, sorted(k for k in reference if not k.startswith("_")))

   def full_benchmark(out_dir, uvspec_exe=None, data_path=None):
       """Run the matrix (needs uvspec), compare against LBL, emit report + plot."""
       results = run_randles2013(out_dir, uvspec_exe=uvspec_exe, data_path=data_path)
       rows = compare_benchmark(results)  # one status row per case/quantity/config
       md, csv = write_report(rows, f"{out_dir}/randles2013.md")
       png = plot_benchmark_overlay(rows, f"{out_dir}/randles2013_overlay.png")
       return format_report(rows), md, csv, png

:func:`~pyradtran.benchmarks.randles2013.run_randles2013` runs the whole
matrix (pass ``cases=`` to restrict it) and returns a nested results dict.

The regression suite runs the real matrix. It is marked ``slow`` and skips
automatically when uvspec or the libRadtran data directory are not found:

.. code-block:: console

   $ python -m pytest tests/test_benchmark_randles.py -v -s   # ~25 s, real uvspec runs
   $ python -m pytest -q -m "not slow"                        # the rest of the suite

A ready-to-run driver for the whole workflow lives at
``examples/randles2013_benchmark/run_benchmark.py``: it runs the matrix,
compares against the reference and writes the Markdown/CSV report and the
overlay PNG to its ``output/`` directory. ``--from-json`` regenerates the
report from an earlier results JSON without re-running uvspec; the exit
code is non-zero if any row FAILs.

Outputs
-------

- ``compare_benchmark(results)`` — one row per (case, quantity, config) with
  ``lbl``, ``ours``, ``abs_diff``, ``rel_diff`` and ``status`` (``PASS`` /
  ``WARN`` / ``FAIL`` / ``n/a``).
- ``format_report(rows)`` — a Markdown report string, grouped by case with a
  status-count summary.
- ``write_report(rows, path)`` — writes the Markdown report plus a CSV twin
  next to it; returns both paths.
- ``plot_benchmark_overlay(rows, path)`` — a PNG overlaying this run's values
  on the LBL reference.

.. note::

   ``run_randles2013`` persists its results as **JSON**, not NetCDF: the
   matrix is a nested, ragged dict (per-case quantities differ), which NetCDF
   cannot represent. The JSON additionally carries a ``results_path`` key
   pointing at the artifact.

Validation result
-----------------

Real run against libRadtran 2.0.6 (uvspec, 2026-08-16): the regression suite
passes — 28 tests in 22.84 s — and the full-matrix comparison yields
**68 rows: 64 PASS / 0 WARN / 0 FAIL** (plus the 4 report-only ``n/a`` rows).
The three largest deviations, all comfortably inside the 8% PASS band:

.. list-table::
   :header-rows: 1

   * - Row (case / quantity / config)
     - LBL
     - pyRadtran
     - rel.
   * - case1 / absorptance / trop75
     - 0.307
     - 0.2858
     - -6.91%
   * - case1 / nir_sfc_down / trop75
     - 101.1
     - 104.95
     - +3.81%
   * - case1 / direct_bb_sfc_down / trop75
     - 179.6
     - 184.26
     - +2.59%

The paper's central case-2b result reproduces: positive TOA forcing at SZA 30
(ours +11.4 / +10.3 vs LBL +11.6 / +10.3 W/m² for ``saw`` / ``trop``) and
negative surface forcing in every configuration.
