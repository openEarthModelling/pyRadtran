# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Highlights: new `pyradtran.data` data layer with bundled libRadtran data, a LEGO-style "blocks" aerosol API, OPAC folding with real Mie phase functions, and a publication-ready visualization/workflow suite. Contains **breaking changes** to the aerosol API (see Removed).

#### Example self-validation + analysis expansion
- `compute_energy_budget` + `assert_energy_conservation` (`pyradtran.core.postprocess`): column energy identity `F_inc = eup_TOA + (1-a)(edir+edn)_surf + F_abs_atm` with hard physical-bound assertions.
- `parse_heating_ascii` (`pyradtran.core.output_parser`): parses libRadtran's wide heating-rate format (K/day per zout per wavelength). `Runner.execute` auto-detects heating mode (scene has `heating_rate` but no `output_user` quantities) and dispatches to it.
- Viz: `plot_size_distributions`, `plot_phase_functions`, `plot_legendre_decay`, `plot_block_spectral_optics`, `plot_drf_spectral` (direct radiative forcing), `plot_spectral_attribution`.
- `examples/multicomponent_viz`: expanded from 10 to 23 plots (per-block spectral optics, scattering phase functions, Legendre decay, heating-rate profile, DRF via no-aerosol baseline, spectral attribution) with a hard energy-conservation assertion and a column-τ-stacking check. Heating rates require a second uvspec invocation (libRadtran's `heating_rate` mode replaces flux output rather than appending a column); the demo runs both and merges.
- `examples/multicomponent_viz/canonical.py`: shared scene config incl. `build_scene_heating` (used by both the demo and the regression test so they cannot drift).
- `tests/fixtures/multicomponent_baseline.json` + `tests/test_multicomponent_regression.py` (slow/uvspec-gated): end-to-end regression baseline for the canonical 3-block scene.
- `scripts/regen_baseline.py` to regenerate the baseline.

### Added

#### Data layer (`pyradtran.data`)
- `DataResolver` with tiered data-root resolution (env var → bundled → system libRadtran); `bundled_only` mode enforced and exposed via `RunnerConfig.bundled_only` for strict offline runs.
- Curated libRadtran data subset bundled in the wheel (~60 MB, 144 files): OPAC aerosol optics, AFGL atmospheres, reptran correlated-k, CRS cross-sections, solar flux.
- `MANIFEST.toml` manifest loader + assets/manifest consistency checker with a CLI.
- `validate_scene` pre-flight check for high-value data references (warns by default, raises when `strict=True`).
- `DataResolver` exported in the public API.

#### LEGO "blocks" aerosol API (`pyradtran.models.blocks`)
- `VerticalProfile`, `MassProfile`, `ExponentialProfile`, and `TabulatedProfile` mass-column profiles.
- `od_to_mass_profile` helper for converting optical-depth profiles.
- `Piece` protocol and `PlacedBlock` abstraction (replaces `LoadedSpecies.evaluate`).
- `read_explicit_aerosol` + `DirectLayerOpticsBlock` for explicit aerosol layer files.
- `name` and `mass_per_particle_kg` fields on species blocks.

#### OPAC folding & Mie phase functions (`pyradtran.optics.opac`)
- OPAC ingredient readers: refractive index, size distribution, and preset profiles.
- `phase_function_to_legendre` converting a real Mie phase function to PMOM Legendre moments.
- Opt-in real Mie phase function on `MieSpecies` (`phase_function='mie'`).
- `BulkSpecies` backed by aerosol3D `BulkAerosolOpticsData`.
- Regression tests validating folded Mie optics against the published OPAC reference (spherical species).

#### Postprocess & workflow (`pyradtran.core.postprocess`, `pyradtran.workflow`)
- `add_budget_vars` / `compute_budget` for transmittance/reflectance/absorbance (T/R/A) budget derivation.
- `evaluate_composite_on_grid` / `evaluate_blocks_on_grid` for gridded composite evaluation.
- Component-attribution orchestration (N+1 leave-one-out).
- Output parser now retains physical `zout` altitudes and exposes a heating-rate column constant.

#### Visualization (`pyradtran.viz`)
- Publication theme with a colorblind-safe palette; matplotlib stays optional via lazy import.
- `plot_spectral`, `plot_flux_profile`, `plot_heating_rate`.
- `plot_budget` (stacked T/R/A) and `plot_rt_overview`.
- `plot_composite_optics` and `plot_block_profiles`.
- `plot_component_attribution`.

#### Examples & docs
- Comprehensive viz + workflow demo: `MieSpecies` → DISORT → all plots + attribution.
- Sphinx user guide, expanded API reference, and rewritten README.

#### YAML configuration front-end (`pyradtran.config`)
- `config_version: 1` YAML schema (strict: scene / aerosol / analysis; block kinds `mie`/`bulk`/`opac_preset`/`explicit_layer`; placements `od_inversion`/`mass`/`exponential`/`tabulated`), `load_config` / `export_config`, and the `run_config` orchestrator (energy-conservation assertion, heating second run, DRF baseline, leave-one-out attribution, plot registry, NetCDF export).
- CLI `pyradtran run|validate|export-config` (also `python -m pyradtran`): `validate` checks a config without invoking uvspec; `export-config` writes canonical YAML.
- Hard round-trip guarantee, enforced by `tests/test_config_roundtrip.py`: a YAML config and its API-built twin emit byte-identical uvspec input and byte-identical `.master` layer files; `export_config` output re-loads identically.
- `examples/multicomponent_viz/canonical.yaml` — YAML twin of the canonical scene, generated from `canonical.py` by `make_yaml.py`.
#### Benchmarks (`pyradtran.benchmarks`)
- Randles et al. (2013) AeroCom shortwave RT intercomparison replication: `run_randles2013` (3 cases × 2 AFGL atmospheres × 2 SZAs × 2 bands = 24 uvspec runs; Å=1 power law, 0–2 km linear taper, HG phase function as PMOM `beta_l = g**l`, LBL-median normalization constants).
- Bundled reference `reference/randles2013_lbl.json`: 64 tabulated LBL values plus a `_meta` block (thresholds: fluxes ±8% PASS / ±12% WARN; RF ≤15% or ≤1.5 W/m²).
- `compare_benchmark` / `format_report` / `write_report` / `plot_benchmark_overlay`: per-row status classification, Markdown + CSV report, LBL-overlay PNG.
- Regression test `tests/test_benchmark_randles.py` (slow, uvspec-gated): real run vs libRadtran 2.0.6 — 68 comparison rows, 64 PASS / 0 WARN / 0 FAIL (+4 report-only n/a).
- Ratified deviation: benchmark results persist as JSON, not NetCDF (nested ragged dict).
### Changed
- `evaluate_blocks_on_grid` (`pyradtran.core.postprocess`) now also returns per-block `ssa` and `g` (was `tau`/`rho_kg_m3` only).
- `OpacPreset` / `OpacCustom` are now `PlacedBlock` factories; OPAC convenience functions expose `output_dir`.
- Runner resolves the libRadtran data path through `DataResolver`.
- **Default `Output.format` is now `"ascii"`** (was `"netcdf"`): uvspec's NetCDF output is broken in many libRadtran builds (a 0-byte `.nc` from a libnetcdf ABI mismatch), so the previous default crashed every simulation that omitted `format=`. ASCII works everywhere and yields an equivalent `xarray.Dataset`; NetCDF remains available via `format="netcdf"`. The convenience functions no longer hardcode `"netcdf"` — they inherit the ASCII default.
- Distribution name corrected to `pyRadtran` (PyPI page title and `pip show` now preserve case; the project URL and wheel filename remain lowercase per PEP 503/625 normalization, which is unavoidable).

### Removed
- **Breaking:** `CompositeAerosol` now takes `pieces`; `LoadedSpecies` and `Species` are removed (`refactor(aerosol)!`).
- Removed the `ParticleOptics.from_aerosol3d()` / `PrecomputedSpecies` coupling layer (superseded by `BulkSpecies` + blocks).
- Removed vestigial `OPACSpecies` (netCDF layer-dump path) and `optics/opac_tables.py`.
- **Breaking:** removed `ExternalFile` / `ExternalAerosol` and the `run_with_aerosol` convenience function — the only non-LEGO aerosol path. Pre-computed explicit files now go through `DirectLayerOpticsBlock` mixed into a `CompositeAerosol`. Also removed the unused `AerosolModel.set_tau_at_wvl` / `king_byrne` fields (never set by any code path).

### Fixed
- ASCII output parser names the 8th uvspec column `heating_rate` (was `col_7`) so `HEATING_RATE_COLUMN` resolves when libRadtran emits heating rates alongside the standard 7 flux columns.
- `_trapz` guard added to `tests/test_composite_aerosol_unit.py` normalization tests (numpy <2.0 compat; production `mie.py` already had the guard).
- `run_satellite` test now skips cleanly when libRadtran's `MPS` satellite-geometry file is absent (not bundled).
- libRadtran integration issues for explicit aerosol coupling.
- ASCII column-name mapping and aerosol-layer file cache collision.
- Mixing-rule HG fallback now uses `g_l` (PMOM) Legendre moments consistently across species.
- NumPy 2.x compatibility (`np.trapz` removed; replaced with `np.trapezoid` / a `_trapz` shim).
- `AttributionLike` made read-only so a frozen `AttributionResult` satisfies the protocol.
- CI branch triggers, lint, and test assertions realigned for green CI.
- Convenience functions no longer crash on the NetCDF parse path: they now use ASCII output (see **Changed**). `_parse_netcdf` raises a clear error pointing at `format='ascii'` when uvspec produces an empty/missing NetCDF file instead of xarray's cryptic backend-mismatch message.
- `run_with_opac_preset` / `run_with_opac_custom` now set `disort_intcor="moments"` on their DISORT solver. The OPAC folding produces Legendre-moment phase functions, which DISORT rejects without this flag (`you need to specify 'disort_intcor moments'`); these functions previously crashed at runtime.
- **i550 wavelength-index bug** in the multicomponent demo and its regression test: `argmin(|wavelength - 0.55|)` compared an nm grid against 0.55 µm, silently selecting 401 nm everywhere a 550 nm scalar was labeled (energy log, heating log, baseline fixture). Also removed a spurious `× 1000` on the DRF wavelength axis (same µm/nm confusion) that made `np.interp(550, ...)` clamp to the 401 nm endpoint. The committed baseline now holds true 550 nm values (edir_surf 798.75, eup_toa 194.79 W/m², F_abs_atm 336.94 W/m²).

- `Runner.execute_many` no longer swallows worker failures: failing scenes were silently returned to callers as `(idx, exception)` result entries; the first failure now raises `RuntimeError(f"scene {i} failed")` chained to the original exception, after cancelling outstanding futures.
- `Runner.execute_many` uses a thread pool instead of a process pool: scenes carrying `CompositeAerosol` can hold unpicklable payloads (aerosol3D bulk size distributions store local closures), which crashed `ProcessPoolExecutor` pickling; uvspec runs as a subprocess and releases the GIL while waiting, so threads parallelize the invocations just as well and nothing needs pickling.
## [0.1.0] - 2026-05-12

### Added
- Initial release of pyRadtran.
- Pythonic wrapper for libRadtran radiative transfer simulations.
- `Scene` builder API with fluent interface for configuring simulations.
- `Runner` for executing libRadtran simulations and parsing output.
- Aerosol model support: OPAC presets, custom OPAC, external aerosol files.
- Composite aerosol support with Mie scattering integration and size distributions.
- Convenience functions for common simulation configurations (3D, cloudy, lidar, polarized, satellite, etc.).
- Full test suite with pytest.
- Documentation with Sphinx and ReadTheDocs theme.
- `__version__` attribute exposed via `importlib.metadata`.
- Codecov coverage reporting in CI.
- Dedicated `lint.yml` workflow with ruff check and format check.
- Automated release notes generation from CHANGELOG in publish workflow.

### Fixed
- NumPy 2.0 compatibility: replaced deprecated `np.trapz` with `np.trapezoid`.
- CI branch triggers aligned to `master`.
- Added `twine check` to PyPI publish workflow.
