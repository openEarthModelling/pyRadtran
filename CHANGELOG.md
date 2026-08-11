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
