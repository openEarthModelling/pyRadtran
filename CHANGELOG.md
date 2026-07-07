# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Highlights: new `pyradtran.data` data layer with bundled libRadtran data, a LEGO-style "blocks" aerosol API, OPAC folding with real Mie phase functions, and a publication-ready visualization/workflow suite. Contains **breaking changes** to the aerosol API (see Removed).

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
- `OpacPreset` / `OpacCustom` are now `PlacedBlock` factories; OPAC convenience functions expose `output_dir`.
- Runner resolves the libRadtran data path through `DataResolver`.
- Distribution name corrected to `pyRadtran` (PyPI page title and `pip show` now preserve case; the project URL and wheel filename remain lowercase per PEP 503/625 normalization, which is unavoidable).

### Removed
- **Breaking:** `CompositeAerosol` now takes `pieces`; `LoadedSpecies` and `Species` are removed (`refactor(aerosol)!`).
- Removed the `ParticleOptics.from_aerosol3d()` / `PrecomputedSpecies` coupling layer (superseded by `BulkSpecies` + blocks).
- Removed vestigial `OPACSpecies` (netCDF layer-dump path) and `optics/opac_tables.py`.

### Fixed
- libRadtran integration issues for explicit aerosol coupling.
- ASCII column-name mapping and aerosol-layer file cache collision.
- Mixing-rule HG fallback now uses `g_l` (PMOM) Legendre moments consistently across species.
- NumPy 2.x compatibility (`np.trapz` removed; replaced with `np.trapezoid` / a `_trapz` shim).
- `AttributionLike` made read-only so a frozen `AttributionResult` satisfies the protocol.
- CI branch triggers, lint, and test assertions realigned for green CI.

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
