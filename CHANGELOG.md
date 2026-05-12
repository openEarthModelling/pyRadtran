# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
