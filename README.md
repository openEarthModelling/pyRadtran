# pyRadtran

[![CI](https://github.com/openEarthModelling/pyRadtran/actions/workflows/ci.yml/badge.svg)](https://github.com/openEarthModelling/pyRadtran/actions/workflows/ci.yml)
[![Lint](https://github.com/openEarthModelling/pyRadtran/actions/workflows/lint.yml/badge.svg)](https://github.com/openEarthModelling/pyRadtran/actions/workflows/lint.yml)
[![codecov](https://codecov.io/gh/openEarthModelling/pyRadtran/branch/master/graph/badge.svg)](https://codecov.io/gh/openEarthModelling/pyRadtran)
[![PyPI version](https://badge.fury.io/py/pyradtran.svg)](https://pypi.org/project/pyradtran/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete Python wrapper for [libRadtran](https://www.libradtran.org) radiative transfer simulations.

pyRadtran provides a Pythonic, type-safe API for configuring and executing libRadtran's `uvspec` radiative transfer code. It features an immutable scene builder with a fluent interface, comprehensive aerosol model support, and high-level convenience functions for common atmospheric science tasks.

## Features

- **Scene Builder API**: Immutable, chainable configuration with type-safe Pydantic models
- **Bundled Data Layer**: a curated libRadtran data subset (~60 MB) ships in the wheel and is resolved automatically by `DataResolver` (env var → bundled → system libRadtran)
- **LEGO Blocks Aerosol API**: separate species optics from vertical placement; externally mix any number of blocks into one `CompositeAerosol`
- **Multiple Solver Support**: DISORT, MYSTIC Monte Carlo, twostream, rodents, and more
- **Aerosol Models**: OPAC presets, custom OPAC species, external optical property files, Mie-species blocks, and bulk optics from Aerosol3D
- **Atmospheric Profiles**: US standard, mid-latitude summer/winter, tropical, sub-arctic summer/winter
- **3D and Cloud Simulations**: Support for 3D radiative transfer and cloudy scenes
- **T/R/A Budget & Grid Diagnostics**: `add_budget_vars`/`compute_budget` and analytic composite/block evaluation without re-running RT
- **Publication Visualization**: spectral, flux-profile, heating-rate, T/R/A budget, overview, composite-optics, block-profile, and component-attribution plots (matplotlib, optional)
- **Component Attribution**: leave-one-out per-block RT attribution workflow
- **High-Level Convenience Functions**: Pre-configured simulations for transmittance, radiance, thermal brightness, lidar, satellite, and more
- **Parallel Execution**: Run multiple simulations in parallel with `Runner.execute_many()`
- **xarray Output**: Simulation results returned as self-describing `xarray.Dataset` objects
- **Full Type Safety**: Python 3.11+ with comprehensive type hints

## Prerequisites

pyRadtran requires the libRadtran **`uvspec` binary** to be installed separately (it is not bundled). A curated subset of the libRadtran **data** files (~60 MB: OPAC aerosol optics, AFGL atmospheres, reptran correlated-k, CRS cross-sections, solar flux) **is** bundled in the wheel and resolved automatically by `DataResolver`. Set `PYRADTRAN_DATA_PATH`, `LIBRADTRAN_DATA_FILES`, or `LIBRADTRANDIR` to use a full libRadtran data tree instead.

### Installing libRadtran

Download and install libRadtran from the [official website](https://www.libradtran.org) or GitHub:

```bash
# Download (example for version 2.0.6)
wget https://github.com/rayference/libradtran/archive/refs/tags/v2.0.6.tar.gz
tar -xzf v2.0.6.tar.gz
cd libradtran-2.0.6

# Build and install
./configure --prefix=/usr/local
make
sudo make install
```

After installation, ensure the `uvspec` binary is on your `PATH`, or set the environment variable:

```bash
export LIBRADTRANDIR=/usr/local/share/libRadtran
# or
export LIBRADTRAN_DATA_FILES=/usr/local/share/libRadtran/data
```

## Installation

```bash
pip install pyradtran
```

For development (includes test dependencies):

```bash
pip install pyradtran[dev]
```

For plotting support:

```bash
pip install pyradtran[plot]
```

## Quick Start

### Basic Transmittance Calculation

```python
from pyradtran import Scene, Runner

scene = (
    Scene()
    .set_atmosphere(profile="us", altitude=2.663)
    .set_source_solar(sza=30.0)
    .set_wavelength(400.0, 700.0)
    .set_solver(method="disort", streams=16)
    .set_output(quantities=["lambda", "edir"], format="ascii", zout=[0, "toa"])
)

# data_path=None -> resolve data via DataResolver (env var -> bundled -> system libRadtran)
result = Runner.execute(scene, data_path=None)

# Plot the result
result.edir.plot()
```

### Using Global Configuration

Avoid repeating `data_path` and `uvspec_exe` on every call:

```python
from pyradtran import Scene, Runner, RunnerConfig

scene = (
    Scene()
    .set_atmosphere(profile="us")
    .set_source_solar(sza=30.0)
    .set_wavelength(400.0, 700.0)
    .set_solver(method="disort", streams=16)
    .set_output(quantities=["lambda", "edir"], format="ascii", zout=[0, "toa"])
)

# Set global defaults once (data_path=None -> DataResolver)
Runner.configure(RunnerConfig(data_path=None))

# Now execute without repeating paths
result = Runner.execute(scene)
```

### Convenience Functions

For common tasks, use the high-level convenience API:

```python
# skip-doc-check: run_with_opac_preset works after the disort_intcor fix, but the
# OPAC aerosol data has spline edge quirks at some wide bands (e.g. 400-700 nm).
from pyradtran import run_solar_transmittance, run_with_opac_preset

# Solar spectral transmittance (data_path omitted -> DataResolver)
transmittance = run_solar_transmittance(
    airmass=2.0, pwv=10.0, ozone=300.0, wl_min=400, wl_max=700
)

# Solar spectral radiance with an OPAC aerosol preset
radiance = run_with_opac_preset(
    preset="continental_average", sza=45.0, wl_min=400, wl_max=700
)
```

### Parallel Batch Execution

Run multiple scenes in parallel:

```python
from pyradtran import Scene, Runner


def _scene(profile):
    return (
        Scene().set_atmosphere(profile=profile)
        .set_source_solar(sza=30.0)
        .set_wavelength(400.0, 700.0)
        .set_solver(method="disort", streams=16)
        .set_output(quantities=["lambda", "edir"], format="ascii", zout=[0, "toa"])
    )


scenes = [_scene(p) for p in ("us", "ms", "mw")]
results = Runner.execute_many(scenes, max_workers=3)
print(len(results))
```

### Composite Aerosol (LEGO blocks)

```python
from pyradtran import Scene, Runner
from pyradtran.models.aerosol_composite import (
    CompositeAerosol, IntegrationConfig, MieSpecies, RefractiveIndex, SizeDistribution,
)
from pyradtran.models.blocks import PlacedBlock, od_to_mass_profile

altitude_km = [8.0, 6.0, 4.0, 2.0, 0.0]
ri = RefractiveIndex(wavelength_um=[0.40, 0.55, 0.70], n_real=[1.53] * 3, k_imag=[0.008] * 3)
sd = SizeDistribution(kind="lognormal", params={"r_g_um": 0.50, "sigma_g": 2.2})
dust = MieSpecies(refractive_index=ri, size_distribution=sd,
                  particle_density_kg_m3=2600.0, integration_config=IntegrationConfig(), name="dust")
piece = PlacedBlock(block=dust, profile=od_to_mass_profile(
    dust, tau_ref=0.20, ref_nm=550.0, altitude_km=altitude_km, scale_height_km=3.0))
aerosol = CompositeAerosol(pieces=[piece], wavelength_grid_um=[0.50, 0.55, 0.60],
                           altitude_grid_km=altitude_km, n_legendre=32, output_dir=".")
scene = (Scene().set_atmosphere(profile="us").set_source_solar(sza=30.0)
         .set_wavelength(500.0, 600.0).set_solver(method="disort", streams=16, disort_intcor="moments")
         .set_output(quantities=["lambda", "edir", "edn", "eup"], format="ascii", zout=[0, 2, "toa"])
         .set_aerosol(aerosol))
result = Runner.execute(scene, data_path=None)
```

## Documentation

Full documentation is available at:

- **User Guide**: [https://openearthmodelling.github.io/pyRadtran/](https://openearthmodelling.github.io/pyRadtran/)
- **API Reference**: [https://openearthmodelling.github.io/pyRadtran/api.html](https://openearthmodelling.github.io/pyRadtran/api.html)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)

## Project Structure

```
pyradtran/
├── core/               # Execution engine (Runner, input builder, output parser, postprocess)
├── data/               # Bundled libRadtran data + DataResolver (tiered data-root resolution)
├── models/             # Configuration models (atmosphere, aerosol, solver, etc.)
│   ├── aerosol.py           # OPAC and external aerosol models
│   ├── aerosol_composite.py # CompositeAerosol, MieSpecies, BulkSpecies, SizeDistribution
│   ├── blocks.py            # LEGO blocks: profiles, PlacedBlock, DirectLayerOpticsBlock
│   ├── atmosphere.py        # Atmospheric profile configuration
│   ├── solver.py            # RTE solver configuration
│   └── ...
├── optics/             # Mie scattering, mixing rules, layer writer, OPAC folding
├── viz/                # Publication plots (RT, composite, block, attribution)
├── workflow/           # RT orchestration (component attribution)
├── scene.py            # Immutable Scene builder API
├── convenience.py      # High-level convenience functions
└── presets.py          # Common altitude and configuration presets
```

## Supported libRadtran Features

pyRadtran supports a growing subset of libRadtran's `uvspec` capabilities:

| Feature | Status |
|---------|--------|
| 1D radiative transfer (DISORT, twostream, rodents) | Supported |
| MYSTIC Monte Carlo | Supported |
| Solar and thermal sources | Supported |
| Standard atmospheric profiles | Supported |
| OPAC aerosol models | Supported |
| Custom aerosol (external files) | Supported |
| Composite aerosol (Mie scattering) | Supported |
| Cloud configurations | Supported |
| 3D radiative transfer | Supported |
| Lidar/SSLidar simulations | Supported |
| Satellite geometry | Supported |
| Polarized radiative transfer | Supported |
| LEGO blocks aerosol API | Supported |
| Bundled data layer (DataResolver) | Supported |
| T/R/A budget postprocessing | Supported |
| Publication visualization suite | Supported |
| Component attribution workflow | Supported |
| Output formats (netCDF, ASCII) | Supported |

See the [design documents](docs/superpowers/specs/) for detailed coverage analysis and planned features.

## Development

```bash
# Clone the repository
git clone https://github.com/openEarthModelling/pyRadtran.git
cd pyRadtran

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=pyradtran --cov-report=html

# Run linting
ruff check src/ tests/
ruff format --check src/ tests/

# Build documentation
cd docs
make html
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

pyRadtran is built on top of the excellent [libRadtran](https://www.libradtran.org) radiative transfer package by Claudia Emde, Robert Buras, and colleagues.

## Citation

If you use pyRadtran in your research, please cite both pyRadtran and libRadtran:

```bibtex
@software{pyradtran,
  author = {Zhang, Fan},
  title = {pyRadtran: A Python wrapper for libRadtran},
  url = {https://github.com/openEarthModelling/pyRadtran},
  year = {2026}
}

@article{libradtran2016,
  author = {Emde, C. and Buras-Schnell, R. and Kylling, A. and Mayer, B. and Gasteiger, J. and Hamann, U. and Kylling, J. and Richter, B. and Pause, C. and Dowling, T. and Bugliaro, L.},
  title = {The libRadtran software package for radiative transfer calculations},
  journal = {Geoscientific Model Development},
  volume = {9},
  pages = {1647--1672},
  year = {2016},
  doi = {10.5194/gmd-9-1647-2016}
}
```
