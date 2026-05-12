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
- **Multiple Solver Support**: DISORT, MYSTIC Monte Carlo, twostream, rodents, and more
- **Aerosol Models**: OPAC presets, custom OPAC species, external optical property files, composite aerosols with Mie scattering
- **Atmospheric Profiles**: US standard, mid-latitude summer/winter, tropical, sub-arctic summer/winter
- **3D and Cloud Simulations**: Support for 3D radiative transfer and cloudy scenes
- **High-Level Convenience Functions**: Pre-configured simulations for transmittance, radiance, thermal brightness, lidar, satellite, and more
- **Parallel Execution**: Run multiple simulations in parallel with `Runner.execute_many()`
- **xarray Output**: Simulation results returned as self-describing `xarray.Dataset` objects
- **Full Type Safety**: Python 3.11+ with comprehensive type hints

## Prerequisites

pyRadtran requires **libRadtran** to be installed separately. libRadtran is not bundled with this package.

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

# Configure the scene
scene = (
    Scene()
    .set_atmosphere(profile="us", altitude=2.663)
    .set_source_solar(sza=30.0)
    .set_wavelength(250.0, 1200.0)
    .set_solver(method="disort", streams=16)
    .set_output(quantities=["lambda", "edir"], quantity="transmittance")
)

# Execute the simulation
result = Runner.execute(scene, data_path="/usr/local/share/libRadtran/data")

# Plot the result
result.edir.plot()
```

### Using Global Configuration

Avoid repeating `data_path` and `uvspec_exe` on every call:

```python
from pyradtran import Runner, RunnerConfig

Runner.configure(
    uvspec_exe="/usr/local/bin/uvspec",
    data_path="/usr/local/share/libRadtran/data",
)

# Now execute without repeating paths
result = Runner.execute(scene)
```

### Convenience Functions

For common tasks, use the high-level convenience API:

```python
from pyradtran import run_solar_transmittance, run_solar_radiance

# Solar spectral transmittance
transmittance = run_solar_transmittance(
    airmass=2.0,
    pwv=10.0,
    ozone=300.0,
    wl_min=300,
    wl_max=1200,
)

# Solar spectral radiance with aerosol
radiance = run_with_opac_preset(
    preset="maritime_clean",
    sza=45.0,
    wl_min=400,
    wl_max=800,
)
```

### Parallel Batch Execution

Run multiple scenes in parallel:

```python
from pyradtran import Runner

scenes = [scene1, scene2, scene3, scene4]
results = Runner.execute_many(scenes, max_workers=4)
```

### Composite Aerosol with Mie Scattering

```python
from pyradtran import CompositeAerosol, MieSpecies, SizeDistribution
from pyradtran.models.aerosol_composite import RefractiveIndex, IntegrationConfig

# Define a Mie species with log-normal size distribution
aerosol = CompositeAerosol(
    species=[
        MieSpecies(
            name="dust",
            size_distribution=SizeDistribution.log_normal(
                r_median=0.5,  # um
                sigma_g=2.0,
            ),
            refractive_index=RefractiveIndex.from_constant(n=1.53, k=0.008),
            density=2.6,  # g/cm3
        )
    ],
    integration=IntegrationConfig(n_legendre=32),
)

scene = Scene().set_aerosol(aerosol)
```

## Documentation

Full documentation is available at:

- **User Guide**: [https://openearthmodelling.github.io/pyRadtran/](https://openearthmodelling.github.io/pyRadtran/)
- **API Reference**: [https://openearthmodelling.github.io/pyRadtran/api.html](https://openearthmodelling.github.io/pyRadtran/api.html)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)

## Project Structure

```
pyradtran/
├── core/               # Execution engine (Runner, input builder, output parser)
├── models/             # Configuration models (atmosphere, aerosol, solver, etc.)
│   ├── aerosol.py      # OPAC and external aerosol models
│   ├── aerosol_composite.py  # Composite aerosol with Mie scattering
│   ├── atmosphere.py   # Atmospheric profile configuration
│   ├── solver.py       # RTE solver configuration
│   ├── source.py       # Radiation source configuration
│   └── ...
├── optics/             # Mie scattering and optical property calculations
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
