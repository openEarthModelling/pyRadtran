# pyRadtran

[![CI](https://github.com/openEarthModelling/pyRadtran/actions/workflows/ci.yml/badge.svg)](https://github.com/openEarthModelling/pyRadtran/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pyradtran.svg)](https://pypi.org/project/pyradtran/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Docs](https://readthedocs.org/projects/pyradtran/badge/?version=latest)](https://pyradtran.readthedocs.io/en/latest/)

A complete Python wrapper for [libRadtran](https://www.libradtran.org) radiative transfer simulations, built on [Pydantic](https://docs.pydantic.dev) for validation and [xarray](https://docs.xarray.dev) for output.

## Prerequisites

This library is a **Python wrapper** around libRadtran. You must install libRadtran (Fortran binary + data files) separately before using pyRadtran.

1. **Install libRadtran**: Follow the official [installation guide](https://libradtran.org/download.php) or use a package manager:
   - **conda**: `conda install -c conda-forge libradtran`
   - **apt (Debian/Ubuntu)**: `sudo apt install libradtran`
   - **Source**: Download from [libradtran.org](https://www.libradtran.org) and compile with `make`

2. **Set the data path**: Ensure the libRadtran data directory is available. You will pass it to `Runner.execute(...)` via the `data_path` argument (e.g. `/usr/local/share/libRadtran/data`).

## Installation

```bash
pip install pyradtran
```

With optional plotting support:

```bash
pip install pyradtran[plot]
```

For development:

```bash
git clone https://github.com/openEarthModelling/pyRadtran.git
cd pyRadtran
pip install -e ".[dev]"
```

## Quick Start

```python
from pyradtran import Scene, Runner

scene = (
    Scene()
    .set_atmosphere(profile="us", altitude=2.663)
    .set_source_solar(sza=30.0)
    .set_wavelength(250.0, 1200.0)
    .set_solver(method="disort", streams=16)
    .set_output(quantities=["lambda", "edir"], quantity="transmittance")
)

result = Runner.execute(scene, data_path="/usr/local/share/libRadtran/data")
result.edir.plot()
```

See the [documentation](https://pyradtran.readthedocs.io) for more examples, including aerosol configurations, 3D scenes, and satellite simulations.

## Key Features

- **Fluent Scene Builder** — configure atmospheres, sources, wavelengths, solvers, and aerosols through a chainable API.
- **Aerosol Modelling** — OPAC presets, custom OPAC, external aerosol files, and composite aerosols with Mie scattering.
- **Convenience Functions** — pre-configured setups for 3D, cloudy, lidar, polarized, satellite, solar, and thermal simulations.
- **Type-Safe** — built on Pydantic v2 for runtime validation and excellent IDE autocompletion.
- **Typed** — ships `py.typed` for full type-checker support.

## Documentation

Full documentation is hosted at [pyradtran.readthedocs.io](https://pyradtran.readthedocs.io), including API reference, tutorials, and example notebooks.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on development setup, testing, and submitting pull requests.

## License

This project is licensed under the [MIT License](LICENSE).

Note: pyRadtran calls the libRadtran binary at runtime. libRadtran is licensed under the GNU GPL v2. Because pyRadtran interacts with libRadtran as a separate program (via subprocess), the pyRadtran code itself is not considered a derivative work under the GPL and is released under the MIT license. Users who use pyRadtran must comply with both the MIT terms and the libRadtran GPL terms.

## Citation

If you use pyRadtran in your research, please cite it. A `CITATION.cff` file is included in this repository for your convenience.
