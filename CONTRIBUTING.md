# Contributing to pyRadtran

Thank you for your interest in contributing! pyRadtran is a scientific tool, and every improvement helps the atmospheric science community.

## Development Setup

```bash
git clone https://github.com/openEarthModelling/pyRadtran.git
cd pyRadtran
pip install -e ".[dev]"
```

## Prerequisites

This library is a Python wrapper around [libRadtran](https://www.libradtran.org). You will need the libRadtran binary and data files installed on your system to run the full test suite or use the library. See the [README](README.md) for installation instructions.

## Running Tests

```bash
pytest tests/ -v
```

Tests that require libRadtran binaries are marked with `@pytest.mark.slow` or `@pytest.mark.integration` depending on the suite. To run only fast unit tests:

```bash
pytest tests/ -v -m "not slow and not integration"
```

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and import sorting.

```bash
ruff check src/ tests/
```

## Submitting Changes

1. Fork the repository and create a branch: `git checkout -b feature/my-feature`
2. Make your changes and add tests if applicable.
3. Ensure all tests pass: `pytest tests/ -v`
4. Run the linter: `ruff check src/ tests/`
5. Open a Pull Request with a clear description of the change and its motivation.

## Reporting Bugs

Please open an issue on GitHub with:
- A minimal reproducing script
- Your Python version and OS
- The libRadtran version you are using (if applicable)

## Release Process

Releases are automated via GitHub Actions when a maintainer pushes a semver tag (`v*.*.*`). Do not push tags manually unless you are a maintainer.
