"""Shared test fixtures for locating uvspec executable and data directories."""

import os
import shutil

import pytest

UVSPEC_EXE = shutil.which("uvspec")

_DATA_CANDIDATES = [
    os.environ.get("LIBRADTRAN_DATA_FILES"),
    "/usr/local/share/libRadtran/data",
    os.path.expanduser("~/libRadtran/data"),
]


@pytest.fixture
def uvspec_exe():
    """Path to uvspec binary, or skip if not found."""
    exe = UVSPEC_EXE
    if exe is None:
        candidate = os.path.join(
            os.path.dirname(__file__), "..", "..", "Radiation", "libRadtran-2.0.6", "bin", "uvspec"
        )
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
        pytest.skip("uvspec not found on PATH")
    return exe


@pytest.fixture
def data_path():
    """Path to libRadtran data directory, or skip if not found."""
    for candidate in _DATA_CANDIDATES:
        if candidate and os.path.isdir(candidate):
            return candidate
    pytest.skip("libRadtran data directory not found")


@pytest.fixture
def has_uvspec(uvspec_exe):
    """Marker fixture — test requires uvspec. Yields exe path."""
    return uvspec_exe
