"""Benchmarks: published radiative-transfer intercomparison case matrices.

Currently ships the Randles et al. (2013) shortwave benchmark
(:mod:`pyradtran.benchmarks.randles2013`) with its bundled LBL reference
values.
"""

from __future__ import annotations

from pyradtran.benchmarks.randles2013 import (
    CASES,
    RandlesAerosol,
    load_reference,
    run_randles2013,
)

__all__ = [
    "CASES",
    "RandlesAerosol",
    "load_reference",
    "run_randles2013",
]
