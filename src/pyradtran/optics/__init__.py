"""Aerosol optics computation subpackage.

Modules:
    mie: BHMIE scattering and size-distribution integration.
    mixing: External mixing rules for multiple aerosol sources.
    layer_writer: explicit-file master + per-layer .LAYER writer.
"""

from pyradtran.optics.layer_writer import write_explicit_aerosol
from pyradtran.optics.mie import bhmie, integrate_size_distribution
from pyradtran.optics.mixing import combine_sources

__all__ = [
    "bhmie",
    "integrate_size_distribution",
    "combine_sources",
    "write_explicit_aerosol",
]
