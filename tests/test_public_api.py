"""Smoke test: public API is importable without matplotlib installed at import time.

(`import pyradtran` must not pull matplotlib; viz symbols import lazily.)
"""
from __future__ import annotations


def test_pyradtran_imports_without_matplotlib_dependency():
    import importlib

    import pyradtran

    importlib.reload(pyradtran)  # no exception -> top-level import is matplotlib-free


def test_public_symbols_exist():
    import pyradtran

    for name in (
        "add_budget_vars",
        "compute_budget",
        "evaluate_composite_on_grid",
        "evaluate_blocks_on_grid",
        "compute_component_attribution",
        "AttributionResult",
    ):
        assert hasattr(pyradtran, name), f"pyradtran missing public symbol: {name}"


def test_viz_namespace_exposes_plot_functions():
    from pyradtran.viz import (  # noqa: F401
        plot_block_profiles,
        plot_budget,
        plot_component_attribution,
        plot_composite_optics,
        plot_flux_profile,
        plot_heating_rate,
        plot_rt_overview,
        plot_spectral,
        set_theme,
    )
