"""pyRadtran visualization layer (pure data-in -> fig-out; lazy matplotlib)."""

from pyradtran.viz._style import get_palette, save, set_theme
from pyradtran.viz.attribution import AttributionLike, plot_component_attribution
from pyradtran.viz.composite import plot_block_profiles, plot_composite_optics
from pyradtran.viz.rt import (
    plot_budget,
    plot_flux_profile,
    plot_heating_rate,
    plot_rt_overview,
    plot_spectral,
)

__all__ = [
    "set_theme",
    "get_palette",
    "save",
    "plot_spectral",
    "plot_flux_profile",
    "plot_heating_rate",
    "plot_budget",
    "plot_rt_overview",
    "plot_composite_optics",
    "plot_block_profiles",
    "plot_component_attribution",
    "AttributionLike",
]
