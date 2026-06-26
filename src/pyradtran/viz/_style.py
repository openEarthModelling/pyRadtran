"""Publication-grade theme, colorblind-safe palette, and save helper.

matplotlib is imported lazily: :func:`require_mpl` is the single chokepoint so
``import pyradtran`` never requires matplotlib.
"""

from __future__ import annotations

from pathlib import Path

# Okabe-Ito colorblind-safe palette.
_OKABE_ITO = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#F0E442",  # yellow
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
    "#000000",  # black
]


def require_mpl():
    """Import and return matplotlib, or raise an actionable ImportError."""
    try:
        import matplotlib
    except ImportError as e:  # pragma: no cover - exercised only without matplotlib
        raise ImportError(
            "pyradtran.viz requires matplotlib; install it with `pip install pyradtran[plot]`"
        ) from e
    return matplotlib


def set_theme(style: str = "publication") -> None:
    """Apply rcParams for the requested style.

    ``publication`` (default): higher DPI, larger fonts, tighter layout.
    ``diagnostic``: lighter settings for quick iteration.
    """
    mpl = require_mpl()
    if style == "publication":
        mpl.rcParams.update(
            {
                "figure.dpi": 150,
                "savefig.dpi": 300,
                "font.size": 11,
                "axes.labelsize": 11,
                "axes.titlesize": 12,
                "legend.fontsize": 9,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "axes.grid": True,
                "grid.alpha": 0.3,
                "figure.autolayout": True,
            }
        )
    elif style == "diagnostic":
        mpl.rcParams.update(
            {
                "figure.dpi": 100,
                "savefig.dpi": 100,
                "font.size": 9,
                "axes.grid": True,
                "grid.alpha": 0.3,
                "figure.autolayout": True,
            }
        )
    else:
        raise ValueError(f"Unknown style: {style!r} (use 'publication' or 'diagnostic')")


def get_palette(n: int) -> list[str]:
    """Return ``n`` colorblind-safe colors (cycling the Okabe-Ito set)."""
    if n <= 0:
        return []
    return [_OKABE_ITO[i % len(_OKABE_ITO)] for i in range(n)]


def save(fig, path, *, formats=("pdf", "png")) -> None:
    """Save ``fig`` to ``path`` with each extension in ``formats`` appended."""
    base = Path(path)
    for ext in formats:
        fig.savefig(base.with_suffix(f".{ext}"))
