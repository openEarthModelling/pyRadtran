"""Tests for viz theme/palette/save (headless)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyradtran.viz._style import get_palette, require_mpl, save, set_theme


def test_require_mpl_returns_matplotlib_module():
    assert require_mpl() is matplotlib


def test_set_theme_changes_a_distinguishable_param():
    set_theme("publication")
    pub_dpi = plt.rcParams["figure.dpi"]
    set_theme("diagnostic")
    diag_dpi = plt.rcParams["figure.dpi"]
    assert pub_dpi != diag_dpi


def test_palette_returns_requested_count_and_consistent():
    a = get_palette(3)
    b = get_palette(5)
    assert len(a) == 3 and len(b) == 5
    # Same palette family: first 3 of the 5-color request equal the 3-color request.
    assert a == b[:3]


def test_save_writes_requested_formats(tmp_path: Path):
    set_theme("publication")
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    out = tmp_path / "fig"
    save(fig, out, formats=("png",))
    assert (tmp_path / "fig.png").exists()
    plt.close(fig)
