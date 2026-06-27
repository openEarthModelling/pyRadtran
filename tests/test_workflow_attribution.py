"""Tests for component-attribution orchestration (no libRadtran)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from pyradtran.workflow.attribution import (
    AttributionResult,
    compute_component_attribution,
)


@dataclass(frozen=True)
class _Piece:
    name: str


@dataclass(frozen=True)
class _Composite:
    pieces: tuple

    def model_copy(self, *, update):
        pieces = update.get("pieces", self.pieces)
        return _Composite(pieces=tuple(pieces))


def _ds(edir: float) -> xr.Dataset:
    return xr.Dataset(
        {"edir": ("wavelength", np.array([edir]))},
        coords={"wavelength": np.array([500.0])},
    )


def test_attribution_contribution_is_full_minus_removed():
    comp = _Composite(pieces=(_Piece("soot"), _Piece("dust")))
    # full=1.0; removing soot -> 0.7; removing dust -> 0.9
    results = {
        ("soot", "dust"): _ds(1.0),
        ("dust",): _ds(0.7),  # soot removed
        ("soot",): _ds(0.9),  # dust removed
    }

    def build_scene(c):
        return c  # the "scene" is just the composite

    def execute_many(scenes):
        return [results[tuple(p.name for p in s.pieces)] for s in scenes]

    result = compute_component_attribution(build_scene, comp, execute_many)
    assert isinstance(result, AttributionResult)
    assert set(result.contributions.keys()) == {"soot", "dust"}
    # contribution(soot) = full - removed(soot) = 1.0 - 0.7 = 0.3
    np.testing.assert_allclose(result.contributions["soot"]["edir"].values, [0.3])
    np.testing.assert_allclose(result.contributions["dust"]["edir"].values, [0.1])
    np.testing.assert_allclose(result.full["edir"].values, [1.0])


def test_attribution_contributions_sum_approximates_full():
    comp = _Composite(pieces=(_Piece("a"), _Piece("b"), _Piece("c")))
    # Additive blocks: full = a+b+c, so contributions sum back to full.
    results = {
        ("a", "b", "c"): _ds(0.9),
        ("b", "c"): _ds(0.5),
        ("a", "c"): _ds(0.6),
        ("a", "b"): _ds(0.7),
    }

    def build_scene(c):
        return c

    def execute_many(scenes):
        return [results[tuple(p.name for p in s.pieces)] for s in scenes]

    result = compute_component_attribution(build_scene, comp, execute_many)
    total = sum(result.contributions[name]["edir"].values[0] for name in ("a", "b", "c"))
    np.testing.assert_allclose(total, 0.9, atol=1e-9)
