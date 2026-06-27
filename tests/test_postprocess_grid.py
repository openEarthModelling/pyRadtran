"""Tests for composite/block grid evaluation helpers."""

from __future__ import annotations

import numpy as np

from pyradtran.core.postprocess import evaluate_blocks_on_grid, evaluate_composite_on_grid
from pyradtran.models.aerosol_composite import LayerOptics


def _layer_optics(n_wl: int, n_layer: int, n_leg: int, tau: float) -> LayerOptics:
    return LayerOptics(
        tau=np.full((n_wl, n_layer), tau),
        ssa=np.full((n_wl, n_layer), 0.5),
        g=np.full((n_wl, n_layer), 0.7),
        legendre_moments=np.zeros((n_wl, n_layer, n_leg)),
    )


class _Profile:
    def evaluate(self, centers):
        return np.full_like(np.asarray(centers, dtype=float), 1e-6)


class _FakePiece:
    def __init__(self, name: str, tau: float):
        self.name = name
        self._tau = tau
        self.profile = _Profile()

    def to_layer_optics(self, wl_um, altitude_km, n_legendre: int = 32) -> LayerOptics:
        return _layer_optics(len(wl_um), len(altitude_km) - 1, n_legendre, self._tau)


class _FakeComposite:
    def __init__(self, pieces):
        self.pieces = pieces

    def evaluate(self, wl_um=None, z_km=None, n_legendre=None) -> LayerOptics:
        n_leg = n_legendre or 32
        return _layer_optics(len(wl_um), len(z_km) - 1, n_leg, 0.9)


def test_evaluate_composite_on_grid_shape_and_coords():
    wl = np.array([0.3, 0.5, 0.8])
    z = np.array([5.0, 3.0, 1.0, 0.0])  # 3 layers
    comp = _FakeComposite([_FakePiece("a", 0.5), _FakePiece("b", 0.4)])
    ds = evaluate_composite_on_grid(comp, wl, z, n_legendre=16)
    assert set(["tau", "ssa", "g"]).issubset(ds.data_vars)
    assert ds["tau"].shape == (3, 3)  # (n_wl, n_layer)
    assert ds.sizes["layer"] == 3
    # layer altitude coord = layer centers
    np.testing.assert_allclose(ds["altitude_km"].values, [4.0, 2.0, 0.5])


def test_evaluate_blocks_on_grid_returns_one_ds_per_piece_with_rho():
    wl = np.array([0.5])
    z = np.array([2.0, 0.0])  # 1 layer
    comp = _FakeComposite([_FakePiece("soot", 0.5), _FakePiece("sulfate", 0.2)])
    out = evaluate_blocks_on_grid(comp, wl, z, n_legendre=8)
    assert set(out.keys()) == {"soot", "sulfate"}
    assert out["soot"]["tau"].shape == (1, 1)
    assert "rho_kg_m3" in out["soot"].data_vars
    np.testing.assert_allclose(out["soot"]["rho_kg_m3"].values, [1e-6])
