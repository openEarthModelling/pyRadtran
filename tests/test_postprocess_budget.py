"""Tests for T/R/A budget derivation in pyradtran.core.postprocess."""
from __future__ import annotations

import numpy as np
import xarray as xr

from pyradtran.core.postprocess import BudgetResult, add_budget_vars, compute_budget


def _flux_ds() -> xr.Dataset:
    wl = np.array([300.0, 500.0, 800.0])
    zout = np.array([0.0, 120.0])  # surface (min) , TOA (max)
    # (n_wl, n_zout); surface attenuated, TOA = incident direct = 1.0
    edir = np.array([[0.5, 1.0], [0.6, 1.0], [0.7, 1.0]])
    edn = np.array([[0.1, 0.0], [0.1, 0.0], [0.1, 0.0]])
    eup = np.array([[0.05, 0.2], [0.06, 0.2], [0.07, 0.2]])
    return xr.Dataset(
        {
            "edir": (("wavelength", "zout"), edir),
            "edn": (("wavelength", "zout"), edn),
            "eup": (("wavelength", "zout"), eup),
        },
        coords={"wavelength": wl, "zout": zout},
    )


def test_add_budget_vars_adds_three_vars_and_closes():
    ds = add_budget_vars(_flux_ds())
    for name in ("transmittance", "reflectance", "absorptance"):
        assert name in ds.data_vars
    T, R, A = ds["transmittance"].values, ds["reflectance"].values, ds["absorptance"].values
    np.testing.assert_allclose(T + R + A, np.ones(3), atol=1e-9)
    # T = (edir+edn)@surface / edir@TOA = (0.6, 0.7, 0.8)
    np.testing.assert_allclose(T, [0.6, 0.7, 0.8])
    # R = eup@TOA / edir@TOA = 0.2
    np.testing.assert_allclose(R, [0.2, 0.2, 0.2])


def test_budget_quantities_in_unit_interval():
    ds = add_budget_vars(_flux_ds())
    for name in ("transmittance", "reflectance", "absorptance"):
        v = ds[name].values
        assert np.all(v >= 0.0) and np.all(v <= 1.0)


def test_compute_budget_returns_dataclass():
    result = compute_budget(_flux_ds())
    assert isinstance(result, BudgetResult)
    np.testing.assert_allclose(result.transmittance, [0.6, 0.7, 0.8])
    assert result.wavelength.tolist() == [300.0, 500.0, 800.0]


def test_add_budget_vars_does_not_mutate_input():
    ds = _flux_ds()
    before = set(ds.data_vars)
    add_budget_vars(ds)
    assert set(ds.data_vars) == before
