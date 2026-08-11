"""B2: column energy budget + conservation assertion.

Identity (per wavelength):  f_inc = f_up_toa + f_abs_surface + f_abs_atm,
where f_inc = edir@toa, f_up_toa = eup@toa, f_abs_surface = (1-albedo)*(edir+edn)@surf,
and f_abs_atm is the residual (atmospheric absorption, must be >= 0).
"""

import numpy as np
import pytest
import xarray as xr

from pyradtran.core.postprocess import (
    EnergyBudget,
    assert_energy_conservation,
    compute_energy_budget,
)


def _rt_dataset(edir_toa, eup_toa, edir_surf, edn_surf):
    """2-level (surface, toa) synthetic dataset. zout sorted ascending."""
    wl = [0.55]
    return xr.Dataset(
        {
            "edir": (("wavelength", "zout"), [[edir_surf, edir_toa]]),
            "edn": (("wavelength", "zout"), [[edn_surf, 0.0]]),
            "eup": (("wavelength", "zout"), [[0.0, eup_toa]]),
        },
        coords={"wavelength": wl, "zout": [0.0, 120.0]},
    )


def test_energy_budget_identity_holds_by_construction():
    ds = _rt_dataset(edir_toa=800.0, eup_toa=200.0, edir_surf=500.0, edn_surf=100.0)
    b = compute_energy_budget(ds, albedo=0.1)
    assert isinstance(b, EnergyBudget)
    assert np.isclose(b.f_incident[0], 800.0)
    assert np.isclose(b.f_up_toa[0], 200.0)
    assert np.isclose(b.f_abs_surface[0], 0.9 * (500.0 + 100.0))
    # residual = 800 - 200 - 540 = 60
    assert np.isclose(b.f_abs_atm[0], 60.0)


def test_assert_energy_conservation_passes_for_physical_fluxes():
    ds = _rt_dataset(edir_toa=800.0, eup_toa=200.0, edir_surf=500.0, edn_surf=100.0)
    b = assert_energy_conservation(ds, albedo=0.1)  # should not raise
    assert b.f_abs_atm[0] >= 0


def test_assert_energy_conservation_raises_when_atmosphere_creates_energy():
    # Impossible: eup@toa + (1-a)(edir+edn)@surf > f_inc  => negative atmospheric absorption.
    ds = _rt_dataset(edir_toa=800.0, eup_toa=500.0, edir_surf=500.0, edn_surf=100.0)
    with pytest.raises(AssertionError, match="[Aa]tmospheric absorption"):
        assert_energy_conservation(ds, albedo=0.1)


def test_assert_energy_conservation_raises_when_toa_upwelling_exceeds_incident():
    ds = _rt_dataset(edir_toa=800.0, eup_toa=900.0, edir_surf=0.0, edn_surf=0.0)
    with pytest.raises(AssertionError, match="[Uu]pwelling"):
        assert_energy_conservation(ds, albedo=0.1)


def test_compute_energy_budget_requires_2d_dataset():
    ds = xr.Dataset({"edir": ("wavelength", [800.0])}, coords={"wavelength": [0.55]})
    with pytest.raises(ValueError, match="zout"):
        compute_energy_budget(ds, albedo=0.1)
