"""Verify removed aerosol entry points are gone (LEGO-only public surface)."""

import pytest


def test_external_file_removed():
    with pytest.raises(ImportError):
        from pyradtran.models.aerosol import ExternalFile  # noqa: F401


def test_external_aerosol_alias_removed():
    with pytest.raises(ImportError):
        from pyradtran.models.aerosol import ExternalAerosol  # noqa: F401


def test_run_with_aerosol_removed():
    with pytest.raises(ImportError):
        from pyradtran.convenience import run_with_aerosol  # noqa: F401


def test_public_exports_removed():
    import pyradtran

    for name in ("ExternalAerosol", "ExternalFile", "run_with_aerosol"):
        assert name not in pyradtran.__all__, f"{name} still exported"


def test_set_tau_at_wvl_and_king_byrne_removed():
    from pyradtran.models.aerosol import AerosolModel

    assert "set_tau_at_wvl" not in AerosolModel.model_fields
    assert "king_byrne" not in AerosolModel.model_fields
