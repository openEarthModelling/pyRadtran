"""Tests for OpacPreset / OpacCustom as PlacedBlock factories (Phase 2)."""

import numpy as np
import pytest

from pyradtran.models.aerosol import OpacPreset, OpacPresetName
from pyradtran.models.aerosol_composite import CompositeAerosol
from pyradtran.models.blocks import PlacedBlock, TabulatedProfile


def _need_data():
    from pyradtran.optics import opac

    try:
        root = opac._opac_root(None)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"OPAC data unavailable: {e}")
    if not (root / "size_distr.cfg").is_file():
        pytest.skip("OPAC data not bundled")


def test_to_placed_blocks_returns_real_phase_pieces():
    _need_data()
    blocks = OpacPreset(name=OpacPresetName.CONTINENTAL_AVERAGE, rh_pct=50.0).to_placed_blocks()
    assert len(blocks) >= 1
    for b in blocks:
        assert isinstance(b, PlacedBlock)
        assert isinstance(b.profile, TabulatedProfile)
        # factory uses the real Mie phase function
        assert getattr(b.block, "phase_function", None) == "mie"
        assert b.block.name.startswith("OPAC:")


def test_to_composite_round_trips_through_evaluate():
    _need_data()
    preset = OpacPreset(name=OpacPresetName.CONTINENTAL_AVERAGE, rh_pct=50.0, n_legendre=8)
    comp = preset.to_composite(wavelength_grid_um=[0.55])
    assert isinstance(comp, CompositeAerosol)
    layer = comp.evaluate(np.array([0.55]))
    assert layer.tau.shape[0] == 1
    assert np.all(layer.tau >= 0)
    assert np.all((layer.ssa >= 0) & (layer.ssa <= 1))


def test_opacpreset_is_not_an_aerosolmodel():
    # The factory must NOT carry to_uvspec_lines; it produces Pieces instead.
    preset = OpacPreset(name=OpacPresetName.CONTINENTAL_AVERAGE)
    assert not hasattr(preset, "to_uvspec_lines")


def test_species_filter_restricts_blocks():
    _need_data()
    blocks = OpacPreset(
        name=OpacPresetName.CONTINENTAL_AVERAGE, rh_pct=50.0, species_names=["soot"]
    ).to_placed_blocks()
    assert all(b.block.name == "OPAC:soot" for b in blocks)
