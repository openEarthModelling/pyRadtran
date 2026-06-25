"""Tests for pyradtran.optics.opac (OPAC ingredient readers)."""

import numpy as np
import pytest

from pyradtran.optics import opac


def _need_data():
    """Skip if the OPAC ingredients are not resolvable on disk."""
    try:
        root = opac._opac_root(None)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"OPAC data not available: {e}")
    if not (root / "size_distr.cfg").is_file():
        pytest.skip(f"OPAC data not bundled at {root}")


def test_species_index_map():
    assert opac._OPAC_SPECIES_TO_INDEX["soot"] == 3
    assert opac._OPAC_SPECIES_TO_INDEX["suso"] == 10
    assert "waso" in opac._HYGROSCOPIC_SPECIES
    assert "soot" not in opac._HYGROSCOPIC_SPECIES


def test_snap_rh_hygroscopic():
    assert opac.snap_rh("ssam", 50) == 50
    assert opac.snap_rh("ssam", 60) == 50  # tie rounds down
    assert opac.snap_rh("ssam", 65) == 70
    assert opac.snap_rh("ssam", 99) == 99
    assert opac.snap_rh("ssam", 100) == 99


def test_snap_rh_nonhygroscopic():
    assert opac.snap_rh("soot", 80) == 0  # soot only has the 00 file


def test_snap_rh_unknown_species():
    with pytest.raises(ValueError):
        opac.snap_rh("nonsense", 50)


def test_read_refractive_index_soot():
    _need_data()
    ri = opac.read_opac_refractive_index("soot", 0)
    # OPAC stores the imaginary part negative; reader returns k >= 0 so that
    # RefractiveIndex.at() -> m = n + i*k (Im>0, the bhmie absorption convention).
    assert all(k >= 0 for k in ri.k_imag)
    m = ri.at(np.array([0.5]))
    # soot00 @0.5um ~ 1.75 + 0.43i (spec §5 reference magnitude)
    assert m[0].real == pytest.approx(1.75, abs=0.05)
    assert m[0].imag == pytest.approx(0.43, abs=0.1)
    assert len(ri.wavelength_um) >= 60


def test_read_size_distribution_soot():
    _need_data()
    sd, rho = opac.read_opac_size_distribution("soot", 0)
    assert sd.kind == "lognormal"
    assert sd.params["r_g_um"] == pytest.approx(0.0118, abs=0.01)
    assert rho == pytest.approx(1000.0, rel=0.01)  # 1.0 g/cm^3 -> 1000 kg/m^3


def test_read_preset_profile_continental_average():
    _need_data()
    prof = opac.read_opac_preset_profile("continental_average")
    # Header row is "# z(km)  inso  waso  soot  suso"
    assert set(prof) == {"inso", "waso", "soot", "suso"}
    for _sp, (z, mass) in prof.items():
        assert z.ndim == 1
        assert mass.shape == z.shape
        assert np.all(mass >= 0)
    assert any(np.any(mass > 0) for _, (_, mass) in prof.items())
