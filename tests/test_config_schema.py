"""T5: config schema error cases, discriminated kinds, and export_config.

Characterization tests for :mod:`pyradtran.config`:

- loader semantic errors in the ``scene`` section (``ValueError``);
- the discriminated ``placement`` union building the matching profile class;
- pydantic constraint violations rejected with ``ValidationError``;
- the ``explicit_layer`` block building a bare direct piece (no file read);
- the ``export_config`` YAML round trip.

All offline — no uvspec invocation, no libRadtran data.
"""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

_EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples" / "multicomponent_viz"
sys.path.insert(0, str(_EXAMPLES_DIR))
try:
    import canonical  # noqa: E402
finally:
    sys.path.pop(0)

from pyradtran.config import export_config, load_config  # noqa: E402
from pyradtran.models.blocks import (  # noqa: E402
    DirectLayerOpticsBlock,
    ExponentialProfile,
    MassProfile,
    PlacedBlock,
    TabulatedProfile,
)

_N_LAYER = len(canonical.ALTITUDE_GRID_KM) - 1

_OD_PLACEMENT = {
    "kind": "od_inversion",
    "tau_ref": canonical.BLOCKS[0]["tau_550"],
    "ref_nm": canonical.REF_NM,
    "scale_height_km": canonical.BLOCKS[0]["scale_height_km"],
}

_MASS_PLACEMENT = {"kind": "mass", "kg_m3_per_layer": [1e-8] * _N_LAYER}


def _mie_block(placement: dict) -> dict:
    """A one-block Mie spec derived from the canonical example's first species."""
    b = canonical.BLOCKS[0]
    return {
        "kind": "mie",
        "name": b["name"],
        "refractive_index": {
            "wavelength_um": canonical._WL_RI,
            "n_real": b["n_real"],
            "k_imag": b["k_imag"],
        },
        "size_distribution": {
            "kind": "lognormal",
            "params": {"r_g_um": b["r_g_um"], "sigma_g": b["sigma_g"]},
        },
        "particle_density_kg_m3": b["density"],
        "placement": placement,
    }


def _config_dict(tmp_path: Path, block: dict) -> dict:
    """A minimal one-block config over the canonical scene and grids."""
    return {
        "config_version": 1,
        "name": "schema_test",
        "scene": dict(canonical.SCENE_KW),
        "aerosol": {
            "wavelength_grid_um": canonical.WAVELENGTHS_UM,
            "altitude_grid_km": canonical.ALTITUDE_GRID_KM,
            "n_legendre": canonical.N_LEGENDRE,
            "output_dir": str(tmp_path),
            "blocks": [block],
        },
    }


# --- Scene-section semantic errors (loader ValueError) ---


def test_solar_source_without_sza_rejected(tmp_path):
    cfg = _config_dict(tmp_path, _mie_block(_MASS_PLACEMENT))
    cfg["scene"]["source"] = {"source": "solar"}
    with pytest.raises(ValueError, match="sza"):
        load_config(cfg)


def test_wavelength_without_min_nm_rejected(tmp_path):
    cfg = _config_dict(tmp_path, _mie_block(_MASS_PLACEMENT))
    cfg["scene"]["wavelength"] = {"max_nm": 699.0}
    with pytest.raises(ValueError, match="min_nm"):
        load_config(cfg)


def test_unknown_source_kind_rejected(tmp_path):
    cfg = _config_dict(tmp_path, _mie_block(_MASS_PLACEMENT))
    cfg["scene"]["source"] = {"source": "flashlight", "sza": 30.0}
    with pytest.raises(ValueError, match="flashlight"):
        load_config(cfg)


# --- Placement discrimination ---


_PLACEMENT_CASES = [
    pytest.param(_OD_PLACEMENT, MassProfile, id="od_inversion"),
    pytest.param(_MASS_PLACEMENT, MassProfile, id="mass"),
    pytest.param(
        {"kind": "exponential", "rho0_kg_m3": 1e-8, "scale_height_km": 1.5},
        ExponentialProfile,
        id="exponential",
    ),
    pytest.param(
        {"kind": "tabulated", "z_km": [10.0, 5.0, 0.0], "kg_m3": [1e-9, 1e-8, 5e-8]},
        TabulatedProfile,
        id="tabulated",
    ),
]


@pytest.mark.parametrize(("placement", "profile_cls"), _PLACEMENT_CASES)
def test_placement_kind_builds_matching_profile(tmp_path, placement, profile_cls):
    loaded = load_config(_config_dict(tmp_path, _mie_block(placement)))
    piece = loaded.aerosol.pieces[0]
    assert isinstance(piece, PlacedBlock)
    assert isinstance(piece.profile, profile_cls)

    kind = placement["kind"]
    if kind == "od_inversion":  # the inversion yields one mass per composite layer
        assert len(piece.profile.kg_m3_per_layer) == _N_LAYER
        assert all(m > 0.0 for m in piece.profile.kg_m3_per_layer)
    elif kind == "mass":
        assert piece.profile.kg_m3_per_layer == tuple(placement["kg_m3_per_layer"])
    elif kind == "exponential":
        assert piece.profile.rho0_kg_m3 == placement["rho0_kg_m3"]
        assert piece.profile.scale_height_km == placement["scale_height_km"]
    else:  # tabulated
        assert piece.profile.z_km == tuple(placement["z_km"])
        assert piece.profile.kg_m3 == tuple(placement["kg_m3"])


# --- Constraint violations (pydantic ValidationError) ---


@pytest.mark.parametrize("tau_ref", [0.0, -0.15], ids=["zero", "negative"])
def test_od_inversion_nonpositive_tau_rejected(tmp_path, tau_ref):
    placement = {"kind": "od_inversion", "tau_ref": tau_ref, "scale_height_km": 1.5}
    with pytest.raises(ValidationError, match="tau_ref"):
        load_config(_config_dict(tmp_path, _mie_block(placement)))


def test_single_level_altitude_grid_rejected(tmp_path):
    cfg = _config_dict(tmp_path, _mie_block(_MASS_PLACEMENT))
    cfg["aerosol"]["altitude_grid_km"] = [5.0]
    with pytest.raises(ValidationError, match="altitude_grid_km"):
        load_config(cfg)


def test_invalid_phase_function_rejected(tmp_path):
    block = _mie_block(_MASS_PLACEMENT)
    block["phase_function"] = "delta"
    with pytest.raises(ValidationError, match="phase_function"):
        load_config(_config_dict(tmp_path, block))


def test_unknown_opac_preset_rejected(tmp_path):
    block = {"kind": "opac_preset", "preset": "not_a_preset"}
    with pytest.raises(ValidationError, match="preset"):
        load_config(_config_dict(tmp_path, block))


# --- explicit_layer: bare direct piece, no file access ---


def test_explicit_layer_block_builds_bare_direct_piece(tmp_path):
    master = tmp_path / "no_such_dir" / "fake.master"
    block = {"kind": "explicit_layer", "name": "fake_explicit", "master_path": str(master)}
    loaded = load_config(_config_dict(tmp_path, block))
    piece = loaded.aerosol.pieces[0]
    assert isinstance(piece, DirectLayerOpticsBlock)
    assert not isinstance(piece, PlacedBlock)  # direct route: no placement wrapper
    assert piece.master_path == str(master)
    assert not master.exists()  # building must not read (or create) the file


# --- export_config round trip ---


def test_export_config_round_trip(tmp_path):
    cfg = _config_dict(tmp_path, _mie_block(_OD_PLACEMENT))
    exported = export_config(cfg, tmp_path / "exported.yaml")
    assert exported.is_file()

    via_file = load_config(exported)
    via_dict = load_config(cfg)
    assert via_file.scene.build_input() == via_dict.scene.build_input()
