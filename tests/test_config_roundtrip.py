"""T2 acceptance: YAML config → Scene → uvspec text round-trip.

A config built from the canonical example's constants must produce the exact
same uvspec input text (modulo the explicit-aerosol file path) as the scene
built via the Python API — and, thanks to the layer writer's content-hash
naming, byte-identical .master/.LAYER files.

T5 extends this with error-case, discriminated-kind, and export tests.
"""

import re
import sys
from pathlib import Path

import pytest

_EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples" / "multicomponent_viz"
sys.path.insert(0, str(_EXAMPLES_DIR))
try:
    import canonical  # noqa: E402
finally:
    sys.path.pop(0)

from pyradtran.config import load_config  # noqa: E402


def _config_dict(tmp_path: Path) -> dict:
    """Config equivalent to canonical.build_composite() + build_scene()."""
    blocks = []
    for b in canonical.BLOCKS:
        blocks.append(
            {
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
                "placement": {
                    "kind": "od_inversion",
                    "tau_ref": b["tau_550"],
                    "ref_nm": canonical.REF_NM,
                    "scale_height_km": b["scale_height_km"],
                },
            }
        )
    return {
        "config_version": 1,
        "name": "roundtrip_test",
        "scene": dict(canonical.SCENE_KW),
        "aerosol": {
            "wavelength_grid_um": canonical.WAVELENGTHS_UM,
            "altitude_grid_km": canonical.ALTITUDE_GRID_KM,
            "n_legendre": canonical.N_LEGENDRE,
            "output_dir": str(tmp_path),
            "blocks": blocks,
        },
    }


def _canonical_texts():
    aerosol = canonical.build_composite()
    scene = canonical.build_scene(aerosol)
    return scene.build_input(), aerosol


_EXPLICIT = re.compile(r"^(aerosol_file explicit )\S+$", re.MULTILINE)


def _mask_paths(text: str) -> str:
    return _EXPLICIT.sub(r"\1<PATH>", text)


def test_yaml_matches_api_uvspec_text(tmp_path):
    cfg_text = load_config(_config_dict(tmp_path)).scene.build_input()
    api_text, _ = _canonical_texts()
    assert _mask_paths(cfg_text) == _mask_paths(api_text)


def test_yaml_master_file_bytes_identical(tmp_path):
    """Same content → same content-hash filename → byte-identical files."""
    _canonical_texts()  # writes the reference explicit files to OUTPUT_DIR
    load_config(_config_dict(tmp_path)).scene.build_input()  # writes to tmp_path

    cfg_master = next(Path(tmp_path).glob("scene_*.master"))
    api_master = canonical.OUTPUT_DIR / cfg_master.name
    assert api_master.is_file(), "canonical master missing from OUTPUT_DIR"

    # The .master embeds absolute .LAYER paths — normalize the directory prefix.
    def _normalized(path: Path) -> bytes:
        return path.read_bytes().replace(str(path.parent).encode(), b"<DIR>")

    assert _normalized(cfg_master) == _normalized(api_master)

    # Layer files are pure numbers — must be byte-identical as-is.
    for cfg_layer in sorted(Path(tmp_path).glob("scene_*_layer_*.LAYER")):
        api_layer = canonical.OUTPUT_DIR / cfg_layer.name
        assert api_layer.is_file(), f"missing layer file {api_layer.name}"
        assert cfg_layer.read_bytes() == api_layer.read_bytes()


def test_load_from_yaml_file_roundtrip(tmp_path):
    """The dict and the YAML-file entry points build identical scenes."""
    import yaml

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(_config_dict(tmp_path)), encoding="utf-8")
    from_dict = load_config(_config_dict(tmp_path / "nested"))
    from_file = load_config(cfg_path)
    assert _mask_paths(from_dict.scene.build_input()) == _mask_paths(from_file.scene.build_input())


def test_generated_canonical_yaml_matches_api(monkeypatch):
    """The committed canonical.yaml (written by make_yaml.py) reproduces the
    canonical.py-built scene: identical uvspec input text modulo the
    explicit-aerosol file path (relative `output` vs absolute OUTPUT_DIR)."""
    monkeypatch.chdir(_EXAMPLES_DIR)
    loaded = load_config("canonical.yaml")
    api_text = canonical.build_scene(canonical.build_composite()).build_input()
    assert _mask_paths(loaded.scene.build_input()) == _mask_paths(api_text)


def test_unknown_block_kind_rejected(tmp_path):
    cfg = _config_dict(tmp_path)
    cfg["aerosol"]["blocks"][0]["kind"] = "miee"  # typo
    with pytest.raises(Exception, match="kind"):
        load_config(cfg)


def test_unknown_top_level_key_rejected(tmp_path):
    cfg = _config_dict(tmp_path)
    cfg["scen"] = {}  # typo
    with pytest.raises(Exception):
        load_config(cfg)


def test_bad_config_version_rejected(tmp_path):
    cfg = _config_dict(tmp_path)
    cfg["config_version"] = 2
    with pytest.raises(Exception, match="config_version"):
        load_config(cfg)
