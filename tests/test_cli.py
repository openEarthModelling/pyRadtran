"""T4: CLI — validate / export-config / run error paths (no uvspec needed).

A real uvspec `run` is exercised by the canonical YAML example (T6) and the
uvspec-gated suites; here `main` is called directly and via `python -m`.
"""

import subprocess
import sys
from pathlib import Path

import yaml

from pyradtran.config import load_config
from pyradtran.config.cli import main

#: Minimal valid config: scene only, no aerosol/analysis sections.
MINIMAL = {
    "config_version": 1,
    "name": "cli_test",
    "scene": {
        "atmosphere": {"profile": "us", "altitude": 0.0},
        "source": {"sza": 30.0},
        "wavelength": {"min_nm": 400, "max_nm": 700},
    },
}


def _write(path: Path, cfg: dict) -> Path:
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_validate_ok(tmp_path, capsys):
    cfg = _write(tmp_path / "config.yaml", MINIMAL)
    assert main(["validate", str(cfg)]) == 0
    out = capsys.readouterr().out
    assert "ok:" in out
    assert "cli_test" in out and "us" in out


def test_validate_bad_config_version(tmp_path, capsys):
    bad = dict(MINIMAL, config_version=2)
    path = _write(tmp_path / "bad.yaml", bad)
    assert main(["validate", str(path)]) == 1
    assert capsys.readouterr().err


def test_export_config_roundtrip(tmp_path, capsys):
    cfg = _write(tmp_path / "config.yaml", MINIMAL)
    out_path = tmp_path / "exported.yaml"
    assert main(["export-config", str(cfg), "-o", str(out_path)]) == 0
    assert str(out_path) in capsys.readouterr().out

    assert load_config(out_path).scene.build_input() == load_config(cfg).scene.build_input()


def test_run_missing_config(tmp_path, capsys):
    missing = tmp_path / "does_not_exist.yaml"
    assert main(["run", str(missing)]) == 1
    assert str(missing) in capsys.readouterr().err


def test_python_dash_m_entry_point(tmp_path):
    cfg = _write(tmp_path / "config.yaml", MINIMAL)
    proc = subprocess.run(
        [sys.executable, "-m", "pyradtran", "validate", str(cfg)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok:" in proc.stdout
