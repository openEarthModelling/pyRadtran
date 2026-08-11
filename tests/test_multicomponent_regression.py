"""B3: canonical multicomponent scene regression — key scalars vs committed baseline.

Guards the end-to-end pipeline (LEGO composite -> explicit .LAYER -> DISORT)
against silent drift. Gated: skipped without uvspec + libRadtran data; marked slow.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.slow

_FIXTURE = Path(__file__).parent / "fixtures" / "multicomponent_baseline.json"
_TOL = 0.02  # 2% — DISORT is deterministic; absorbs minor libRadtran-version drift
_EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples" / "multicomponent_viz"


@pytest.fixture
def canonical_rt(uvspec_exe, data_path):
    """Run the canonical scene once per session and return the dataset."""
    sys.path.insert(0, str(_EXAMPLES_DIR))
    try:
        from canonical import build_composite, build_scene  # noqa: E402
    finally:
        sys.path.pop(0)

    from pyradtran.core.runner import Runner

    scene = build_scene(build_composite())
    return Runner.execute(scene, uvspec_exe=uvspec_exe, data_path=data_path)


def _baseline():
    if not _FIXTURE.is_file():
        pytest.skip("baseline fixture missing — run scripts/regen_baseline.py")
    return json.loads(_FIXTURE.read_text())


def test_scalar_baseline_matches(canonical_rt):
    rt = canonical_rt
    base = _baseline()
    i550 = int(np.argmin(np.abs(rt["wavelength"].values - 0.55)))
    surf_idx = int(np.argmin(rt["zout"].values))
    toa_idx = int(np.argmax(rt["zout"].values))
    checks = {
        "edir_surf_550nm": float(rt["edir"].isel(wavelength=i550, zout=surf_idx).values),
        "edn_surf_550nm": float(rt["edn"].isel(wavelength=i550, zout=surf_idx).values),
        "eup_surf_550nm": float(rt["eup"].isel(wavelength=i550, zout=surf_idx).values),
        "eup_toa_550nm": float(rt["eup"].isel(wavelength=i550, zout=toa_idx).values),
    }
    for key, got in checks.items():
        expected = float(base[key])
        assert np.isclose(got, expected, rtol=_TOL), (
            f"{key}: got {got:.4f}, baseline {expected:.4f} (>{_TOL:.0%} drift)"
        )


def test_atmospheric_absorption_matches(canonical_rt):
    """F_abs_atm@550 from a fresh run must match the committed baseline."""
    from pyradtran.core.postprocess import compute_energy_budget

    rt = canonical_rt
    base = _baseline()
    budget = compute_energy_budget(rt, albedo=0.1)
    i550 = int(np.argmin(np.abs(budget.wavelength - 0.55)))
    got = float(budget.f_abs_atm[i550])
    expected = float(base["F_abs_atm_550nm"])
    assert np.isclose(got, expected, rtol=_TOL), (
        f"F_abs_atm@550: got {got:.4f}, baseline {expected:.4f} (>{_TOL:.0%} drift)"
    )
