"""Regenerate tests/fixtures/multicomponent_baseline.json.

Usage: MPLBACKEND=Agg python scripts/regen_baseline.py
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    target = ROOT / "tests" / "fixtures" / "multicomponent_baseline.json"
    demo = ROOT / "examples" / "multicomponent_viz" / "run_demo.py"
    env = {**os.environ, "MPLBACKEND": "Agg"}
    subprocess.run([sys.executable, str(demo), "--dump-baseline", str(target)], env=env, check=True)
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
