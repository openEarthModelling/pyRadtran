#!/usr/bin/env python3
"""Check consistency between MANIFEST.toml and the bundled assets/ directory.

Exit code 0 = consistent, 1 = mismatch. Intended for CI and manual use after
curating data files (Phase B).

Usage:
    python scripts/build_data_manifest.py            # check
    python scripts/build_data_manifest.py --assets   # list on-disk files + sizes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "src" / "pyradtran" / "data" / "assets"

# Make src importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pyradtran.data.manifest import check_consistency, load_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets", action="store_true", help="list on-disk asset files with sizes, then exit"
    )
    args = parser.parse_args()

    if args.assets:
        if not _ASSETS_DIR.is_dir():
            print("no assets/ directory")
            return 0
        for f in sorted(_ASSETS_DIR.rglob("*")):
            if f.is_file() and f.name != ".gitkeep":
                print(f"{f.stat().st_size:>10}  {f.relative_to(_ASSETS_DIR)}")
        return 0

    messages = check_consistency(load_manifest(), _ASSETS_DIR)
    if messages:
        print("Manifest / assets inconsistency:", file=sys.stderr)
        for m in messages:
            print(f"  - {m}", file=sys.stderr)
        return 1
    print("OK: manifest and assets/ are consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
