"""Asset manifest: the registry of data files bundled with pyRadtran.

The manifest maps logical (category, name) references to on-disk relative
paths under the data root (bundled or external libRadtran). All paths are
relative to the libRadtran data-files root so the same manifest works against
either the bundled subset or a full external libRadtran install.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

_MANIFEST_PATH = Path(__file__).resolve().parent / "MANIFEST.toml"


@dataclass(frozen=True)
class Asset:
    """A single bundled data asset.

    Attributes:
        category: Logical group (atmosphere_profile, solar_flux, ckd, ...).
        name: The user-facing reference value for the consuming uvspec keyword
              (e.g. "US-standard", "kurudz_1.0nm.dat", "reptran coarse").
        uvspec_keyword: The uvspec keyword that consumes this asset.
        paths: One or more paths relative to the data root.
    """

    category: str
    name: str
    uvspec_keyword: str
    paths: tuple[str, ...]


@dataclass(frozen=True)
class ValidationIssue:
    """A single data-reference problem found when validating a Scene.

    Attributes:
        severity: "warning" (default) or "error".
        category: Manifest category of the missing/referenced asset.
        name: The user-facing reference value that could not be satisfied.
        message: Human-readable explanation.
    """

    severity: str
    category: str
    name: str
    message: str


def load_manifest() -> list[Asset]:
    """Load bundled MANIFEST.toml into a list of Asset objects.

    Returns an empty list when the manifest contains no ``[[assets]]`` entries
    (Phase A: no data files committed yet).
    """
    if not _MANIFEST_PATH.is_file():
        return []
    with open(_MANIFEST_PATH, "rb") as f:
        doc = tomllib.load(f)
    assets: list[Asset] = []
    for entry in doc.get("assets", []):
        path = entry.get("path")
        files = entry.get("files")
        if path is not None:
            paths = (path,)
        elif files is not None:
            paths = tuple(files)
        else:
            raise ValueError(f"Manifest asset {entry.get('name')!r} has neither 'path' nor 'files'")
        assets.append(
            Asset(
                category=entry["category"],
                name=entry["name"],
                uvspec_keyword=entry["uvspec_keyword"],
                paths=paths,
            )
        )
    return assets


def check_consistency(assets: list[Asset], assets_dir: Path) -> list[str]:
    """Return a list of human-readable inconsistency messages.

    Consistent == every manifest path exists on disk AND every file under
    assets_dir is claimed by some manifest asset. An empty assets_dir and an
    empty manifest are consistent (Phase A state).
    """
    messages: list[str] = []

    # Every manifest path must exist on disk.
    declared: set[str] = set()
    for a in assets:
        for p in a.paths:
            declared.add(p)
            if not (assets_dir / p).is_file():
                messages.append(f"manifest path missing on disk: {p}")

    # Every on-disk file must be declared (ignore .gitkeep).
    on_disk: set[str] = set()
    if assets_dir.is_dir():
        for f in assets_dir.rglob("*"):
            if f.is_file() and f.name != ".gitkeep":
                on_disk.add(str(f.relative_to(assets_dir)))
    orphaned = on_disk - declared
    for p in sorted(orphaned):
        messages.append(f"file on disk not in manifest: {p}")

    return messages
