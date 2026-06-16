"""DataResolver: locate libRadtran data files with tiered fallback.

Resolution priority for the data root (``data_files_path``):
    1. explicit ``data_root`` argument (user is most explicit)
    2. ``LIBRADTRAN_DATA_FILES`` environment variable
    3. ``LIBRADTRANDIR`` environment variable (its ``data/`` subdir)
    4. bundled subset at ``pyradtran/data/assets``

``bundled_only=True`` overrides 1-3 and always uses the bundled root
(implemented in a later task; here it is accepted but not yet enforced).
"""

from __future__ import annotations

import os
from pathlib import Path

from pyradtran.data.manifest import Asset, load_manifest

_BUNDLED_ROOT = Path(__file__).resolve().parent / "assets"


class DataResolver:
    """Resolve logical data references to absolute paths under the data root."""

    def __init__(
        self, *, data_root: str | os.PathLike | None = None, bundled_only: bool = False
    ) -> None:
        self._explicit_root: Path | None = (
            Path(data_root).resolve() if data_root is not None else None
        )
        self._bundled_only = bundled_only
        self._manifest: list[Asset] = load_manifest()
        self._cached_root: Path | None = None

    # -- data root -------------------------------------------------------

    @property
    def data_root(self) -> Path:
        """The effective data_files_path used for this resolver."""
        if self._cached_root is None:
            self._cached_root = self._resolve_root()
        return self._cached_root

    def _resolve_root(self) -> Path:
        if self._explicit_root is not None:
            if not self._explicit_root.is_dir():
                raise FileNotFoundError(f"Data directory not found: {self._explicit_root}")
            return self._explicit_root

        val = os.environ.get("LIBRADTRAN_DATA_FILES")
        if val and Path(val).is_dir():
            return Path(val).resolve()

        val = os.environ.get("LIBRADTRANDIR")
        if val:
            candidate = Path(val) / "data"
            if candidate.is_dir():
                return candidate.resolve()

        return _BUNDLED_ROOT

    # -- asset lookup ----------------------------------------------------

    def _find_asset(self, category: str, name: str) -> Asset | None:
        for a in self._manifest:
            if a.category == category and a.name == name:
                return a
        return None

    def resolve(self, category: str, name: str) -> Path:
        """Return the absolute path of an asset's first file.

        Raises FileNotFoundError if the (category, name) is unknown to the
        bundled manifest or its file is missing on disk.
        """
        asset = self._find_asset(category, name)
        if asset is None:
            raise FileNotFoundError(
                f"No bundled asset for {category}/{name!r}. "
                f"Set LIBRADTRAN_DATA_FILES or install libRadtran."
            )
        path = self.data_root / asset.paths[0]
        if not path.exists():
            raise FileNotFoundError(
                f"{category}/{name!r} not found at {path} (data root: {self.data_root})"
            )
        return path

    def is_available(self, category: str, name: str) -> bool:
        """True if the asset is present on disk.

        Unknown (category, name) -- not in the bundled manifest -- are treated
        as permissively available (assumed resolvable via an external data root).
        """
        asset = self._find_asset(category, name)
        if asset is None:
            return True
        return all((self.data_root / p).exists() for p in asset.paths)

    def list_bundled(self, category: str | None = None) -> list[Asset]:
        """List bundled assets, optionally filtered by category."""
        if category is None:
            return list(self._manifest)
        return [a for a in self._manifest if a.category == category]
