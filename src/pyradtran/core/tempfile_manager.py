"""Temporary file lifecycle management for uvspec data files."""

from __future__ import annotations

import contextlib
import tempfile
import uuid
from pathlib import Path

import numpy as np


class TempFileManager:
    """Manage temporary data files for uvspec execution.

    Args:
        temp_dir: Directory for temporary files. Defaults to system temp.
        keep_temp: If True, don't delete files on cleanup (for debugging).

    Usage::

        with TempFileManager() as mgr:
            path = mgr.write_array("extinction.dat", extinction_data)
        # files cleaned up on exit
    """

    def __init__(self, temp_dir: str | None = None, keep_temp: bool = False):
        self._keep_temp = keep_temp
        self._base_dir = Path(temp_dir) if temp_dir else Path(tempfile.getdtemp()) / "pyradtran"
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._files: list[Path] = []

    @property
    def files(self) -> list[str]:
        """List of tracked temporary file paths."""
        return [str(p) for p in self._files]

    def write_array(self, name: str, data: np.ndarray) -> str:
        """Write numpy array to a temporary file, return the path."""
        unique_name = f"{uuid.uuid4().hex[:8]}_{name}"
        path = self._base_dir / unique_name
        np.savetxt(path, data)
        self._files.append(path)
        return str(path)

    def write_text(self, name: str, content: str) -> str:
        """Write text content to a temporary file, return the path."""
        unique_name = f"{uuid.uuid4().hex[:8]}_{name}"
        path = self._base_dir / unique_name
        path.write_text(content)
        self._files.append(path)
        return str(path)

    def cleanup(self) -> None:
        """Delete all tracked temporary files."""
        if self._keep_temp:
            return
        for path in self._files:
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)
        self._files.clear()

    def __enter__(self) -> TempFileManager:
        return self

    def __exit__(self, *args) -> None:
        self.cleanup()
