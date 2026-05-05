"""Subprocess execution engine for uvspec.

Runner is decoupled from Scene. Scene holds configuration;
Runner executes it by building input, running uvspec, and parsing output.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import xarray as xr

from pyradtran.core.output_parser import parse_output

if TYPE_CHECKING:
    from pyradtran.scene import Scene


@dataclass
class RunnerConfig:
    """Configuration for uvspec execution.

    Attributes:
        uvspec_exe: Path to uvspec binary. Auto-detected via PATH if None.
        data_path: Path to libRadtran data directory. Uses env vars if None.
        max_workers: Maximum parallel workers for execute_many().
        keep_temp: Keep temporary files after execution (for debugging).
        timeout: Maximum uvspec execution time in seconds. None = no timeout.
    """

    uvspec_exe: str | None = None
    data_path: str | None = None
    max_workers: int = 1
    keep_temp: bool = False
    timeout: int | None = None


class Runner:
    """Execute uvspec simulations.

    Usage::

        result = Runner.execute(scene, data_path="/path/to/data")
        results = Runner.execute_many(scenes, max_workers=4)

    Global defaults may be set once via :py:meth:`configure` so that
    ``uvspec_exe`` and ``data_path`` do not need to be repeated on
    every call::

        Runner.configure(uvspec_exe="/opt/libRadtran/bin/uvspec",
                         data_path="/opt/libRadtran/data")
        result = Runner.execute(scene)
    """

    _config: RunnerConfig = RunnerConfig()

    @classmethod
    def configure(
        cls, config: RunnerConfig | None = None, **kwargs
    ) -> RunnerConfig:
        """Set global default configuration for all Runner executions.

        Args:
            config: A complete ``RunnerConfig`` instance, or ``None``.
            **kwargs: Individual fields (e.g. ``uvspec_exe=..., data_path=...``).

        Returns:
            The newly set configuration.
        """
        if config is not None:
            cls._config = config
        else:
            cls._config = RunnerConfig(**kwargs)
        return cls._config

    @staticmethod
    def _find_uvspec(exe: str | None = None) -> str:
        """Locate uvspec binary."""
        if exe is not None:
            if not os.path.isfile(exe):
                raise FileNotFoundError(f"uvspec not found at: {exe}")
            return os.path.abspath(exe)
        exe = shutil.which("uvspec")
        if exe is None:
            raise FileNotFoundError(
                "uvspec not found on PATH. Install libRadtran or set uvspec_exe explicitly."
            )
        return exe

    @staticmethod
    def _find_data_path(path: str | None = None) -> str:
        """Locate libRadtran data directory."""
        if path is not None:
            if not os.path.isdir(path):
                raise FileNotFoundError(f"Data directory not found: {path}")
            return os.path.abspath(path)
        for env_var in ("LIBRADTRAN_DATA_FILES", "LIBRADTRANDIR"):
            val = os.environ.get(env_var)
            if val:
                if env_var == "LIBRADTRANDIR":
                    val = os.path.join(val, "data")
                if os.path.isdir(val):
                    return val
        raise FileNotFoundError(
            "libRadtran data directory not found. "
            "Set LIBRADTRAN_DATA_FILES or pass data_path explicitly."
        )

    @staticmethod
    def execute(
        scene: Scene,
        uvspec_exe: str | None = None,
        data_path: str | None = None,
        keep_temp: bool | None = None,
        timeout: int | None = None,
        config: RunnerConfig | None = None,
    ) -> xr.Dataset:
        """Execute a single uvspec simulation.

        Args:
            scene: Configured Scene object.
            uvspec_exe: Path to uvspec binary. Auto-detected if None.
            data_path: Path to libRadtran data directory. Auto-detected if None.
            keep_temp: Keep temporary files after execution.
            timeout: Maximum execution time in seconds.
            config: Optional ``RunnerConfig`` overriding global defaults for
                this call only.

        Returns:
            xarray.Dataset with simulation results.
        """
        cfg = config or Runner._config
        uvspec = Runner._find_uvspec(
            uvspec_exe if uvspec_exe is not None else cfg.uvspec_exe
        )
        resolved_data_path = Runner._find_data_path(
            data_path if data_path is not None else cfg.data_path
        )
        resolved_keep_temp = (
            keep_temp if keep_temp is not None else cfg.keep_temp
        )
        resolved_timeout = (
            timeout if timeout is not None else cfg.timeout
        )

        input_text = scene.build_input(data_files_path=resolved_data_path)

        with TemporaryDirectory(prefix="pyradtran_") as tmp_dir:
            inp_file = Path(tmp_dir) / "uvspec.inp"

            output_format = scene.output.format if scene.output else "netcdf"
            ext = "nc" if output_format == "netcdf" else "out"
            out_file = Path(tmp_dir) / f"uvspec.{ext}"

            inp_file.write_text(input_text)

            env = os.environ.copy()
            env["LIBRADTRAN_DATA_FILES"] = resolved_data_path

            with open(inp_file) as stdin_f, open(out_file, "w") as stdout_f:
                proc = subprocess.run(
                    [uvspec],
                    stdin=stdin_f,
                    stdout=stdout_f,
                    stderr=subprocess.PIPE,
                    env=env,
                    timeout=resolved_timeout,
                    text=True,
                )

            if proc.returncode != 0:
                raise RuntimeError(
                    f"uvspec failed (exit code {proc.returncode}):\n"
                    f"stderr: {proc.stderr}"
                )

            n_zout = len(scene.output.zout) if scene.output and scene.output.zout else 1
            result = parse_output(out_file, format=output_format, n_zout=n_zout)

        result.attrs["input_config"] = input_text.strip()
        result.attrs["uvspec_exe"] = uvspec

        return result

    @staticmethod
    def execute_many(
        scenes: list[Scene],
        uvspec_exe: str | None = None,
        data_path: str | None = None,
        max_workers: int | None = None,
        keep_temp: bool | None = None,
        timeout: int | None = None,
        config: RunnerConfig | None = None,
    ) -> list[xr.Dataset]:
        """Execute multiple uvspec simulations in parallel.

        Args:
            scenes: List of configured Scene objects.
            uvspec_exe: Path to uvspec binary. Auto-detected if None.
            data_path: Path to libRadtran data directory. Auto-detected if None.
            max_workers: Maximum parallel workers.
            keep_temp: Keep temporary files after execution.
            timeout: Maximum execution time in seconds per scene.
            config: Optional ``RunnerConfig`` overriding global defaults.

        Returns:
            List of xarray.Dataset results, one per scene.
        """
        cfg = config or Runner._config
        resolved_max_workers = (
            max_workers if max_workers is not None else cfg.max_workers
        )
        resolved_uvspec_exe = (
            uvspec_exe if uvspec_exe is not None else cfg.uvspec_exe
        )
        resolved_data_path = (
            data_path if data_path is not None else cfg.data_path
        )
        resolved_keep_temp = (
            keep_temp if keep_temp is not None else cfg.keep_temp
        )
        resolved_timeout = (
            timeout if timeout is not None else cfg.timeout
        )

        results = []
        with ProcessPoolExecutor(max_workers=resolved_max_workers) as executor:
            futures = {
                executor.submit(
                    Runner.execute,
                    scene,
                    uvspec_exe=resolved_uvspec_exe,
                    data_path=resolved_data_path,
                    keep_temp=resolved_keep_temp,
                    timeout=resolved_timeout,
                ): i
                for i, scene in enumerate(scenes)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results.append((idx, future.result()))
                except Exception as e:
                    results.append((idx, e))

        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]
