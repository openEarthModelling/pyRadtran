"""Command-line interface: ``pyradtran run | validate | export-config``.

argparse stdlib only. :func:`main` returns a process exit code so it can be
called directly from tests; config load/validation errors are reported as a
single concise stderr message, while anything else propagates with its
traceback (far more useful during development).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from pydantic import ValidationError

from pyradtran.config.loader import export_config, load_config
from pyradtran.config.orchestrator import RunResult, run_config
from pyradtran.config.schema import PyRadtranConfig

#: Wavelength (nm) at which budget/DRF scalars are summarized.
_SUMMARY_NM = 550.0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. ``argv`` defaults to ``sys.argv[1:]``."""
    args = _build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValidationError, ValueError, TypeError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyradtran",
        description="Run libRadtran experiments described by YAML configs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="execute a config: main run + analysis intents")
    run.add_argument("config", help="path to the YAML config")
    run.add_argument("--uvspec", help="path to the uvspec executable")
    run.add_argument("--data-path", help="libRadtran data directory")
    run.add_argument("--plot-dir", help="directory for requested plots")
    run.set_defaults(func=_cmd_run)

    validate = sub.add_parser("validate", help="validate a config (no uvspec invocation)")
    validate.add_argument("config", help="path to the YAML config")
    validate.set_defaults(func=_cmd_validate)

    export = sub.add_parser("export-config", help="write a config as canonical YAML")
    export.add_argument("config", help="path to the YAML config")
    export.add_argument("--output", "-o", required=True, help="output YAML path")
    export.set_defaults(func=_cmd_export)

    return parser


# --- Commands ---


def _cmd_run(args: argparse.Namespace) -> int:
    loaded = load_config(args.config)
    result = run_config(
        loaded,
        uvspec_exe=args.uvspec,
        data_path=args.data_path,
        plot_dir=args.plot_dir,
    )
    _print_summary(loaded.config, result)
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config).config
    profile = cfg.scene.atmosphere.get("profile", "?")
    n_blocks = len(cfg.aerosol.blocks) if cfg.aerosol else 0
    print(f"ok: {cfg.name} — scene {profile}, {n_blocks} aerosol blocks, intents: {_intents(cfg)}")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    cfg = load_config(args.config).config
    print(export_config(cfg, args.output))
    return 0


# --- Summary rendering ---


def _print_summary(cfg: PyRadtranConfig, result: RunResult) -> None:
    """Human-readable one-block-per-topic digest of a finished run."""
    print(f"config: {cfg.name}")
    sizes = " ".join(f"{dim}={n}" for dim, n in result.rt.sizes.items())
    print(f"rt: {sizes} — data_vars: {', '.join(result.rt.data_vars)}")

    if result.budget is not None:
        b, i = result.budget, _nearest_index(result.budget.wavelength)
        print(
            f"energy budget @{b.wavelength[i]:.0f} nm: "
            f"F_inc={b.f_incident[i]:.4g} up_TOA={b.f_up_toa[i]:.4g} "
            f"abs_surf={b.f_abs_surface[i]:.4g} abs_atm={b.f_abs_atm[i]:.4g} W/m²"
        )
    if result.drf is not None:
        d, i = result.drf, _nearest_index(result.drf.wavelength_nm)
        print(
            f"drf @{d.wavelength_nm[i]:.0f} nm: "
            f"TOA={d.toa[i]:.4g} surface={d.surface[i]:.4g} "
            f"atmosphere={d.atmosphere[i]:.4g} W/m²"
        )
    if "heating_rate" in result.rt:
        print("heating_rate: merged")
    if result.attribution is not None:
        print(f"attribution: {', '.join(result.attribution.contributions)}")
    for fig in result.figures:
        print(f"figure: {fig}")
    if result.netcdf_path is not None:
        print(f"netcdf: {result.netcdf_path}")


def _nearest_index(wavelengths: np.ndarray) -> int:
    """Index of the grid point closest to the summary wavelength."""
    return int(np.argmin(np.abs(np.asarray(wavelengths, dtype=float) - _SUMMARY_NM)))


def _intents(cfg: PyRadtranConfig) -> list[str]:
    """Names of the enabled analysis intents, in schema declaration order."""
    analysis = cfg.analysis
    if analysis is None:
        return []
    intents: list[str] = []
    if analysis.energy_conservation is not None:
        intents.append("energy_conservation")
    intents.extend(flag for flag in ("drf", "heating", "attribution") if getattr(analysis, flag))
    if analysis.plots:
        intents.append("plots")
    if analysis.save_netcdf:
        intents.append("save_netcdf")
    return intents
