"""Component-attribution workflow.

Drives N+1 RT runs (one full composite + one leave-one-out per piece) via an
injected ``execute_many`` and returns per-block contributions
``full - leave_one_out``. The plot layer consumes :class:`AttributionResult`
by duck-typing; this module owns the data contract and the orchestration.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import xarray as xr


@dataclass(frozen=True)
class AttributionResult:
    """Full-field run plus per-block contribution datasets."""

    full: xr.Dataset
    contributions: dict[str, xr.Dataset]


def compute_component_attribution(
    build_scene: Callable,
    composite,
    execute_many: Callable[[list], list[xr.Dataset]],
) -> AttributionResult:
    """Compute per-block RT contributions by leave-one-out subtraction.

    Args:
        build_scene: Callable mapping a (possibly leave-one-out) composite to a
            runnable :class:`~pyradtran.scene.Scene`.
        composite: A composite with ``.pieces`` and ``.model_copy(update=...)``.
        execute_many: Callable running a list of scenes in parallel and returning
            the parsed datasets in the same order (e.g.
            ``lambda scenes: Runner.execute_many(scenes, uvspec_exe=..., data_path=...)``).

    Returns:
        :class:`AttributionResult` whose ``contributions[piece.name]`` is
        ``full - leave_piece_out``.
    """
    pieces = list(composite.pieces)
    names = [getattr(p, "name", f"piece_{i}") for i, p in enumerate(pieces)]

    scenes = [build_scene(composite)]
    for i in range(len(pieces)):
        remaining = [p for j, p in enumerate(pieces) if j != i]
        sub = composite.model_copy(update={"pieces": remaining})
        scenes.append(build_scene(sub))

    datasets = execute_many(scenes)
    full = datasets[0]

    contributions: dict[str, xr.Dataset] = {}
    for i, name in enumerate(names):
        removed = datasets[i + 1]
        contributions[name] = full - removed
    return AttributionResult(full=full, contributions=contributions)
