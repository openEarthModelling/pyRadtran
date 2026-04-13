"""Monte Carlo configuration model (Phase 3 placeholder)."""

from pyradtran.models.base import UvspecOption


class McConfig(UvspecOption):
    """Monte Carlo configuration placeholder -- implemented in Phase 3."""

    def to_uvspec_lines(self) -> list[str]:
        return []
