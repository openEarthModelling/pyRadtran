"""Cloud configuration model (Phase 2 placeholder)."""

from pyradtran.models.base import UvspecOption


class CloudConfig(UvspecOption):
    """Cloud configuration placeholder -- implemented in Phase 2."""

    def to_uvspec_lines(self) -> list[str]:
        return []
