"""Advanced options model (Phase 4 placeholder)."""

from pyradtran.models.base import UvspecOption


class AdvancedConfig(UvspecOption):
    """Advanced options placeholder -- implemented in Phase 4."""

    def to_uvspec_lines(self) -> list[str]:
        return []
