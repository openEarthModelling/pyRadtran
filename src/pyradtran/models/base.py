"""Base class for all uvspec option models."""

from pydantic import BaseModel, ConfigDict


class UvspecOption(BaseModel):
    """Base class for uvspec keyword group models.

    All models are frozen (immutable) and reject extra fields to catch
    typos at the Python level rather than at uvspec runtime.

    Subclasses must implement ``to_uvspec_lines()`` to serialize
    their configuration to uvspec input file format.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
    )

    def to_uvspec_lines(self) -> list[str]:
        """Serialize this configuration to uvspec input file lines.

        Returns:
            List of strings, each being one line of a uvspec input file.
        """
        raise NotImplementedError

    def to_uvspec_items(self) -> list[tuple[int, str]]:
        """Return (phase, keyword_line) pairs for priority-sorted output.

        Subclasses may override to assign different phases to different keywords.
        Default phase is 9 (output section).
        """
        default_phase = 9
        return [(default_phase, line) for line in self.to_uvspec_lines()]
