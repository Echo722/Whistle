"""Public message for ``POST /v1/assets/{id}/materialize`` (``materialize-v1``).

Kept out of ``contracts.models`` on purpose, the same way ``reaper_status`` is:
the 0.1.0 wire contract and its exported schema stay exactly what 0.1.0 froze, so
this is an additive 0.1.x message rather than a change to any existing field.

Why it exists: a drag out of this window must carry a real file, and the desktop
is never given a filesystem path. This message is the one place a path crosses
the wire — loopback and token-guarded, and only for the asset the caller named.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from audiosearch.contracts.models import PROTOCOL_VERSION, Seconds, Versioned

MATERIALIZE_KIND = "materialize-v1"

NonEmpty = Field(min_length=1, max_length=4096)


class _Local(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class MaterializeRequest(_Local):
    """``None`` for both bounds means "the source file itself"."""

    start_seconds: Seconds | None = None
    end_seconds: Seconds | None = None

    @model_validator(mode="after")
    def range_is_whole(self) -> Self:
        if (self.start_seconds is None) != (self.end_seconds is None):
            raise ValueError("start_seconds and end_seconds go together")
        if (
            self.start_seconds is not None
            and self.end_seconds is not None
            and self.end_seconds <= self.start_seconds
        ):
            raise ValueError("end_seconds must be greater than start_seconds")
        return self


class MaterializedFile(Versioned):
    """A local path the caller may hand to the operating system's drag layer."""

    kind: Literal["source", "rendered"]
    path: str
    media_type: str
    protocol_version: str = PROTOCOL_VERSION


__all__ = ["MATERIALIZE_KIND", "MaterializeRequest", "MaterializedFile"]
