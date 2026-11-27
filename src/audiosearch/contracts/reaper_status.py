"""Public read-only REAPER status message (``reaper-status-v1``).

**Status: frozen.** These models describe what ``GET /v1/reaper/status``
returns. The live service answers this shape — ``503 bridge_unavailable`` when
no bridge is mounted — and the desktop takes instance/session/project/track IDs
from it instead of reading bridge files itself. The models live outside
``contracts.models`` on purpose: the 0.1.0 wire contract and its exported schema
stay exactly what 0.1.0 froze, so this message is an additive 0.1.x increment
rather than a change to any existing field.

Why it is needed: the desktop has to fill ``CommandContext.reaper_instance_id``,
``CommandContext.session_id`` and ``InsertTarget.project_id/track_id`` with IDs
that only the bridge knows. 0.1.0 has no message carrying them, and inventing
them client-side would address commands at a session that does not exist.

Design rules:

* every field is a *report* from the bridge, never an instruction;
* IDs are opaque strings; names and indices are display-only;
* ``kind`` distinguishes a real bridge from a simulated one, and a simulated
  bridge may never be mounted on ``data_kind="local"``;
* the payload is a plain JSON object with the usual top-level
  ``protocol_version``, so it composes with the rest of the contract family;
* it is a strict superset problem — adding it later is an additive 0.1.x change
  and no existing message changes.

The mapping from the internal ``audiosearch.integrations.reaper.BridgeStatus`` is
explicit (``from_bridge_status``) so the public contract never depends on the
bridge implementation.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints, model_validator

from audiosearch.contracts.models import PROTOCOL_VERSION

STATUS_KIND = "reaper-status-v1"

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
Count = Annotated[int, Field(ge=0, strict=True)]


class _Report(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class BridgeConnection(StrEnum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    STALE = "stale"
    CLOSED = "closed"


class CommandName(StrEnum):
    """Public command names; deliberately not the bridge's dotted internals."""

    PLAY = "play"
    STOP = "stop"
    SEEK = "seek"
    INSERT = "insert"


class BridgeTrack(_Report):
    track_id: NonEmpty
    name: str = ""
    # Display only: never used to address a track.
    index: Count = 0


class BridgeProject(_Report):
    project_id: NonEmpty
    # Bumped when this project's track list changes; not needed to address it.
    project_version: Count = 0
    name: str = ""
    is_current: bool = False
    # Where an insert would land without the operator naming a target. Reported
    # for the current project only; ``selected_track_id`` is None when nothing is
    # selected, which the desktop sends as "create a track".
    edit_cursor_seconds: Annotated[float, Field(ge=0)] | None = None
    selected_track_id: str | None = None
    tracks: list[BridgeTrack] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_tracks(self) -> Self:
        ids = [track.track_id for track in self.tracks]
        if len(set(ids)) != len(ids):
            raise ValueError("track IDs must be unique within a project")
        return self


class BridgePlayback(_Report):
    generation: Count | None = None
    state: Literal["idle", "playing", "stopped"] = "idle"
    request_id: UUID | None = None
    source_position_seconds: Annotated[float, Field(ge=0)] | None = None


class ReaperStatus(_Report):
    """What the bridge currently reports; shaped like every other message."""

    protocol_version: Literal["0.1.0"] = PROTOCOL_VERSION
    status_kind: Literal["reaper-status-v1"] = STATUS_KIND
    # "fake" means every command result is simulated; never REAPER evidence.
    kind: Literal["fake", "reaper"]
    state: BridgeConnection
    reaper_instance_id: NonEmpty | None = None
    session_id: UUID | None = None
    connected_at: AwareDatetime | None = None
    last_heartbeat_at: AwareDatetime | None = None
    heartbeat_timeout_seconds: Annotated[float, Field(gt=0)]
    # Only commands the bridge has actually verified.
    commands: list[CommandName] = Field(default_factory=list)
    channel_selection: bool = False
    output_to_track: bool = False
    reaper_version: str | None = None
    projects: list[BridgeProject] = Field(default_factory=list)
    discovery_version: Count = 0
    playback: BridgePlayback = Field(default_factory=BridgePlayback)
    reason: str | None = None
    observed_at: AwareDatetime

    @model_validator(mode="after")
    def identity_matches_state(self) -> Self:
        # A "connected" report without an instance and session would let a
        # caller build a CommandContext that can never be accepted.
        if self.state is BridgeConnection.CONNECTED:
            if self.reaper_instance_id is None or self.session_id is None:
                raise ValueError("connected status needs reaper_instance_id and session_id")
        return self

    @property
    def usable(self) -> bool:
        """Commands may only be built from a live session."""
        return self.state is BridgeConnection.CONNECTED and self.session_id is not None

    def project(self, project_id: str) -> BridgeProject | None:
        return next((item for item in self.projects if item.project_id == project_id), None)


__all__ = [
    "STATUS_KIND",
    "BridgeConnection",
    "BridgePlayback",
    "BridgeProject",
    "BridgeTrack",
    "CommandName",
    "ReaperStatus",
]
