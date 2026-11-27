"""Internal bridge status snapshot (``reaper-adapter-v0``).

These models are **not** part of the public 0.1.0 wire contract. They describe
what an in-process bridge adapter knows about the REAPER side so the service can
fill ``CommandContext`` / ``InsertTarget`` and report honest capabilities. They
are JSON-serialisable so a later public status message can reuse them verbatim
(see ``docs/contracts/reaper-adapter-v0.md``, proposals for Lead).

Identity rules:

* ``reaper_instance_id`` names one REAPER process configuration (stable across
  restarts of the same test instance); ``session_id`` is new for every bridge
  connection. A disconnect or restart always yields a new ``session_id``.
* ``project_id`` is an opaque ID issued by the bridge and only valid inside the
  ``session_id`` that reported it. When a project tab loads different content the
  bridge issues a *new* ``project_id``; an old ID is then simply unknown.
* ``track_id`` is opaque (the real bridge uses REAPER's track GUID). Names and
  indices are display-only and never identify a target.
* REAPER pointers never leave the REAPER process.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints

from audiosearch.contracts.models import AssetRef, SourceRange

ADAPTER_PROTOCOL = "reaper-adapter-v0"

_Id = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
_Count = Annotated[int, Field(ge=0, strict=True)]

PlaybackOptionName = Literal["gain_db", "play_rate", "pitch_semitones", "preserve_pitch", "loop"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ConnectionState(StrEnum):
    DISCONNECTED = "disconnected"  # never connected, REAPER absent, or link lost
    CONNECTING = "connecting"  # connect() in progress
    CONNECTED = "connected"  # session live and heartbeat fresh
    STALE = "stale"  # session known but heartbeat older than heartbeat_timeout_seconds
    CLOSED = "closed"  # close() was called; this adapter object is finished


class BridgeCommand(StrEnum):
    PLAYBACK_START = "playback.start"
    PLAYBACK_STOP = "playback.stop"
    PLAYBACK_SEEK = "playback.seek"
    INSERT = "reaper.insert"


class BridgeCapabilities(_Model):
    """What the connected bridge has actually verified it can do."""

    commands: list[BridgeCommand] = Field(default_factory=list)
    # PlaybackOptions fields that may take a NON-default value in playback.start.
    # Anything not listed must stay at its PlaybackOptions() default or the
    # command is rejected with invalid_request.
    playback_options: list[PlaybackOptionName] = Field(default_factory=list)
    # Same rule for InsertRequest.apply_playback_options (null is always allowed).
    insert_playback_options: list[PlaybackOptionName] = Field(default_factory=list)
    # False: SourceRange.channels must be null (all channels).
    channel_selection: bool = False
    # False: PlaybackStartRequest.output_project_id/output_track_id must be null.
    output_to_track: bool = False
    reaper_version: str | None = None
    # Extension name -> detected version; None means checked and missing.
    extensions: dict[str, str | None] = Field(default_factory=dict)


class TrackInfo(_Model):
    track_id: _Id
    name: str = ""  # display only
    index: _Count = 0  # display only, 0-based position at discovery time


class ProjectInfo(_Model):
    project_id: _Id
    # Bumped whenever this project's track list changes (add/remove/reorder).
    project_version: _Count = 0
    name: str = ""  # display only (never a target identity)
    is_current: bool = False  # REAPER's active project tab at discovery time
    # Where a "drop it here" insert would land without the operator naming a
    # target. Reported by the bridge for the current project only.
    edit_cursor_seconds: float | None = None
    selected_track_id: str | None = None
    tracks: list[TrackInfo] = Field(default_factory=list)

    def track(self, track_id: str) -> TrackInfo | None:
        return next((track for track in self.tracks if track.track_id == track_id), None)


class PlaybackSnapshot(_Model):
    # Highest generation accepted in this session (None: none yet). Resets with
    # every new session.
    generation: _Count | None = None
    state: Literal["idle", "playing", "stopped"] = "idle"
    request_id: UUID | None = None  # request that started this generation
    asset: AssetRef | None = None
    source_range: SourceRange | None = None
    # Last known/commanded position in ORIGINAL file seconds. Not a live playhead.
    source_position_seconds: Annotated[float, Field(ge=0)] | None = None


class BridgeStatus(_Model):
    adapter_protocol: Literal["reaper-adapter-v0"] = ADAPTER_PROTOCOL
    # "fake" results are simulated and must never be reported as REAPER evidence.
    kind: Literal["fake", "reaper"]
    state: ConnectionState
    reaper_instance_id: _Id | None = None
    session_id: UUID | None = None
    connected_at: AwareDatetime | None = None
    last_heartbeat_at: AwareDatetime | None = None
    heartbeat_timeout_seconds: Annotated[float, Field(gt=0)]
    capabilities: BridgeCapabilities | None = None
    projects: list[ProjectInfo] = Field(default_factory=list)
    # Bumped on any project/track discovery change in this adapter.
    discovery_version: _Count = 0
    playback: PlaybackSnapshot = Field(default_factory=PlaybackSnapshot)
    # Human-readable (Chinese) reason for non-connected states.
    reason: str | None = None
    observed_at: AwareDatetime

    @property
    def is_ready(self) -> bool:
        return self.state is ConnectionState.CONNECTED and self.session_id is not None

    def project(self, project_id: str) -> ProjectInfo | None:
        return next((p for p in self.projects if p.project_id == project_id), None)

    def supports(self, command: BridgeCommand) -> bool:
        return self.capabilities is not None and command in self.capabilities.commands


__all__ = [
    "ADAPTER_PROTOCOL",
    "BridgeCapabilities",
    "BridgeCommand",
    "BridgeStatus",
    "ConnectionState",
    "PlaybackOptionName",
    "PlaybackSnapshot",
    "ProjectInfo",
    "TrackInfo",
]
