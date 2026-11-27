"""The bridge adapter interface every playback backend implements.

Contract summary (details: ``docs/contracts/reaper-adapter-v0.md``):

* Construction and import have no side effects: no connection, no file channel,
  no REAPER command. Only ``connect()`` touches the backend.
* ``connect()`` / ``refresh()`` return a ``BridgeStatus`` and do not raise for
  environment problems (REAPER absent, extension missing): the status says so.
* Command methods take the existing public request models and always return a
  ``CommandReceipt`` with a terminal status: ``completed``, ``rejected`` or
  ``uncertain``. v0 adapters never return ``accepted``.
* Every method is thread-safe and blocks at most ``timeout`` seconds, further
  capped by ``context.expires_at``.
* ``receipt(request_id)`` reports a known result without executing anything.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable
from uuid import UUID

from audiosearch.contracts.models import (
    CommandReceipt,
    InsertRequest,
    PlaybackSeekRequest,
    PlaybackStartRequest,
    PlaybackStopRequest,
)

from .status import BridgeStatus

DEFAULT_CONNECT_TIMEOUT = 5.0
DEFAULT_PREVIEW_TIMEOUT = 2.0
DEFAULT_INSERT_TIMEOUT = 5.0
DEFAULT_HEARTBEAT_TIMEOUT = 3.0
# A command whose issued_at lies further in the future than this is refused.
MAX_CLOCK_SKEW_SECONDS = 5.0


@runtime_checkable
class PlaybackBridge(Protocol):
    @property
    def kind(self) -> Literal["fake", "reaper"]: ...

    def connect(self, *, timeout: float = DEFAULT_CONNECT_TIMEOUT) -> BridgeStatus: ...

    def close(self) -> None: ...

    def status(self) -> BridgeStatus: ...

    def refresh(self, *, timeout: float = DEFAULT_CONNECT_TIMEOUT) -> BridgeStatus: ...

    def start_playback(
        self, request: PlaybackStartRequest, *, timeout: float = DEFAULT_PREVIEW_TIMEOUT
    ) -> CommandReceipt: ...

    def stop_playback(
        self, request: PlaybackStopRequest, *, timeout: float = DEFAULT_PREVIEW_TIMEOUT
    ) -> CommandReceipt: ...

    def seek_playback(
        self, request: PlaybackSeekRequest, *, timeout: float = DEFAULT_PREVIEW_TIMEOUT
    ) -> CommandReceipt: ...

    def insert(
        self, request: InsertRequest, *, timeout: float = DEFAULT_INSERT_TIMEOUT
    ) -> CommandReceipt: ...

    def receipt(self, request_id: UUID) -> CommandReceipt | None: ...


__all__ = [
    "DEFAULT_CONNECT_TIMEOUT",
    "DEFAULT_HEARTBEAT_TIMEOUT",
    "DEFAULT_INSERT_TIMEOUT",
    "DEFAULT_PREVIEW_TIMEOUT",
    "MAX_CLOCK_SKEW_SECONDS",
    "PlaybackBridge",
]
