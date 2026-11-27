"""Playback state plus the cursor that makes the position real.

``PlayerCore`` owns the answer to "where are we" and knows nothing about audio
devices: the sink asks it for frames, hands them to the device, and the cursor
advances only for frames that were actually delivered. That is the whole reason
the playhead stops being an estimate.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Literal, Protocol

import numpy as np

from .buffer import PlaybackBuffer

PlayerState = Literal["empty", "paused", "playing", "finished"]


class AudioSink(Protocol):
    """Where decoded frames go. One implementation per output device kind."""

    def start(self, pull: Callable[[int], np.ndarray], channel_count: int) -> None: ...

    def stop(self) -> None: ...


class PlayerCore:
    def __init__(self, sink: AudioSink) -> None:
        self._sink = sink
        self._lock = threading.Lock()
        self._buffer: PlaybackBuffer | None = None
        self._cursor = 0  # frames, at the buffer's own rate
        self._state: PlayerState = "empty"
        self._finished = threading.Event()

    # ------------------------------------------------------------------ state
    @property
    def state(self) -> PlayerState:
        with self._lock:
            return self._state

    @property
    def position_seconds(self) -> float:
        with self._lock:
            if self._buffer is None:
                return 0.0
            return self._cursor / self._buffer.sample_rate_hz

    @property
    def duration_seconds(self) -> float:
        with self._lock:
            return 0.0 if self._buffer is None else self._buffer.duration_seconds

    def wait_finished(self, timeout: float | None = None) -> bool:
        """Block until the current buffer plays to its end (or stop/pause)."""
        return self._finished.wait(timeout)

    # ----------------------------------------------------------------- loading
    def load(self, buffer: PlaybackBuffer) -> None:
        """Swap the source. Always starts paused at the beginning."""
        self._sink.stop()
        with self._lock:
            self._buffer = buffer
            self._cursor = 0
            self._state = "paused"
        self._finished.clear()

    # ---------------------------------------------------------------- transport
    def play(self, start_seconds: float | None = None) -> None:
        """Play from ``start_seconds``; ``None`` continues from the cursor."""
        with self._lock:
            buffer = self._buffer
            if buffer is None:
                return
            if start_seconds is not None:
                self._cursor = self._clamp(int(round(start_seconds * buffer.sample_rate_hz)))
            elif self._state == "finished":
                self._cursor = 0
            if self._state == "playing":
                return
            self._state = "playing"
        self._finished.clear()
        self._sink.start(self._pull, buffer.channel_count)

    def pause(self) -> None:
        with self._lock:
            if self._state != "playing":
                return
            self._state = "paused"
        self._finished.set()
        self._sink.stop()

    def stop(self) -> None:
        """Stop and rewind: the next play starts at the beginning."""
        with self._lock:
            was_playing = self._state == "playing"
            self._cursor = 0
            if self._buffer is not None:
                self._state = "paused"
        self._finished.set()
        if was_playing:
            self._sink.stop()

    # -------------------------------------------------------------------- pull
    def _pull(self, frames: int) -> np.ndarray:
        """Hand ``frames`` to the sink and advance by what was delivered.

        Called from the device's callback thread: the lock is held only for the
        cursor arithmetic, never for the copy.
        """
        with self._lock:
            buffer = self._buffer
            if buffer is None or self._state != "playing":
                return np.zeros((max(frames, 0), 1), dtype=np.float32)
            start = self._cursor
            stop = min(start + max(frames, 0), buffer.frames)
            block = buffer.samples[start:stop]
            self._cursor = stop
            reached_end = stop >= buffer.frames
            if reached_end:
                self._state = "finished"
        if reached_end:
            # Never call self._sink.stop() from the audio callback thread: PortAudio
            # forbids stopping/closing the stream inside its own callback and doing so
            # corrupts the heap. Signal finished so the sink raises CallbackStop.
            self._finished.set()
        return block

    def _clamp(self, frame: int) -> int:
        assert self._buffer is not None
        return max(0, min(frame, self._buffer.frames))


__all__ = ["AudioSink", "PlayerCore", "PlayerState"]
