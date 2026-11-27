"""The transport state machine, driven by a fake sink.

The fake stands in for the output device: it holds the callback and hands it
frame counts, which is exactly what PortAudio does. Everything asserted here is
arithmetic on delivered frames, so these tests need no sound card.
"""

from __future__ import annotations

import numpy as np
import pytest

from audiosearch.audition.buffer import PlaybackBuffer
from audiosearch.audition.player import PlayerCore

RATE = 4800  # 0.1 s == 480 frames; small numbers keep the assertions readable


def buffer_of(*, seconds: float = 1.0, channels: int = 2, rate: int = RATE) -> PlaybackBuffer:
    frames = int(seconds * rate)
    return PlaybackBuffer(
        samples=np.zeros((frames, channels), dtype=np.float32),
        sample_rate_hz=rate,
        duration_seconds=seconds,
        source_sample_rate_hz=rate,
        resampled=False,
    )


class FakeSink:
    def __init__(self) -> None:
        self._pull = None
        self.starts = 0
        self.stops = 0
        self.delivered = 0
        self.channel_count = 0

    def start(self, pull, channel_count: int) -> None:
        self._pull = pull
        self.starts += 1
        self.channel_count = channel_count

    def stop(self) -> None:
        self.stops += 1

    def advance(self, frames: int) -> np.ndarray:
        """Deliver ``frames`` the way the device would, returning what came out."""
        block = self._pull(frames)
        self.delivered += len(block)
        return block


def test_play_from_a_position_starts_there():
    sink = FakeSink()
    core = PlayerCore(sink)
    core.load(buffer_of(seconds=1.0))

    core.play(0.5)

    assert core.state == "playing"
    assert core.position_seconds == pytest.approx(0.5)


def test_pull_advances_by_delivered_frames():
    sink = FakeSink()
    core = PlayerCore(sink)
    core.load(buffer_of(seconds=1.0))
    core.play(0.5)

    block = sink.advance(480)

    assert block.shape == (480, 2)
    assert core.position_seconds == pytest.approx(0.6)


def test_pause_keeps_the_position_and_resume_continues():
    sink = FakeSink()
    core = PlayerCore(sink)
    core.load(buffer_of(seconds=1.0))
    core.play(0.0)
    sink.advance(960)
    stops_before = sink.stops

    core.pause()

    assert core.state == "paused"
    assert core.position_seconds == pytest.approx(0.2)
    assert sink.stops == stops_before + 1

    core.play()

    assert core.state == "playing"
    assert core.position_seconds == pytest.approx(0.2)
    assert sink.starts == 2


def test_reaching_the_end_finishes_without_stopping_sink_inside_callback():
    sink = FakeSink()
    core = PlayerCore(sink)
    core.load(buffer_of(seconds=0.2))
    stops_after_load = sink.stops
    core.play(0.0)

    sink.advance(480)
    tail = sink.advance(480)

    assert len(tail) == 480
    assert core.state == "finished"
    assert core.position_seconds == pytest.approx(0.2)
    assert core.wait_finished(timeout=0.1) is True
    # _pull runs inside PortAudio's callback thread, so it must never call sink.stop().
    assert sink.stops == stops_after_load


def test_playing_again_after_the_end_restarts_from_zero():
    sink = FakeSink()
    core = PlayerCore(sink)
    core.load(buffer_of(seconds=0.1))
    core.play(0.0)
    sink.advance(480)

    core.play()

    assert core.state == "playing"
    assert core.position_seconds == pytest.approx(0.0)


def test_stop_rewinds_to_the_beginning():
    sink = FakeSink()
    core = PlayerCore(sink)
    core.load(buffer_of(seconds=1.0))
    core.play(0.0)
    sink.advance(960)

    core.stop()

    assert core.state == "paused"
    assert core.position_seconds == pytest.approx(0.0)


def test_loading_a_new_buffer_resets_the_cursor():
    sink = FakeSink()
    core = PlayerCore(sink)
    core.load(buffer_of(seconds=1.0))
    core.play(0.5)
    sink.advance(480)

    core.load(buffer_of(seconds=2.0))

    assert core.state == "paused"
    assert core.position_seconds == pytest.approx(0.0)
    assert core.duration_seconds == pytest.approx(2.0)


def test_pull_without_a_buffer_is_silence_not_a_crash():
    core = PlayerCore(FakeSink())
    block = core._pull(128)
    assert block.shape[0] == 128
    assert not block.any()
