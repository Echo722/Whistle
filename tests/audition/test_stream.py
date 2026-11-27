"""Behavioral tests for StreamingResampler and StreamingPlayer."""

from __future__ import annotations

import threading
import wave
from pathlib import Path

import numpy as np
import pytest

from audiosearch.audioio.resample import resample
from audiosearch.audition.stream import (
    STREAM_KAISER_BETA,
    STREAM_ZERO_CROSSINGS,
    FractionalResampler,
    StreamingPlayer,
    StreamingResampler,
    StreamingWSOLA,
)


class _FakeSink:
    def __init__(self) -> None:
        self.pull = None
        self.started = threading.Event()
        self.stopped = 0
        self.closed = False
        self.device_error: str | None = None

    def available(self) -> bool:
        return True

    def start(self, pull, channel_count: int = 2) -> None:
        self.pull = pull
        self.started.set()

    def stop(self) -> None:
        self.stopped += 1
        self.pull = None

    def close(self) -> None:
        self.closed = True
        self.pull = None


@pytest.mark.parametrize("src_rate", [96000, 44100, 48000])
def test_streaming_resampler_matches_whole_file_output(src_rate: int):
    rng = np.random.default_rng(42)
    full = (rng.standard_normal((src_rate // 2, 2)) * 0.25).astype(np.float32)

    expected = resample(
        full,
        src_rate,
        48000,
        zero_crossings=STREAM_ZERO_CROSSINGS,
        beta=STREAM_KAISER_BETA,
    )

    resampler = StreamingResampler(src_rate, 48000, 2)
    chunks: list[np.ndarray] = []
    step = 1373  # deliberately prime/unaligned block size
    for start in range(0, len(full), step):
        part = full[start : start + step]
        eof = start + step >= len(full)
        out = resampler.process(part, eof=eof)
        if len(out):
            chunks.append(out)
    actual = np.concatenate(chunks, axis=0)

    assert actual.shape == expected.shape
    assert np.allclose(actual, expected, atol=1e-5)


def test_streaming_player_handles_long_files_pause_resume_and_seek(tmp_path: Path):
    # > 120 seconds (130 s @ 8 kHz mono -> tiny file, exceeds old 120s cap)
    wav_path = tmp_path / "long_130s.wav"
    rate = 8000
    total_frames = rate * 130
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(np.full(total_frames, 4000, dtype="<i2").tobytes())

    sink = _FakeSink()
    player = StreamingPlayer(sink, 48000)
    ready = threading.Event()
    errors: list[str] = []

    try:
        player.load_and_play(
            wav_path,
            start_seconds=60.0,
            on_ready=lambda _g, _s: ready.set(),
            on_error=lambda _g, msg: errors.append(msg),
        )
        assert ready.wait(timeout=3.0)
        assert not errors
        assert player.state == "playing"
        assert player.position_seconds == pytest.approx(60.0, abs=0.01)

        block1 = sink.pull(1024)
        assert block1.shape[0] == 1024
        pos_after_pull = player.position_seconds
        assert pos_after_pull > 60.0

        # Pause preserves position; resume continues without rewinding
        player.pause()
        assert player.state == "paused"
        assert player.position_seconds == pytest.approx(pos_after_pull, abs=1e-6)
        assert player.resume() is True
        assert player.state == "playing"
        block2 = sink.pull(512)
        assert block2.shape[0] == 512
        assert player.position_seconds > pos_after_pull
    finally:
        player.close()
        assert sink.closed is True


def test_streaming_player_resume_after_stop_replays_cleanly(tmp_path: Path):
    sr = 48000
    t = np.arange(sr // 2, dtype=np.float64) / sr
    sine = (0.4 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    wav_path = tmp_path / "stop_resume.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((sine * 32767.0).astype("<i2").tobytes())

    sink = _FakeSink()
    player = StreamingPlayer(sink, sr)
    ready_gens: list[int] = []
    ready = threading.Event()

    def _on_ready(g: int, _s: float) -> None:
        ready_gens.append(g)
        ready.set()

    try:
        gen1 = player.load_and_play(wav_path, 0.0, on_ready=_on_ready)
        assert ready.wait(timeout=3.0)
        assert player.state == "playing"
        assert player.has_buffered_audio is True

        _ = sink.pull(2048)
        player.pause()
        assert player.state == "paused"
        assert player.has_buffered_audio is True
        assert player.resume() is True
        assert player.state == "playing"

        # Stop clears buffered audio (like Esc), subsequent resume() must cleanly replay from start
        player.stop()
        assert player.state == "paused"
        assert player.has_buffered_audio is False
        ready.clear()
        assert player.resume() is True
        assert ready.wait(timeout=3.0)
        assert player.state == "playing"
        assert len(ready_gens) == 2 and ready_gens[1] > gen1
        block = sink.pull(1024)
        assert block.shape[0] == 1024
        assert float(np.max(np.abs(block))) > 0.05
    finally:
        player.close()


def test_fractional_resampler_continuity_and_player_loop_volume_pitch_rate(tmp_path: Path):
    # Verify FractionalResampler produces smooth sine output at 2.0x speed across chunk boundaries.
    rate = 48000
    t = np.arange(rate, dtype=np.float64) / rate
    sine = np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)[:, np.newaxis]
    fr = FractionalResampler(rate, rate, 1, speed=2.0)
    parts = [fr.process(sine[i : i + 1000], eof=(i + 1000 >= len(sine))) for i in range(0, len(sine), 1000)]
    out = np.concatenate(parts, axis=0)
    # At 2.0x speed, output duration is ~0.5s (24000 frames) and max adjacent jump is bounded.
    assert 23990 <= len(out) <= 24010
    assert np.max(np.abs(np.diff(out[:, 0]))) < 0.25

    wav_path = tmp_path / "tone_1s.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes((sine[:, 0] * 16000).astype("<i2").tobytes())

    sink = _FakeSink()
    player = StreamingPlayer(sink, rate)
    ready = threading.Event()
    try:
        player.set_loop(True)
        player.set_selection((0.2, 0.5))
        player.set_volume_gain(0.5)
        player.set_pitch_rate(12.0, 1.0)  # +12 st => 2.0x speed
        player.load_and_play(wav_path, 0.2, end_seconds=0.5, on_ready=lambda _g, _s: ready.set())
        assert ready.wait(timeout=3.0)

        # Pull 0.4s wall-clock (> 0.3s selection at 2x speed => wraps cleanly without finishing)
        pulled = [sink.pull(4800) for _ in range(4)]
        assert all(b.shape[0] > 0 for b in pulled)
        assert np.max(np.abs(pulled[0])) <= 0.35  # 0.5 volume gain applied
        assert player.state == "playing"
        assert 0.2 <= player.position_seconds <= 0.55
    finally:
        player.close()


def test_streaming_wsola_independent_pitch_rate_and_short_sfx():
    sr = 48000
    t = np.arange(sr * 2, dtype=np.float64) / sr
    sine = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    stereo = np.stack([sine, sine], axis=1)

    pitch = 5.0
    rate = 1.25
    pitch_ratio = 2.0 ** (pitch / 12.0)
    resampler = FractionalResampler(sr, sr, 2, speed=1.0)
    resampler.set_speed(pitch_ratio)
    wsola = StreamingWSOLA(sr, 2)

    chunks = []
    for i in range(0, len(stereo), 1920):
        eof = i + 1920 >= len(stereo)
        mid = resampler.process(stereo[i : i + 1920], eof=eof)
        out = wsola.process(mid, rate / pitch_ratio)
        if len(out):
            chunks.append(out)
    tail = wsola.flush(rate / pitch_ratio)
    if len(tail):
        chunks.append(tail)

    full = np.concatenate(chunks, axis=0)
    assert abs(len(full) - int(len(stereo) / rate)) < 2500
    seg = full[2048:-2048, 0]
    peak_hz = float(np.fft.rfftfreq(len(seg), 1.0 / sr)[np.argmax(np.abs(np.fft.rfft(seg)))])
    assert abs(peak_hz - 440.0 * pitch_ratio) < 5.0
    assert float(np.sqrt(np.mean(seg**2))) > 0.30

    # Short 0.08s SFX flushes cleanly without hanging or returning empty
    short = stereo[: int(sr * 0.08)]
    res_s = FractionalResampler(sr, sr, 2, speed=pitch_ratio)
    wsola_s = StreamingWSOLA(sr, 2)
    y1 = wsola_s.process(res_s.process(short, eof=True), rate / pitch_ratio)
    y2 = wsola_s.flush(rate / pitch_ratio)
    out_s = np.concatenate([y1, y2], axis=0)
    assert len(out_s) > 1000


def test_player_live_pitch_rate_preserve_pitch_without_restart(tmp_path: Path):
    sr = 48000
    t = np.arange(sr * 3, dtype=np.float64) / sr
    sine = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    wav_path = tmp_path / "live_tone.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((sine * 32767.0).astype("<i2").tobytes())

    sink = _FakeSink()
    player = StreamingPlayer(sink, sr)
    ready_gens: list[int] = []
    ready = threading.Event()
    try:
        gen = player.load_and_play(
            wav_path,
            0.0,
            on_ready=lambda g, _s: (ready_gens.append(g), ready.set()),
        )
        assert ready.wait(timeout=3.0)
        stops_before = sink.stopped

        # Pull initial frames, then enable Preserve Pitch + change Pitch/Rate live
        b0 = sink.pull(2048)
        assert b0.shape[0] == 2048
        player.set_preserve_pitch(True)
        player.set_pitch_rate(7.0, 1.5)

        # No decoder restart (same generation, no sink.stop() call)
        assert sink.stopped == stops_before
        assert len(ready_gens) == 1 and ready_gens[0] == gen

        pulled = []
        for _ in range(12):
            import time
            time.sleep(0.035)
            pulled.append(sink.pull(4096))
        merged = np.concatenate(pulled, axis=0)
        assert float(np.sqrt(np.mean(merged[-8192:, 0] ** 2))) > 0.20
        seg = merged[-8192:, 0]
        peak_hz = float(np.fft.rfftfreq(len(seg), 1.0 / sr)[np.argmax(np.abs(np.fft.rfft(seg)))])
        assert abs(peak_hz - 440.0 * (2.0 ** (7.0 / 12.0))) < 15.0
        assert player.position_seconds > 0.5
    finally:
        player.close()


def test_output_device_enumeration_resolve_and_sink_reconfigure(monkeypatch):
    import sys
    import types

    from audiosearch.audition import sink as sink_mod

    fake_hostapis = [
        {"name": "MME", "default_output_device": 0},
        {"name": "Windows WASAPI", "default_output_device": 2},
    ]
    fake_devices = [
        {"name": "Speakers (MME)", "hostapi": 0, "max_output_channels": 2, "default_samplerate": 44100.0},
        {"name": "Mic Input Only", "hostapi": 1, "max_output_channels": 0, "default_samplerate": 48000.0},
        {"name": "Speakers (Realtek)", "hostapi": 1, "max_output_channels": 2, "default_samplerate": 48000.0},
        {"name": "Focusrite Scarlett 4i4 USB", "hostapi": 1, "max_output_channels": 4, "default_samplerate": 96000.0},
        {"name": "Focusrite Scarlett 4i4 USB", "hostapi": 1, "max_output_channels": 2, "default_samplerate": 48000.0},
    ]
    fake_sd = types.SimpleNamespace(
        query_hostapis=lambda: fake_hostapis,
        query_devices=lambda *a, **kw: fake_devices if not a and not kw else fake_devices[a[0]],
        default=types.SimpleNamespace(hostapi=1),
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    devices = sink_mod.list_output_devices()
    assert [d.name for d in devices] == [
        "Speakers (Realtek)",
        "Focusrite Scarlett 4i4 USB",
    ]
    assert devices[0].is_default is True
    assert devices[1].max_output_channels == 4 and devices[1].default_samplerate == 96000

    idx_def, name_def, rate_def, matched_def = sink_mod.resolve_output_device("")
    assert (idx_def, name_def, rate_def, matched_def) == (2, "Speakers (Realtek)", 48000, True)

    idx_ext, name_ext, rate_ext, matched_ext = sink_mod.resolve_output_device(
        "focusrite scarlett 4i4 usb"
    )
    assert (idx_ext, name_ext, rate_ext, matched_ext) == (
        3,
        "Focusrite Scarlett 4i4 USB",
        96000,
        True,
    )

    idx_fb, name_fb, rate_fb, matched_fb = sink_mod.resolve_output_device("Missing Device")
    assert (idx_fb, name_fb, rate_fb, matched_fb) == (2, "Speakers (Realtek)", 48000, False)

    closed_flags: list[bool] = []

    class _DummyStream:
        active = True

        def stop(self) -> None:
            closed_flags.append(True)

        def close(self) -> None:
            closed_flags.append(True)

    sd_sink = sink_mod.SoundDeviceSink(48000)
    sd_sink._stream = _DummyStream()
    sd_sink.device_error = "stale"
    sd_sink.reconfigure(96000, device=3)
    assert sd_sink.sample_rate_hz == 96000
    assert sd_sink._device == 3
    assert sd_sink._stream is None
    assert sd_sink.device_error is None
    assert len(closed_flags) == 2

    player = StreamingPlayer(_FakeSink(), 48000)
    try:
        player.set_target_rate(96000)
        assert player._target_rate == 96000
    finally:
        player.close()
