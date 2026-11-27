"""Decoding a fetched file into the buffer the player consumes.

The buffer is always float32 at the *output* rate, so the audio callback never
converts anything: position, duration and the samples share one time base.
"""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from audiosearch.audition.buffer import (
    TooLongError,
    UnsupportedFormatError,
    decode_for_playback,
)


def sine_wav_bytes(*, seconds: float = 1.0, rate: int = 48000, channels: int = 2) -> bytes:
    frames = int(seconds * rate)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        t = np.arange(frames) / rate
        tone = (0.5 * np.sin(2 * np.pi * 440.0 * t) * 32767).astype("<i2")
        audio.writeframes(np.repeat(tone[:, None], channels, axis=1).tobytes())
    return buffer.getvalue()


def test_decodes_a_wav_at_its_own_rate():
    buffer = decode_for_playback(
        sine_wav_bytes(seconds=0.5, rate=48000, channels=2), target_rate=48000
    )

    assert buffer.samples.dtype == np.float32
    assert buffer.samples.shape == (24000, 2)
    assert buffer.sample_rate_hz == 48000
    assert buffer.source_sample_rate_hz == 48000
    assert buffer.resampled is False
    assert buffer.duration_seconds == pytest.approx(0.5, abs=1e-3)


def test_resamples_when_the_device_rate_differs():
    buffer = decode_for_playback(
        sine_wav_bytes(seconds=0.5, rate=96000, channels=1), target_rate=48000
    )

    assert buffer.samples.shape == (24000, 1)
    assert buffer.sample_rate_hz == 48000
    assert buffer.source_sample_rate_hz == 96000
    assert buffer.resampled is True
    # Time keeps its meaning across the conversion.
    assert buffer.duration_seconds == pytest.approx(0.5, abs=1e-3)


def test_resampling_preserves_the_signal():
    source = sine_wav_bytes(seconds=0.5, rate=96000, channels=1)
    buffer = decode_for_playback(source, target_rate=48000)

    peak = float(np.max(np.abs(buffer.samples)))
    assert peak == pytest.approx(0.5, abs=0.05)


def test_refuses_files_longer_than_the_cap():
    with pytest.raises(TooLongError):
        decode_for_playback(sine_wav_bytes(seconds=3.0), target_rate=48000, max_seconds=2.0)


def test_undecodable_bytes_raise_a_typed_error():
    with pytest.raises(UnsupportedFormatError):
        decode_for_playback(b"this is not audio", target_rate=48000)
