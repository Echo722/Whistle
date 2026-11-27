"""The output-device wrapper: device probing and the callback's arithmetic.

The callback is tested directly with a fake ``outdata`` buffer, so the padding
and channel rules hold on a machine with no sound card. Only the last test needs
real hardware and skips without it.
"""

from __future__ import annotations

import numpy as np
import pytest

from audiosearch.audition.sink import SoundDeviceSink


def test_sink_reports_unavailable_when_portaudio_sees_no_output(monkeypatch):
    import sounddevice

    monkeypatch.setattr(sounddevice, "query_devices", lambda *a, **k: [])
    sink = SoundDeviceSink(48000, 2)

    assert sink.available() is False
    assert "没有可用的音频输出设备" in (sink.device_error or "")


def test_sink_reports_available_when_a_device_has_outputs(monkeypatch):
    import sounddevice

    monkeypatch.setattr(
        sounddevice, "query_devices", lambda *a, **k: [{"max_output_channels": 2}]
    )
    sink = SoundDeviceSink(48000, 2)

    assert sink.available() is True
    assert sink.device_error is None


def test_callback_pads_a_short_block_with_silence():
    sink = SoundDeviceSink(48000, 2)
    pull = lambda frames: np.full((frames // 2, 2), 0.5, dtype=np.float32)  # noqa: E731
    outdata = np.ones((64, 2), dtype=np.float32)

    sink._callback(pull)(outdata, 64, None, None)

    assert np.allclose(outdata[:32], 0.5)
    assert not outdata[32:].any()


def test_callback_drops_surplus_channels_instead_of_crashing():
    sink = SoundDeviceSink(48000, 2)
    pull = lambda frames: np.full((frames, 4), 0.25, dtype=np.float32)  # noqa: E731
    outdata = np.zeros((32, 2), dtype=np.float32)

    sink._callback(pull)(outdata, 32, None, None)

    assert np.allclose(outdata, 0.25)


def test_callback_upmixes_mono_to_stereo():
    sink = SoundDeviceSink(48000, 2)
    pull = lambda frames: np.full((frames, 1), 0.3, dtype=np.float32)  # noqa: E731
    outdata = np.zeros((32, 2), dtype=np.float32)

    sink._callback(pull)(outdata, 32, None, None)

    assert np.allclose(outdata[:, 0], 0.3)
    assert np.allclose(outdata[:, 1], 0.3)


def test_stop_clears_pull_and_outputs_silence_without_closing_stream():
    class FakeStream:
        active = True
        stopped = False
        closed = False

        def stop(self):
            self.stopped = True

        def close(self):
            self.closed = True

    sink = SoundDeviceSink(48000, 2)
    fake = FakeStream()
    sink._stream = fake
    sink._pull = lambda frames: np.full((frames, 2), 0.5, dtype=np.float32)

    sink.stop()
    outdata = np.ones((32, 2), dtype=np.float32)
    sink._stream_callback(outdata, 32, None, None)

    assert not outdata.any()
    assert fake.stopped is False
    assert fake.closed is False

    sink.close()
    assert fake.stopped is True
    assert fake.closed is True
    assert sink._stream is None


def _has_output_device() -> bool:
    try:
        import sounddevice as sd

        return any(int(d.get("max_output_channels", 0)) > 0 for d in sd.query_devices())
    except Exception:
        return False


@pytest.mark.skipif(not _has_output_device(), reason="没有音频输出设备")
def test_a_real_stream_pulls_frames():
    from audiosearch.audition.buffer import PlaybackBuffer
    from audiosearch.audition.player import PlayerCore

    rate = 48000
    buffer = PlaybackBuffer(
        samples=np.zeros((rate // 10, 2), dtype=np.float32),  # 100 ms
        sample_rate_hz=rate,
        duration_seconds=0.1,
        source_sample_rate_hz=rate,
        resampled=False,
    )
    sink = SoundDeviceSink(rate, 2)
    try:
        core = PlayerCore(sink)
        core.load(buffer)
        core.play(0.0)

        assert core.wait_finished(timeout=3.0) is True
        assert core.position_seconds == pytest.approx(0.1, abs=0.01)
        assert sink.device_error is None
    finally:
        sink.close()
