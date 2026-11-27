"""Tests for pure-Python WAV fallback when libsndfile rejects valid commercial WAVs."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audiosearch.ai.indexer import decode_audio_windows_mono
from audiosearch.audioio.decode import open_audio_file
from audiosearch.audition.buffer import decode_file_for_playback
from audiosearch.audition.render import render_range
from audiosearch.library.scanner import LibraryScanner
from audiosearch.library.store import LibraryStore


def _make_bad_peak_wav(path: Path, *, sample_rate: int = 48000, frames: int = 12000) -> None:
    """Create a valid 24-bit stereo WAV with a 16-byte PEAK chunk that libsndfile rejects."""
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(frames, dtype=np.float64) / sample_rate
    left = np.sin(2.0 * np.pi * 440.0 * t) * 0.4
    right = np.sin(2.0 * np.pi * 880.0 * t) * 0.4
    interleaved = np.column_stack([left, right])
    ints = np.clip(np.round(interleaved * ((1 << 23) - 1)), -(1 << 23), (1 << 23) - 1).astype("<i4")
    raw_bytes = ints.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()

    channels = 2
    bits = 24
    block_align = channels * (bits // 8)
    byte_rate = sample_rate * block_align

    fmt_body = struct.pack("<HHIIHH", 1, channels, sample_rate, byte_rate, block_align, bits)
    fmt_chunk = b"fmt " + struct.pack("<I", len(fmt_body)) + fmt_body
    data_chunk = b"data" + struct.pack("<I", len(raw_bytes)) + raw_bytes
    # Version(4) + timestamp(4) + 1 channel peak(8) = 16 bytes on a stereo file -> LibsndfileError: Bad 'PEAK' chunk.
    peak_chunk = b"PEAK" + struct.pack("<I", 16) + struct.pack("<IIfI", 0, 0, 0.4, 0)
    riff_body = b"WAVE" + fmt_chunk + data_chunk + peak_chunk
    path.write_bytes(b"RIFF" + struct.pack("<I", len(riff_body)) + riff_body)


def test_bad_peak_wav_rescued_across_scan_playback_render_and_ai(tmp_path: Path):
    sfx = tmp_path / "sfx"
    wav_path = sfx / "bad_peak_24bit.wav"
    _make_bad_peak_wav(wav_path, sample_rate=48000, frames=14400)
    (sfx / "._mac_sidecar.wav").write_bytes(b"\x00\x05\x16\x07macOS_dotfile")

    with pytest.raises(sf.LibsndfileError):
        sf.info(str(wav_path))

    db_path = tmp_path / "fallback.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(sfx)
    report = LibraryScanner(db_path).scan_root(root.id)
    assert report.status == "completed"
    assert report.added == 1
    assert report.errors == 0

    rows = store.query_page(offset=0, limit=10)
    assert len(rows) == 1
    rec = rows[0]
    assert rec.filename == "bad_peak_24bit.wav"
    assert rec.sample_rate == 48000
    assert rec.channels == 2
    assert rec.bit_depth == 24
    assert abs(rec.duration_sec - 0.3) < 0.01
    store.close()

    with open_audio_file(wav_path) as handle:
        assert handle.samplerate == 48000
        assert handle.channels == 2
        assert handle.frames == 14400
        assert handle.subtype == "PCM_24"
        assert handle.seekable() is True
        part1 = handle.read(4096, dtype="float32", always_2d=True)
        assert part1.shape == (4096, 2)
        handle.seek(4800)
        part2 = handle.read(2048, dtype="float32", always_2d=True)
        assert part2.shape == (2048, 2)
        assert float(np.max(np.abs(part2))) > 0.1

    buf = decode_file_for_playback(wav_path, target_rate=48000)
    assert buf.frames == 14400
    assert buf.channel_count == 2
    assert float(np.max(np.abs(buf.samples))) > 0.1

    rendered = render_range(
        wav_path,
        start_seconds=0.0,
        end_seconds=0.1,
        out_dir=tmp_path / "renders",
    )
    assert rendered.is_file()
    rinfo = sf.info(str(rendered))
    assert rinfo.samplerate == 48000
    assert rinfo.frames == 4800

    segs, sr = decode_audio_windows_mono(wav_path, [(0.0, 0.25)])
    assert sr == 48000
    assert len(segs) == 1
    assert segs[0].dtype == np.float32
    assert segs[0].shape[0] == 12000
    assert float(np.max(np.abs(segs[0]))) > 0.1


def test_tail_seek_error_falls_back_to_pedalboard_and_streaming_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Simulate libsndfile raising an error on seek/EOF read, as seen on some MP3s."""
    import soundfile as sf

    import audiosearch.audioio.decode as decode_mod

    wav_path = tmp_path / "tail_quirk.wav"
    sr = 44100
    frames = sr * 2  # 2.0s
    t = np.arange(frames, dtype=np.float32) / sr
    tone = (np.sin(2.0 * np.pi * 440.0 * t) * 0.35).astype(np.float32)
    sf.write(str(wav_path), tone, sr, subtype="PCM_16")

    real_open = decode_mod.open_audio_file

    class _SeekFailHandle:
        """Proxy that succeeds for initial chunks but raises LibsndfileError on seek > 0 or late read."""

        def __init__(self, inner) -> None:
            self._inner = inner
            self._pos = 0

        @property
        def samplerate(self) -> int:
            return self._inner.samplerate

        @property
        def frames(self) -> int:
            return self._inner.frames

        @property
        def channels(self) -> int:
            return self._inner.channels

        @property
        def subtype(self) -> str:
            return self._inner.subtype

        def __len__(self) -> int:
            return len(self._inner)

        def seekable(self) -> bool:
            return True

        def seek(self, pos: int) -> int:
            if pos > 0:
                raise RuntimeError("simulated libsndfile seek failure")
            self._pos = 0
            return self._inner.seek(0)

        def read(self, count: int = -1, dtype: str = "float32", always_2d: bool = True):
            if count < 0 or self._pos + count >= self._inner.frames:
                raise RuntimeError("simulated libsndfile EOF internal error")
            arr = self._inner.read(count, dtype=dtype, always_2d=always_2d)
            self._pos += int(arr.shape[0])
            return arr

        def close(self) -> None:
            self._inner.close()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr(decode_mod, "open_audio_file", lambda p: _SeekFailHandle(real_open(p)))

    # 1. Both AI window decode & audition playback recover via pedalboard fallback
    segs, out_sr = decode_audio_windows_mono(wav_path, [(0.0, 1.0), (1.0, 2.0)])
    assert out_sr == sr
    assert len(segs) == 2
    assert segs[0].shape[0] == sr
    assert segs[1].shape[0] == sr
    assert float(np.max(np.abs(segs[1]))) > 0.1

    buf = decode_file_for_playback(wav_path, target_rate=48000)
    assert buf.sample_rate_hz == 48000
    assert buf.frames == 48000 * 2
    assert float(np.max(np.abs(buf.samples))) > 0.1

    # 2. If pedalboard also fails, sequential partial block fallback still rescues prefix frames
    def _no_pedalboard(_path):
        raise RuntimeError("pedalboard unavailable")

    monkeypatch.setattr(decode_mod, "open_pedalboard_audio", _no_pedalboard)
    segs_partial, out_sr2 = decode_audio_windows_mono(wav_path, [(0.0, 1.0)])
    assert out_sr2 == sr
    assert len(segs_partial) == 1
    assert segs_partial[0].shape[0] == sr
    assert float(np.max(np.abs(segs_partial[0]))) > 0.1
