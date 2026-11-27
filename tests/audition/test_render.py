"""Behavioral tests for multi-format selection rendering and directory pruning."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from audiosearch.audition.render import prune_render_dir, render_range


def test_render_range_supports_wav_flac_ogg_and_fade(tmp_path: Path):
    rate = 48000
    samples = np.full((rate, 2), 0.5, dtype=np.float32)

    for ext, fmt in [("wav", "WAV"), ("flac", "FLAC"), ("ogg", "OGG")]:
        src = tmp_path / f"src.{ext}"
        sf.write(str(src), samples, rate, format=fmt)

        out_raw = render_range(
            src, start_seconds=0.25, end_seconds=0.75, out_dir=tmp_path / "renders"
        )
        assert out_raw.exists()
        with sf.SoundFile(str(out_raw)) as rendered:
            assert rendered.samplerate == rate
            assert rendered.channels == 2
            assert abs(rendered.frames - rate // 2) <= 2

        out_fade = render_range(
            src,
            start_seconds=0.25,
            end_seconds=0.75,
            out_dir=tmp_path / "renders",
            fade_ms=5.0,
        )
        data, _ = sf.read(str(out_fade), dtype="float32")
        assert abs(float(data[0, 0])) < 0.05
        assert abs(float(data[-1, 0])) < 0.05
        assert float(data[len(data) // 2, 0]) > 0.3


def test_prune_render_dir_keeps_newest(tmp_path: Path):
    out_dir = tmp_path / "renders"
    out_dir.mkdir()
    base = time.time() - 100
    for idx in range(5):
        p = out_dir / f"clip_{idx}.wav"
        p.write_bytes(b"RIFF")
        os.utime(p, (base + idx, base + idx))

    removed = prune_render_dir(out_dir, max_files=2)
    assert removed == 3
    remaining = sorted(p.name for p in out_dir.iterdir())
    assert remaining == ["clip_3.wav", "clip_4.wav"]


def test_render_range_full_file_and_baked_pitch_rate(tmp_path: Path):
    sr = 48000
    t = np.arange(sr, dtype=np.float64) / sr
    sine = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    stereo = np.stack([sine, sine], axis=1)
    src = tmp_path / "tone.wav"
    sf.write(str(src), stereo, sr, format="WAV")

    # 1. Varispeed full-file bake (+12 st -> 2x speed, ~0.5s, 880 Hz)
    out_varispeed = render_range(
        src,
        start_seconds=0.0,
        end_seconds=None,
        out_dir=tmp_path / "renders",
        pitch_semitones=12.0,
        rate=1.0,
        preserve_pitch=False,
    )
    assert out_varispeed.exists()
    data_v, fs_v = sf.read(str(out_varispeed), dtype="float32")
    assert fs_v == sr
    assert 23800 <= len(data_v) <= 24200
    peak_v = float(np.fft.rfftfreq(len(data_v), 1.0 / sr)[np.argmax(np.abs(np.fft.rfft(data_v[:, 0])))])
    assert abs(peak_v - 880.0) < 5.0

    # 2. Preserve-pitch full-file bake (+5 st, 1.25x -> ~0.8s, ~587.3 Hz, strong RMS)
    out_pp = render_range(
        src,
        start_seconds=0.0,
        end_seconds=None,
        out_dir=tmp_path / "renders",
        pitch_semitones=5.0,
        rate=1.25,
        preserve_pitch=True,
    )
    assert out_pp.exists() and "_pp" in out_pp.stem
    data_pp, fs_pp = sf.read(str(out_pp), dtype="float32")
    assert fs_pp == sr
    assert 37500 <= len(data_pp) <= 39500
    peak_pp = float(np.fft.rfftfreq(len(data_pp), 1.0 / sr)[np.argmax(np.abs(np.fft.rfft(data_pp[:, 0])))])
    assert abs(peak_pp - 440.0 * (2.0 ** (5.0 / 12.0))) < 5.0
    assert float(np.sqrt(np.mean(data_pp**2))) > 0.25
