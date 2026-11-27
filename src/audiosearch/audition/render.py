"""Render a source range to a real WAV file on disk across all soundfile formats."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from audiosearch.audioio.decode import open_audio_file
from audiosearch.audition.stream import FractionalResampler

RENDER_DIR_NAME = "renders"
_HIGH_BIT_SUBTYPES = frozenset({"PCM_24", "PCM_32", "FLOAT", "DOUBLE"})


def render_range(
    source: Path,
    *,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    out_dir: Path,
    fade_ms: float = 0.0,
    pitch_semitones: float = 0.0,
    rate: float = 1.0,
    preserve_pitch: bool = False,
) -> Path:
    """Write ``[start_seconds, end_seconds)`` (or full file) of ``source`` as WAV under ``out_dir``.

    Supports WAV, AIF/AIFF, FLAC, OGG, MP3, and OPUS via libsndfile. Optionally bakes
    varispeed or Rubber Band pitch/rate changes and linear boundary fades.
    """
    if start_seconds < 0 or (end_seconds is not None and end_seconds <= start_seconds):
        raise ValueError(f"无效选区：[{start_seconds}, {end_seconds})")

    pitch_clamped = max(-24.0, min(float(pitch_semitones), 24.0))
    rate_clamped = max(0.25, min(float(rate), 4.0))

    out_dir.mkdir(parents=True, exist_ok=True)
    with open_audio_file(source) as reader:
        sr = int(reader.samplerate)
        total_frames = int(reader.frames)
        if sr <= 0 or total_frames <= 0:
            raise ValueError("音频没有可播放的采样")
        eff_end_seconds = (total_frames / sr) if end_seconds is None else float(end_seconds)
        first = min(max(int(round(start_seconds * sr)), 0), total_frames)
        last = min(max(int(round(eff_end_seconds * sr)), first), total_frames)
        count = last - first
        if count <= 0:
            raise ValueError("选区不包含任何有效采样帧")
        reader.seek(first)
        block = reader.read(frames=count, dtype="float32", always_2d=True)
        subtype = "PCM_24" if str(reader.subtype or "") in _HIGH_BIT_SUBTYPES else "PCM_16"

    has_tone_change = abs(pitch_clamped) > 1e-4 or abs(rate_clamped - 1.0) > 1e-4
    if has_tone_change:
        if preserve_pitch:
            import pedalboard

            hq = block.shape[0] <= 48000 * 10
            stretched = pedalboard.time_stretch(
                np.ascontiguousarray(block.T, dtype=np.float32),
                float(sr),
                stretch_factor=rate_clamped,
                pitch_shift_in_semitones=pitch_clamped,
                high_quality=hq,
                preserve_formants=False,
            )
            block = np.ascontiguousarray(stretched.T, dtype=np.float32)
        else:
            speed = rate_clamped * (2.0 ** (pitch_clamped / 12.0))
            block = FractionalResampler(sr, sr, block.shape[1], speed=speed).process(
                block, eof=True
            )

    if fade_ms > 0 and block.shape[0] > 1:
        fade_frames = min(int(round((fade_ms / 1000.0) * sr)), block.shape[0] // 2)
        if fade_frames > 0:
            ramp_in = np.linspace(0.0, 1.0, fade_frames, dtype=np.float32)[:, np.newaxis]
            ramp_out = np.linspace(1.0, 0.0, fade_frames, dtype=np.float32)[:, np.newaxis]
            block[:fade_frames] *= ramp_in
            block[-fade_frames:] *= ramp_out

    block = np.clip(block, -1.0, 1.0)
    start_ms = int(round(start_seconds * 1000))
    end_ms = int(round(eff_end_seconds * 1000))
    tone_suffix = ""
    if has_tone_change:
        pp = "_pp" if preserve_pitch else ""
        tone_suffix = f"_p{pitch_clamped:+.1f}_r{rate_clamped:.2f}{pp}"
    fade_suffix = "_fade" if fade_ms > 0 else ""
    target = out_dir / f"{source.stem}_{start_ms}-{end_ms}{tone_suffix}{fade_suffix}.wav"
    sf.write(str(target), block, sr, subtype=subtype, format="WAV")
    return target


def prune_render_dir(out_dir: Path, *, max_files: int = 50) -> int:
    """Remove oldest rendered WAV files beyond ``max_files``."""
    if not out_dir.exists() or max_files < 0:
        return 0
    files = [p for p in out_dir.iterdir() if p.is_file() and p.suffix.lower() == ".wav"]
    if len(files) <= max_files:
        return 0
    files.sort(key=lambda p: p.stat().st_mtime_ns)
    removed = 0
    for old in files[: len(files) - max_files]:
        try:
            old.unlink()
            removed += 1
        except OSError:
            pass
    return removed


__all__ = ["RENDER_DIR_NAME", "prune_render_dir", "render_range"]
