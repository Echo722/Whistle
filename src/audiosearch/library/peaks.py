"""Local multi-channel waveform peak computation, spectral analysis, range zoom, and serialization."""

from __future__ import annotations

import colorsys
import functools
import math
import struct
import threading
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from audiosearch.audioio.decode import open_wav
from audiosearch.audioio.errors import UnsupportedFormatError

_HEADER_FMT = "<dddIII"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_SPECTRAL_MAGIC = b"SPK2"

SPECTRAL_MIN_HZ = 30.0
SPECTRAL_MAX_HZ = 18000.0
SPECTRAL_BANDS = 32
SPECTRAL_NOISE_FLOOR = 0.14
SPECTRAL_TONAL_CEIL = 0.56

_SPECTRAL_ANCHORS: tuple[tuple[float, float, float, float], ...] = (
    (0.00, 0.0, 0.88, 0.90),
    (0.18, 18.0, 0.88, 0.94),
    (0.33, 38.0, 0.88, 0.96),
    (0.47, 58.0, 0.86, 0.95),
    (0.63, 122.0, 0.80, 0.88),
    (0.78, 196.0, 0.80, 0.94),
    (0.90, 238.0, 0.72, 0.98),
    (1.00, 278.0, 0.74, 0.96),
)
_NOISE_RGB = (132.0, 140.0, 150.0)

THUMBNAIL_POINTS = 512
LARGE_WAV_STRIDE_BYTES = 10 * 1024 * 1024
LARGE_COMPRESSED_SECONDS = 45.0
MIN_THUMB_WIDTH_FRAC = 0.20
THUMB_REF_MIN_SECONDS = 0.08
THUMB_REF_MAX_SECONDS = 15.0


def _load_soundfile():
    import soundfile as sf

    return sf


@dataclass(frozen=True, slots=True)
class PeakData:
    """Per-channel (min, max) peak buckets in [-1.0, 1.0] and optional spectral tags."""

    duration_seconds: float
    sample_rate_hz: int
    channel_count: int
    points: int
    # Shape: (channels, points, 2) float32 where [:, :, 0] is min and [:, :, 1] is max.
    peaks: np.ndarray
    start_seconds: float = 0.0
    end_seconds: float = 0.0
    # Optional shape: (channels, points, 2) uint8 where [:, :, 0] is freq_u8 and [:, :, 1] is tonality_u8.
    spectral: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.end_seconds <= self.start_seconds and self.duration_seconds > 0:
            object.__setattr__(self, "end_seconds", self.duration_seconds)


class _PeakCache:
    def __init__(self, max_entries: int = 64) -> None:
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._entries: OrderedDict[tuple[str, int, int, float, float], PeakData] = OrderedDict()

    def get(self, key: tuple[str, int, int, float, float]) -> PeakData | None:
        with self._lock:
            value = self._entries.get(key)
            if value is not None:
                self._entries.move_to_end(key)
            return value

    def put(self, key: tuple[str, int, int, float, float], value: PeakData) -> None:
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)


_CACHE = _PeakCache()


def spectral_fft_size(sample_rate: int) -> int:
    """Choose an FFT window size appropriate for the sample rate."""
    if sample_rate <= 24000:
        return 512
    if sample_rate <= 56000:
        return 1024
    return 2048


@functools.lru_cache(maxsize=16)
def _spectral_tables(
    n_fft: int, sample_rate: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    sr = max(1, int(sample_rate))
    win = np.hanning(n_fft).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    f_lo = max(20.0, float(freqs[1]) if len(freqs) > 1 else 20.0)
    f_hi = min(SPECTRAL_MAX_HZ, float(sr) * 0.495)
    valid = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(valid):
        valid = np.ones_like(freqs, dtype=bool)
        if len(valid) > 1:
            valid[0] = False

    fv = freqs[valid].astype(np.float64)
    log_span = math.log2(SPECTRAL_MAX_HZ / SPECTRAL_MIN_HZ)
    log_f = np.clip(
        (np.log2(np.maximum(fv, SPECTRAL_MIN_HZ)) - math.log2(SPECTRAL_MIN_HZ)) / log_span,
        0.0,
        1.0,
    )

    lo_edge = max(float(fv[0]), 1.0)
    hi_edge = max(float(fv[-1]) * 1.0001, lo_edge * 1.1)
    edges = np.geomspace(lo_edge, hi_edge, SPECTRAL_BANDS + 1)
    bidx = np.clip(np.searchsorted(edges, fv, side="right") - 1, 0, SPECTRAL_BANDS - 1)
    band_mat = np.zeros((len(fv), SPECTRAL_BANDS), dtype=np.float64)
    band_mat[np.arange(len(fv)), bidx] = 1.0
    active_bands = band_mat.sum(axis=0) > 0
    band_mat = band_mat[:, active_bands]

    log_n_bins = math.log(max(len(fv), 2))
    log_n_bands = math.log(max(int(band_mat.shape[1]), 2))
    return win, valid, log_f, band_mat, log_n_bins, log_n_bands


def analyze_spectral_frames(windows: np.ndarray, sample_rate: int) -> np.ndarray:
    """Compute quantized ``(freq_u8, tonality_u8)`` pairs for ``(n_windows, n_fft)`` frames."""
    arr = np.asarray(windows, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return np.zeros((0, 2), dtype=np.uint8)

    n_fft = int(arr.shape[1])
    win, valid, log_f, band_mat, log_n_bins, log_n_bands = _spectral_tables(n_fft, int(sample_rate))

    centered = arr - arr.mean(axis=-1, keepdims=True)
    spectrum = np.fft.rfft(centered * win, axis=-1)[:, valid]
    pwr = (spectrum.real.astype(np.float64) ** 2) + (spectrum.imag.astype(np.float64) ** 2)
    total_pwr = pwr.sum(axis=-1)

    weights = pwr * np.sqrt(pwr)
    f_norm = np.clip((weights @ log_f) / (weights.sum(axis=-1) + 1e-20), 0.0, 1.0)

    pf = pwr / (total_pwr[:, None] + 1e-20)
    h_fine = -np.sum(pf * np.log(pf + 1e-20), axis=-1) / log_n_bins
    fine_ton = 1.0 - h_fine

    bp = pwr @ band_mat
    pb = bp / (bp.sum(axis=-1, keepdims=True) + 1e-20)
    h_band = -np.sum(pb * np.log(pb + 1e-20), axis=-1) / log_n_bands
    band_ton = 1.0 - h_band

    comb = np.minimum(fine_ton, band_ton)
    ton = np.clip(
        (comb - SPECTRAL_NOISE_FLOOR) / (SPECTRAL_TONAL_CEIL - SPECTRAL_NOISE_FLOOR),
        0.0,
        1.0,
    )

    silent = total_pwr <= 1e-12
    if np.any(silent):
        f_norm[silent] = 0.0
        ton[silent] = 0.0

    out = np.empty((arr.shape[0], 2), dtype=np.uint8)
    out[:, 0] = np.clip(np.rint(f_norm * 255.0), 0, 255).astype(np.uint8)
    out[:, 1] = np.clip(np.rint(ton * 255.0), 0, 255).astype(np.uint8)
    return out


def spectral_color_rgb(
    freq_u8: int, tonality_u8: int, *, played: bool = False
) -> tuple[int, int, int]:
    """Map ``(freq_u8, tonality_u8)`` to an RGB color readable on a dark workbench."""
    fn = max(0.0, min(1.0, float(freq_u8) / 255.0))
    tn = max(0.0, min(1.0, float(tonality_u8) / 255.0))

    hue_deg, sat, val = _SPECTRAL_ANCHORS[-1][1:]
    for i in range(len(_SPECTRAL_ANCHORS) - 1):
        n0, h0, s0, v0 = _SPECTRAL_ANCHORS[i]
        n1, h1, s1, v1 = _SPECTRAL_ANCHORS[i + 1]
        if fn <= n1:
            t = 0.0 if n1 <= n0 else (fn - n0) / (n1 - n0)
            hue_deg = h0 + (h1 - h0) * t
            sat = s0 + (s1 - s0) * t
            val = v0 + (v1 - v0) * t
            break

    tr, tg, tb = colorsys.hsv_to_rgb(hue_deg / 360.0, sat, val)
    t_vis = tn**0.85
    r = _NOISE_RGB[0] + (tr * 255.0 - _NOISE_RGB[0]) * t_vis
    g = _NOISE_RGB[1] + (tg * 255.0 - _NOISE_RGB[1]) * t_vis
    b = _NOISE_RGB[2] + (tb * 255.0 - _NOISE_RGB[2]) * t_vis

    if played:
        return (
            max(28, min(255, int(round(r * 0.70)))),
            max(28, min(255, int(round(g * 0.70)))),
            max(28, min(255, int(round(b * 0.70)))),
        )
    return (
        max(0, min(255, int(round(r)))),
        max(0, min(255, int(round(g)))),
        max(0, min(255, int(round(b)))),
    )


def serialize_peaks(data: PeakData) -> bytes:
    """Serialize ``PeakData`` into a compact binary blob (v2 when spectral data is present)."""
    header = struct.pack(
        _HEADER_FMT,
        float(data.duration_seconds),
        float(data.start_seconds),
        float(data.end_seconds),
        int(data.sample_rate_hz),
        int(data.channel_count),
        int(data.points),
    )
    arr = np.ascontiguousarray(data.peaks, dtype=np.float32)
    spec = data.spectral
    if spec is not None and spec.shape == (data.channel_count, data.points, 2):
        spec_u8 = np.ascontiguousarray(spec, dtype=np.uint8)
        return _SPECTRAL_MAGIC + header + arr.tobytes() + spec_u8.tobytes()
    return header + arr.tobytes()


def deserialize_peaks(blob: bytes) -> PeakData:
    """Restore ``PeakData`` from a ``waveform_cache`` v1 or v2 blob."""
    if len(blob) >= 4 + _HEADER_SIZE and blob[:4] == _SPECTRAL_MAGIC:
        offset = 4
        duration, start_sec, end_sec, rate, channels, points = struct.unpack(
            _HEADER_FMT, blob[offset : offset + _HEADER_SIZE]
        )
        peaks_bytes = channels * points * 2 * 4
        spec_bytes = channels * points * 2
        payload = blob[offset + _HEADER_SIZE :]
        if len(payload) != peaks_bytes + spec_bytes:
            raise ValueError("Invalid spectral peak blob payload size")
        arr = np.frombuffer(payload[:peaks_bytes], dtype=np.float32).reshape(
            (channels, points, 2)
        ).copy()
        spec = np.frombuffer(payload[peaks_bytes:], dtype=np.uint8).reshape(
            (channels, points, 2)
        ).copy()
        return PeakData(
            duration_seconds=duration,
            sample_rate_hz=rate,
            channel_count=channels,
            points=points,
            peaks=arr,
            start_seconds=start_sec,
            end_seconds=end_sec,
            spectral=spec,
        )

    if len(blob) < _HEADER_SIZE:
        raise ValueError("Invalid peak blob header")
    duration, start_sec, end_sec, rate, channels, points = struct.unpack(
        _HEADER_FMT, blob[:_HEADER_SIZE]
    )
    expected_bytes = channels * points * 2 * 4
    payload = blob[_HEADER_SIZE:]
    if len(payload) != expected_bytes:
        raise ValueError("Invalid peak blob payload size")
    arr = np.frombuffer(payload, dtype=np.float32).reshape((channels, points, 2)).copy()
    return PeakData(
        duration_seconds=duration,
        sample_rate_hz=rate,
        channel_count=channels,
        points=points,
        peaks=arr,
        start_seconds=start_sec,
        end_seconds=end_sec,
    )


def _extract_centered_window(
    buf: np.ndarray, buf_start: int, center_frame: int, n_fft: int, ch: int
) -> np.ndarray:
    half = n_fft // 2
    w0 = center_frame - half
    w1 = w0 + n_fft
    b0 = w0 - buf_start
    b1 = w1 - buf_start
    buf_len = int(buf.shape[0])
    if b0 >= 0 and b1 <= buf_len:
        return buf[b0:b1, ch]
    win = np.zeros(n_fft, dtype=np.float32)
    src0 = max(0, b0)
    src1 = min(buf_len, b1)
    if src1 > src0:
        dst0 = src0 - b0
        win[dst0 : dst0 + (src1 - src0)] = buf[src0:src1, ch]
    return win


def _extract_mono_centered_window(
    buf_mono: np.ndarray, buf_start: int, center_frame: int, n_fft: int
) -> np.ndarray:
    half = n_fft // 2
    w0 = center_frame - half
    w1 = w0 + n_fft
    b0 = w0 - buf_start
    b1 = w1 - buf_start
    buf_len = int(buf_mono.shape[0])
    if b0 >= 0 and b1 <= buf_len:
        return buf_mono[b0:b1]
    win = np.zeros(n_fft, dtype=np.float32)
    src0 = max(0, b0)
    src1 = min(buf_len, b1)
    if src1 > src0:
        dst0 = src0 - b0
        win[dst0 : dst0 + (src1 - src0)] = buf_mono[src0:src1]
    return win


def _flush_channel_batch(
    spectral: np.ndarray,
    ch: int,
    bucket_indices: list[int],
    windows: list[np.ndarray],
    sample_rate: int,
) -> None:
    if not bucket_indices:
        return
    tags = analyze_spectral_frames(np.stack(windows, axis=0), sample_rate)
    spectral[ch, bucket_indices, :] = tags
    bucket_indices.clear()
    windows.clear()


def _compute_peaks_from_chunks(
    chunks: Iterator[np.ndarray],
    *,
    total_frames: int,
    sample_rate: int,
    channels: int,
    first_frame: int,
    last_frame: int,
    points: int,
) -> PeakData:
    full_duration = total_frames / sample_rate
    actual_start = first_frame / sample_rate
    actual_end = last_frame / sample_rate
    span_frames = max(1, last_frame - first_frame)
    buckets = min(points, span_frames)

    mins = np.full((channels, buckets), np.inf, dtype=np.float32)
    maxs = np.full((channels, buckets), -np.inf, dtype=np.float32)
    spectral = np.zeros((channels, buckets, 2), dtype=np.uint8)

    n_fft = spectral_fft_size(sample_rate)
    half = n_fft // 2
    dense_mode = (span_frames / buckets) <= n_fft

    if dense_mode:
        hop = min(n_fft // 4, max(1, span_frames // 64))
        spec_stride = max(1, int((hop * buckets) // max(span_frames, 1)))
        anchor_buckets = list(range(0, buckets, spec_stride))
        anchor_centers = [
            min(span_frames - 1, int(((b + 0.5) * span_frames) // buckets)) for b in anchor_buckets
        ]
        next_anchor = 0
    else:
        spec_stride = 1
        anchor_buckets = []
        anchor_centers = []
        next_anchor = 0
        peak_amp = np.full((channels, buckets), -1.0, dtype=np.float32)
        peak_frame = np.zeros((channels, buckets), dtype=np.int64)
        pending_sparse: dict[tuple[int, int], int] = {}

    prev_tail = np.zeros((0, channels), dtype=np.float32)
    frame_offset = 0
    batch_buckets: list[list[int]] = [[] for _ in range(channels)]
    batch_windows: list[list[np.ndarray]] = [[] for _ in range(channels)]

    for block in chunks:
        n = int(block.shape[0])
        if n == 0:
            continue
        end_offset = frame_offset + n
        buf = block if prev_tail.shape[0] == 0 else np.concatenate([prev_tail, block], axis=0)
        buf_start = frame_offset - int(prev_tail.shape[0])

        # Flush deferred sparse peak windows whose right tail is now inside buf.
        if not dense_mode and pending_sparse:
            still_pending: dict[tuple[int, int], int] = {}
            for (ch, bucket), pf in pending_sparse.items():
                if pf - half + n_fft <= end_offset:
                    win = _extract_centered_window(buf, buf_start, pf, n_fft, ch)
                    batch_buckets[ch].append(bucket)
                    batch_windows[ch].append(win)
                    if len(batch_buckets[ch]) >= 256:
                        _flush_channel_batch(
                            spectral, ch, batch_buckets[ch], batch_windows[ch], sample_rate
                        )
                else:
                    still_pending[(ch, bucket)] = pf
            pending_sparse = still_pending

        b_start = int((frame_offset * buckets) // span_frames)
        b_end = min(buckets - 1, int(((end_offset - 1) * buckets) // span_frames))

        if b_start == b_end:
            blk_min = block.min(axis=0)
            blk_max = block.max(axis=0)
            mins[:, b_start] = np.minimum(mins[:, b_start], blk_min)
            maxs[:, b_start] = np.maximum(maxs[:, b_start], blk_max)
            if not dense_mode:
                abs_block = np.abs(block)
                idx_max = np.argmax(abs_block, axis=0)
                val_max = abs_block[idx_max, np.arange(channels)]
                for ch in range(channels):
                    if float(val_max[ch]) > float(peak_amp[ch, b_start]):
                        peak_amp[ch, b_start] = val_max[ch]
                        pf = frame_offset + int(idx_max[ch])
                        peak_frame[ch, b_start] = pf
                        if pf - half + n_fft <= end_offset:
                            win = _extract_centered_window(buf, buf_start, pf, n_fft, ch)
                            batch_buckets[ch].append(b_start)
                            batch_windows[ch].append(win)
                            if len(batch_buckets[ch]) >= 256:
                                _flush_channel_batch(
                                    spectral, ch, batch_buckets[ch], batch_windows[ch], sample_rate
                                )
                        else:
                            pending_sparse[(ch, b_start)] = pf
        else:
            abs_block = None if dense_mode else np.abs(block)
            for bucket in range(b_start, b_end + 1):
                f0 = max(0, int((bucket * span_frames) // buckets) - frame_offset)
                f1 = min(
                    n,
                    int(((bucket + 1) * span_frames + buckets - 1) // buckets) - frame_offset,
                )
                if f1 <= f0:
                    f1 = min(n, f0 + 1)
                if f0 < n and f1 > f0:
                    sl = block[f0:f1]
                    mins[:, bucket] = np.minimum(mins[:, bucket], sl.min(axis=0))
                    maxs[:, bucket] = np.maximum(maxs[:, bucket], sl.max(axis=0))
                    if abs_block is not None:
                        abs_sl = abs_block[f0:f1]
                        idx_max = np.argmax(abs_sl, axis=0)
                        val_max = abs_sl[idx_max, np.arange(channels)]
                        for ch in range(channels):
                            if float(val_max[ch]) > float(peak_amp[ch, bucket]):
                                peak_amp[ch, bucket] = val_max[ch]
                                pf = frame_offset + f0 + int(idx_max[ch])
                                peak_frame[ch, bucket] = pf
                                if pf - half + n_fft <= end_offset:
                                    win = _extract_centered_window(buf, buf_start, pf, n_fft, ch)
                                    batch_buckets[ch].append(bucket)
                                    batch_windows[ch].append(win)
                                    if len(batch_buckets[ch]) >= 256:
                                        _flush_channel_batch(
                                            spectral,
                                            ch,
                                            batch_buckets[ch],
                                            batch_windows[ch],
                                            sample_rate,
                                        )
                                else:
                                    pending_sparse[(ch, bucket)] = pf

        if dense_mode:
            while next_anchor < len(anchor_buckets):
                center = anchor_centers[next_anchor]
                if center - half + n_fft > end_offset:
                    break
                b_idx = anchor_buckets[next_anchor]
                for ch in range(channels):
                    win = _extract_centered_window(buf, buf_start, center, n_fft, ch)
                    batch_buckets[ch].append(b_idx)
                    batch_windows[ch].append(win)
                    if len(batch_buckets[ch]) >= 256:
                        _flush_channel_batch(
                            spectral, ch, batch_buckets[ch], batch_windows[ch], sample_rate
                        )
                next_anchor += 1

        tail_keep = min(n_fft, int(buf.shape[0]))
        prev_tail = buf[-tail_keep:].copy() if tail_keep > 0 else np.zeros((0, channels), dtype=np.float32)
        frame_offset = end_offset

    # Flush remaining windows at EOF with zero-padding.
    eof_buf = prev_tail
    eof_start = frame_offset - int(prev_tail.shape[0])
    if dense_mode:
        while next_anchor < len(anchor_buckets):
            center = anchor_centers[next_anchor]
            b_idx = anchor_buckets[next_anchor]
            for ch in range(channels):
                win = _extract_centered_window(eof_buf, eof_start, center, n_fft, ch)
                batch_buckets[ch].append(b_idx)
                batch_windows[ch].append(win)
            next_anchor += 1
    elif pending_sparse:
        for (ch, bucket), pf in pending_sparse.items():
            win = _extract_centered_window(eof_buf, eof_start, pf, n_fft, ch)
            batch_buckets[ch].append(bucket)
            batch_windows[ch].append(win)

    for ch in range(channels):
        _flush_channel_batch(spectral, ch, batch_buckets[ch], batch_windows[ch], sample_rate)

    if dense_mode and spec_stride > 1:
        for idx, anchor_b in enumerate(anchor_buckets):
            end_b = anchor_buckets[idx + 1] if idx + 1 < len(anchor_buckets) else buckets
            if end_b > anchor_b + 1:
                spectral[:, anchor_b + 1 : end_b, :] = spectral[:, anchor_b : anchor_b + 1, :]

    empty = ~np.isfinite(mins)
    mins[empty] = 0.0
    maxs[empty] = 0.0
    peaks = np.empty((channels, buckets, 2), dtype=np.float32)
    peaks[:, :, 0] = np.clip(mins, -1.0, 1.0)
    peaks[:, :, 1] = np.clip(maxs, -1.0, 1.0)

    return PeakData(
        duration_seconds=full_duration,
        sample_rate_hz=sample_rate,
        channel_count=channels,
        points=buckets,
        peaks=peaks,
        start_seconds=actual_start,
        end_seconds=actual_end,
        spectral=spectral,
    )


def compute_file_peaks(
    path: str | Path,
    *,
    points: int = 1024,
    mtime_ns: int = 0,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
) -> PeakData:
    """Decode peaks and spectral tags from a local audio file (full file or zoomed)."""
    if points <= 0:
        raise ValueError("points must be positive")
    file_path = Path(path)
    key_start = round(max(0.0, start_seconds), 6)
    key_end = -1.0 if end_seconds is None else round(end_seconds, 6)
    cache_key = (str(file_path), mtime_ns, points, key_start, key_end)
    if mtime_ns:
        cached = _CACHE.get(cache_key)
        if cached is not None:
            return cached

    result: PeakData | None = None
    if file_path.suffix.lower() == ".wav":
        try:
            with open_wav(file_path) as wav:
                total_frames = wav.info.frames
                sample_rate = wav.info.sample_rate_hz
                channels = max(int(wav.info.channel_count), 1)
                if total_frames <= 0 or sample_rate <= 0:
                    raise ValueError("音频没有可绘制的采样")
                first_frame = min(
                    max(int(round(max(0.0, start_seconds) * sample_rate)), 0), total_frames - 1
                )
                last_frame = (
                    total_frames
                    if end_seconds is None
                    else min(max(int(round(end_seconds * sample_rate)), first_frame + 1), total_frames)
                )
                result = _compute_peaks_from_chunks(
                    wav.iter_frames(first_frame, last_frame, block_frames=65536),
                    total_frames=total_frames,
                    sample_rate=sample_rate,
                    channels=channels,
                    first_frame=first_frame,
                    last_frame=last_frame,
                    points=points,
                )
        except UnsupportedFormatError:
            result = None

    if result is None:
        sf = _load_soundfile()
        with sf.SoundFile(str(file_path)) as handle:
            total_frames = int(handle.frames)
            sample_rate = int(handle.samplerate)
            channels = max(int(handle.channels), 1)
            if total_frames <= 0 or sample_rate <= 0:
                raise ValueError("音频没有可绘制的采样")
            first_frame = min(
                max(int(round(max(0.0, start_seconds) * sample_rate)), 0), total_frames - 1
            )
            last_frame = (
                total_frames
                if end_seconds is None
                else min(max(int(round(end_seconds * sample_rate)), first_frame + 1), total_frames)
            )
            if first_frame > 0:
                handle.seek(first_frame)

            def _sf_chunks() -> Iterator[np.ndarray]:
                remaining = max(1, last_frame - first_frame)
                while remaining > 0:
                    to_read = min(65536, remaining)
                    block = handle.read(frames=to_read, dtype="float32", always_2d=True)
                    if block.shape[0] == 0:
                        break
                    remaining -= int(block.shape[0])
                    yield block

            result = _compute_peaks_from_chunks(
                _sf_chunks(),
                total_frames=total_frames,
                sample_rate=sample_rate,
                channels=channels,
                first_frame=first_frame,
                last_frame=last_frame,
                points=points,
            )

    if mtime_ns:
        _CACHE.put(cache_key, result)
    return result


def encode_thumbnail_amplitudes(amps: np.ndarray) -> bytes:
    """Quantize mono peak amplitudes in ``[0.0, 1.0]`` with sqrt (gamma 0.5) to ``uint8``."""
    arr = np.asarray(amps, dtype=np.float32)
    if arr.size == 0:
        return b""
    quantized = np.clip(np.rint(np.sqrt(np.clip(arr, 0.0, 1.0)) * 255.0), 0, 255)
    return quantized.astype(np.uint8).tobytes()


def shape_thumbnail_envelope(blob: bytes) -> np.ndarray:
    """Decode thumbnail blob into true linear absolute peak amplitudes in ``[0.0, 1.0]``."""
    if not blob:
        return np.zeros(0, dtype=np.float32)
    raw = (np.frombuffer(blob, dtype=np.uint8).astype(np.float32) / 255.0) ** 2
    return np.clip(raw, 0.0, 1.0).astype(np.float32, copy=False)


def thumbnail_width_fraction(duration_seconds: float) -> float:
    """Map audio duration to inline waveform width fraction in ``[0.20, 1.00]`` (log scale)."""
    dur = float(duration_seconds)
    if dur <= 0.0:
        return 1.0
    if dur <= THUMB_REF_MIN_SECONDS:
        return MIN_THUMB_WIDTH_FRAC
    if dur >= THUMB_REF_MAX_SECONDS:
        return 1.0
    span = math.log(THUMB_REF_MAX_SECONDS / THUMB_REF_MIN_SECONDS)
    t = math.log(dur / THUMB_REF_MIN_SECONDS) / span
    return MIN_THUMB_WIDTH_FRAC + (1.0 - MIN_THUMB_WIDTH_FRAC) * t


def peaks_to_thumbnail(data: PeakData, points: int = THUMBNAIL_POINTS) -> bytes:
    """Downsample full-file ``PeakData`` into a compact mono thumbnail blob."""
    if points <= 0:
        raise ValueError("points must be positive")
    if data.peaks.size == 0 or data.points <= 0:
        return b""
    mono = np.max(np.abs(data.peaks), axis=(0, 2)).astype(np.float32, copy=False)
    total = int(mono.shape[0])
    if total <= points:
        return encode_thumbnail_amplitudes(mono)
    buckets = min(points, total)
    amps = np.zeros(buckets, dtype=np.float32)
    for bucket in range(buckets):
        i0 = (bucket * total) // buckets
        i1 = ((bucket + 1) * total) // buckets
        if i1 <= i0:
            i1 = min(total, i0 + 1)
        amps[bucket] = float(np.max(mono[i0:i1]))
    return encode_thumbnail_amplitudes(amps)


def peaks_to_thumbnail_spectral(data: PeakData, points: int = THUMBNAIL_POINTS) -> bytes:
    """Downsample ``PeakData.spectral`` by picking the loudest-bucket spectral tag."""
    if points <= 0:
        raise ValueError("points must be positive")
    spec = data.spectral
    if (
        spec is None
        or data.peaks.size == 0
        or data.points <= 0
        or spec.shape != (data.channel_count, data.points, 2)
    ):
        return b""
    ch_amp = np.max(np.abs(data.peaks), axis=2)  # (channels, points)
    ch_best = np.argmax(ch_amp, axis=0)
    mono = ch_amp[ch_best, np.arange(data.points)]
    total = int(mono.shape[0])
    if total <= points:
        selected = spec[ch_best, np.arange(total), :]
        return np.ascontiguousarray(selected, dtype=np.uint8).tobytes()
    buckets = min(points, total)
    out = np.empty((buckets, 2), dtype=np.uint8)
    for bucket in range(buckets):
        i0 = (bucket * total) // buckets
        i1 = ((bucket + 1) * total) // buckets
        if i1 <= i0:
            i1 = min(total, i0 + 1)
        idx = i0 + int(np.argmax(mono[i0:i1]))
        out[bucket] = spec[int(ch_best[idx]), idx, :]
    return out.tobytes()


def decode_thumbnail_spectral(blob: bytes) -> np.ndarray:
    """Decode thumbnail spectral bytes into ``(N, 2)`` ``uint8`` array."""
    if not blob or len(blob) % 2 != 0:
        return np.zeros((0, 2), dtype=np.uint8)
    return np.frombuffer(blob, dtype=np.uint8).reshape(-1, 2).copy()


def resample_spectral_columns(
    envelope: np.ndarray, spectral: np.ndarray, cols: int
) -> tuple[np.ndarray, np.ndarray]:
    """Resample 1D peak envelope and ``(N, 2)`` spectral tags into ``cols`` pixel columns."""
    n_pts = int(envelope.shape[0])
    if cols <= 0 or n_pts <= 0:
        return np.zeros(0, dtype=np.float32), np.zeros((0, 2), dtype=np.uint8)
    if spectral.shape[0] != n_pts:
        spec_arr = np.zeros((n_pts, 2), dtype=np.uint8)
    else:
        spec_arr = spectral
    if cols == n_pts:
        return envelope.astype(np.float32, copy=False), spec_arr.astype(np.uint8, copy=False)
    if cols > n_pts:
        xp = np.arange(n_pts, dtype=np.float32)
        x = np.linspace(0.0, float(n_pts - 1), cols, dtype=np.float32)
        env_out = np.interp(x, xp, envelope).astype(np.float32, copy=False)
        idx = np.clip(np.rint(x).astype(np.intp), 0, n_pts - 1)
        return env_out, spec_arr[idx].astype(np.uint8, copy=False)

    env_out = np.empty(cols, dtype=np.float32)
    spec_out = np.empty((cols, 2), dtype=np.uint8)
    for col in range(cols):
        i0 = (col * n_pts) // cols
        i1 = ((col + 1) * n_pts) // cols
        if i1 <= i0:
            i1 = min(n_pts, i0 + 1)
        best = i0 + int(np.argmax(envelope[i0:i1]))
        env_out[col] = float(envelope[best])
        spec_out[col] = spec_arr[best]
    return env_out, spec_out


def _accumulate_block_peaks(
    amps: np.ndarray,
    block: np.ndarray,
    frame_offset: int,
    total_frames: int,
    buckets: int,
) -> None:
    n = int(block.shape[0])
    if n <= 0:
        return
    end_offset = frame_offset + n
    b_start = int((frame_offset * buckets) // total_frames)
    b_end = min(buckets - 1, int(((end_offset - 1) * buckets) // total_frames))
    if b_start == b_end:
        val = float(np.max(np.abs(block)))
        if val > amps[b_start]:
            amps[b_start] = val
        return
    abs_block = np.abs(block)
    for bucket in range(b_start, b_end + 1):
        f0 = max(0, int((bucket * total_frames) // buckets) - frame_offset)
        f1 = min(
            n,
            int(((bucket + 1) * total_frames + buckets - 1) // buckets) - frame_offset,
        )
        if f1 <= f0:
            f1 = min(n, f0 + 1)
        if f0 < n and f1 > f0:
            val = float(np.max(abs_block[f0:f1]))
            if val > amps[bucket]:
                amps[bucket] = val


def _slice_mono_fft_window(block: np.ndarray, n_fft: int) -> np.ndarray:
    mono = block.mean(axis=1) if block.ndim == 2 and block.shape[1] > 1 else block.reshape(-1)
    n = int(mono.shape[0])
    if n >= n_fft:
        start = (n - n_fft) // 2
        return mono[start : start + n_fft].astype(np.float32, copy=False)
    win = np.zeros(n_fft, dtype=np.float32)
    if n > 0:
        win[:n] = mono
    return win


def _compute_mono_thumb_and_spectral_from_chunks(
    chunks: Iterator[np.ndarray],
    *,
    total_frames: int,
    sample_rate: int,
    buckets: int,
) -> tuple[bytes, bytes]:
    amps = np.zeros(buckets, dtype=np.float32)
    spectral = np.zeros((buckets, 2), dtype=np.uint8)
    n_fft = min(512, spectral_fft_size(sample_rate))
    half = n_fft // 2
    dense_mode = (total_frames / buckets) <= n_fft

    if dense_mode:
        hop = min(n_fft // 4, max(1, total_frames // 64))
        spec_stride = max(1, int((hop * buckets) // max(total_frames, 1)))
        anchor_buckets = list(range(0, buckets, spec_stride))
        anchor_centers = [
            min(total_frames - 1, int(((b + 0.5) * total_frames) // buckets)) for b in anchor_buckets
        ]
        next_anchor = 0
    else:
        spec_stride = 1
        anchor_buckets = []
        anchor_centers = []
        next_anchor = 0
        peak_amp = np.full(buckets, -1.0, dtype=np.float32)
        pending_sparse: dict[int, int] = {}

    prev_tail = np.zeros(0, dtype=np.float32)
    frame_offset = 0
    batch_buckets: list[int] = []
    batch_windows: list[np.ndarray] = []

    def _flush() -> None:
        if not batch_buckets:
            return
        tags = analyze_spectral_frames(np.stack(batch_windows, axis=0), sample_rate)
        spectral[batch_buckets, :] = tags
        batch_buckets.clear()
        batch_windows.clear()

    for block in chunks:
        n = int(block.shape[0])
        if n == 0:
            continue
        mono = (
            block.mean(axis=1, dtype=np.float32)
            if block.ndim == 2 and block.shape[1] > 1
            else block.reshape(-1).astype(np.float32, copy=False)
        )
        end_offset = frame_offset + n
        buf = mono if prev_tail.size == 0 else np.concatenate([prev_tail, mono])
        buf_start = frame_offset - int(prev_tail.size)

        if not dense_mode and pending_sparse:
            still_pending: dict[int, int] = {}
            for bucket, pf in pending_sparse.items():
                if pf - half + n_fft <= end_offset:
                    batch_buckets.append(bucket)
                    batch_windows.append(_extract_mono_centered_window(buf, buf_start, pf, n_fft))
                    if len(batch_buckets) >= 256:
                        _flush()
                else:
                    still_pending[bucket] = pf
            pending_sparse = still_pending

        b_start = int((frame_offset * buckets) // total_frames)
        b_end = min(buckets - 1, int(((end_offset - 1) * buckets) // total_frames))
        if b_start == b_end:
            abs_mono = np.abs(block)
            val = float(np.max(abs_mono))
            if val > amps[b_start]:
                amps[b_start] = val
            if not dense_mode:
                abs_m = np.abs(mono)
                idx_max = int(np.argmax(abs_m))
                m_val = float(abs_m[idx_max])
                if m_val > float(peak_amp[b_start]):
                    peak_amp[b_start] = m_val
                    pf = frame_offset + idx_max
                    if pf - half + n_fft <= end_offset:
                        batch_buckets.append(b_start)
                        batch_windows.append(
                            _extract_mono_centered_window(buf, buf_start, pf, n_fft)
                        )
                        if len(batch_buckets) >= 256:
                            _flush()
                    else:
                        pending_sparse[b_start] = pf
        else:
            abs_block = np.abs(block)
            abs_mono = None if dense_mode else np.abs(mono)
            for bucket in range(b_start, b_end + 1):
                f0 = max(0, int((bucket * total_frames) // buckets) - frame_offset)
                f1 = min(
                    n,
                    int(((bucket + 1) * total_frames + buckets - 1) // buckets) - frame_offset,
                )
                if f1 <= f0:
                    f1 = min(n, f0 + 1)
                if f0 < n and f1 > f0:
                    val = float(np.max(abs_block[f0:f1]))
                    if val > amps[bucket]:
                        amps[bucket] = val
                    if abs_mono is not None:
                        abs_sl = abs_mono[f0:f1]
                        idx_max = int(np.argmax(abs_sl))
                        m_val = float(abs_sl[idx_max])
                        if m_val > float(peak_amp[bucket]):
                            peak_amp[bucket] = m_val
                            pf = frame_offset + f0 + idx_max
                            if pf - half + n_fft <= end_offset:
                                batch_buckets.append(bucket)
                                batch_windows.append(
                                    _extract_mono_centered_window(buf, buf_start, pf, n_fft)
                                )
                                if len(batch_buckets) >= 256:
                                    _flush()
                            else:
                                pending_sparse[bucket] = pf

        if dense_mode:
            while next_anchor < len(anchor_buckets):
                center = anchor_centers[next_anchor]
                if center - half + n_fft > end_offset:
                    break
                batch_buckets.append(anchor_buckets[next_anchor])
                batch_windows.append(_extract_mono_centered_window(buf, buf_start, center, n_fft))
                if len(batch_buckets) >= 256:
                    _flush()
                next_anchor += 1

        tail_keep = min(n_fft, int(buf.size))
        prev_tail = buf[-tail_keep:].copy() if tail_keep > 0 else np.zeros(0, dtype=np.float32)
        frame_offset = end_offset

    eof_buf = prev_tail
    eof_start = frame_offset - int(prev_tail.size)
    if dense_mode:
        while next_anchor < len(anchor_buckets):
            center = anchor_centers[next_anchor]
            batch_buckets.append(anchor_buckets[next_anchor])
            batch_windows.append(_extract_mono_centered_window(eof_buf, eof_start, center, n_fft))
            next_anchor += 1
    elif pending_sparse:
        for bucket, pf in pending_sparse.items():
            batch_buckets.append(bucket)
            batch_windows.append(_extract_mono_centered_window(eof_buf, eof_start, pf, n_fft))

    _flush()

    if dense_mode and spec_stride > 1:
        for idx, anchor_b in enumerate(anchor_buckets):
            end_b = anchor_buckets[idx + 1] if idx + 1 < len(anchor_buckets) else buckets
            if end_b > anchor_b + 1:
                spectral[anchor_b + 1 : end_b, :] = spectral[anchor_b : anchor_b + 1, :]

    return encode_thumbnail_amplitudes(amps), spectral.tobytes()


def compute_thumbnail_with_spectral(
    path: str | Path,
    *,
    points: int = THUMBNAIL_POINTS,
    stride_threshold_bytes: int = LARGE_WAV_STRIDE_BYTES,
) -> tuple[bytes, bytes]:
    """Compute mono amplitude thumbnail and paired ``(buckets, 2)`` spectral bytes in one pass."""
    if points <= 0:
        raise ValueError("points must be positive")
    file_path = Path(path)

    if file_path.suffix.lower() == ".wav":
        try:
            with open_wav(file_path) as wav:
                total_frames = wav.info.frames
                sample_rate = wav.info.sample_rate_hz
                if total_frames <= 0 or sample_rate <= 0:
                    return b"", b""
                buckets = min(points, total_frames)
                if wav.info.data_bytes > stride_threshold_bytes:
                    amps = np.zeros(buckets, dtype=np.float32)
                    n_fft = min(512, spectral_fft_size(sample_rate))
                    windows = np.zeros((buckets, n_fft), dtype=np.float32)
                    for bucket in range(buckets):
                        f0 = (bucket * total_frames) // buckets
                        f1 = ((bucket + 1) * total_frames) // buckets
                        span = max(1, f1 - f0)
                        take = min(1024, span)
                        start = f0 + max(0, (span - take) // 2)
                        block = wav.read_frames(start, take)
                        if block.shape[0] > 0:
                            amps[bucket] = float(np.max(np.abs(block)))
                            windows[bucket] = _slice_mono_fft_window(block, n_fft)
                    spectral = analyze_spectral_frames(windows, sample_rate)
                    return encode_thumbnail_amplitudes(amps), spectral.tobytes()

                return _compute_mono_thumb_and_spectral_from_chunks(
                    wav.iter_frames(0, total_frames, block_frames=65536),
                    total_frames=total_frames,
                    sample_rate=sample_rate,
                    buckets=buckets,
                )
        except UnsupportedFormatError:
            pass

    sf = _load_soundfile()
    with sf.SoundFile(str(file_path)) as handle:
        total_frames = int(handle.frames)
        sample_rate = int(handle.samplerate)
        if total_frames <= 0 or sample_rate <= 0:
            return b"", b""
        buckets = min(points, total_frames)
        duration = total_frames / sample_rate
        if duration > LARGE_COMPRESSED_SECONDS and handle.seekable():
            amps = np.zeros(buckets, dtype=np.float32)
            n_fft = min(512, spectral_fft_size(sample_rate))
            windows = np.zeros((buckets, n_fft), dtype=np.float32)
            try:
                for bucket in range(buckets):
                    f0 = (bucket * total_frames) // buckets
                    f1 = ((bucket + 1) * total_frames) // buckets
                    span = max(1, f1 - f0)
                    take = min(1024, span)
                    start = f0 + max(0, (span - take) // 2)
                    handle.seek(start)
                    block = handle.read(frames=take, dtype="float32", always_2d=True)
                    if block.shape[0] > 0:
                        amps[bucket] = float(np.max(np.abs(block)))
                        windows[bucket] = _slice_mono_fft_window(block, n_fft)
                spectral = analyze_spectral_frames(windows, sample_rate)
                return encode_thumbnail_amplitudes(amps), spectral.tobytes()
            except Exception:
                handle.seek(0)

        def _sf_chunks() -> Iterator[np.ndarray]:
            remaining = total_frames
            while remaining > 0:
                to_read = min(65536, remaining)
                block = handle.read(frames=to_read, dtype="float32", always_2d=True)
                if block.shape[0] == 0:
                    break
                remaining -= int(block.shape[0])
                yield block

        return _compute_mono_thumb_and_spectral_from_chunks(
            _sf_chunks(),
            total_frames=total_frames,
            sample_rate=sample_rate,
            buckets=buckets,
        )


def compute_thumbnail_peaks(
    path: str | Path,
    *,
    points: int = THUMBNAIL_POINTS,
    stride_threshold_bytes: int = LARGE_WAV_STRIDE_BYTES,
) -> bytes:
    """Compute a compact mono amplitude thumbnail, using sparse stride reads on large files."""
    if points <= 0:
        raise ValueError("points must be positive")
    file_path = Path(path)

    if file_path.suffix.lower() == ".wav":
        try:
            with open_wav(file_path) as wav:
                total_frames = wav.info.frames
                if total_frames <= 0:
                    return b""
                buckets = min(points, total_frames)
                amps = np.zeros(buckets, dtype=np.float32)
                if wav.info.data_bytes > stride_threshold_bytes:
                    for bucket in range(buckets):
                        f0 = (bucket * total_frames) // buckets
                        f1 = ((bucket + 1) * total_frames) // buckets
                        span = max(1, f1 - f0)
                        take = min(1024, span)
                        start = f0 + max(0, (span - take) // 2)
                        block = wav.read_frames(start, take)
                        if block.shape[0] > 0:
                            amps[bucket] = float(np.max(np.abs(block)))
                else:
                    chunk_frames = 65536
                    frame_offset = 0
                    while frame_offset < total_frames:
                        take = min(chunk_frames, total_frames - frame_offset)
                        block = wav.read_frames(frame_offset, take)
                        n = block.shape[0]
                        if n == 0:
                            break
                        _accumulate_block_peaks(amps, block, frame_offset, total_frames, buckets)
                        frame_offset += n
                return encode_thumbnail_amplitudes(amps)
        except UnsupportedFormatError:
            pass

    sf = _load_soundfile()
    with sf.SoundFile(str(file_path)) as handle:
        total_frames = int(handle.frames)
        sample_rate = int(handle.samplerate)
        if total_frames <= 0 or sample_rate <= 0:
            return b""
        buckets = min(points, total_frames)
        duration = total_frames / sample_rate
        amps = np.zeros(buckets, dtype=np.float32)
        used_stride = False
        if duration > LARGE_COMPRESSED_SECONDS and handle.seekable():
            try:
                for bucket in range(buckets):
                    f0 = (bucket * total_frames) // buckets
                    f1 = ((bucket + 1) * total_frames) // buckets
                    span = max(1, f1 - f0)
                    take = min(1024, span)
                    start = f0 + max(0, (span - take) // 2)
                    handle.seek(start)
                    block = handle.read(frames=take, dtype="float32", always_2d=True)
                    if block.shape[0] > 0:
                        amps[bucket] = float(np.max(np.abs(block)))
                used_stride = True
            except Exception:
                amps.fill(0.0)
                handle.seek(0)
        if not used_stride:
            chunk_frames = 65536
            frame_offset = 0
            while frame_offset < total_frames:
                to_read = min(chunk_frames, total_frames - frame_offset)
                block = handle.read(frames=to_read, dtype="float32", always_2d=True)
                n = block.shape[0]
                if n == 0:
                    break
                _accumulate_block_peaks(amps, block, frame_offset, total_frames, buckets)
                frame_offset += n
        return encode_thumbnail_amplitudes(amps)
