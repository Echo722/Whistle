"""True per-channel waveform peaks for an original-file range.

``WaveformProvider.waveform`` streams the requested range once and reduces it
into the requested number of equally spaced buckets, so memory stays bounded
regardless of file length. Every frame inside the range contributes to exactly
one bucket; nothing is decimated away and no synthetic values are produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from audiosearch.contracts.models import (
    Peak,
    WaveformData,
    WaveformRequest,
    validate_source_range,
)

from .catalog import ManifestCatalog
from .decode import DEFAULT_BLOCK_FRAMES, WavReader, open_wav
from .errors import AudioError, SourceRangeError

# Buckets smaller than this many frames would give single-sample "peaks" that
# look like a waveform but carry no envelope. We still honour the request.
_MIN_FRAMES_PER_BUCKET_HINT = 4


@dataclass(frozen=True, slots=True)
class PeakResult:
    """Peaks plus the facts a caller may want to report."""

    channel_indices: list[int]
    minima: np.ndarray  # (points, channels)
    maxima: np.ndarray  # (points, channels)
    frames: int
    clamped_frames: int
    empty_buckets: int


def _bucket_ids(frame_offsets: np.ndarray, points: int, total_frames: int) -> np.ndarray:
    return (frame_offsets * points) // total_frames


def compute_peaks(
    reader: WavReader,
    *,
    start_seconds: float,
    end_seconds: float,
    points: int,
    channels: list[int] | None = None,
    block_frames: int = DEFAULT_BLOCK_FRAMES,
) -> PeakResult:
    """Reduce one range into ``points`` buckets for each selected channel."""
    if points < 1:
        raise SourceRangeError("points must be positive")
    first, last = reader.frame_bounds(start_seconds, end_seconds)
    total = last - first
    indices = list(range(reader.channel_count)) if channels is None else list(channels)
    for channel in indices:
        if not 0 <= channel < reader.channel_count:
            raise SourceRangeError(f"source channel {channel} is out of range")

    # Keep one block below a fixed byte budget even for wide files.
    block_frames = max(1, min(block_frames, (8 * 1024 * 1024) // reader.info.block_align))
    minima = np.full((points, len(indices)), np.inf, dtype=np.float32)
    maxima = np.full((points, len(indices)), -np.inf, dtype=np.float32)
    position = first
    while position < last:
        take = min(block_frames, last - position)
        block = reader.read_frames(position, take)
        if block.shape[0] == 0:
            break
        block = block[:, indices]
        offsets = np.arange(position - first, position - first + block.shape[0], dtype=np.int64)
        ids = _bucket_ids(offsets, points, total)
        # Segment the block at bucket changes, then reduce each segment once.
        cuts = np.flatnonzero(np.diff(ids)) + 1
        starts = np.concatenate(([0], cuts))
        segment_ids = ids[starts]
        block_min = np.fmin.reduceat(block, starts, axis=0)
        block_max = np.fmax.reduceat(block, starts, axis=0)
        minima[segment_ids] = np.fmin(minima[segment_ids], block_min)
        maxima[segment_ids] = np.fmax(maxima[segment_ids], block_max)
        position += block.shape[0]

    empty = np.flatnonzero(~np.isfinite(minima[:, 0]))
    if empty.size:
        # Only reachable when the range holds fewer frames than buckets: each
        # such bucket then carries the sample living at its start offset.
        span = reader.read_frames(first, total)[:, indices]
        for bucket in empty:
            frame = min(total - 1, (int(bucket) * total) // points)
            minima[bucket, :] = span[frame, :]
            maxima[bucket, :] = span[frame, :]

    clamped = int(np.count_nonzero(np.abs(minima) > 1.0) + np.count_nonzero(np.abs(maxima) > 1.0))
    # The wire contract bounds peaks to [-1, 1]; float WAVs can exceed it.
    minima = np.clip(minima, -1.0, 1.0)
    maxima = np.clip(maxima, -1.0, 1.0)
    return PeakResult(
        channel_indices=indices,
        minima=minima,
        maxima=maxima,
        frames=total,
        clamped_frames=clamped,
        empty_buckets=int(empty.size),
    )


class WavWaveformProvider:
    """``WaveformProvider`` implementation over a fixed local manifest."""

    def __init__(self, catalog: ManifestCatalog) -> None:
        self.catalog = catalog

    def waveform(self, request: WaveformRequest) -> WaveformData:
        asset = self.catalog.resolve(request.asset, verify=True).asset
        try:
            validate_source_range(asset, request.source_range)
        except ValueError as exc:
            raise SourceRangeError(str(exc)) from exc

        path: Path = self.catalog.path_for(asset)
        with open_wav(path) as reader:
            try:
                result = compute_peaks(
                    reader,
                    start_seconds=request.source_range.start_seconds,
                    end_seconds=request.source_range.end_seconds,
                    points=request.points,
                    channels=request.source_range.channels,
                )
            except AudioError:
                raise
            except (ValueError, OSError) as exc:
                raise SourceRangeError(f"cannot read the requested range: {exc}") from exc

        peaks = [
            [
                Peak(
                    minimum=float(result.minima[bucket, channel]),
                    maximum=float(result.maxima[bucket, channel]),
                )
                for bucket in range(request.points)
            ]
            for channel in range(len(result.channel_indices))
        ]
        return WaveformData(
            asset=request.asset,
            source_range=request.source_range,
            data_kind="local",
            status="ready",
            points=request.points,
            channel_indices=result.channel_indices,
            peaks=peaks,
            message=_describe(result, request.points),
        )


def _describe(result: PeakResult, points: int) -> str:
    parts = [f"decoded {result.frames} frames inside the requested source range"]
    if result.empty_buckets:
        parts.append(
            f"{result.empty_buckets} buckets had no whole frame and hold the sample at their start"
        )
    if result.clamped_frames:
        parts.append(f"{result.clamped_frames} peak values were clamped to [-1, 1]")
    if result.frames < points * _MIN_FRAMES_PER_BUCKET_HINT:
        parts.append(f"range is short: fewer than {_MIN_FRAMES_PER_BUCKET_HINT} frames per bucket")
    return "; ".join(parts)


__all__ = ["PeakResult", "WavWaveformProvider", "compute_peaks"]
