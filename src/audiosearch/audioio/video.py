"""Read-only video probing, audio-track decoding, and frame extraction powered by PyAV."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import DecodeError, UnsupportedFormatError

VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"})
VIDEO_EXT_NAMES = frozenset(ext.lstrip(".") for ext in VIDEO_EXTENSIONS)


def is_video_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def is_video_ext(ext: str | None) -> bool:
    return str(ext or "").strip().lower().lstrip(".") in VIDEO_EXT_NAMES


@dataclass(frozen=True, slots=True)
class VideoProbeInfo:
    """Technical metadata and optional container tags probed from a video file."""

    duration_sec: float
    width: int
    height: int
    fps: float
    video_codec: str
    has_audio: bool
    audio_sample_rate: int
    audio_channels: int
    audio_codec: str
    title: str
    comment: str
    tags: dict[str, str]


def _import_av():
    try:
        import av

        return av
    except ImportError as exc:
        raise UnsupportedFormatError("未安装 PyAV 视频解码组件，无法读取视频文件") from exc


def _stream_duration_sec(container: Any, stream: Any, av: Any) -> float:
    if getattr(container, "duration", None):
        dur = float(container.duration) / float(av.time_base)
        if dur > 0.0:
            return dur
    if stream is not None and getattr(stream, "duration", None) and getattr(stream, "time_base", None):
        dur = float(stream.duration * stream.time_base)
        if dur > 0.0:
            return dur
    return 0.0


def probe_video(path: str | Path) -> VideoProbeInfo:
    """Inspect video resolution, fps, audio stream specs, and container tags without modifying disk."""
    av = _import_av()
    file_path = Path(path)
    try:
        container = av.open(str(file_path), mode="r")
    except Exception as exc:
        raise DecodeError(f"无法打开视频文件：{file_path.name}") from exc

    try:
        vstream = container.streams.video[0] if container.streams.video else None
        astream = container.streams.audio[0] if container.streams.audio else None
        duration = _stream_duration_sec(container, vstream or astream, av)

        width = int(getattr(vstream, "width", 0) or 0)
        height = int(getattr(vstream, "height", 0) or 0)
        fps = 0.0
        if vstream is not None:
            rate = getattr(vstream, "average_rate", None) or getattr(vstream, "guessed_rate", None)
            if rate is not None:
                try:
                    fps = float(rate)
                except Exception:
                    fps = 0.0
        video_codec = (
            str(getattr(vstream.codec_context, "name", "") or "")
            if vstream is not None
            else ""
        )

        has_audio = astream is not None
        audio_sr = (
            int(getattr(astream.codec_context, "sample_rate", 0) or 0)
            if astream is not None
            else 0
        )
        audio_ch = (
            int(getattr(astream.codec_context, "channels", 0) or 0)
            if astream is not None
            else 0
        )
        audio_codec = (
            str(getattr(astream.codec_context, "name", "") or "")
            if astream is not None
            else ""
        )

        raw_tags = dict(getattr(container, "metadata", {}) or {})
        tags = {str(k): str(v) for k, v in raw_tags.items() if v is not None and str(v).strip()}
        tags_lc = {k.lower(): v.strip() for k, v in tags.items()}
        title = tags_lc.get("title", "")
        comment = tags_lc.get("description") or tags_lc.get("comment") or ""

        return VideoProbeInfo(
            duration_sec=duration,
            width=width,
            height=height,
            fps=fps,
            video_codec=video_codec,
            has_audio=has_audio,
            audio_sample_rate=audio_sr,
            audio_channels=audio_ch,
            audio_codec=audio_codec,
            title=title,
            comment=comment,
            tags=tags,
        )
    finally:
        container.close()


class PyAvAudioReader:
    """Read-only SoundFile-compatible audio stream facade for video containers."""

    def __init__(self, path: str | Path) -> None:
        av = _import_av()
        self._av = av
        self._path = Path(path)
        try:
            self._container = av.open(str(self._path), mode="r")
        except Exception as exc:
            raise DecodeError(f"无法打开视频音轨：{self._path.name}") from exc

        if not self._container.streams.audio:
            self._container.close()
            raise UnsupportedFormatError("视频文件不含音轨")

        self._astream = self._container.streams.audio[0]
        self._samplerate = max(1, int(getattr(self._astream.codec_context, "sample_rate", 0) or 48000))
        self._channels = max(1, int(getattr(self._astream.codec_context, "channels", 0) or 2))
        self._subtype = str(getattr(self._astream.codec_context, "name", "") or "AAC").upper()
        dur_sec = _stream_duration_sec(self._container, self._astream, av)
        self._frames = max(0, int(round(dur_sec * self._samplerate)))

        self._resampler = self._make_resampler()
        self._packet_iter = self._container.demux(self._astream)
        self._queue: deque[np.ndarray] = deque()
        self._queued_frames = 0
        self._pos = 0
        self._eof = False
        self._trim_until_sec: float | None = None

    def _make_resampler(self):
        return self._av.AudioResampler(
            format="fltp",
            layout=self._astream.layout,
            rate=self._samplerate,
        )

    @property
    def samplerate(self) -> int:
        return self._samplerate

    @property
    def frames(self) -> int:
        return self._frames

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def subtype(self) -> str:
        return self._subtype

    def __len__(self) -> int:
        return self._frames

    def seekable(self) -> bool:
        return True

    def seek(self, frame: int) -> int:
        target = max(0, int(frame))
        if self._frames > 0:
            target = min(target, self._frames)
        self._pos = target
        self._queue.clear()
        self._queued_frames = 0
        self._eof = False
        target_sec = target / float(self._samplerate)
        self._trim_until_sec = target_sec if target > 0 else None
        self._resampler = self._make_resampler()
        try:
            self._container.seek(
                int(target_sec * self._av.time_base),
                any_frame=False,
                backward=True,
            )
            try:
                self._astream.codec_context.flush_buffers()
            except Exception:
                pass
            self._packet_iter = self._container.demux(self._astream)
        except Exception as exc:
            self._eof = True
            raise DecodeError(f"视频音轨定位失败：{exc}") from exc
        return self._pos

    def _convert_frame(self, aframe: Any) -> np.ndarray:
        resampled = self._resampler.resample(aframe)
        if not resampled:
            return np.zeros((0, self._channels), dtype=np.float32)
        if not isinstance(resampled, list):
            resampled = [resampled]
        parts: list[np.ndarray] = []
        for rf in resampled:
            nd = rf.to_ndarray()
            if nd.size == 0:
                continue
            if getattr(rf.format, "is_planar", False):
                arr = np.ascontiguousarray(nd.T, dtype=np.float32)
            elif nd.ndim == 2:
                arr = np.ascontiguousarray(nd.reshape(-1, self._channels), dtype=np.float32)
            else:
                arr = np.ascontiguousarray(nd.reshape(-1, self._channels), dtype=np.float32)
            if self._trim_until_sec is not None:
                pts_val = getattr(rf, "pts", None)
                tb = getattr(rf, "time_base", None) or getattr(self._astream, "time_base", None)
                if pts_val is not None and tb is not None:
                    f_start = float(pts_val * tb)
                    f_end = f_start + arr.shape[0] / float(self._samplerate)
                    if f_end <= self._trim_until_sec:
                        continue
                    if f_start < self._trim_until_sec:
                        drop = min(
                            arr.shape[0],
                            max(0, int(round((self._trim_until_sec - f_start) * self._samplerate))),
                        )
                        arr = arr[drop:]
                    self._trim_until_sec = None
                else:
                    self._trim_until_sec = None
            if arr.shape[0] > 0:
                parts.append(arr)
        if not parts:
            return np.zeros((0, self._channels), dtype=np.float32)
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)

    def _fill_until(self, needed: int) -> None:
        while self._queued_frames < needed and not self._eof:
            try:
                packet = next(self._packet_iter)
            except StopIteration:
                # Flush buffered resampler tail.
                tail = self._resampler.resample(None)
                if tail:
                    items = tail if isinstance(tail, list) else [tail]
                    for rf in items:
                        nd = rf.to_ndarray()
                        if nd.size > 0:
                            arr = (
                                np.ascontiguousarray(nd.T, dtype=np.float32)
                                if getattr(rf.format, "is_planar", False)
                                else np.ascontiguousarray(nd.reshape(-1, self._channels), dtype=np.float32)
                            )
                            if arr.shape[0] > 0:
                                self._queue.append(arr)
                                self._queued_frames += int(arr.shape[0])
                self._eof = True
                break

            if getattr(packet, "dts", None) is None and getattr(packet, "size", 1) == 0:
                continue
            try:
                decoded = packet.decode()
            except Exception:
                continue
            for frame in decoded:
                arr = self._convert_frame(frame)
                if arr.shape[0] > 0:
                    self._queue.append(arr)
                    self._queued_frames += int(arr.shape[0])

    def read(
        self,
        frames: int = -1,
        dtype: str = "float32",
        always_2d: bool = True,
    ) -> np.ndarray:
        if frames < 0:
            self._fill_until(1 << 30)
            want = self._queued_frames
        else:
            want = max(0, int(frames))
            if want > 0:
                self._fill_until(want)

        if want <= 0 or self._queued_frames <= 0:
            out = np.zeros((0, self._channels), dtype=dtype)
            return out[:, 0] if (not always_2d and self._channels == 1) else out

        take = min(want, self._queued_frames)
        chunks: list[np.ndarray] = []
        rem = take
        while rem > 0 and self._queue:
            head = self._queue[0]
            avail = int(head.shape[0])
            if avail <= rem:
                chunks.append(self._queue.popleft())
                self._queued_frames -= avail
                rem -= avail
            else:
                chunks.append(head[:rem])
                self._queue[0] = head[rem:]
                self._queued_frames -= rem
                rem = 0

        arr = chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)
        self._pos += int(arr.shape[0])
        if dtype != "float32":
            arr = arr.astype(dtype, copy=False)
        if not always_2d and self._channels == 1:
            return arr[:, 0]
        return np.ascontiguousarray(arr)

    def close(self) -> None:
        try:
            self._container.close()
        except Exception:
            pass

    def __enter__(self) -> PyAvAudioReader:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


def open_video_audio(path: str | Path) -> PyAvAudioReader:
    """Open the first audio stream inside a video file for streaming PCM reads."""
    return PyAvAudioReader(path)


def fit_even_size(src_w: int, src_h: int, max_w: int, max_h: int) -> tuple[int, int]:
    """Scale ``(src_w, src_h)`` proportionally within ``(max_w, max_h)``, rounding to even >= 2."""
    sw = max(2, int(src_w))
    sh = max(2, int(src_h))
    mw = max(2, int(max_w))
    mh = max(2, int(max_h))
    ratio = min(float(mw) / float(sw), float(mh) / float(sh), 1.0)
    out_w = max(2, int(round(sw * ratio)) & ~1)
    out_h = max(2, int(round(sh * ratio)) & ~1)
    return min(mw, out_w), min(mh, out_h)


def frame_to_rgb24(frame: Any, max_w: int, max_h: int) -> tuple[bytes, int, int]:
    """Convert an ``av.VideoFrame`` to tightly packed RGB24 bytes scaled inside ``(max_w, max_h)``."""
    out_w, out_h = fit_even_size(frame.width, frame.height, max_w, max_h)
    rgb = frame.reformat(width=out_w, height=out_h, format="rgb24")
    plane = bytes(rgb.planes[0])
    stride = rgb.planes[0].line_size
    expected_stride = out_w * 3
    if stride == expected_stride:
        return plane[: out_h * expected_stride], out_w, out_h
    arr = np.frombuffer(plane, dtype=np.uint8).reshape((out_h, stride))[:, :expected_stride]
    return np.ascontiguousarray(arr).tobytes(), out_w, out_h


__all__ = [
    "PyAvAudioReader",
    "VIDEO_EXTENSIONS",
    "VIDEO_EXT_NAMES",
    "VideoProbeInfo",
    "fit_even_size",
    "frame_to_rgb24",
    "is_video_ext",
    "is_video_path",
    "open_video_audio",
    "probe_video",
]
