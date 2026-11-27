"""Persistent WASAPI/PortAudio output sink for local audition.

Why one persistent stereo stream instead of opening/closing per clip:
repeatedly tearing down and re-creating Windows audio streams (especially across
mono/stereo channel-count switches) stalls drivers and can leave a device
silently muted while the callback keeps advancing frames. Keeping one stream
open at the device's own sample rate and switching the pull source to ``None``
on pause/stop gives instant, glitch-free switching.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# Large enough to ride out Qt repaints and Python GIL pauses without xruns,
# while staying ~42 ms at 48 kHz for crisp play/pause responsiveness.
DEFAULT_BLOCKSIZE = 2048


@dataclass(frozen=True, slots=True)
class OutputDeviceInfo:
    """Selectable local audio output endpoint metadata."""

    index: int
    name: str
    hostapi_name: str
    default_samplerate: int
    max_output_channels: int
    is_default: bool = False


def list_output_devices() -> list[OutputDeviceInfo]:
    """Return available output devices, preferring Windows WASAPI shared endpoints."""
    try:
        import sounddevice as sd

        hostapis = list(sd.query_hostapis())
        devices = list(sd.query_devices())
    except Exception:
        return []

    target_host_idx: int | None = None
    for idx, host in enumerate(hostapis):
        if "wasapi" in str(host.get("name", "")).lower():
            target_host_idx = idx
            break
    if target_host_idx is None:
        try:
            target_host_idx = int(sd.default.hostapi)
        except Exception:
            target_host_idx = 0 if hostapis else None

    default_out_idx = -1
    if target_host_idx is not None and 0 <= target_host_idx < len(hostapis):
        default_out_idx = int(hostapis[target_host_idx].get("default_output_device", -1))

    collected: dict[str, OutputDeviceInfo] = {}
    for dev_idx, dev in enumerate(devices):
        max_ch = int(dev.get("max_output_channels", 0) or 0)
        if max_ch < 1:
            continue
        dev_host = int(dev.get("hostapi", -1))
        if target_host_idx is not None and dev_host != target_host_idx:
            continue
        name = str(dev.get("name", "") or f"Output #{dev_idx}").strip()
        host_name = (
            str(hostapis[dev_host].get("name", ""))
            if 0 <= dev_host < len(hostapis)
            else ""
        )
        raw_rate = int(round(float(dev.get("default_samplerate") or 48000)))
        rate = raw_rate if raw_rate > 0 else 48000
        is_def = dev_idx == default_out_idx
        info = OutputDeviceInfo(
            index=dev_idx,
            name=name,
            hostapi_name=host_name,
            default_samplerate=rate,
            max_output_channels=max_ch,
            is_default=is_def,
        )
        prev = collected.get(name)
        if prev is None or (info.is_default and not prev.is_default) or (
            not prev.is_default and info.max_output_channels > prev.max_output_channels
        ):
            collected[name] = info

    if not collected and target_host_idx is not None:
        for dev_idx, dev in enumerate(devices):
            max_ch = int(dev.get("max_output_channels", 0) or 0)
            if max_ch < 1:
                continue
            dev_host = int(dev.get("hostapi", -1))
            name = str(dev.get("name", "") or f"Output #{dev_idx}").strip()
            host_name = (
                str(hostapis[dev_host].get("name", ""))
                if 0 <= dev_host < len(hostapis)
                else ""
            )
            raw_rate = int(round(float(dev.get("default_samplerate") or 48000)))
            rate = raw_rate if raw_rate > 0 else 48000
            if name not in collected:
                collected[name] = OutputDeviceInfo(
                    index=dev_idx,
                    name=name,
                    hostapi_name=host_name,
                    default_samplerate=rate,
                    max_output_channels=max_ch,
                    is_default=False,
                )

    items = list(collected.values())
    items.sort(key=lambda d: (0 if d.is_default else 1, d.index))
    return items


def resolve_output_device(
    device_name: str | None = None, *, fallback_rate: int = 48000
) -> tuple[int | None, str, int, bool]:
    """Resolve ``(device_index, resolved_name, samplerate, matched)`` for ``device_name``."""
    target = str(device_name or "").strip()
    devices = list_output_devices()
    default_dev = next((d for d in devices if d.is_default), devices[0] if devices else None)

    if target:
        for dev in devices:
            if dev.name == target:
                return dev.index, dev.name, dev.default_samplerate, True
        target_cf = target.casefold()
        for dev in devices:
            if dev.name.casefold() == target_cf:
                return dev.index, dev.name, dev.default_samplerate, True
        if default_dev is not None:
            return default_dev.index, default_dev.name, default_dev.default_samplerate, False
        idx, rate = _preferred_output_device(fallback_rate)
        return idx, "", rate, False

    if default_dev is not None:
        return default_dev.index, default_dev.name, default_dev.default_samplerate, True
    idx, rate = _preferred_output_device(fallback_rate)
    return idx, "", rate, True


def _preferred_output_device(fallback_rate: int = 48000) -> tuple[int | None, int]:
    """Return ``(device_index, default_samplerate)``, preferring WASAPI shared output."""
    try:
        import sounddevice as sd

        for host in sd.query_hostapis():
            if "wasapi" not in str(host.get("name", "")).lower():
                continue
            index = int(host.get("default_output_device", -1))
            if index < 0:
                continue
            info = sd.query_devices(index)
            if int(info.get("max_output_channels", 0)) >= 1:
                rate = int(info.get("default_samplerate") or fallback_rate)
                return index, (rate if rate > 0 else fallback_rate)
        default_info = sd.query_devices(kind="output")
        if int(default_info.get("max_output_channels", 0)) >= 1:
            rate = int(default_info.get("default_samplerate") or fallback_rate)
            return None, (rate if rate > 0 else fallback_rate)
    except Exception:
        pass
    return None, fallback_rate


def default_output_rate(fallback: int = 48000) -> int:
    """The preferred output device's own rate, avoiding runtime resampling."""
    _, rate = _preferred_output_device(fallback)
    return rate


def _close_stream(stream: object | None) -> None:
    if stream is None:
        return
    try:
        stream.stop()  # type: ignore[attr-defined]
        stream.close()  # type: ignore[attr-defined]
    except Exception:  # a stream already torn down by the device
        pass


class SoundDeviceSink:
    def __init__(
        self, sample_rate_hz: int, channel_count: int = 2, *, device: int | None = None
    ) -> None:
        self.sample_rate_hz = sample_rate_hz
        self.channel_count = max(1, min(channel_count, 2))
        self.device_error: str | None = None
        self._device = device
        self._lock = threading.Lock()
        self._stream = None
        self._pull: Callable[[int], np.ndarray] | None = None

    # ------------------------------------------------------------------ probe
    def available(self) -> bool:
        """Whether PortAudio can see an output device at all."""
        try:
            import sounddevice as sd

            devices = sd.query_devices()
        except Exception as exc:  # missing dependency, absent PortAudio
            self.device_error = f"音频输出不可用：{exc}"
            return False
        for device in devices:
            if int(device.get("max_output_channels", 0)) > 0:
                self.device_error = None
                return True
        self.device_error = "没有可用的音频输出设备"
        return False

    # ---------------------------------------------------------------- transport
    def start(self, pull: Callable[[int], np.ndarray], _channel_count: int = 2) -> None:
        with self._lock:
            self._pull = pull
            stream = self._stream
            if stream is not None and bool(getattr(stream, "active", True)):
                self.device_error = None
                return
            dead, self._stream = self._stream, None
        _close_stream(dead)

        try:
            stream = self._open_stream(self._device)
        except Exception as exc:
            with self._lock:
                self._pull = None
                self._stream = None
            self.device_error = f"无法打开音频输出：{exc}"
            return

        with self._lock:
            self._stream = stream
        self.device_error = None

    def stop(self) -> None:
        """Silence playback immediately while keeping the persistent stream open."""
        with self._lock:
            self._pull = None

    def close(self) -> None:
        """Stop and close the underlying PortAudio stream (call on window shutdown)."""
        with self._lock:
            self._pull = None
            stream, self._stream = self._stream, None
        _close_stream(stream)

    def reconfigure(self, sample_rate_hz: int, *, device: int | None = None) -> None:
        """Switch target output device / rate cleanly by closing any active stream."""
        with self._lock:
            self._pull = None
            stream, self._stream = self._stream, None
            self.sample_rate_hz = max(8000, int(sample_rate_hz))
            self._device = device
            self.device_error = None
        _close_stream(stream)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ internals
    def _open_stream(self, device: int | None):
        import sounddevice as sd

        candidates = (device, None) if device is not None else (None,)
        last_exc: Exception | None = None
        for candidate in candidates:
            try:
                stream = sd.OutputStream(
                    samplerate=self.sample_rate_hz,
                    channels=self.channel_count,
                    dtype="float32",
                    blocksize=DEFAULT_BLOCKSIZE,
                    latency="high",
                    device=candidate,
                    callback=self._stream_callback,
                )
                stream.start()
                return stream
            except Exception as exc:
                last_exc = exc
        assert last_exc is not None
        raise last_exc

    def _stream_callback(self, outdata, frames, _time, status) -> None:  # noqa: ANN001 - PortAudio API
        pull = self._pull
        if pull is None:
            outdata.fill(0.0)
            return
        self._fill_outdata(pull, outdata, frames, status)

    def _fill_outdata(self, pull: Callable[[int], np.ndarray], outdata, frames, status) -> None:  # noqa: ANN001
        block = pull(frames)
        block = np.asarray(block, dtype=np.float32)
        if block.ndim == 1:
            block = block[:, np.newaxis]
        outdata.fill(0.0)
        written = min(block.shape[0], frames)
        if written > 0:
            if block.shape[1] == 1 and outdata.shape[1] >= 2:
                mono = block[:written, 0]
                outdata[:written, 0] = mono
                outdata[:written, 1] = mono
            else:
                channels = min(block.shape[1], outdata.shape[1])
                outdata[:written, :channels] = block[:written, :channels]
        if status:  # under/overflow: audible as a gap, never fatal
            self.device_error = f"音频输出异常：{status}"

    def _callback(self, pull: Callable[[int], np.ndarray]):
        def callback(outdata, frames, _time, status) -> None:  # noqa: ANN001 - PortAudio API
            self._fill_outdata(pull, outdata, frames, status)

        return callback


__all__ = [
    "DEFAULT_BLOCKSIZE",
    "OutputDeviceInfo",
    "SoundDeviceSink",
    "_preferred_output_device",
    "default_output_rate",
    "list_output_devices",
    "resolve_output_device",
]
