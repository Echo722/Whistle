"""Local file audition controller backed by StreamingPlayer + persistent WASAPI sink."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal, Slot

from audiosearch.audition.sink import (
    OutputDeviceInfo,
    SoundDeviceSink,
    list_output_devices,
    resolve_output_device,
)
from audiosearch.audition.stream import StreamingPlayer


class _StreamBridge(QObject):
    ready = Signal(int, float)
    failed = Signal(int, str)
    finished = Signal(int)


class AuditionController(QObject):
    """Play / pause / resume / seek / loop / volume / pitch / rate from local files."""

    positionChanged = Signal(int, float)  # (file_id, seconds)
    stateChanged = Signal(str)  # empty | playing | paused | finished
    failed = Signal(str)

    def __init__(
        self,
        *,
        rate: int | None = None,
        device_name: str | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._requested_device_name = str(device_name or "").strip()
        dev_idx, active_name, resolved_rate, matched = resolve_output_device(
            self._requested_device_name
        )
        self._rate = int(rate) if rate is not None else int(resolved_rate)
        self._active_device_name = active_name
        self._device_fallback = bool(self._requested_device_name and not matched)
        self._sink = SoundDeviceSink(self._rate, device=dev_idx)
        self._player = StreamingPlayer(self._sink, self._rate)
        self._bridge = _StreamBridge(self)
        self._bridge.ready.connect(self._on_ready)
        self._bridge.failed.connect(self._on_failed)
        self._bridge.finished.connect(self._on_finished)

        self._closed = False
        self._file_id: int | None = None
        self._current_path: Path | None = None
        self._active_gen = 0

        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)

    @property
    def output_rate_hz(self) -> int:
        return self._rate

    @property
    def requested_device_name(self) -> str:
        return self._requested_device_name

    @property
    def active_device_name(self) -> str:
        return self._active_device_name

    @property
    def device_fallback(self) -> bool:
        return self._device_fallback

    def list_output_devices(self) -> list[OutputDeviceInfo]:
        return list_output_devices()

    def set_output_device(
        self, device_name: str | None, *, refresh_portaudio: bool = False
    ) -> tuple[bool, str]:
        """Switch audio output endpoint live and update target sample rate."""
        requested = str(device_name or "").strip()
        self.stop()
        self._sink.close()
        if refresh_portaudio:
            try:
                import sounddevice as sd

                sd._terminate()
                sd._initialize()
            except Exception:
                pass
        dev_idx, active_name, resolved_rate, matched = resolve_output_device(requested)
        self._requested_device_name = requested
        self._active_device_name = active_name
        self._device_fallback = bool(requested and not matched)
        self._rate = int(resolved_rate)
        self._player.set_target_rate(self._rate)
        self._sink.reconfigure(self._rate, device=dev_idx)
        self._sink.available()
        return matched, active_name

    @property
    def device_error(self) -> str | None:
        return self._sink.device_error

    @property
    def current_file_id(self) -> int | None:
        return self._file_id

    @property
    def state(self) -> str:
        return self._player.state

    @property
    def position_seconds(self) -> float:
        return self._player.position_seconds

    @property
    def pitch_semitones(self) -> float:
        return self._player.pitch_semitones

    @property
    def rate(self) -> float:
        return self._player.rate

    @property
    def preserve_pitch(self) -> bool:
        return self._player.preserve_pitch

    @property
    def can_resume_buffered(self) -> bool:
        return self._player.state == "paused" and self._player.has_buffered_audio

    def device_available(self) -> bool:
        return self._sink.available()

    def set_volume_db(self, db: float) -> None:
        gain = 0.0 if db <= -48.0 else 10.0 ** (db / 20.0)
        self._player.set_volume_gain(gain)

    def set_loop(self, enabled: bool) -> None:
        self._player.set_loop(enabled)

    def set_selection(self, selection: tuple[float, float] | None) -> None:
        self._player.set_selection(selection)

    def set_pitch_rate(self, pitch_semitones: float, rate: float) -> None:
        self._player.set_pitch_rate(pitch_semitones, rate)

    def set_preserve_pitch(self, enabled: bool) -> None:
        self._player.set_preserve_pitch(enabled)

    def _emit_ready(self, gen: int, start_sec: float) -> None:
        try:
            self._bridge.ready.emit(gen, start_sec)
        except RuntimeError:
            pass

    def _emit_failed(self, gen: int, message: str) -> None:
        try:
            self._bridge.failed.emit(gen, message)
        except RuntimeError:
            pass

    def _emit_finished(self, gen: int) -> None:
        try:
            self._bridge.finished.emit(gen)
        except RuntimeError:
            pass

    def play_file(
        self,
        file_id: int,
        path: Path,
        start_seconds: float = 0.0,
        *,
        end_seconds: float | None = None,
    ) -> None:
        if self._closed:
            return
        self._file_id = file_id
        self._current_path = path
        self._timer.stop()
        self._active_gen = self._player.load_and_play(
            path,
            start_seconds,
            end_seconds=end_seconds,
            on_ready=self._emit_ready,
            on_error=self._emit_failed,
            on_finished=self._emit_finished,
        )

    def toggle_pause_resume(self) -> None:
        if self._closed or self._file_id is None:
            return
        state = self._player.state
        if state == "playing":
            self._player.pause()
            self._timer.stop()
            self.stateChanged.emit("paused")
            self.positionChanged.emit(self._file_id, self._player.position_seconds)
            return

        if state == "paused" and self._player.has_buffered_audio:
            resumed = self._player.resume(
                on_ready=self._emit_ready,
                on_error=self._emit_failed,
                on_finished=self._emit_finished,
            )
            if resumed:
                if self._sink.device_error:
                    self.failed.emit(self._sink.device_error)
                    return
                self._timer.start()
                self.stateChanged.emit("playing")
                self.positionChanged.emit(self._file_id, self._player.position_seconds)
            return

        if self._current_path is not None:
            sel = self._player.selection
            start = sel[0] if sel is not None else 0.0
            end = sel[1] if sel is not None else None
            self.play_file(self._file_id, self._current_path, start, end_seconds=end)

    def stop(self) -> None:
        self._active_gen += 1
        self._player.stop()
        self._timer.stop()
        if self._file_id is not None:
            self.positionChanged.emit(self._file_id, 0.0)
        self.stateChanged.emit(self._player.state)

    def close(self) -> None:
        self._closed = True
        self._timer.stop()
        self._player.close()

    @Slot(int, float)
    def _on_ready(self, gen: int, start_sec: float) -> None:
        if self._closed or gen != self._active_gen or self._file_id is None:
            return
        if self._sink.device_error:
            self._timer.stop()
            self.stateChanged.emit("paused")
            self.failed.emit(self._sink.device_error)
            return
        self._timer.start()
        self.stateChanged.emit("playing")
        self.positionChanged.emit(self._file_id, start_sec)

    @Slot(int, str)
    def _on_failed(self, gen: int, message: str) -> None:
        if self._closed or gen != self._active_gen:
            return
        self._timer.stop()
        self.stateChanged.emit("paused")
        self.failed.emit(message)

    @Slot(int)
    def _on_finished(self, gen: int) -> None:
        if self._closed or gen != self._active_gen:
            return
        self._timer.stop()
        self._sink.stop()
        self.stateChanged.emit("finished")

    def _tick(self) -> None:
        if self._file_id is None:
            self._timer.stop()
            return
        pos = self._player.position_seconds
        self.positionChanged.emit(self._file_id, pos)
        if self._player.state != "playing":
            self._timer.stop()
            self.stateChanged.emit(self._player.state)
