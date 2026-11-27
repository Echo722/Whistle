"""Behavioral tests for #013 video scanning, audio-track waveforms, filtering, and Inspector preview."""

from __future__ import annotations

import os
import time
import wave
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from audiosearch.audioio.decode import open_audio_file
from audiosearch.audioio.errors import UnsupportedFormatError
from audiosearch.audioio.video import is_video_path, probe_video
from audiosearch.browser.main_window import BrowserWindow, create_app
from audiosearch.browser.settings import load_settings
from audiosearch.library.metadata_writer import SUPPORTED_WRITE_EXTS
from audiosearch.library.peaks import compute_file_peaks, compute_thumbnail_with_spectral
from audiosearch.library.scanner import LibraryScanner
from audiosearch.library.store import LibraryStore, QueryFilter

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _write_wav(path: Path, seconds: float = 0.4, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(sr * seconds)
    t = np.arange(n, dtype=np.float64) / sr
    tone = (0.4 * np.sin(2.0 * np.pi * 440.0 * t) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(tone.tobytes())


def _write_test_mp4(
    path: Path,
    *,
    width: int = 320,
    height: int = 180,
    fps: int = 15,
    duration_sec: float = 1.0,
    with_audio: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    try:
        vstream = container.add_stream("mpeg4", rate=fps)
        vstream.width = width
        vstream.height = height
        vstream.pix_fmt = "yuv420p"

        astream = None
        if with_audio:
            astream = container.add_stream("aac", rate=44100)
            astream.layout = "stereo"

        n_frames = max(1, int(round(fps * duration_sec)))
        for idx in range(n_frames):
            rgb = np.zeros((height, width, 3), dtype=np.uint8)
            rgb[:, :, 0] = (idx * 25) % 255
            rgb[:, :, 1] = 90
            rgb[:, :, 2] = 200
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for pkt in vstream.encode(frame):
                container.mux(pkt)
        for pkt in vstream.encode():
            container.mux(pkt)

        if astream is not None:
            total_samples = int(44100 * duration_sec)
            chunk_sz = 1024
            cursor = 0
            while cursor < total_samples:
                take = min(chunk_sz, total_samples - cursor)
                t = (np.arange(cursor, cursor + take, dtype=np.float64)) / 44100.0
                wave_f = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
                stereo = np.stack([wave_f, wave_f], axis=0)
                aframe = av.AudioFrame.from_ndarray(stereo, format="fltp", layout="stereo")
                aframe.sample_rate = 44100
                aframe.pts = cursor
                aframe.time_base = Fraction(1, 44100)
                for pkt in astream.encode(aframe):
                    container.mux(pkt)
                cursor += take
            for pkt in astream.encode():
                container.mux(pkt)
    finally:
        container.close()


class _FakeAudition(QObject):
    positionChanged = Signal(int, float)
    stateChanged = Signal(str)
    failed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.output_rate_hz = 48000
        self.device_error = None
        self.state = "empty"
        self.can_resume_buffered = False
        self.position_seconds = 0.0
        self.current_file_id: int | None = None
        self.play_calls: list[tuple[int, str, float]] = []

    def set_volume_db(self, _db: float) -> None:
        pass

    def set_loop(self, _enabled: bool) -> None:
        pass

    def set_selection(self, _sel) -> None:
        pass

    def set_pitch_rate(self, _pitch: float, _rate: float) -> None:
        pass

    def set_preserve_pitch(self, _enabled: bool) -> None:
        pass

    def play_file(
        self, file_id: int, path: Path, start_seconds: float = 0.0, *, end_seconds=None
    ) -> None:
        self.current_file_id = file_id
        self.play_calls.append((file_id, path.name, float(start_seconds)))
        self.state = "playing"
        self.can_resume_buffered = True
        self.position_seconds = start_seconds
        self.stateChanged.emit("playing")

    def toggle_pause_resume(self) -> None:
        self.state = "paused" if self.state == "playing" else "playing"
        self.stateChanged.emit(self.state)

    def stop(self) -> None:
        self.can_resume_buffered = False
        self.state = "paused" if self.current_file_id is not None else "empty"
        self.stateChanged.emit(self.state)

    def close(self) -> None:
        pass


class _FakeSpotClient:
    def status(self):
        return type(
            "S",
            (),
            {"connected": False, "track_name": "", "project_name": "", "cursor_sec": 0.0},
        )()


def _wait_until(cond, timeout_sec: float = 4.0) -> None:
    app = QApplication.instance()
    assert app is not None
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        app.processEvents()
        if cond():
            return
        time.sleep(0.015)
    assert cond(), "Timed out waiting for condition"


def test_video_probe_audio_decode_peaks_scanner_and_filters(tmp_path: Path) -> None:
    media_dir = tmp_path / "media"
    wav_path = media_dir / "sfx.wav"
    av_mp4 = media_dir / "skill_with_audio.mp4"
    silent_mp4 = media_dir / "silent_portrait.mp4"

    _write_wav(wav_path)
    _write_test_mp4(av_mp4, width=320, height=180, fps=15, duration_sec=0.9, with_audio=True)
    _write_test_mp4(silent_mp4, width=180, height=320, fps=15, duration_sec=0.9, with_audio=False)

    assert is_video_path(av_mp4) is True
    assert is_video_path(wav_path) is False
    assert "mp4" not in SUPPORTED_WRITE_EXTS

    info_av = probe_video(av_mp4)
    assert info_av.width == 320 and info_av.height == 180
    assert info_av.fps > 10.0 and info_av.duration_sec > 0.5
    assert info_av.has_audio is True and info_av.audio_sample_rate == 44100

    info_silent = probe_video(silent_mp4)
    assert info_silent.width == 180 and info_silent.height == 320
    assert info_silent.has_audio is False

    # Audio reader & waveform extraction on MP4 with audio
    with open_audio_file(av_mp4) as reader:
        assert reader.samplerate == 44100
        assert reader.channels == 2
        pcm = reader.read(4096)
        assert pcm.shape == (4096, 2)
        assert float(np.max(np.abs(pcm))) > 0.05
        reader.seek(22050)
        pcm2 = reader.read(1024)
        assert pcm2.shape[0] > 0

    with pytest.raises(UnsupportedFormatError):
        open_audio_file(silent_mp4)

    peaks = compute_file_peaks(av_mp4, points=128)
    assert peaks.peaks.shape == (2, 128, 2)
    assert float(np.max(np.abs(peaks.peaks))) > 0.05
    thumb, spec = compute_thumbnail_with_spectral(av_mp4)
    assert len(thumb) > 0 and len(spec) > 0

    assert compute_thumbnail_with_spectral(silent_mp4) == (b"", b"")

    # Scanner include_videos=False vs include_videos=True + media filters
    db_no_vid = tmp_path / "no_vid.sqlite3"
    s1 = LibraryStore(db_no_vid)
    r1 = s1.add_root(media_dir)
    s1.close()
    LibraryScanner(db_no_vid, include_videos=False).scan_root(r1.id)
    s1 = LibraryStore(db_no_vid)
    try:
        assert s1.total_file_count() == 1
    finally:
        s1.close()

    db_with_vid = tmp_path / "with_vid.sqlite3"
    s2 = LibraryStore(db_with_vid)
    r2 = s2.add_root(media_dir)
    s2.close()
    LibraryScanner(db_with_vid, include_videos=True).scan_root(r2.id)
    s2 = LibraryStore(db_with_vid)
    try:
        assert s2.total_file_count() == 3
        assert s2.query_count(filters=QueryFilter(ext="__audio__")) == 1
        assert s2.query_count(filters=QueryFilter(ext="__video__")) == 2
        assert s2.query_count(filters=QueryFilter(ext="mp4")) == 2
        silent_rec = next(
            r for r in s2.query_page(offset=0, limit=10) if r.filename == "silent_portrait.mp4"
        )
        assert silent_rec.sample_rate == 0
        fields = dict((k, v) for k, v, _ in s2.get_metadata_fields(silent_rec.id))
        assert fields["video_resolution"] == "180x320"
        assert fields["audio_codec"] == "无音轨"
    finally:
        s2.close()


def test_inspector_embedded_video_preview_silent_clock_and_settings(tmp_path: Path) -> None:
    app = create_app([])
    media_dir = tmp_path / "ui_media"
    _write_wav(media_dir / "a_audio.wav")
    _write_test_mp4(
        media_dir / "b_video_av.mp4", width=320, height=180, fps=15, duration_sec=0.8, with_audio=True
    )
    _write_test_mp4(
        media_dir / "c_video_silent.mp4",
        width=180,
        height=320,
        fps=15,
        duration_sec=1.2,
        with_audio=False,
    )

    db_path = tmp_path / "ui_video.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(media_dir)
    store.close()
    LibraryScanner(db_path, include_videos=True).scan_root(root.id)

    aud = _FakeAudition()
    win = BrowserWindow(db_path, audition=aud, spot_client=_FakeSpotClient())
    win.show()
    _wait_until(lambda: win.table_model.rowCount() == 3)
    try:
        # Row 0: WAV -> cover stack index 0
        win.autoplay_btn.setChecked(False)
        win._on_audition_row(0)
        assert win.inspector.media_stack.currentIndex() == 0

        # Row 1: Video with audio -> Inspector switches to video canvas & decodes poster frame
        win._on_audition_row(1)
        assert win.inspector.media_stack.currentIndex() == 1
        assert win.inspector.edit_meta_btn.isEnabled() is False
        _wait_until(lambda: win.inspector.video_canvas.has_image())
        assert "320×180" in win.inspector.video_canvas._meta_text

        # Audio-master position signal updates video canvas playback state & frame sync
        aud.state = "playing"
        aud.stateChanged.emit("playing")
        aud.positionChanged.emit(win._current_file.id, 0.35)
        app.processEvents()
        assert win.inspector.video_canvas._playing is True

        # Row 2: Silent portrait video -> shows "无音轨", plays/pauses/stops via wall clock
        win._on_audition_row(2)
        assert win.inspector.media_stack.currentIndex() == 1
        _wait_until(lambda: win.inspector.video_canvas.has_image())
        assert "无音轨" in win.inspector._pill_sr.text()
        assert "无音轨" in win.inspector.video_canvas._meta_text

        win._on_play_button()
        assert win._silent_video_state == "playing"
        assert bool(win.play_btn.property("playing")) is True
        _wait_until(lambda: win._silent_video_pos > 0.08, timeout_sec=1.5)

        # Pause with Space, stop with Esc
        win._on_play_button()
        assert win._silent_video_state == "paused"
        assert bool(win.play_btn.property("playing")) is False

        esc_ev = QKeyEvent(
            QKeyEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier
        )
        assert win._handle_key_press(esc_ev) is True
        assert win._silent_video_pos == 0.0

        # Switching back to WAV restores cover card & metadata editability
        win._on_audition_row(0)
        assert win.inspector.media_stack.currentIndex() == 0
        assert win.inspector.edit_meta_btn.isEnabled() is True

        # Filter combo "仅视频" vs "仅音频"
        vid_idx = win.filter_ext_combo.findData("__video__")
        assert vid_idx >= 0
        win.filter_ext_combo.setCurrentIndex(vid_idx)
        _wait_until(lambda: win.table_model.rowCount() == 2)

        aud_idx = win.filter_ext_combo.findData("__audio__")
        assert aud_idx >= 0
        win.filter_ext_combo.setCurrentIndex(aud_idx)
        _wait_until(lambda: win.table_model.rowCount() == 1)

        # Settings toggle index_video_files persists cleanly
        win.set_index_video_files(False)
    finally:
        win.close()
        app.processEvents()

    saved = load_settings(db_path.parent / "browser-settings.json")
    assert saved.index_video_files is False
