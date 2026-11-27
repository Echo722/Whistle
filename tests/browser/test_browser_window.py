"""Offscreen PySide6 behavioral tests for BrowserWindow."""

from __future__ import annotations

import json
import os
import wave
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QPoint, QPointF, Qt, Signal
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import audiosearch.browser.file_table as file_table_mod
import audiosearch.browser.main_window as main_window_mod
from audiosearch.browser.file_table import COLUMNS, WAVEFORM_COL
from audiosearch.browser.main_window import BrowserWindow, create_app
from audiosearch.browser.settings import settings_path_for_db
from audiosearch.browser.waveform_delegate import WaveformItemDelegate
from audiosearch.library.scanner import LibraryScanner
from audiosearch.library.store import LibraryStore

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class FakeAuditionController(QObject):
    positionChanged = Signal(int, float)
    stateChanged = Signal(str)
    failed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.output_rate_hz = 48000
        self.device_error: str | None = None
        self.state = "empty"
        self.position_seconds = 0.0
        self.current_file_id: int | None = None
        self.play_calls: list[tuple[int, str, float]] = []
        self.toggle_calls = 0
        self.stop_calls = 0
        self.closed = False
        self.volume_db = 0.0
        self.loop = False
        self.selection: tuple[float, float] | None = None
        self.pitch_rate = (0.0, 1.0)
        self.preserve_pitch = False

    def set_volume_db(self, db: float) -> None:
        self.volume_db = db

    def set_loop(self, enabled: bool) -> None:
        self.loop = enabled

    def set_selection(self, sel: tuple[float, float] | None) -> None:
        self.selection = sel

    def set_pitch_rate(self, pitch: float, rate: float) -> None:
        self.pitch_rate = (pitch, rate)

    def set_preserve_pitch(self, enabled: bool) -> None:
        self.preserve_pitch = bool(enabled)

    def play_file(
        self,
        file_id: int,
        path: Path,
        start_seconds: float = 0.0,
        *,
        end_seconds: float | None = None,
    ) -> None:
        self.current_file_id = file_id
        self.play_calls.append((file_id, path.name, start_seconds))
        self.state = "playing"
        self.position_seconds = start_seconds
        self.stateChanged.emit("playing")

    def toggle_pause_resume(self) -> None:
        self.toggle_calls += 1
        if self.state == "playing":
            self.state = "paused"
        elif self.state == "paused":
            self.state = "playing"
        self.stateChanged.emit(self.state)

    def stop(self) -> None:
        self.stop_calls += 1
        self.state = "paused"
        self.position_seconds = 0.0
        self.stateChanged.emit("paused")

    def close(self) -> None:
        self.closed = True


def _make_wav(path: Path, frames: int = 4800) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(48000)
        wf.writeframes((np.zeros(frames, dtype="<i2")).tobytes())


def _pump(ms: int = 80) -> None:
    QTest.qWait(ms)
    QApplication.processEvents()


def _wait_until(predicate, timeout_ms: int = 2000) -> None:
    elapsed = 0
    while elapsed < timeout_ms:
        QApplication.processEvents()
        if predicate():
            return
        QTest.qWait(25)
        elapsed += 25
    QApplication.processEvents()
    assert predicate()


def test_browser_interactions_and_me_keys(tmp_path):
    create_app([])
    sfx = tmp_path / "sfx"
    _make_wav(sfx / "Alpha" / "alpha_01.wav")
    _make_wav(sfx / "Beta" / "beta_02.wav")

    db_path = tmp_path / "browser.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(sfx)
    LibraryScanner(db_path).scan_root(root.id)
    store.close()

    fake_audition = FakeAuditionController()
    win = BrowserWindow(db_path, audition=fake_audition)
    win.show()
    _wait_until(lambda: win.table_model.rowCount() == 2)

    try:
        assert win.table_model.rowCount() == 2

        # 1. Single-click row auto-plays from 0.0
        rect0 = win.table_view.visualRect(win.table_model.index(0, 0))
        QTest.mouseClick(win.table_view.viewport(), Qt.MouseButton.LeftButton, pos=rect0.center())
        _pump(80)
        assert len(fake_audition.play_calls) == 1
        assert fake_audition.play_calls[0][1] == "alpha_01.wav"
        assert fake_audition.play_calls[0][2] == 0.0

        # 2. Down arrow moves to next row and auto-plays
        QTest.keyClick(win.table_view, Qt.Key.Key_Down)
        _pump(80)
        assert len(fake_audition.play_calls) == 2
        assert fake_audition.play_calls[1][1] == "beta_02.wav"

        # 3. Left/Right arrow does NOT seek/play
        before_calls = len(fake_audition.play_calls)
        QTest.keyClick(win.table_view, Qt.Key.Key_Left)
        QTest.keyClick(win.table_view, Qt.Key.Key_Right)
        _pump(40)
        assert len(fake_audition.play_calls) == before_calls

        # 4. Space pauses/resumes when table is focused
        QTest.keyClick(win.table_view, Qt.Key.Key_Space)
        _pump(40)
        assert fake_audition.toggle_calls == 1
        assert fake_audition.state == "paused"
        QTest.keyClick(win.table_view, Qt.Key.Key_Space)
        _pump(40)
        assert fake_audition.toggle_calls == 2
        assert fake_audition.state == "playing"

        # 5. Waveform click seeks & plays > 0
        wf_rect = win.waveform._plot_rect()
        QTest.mouseClick(
            win.waveform,
            Qt.MouseButton.LeftButton,
            pos=QPoint(int(wf_rect.center().x()), int(wf_rect.center().y())),
        )
        _pump(60)
        assert fake_audition.play_calls[-1][2] > 0.0

        # 6. Esc stops playback
        QTest.keyClick(win.table_view, Qt.Key.Key_Escape)
        _pump(40)
        assert fake_audition.stop_calls == 1

        # 7. Ctrl+F focuses search input; Space types a space without toggling playback
        QTest.keyClick(win.table_view, Qt.Key.Key_F, Qt.KeyboardModifier.ControlModifier)
        _pump(40)
        assert win.focusWidget() is win.search_input
        toggles_before = fake_audition.toggle_calls
        QTest.keyClicks(win.search_input, "beta ")
        _wait_until(lambda: win.table_model.rowCount() == 1)
        assert win.search_input.text() == "beta "
        assert fake_audition.toggle_calls == toggles_before
        assert win.table_model.record_at(0).filename == "beta_02.wav"

        # 8. Clear search and verify column sort signal works
        win.search_input.clear()
        _wait_until(lambda: win.table_model.rowCount() == 2)
        win.table_model.sort(0, Qt.SortOrder.DescendingOrder)
        _wait_until(lambda: win.table_model.record_at(0).filename == "beta_02.wav")

        # 9. Folder tree selection + Alt+Left / Alt+Right history
        root_item = win.folder_tree.find_root_item(root.id)
        assert root_item is not None
        win.folder_tree.expandItem(root_item)
        _pump(40)
        alpha_item = root_item.child(0)
        win.folder_tree.setCurrentItem(alpha_item)
        _wait_until(lambda: win.table_model.rowCount() == 1)
        QTest.keyClick(win.table_view, Qt.Key.Key_Left, Qt.KeyboardModifier.AltModifier)
        _wait_until(lambda: win.table_model.rowCount() == 2)
        QTest.keyClick(win.table_view, Qt.Key.Key_Right, Qt.KeyboardModifier.AltModifier)
        _wait_until(lambda: win.table_model.rowCount() == 1)

        # 10. mimeData exports valid local URL and downward drag keeps initial row
        mime = win.table_model.mimeData([win.table_model.index(0, 0)])
        assert mime.hasUrls()
        assert Path(mime.urls()[0].toLocalFile()).name == "alpha_01.wav"

        # Reset to all 2 rows (still sorted descending: row 0=beta_02, row 1=alpha_01)
        # and verify downward drag from row 0 drags row 0, not row 1.
        QTest.keyClick(win.table_view, Qt.Key.Key_Left, Qt.KeyboardModifier.AltModifier)
        _wait_until(lambda: win.table_model.rowCount() == 2)
        expected_drag_name = win.table_model.record_at(0).filename
        assert expected_drag_name == "beta_02.wav"
        dragged: list[str] = []
        orig_drag = file_table_mod.start_file_drag
        file_table_mod.start_file_drag = lambda _w, p: dragged.append(Path(p).name) or True
        try:
            vp = win.table_view.viewport()
            p_row0 = win.table_view.visualRect(win.table_model.index(0, 0)).center()
            p_down = QPoint(p_row0.x(), p_row0.y() + 36)
            QApplication.sendEvent(
                vp,
                QMouseEvent(
                    QMouseEvent.Type.MouseButtonPress,
                    p_row0,
                    Qt.MouseButton.LeftButton,
                    Qt.MouseButton.LeftButton,
                    Qt.KeyboardModifier.NoModifier,
                ),
            )
            QApplication.sendEvent(
                vp,
                QMouseEvent(
                    QMouseEvent.Type.MouseMove,
                    p_down,
                    Qt.MouseButton.NoButton,
                    Qt.MouseButton.LeftButton,
                    Qt.KeyboardModifier.NoModifier,
                ),
            )
            QApplication.sendEvent(
                vp,
                QMouseEvent(
                    QMouseEvent.Type.MouseButtonRelease,
                    p_down,
                    Qt.MouseButton.LeftButton,
                    Qt.MouseButton.NoButton,
                    Qt.KeyboardModifier.NoModifier,
                ),
            )
            assert dragged == [expected_drag_name]
        finally:
            file_table_mod.start_file_drag = orig_drag
    finally:
        win.close()


class _FakeSpotClient:
    def __init__(self, connected: bool = True) -> None:
        from audiosearch.browser.reaper_spot import SpotBridgeStatus

        self.connected = connected
        self._status_cls = SpotBridgeStatus
        self.spot_calls: list[dict] = []

    def status(self):
        return self._status_cls(
            connected=self.connected,
            project_name="Demo.rpp" if self.connected else "",
            track_name="SFX_01" if self.connected else "",
            cursor_sec=2.0,
        )

    def spot(self, path: Path, **kwargs):
        from audiosearch.browser.reaper_spot import SpotResult

        self.spot_calls.append({"path": path, **kwargs})
        return SpotResult(
            ok=self.connected,
            message="ok" if self.connected else "offline",
            track_name="SFX_01",
            position=2.0,
            length=kwargs.get("end_sec") or 1.0,
        )


def test_me_single_click_dedupe_and_reaper_spot(tmp_path):
    create_app([])
    sfx = tmp_path / "sfx"
    sr = 48000
    t = np.arange(sr, dtype=np.float64) / sr
    sine = (0.3 * np.sin(2.0 * np.pi * 440.0 * t) * 32767.0).astype("<i2")
    for name in ("clip_a.wav", "clip_b.wav"):
        p = sfx / name
        p.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(p), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(sine.tobytes())

    db_path = tmp_path / "spot.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(sfx)
    LibraryScanner(db_path).scan_root(root.id)
    store.close()

    fake = FakeAuditionController()
    fake_spot = _FakeSpotClient(connected=False)
    win = BrowserWindow(db_path, audition=fake, spot_client=fake_spot)
    win.show()
    _wait_until(lambda: win.table_model.rowCount() == 2)
    try:
        assert "REAPER 未连接" in win.status_mode.text()

        # 1. First click on row 0 plays once
        rect0 = win.table_view.visualRect(win.table_model.index(0, 0))
        QTest.mouseClick(win.table_view.viewport(), Qt.MouseButton.LeftButton, pos=rect0.center())
        _pump(80)
        assert len(fake.play_calls) == 1
        assert fake.play_calls[0][1] == "clip_a.wav"

        # Draw selection on waveform, then click already-selected row 0 again -> does NOT replay or clear selection
        win.waveform.set_selection((0.2, 0.6))
        win._selection = (0.2, 0.6)
        QTest.mouseClick(win.table_view.viewport(), Qt.MouseButton.LeftButton, pos=rect0.center())
        _pump(60)
        assert len(fake.play_calls) == 1
        assert win.waveform.selection == (0.2, 0.6)

        # 2. Boundary Key_Up on row 0 does NOT replay; Key_Down to row 1 plays once;
        # second Key_Down on last row does NOT replay
        QTest.keyClick(win.table_view, Qt.Key.Key_Up)
        _pump(40)
        assert len(fake.play_calls) == 1
        QTest.keyClick(win.table_view, Qt.Key.Key_Down)
        _pump(80)
        assert len(fake.play_calls) == 2
        assert fake.play_calls[1][1] == "clip_b.wav"
        QTest.keyClick(win.table_view, Qt.Key.Key_Down)
        _pump(40)
        assert len(fake.play_calls) == 2

        # 3. Clicking drawn waveform bar on already-selected row 1 STILL seeks
        r1 = win.table_model.record_at(1)
        _wait_until(lambda: win.thumbnail_manager.get_cached(r1.id, r1.mtime_ns) is not None)
        cell1 = win.table_view.visualRect(win.table_model.index(1, WAVEFORM_COL))
        wrect1 = WaveformItemDelegate.waveform_rect(cell1, r1.duration_sec)
        QTest.mouseClick(win.table_view.viewport(), Qt.MouseButton.LeftButton, pos=wrect1.center())
        _pump(60)
        assert len(fake.play_calls) == 3
        assert fake.play_calls[2][2] > 0.2

        # 4. Offline Spot -> non-blocking status bar message, no spot() call sent
        QTest.mouseDClick(win.table_view.viewport(), Qt.MouseButton.LeftButton, pos=rect0.center())
        _pump(40)
        assert len(fake_spot.spot_calls) == 0
        assert "未连接" in win.statusBar().currentMessage()

        # 5. Connected Spot: double-click row 0 inserts full file; Enter with selection + tone spots selection + tone
        fake_spot.connected = True
        win._refresh_reaper_status()
        assert "REAPER 已连接" in win.status_mode.text()

        QTest.mouseDClick(win.table_view.viewport(), Qt.MouseButton.LeftButton, pos=rect0.center())
        _pump(40)
        assert len(fake_spot.spot_calls) == 1
        assert fake_spot.spot_calls[0]["path"].name == "clip_a.wav"
        assert fake_spot.spot_calls[0]["start_sec"] == 0.0
        assert fake_spot.spot_calls[0]["end_sec"] is None

        # Select row 0 cleanly, set waveform selection + pitch/rate/preserve_pitch, then press Enter
        QTest.mouseClick(win.table_view.viewport(), Qt.MouseButton.LeftButton, pos=rect0.center())
        _pump(60)
        win.waveform.set_selection((0.25, 0.75))
        win._selection = (0.25, 0.75)
        win.pitch_knob.setValue(4.0)
        win.rate_knob.setValue(1.25)
        win.preserve_pitch_btn.setChecked(True)
        win.bake_drag_btn.setChecked(True)

        QTest.keyClick(win.waveform, Qt.Key.Key_Return)
        _pump(40)
        assert len(fake_spot.spot_calls) == 2
        call = fake_spot.spot_calls[1]
        assert call["path"].name == "clip_a.wav"
        assert abs(call["start_sec"] - 0.25) < 1e-6
        assert abs(call["end_sec"] - 0.75) < 1e-6
        assert call["pitch_semitones"] == 4.0
        assert call["rate"] == 1.25
        assert call["preserve_pitch"] is True
        assert call["apply_tone"] is True
    finally:
        win.close()
        assert fake_audition.closed is True


def test_r2_transport_selection_zoom_drag_and_settings_persistence(tmp_path):
    create_app([])
    sfx = tmp_path / "sfx"
    _make_wav(sfx / "tone_1s.wav", frames=48000)
    _make_wav(sfx / "tone_2s.wav", frames=48000)

    db_path = tmp_path / "r2.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(sfx)
    LibraryScanner(db_path).scan_root(root.id)
    store.close()

    fake = FakeAuditionController()
    win = BrowserWindow(db_path, audition=fake)
    win.show()
    _wait_until(lambda: win.table_model.rowCount() == 2)

    try:
        # 1. Disable Autoplay -> clicking row inspects & loads waveform without playing
        win.autoplay_btn.setChecked(False)
        rect0 = win.table_view.visualRect(win.table_model.index(0, 0))
        QTest.mouseClick(win.table_view.viewport(), Qt.MouseButton.LeftButton, pos=rect0.center())
        _wait_until(lambda: win.waveform.duration > 0.5)
        assert len(fake.play_calls) == 0

        # Space starts playing selected file when Autoplay was off
        QTest.keyClick(win.table_view, Qt.Key.Key_Space)
        _pump(40)
        assert len(fake.play_calls) == 1

        # 2. Loop, Volume, Pitch, Rate, Preserve Pitch & toggles propagate and reset
        win.loop_btn.setChecked(True)
        assert fake.loop is True
        win.volume_knob.setValue(-12.0)
        assert fake.volume_db == -12.0
        win.preserve_pitch_btn.setChecked(True)
        assert fake.preserve_pitch is True
        win.pitch_knob.setValue(5.0)
        win.rate_knob.setValue(1.5)
        assert fake.pitch_rate == (5.0, 1.5)
        win.reset_tone_btn.click()
        assert fake.pitch_rate == (0.0, 1.0)

        # 3. Waveform drag establishes selection and replays selection when Autoplay/playing;
        # Esc clears selection before stopping
        win.autoplay_btn.setChecked(True)
        calls_before_sel = len(fake.play_calls)
        wf_rect = win.waveform._plot_rect()
        x1 = int(wf_rect.left() + wf_rect.width() * 0.2)
        x2 = int(wf_rect.left() + wf_rect.width() * 0.6)
        y = int(wf_rect.center().y())
        QApplication.sendEvent(
            win.waveform,
            QMouseEvent(
                QMouseEvent.Type.MouseButtonPress,
                QPoint(x1, y),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        QApplication.sendEvent(
            win.waveform,
            QMouseEvent(
                QMouseEvent.Type.MouseMove,
                QPoint(x2, y),
                Qt.MouseButton.NoButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        QApplication.sendEvent(
            win.waveform,
            QMouseEvent(
                QMouseEvent.Type.MouseButtonRelease,
                QPoint(x2, y),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        assert win.waveform.selection is not None
        assert fake.selection is not None
        s0, s1 = win.waveform.selection
        assert 0.1 < s0 < s1 < 0.7
        assert len(fake.play_calls) == calls_before_sel + 1
        assert abs(fake.play_calls[-1][2] - s0) < 0.02
        win.autoplay_btn.setChecked(False)

        # 4. Dragging inside selection renders WAV (with Alt => _fade.wav) and starts drag
        dragged: list[str] = []
        orig_drag = main_window_mod.start_file_drag
        main_window_mod.start_file_drag = lambda _w, p: dragged.append(str(p)) or True
        try:
            xm = int((x1 + x2) / 2)
            QApplication.sendEvent(
                win.waveform,
                QMouseEvent(
                    QMouseEvent.Type.MouseButtonPress,
                    QPoint(xm, y),
                    Qt.MouseButton.LeftButton,
                    Qt.MouseButton.LeftButton,
                    Qt.KeyboardModifier.AltModifier,
                ),
            )
            QApplication.sendEvent(
                win.waveform,
                QMouseEvent(
                    QMouseEvent.Type.MouseMove,
                    QPoint(xm + 30, y + 30),
                    Qt.MouseButton.NoButton,
                    Qt.MouseButton.LeftButton,
                    Qt.KeyboardModifier.AltModifier,
                ),
            )
            QApplication.sendEvent(
                win.waveform,
                QMouseEvent(
                    QMouseEvent.Type.MouseButtonRelease,
                    QPoint(xm + 30, y + 30),
                    Qt.MouseButton.LeftButton,
                    Qt.MouseButton.NoButton,
                    Qt.KeyboardModifier.NoModifier,
                ),
            )
            assert len(dragged) == 1
            assert dragged[0].endswith("_fade.wav")
            assert Path(dragged[0]).exists()
        finally:
            main_window_mod.start_file_drag = orig_drag

        # First Esc clears selection without calling stop(); second Esc calls stop()
        stops_before = fake.stop_calls
        QTest.keyClick(win.waveform, Qt.Key.Key_Escape)
        assert win.waveform.selection is None
        assert fake.stop_calls == stops_before
        QTest.keyClick(win.waveform, Qt.Key.Key_Escape)
        assert fake.stop_calls == stops_before + 1

        # 5. Wheel zooms in; double-click resets view
        QApplication.sendEvent(
            win.waveform,
            QWheelEvent(
                QPointF(wf_rect.center()),
                QPointF(wf_rect.center()),
                QPoint(0, 0),
                QPoint(0, 120),
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
                Qt.ScrollPhase.NoScrollPhase,
                False,
            ),
        )
        v0, v1 = win.waveform.view
        assert (v1 - v0) < win.waveform.duration
        QTest.mouseDClick(win.waveform, Qt.MouseButton.LeftButton, pos=wf_rect.center().toPoint())
        v0_reset, v1_reset = win.waveform.view
        assert abs((v1_reset - v0_reset) - win.waveform.duration) < 1e-4

        # Customize column width to check settings persistence
        win.table_view.setColumnWidth(0, 315)
    finally:
        win.close()

    # 6. Reopen window and verify settings persisted
    fake2 = FakeAuditionController()
    win2 = BrowserWindow(db_path, audition=fake2)
    try:
        assert win2.autoplay_btn.isChecked() is False
        assert win2.loop_btn.isChecked() is True
        assert win2.volume_knob.value() == -12.0
        assert win2.preserve_pitch_btn.isChecked() is True
        assert win2.reset_switch_btn.isChecked() is True
        assert win2.bake_drag_btn.isChecked() is True
        assert win2.table_view.columnWidth(0) == 315
    finally:
        win2.close()


def test_waveform_column_seek_drag_lazy_cache_and_settings(tmp_path):
    create_app([])
    sfx = tmp_path / "sfx"
    # Make short (0.2s), medium (2.0s), and long (15.0s) WAVs to verify log-scaled widths.
    _make_wav(sfx / "alpha.wav", frames=96000)  # 2.0s
    _make_wav(sfx / "beta.wav", frames=9600)    # 0.2s
    _make_wav(sfx / "gamma.wav", frames=720000) # 15.0s

    db_path = tmp_path / "wf_col.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(sfx)
    LibraryScanner(db_path).scan_root(root.id)
    store.close()

    # Seed legacy settings missing "waveform" in column_order to verify migration.
    settings_path = settings_path_for_db(db_path)
    settings_path.write_text(
        json.dumps(
            {
                "autoplay": False,
                "column_order": [key for key, _, _ in COLUMNS if key != "waveform"],
                "hidden_columns": ["rel_path"],
            }
        ),
        encoding="utf-8",
    )

    fake = FakeAuditionController()
    win = BrowserWindow(db_path, audition=fake)
    win.show()
    _wait_until(lambda: win.table_model.rowCount() == 3)

    try:
        # 1. Default layout: waveform column placed right after filename, row height = 28px, hidden_columns applied.
        header = win.table_view.horizontalHeader()
        assert COLUMNS[0][0] == "filename"
        assert COLUMNS[WAVEFORM_COL][0] == "waveform"
        assert header.visualIndex(WAVEFORM_COL) == 1
        assert win.table_view.verticalHeader().defaultSectionSize() == 28
        assert win.table_view.isColumnHidden(len(COLUMNS) - 1) is True

        # 2. Lazy thumbnail loader populates visible rows and caches them into SQLite.
        r0 = win.table_model.record_at(0)
        r1 = win.table_model.record_at(1)
        r2 = win.table_model.record_at(2)
        assert r0 is not None and r1 is not None and r2 is not None
        _wait_until(
            lambda: win.thumbnail_manager.get_cached(r0.id, r0.mtime_ns) is not None,
            timeout_ms=5000,
        )
        _wait_until(
            lambda: win.thumbnail_manager.get_cached(r1.id, r1.mtime_ns) is not None,
            timeout_ms=5000,
        )
        _wait_until(
            lambda: win.thumbnail_manager.get_cached(r2.id, r2.mtime_ns) is not None,
            timeout_ms=5000,
        )

        # Log-scaled waveform_rect widths: short (0.2s) < medium (2.0s) < long (15.0s)
        cell0 = win.table_view.visualRect(win.table_model.index(0, WAVEFORM_COL))
        cell1 = win.table_view.visualRect(win.table_model.index(1, WAVEFORM_COL))
        cell2 = win.table_view.visualRect(win.table_model.index(2, WAVEFORM_COL))
        wrect0 = WaveformItemDelegate.waveform_rect(cell0, r0.duration_sec)
        wrect1 = WaveformItemDelegate.waveform_rect(cell1, r1.duration_sec)
        wrect2 = WaveformItemDelegate.waveform_rect(cell2, r2.duration_sec)
        assert wrect1.width() < wrect0.width() < wrect2.width()

        # 3. Clicking drawn waveform center starts playback near 0.5 * duration even with Autoplay off.
        QTest.mouseClick(win.table_view.viewport(), Qt.MouseButton.LeftButton, pos=wrect0.center())
        _pump(80)
        assert len(fake.play_calls) == 1
        assert fake.play_calls[0][1] == "alpha.wav"
        assert 0.85 <= fake.play_calls[0][2] <= 1.15

        # Same row click near 25% of drawn waveform seeks again without resetting to 0.
        x_quarter = int(wrect0.left() + wrect0.width() * 0.25)
        QTest.mouseClick(
            win.table_view.viewport(),
            Qt.MouseButton.LeftButton,
            pos=QPoint(x_quarter, wrect0.center().y()),
        )
        _pump(60)
        assert len(fake.play_calls) == 2
        assert 0.35 <= fake.play_calls[1][2] <= 0.65

        # Clicking row 1 short waveform center switches file and plays near 0.5 * 0.2s.
        QTest.mouseClick(win.table_view.viewport(), Qt.MouseButton.LeftButton, pos=wrect1.center())
        _pump(80)
        assert len(fake.play_calls) == 3
        assert fake.play_calls[2][1] == "beta.wav"
        assert 0.05 <= fake.play_calls[2][2] <= 0.15

        # Clicking the blank right area of short row 1 does NOT seek/play when Autoplay is off.
        blank_x = cell1.right() - 10
        assert blank_x > wrect1.right() + 4
        QTest.mouseClick(
            win.table_view.viewport(),
            Qt.MouseButton.LeftButton,
            pos=QPoint(blank_x, cell1.center().y()),
        )
        _pump(60)
        assert len(fake.play_calls) == 3

        # 4. Dragging from waveform cell triggers whole-file drag and does NOT emit seek.
        calls_before_drag = len(fake.play_calls)
        dragged: list[str] = []
        orig_drag = file_table_mod.start_file_drag
        file_table_mod.start_file_drag = lambda _w, p: dragged.append(Path(p).name) or True
        try:
            vp = win.table_view.viewport()
            p_start = wrect0.center()
            p_end = QPoint(p_start.x() + 25, p_start.y() + 25)
            QApplication.sendEvent(
                vp,
                QMouseEvent(
                    QMouseEvent.Type.MouseButtonPress,
                    p_start,
                    Qt.MouseButton.LeftButton,
                    Qt.MouseButton.LeftButton,
                    Qt.KeyboardModifier.NoModifier,
                ),
            )
            QApplication.sendEvent(
                vp,
                QMouseEvent(
                    QMouseEvent.Type.MouseMove,
                    p_end,
                    Qt.MouseButton.NoButton,
                    Qt.MouseButton.LeftButton,
                    Qt.KeyboardModifier.NoModifier,
                ),
            )
            QApplication.sendEvent(
                vp,
                QMouseEvent(
                    QMouseEvent.Type.MouseButtonRelease,
                    p_end,
                    Qt.MouseButton.LeftButton,
                    Qt.MouseButton.NoButton,
                    Qt.KeyboardModifier.NoModifier,
                ),
            )
            assert dragged == ["alpha.wav"]
            assert len(fake.play_calls) == calls_before_drag
        finally:
            file_table_mod.start_file_drag = orig_drag

        # 5. Clicking waveform header does not change sort column or indicator.
        win.table_model.sort(WAVEFORM_COL, Qt.SortOrder.DescendingOrder)
        header.sectionClicked.emit(WAVEFORM_COL)
        assert win.table_model.sort_column == "filename"
        assert header.sortIndicatorSection() == 0

        # Hide waveform column via helper and verify persistence across close/reopen.
        win.table_view._toggle_column_visible(WAVEFORM_COL, "waveform", False)
        assert win.table_view.isColumnHidden(WAVEFORM_COL) is True
    finally:
        win.close()

    # Verify thumbnails persisted to SQLite and hidden_columns persisted to settings.
    verify_store = LibraryStore(db_path)
    try:
        cached_map = verify_store.get_cached_thumbnails([(r0.id, r0.mtime_ns), (r1.id, r1.mtime_ns)])
        assert set(cached_map.keys()) == {r0.id, r1.id}
    finally:
        verify_store.close()

    fake2 = FakeAuditionController()
    win2 = BrowserWindow(db_path, audition=fake2)
    try:
        assert win2.table_view.isColumnHidden(WAVEFORM_COL) is True
    finally:
        win2.close()


def test_drag_knob_and_me_waveform_and_baked_drag(tmp_path):
    create_app([])
    from audiosearch.browser.drag_knob import DragValueControl

    # 1. DragValueControl vertical drag, Shift fine mode, wheel, and double-click reset
    knob = DragValueControl(
        "变调",
        minimum=-24.0,
        maximum=24.0,
        default=0.0,
        value_format=lambda v: f"{v:+.1f} st",
        drag_range_px=160.0,
        wheel_step=1.0,
        decimals=1,
    )
    knob.resize(116, 30)
    knob.show()
    try:
        QApplication.sendEvent(
            knob,
            QMouseEvent(
                QMouseEvent.Type.MouseButtonPress,
                QPoint(50, 20),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        QApplication.sendEvent(
            knob,
            QMouseEvent(
                QMouseEvent.Type.MouseMove,
                QPoint(50, 0),  # +20 px up => +6.0 st
                Qt.MouseButton.NoButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        assert abs(knob.value() - 6.0) < 0.2
        QApplication.sendEvent(
            knob,
            QMouseEvent(
                QMouseEvent.Type.MouseMove,
                QPoint(50, -40),  # +40 px up with Shift => +1.2 st fine
                Qt.MouseButton.NoButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.ShiftModifier,
            ),
        )
        assert abs(knob.value() - 7.2) < 0.2
        QApplication.sendEvent(
            knob,
            QMouseEvent(
                QMouseEvent.Type.MouseButtonRelease,
                QPoint(50, -40),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        QApplication.sendEvent(
            knob,
            QWheelEvent(
                QPointF(50, 15),
                QPointF(50, 15),
                QPoint(0, 0),
                QPoint(0, 120),
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
                Qt.ScrollPhase.NoScrollPhase,
                False,
            ),
        )
        assert abs(knob.value() - 8.2) < 0.2
        QTest.mouseDClick(knob, Qt.MouseButton.LeftButton, pos=QPoint(50, 15))
        assert knob.value() == 0.0
    finally:
        knob.close()

    # 2. Full BrowserWindow ME waveform UX + reset_on_switch + baked drag
    sfx = tmp_path / "sfx"
    sr = 48000
    t = np.arange(sr, dtype=np.float64) / sr
    sine = (0.4 * np.sin(2.0 * np.pi * 440.0 * t) * 32767.0).astype("<i2")
    for name in ("one.wav", "two.wav"):
        p = sfx / name
        p.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(p), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(sine.tobytes())

    db_path = tmp_path / "me_ux.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(sfx)
    LibraryScanner(db_path).scan_root(root.id)
    store.close()

    fake = FakeAuditionController()
    win = BrowserWindow(db_path, audition=fake)
    win.show()
    _wait_until(lambda: win.table_model.rowCount() == 2)
    try:
        # Select row 0 (one.wav) -> plays
        rect0 = win.table_view.visualRect(win.table_model.index(0, 0))
        QTest.mouseClick(win.table_view.viewport(), Qt.MouseButton.LeftButton, pos=rect0.center())
        _pump(80)
        assert fake.play_calls[-1][1] == "one.wav"

        # Draw selection [0.25, 0.75] on waveform -> emits selectionCommitted & auto-plays selection
        wf_rect = win.waveform._plot_rect()
        x1 = int(wf_rect.left() + wf_rect.width() * 0.25)
        x2 = int(wf_rect.left() + wf_rect.width() * 0.75)
        y = int(wf_rect.center().y())
        QApplication.sendEvent(
            win.waveform,
            QMouseEvent(
                QMouseEvent.Type.MouseButtonPress,
                QPoint(x1, y),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        QApplication.sendEvent(
            win.waveform,
            QMouseEvent(
                QMouseEvent.Type.MouseMove,
                QPoint(x2, y),
                Qt.MouseButton.NoButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        QApplication.sendEvent(
            win.waveform,
            QMouseEvent(
                QMouseEvent.Type.MouseButtonRelease,
                QPoint(x2, y),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        assert win.waveform.selection is not None
        s0, s1 = win.waveform.selection
        assert abs(s0 - 0.25) < 0.03 and abs(s1 - 0.75) < 0.03
        assert abs(fake.play_calls[-1][2] - s0) < 0.03

        # Hover within 6 px of right edge -> SizeHorCursor; drag right to trim selection
        edge_x = int(win.waveform._time_to_x(s1, wf_rect))
        QApplication.sendEvent(
            win.waveform,
            QMouseEvent(
                QMouseEvent.Type.MouseMove,
                QPoint(edge_x, y),
                Qt.MouseButton.NoButton,
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        assert win.waveform.cursor().shape() == Qt.CursorShape.SizeHorCursor
        calls_before_trim = len(fake.play_calls)
        new_edge_x = int(wf_rect.left() + wf_rect.width() * 0.85)
        QApplication.sendEvent(
            win.waveform,
            QMouseEvent(
                QMouseEvent.Type.MouseButtonPress,
                QPoint(edge_x, y),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        QApplication.sendEvent(
            win.waveform,
            QMouseEvent(
                QMouseEvent.Type.MouseMove,
                QPoint(new_edge_x, y),
                Qt.MouseButton.NoButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        QApplication.sendEvent(
            win.waveform,
            QMouseEvent(
                QMouseEvent.Type.MouseButtonRelease,
                QPoint(new_edge_x, y),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        assert abs(win.waveform.selection[1] - 0.85) < 0.03
        assert len(fake.play_calls) == calls_before_trim + 1

        # Single-click INSIDE selection seeks without clearing selection
        mid_x = int(wf_rect.left() + wf_rect.width() * 0.5)
        QTest.mouseClick(win.waveform, Qt.MouseButton.LeftButton, pos=QPoint(mid_x, y))
        _pump(40)
        assert win.waveform.selection is not None
        assert abs(fake.play_calls[-1][2] - 0.5) < 0.03

        # Single-click OUTSIDE selection clears selection and seeks
        out_x = int(wf_rect.left() + wf_rect.width() * 0.05)
        QTest.mouseClick(win.waveform, Qt.MouseButton.LeftButton, pos=QPoint(out_x, y))
        _pump(40)
        assert win.waveform.selection is None
        assert fake.selection is None
        assert abs(fake.play_calls[-1][2] - 0.05) < 0.03

        # Middle-button pan when zoomed in moves view window
        QApplication.sendEvent(
            win.waveform,
            QWheelEvent(
                QPointF(wf_rect.center()),
                QPointF(wf_rect.center()),
                QPoint(0, 0),
                QPoint(0, 120),
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
                Qt.ScrollPhase.NoScrollPhase,
                False,
            ),
        )
        v0_before, v1_before = win.waveform.view
        QApplication.sendEvent(
            win.waveform,
            QMouseEvent(
                QMouseEvent.Type.MouseButtonPress,
                QPoint(mid_x, y),
                Qt.MouseButton.MiddleButton,
                Qt.MouseButton.MiddleButton,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        QApplication.sendEvent(
            win.waveform,
            QMouseEvent(
                QMouseEvent.Type.MouseMove,
                QPoint(mid_x - 40, y),
                Qt.MouseButton.NoButton,
                Qt.MouseButton.MiddleButton,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        QApplication.sendEvent(
            win.waveform,
            QMouseEvent(
                QMouseEvent.Type.MouseButtonRelease,
                QPoint(mid_x - 40, y),
                Qt.MouseButton.MiddleButton,
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        assert win.waveform.view[0] > v0_before

        # Reset-on-switch toggle: ON resets Pitch/Rate on row switch; OFF retains them
        win.pitch_knob.setValue(5.0)
        win.rate_knob.setValue(1.25)
        rect1 = win.table_view.visualRect(win.table_model.index(1, 0))
        QTest.mouseClick(win.table_view.viewport(), Qt.MouseButton.LeftButton, pos=rect1.center())
        _pump(80)
        assert win.pitch_knob.value() == 0.0
        assert win.rate_knob.value() == 1.0

        win.reset_switch_btn.setChecked(False)
        win.pitch_knob.setValue(5.0)
        win.rate_knob.setValue(1.25)
        win.preserve_pitch_btn.setChecked(True)
        QTest.mouseClick(win.table_view.viewport(), Qt.MouseButton.LeftButton, pos=rect0.center())
        _pump(80)
        assert win.pitch_knob.value() == 5.0
        assert win.rate_knob.value() == 1.25

        # Baked table drag: bake_drag_btn ON => rendered WAV; OFF => original WAV
        dragged_paths: list[Path] = []
        orig_drag = file_table_mod.start_file_drag
        file_table_mod.start_file_drag = lambda _w, p: dragged_paths.append(Path(p)) or True
        try:
            vp = win.table_view.viewport()
            p0 = rect0.center()
            p1 = QPoint(p0.x() + 25, p0.y())
            for typ, pt, btn, btns in [
                (QMouseEvent.Type.MouseButtonPress, p0, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton),
                (QMouseEvent.Type.MouseMove, p1, Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton),
                (QMouseEvent.Type.MouseButtonRelease, p1, Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton),
            ]:
                QApplication.sendEvent(vp, QMouseEvent(typ, pt, btn, btns, Qt.KeyboardModifier.NoModifier))
            assert len(dragged_paths) == 1
            assert "_p+5.0_r1.25_pp.wav" in dragged_paths[0].name
            assert dragged_paths[0].exists()

            win.bake_drag_btn.setChecked(False)
            for typ, pt, btn, btns in [
                (QMouseEvent.Type.MouseButtonPress, p0, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton),
                (QMouseEvent.Type.MouseMove, p1, Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton),
                (QMouseEvent.Type.MouseButtonRelease, p1, Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton),
            ]:
                QApplication.sendEvent(vp, QMouseEvent(typ, pt, btn, btns, Qt.KeyboardModifier.NoModifier))
            assert len(dragged_paths) == 2
            assert dragged_paths[1].name == "one.wav"
        finally:
            file_table_mod.start_file_drag = orig_drag
    finally:
        win.close()


class _FakeSpotClient:
    def __init__(self, connected: bool = True) -> None:
        from audiosearch.browser.reaper_spot import SpotBridgeStatus

        self.connected = connected
        self._status_cls = SpotBridgeStatus
        self.spot_calls: list[dict] = []

    def status(self):
        return self._status_cls(
            connected=self.connected,
            project_name="Demo.rpp" if self.connected else "",
            track_name="SFX_01" if self.connected else "",
            cursor_sec=2.0,
        )

    def spot(self, path: Path, **kwargs):
        from audiosearch.browser.reaper_spot import SpotResult

        self.spot_calls.append({"path": path, **kwargs})
        return SpotResult(
            ok=self.connected,
            message="ok" if self.connected else "offline",
            track_name="SFX_01",
            position=2.0,
            length=kwargs.get("end_sec") or 1.0,
        )


def test_me_single_click_dedupe_and_reaper_spot(tmp_path):
    create_app([])
    sfx = tmp_path / "sfx"
    sr = 48000
    t = np.arange(sr, dtype=np.float64) / sr
    sine = (0.3 * np.sin(2.0 * np.pi * 440.0 * t) * 32767.0).astype("<i2")
    for name in ("clip_a.wav", "clip_b.wav"):
        p = sfx / name
        p.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(p), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(sine.tobytes())

    db_path = tmp_path / "spot.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(sfx)
    LibraryScanner(db_path).scan_root(root.id)
    store.close()

    fake = FakeAuditionController()
    fake_spot = _FakeSpotClient(connected=False)
    win = BrowserWindow(db_path, audition=fake, spot_client=fake_spot)
    win.show()
    _wait_until(lambda: win.table_model.rowCount() == 2)
    try:
        assert "REAPER 未连接" in win.status_mode.text()

        # 1. First click on row 0 plays once
        rect0 = win.table_view.visualRect(win.table_model.index(0, 0))
        QTest.mouseClick(win.table_view.viewport(), Qt.MouseButton.LeftButton, pos=rect0.center())
        _pump(80)
        assert len(fake.play_calls) == 1
        assert fake.play_calls[0][1] == "clip_a.wav"

        # Draw selection on waveform, then click already-selected row 0 again -> does NOT replay or clear selection
        win.waveform.set_selection((0.2, 0.6))
        win._selection = (0.2, 0.6)
        QTest.mouseClick(win.table_view.viewport(), Qt.MouseButton.LeftButton, pos=rect0.center())
        _pump(60)
        assert len(fake.play_calls) == 1
        assert win.waveform.selection == (0.2, 0.6)

        # 2. Boundary Key_Up on row 0 does NOT replay; Key_Down to row 1 plays once;
        # second Key_Down on last row does NOT replay
        QTest.keyClick(win.table_view, Qt.Key.Key_Up)
        _pump(40)
        assert len(fake.play_calls) == 1
        QTest.keyClick(win.table_view, Qt.Key.Key_Down)
        _pump(80)
        assert len(fake.play_calls) == 2
        assert fake.play_calls[1][1] == "clip_b.wav"
        QTest.keyClick(win.table_view, Qt.Key.Key_Down)
        _pump(40)
        assert len(fake.play_calls) == 2

        # 3. Clicking drawn waveform bar on already-selected row 1 STILL seeks
        r1 = win.table_model.record_at(1)
        _wait_until(lambda: win.thumbnail_manager.get_cached(r1.id, r1.mtime_ns) is not None)
        cell1 = win.table_view.visualRect(win.table_model.index(1, WAVEFORM_COL))
        wrect1 = WaveformItemDelegate.waveform_rect(cell1, r1.duration_sec)
        QTest.mouseClick(win.table_view.viewport(), Qt.MouseButton.LeftButton, pos=wrect1.center())
        _pump(60)
        assert len(fake.play_calls) == 3
        assert fake.play_calls[2][2] > 0.2

        # 4. Offline Spot -> non-blocking status bar message, no spot() call sent
        QTest.mouseDClick(win.table_view.viewport(), Qt.MouseButton.LeftButton, pos=rect0.center())
        _pump(40)
        assert len(fake_spot.spot_calls) == 0
        assert "未连接" in win.statusBar().currentMessage()

        # 5. Connected Spot: double-click row 0 inserts full file; Enter with selection + tone spots selection + tone
        fake_spot.connected = True
        win._refresh_reaper_status()
        assert "REAPER 已连接" in win.status_mode.text()

        QTest.mouseDClick(win.table_view.viewport(), Qt.MouseButton.LeftButton, pos=rect0.center())
        _pump(40)
        assert len(fake_spot.spot_calls) == 1
        assert fake_spot.spot_calls[0]["path"].name == "clip_a.wav"
        assert fake_spot.spot_calls[0]["start_sec"] == 0.0
        assert fake_spot.spot_calls[0]["end_sec"] is None

        # Select row 0 cleanly, set waveform selection + pitch/rate/preserve_pitch, then press Enter
        QTest.mouseClick(win.table_view.viewport(), Qt.MouseButton.LeftButton, pos=rect0.center())
        _pump(60)
        win.waveform.set_selection((0.25, 0.75))
        win._selection = (0.25, 0.75)
        win.pitch_knob.setValue(4.0)
        win.rate_knob.setValue(1.25)
        win.preserve_pitch_btn.setChecked(True)
        win.bake_drag_btn.setChecked(True)

        QTest.keyClick(win.waveform, Qt.Key.Key_Return)
        _pump(40)
        assert len(fake_spot.spot_calls) == 2
        call = fake_spot.spot_calls[1]
        assert call["path"].name == "clip_a.wav"
        assert abs(call["start_sec"] - 0.25) < 1e-6
        assert abs(call["end_sec"] - 0.75) < 1e-6
        assert call["pitch_semitones"] == 4.0
        assert call["rate"] == 1.25
        assert call["preserve_pitch"] is True
        assert call["apply_tone"] is True
    finally:
        win.close()


def test_spectral_peaks_toggle_upgrade_and_persistence(tmp_path):
    create_app([])
    from audiosearch.browser.settings import load_settings, save_settings
    from audiosearch.library.peaks import PeakData

    sfx = tmp_path / "sfx"
    sr = 48000
    t = np.arange(sr, dtype=np.float32) / sr
    tone = 0.5 * np.sin(2.0 * np.pi * 880.0 * t)
    wav_path = sfx / "synth_880.wav"
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(np.clip(np.rint(tone * 32767.0), -32768, 32767).astype("<i2").tobytes())

    db_path = tmp_path / "spectral_ui.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(sfx)
    LibraryScanner(db_path).scan_root(root.id)
    rec = store.query_page(offset=0, limit=10)[0]

    # Seed legacy v1 PeakData (spectral=None) in waveform_cache before opening UI.
    legacy_peaks = PeakData(
        duration_seconds=1.0,
        sample_rate_hz=sr,
        channel_count=1,
        points=main_window_mod.OVERVIEW_POINTS,
        peaks=np.full((1, main_window_mod.OVERVIEW_POINTS, 2), 0.3, dtype=np.float32),
        spectral=None,
    )
    store.save_cached_peaks(
        rec.id,
        rec.mtime_ns,
        legacy_peaks,
        requested_points=main_window_mod.OVERVIEW_POINTS,
    )
    store.close()

    # Verify settings default & roundtrip helper directly.
    sp = settings_path_for_db(db_path)
    assert load_settings(sp).spectral_peaks is False

    fake = FakeAuditionController()
    win = BrowserWindow(db_path, audition=fake)
    win.show()
    _wait_until(lambda: win.table_model.rowCount() == 1)
    try:
        assert win.spectral_peaks_btn.isChecked() is False
        assert win.waveform.spectral_peaks is False
        assert win.table_view._waveform_delegate._spectral_peaks is False

        # Select row 0 -> loads legacy v1 peak (spectral=None)
        rect0 = win.table_view.visualRect(win.table_model.index(0, 0))
        QTest.mouseClick(win.table_view.viewport(), Qt.MouseButton.LeftButton, pos=rect0.center())
        _wait_until(lambda: win.waveform._overview_peaks is not None)
        assert win.waveform._overview_peaks.spectral is None

        # Toggle ON -> syncs waveform + delegate and upgrades legacy v1 cache to v2 spectral
        win.spectral_peaks_btn.click()
        assert win.spectral_peaks_btn.isChecked() is True
        assert win.waveform.spectral_peaks is True
        assert win.table_view._waveform_delegate._spectral_peaks is True
        _wait_until(
            lambda: win.waveform._overview_peaks is not None
            and win.waveform._overview_peaks.spectral is not None
        )
        _wait_until(
            lambda: win.thumbnail_manager.get_cached_spectral(rec.id, rec.mtime_ns) is not None
        )

        # Signal from waveform context menu toggles button OFF and back ON
        win.waveform.spectralPeaksToggled.emit(False)
        assert win.spectral_peaks_btn.isChecked() is False
        assert win.waveform.spectral_peaks is False
        assert win.table_view._waveform_delegate._spectral_peaks is False
        win.waveform.spectralPeaksToggled.emit(True)
        assert win.spectral_peaks_btn.isChecked() is True
    finally:
        win.close()

    # Verify persistence across window reopen
    assert load_settings(sp).spectral_peaks is True
    fake2 = FakeAuditionController()
    win2 = BrowserWindow(db_path, audition=fake2)
    try:
        assert win2.spectral_peaks_btn.isChecked() is True
        assert win2.waveform.spectral_peaks is True
        assert win2.table_view._waveform_delegate._spectral_peaks is True
    finally:
        win2.close()
        s_off = load_settings(sp)
        s_off.spectral_peaks = False
        save_settings(sp, s_off)
        assert load_settings(sp).spectral_peaks is False
