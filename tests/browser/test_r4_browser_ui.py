"""Offscreen PySide6 behavioral tests for R4 UI workflows in BrowserWindow."""

from __future__ import annotations

import json
import math
import os
import shutil
import struct
import time
import wave
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QItemSelectionModel, QObject, QPoint, Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QAbstractItemView, QApplication, QFileDialog

from audiosearch.browser.collections_tree import (
    KIND_COLLECTION,
    KIND_COLLECTION_FOLDER,
    KIND_DUPLICATES,
    KIND_FAVORITES,
    SidebarViewTarget,
)
from audiosearch.browser.file_table import COLUMNS, FAV_COL, MARK_COL, WAVEFORM_COL
from audiosearch.browser.folder_tree import _COUNT_ROLE
from audiosearch.browser.main_window import BrowserWindow, create_app
from audiosearch.browser.settings import settings_path_for_db
from audiosearch.library.scanner import LibraryScanner
from audiosearch.library.store import LibraryStore


class _FakeAudition(QObject):
    positionChanged = Signal(float)
    stateChanged = Signal(str)
    failed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.current_file_id: int | None = None
        self.state = "empty"
        self.can_resume_buffered = False
        self.output_rate_hz = 48000
        self.device_error: str | None = None
        self.position_seconds = 0.0
        self.play_calls: list[int] = []

    def set_loop(self, _loop: bool) -> None:
        pass

    def set_volume_db(self, _db: float) -> None:
        pass

    def set_preserve_pitch(self, _enabled: bool) -> None:
        pass

    def set_pitch_rate(self, _pitch: float, _rate: float) -> None:
        pass

    def set_selection(self, _sel: tuple[float, float] | None) -> None:
        pass

    def play_file(
        self, file_id: int, _path: Path, start_seconds: float = 0.0, *, end_seconds: float | None = None
    ) -> None:
        self.current_file_id = file_id
        self.position_seconds = start_seconds
        self.state = "playing"
        self.can_resume_buffered = True
        self.play_calls.append(file_id)
        self.stateChanged.emit("playing")

    def toggle_pause_resume(self) -> None:
        if self.state == "playing":
            self.state = "paused"
            self.stateChanged.emit("paused")
        elif self.state == "paused":
            self.state = "playing"
            self.stateChanged.emit("playing")

    def stop(self) -> None:
        self.can_resume_buffered = False
        self.state = "paused" if self.current_file_id is not None else "empty"
        self.stateChanged.emit(self.state)

    def close(self) -> None:
        pass


class _FakeSpotClient:
    def status(self):
        return type("S", (), {"connected": False, "track_name": "", "project_name": "", "cursor_sec": 0.0})()


def _write_wav(path: Path, *, duration_sec: float = 0.3, freq: float = 440.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sr = 44100
    n = max(1, int(round(duration_sec * sr)))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = bytearray()
        for i in range(n):
            val = int(10000 * math.sin(2.0 * math.pi * freq * (i / sr)))
            frames.extend(struct.pack("<h", val))
        wf.writeframes(bytes(frames))


def _pump(app: QApplication, *, timeout_sec: float = 2.0, cond=None) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        app.processEvents()
        if cond is not None and cond():
            return
        time.sleep(0.01)


def _setup_window(tmp_path: Path) -> tuple[QApplication, BrowserWindow, Path, Path]:
    app = create_app()
    lib_dir = tmp_path / "audio_root"
    _write_wav(lib_dir / "ui" / "click_short.wav", duration_sec=0.25, freq=500.0)
    _write_wav(lib_dir / "ui" / "confirm_mid.wav", duration_sec=1.4, freq=660.0)
    _write_wav(lib_dir / "sfx" / "explosion_long.wav", duration_sec=0.6, freq=180.0)

    db_path = tmp_path / "library.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(lib_dir)
    store.close()
    LibraryScanner(db_path).scan_root(root.id)

    win = BrowserWindow(db_path, audition=_FakeAudition(), spot_client=_FakeSpotClient())
    win.show()
    _pump(app, timeout_sec=2.0, cond=lambda: win.table_model.rowCount() == 3)
    return app, win, db_path, lib_dir


def test_multiselect_favorite_note_and_text_focus_guard(tmp_path: Path) -> None:
    app, win, db_path, _lib_dir = _setup_window(tmp_path)
    try:
        assert win.table_view.selectionMode() == QAbstractItemView.SelectionMode.ExtendedSelection

        # Select row 0 and toggle favorite directly
        rec0 = win.table_model.record_at(0)
        assert rec0 is not None
        win.table_view.toggleFavoriteRequested.emit(rec0.id)
        _pump(app, timeout_sec=0.2)
        assert win.store.get_file(rec0.id).favorite is True
        assert win.table_model.data(win.table_model.index(0, FAV_COL)) == "★"
        fav_item = win.folder_tree.find_quick_item(KIND_FAVORITES)
        assert fav_item is not None and fav_item.data(0, _COUNT_ROLE) == "1"

        # Multi-select batch favorite
        rec1 = win.table_model.record_at(1)
        assert rec1 is not None
        win.table_view.batchFavoriteRequested.emit([rec0.id, rec1.id], True)
        _pump(app, timeout_sec=0.2)
        assert win.store.favorite_count() == 2

        # Select row 0 via selectionModel + type Space inside inspector note_edit (must NOT play audio!)
        win._on_audition_row(0)
        win.inspector.note_edit.setFocus()
        _pump(app, timeout_sec=0.1)
        calls_before = len(win.audition.play_calls)
        space_ev = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier, " ")
        assert win._handle_key_press(space_ev) is False
        assert len(win.audition.play_calls) == calls_before

        win.inspector.note_edit.setPlainText("游戏结算主角音效")
        win.inspector.flush_pending_note()
        _pump(app, timeout_sec=0.1)
        assert win.store.get_file(rec0.id).note == "游戏结算主角音效"
    finally:
        win.close()
        app.processEvents()


def test_collections_drop_filter_bar_and_history_replay(tmp_path: Path) -> None:
    app, win, _db_path, _lib_dir = _setup_window(tmp_path)
    try:
        coll = win.store.create_collection("UI与界面")
        win.folder_tree.reload()
        rec_click = next(
            win.table_model.record_at(i)
            for i in range(win.table_model.rowCount())
            if win.table_model.record_at(i).filename == "click_short.wav"
        )
        # Drop single URL onto collection (simulating single-file drag fallback)
        win.table_view.selectRow(0)
        win._on_urls_dropped_on_collection(coll.id, [str(rec_click.full_path)])
        win._on_sidebar_view_selected(SidebarViewTarget(KIND_COLLECTION, collection_id=coll.id, display_path="UI与界面"))
        _pump(app, timeout_sec=2.0, cond=lambda: win.table_model.rowCount() == 1)
        assert win.table_model.record_at(0).filename == "click_short.wav"

        # Return to all files and verify collapsible filter bar (>1s duration filter)
        win._active_sidebar = None
        win.filter_toggle_btn.setChecked(True)
        assert win.filter_bar.isVisible()
        idx_1_5 = win.filter_dur_combo.findData("1_5")
        win.filter_dur_combo.setCurrentIndex(idx_1_5)
        _pump(
            app,
            timeout_sec=2.0,
            cond=lambda: (
                win.table_model.rowCount() == 1
                and win.table_model.record_at(0) is not None
                and win.table_model.record_at(0).filename == "confirm_mid.wav"
            ),
        )
        assert win.table_model.record_at(0).filename == "confirm_mid.wav"

        win.filter_reset_btn.click()
        _pump(app, timeout_sec=2.0, cond=lambda: win.table_model.rowCount() == 3)

        # Search history record + replay
        win.search_input.setText("explosion")
        win._on_search_return_pressed()
        _pump(app, timeout_sec=1.5, cond=lambda: win.table_model.rowCount() == 1)
        hist = win.store.list_search_history()
        assert any(h.query == "explosion" for h in hist)

        win.search_input.clear()
        _pump(app, timeout_sec=1.5, cond=lambda: win.table_model.rowCount() == 3)
        target_entry = next(h for h in hist if h.query == "explosion")
        win._apply_history_entry(target_entry)
        _pump(app, timeout_sec=1.5, cond=lambda: win.table_model.rowCount() == 1)
        assert win.table_model.record_at(0).filename == "explosion_long.wav"
    finally:
        win.close()
        app.processEvents()


def test_duplicate_ui_hide_unhide_and_root_relocate(tmp_path: Path, monkeypatch) -> None:
    app, win, db_path, lib_dir = _setup_window(tmp_path)
    try:
        # Create an exact byte duplicate & rescan
        src = lib_dir / "ui" / "click_short.wav"
        dup = lib_dir / "sfx" / "click_short_copy.wav"
        shutil.copy2(src, dup)
        root = win.store.list_roots()[0]
        LibraryScanner(db_path).scan_root(root.id)
        _pump(app, timeout_sec=0.1)

        # Record pre-switch layout geometry to verify zero page shift when entering duplicates view
        _pump(app, timeout_sec=0.1)
        y_tree_before = win.folder_tree.mapTo(win, QPoint(0, 0)).y()
        y_table_before = win.table_view.mapTo(win, QPoint(0, 0)).y()
        y_insp_before = win.inspector.mapTo(win, QPoint(0, 0)).y()
        splitter_sizes_before = win.h_splitter.sizes()

        # Enter duplicate finder UI in exact mode
        win._on_sidebar_view_selected(SidebarViewTarget(KIND_DUPLICATES, display_path="重复文件"))
        _pump(app, timeout_sec=0.1)
        assert win.folder_tree.mapTo(win, QPoint(0, 0)).y() == y_tree_before
        assert win.table_view.mapTo(win, QPoint(0, 0)).y() == y_table_before
        assert win.inspector.mapTo(win, QPoint(0, 0)).y() == y_insp_before
        assert win.h_splitter.sizes() == splitter_sizes_before
        from audiosearch.browser.file_table import COLUMNS as TABLE_COLS

        rel_col = [c[0] for c in TABLE_COLS].index("relevance")
        reason_col = [c[0] for c in TABLE_COLS].index("hit_reason")
        dur_col = [c[0] for c in TABLE_COLS].index("duration_sec")
        assert win.table_model.headerData(rel_col, Qt.Orientation.Horizontal) == "重复状态"
        assert win.table_model.headerData(reason_col, Qt.Orientation.Horizontal) == "重复分组"

        win._set_duplicate_mode("exact")
        win._toggle_duplicate_scan()
        _pump(
            app,
            timeout_sec=3.0,
            cond=lambda: win._dup_thread is not None and not win._dup_thread.is_alive() and win.table_model.rowCount() == 2,
        )
        assert win.dup_banner.isVisible()
        assert win.h_splitter.sizes() == splitter_sizes_before
        assert (
            win.table_model.data(
                win.table_model.index(0, rel_col), Qt.ItemDataRole.TextAlignmentRole
            )
            is None
        )
        assert win.table_model.data(
            win.table_model.index(0, dur_col), Qt.ItemDataRole.TextAlignmentRole
        ) == int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        # Hide non-primary copies in library; verify disk files untouched!
        win._on_hide_all_non_primary()
        assert win.store.hidden_file_count() == 1
        assert src.exists() and dup.exists()

        win._exit_duplicate_mode()
        _pump(app, timeout_sec=2.0, cond=lambda: win.table_model.rowCount() == 3)
        assert win.table_model.headerData(rel_col, Qt.Orientation.Horizontal) == "相关度"
        assert win.table_model.headerData(reason_col, Qt.Orientation.Horizontal) == "命中说明"
        assert win.table_view.mapTo(win, QPoint(0, 0)).y() == y_table_before

        # Restore hidden
        win._on_unhide_all_duplicates()
        _pump(app, timeout_sec=2.0, cond=lambda: win.table_model.rowCount() == 4)

        # Relocate root folder to a new drive path
        moved_dir = tmp_path / "relocated_root"
        shutil.move(str(lib_dir), str(moved_dir))
        monkeypatch.setattr(
            QFileDialog,
            "getExistingDirectory",
            staticmethod(lambda *_a, **_kw: str(moved_dir)),
        )
        win._on_relocate_root(root.id)
        _pump(app, timeout_sec=2.0)
        assert win.store.get_root(root.id).is_online is True
        assert Path(win.store.get_root(root.id).path).resolve() == moved_dir.resolve()
    finally:
        win.close()
        app.processEvents()


def test_subfolder_offline_ui_hide_relocate_and_folder_to_collection(
    tmp_path: Path, monkeypatch
) -> None:
    app, win, _db_path, lib_dir = _setup_window(tmp_path)
    try:
        root = win.store.list_roots()[0]
        ui_folder = next(f for f in win.store.list_folders(root.id, None) if f.name == "ui")

        # Verify FolderTree -> BrowserWindow 5-arg addFolderToCollectionRequested works cleanly
        coll = win.store.create_collection("合集B")
        win.folder_tree.addFolderToCollectionRequested.emit(
            coll.id, root.id, ui_folder.id, ui_folder.rel_path, ui_folder.name
        )
        _pump(app, timeout_sec=0.2)
        assert win.store.query_count(filters=win._current_filters()) == 3
        cf = win.store.list_collection_folders(coll.id)
        assert len(cf) == 1 and cf[0].folder_rel_path == "ui"

        # Star a file inside 'ui', then move 'ui' outside to simulate offline subfolder
        click_row = next(
            i
            for i in range(win.table_model.rowCount())
            if win.table_model.record_at(i).filename == "click_short.wav"
        )
        click_rec = win.table_model.record_at(click_row)
        win.store.set_favorites([click_rec.id], True)

        moved_ui = tmp_path / "ui_moved_away"
        shutil.move(str(lib_dir / "ui"), str(moved_ui))

        # Turn on 隐藏离线, then click the moved file -> immediate offline writeback & hide!
        win.filter_toggle_btn.setChecked(True)
        win.filter_online_cb.setChecked(True)
        _pump(app, timeout_sec=1.5, cond=lambda: win.table_model.rowCount() == 3)
        assert win._prepare_file_selection(click_rec) is False
        _pump(app, timeout_sec=2.0, cond=lambda: win.table_model.rowCount() == 1)
        assert win.table_model.record_at(0).filename == "explosion_long.wav"
        assert win.store.get_file(click_rec.id).is_available is False

        # FolderTree marks moved subfolder offline cleanly without emoji prefixes
        root_item = win.folder_tree.find_root_item(root.id)
        assert root_item is not None
        win.folder_tree.expandItem(root_item)
        _pump(app, timeout_sec=0.1)
        assert any(
            root_item.child(i).text(0) == "ui"
            and win.folder_tree.is_item_offline(root_item.child(i))
            for i in range(root_item.childCount())
        )

        # Relocate subfolder via UI handler -> restores online, visible, and preserves ★
        monkeypatch.setattr(
            QFileDialog,
            "getExistingDirectory",
            staticmethod(lambda *_a, **_kw: str(moved_ui)),
        )
        win._on_relocate_folder(root.id, ui_folder.id)
        _pump(app, timeout_sec=2.0, cond=lambda: win.table_model.rowCount() == 3)
        fresh = win.store.get_file(click_rec.id)
        assert fresh is not None and fresh.is_available is True and fresh.favorite is True
        assert fresh.full_path == (moved_ui / "click_short.wav").resolve()
    finally:
        win.close()
        app.processEvents()


def test_single_file_moved_marks_offline_in_ui_and_hides_immediately(tmp_path: Path) -> None:
    app, win, _db_path, lib_dir = _setup_window(tmp_path)
    try:
        # Remove only ONE file inside ui/ while ui/confirm_mid.wav remains on disk
        missing_path = lib_dir / "ui" / "click_short.wav"
        moved_away = tmp_path / "click_short_moved.wav"
        shutil.move(str(missing_path), str(moved_away))

        click_row = next(
            i
            for i in range(win.table_model.rowCount())
            if win.table_model.record_at(i).filename == "click_short.wav"
        )
        click_rec = win.table_model.record_at(click_row)

        # Clicking the missing file marks it offline immediately (without being overwritten back to 1)
        assert win._prepare_file_selection(click_rec) is False
        fresh = win.store.get_file(click_rec.id)
        assert fresh is not None and fresh.is_available is False
        assert "离线" in win.inspector.meta_label.text()

        # Toggling 隐藏离线 immediately hides click_short.wav while keeping confirm_mid.wav
        win.filter_toggle_btn.setChecked(True)
        win.filter_online_cb.setChecked(True)
        _pump(app, timeout_sec=2.0, cond=lambda: win.table_model.rowCount() == 2)
        visible_names = {
            win.table_model.record_at(i).filename for i in range(win.table_model.rowCount())
        }
        assert visible_names == {"confirm_mid.wav", "explosion_long.wav"}
    finally:
        win.close()
        app.processEvents()


def test_directory_browse_shows_subfolders_and_search_shows_audio_files(tmp_path: Path) -> None:
    app = create_app()
    lib_dir = tmp_path / "me_root"
    _write_wav(lib_dir / "demo_top.wav", duration_sec=0.2)
    _write_wav(lib_dir / "ui" / "click_short.wav", duration_sec=0.25)
    _write_wav(lib_dir / "ui" / "confirm_mid.wav", duration_sec=0.4)
    _write_wav(lib_dir / "sfx" / "explosion_long.wav", duration_sec=0.5)

    db_path = tmp_path / "me_browse.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(lib_dir)
    store.close()
    LibraryScanner(db_path).scan_root(root.id)

    win = BrowserWindow(db_path, audition=_FakeAudition(), spot_client=_FakeSpotClient())
    win.show()
    _pump(app, timeout_sec=2.0, cond=lambda: win.table_model.rowCount() == 4)
    try:
        # Select root in FolderTree -> middle table shows 2 subfolders + 1 root file (NOT child files!)
        root_item = win.folder_tree.find_root_item(root.id)
        assert root_item is not None
        win.folder_tree.setCurrentItem(root_item)
        _pump(
            app,
            timeout_sec=2.0,
            cond=lambda: win.table_model.folder_count == 2 and win.table_model.rowCount() == 3,
        )
        assert win.table_model.folder_at(0)[0].name == "sfx"
        assert win.table_model.folder_at(1)[0].name == "ui"
        assert win.table_model.record_at(0) is None
        assert win.table_model.record_at(2).filename == "demo_top.wav"

        # Double-clicking / activating subfolder 'ui' drills down and shows ui's audio files
        ui_folder = win.table_model.folder_at(1)[0]
        win._on_table_folder_activated(ui_folder)
        _pump(
            app,
            timeout_sec=2.0,
            cond=lambda: win.table_model.folder_count == 0 and win.table_model.rowCount() == 2,
        )
        assert {
            win.table_model.record_at(i).filename for i in range(win.table_model.rowCount())
        } == {"click_short.wav", "confirm_mid.wav"}

        # Go back to root, then type a search query -> immediately switches to recursive audio results
        win.go_back()
        _pump(
            app,
            timeout_sec=2.0,
            cond=lambda: win.table_model.folder_count == 2 and win.table_model.rowCount() == 3,
        )
        win.search_input.setText("click")
        _pump(
            app,
            timeout_sec=2.0,
            cond=lambda: win.table_model.folder_count == 0 and win.table_model.rowCount() == 1,
        )
        assert win.table_model.record_at(0).filename == "click_short.wav"

        # Clearing search restores directory browse view with folders + root file
        win.search_input.clear()
        _pump(
            app,
            timeout_sec=2.0,
            cond=lambda: win.table_model.folder_count == 2 and win.table_model.rowCount() == 3,
        )
    finally:
        win.close()
        app.processEvents()


def test_local_divider_collapsible_and_table_drilldown_keeps_tree_collapsed(
    tmp_path: Path,
) -> None:
    app = create_app()
    lib_dir = tmp_path / "me_root"
    _write_wav(lib_dir / "demo_top.wav", duration_sec=0.2)
    _write_wav(lib_dir / "ui" / "nested" / "sub_beep.wav", duration_sec=0.3)
    _write_wav(lib_dir / "sfx" / "boom.wav", duration_sec=0.4)

    db_path = tmp_path / "me_collapse.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(lib_dir)
    store.close()
    LibraryScanner(db_path).scan_root(root.id)

    win = BrowserWindow(db_path, audition=_FakeAudition(), spot_client=_FakeSpotClient())
    win.show()
    _pump(app, timeout_sec=2.0, cond=lambda: win.table_model.rowCount() == 3)
    try:
        local_div = win.folder_tree.find_local_divider_item()
        assert local_div is not None
        assert local_div.childCount() == 1
        assert local_div.isExpanded() is False

        # Clicking the 本地目录 row toggles expanding & collapsing all local roots
        div_rect = win.folder_tree.visualItemRect(local_div)
        QTest.mouseClick(win.folder_tree.viewport(), Qt.MouseButton.LeftButton, pos=div_rect.center())
        _pump(app, timeout_sec=0.5, cond=lambda: local_div.isExpanded() is True)
        assert local_div.isExpanded() is True

        QTest.mouseClick(win.folder_tree.viewport(), Qt.MouseButton.LeftButton, pos=div_rect.center())
        _pump(app, timeout_sec=0.5, cond=lambda: local_div.isExpanded() is False)
        assert local_div.isExpanded() is False

        QTest.mouseClick(win.folder_tree.viewport(), Qt.MouseButton.LeftButton, pos=div_rect.center())
        _pump(app, timeout_sec=0.5, cond=lambda: local_div.isExpanded() is True)
        assert local_div.isExpanded() is True

        # Select root (kept collapsed), then drill down ui -> nested from middle table
        root_item = win.folder_tree.find_root_item(root.id)
        assert root_item is not None
        win.folder_tree.setCurrentItem(root_item)
        _pump(
            app,
            timeout_sec=2.0,
            cond=lambda: win.table_model.folder_count == 2 and win.table_model.rowCount() == 3,
        )
        assert root_item.isExpanded() is False

        ui_folder = win.table_model.folder_at(1)[0]
        assert ui_folder.name == "ui"
        win._on_table_folder_activated(ui_folder)
        _pump(
            app,
            timeout_sec=2.0,
            cond=lambda: win.table_model.folder_count == 1 and win.table_model.rowCount() == 1,
        )
        assert root_item.isExpanded() is False
        assert win.folder_tree.currentItem() is root_item

        nested_folder = win.table_model.folder_at(0)[0]
        assert nested_folder.name == "nested"
        win._on_table_folder_activated(nested_folder)
        _pump(
            app,
            timeout_sec=2.0,
            cond=lambda: (
                win.table_model.folder_count == 0
                and win.table_model.rowCount() == 1
                and win.table_model.record_at(0) is not None
                and win.table_model.record_at(0).filename == "sub_beep.wav"
            ),
        )
        assert root_item.isExpanded() is False
        assert win.folder_tree.currentItem() is root_item

        # Clicking the already-highlighted ancestor root_item returns back up to root_item
        root_rect = win.folder_tree.visualItemRect(root_item)
        QTest.mouseClick(
            win.folder_tree.viewport(), Qt.MouseButton.LeftButton, pos=root_rect.center()
        )
        _pump(
            app,
            timeout_sec=2.0,
            cond=lambda: win.table_model.folder_count == 2 and win.table_model.rowCount() == 3,
        )
        assert root_item.isExpanded() is False

        # Collapsing 本地目录 preserves collapsed state across reload()
        QTest.mouseClick(win.folder_tree.viewport(), Qt.MouseButton.LeftButton, pos=div_rect.center())
        _pump(app, timeout_sec=0.5, cond=lambda: local_div.isExpanded() is False)
        win.folder_tree.reload(keep_location=win._location)
        _pump(app, timeout_sec=0.2)
        reloaded_div = win.folder_tree.find_local_divider_item()
        reloaded_root = win.folder_tree.find_root_item(root.id)
        assert reloaded_div is not None and reloaded_div.isExpanded() is False
        assert reloaded_root is not None and reloaded_root.isExpanded() is False
    finally:
        win.close()
        app.processEvents()


def test_mark_column_per_display_lifecycle_toggle_clear_and_migration(
    tmp_path: Path,
) -> None:
    app, win, db_path, _lib_dir = _setup_window(tmp_path)
    try:
        # 1. Default visual layout: Mark placed directly left of 名称 (visualIndex 0 -> Mark, 1 -> 名称, 2 -> 波形)
        header = win.table_view.horizontalHeader()
        assert header.visualIndex(MARK_COL) == 0
        assert header.visualIndex(0) == 1
        assert header.visualIndex(WAVEFORM_COL) == 2

        # 2. Initially unmarked
        for r in range(win.table_model.rowCount()):
            assert win.table_model.data(win.table_model.index(r, MARK_COL)) == ""

        # 3. With Autoplay OFF, selecting row 0 does NOT mark it; pressing Play marks row 0 only
        rec0 = win.table_model.record_at(0)
        rec1 = win.table_model.record_at(1)
        assert rec0 is not None and rec1 is not None
        win.autoplay_btn.setChecked(False)
        win._on_audition_row(0)
        assert win.table_model.data(win.table_model.index(0, MARK_COL)) == ""

        win._on_play_button()
        assert win.table_model.data(win.table_model.index(0, MARK_COL)) == "•"
        assert win.table_model.data(win.table_model.index(1, MARK_COL)) == ""

        # 4. With Autoplay ON, auditioning row 1 marks row 1; sorting SAME display preserves marks
        win.autoplay_btn.setChecked(True)
        win._on_audition_row(1)
        assert win.table_model.data(win.table_model.index(1, MARK_COL)) == "•"

        win.table_model.sort(MARK_COL, Qt.SortOrder.DescendingOrder)
        header.sectionClicked.emit(MARK_COL)
        assert win.table_model.sort_column == "filename"

        win.table_model.sort(0, Qt.SortOrder.DescendingOrder)
        _pump(
            app,
            timeout_sec=2.0,
            cond=lambda: (
                win.table_model.record_at(0) is not None
                and win.table_model.record_at(0).filename == "explosion_long.wav"
            ),
        )
        for r in range(win.table_model.rowCount()):
            rec = win.table_model.record_at(r)
            assert rec is not None
            expected = "•" if rec.id in (rec0.id, rec1.id) else ""
            assert win.table_model.data(win.table_model.index(r, MARK_COL)) == expected

        # 5. Searching starts a fresh display (clears marks); marking in search & clearing search clears again
        win.search_input.setText("click")
        _pump(app, timeout_sec=1.5, cond=lambda: win.table_model.rowCount() == 1)
        assert win.table_model.data(win.table_model.index(0, MARK_COL)) == ""
        assert win.table_model.played_mark_count() == 0

        win._on_audition_row(0)
        assert win.table_model.data(win.table_model.index(0, MARK_COL)) == "•"
        win.search_input.clear()
        _pump(app, timeout_sec=1.5, cond=lambda: win.table_model.rowCount() == 3)
        assert win.table_model.played_mark_count() == 0

        # 6. Directory navigation (drill-down -> mark -> Back -> Forward) clears marks!
        root = win.store.list_roots()[0]
        root_item = win.folder_tree.find_root_item(root.id)
        assert root_item is not None
        win.folder_tree.setCurrentItem(root_item)
        _pump(app, timeout_sec=2.0, cond=lambda: win.table_model.folder_count == 2)
        assert win.table_model.data(win.table_model.index(0, MARK_COL)) is None

        ui_folder = next(
            win.table_model.folder_at(i)[0]
            for i in range(win.table_model.folder_count)
            if win.table_model.folder_at(i)[0].name == "ui"
        )
        win._on_table_folder_activated(ui_folder)
        _pump(
            app,
            timeout_sec=2.0,
            cond=lambda: win.table_model.folder_count == 0 and win.table_model.rowCount() == 2,
        )
        ui_rec0 = win.table_model.record_at(0)
        assert ui_rec0 is not None
        win._on_audition_row(0)
        assert win.table_model.data(win.table_model.index(0, MARK_COL)) == "•"

        win.go_back()
        _pump(app, timeout_sec=2.0, cond=lambda: win.table_model.folder_count == 2)
        assert win.table_model.played_mark_count() == 0

        win.go_forward()
        _pump(
            app,
            timeout_sec=2.0,
            cond=lambda: win.table_model.folder_count == 0 and win.table_model.rowCount() == 2,
        )
        assert win.table_model.played_mark_count() == 0
        assert win.table_model.data(win.table_model.index(0, MARK_COL)) == ""

        # 7. Switching sidebar view clears marks; manual toggle & clear all work
        win._on_audition_row(0)
        assert win.table_model.is_played_marked(ui_rec0.id) is True
        win._on_sidebar_view_selected(SidebarViewTarget(KIND_FAVORITES, display_path="我的收藏"))
        _pump(app, timeout_sec=1.5, cond=lambda: win.table_model.rowCount() == 0)
        assert win.table_model.played_mark_count() == 0

        all_item = win.folder_tree.find_quick_item("all")
        assert all_item is not None
        win.folder_tree.setCurrentItem(all_item)
        _pump(app, timeout_sec=2.0, cond=lambda: win.table_model.rowCount() == 3)

        win.table_view.setPlayedMarkRequested.emit([rec0.id], True)
        assert win.table_model.is_played_marked(rec0.id) is True
        win.table_view.setPlayedMarkRequested.emit([rec0.id], False)
        assert win.table_model.is_played_marked(rec0.id) is False
        win.table_view.setPlayedMarkRequested.emit([rec0.id, rec1.id], True)
        assert win.table_model.played_mark_count() == 2
        win.table_view.clearPlayedMarksRequested.emit()
        assert win.table_model.played_mark_count() == 0
    finally:
        win.close()
        app.processEvents()

    # 8. Legacy column_order without "mark" migrates Mark directly before filename
    s_path = settings_path_for_db(db_path)
    legacy = json.loads(s_path.read_text(encoding="utf-8"))
    legacy["column_order"] = [k for k, _, _ in COLUMNS if k != "mark"]
    s_path.write_text(json.dumps(legacy), encoding="utf-8")

    win2 = BrowserWindow(db_path, audition=_FakeAudition(), spot_client=_FakeSpotClient())
    win2.show()
    _pump(app, timeout_sec=2.0, cond=lambda: win2.table_model.rowCount() == 3)
    try:
        assert win2.table_model.played_mark_count() == 0
        h2 = win2.table_view.horizontalHeader()
        assert h2.visualIndex(MARK_COL) == 0
        assert h2.visualIndex(0) == 1
    finally:
        win2.close()
        app.processEvents()


def test_ui_tracking_fixes_filter_jitter_inspector_play_btn_and_title_bar(tmp_path: Path) -> None:
    app, win, _db_path, _lib_dir = _setup_window(tmp_path)
    try:
        # 1. Custom dark title bar controls exist
        assert win.title_bar.isVisible()
        assert win.title_bar.min_btn is not None
        assert win.title_bar.max_btn is not None
        assert win.title_bar.close_btn is not None

        # 2. Toggling filter panel open & closed must not shift tree, inspector, or waveform
        _pump(app, timeout_sec=0.1)
        y_tree_0 = win.folder_tree.mapTo(win, QPoint(0, 0)).y()
        y_insp_0 = win.inspector.mapTo(win, QPoint(0, 0)).y()
        y_wave_0 = win.waveform.mapTo(win, QPoint(0, 0)).y()

        win.filter_toggle_btn.setChecked(True)
        _pump(app, timeout_sec=0.1)
        assert win.filter_bar.isVisible() is True
        assert win.folder_tree.mapTo(win, QPoint(0, 0)).y() == y_tree_0
        assert win.inspector.mapTo(win, QPoint(0, 0)).y() == y_insp_0
        assert win.waveform.mapTo(win, QPoint(0, 0)).y() == y_wave_0

        win.filter_toggle_btn.setChecked(False)
        _pump(app, timeout_sec=0.1)
        assert win.filter_bar.isVisible() is False
        assert win.folder_tree.mapTo(win, QPoint(0, 0)).y() == y_tree_0
        assert win.inspector.mapTo(win, QPoint(0, 0)).y() == y_insp_0
        assert win.waveform.mapTo(win, QPoint(0, 0)).y() == y_wave_0

        # 3. Inspector core info card displays promoted brand / path & keeps meta_label
        rec0 = win.table_model.record_at(0)
        assert rec0 is not None
        win.store.update_file_metadata(
            rec0.id,
            {"designer": "Rock The Speakerbox", "library": "SC Physics"},
            source="user",
        )
        fresh = win.store.get_file(rec0.id)
        assert fresh is not None
        win.inspector.show_file(fresh, win.store.get_metadata_fields(fresh.id))
        assert win.inspector.info_card.isVisible() is True
        assert win.inspector.brand_value_label.text() == "Rock The Speakerbox"
        assert win.inspector.package_value_label.text() == "SC Physics"
        assert Path(fresh.rel_path).parent.name in win.inspector.path_breadcrumb_label.text()
        assert "#E89274" in win.inspector.path_breadcrumb_label.text()
        assert win.inspector.path_value_label.text() == str(fresh.full_path.parent)
        assert "Rock The Speakerbox" in win.inspector.meta_label.text()

        # Horizontal scrolling must be per-pixel
        assert win.table_view.horizontalScrollMode() == QAbstractItemView.ScrollMode.ScrollPerPixel

        # 4. Play button uses vector icon & Space -> Space -> Esc -> Space cleanly replays
        assert win.play_btn.text() == ""
        assert bool(win.play_btn.property("playing")) is False
        win._on_audition_row(0)
        win.table_view.setFocus()
        _pump(app, timeout_sec=0.1)
        assert bool(win.play_btn.property("playing")) is True
        plays_after_row = len(win.audition.play_calls)

        space_ev = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier)
        assert win._handle_key_press(space_ev) is True
        assert win.audition.state == "paused"
        assert bool(win.play_btn.property("playing")) is False
        assert win._handle_key_press(space_ev) is True
        assert win.audition.state == "playing"

        esc_ev = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
        assert win._handle_key_press(esc_ev) is True
        assert win.audition.state in ("paused", "empty")
        assert bool(win.play_btn.property("playing")) is False

        assert win._handle_key_press(space_ev) is True
        _pump(app, timeout_sec=0.1)
        assert win.audition.state == "playing"
        assert bool(win.play_btn.property("playing")) is True
        assert len(win.audition.play_calls) == plays_after_row + 1

        # 5. PageLoader eviction never leaks evicted pages in _in_flight_pages
        win.table_model._file_total_count = 2000
        for p_idx in range(1, 7):
            win.table_model.request_page_async(p_idx, force=True)
        assert len(win.table_model._in_flight_pages) <= 5
        _pump(app, timeout_sec=1.0, cond=lambda: not win.table_model._in_flight_pages)
        assert not win.table_model._in_flight_pages
    finally:
        win.close()
        app.processEvents()


def test_global_scope_plan_a_path_search_and_jump_to_folder(tmp_path: Path) -> None:
    app = create_app()
    lib_dir = tmp_path / "Sound Effects"
    _write_wav(lib_dir / "PackA" / "alpha_one.wav", duration_sec=0.3, freq=440.0)
    _write_wav(lib_dir / "PackA" / "Sub" / "sub_item.wav", duration_sec=0.3, freq=520.0)
    _write_wav(lib_dir / "PackC" / "orphan.wav", duration_sec=0.3, freq=600.0)
    # Seed > 200 files in PackB so target file sits on page 2 (row >= 200)
    for i in range(215):
        _write_wav(lib_dir / "PackB" / f"tone_{i:03d}.wav", duration_sec=0.2, freq=300.0)
    target_path = lib_dir / "PackB" / "tone_210.wav"

    db_path = tmp_path / "nav_jump.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(lib_dir)
    LibraryScanner(db_path).scan_root(root.id)
    folders = store.list_folders(root.id)
    pack_a_folder = next(f for f in folders if f.name == "PackA")
    pack_b_folder = next(f for f in folders if f.name == "PackB")
    coll = store.create_collection("常用快捷")
    store.add_folder_to_collection(coll.id, root.id, pack_a_folder.id, pack_a_folder.rel_path, label="PackA")
    store.add_folder_to_collection(coll.id, root.id, pack_b_folder.id, pack_b_folder.rel_path, label="PackB")
    sub_rec_id = next(
        r.id for r in store.query_page(offset=0, limit=10, query="sub_item.wav")
    )
    orphan_rec_id = next(
        r.id for r in store.query_page(offset=0, limit=10, query="orphan.wav")
    )
    store.close()

    fake_audition = _FakeAudition()
    win = BrowserWindow(db_path, audition=fake_audition, spot_client=_FakeSpotClient())
    win.show()
    _pump(app, timeout_sec=2.0, cond=lambda: win.table_model.rowCount() == 218)
    try:
        # 1. Plan A: Turn ON global scope; empty search box + click shortcut browses ONLY that folder
        win.scope_btn.setChecked(True)
        assert win.scope_btn.isChecked() is True

        coll_item = next(
            win.folder_tree.topLevelItem(i)
            for i in range(win.folder_tree.topLevelItemCount())
            if win.folder_tree.topLevelItem(i).text(0) == "常用快捷"
        )
        cf_a = coll_item.child(0)
        cf_b = coll_item.child(1)
        assert cf_a.text(0) == "PackA" and cf_b.text(0) == "PackB"

        win.folder_tree.setCurrentItem(cf_a)
        _pump(
            app,
            timeout_sec=2.0,
            cond=lambda: win.table_model.folder_count == 1 and win.table_model.rowCount() == 2,
        )
        assert win.table_model.record_at(1).filename == "alpha_one.wav"

        # 2. With global scope ON, typing keyword searches whole library
        win.search_input.setText("tone_210")
        _pump(
            app,
            timeout_sec=2.0,
            cond=lambda: win.table_model.rowCount() == 1
            and win.table_model.record_at(0) is not None
            and win.table_model.record_at(0).filename == "tone_210.wav",
        )

        # 3. Clicking the ALREADY-highlighted shortcut (cf_a) clears search & returns to PackA!
        cf_a_rect = win.folder_tree.visualItemRect(cf_a)
        QTest.mouseClick(win.folder_tree.viewport(), Qt.MouseButton.LeftButton, pos=cf_a_rect.center())
        _pump(
            app,
            timeout_sec=2.0,
            cond=lambda: win.search_input.text() == ""
            and win.table_model.folder_count == 1
            and win.table_model.rowCount() == 2,
        )

        # 4. Clicking "全部素材" keeps global search text active
        win.search_input.setText("tone_210")
        _pump(app, timeout_sec=2.0, cond=lambda: win.table_model.rowCount() == 1)
        all_item = win.folder_tree.find_quick_item("all")
        assert all_item is not None
        win.folder_tree.setCurrentItem(all_item)
        _pump(app, timeout_sec=1.0, cond=lambda: win.table_model.rowCount() == 1)
        assert win.search_input.text() == "tone_210"

        # 5. Turning OFF global scope restricts search to current shortcut (PackA has 0 tone_210)
        win.scope_btn.setChecked(False)
        win.folder_tree.setCurrentItem(cf_a)
        _pump(app, timeout_sec=2.0, cond=lambda: win.table_model.rowCount() == 0)
        win.search_input.clear()
        _pump(app, timeout_sec=2.0, cond=lambda: win.table_model.rowCount() == 2)

        # 6. Shortcut navigation records history & enables up_btn
        assert win.up_btn.isEnabled() is True
        win.go_up()
        _pump(
            app,
            timeout_sec=2.0,
            cond=lambda: win.table_model.folder_count == 3 and win.table_model.rowCount() == 3,
        )
        win.go_back()
        _pump(
            app,
            timeout_sec=2.0,
            cond=lambda: win.table_model.folder_count == 1 and win.table_model.rowCount() == 2,
        )

        # 7. Path search in search_input (from Favorites view, with quotes & Windows backslashes)
        fav_item = win.folder_tree.find_quick_item(KIND_FAVORITES)
        assert fav_item is not None
        win.folder_tree.setCurrentItem(fav_item)
        _pump(app, timeout_sec=1.5, cond=lambda: win.table_model.rowCount() == 0)

        win.search_input.setText(f'"{str(target_path).replace("/", "\\")}"')
        _pump(
            app,
            timeout_sec=2.0,
            cond=lambda: win.table_model.rowCount() == 1
            and win.table_model.record_at(0) is not None
            and win.table_model.record_at(0).filename == "tone_210.wav",
        )
        target_rec = win.table_model.record_at(0)
        assert target_rec is not None

        # Relative backslash search works too
        win.search_input.setText(r"PackB\tone_210")
        win.folder_tree.setCurrentItem(all_item)
        _pump(
            app,
            timeout_sec=2.0,
            cond=lambda: win.table_model.rowCount() == 1
            and win.table_model.record_at(0) is not None
            and win.table_model.record_at(0).id == target_rec.id,
        )

        # 8. Jump to folder: prefer shortcut, otherwise highlight 本地目录 without expanding
        local_div = win.folder_tree.find_local_divider_item()
        assert local_div is not None
        win.folder_tree.set_local_roots_expanded(False)
        assert local_div.isExpanded() is False

        # Collapse collection first to confirm jumping expands the matching collection to reveal cf_b
        win.folder_tree.collapse_all_collections()
        assert coll_item.isExpanded() is False

        # Apply an ext filter that would hide wav if not cleared (use ogg)
        ogg_idx = win.filter_ext_combo.findData("ogg")
        win.filter_ext_combo.setCurrentIndex(ogg_idx)
        plays_before = len(fake_audition.play_calls)

        # 8a. Exact shortcut match (PackB): highlights cf_b, keeps 本地目录 collapsed, reveals row 210
        win.table_view.navigateToFileFolderRequested.emit(target_rec.id)
        _pump(
            app,
            timeout_sec=3.0,
            cond=lambda: (
                win.table_model.rowCount() == 215
                and win.table_view.currentIndex().isValid()
                and win.table_view.currentIndex().row() == 210
            ),
        )
        assert win.search_input.text() == ""
        assert win.filter_ext_combo.currentIndex() == 0
        assert local_div.isExpanded() is False
        assert win.folder_tree.is_local_divider_highlighted() is False
        assert coll_item.isExpanded() is True
        assert win.folder_tree.currentItem() is cf_b
        assert win.table_model.record_at(210).id == target_rec.id
        assert len(fake_audition.play_calls) == plays_before

        # 8b. Ancestor shortcut match (PackA/Sub -> cf_a): keeps 本地目录 & cf_a collapsed, clicking cf_a goes back up to PackA
        win.table_view.navigateToFileFolderRequested.emit(sub_rec_id)
        _pump(
            app,
            timeout_sec=2.0,
            cond=lambda: (
                win.table_model.folder_count == 0
                and win.table_model.rowCount() == 1
                and win.table_model.record_at(0) is not None
                and win.table_model.record_at(0).id == sub_rec_id
            ),
        )
        assert local_div.isExpanded() is False
        assert cf_a.isExpanded() is False
        assert win.folder_tree.currentItem() is cf_a
        assert win.table_view.currentIndex().row() == 0
        QTest.mouseClick(
            win.folder_tree.viewport(),
            Qt.MouseButton.LeftButton,
            pos=win.folder_tree.visualItemRect(cf_a).center(),
        )
        _pump(
            app,
            timeout_sec=2.0,
            cond=lambda: win.table_model.folder_count == 1 and win.table_model.rowCount() == 2,
        )

        # 8c. Unmounted folder (PackC) with 本地目录 collapsed: keeps 本地目录 collapsed & highlights divider
        win.table_view.navigateToFileFolderRequested.emit(orphan_rec_id)
        _pump(
            app,
            timeout_sec=2.0,
            cond=lambda: (
                win.table_model.folder_count == 0
                and win.table_model.rowCount() == 1
                and win.table_model.record_at(0) is not None
                and win.table_model.record_at(0).id == orphan_rec_id
            ),
        )
        assert local_div.isExpanded() is False
        assert win.folder_tree.currentItem() is None
        assert win.folder_tree.is_local_divider_highlighted() is True
        assert win.table_view.currentIndex().row() == 0

        # 8d. Unmounted folder (PackC) with 本地目录 expanded (root collapsed): highlights root without expanding root
        win.folder_tree.set_local_roots_expanded(True)
        root_item = win.folder_tree.find_root_item(root.id)
        assert root_item is not None and root_item.isExpanded() is False
        win.folder_tree.setCurrentItem(all_item)
        _pump(app, timeout_sec=1.5, cond=lambda: win.table_model.rowCount() == 218)

        win.table_view.navigateToFileFolderRequested.emit(orphan_rec_id)
        _pump(
            app,
            timeout_sec=2.0,
            cond=lambda: (
                win.table_model.folder_count == 0
                and win.table_model.rowCount() == 1
                and win.table_model.record_at(0) is not None
                and win.table_model.record_at(0).id == orphan_rec_id
            ),
        )
        assert root_item.isExpanded() is False
        assert win.folder_tree.currentItem() is root_item
        assert win.folder_tree.is_local_divider_highlighted() is False

        # 8e. Inspector path action buttons style & trigger via "跳转目录"
        assert win.inspector.jump_folder_btn.property("variant") == "cardAction"
        assert win.inspector.reveal_folder_btn.property("variant") == "cardAction"
        assert win.inspector.copy_path_btn.property("variant") == "cardAction"
        assert win.inspector.reveal_folder_btn.text() == "资源管理器"
        assert win.inspector.copy_path_btn.text() == "复制路径"
        win.inspector.show_file(target_rec, [])
        win.folder_tree.setCurrentItem(fav_item)
        _pump(app, timeout_sec=1.5, cond=lambda: win.table_model.rowCount() == 0)
        win.inspector.jump_folder_btn.click()
        _pump(
            app,
            timeout_sec=3.0,
            cond=lambda: (
                win.table_model.rowCount() == 215
                and win.table_view.currentIndex().isValid()
                and win.table_view.currentIndex().row() == 210
            ),
        )
        assert win.folder_tree.currentItem() is cf_b
        assert len(fake_audition.play_calls) == plays_before
        assert KIND_COLLECTION_FOLDER == "collection_folder"
    finally:
        win.close()
        app.processEvents()


def test_sidebar_branch_hitbox_and_settings_dialog_live_scale_and_audio_device(
    tmp_path: Path, monkeypatch
) -> None:
    from PySide6.QtWidgets import QPushButton

    from audiosearch.audition.sink import OutputDeviceInfo
    from audiosearch.browser.settings_dialog import SettingsDialog

    class _AuditionWithDevices(_FakeAudition):
        def __init__(self) -> None:
            super().__init__()
            self.active_device_name = "Speakers (Realtek)"
            self.device_fallback = False
            self.refresh_calls: list[tuple[str, bool]] = []

        def list_output_devices(self):
            return [
                OutputDeviceInfo(0, "Speakers (Realtek)", "Windows WASAPI", 48000, 2, is_default=True),
                OutputDeviceInfo(1, "Scarlett 4i4 USB", "Windows WASAPI", 96000, 4, is_default=False),
            ]

        def set_output_device(self, name: str | None, *, refresh_portaudio: bool = False):
            req = (name or "").strip()
            self.refresh_calls.append((req, refresh_portaudio))
            if req == "Scarlett 4i4 USB":
                self.active_device_name = "Scarlett 4i4 USB"
                self.output_rate_hz = 96000
                self.device_fallback = False
                return True, self.active_device_name
            self.active_device_name = "Speakers (Realtek)"
            self.output_rate_hz = 48000
            self.device_fallback = False
            return True, self.active_device_name

    app = create_app()
    lib_dir = tmp_path / "audio_root_009_010"
    _write_wav(lib_dir / "Magic" / "spell.wav", duration_sec=0.25)
    db_path = tmp_path / "ui_009_010.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(lib_dir)
    store.close()
    LibraryScanner(db_path).scan_root(root.id)

    aud = _AuditionWithDevices()
    win = BrowserWindow(db_path, audition=aud, spot_client=_FakeSpotClient())
    win.show()
    _pump(app, timeout_sec=2.0, cond=lambda: win.table_model.rowCount() == 1)
    try:
        # #009: Left safe margin & enlarged indentation + clickable branch hit rect
        assert win.folder_tree.contentsRect().left() >= 10
        assert win.folder_tree.indentation() >= 18
        assert win.title_bar.settings_btn is not None
        assert win.title_bar.is_interactive_child(win.title_bar.settings_btn.geometry().center()) is True
        assert isinstance(win.status_audio, QPushButton)

        local_div = win.folder_tree.find_local_divider_item()
        assert local_div is not None
        win.folder_tree.set_local_roots_expanded(False)
        assert local_div.isExpanded() is False
        div_idx = win.folder_tree.indexFromItem(local_div)
        hit_rect = win.folder_tree._branch_hit_rect(div_idx)
        assert hit_rect.width() >= 20
        QTest.mouseClick(win.folder_tree.viewport(), Qt.MouseButton.LeftButton, pos=hit_rect.center())
        _pump(app, timeout_sec=0.1)
        assert local_div.isExpanded() is True
        QTest.mouseClick(win.folder_tree.viewport(), Qt.MouseButton.LeftButton, pos=hit_rect.center())
        _pump(app, timeout_sec=0.1)
        assert local_div.isExpanded() is False

        # #010: SettingsDialog live 4-tier scaling, audio device switch, AI limit sync & reset
        dlg = SettingsDialog(win, initial_tab="appearance", parent=win)
        try:
            assert set(dlg.scale_buttons.keys()) == {"compact", "standard", "large", "xlarge"}
            assert dlg.scale_buttons["standard"].isChecked() is True

            dlg.scale_buttons["xlarge"].click()
            _pump(app, timeout_sec=0.05)
            assert win._ui_scale == "xlarge"
            assert win.table_view.verticalHeader().defaultSectionSize() == 36
            assert win.folder_tree.indentation() == 23
            assert win.volume_knob.sizeHint().height() == 34
            assert "font-size: 16px;" in app.styleSheet()

            dlg.scale_buttons["compact"].click()
            _pump(app, timeout_sec=0.05)
            assert win._ui_scale == "compact"
            assert win.table_view.verticalHeader().defaultSectionSize() == 25

            dlg.select_tab("audio")
            dev_idx = dlg.device_combo.findData("Scarlett 4i4 USB")
            assert dev_idx >= 0
            dlg.device_combo.setCurrentIndex(dev_idx)
            _pump(app, timeout_sec=0.05)
            assert win._output_device_name == "Scarlett 4i4 USB"
            assert "Scarlett" in win.status_audio.text()
            assert "96kHz" in win.status_audio.text()
            assert "Scarlett 4i4 USB" in dlg.device_status_label.text()

            dlg.refresh_devices_btn.click()
            assert ("Scarlett 4i4 USB", True) in aud.refresh_calls

            dlg.select_tab("search")
            idx_2000 = dlg.ai_limit_combo.findData(2000)
            assert idx_2000 >= 0
            dlg.ai_limit_combo.setCurrentIndex(idx_2000)
            assert win.ai_limit_combo.currentData() == 2000

            dlg.reset_defaults_btn.click()
            assert win._ui_scale == "standard"
            assert win._output_device_name == ""
            assert "48kHz" in win.status_audio.text()
        finally:
            dlg.close()

        opened_tabs: list[str] = []
        monkeypatch.setattr(win, "_open_settings_dialog", lambda tab="appearance": opened_tabs.append(tab))
        comma_ev = QKeyEvent(
            QKeyEvent.Type.KeyPress, Qt.Key.Key_Comma, Qt.KeyboardModifier.ControlModifier
        )
        assert win._handle_key_press(comma_ev) is True
        win.status_audio.click()
        assert opened_tabs == ["appearance", "audio"]
    finally:
        win.close()
        app.processEvents()
