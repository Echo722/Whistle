"""Behavioral tests for debounced auto-save, sidebar collapse persistence, hidden column widths, and maximized window restore."""

from __future__ import annotations

import json
import os
import time
import wave
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from audiosearch.browser.file_table import COLUMNS
from audiosearch.browser.main_window import BrowserWindow, create_app
from audiosearch.browser.settings import load_settings
from audiosearch.library.scanner import LibraryScanner
from audiosearch.library.store import LibraryStore
from tests.ai.test_browser_ai_ui import _FakeAudition, _wait_until

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _make_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(np.zeros(1600, dtype="<i2").tobytes())


def _pump_for(seconds: float) -> None:
    app = QApplication.instance()
    assert app is not None
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.015)


def test_debounced_autosave_without_close_and_sidebar_restore(tmp_path: Path):
    create_app([])
    sfx = tmp_path / "sfx"
    _make_wav(sfx / "Magic" / "spell_01.wav")
    _make_wav(sfx / "UI" / "click_01.wav")

    db_path = tmp_path / "persist.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(sfx)
    LibraryScanner(db_path).scan_root(root.id)
    coll = store.create_collection("常用音效")
    store.close()

    win = BrowserWindow(db_path, audition=_FakeAudition())
    win.show()
    _wait_until(lambda: win.table_model.rowCount() == 2)

    try:
        # 1. Expand 本地目录 and collapse collection in sidebar tree
        local_div = win.folder_tree.find_local_divider_item()
        assert local_div is not None
        local_div.setExpanded(True)
        for i in range(win.folder_tree.topLevelItemCount()):
            item = win.folder_tree.topLevelItem(i)
            if item.text(0) == "常用音效":
                item.setExpanded(False)
                break

        # 2. Toggle global scope
        win.scope_btn.setChecked(True)

        # 3. Resize filename column and hide package column
        col_keys = [c[0] for c in COLUMNS]
        fn_idx = col_keys.index("filename")
        pkg_idx = col_keys.index("package")
        win.table_view.setColumnWidth(fn_idx, 345)
        win.table_view.setColumnWidth(pkg_idx, 180)
        win.table_view.setColumnHidden(pkg_idx, True)
        win.table_view.columnVisibilityChanged.emit()

        # Wait past 400ms debounce WITHOUT calling close() (simulating abrupt process kill)
        _pump_for(0.55)

        settings_file = db_path.parent / "browser-settings.json"
        assert settings_file.is_file()
        raw = json.loads(settings_file.read_text(encoding="utf-8"))
        assert raw["column_widths"]["filename"] == 345
        assert raw["column_widths"]["package"] == 180
        assert raw["column_widths"]["relevance"] >= 24
        assert raw["column_widths"]["hit_reason"] >= 24
        assert raw["local_roots_expanded"] is True
        assert raw["collapsed_collections"] == [coll.id]
        assert raw["global_scope"] is True
        assert "package" in raw["hidden_columns"]
    finally:
        win.close()

    # Reopen a fresh window and verify everything restores cleanly
    win2 = BrowserWindow(db_path, audition=_FakeAudition())
    win2.show()
    _wait_until(lambda: win2.table_model.rowCount() == 2)
    try:
        assert win2.table_view.columnWidth(fn_idx) == 345
        assert win2.table_view.isColumnHidden(pkg_idx) is True
        assert win2.folder_tree.is_local_roots_expanded() is True
        assert win2.folder_tree.find_local_divider_item().isExpanded() is True
        assert win2.folder_tree.collapsed_collection_ids() == [coll.id]
        for i in range(win2.folder_tree.topLevelItemCount()):
            item = win2.folder_tree.topLevelItem(i)
            if item.text(0) == "常用音效":
                assert item.isExpanded() is False
        assert win2.scope_btn.isChecked() is True
        assert "全局" in win2.scope_btn.text()

        # Un-hiding package or switching to AI mode restores valid non-zero widths
        win2.table_view.set_ai_columns_visible(True)
        rel_idx = col_keys.index("relevance")
        hit_idx = col_keys.index("hit_reason")
        assert win2.table_view.columnWidth(rel_idx) >= 60
        assert win2.table_view.columnWidth(hit_idx) >= 100
    finally:
        win2.close()


def test_legacy_zero_hidden_widths_self_heal_and_maximized_persists(tmp_path: Path):
    create_app([])
    db_path = tmp_path / "legacy_fix.sqlite3"
    store = LibraryStore(db_path)
    store.close()

    settings_path = db_path.parent / "browser-settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "window_width": 1440,
                "window_height": 900,
                "window_maximized": True,
                "column_widths": {
                    "filename": 310,
                    "relevance": 0,
                    "hit_reason": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = load_settings(settings_path)
    assert loaded.column_widths == {"filename": 310}
    assert loaded.window_maximized is True

    win = BrowserWindow(db_path, audition=_FakeAudition())
    win.show()
    _pump_for(0.05)
    try:
        assert bool(win.windowState() & Qt.WindowState.WindowMaximized)
        collected = win._collect_settings()
        assert collected.column_widths["relevance"] >= 60
        assert collected.column_widths["hit_reason"] >= 100
        assert collected.window_maximized is True
        assert collected.window_width >= 640
        assert collected.window_height >= 480
    finally:
        win.close()


def test_keyword_and_ai_column_layouts_and_limit_persist_independently(tmp_path: Path):
    create_app([])
    sfx = tmp_path / "sfx_cols"
    _make_wav(sfx / "Magic" / "spell_01.wav")
    db_path = tmp_path / "cols.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(sfx)
    LibraryScanner(db_path).scan_root(root.id)
    store.close()

    col_keys = [c[0] for c in COLUMNS]
    rel_idx = col_keys.index("relevance")
    hit_idx = col_keys.index("hit_reason")
    fn_idx = col_keys.index("filename")
    path_idx = col_keys.index("rel_path")

    win = BrowserWindow(db_path, audition=_FakeAudition())
    win.show()
    _wait_until(lambda: win.table_model.rowCount() == 1)
    try:
        # 1. Customize Keyword mode column widths & visibility
        win.table_view.setColumnWidth(fn_idx, 310)
        win.table_view.setColumnWidth(path_idx, 420)
        assert win.table_view.isColumnHidden(rel_idx) is True

        # 2. Switch to AI mode and customize AI-specific widths + limit + hide rel_path only in AI
        win.mode_ai_btn.click()
        assert win.table_view.isColumnHidden(rel_idx) is False
        win.table_view.setColumnWidth(fn_idx, 240)
        win.table_view.setColumnWidth(rel_idx, 96)
        win.table_view.setColumnWidth(hit_idx, 230)
        win.table_view.setColumnHidden(path_idx, True)
        win.table_view.columnVisibilityChanged.emit()
        idx_1000 = win.ai_limit_combo.findData(1000)
        assert idx_1000 >= 0
        win.ai_limit_combo.setCurrentIndex(idx_1000)

        # 3. Switching back to Keyword restores Keyword widths & keeps rel_path visible
        win.mode_keyword_btn.click()
        assert win.table_view.isColumnHidden(rel_idx) is True
        assert win.table_view.isColumnHidden(path_idx) is False
        assert win.table_view.columnWidth(fn_idx) == 310
        assert win.table_view.columnWidth(path_idx) == 420

        # 4. Switching to AI again restores AI widths & hidden rel_path without touching Keyword
        win.mode_ai_btn.click()
        assert win.table_view.isColumnHidden(rel_idx) is False
        assert win.table_view.isColumnHidden(path_idx) is True
        assert win.table_view.columnWidth(fn_idx) == 240
        assert win.table_view.columnWidth(rel_idx) == 96
        assert win.table_view.columnWidth(hit_idx) == 230
    finally:
        win.close()

    saved = load_settings(db_path.parent / "browser-settings.json")
    assert saved.column_widths["filename"] == 310
    assert saved.column_widths["rel_path"] == 420
    assert "rel_path" not in saved.hidden_columns
    assert saved.ai_column_widths["filename"] == 240
    assert saved.ai_column_widths["relevance"] == 96
    assert saved.ai_column_widths["hit_reason"] == 230
    assert "rel_path" in saved.ai_hidden_columns
    assert saved.ai_result_limit == 1000

    # Reopen and verify both independent layouts survive restart cleanly
    win2 = BrowserWindow(db_path, audition=_FakeAudition())
    win2.show()
    _wait_until(lambda: win2.table_model.rowCount() == 1)
    try:
        assert win2.table_view.columnWidth(fn_idx) == 310
        assert win2.table_view.isColumnHidden(path_idx) is False
        assert win2.ai_limit_combo.currentData() == 1000
        win2.mode_ai_btn.click()
        assert win2.table_view.columnWidth(fn_idx) == 240
        assert win2.table_view.columnWidth(rel_idx) == 96
        assert win2.table_view.isColumnHidden(path_idx) is True
    finally:
        win2.close()


def test_ui_scale_and_output_device_persist_across_restart(tmp_path: Path):
    app = create_app([])
    sfx = tmp_path / "sfx_scale"
    _make_wav(sfx / "Magic" / "spell_01.wav")
    db_path = tmp_path / "scale.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(sfx)
    LibraryScanner(db_path).scan_root(root.id)
    store.close()

    settings_path = db_path.parent / "browser-settings.json"
    settings_path.write_text(json.dumps({"ui_scale": "invalid_key"}), encoding="utf-8")
    assert load_settings(settings_path).ui_scale == "standard"

    win = BrowserWindow(db_path, audition=_FakeAudition())
    win.show()
    _wait_until(lambda: win.table_model.rowCount() == 1)
    try:
        assert win._ui_scale == "standard"
        win.apply_ui_scale("large")
        win.set_output_device("Scarlett 4i4 USB")
        assert win.table_view.verticalHeader().defaultSectionSize() == 33
        assert win.folder_tree.indentation() == 21
        assert "font-size: 15px;" in app.styleSheet()
    finally:
        win.close()

    saved = load_settings(settings_path)
    assert saved.ui_scale == "large"
    assert saved.output_device_name == "Scarlett 4i4 USB"

    win2 = BrowserWindow(db_path, audition=_FakeAudition())
    win2.show()
    _wait_until(lambda: win2.table_model.rowCount() == 1)
    try:
        assert win2._ui_scale == "large"
        assert win2._output_device_name == "Scarlett 4i4 USB"
        assert win2.table_view.verticalHeader().defaultSectionSize() == 33
        assert win2.folder_tree.indentation() == 21
    finally:
        win2.close()
