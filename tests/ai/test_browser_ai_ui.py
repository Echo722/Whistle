"""Offscreen PySide6 behavioral tests for BrowserWindow R3 AI UI features."""

from __future__ import annotations

import os
import threading
import wave
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from audiosearch.ai.search_engine import AiSearchHit, AiSearchResponse
from audiosearch.browser.collections_tree import KIND_FAVORITES
from audiosearch.browser.file_table import COLUMNS
from audiosearch.browser.main_window import BrowserWindow, create_app
from audiosearch.browser.settings import load_settings, settings_path_for_db
from audiosearch.library.scanner import LibraryScanner
from audiosearch.library.store import LibraryStore

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class _FakeAudition(QObject):
    positionChanged = Signal(int, float)
    stateChanged = Signal(str)
    failed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.output_rate_hz = 48000
        self.device_error = None
        self.state = "empty"
        self.position_seconds = 0.0
        self.current_file_id: int | None = None
        self.play_calls: list[tuple[int, str, float]] = []

    def set_volume_db(self, _db: float) -> None:
        return None

    def set_loop(self, _enabled: bool) -> None:
        return None

    def set_selection(self, _sel) -> None:
        return None

    def set_pitch_rate(self, _pitch: float, _rate: float) -> None:
        return None

    def set_preserve_pitch(self, _enabled: bool) -> None:
        return None

    def play_file(
        self, file_id: int, path: Path, start_seconds: float = 0.0, *, end_seconds=None
    ) -> None:
        self.current_file_id = file_id
        self.play_calls.append((file_id, path.name, float(start_seconds)))
        self.state = "playing"
        self.position_seconds = start_seconds
        self.stateChanged.emit("playing")

    def toggle_pause_resume(self) -> None:
        return None

    def stop(self) -> None:
        self.state = "paused"
        self.stateChanged.emit("paused")

    def close(self) -> None:
        return None


def _make_wav(path: Path, seconds: float = 0.5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sr = 16000
    frames = int(sr * seconds)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(np.zeros(frames, dtype="<i2").tobytes())


class _FakeCoverage:
    overall_percent = 100.0
    text_files = 2
    audio_files = 2
    total_files = 2


class _FakeVectorIndex:
    def coverage(self, _total: int):
        return _FakeCoverage()

    def close(self) -> None:
        return None


class _FakeAiEngine:
    def __init__(self, long_rec, short_rec) -> None:
        self.long_rec = long_rec
        self.short_rec = short_rec
        self.calls: list[tuple[str, dict]] = []
        self.thread_ids: list[int] = []

    def search_semantic(self, **kwargs) -> AiSearchResponse:
        self.thread_ids.append(threading.get_ident())
        self.calls.append(("semantic", kwargs))
        hits = (
            AiSearchHit(
                file_id=self.long_rec.id,
                rank=1,
                score=0.032,
                relevance_pct=96,
                hit_reason="文本+声波 · 0:15-0:25",
                text_score=0.71,
                audio_score=0.35,
                hit_start_sec=15.0,
                hit_end_sec=25.0,
            ),
            AiSearchHit(
                file_id=self.short_rec.id,
                rank=2,
                score=0.025,
                relevance_pct=78,
                hit_reason="声波高置信(无描述)",
                text_score=None,
                audio_score=0.33,
            ),
        )
        return AiSearchResponse(
            mode="semantic",
            query_text=kwargs["query"],
            hits=hits,
            records=(self.long_rec, self.short_rec),
            total_indexed_text=2,
            total_indexed_audio=2,
            elapsed_ms=12.4,
            audio_prompt="A clear sound effect of monster roar.",
        )

    def find_similar_file(self, **kwargs) -> AiSearchResponse:
        self.thread_ids.append(threading.get_ident())
        self.calls.append(("similar_file", kwargs))
        excl = bool(kwargs.get("exclude_same_folder_or_package"))
        recs = (self.long_rec,) if excl else (self.long_rec, self.short_rec)
        hits = tuple(
            AiSearchHit(
                file_id=r.id,
                rank=i + 1,
                score=0.91 - i * 0.05,
                relevance_pct=91 - i * 5,
                hit_reason=f"音色相似 {91 - i * 5}%",
                audio_score=0.91 - i * 0.05,
                hit_start_sec=15.0 if r.duration_sec > 10.0 else None,
                hit_end_sec=25.0 if r.duration_sec > 10.0 else None,
            )
            for i, r in enumerate(recs)
        )
        return AiSearchResponse(
            mode="similar",
            query_text=self.short_rec.filename,
            hits=hits,
            records=recs,
            total_indexed_text=2,
            total_indexed_audio=2,
            elapsed_ms=5.2,
            seed_file_id=kwargs["file_id"],
            seed_label=self.short_rec.filename,
        )

    def find_similar_range(self, **kwargs) -> AiSearchResponse:
        self.thread_ids.append(threading.get_ident())
        self.calls.append(("similar_range", kwargs))
        hit = AiSearchHit(
            file_id=self.short_rec.id,
            rank=1,
            score=0.89,
            relevance_pct=89,
            hit_reason="音色相似 89%",
            audio_score=0.89,
        )
        return AiSearchResponse(
            mode="similar",
            query_text=f"{self.long_rec.filename} [0:12-0:18]",
            hits=(hit,),
            records=(self.short_rec,),
            total_indexed_text=2,
            total_indexed_audio=2,
            elapsed_ms=8.1,
            seed_file_id=kwargs["file_id"],
            seed_label=f"{self.long_rec.filename} [0:12-0:18]",
            seed_range=(kwargs["start_sec"], kwargs["end_sec"]),
        )


def _wait_until(predicate, timeout_ms: int = 2500) -> None:
    elapsed = 0
    while elapsed < timeout_ms:
        QApplication.processEvents()
        if predicate():
            return
        QTest.qWait(25)
        elapsed += 25
    QApplication.processEvents()
    assert predicate()


def test_browser_ai_mode_long_file_jump_and_find_similar(tmp_path):
    create_app([])
    sfx = tmp_path / "sfx"
    _make_wav(sfx / "PackA" / "long_monster_26s.wav", seconds=26.0)
    _make_wav(sfx / "PackB" / "short_roar.wav", seconds=0.6)

    db_path = tmp_path / "browser.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(sfx)
    LibraryScanner(db_path).scan_root(root.id)
    recs = {r.filename: r for r in next(store.iter_files_for_indexing())}
    store.close()

    fake_audition = _FakeAudition()
    fake_engine = _FakeAiEngine(recs["long_monster_26s.wav"], recs["short_roar.wav"])
    win = BrowserWindow(
        db_path,
        audition=fake_audition,
        ai_engine=fake_engine,
        ai_index=_FakeVectorIndex(),
    )
    win.show()
    _wait_until(lambda: win.table_model.rowCount() == 2)

    col_keys = [c[0] for c in COLUMNS]
    rel_col = col_keys.index("relevance")
    reason_col = col_keys.index("hit_reason")

    try:
        # 1. Starts in Keyword mode with AI columns hidden
        assert win.mode_keyword_btn.isChecked() is True
        assert win.mode_ai_btn.isChecked() is False
        assert win.table_view.isColumnHidden(rel_col) is True
        assert win.table_view.isColumnHidden(reason_col) is True
        assert win.ai_banner.isVisible() is False

        # 2. Switch to AI Semantic mode & press Enter -> shows AI columns & hits
        win.mode_ai_btn.click()
        assert win.mode_ai_btn.isChecked() is True
        assert win.table_view.isColumnHidden(rel_col) is False
        assert win.table_view.isColumnHidden(reason_col) is False

        win.search_input.setText("怪物低吼咆哮")
        QTest.keyClick(win.search_input, Qt.Key.Key_Return)
        _wait_until(lambda: win.table_model.is_ai_mode and win.table_model.rowCount() == 2)
        assert win.table_model.data(win.table_model.index(0, rel_col)) == "96%"
        assert "0:15-0:25" in win.table_model.data(win.table_model.index(0, reason_col))

        # 3. Single-click long AI hit highlights [15, 25) on waveform & auto-seeks to 15.0s!
        rect0 = win.table_view.visualRect(win.table_model.index(0, 0))
        QTest.mouseClick(win.table_view.viewport(), Qt.MouseButton.LeftButton, pos=rect0.center())
        _wait_until(lambda: len(fake_audition.play_calls) == 1)
        assert fake_audition.play_calls[-1][1] == "long_monster_26s.wav"
        assert fake_audition.play_calls[-1][2] == pytest.approx(15.0, abs=1e-3)
        assert win.waveform.highlight_range == pytest.approx((15.0, 25.0))
        assert "AI 匹配" in win.inspector.detail_label.text()

        # 4. Find Similar from table row + Exclude same folder/package toggle re-queries
        win.table_view.findSimilarRowRequested.emit(1)
        _wait_until(
            lambda: any(c[0] == "similar_file" for c in fake_engine.calls)
            and win.exclude_same_pkg_btn.isVisible()
        )
        assert win.table_model.rowCount() == 2
        win.exclude_same_pkg_btn.click()
        _wait_until(lambda: win.table_model.rowCount() == 1)
        assert fake_engine.calls[-1][1]["exclude_same_folder_or_package"] is True

        # 5. Range-based Find Similar via WaveformWidget signal
        win.waveform.findSimilarRangeRequested.emit((12.0, 18.0))
        _wait_until(lambda: any(c[0] == "similar_range" for c in fake_engine.calls))
        assert fake_engine.calls[-1][1]["start_sec"] == pytest.approx(12.0)
        assert fake_engine.calls[-1][1]["end_sec"] == pytest.approx(18.0)

        # 6. Switching back to Keyword mode restores local FTS5 list & hides AI columns
        win.search_input.clear()
        win.mode_keyword_btn.click()
        _wait_until(lambda: not win.table_model.is_ai_mode and win.table_model.rowCount() == 2)
        assert win.table_view.isColumnHidden(rel_col) is True
        assert win.ai_banner.isVisible() is False
    finally:
        win.close()

    # Saved settings must NOT include relevance/hit_reason in hidden_columns
    saved = load_settings(settings_path_for_db(db_path))
    assert "relevance" not in saved.hidden_columns
    assert "hit_reason" not in saved.hidden_columns


def test_ai_mode_idle_sidebar_navigation_updates_table(tmp_path):
    create_app([])
    sfx = tmp_path / "sfx"
    _make_wav(sfx / "PackA" / "long_monster_26s.wav", seconds=26.0)
    _make_wav(sfx / "PackB" / "short_roar.wav", seconds=0.6)

    db_path = tmp_path / "ai_nav.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(sfx)
    LibraryScanner(db_path).scan_root(root.id)
    recs = {r.filename: r for r in next(store.iter_files_for_indexing())}
    store.close()

    fake_audition = _FakeAudition()
    fake_engine = _FakeAiEngine(recs["long_monster_26s.wav"], recs["short_roar.wav"])
    win = BrowserWindow(
        db_path,
        audition=fake_audition,
        ai_engine=fake_engine,
        ai_index=_FakeVectorIndex(),
    )
    win.show()
    _wait_until(lambda: win.table_model.rowCount() == 2)

    try:
        # Switch to ✦ AI 语义 with empty search input
        win.mode_ai_btn.click()
        assert win.mode_ai_btn.isChecked() is True

        # Clicking folders in left tree updates right table immediately
        root_item = win.folder_tree.find_root_item(root.id)
        assert root_item is not None
        win.folder_tree.expandItem(root_item)
        _wait_until(lambda: root_item.childCount() == 2)
        pack_a = root_item.child(0)
        pack_b = root_item.child(1)
        assert pack_a.text(0) == "PackA" and pack_b.text(0) == "PackB"

        win.folder_tree.setCurrentItem(pack_a)
        _wait_until(
            lambda: (
                win.table_model.rowCount() == 1
                and win.table_model.record_at(0) is not None
                and win.table_model.record_at(0).filename == "long_monster_26s.wav"
            )
        )

        win.folder_tree.setCurrentItem(pack_b)
        _wait_until(
            lambda: (
                win.table_model.rowCount() == 1
                and win.table_model.record_at(0) is not None
                and win.table_model.record_at(0).filename == "short_roar.wav"
            )
        )

        # Clicking quick sidebar views (我的收藏 -> 全部素材) also updates right table
        fav_item = win.folder_tree.find_quick_item(KIND_FAVORITES)
        all_item = win.folder_tree.find_quick_item("all")
        assert fav_item is not None and all_item is not None

        win.folder_tree.setCurrentItem(fav_item)
        _wait_until(lambda: win.table_model.rowCount() == 0)

        win.folder_tree.setCurrentItem(all_item)
        _wait_until(lambda: win.table_model.rowCount() == 2)

        # Running AI search then clearing input restores the selected folder/view
        win.folder_tree.setCurrentItem(pack_a)
        _wait_until(lambda: win.table_model.rowCount() == 1)
        win.search_input.setText("怪物低吼")
        QTest.keyClick(win.search_input, Qt.Key.Key_Return)
        _wait_until(lambda: win.table_model.is_ai_mode and win.table_model.rowCount() == 2)

        win.search_input.clear()
        _wait_until(
            lambda: (
                not win.table_model.is_ai_mode
                and win.table_model.rowCount() == 1
                and win.table_model.record_at(0) is not None
                and win.table_model.record_at(0).filename == "long_monster_26s.wav"
            )
        )
    finally:
        win.close()


def test_ai_mode_with_query_text_sidebar_navigation_does_not_auto_search(tmp_path):
    create_app([])
    sfx = tmp_path / "sfx"
    _make_wav(sfx / "PackA" / "long_monster_26s.wav", seconds=26.0)
    _make_wav(sfx / "PackB" / "short_roar.wav", seconds=0.6)

    db_path = tmp_path / "ai_nav_text.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(sfx)
    LibraryScanner(db_path).scan_root(root.id)
    recs = {r.filename: r for r in next(store.iter_files_for_indexing())}
    store.close()

    fake_audition = _FakeAudition()
    fake_engine = _FakeAiEngine(recs["long_monster_26s.wav"], recs["short_roar.wav"])
    win = BrowserWindow(
        db_path,
        audition=fake_audition,
        ai_engine=fake_engine,
        ai_index=_FakeVectorIndex(),
    )
    win.show()
    _wait_until(lambda: win.table_model.rowCount() == 2)

    try:
        # 1. Type text in Keyword mode then switch to AI mode -> must NOT auto-run AI search
        win.search_input.setText("怪物低吼咆哮")
        QTest.qWait(150)
        win.mode_ai_btn.click()
        assert win.mode_ai_btn.isChecked() is True
        QTest.qWait(120)
        assert fake_engine.calls == []
        assert "Enter" in win.ai_banner_label.text()

        # 2. Clicking folders & sidebar tabs in AI mode with text updates right table immediately without running AI
        root_item = win.folder_tree.find_root_item(root.id)
        assert root_item is not None
        win.folder_tree.expandItem(root_item)
        _wait_until(lambda: root_item.childCount() == 2)
        pack_a = root_item.child(0)
        pack_b = root_item.child(1)
        fav_item = win.folder_tree.find_quick_item(KIND_FAVORITES)
        all_item = win.folder_tree.find_quick_item("all")
        assert fav_item is not None and all_item is not None

        win.folder_tree.setCurrentItem(pack_a)
        _wait_until(
            lambda: (
                win.table_model.rowCount() == 1
                and win.table_model.record_at(0) is not None
                and win.table_model.record_at(0).filename == "long_monster_26s.wav"
            )
        )
        assert fake_engine.calls == []
        assert win.search_input.text() == "怪物低吼咆哮"

        win.folder_tree.setCurrentItem(pack_b)
        _wait_until(
            lambda: (
                win.table_model.rowCount() == 1
                and win.table_model.record_at(0) is not None
                and win.table_model.record_at(0).filename == "short_roar.wav"
            )
        )
        win.folder_tree.setCurrentItem(fav_item)
        _wait_until(lambda: win.table_model.rowCount() == 0)
        win.folder_tree.setCurrentItem(all_item)
        _wait_until(lambda: win.table_model.rowCount() == 2)
        assert fake_engine.calls == []

        # 3. Explicitly pressing Enter in PackA runs AI search scoped to PackA
        win.folder_tree.setCurrentItem(pack_a)
        _wait_until(lambda: win.table_model.rowCount() == 1)
        QTest.keyClick(win.search_input, Qt.Key.Key_Return)
        _wait_until(lambda: len(fake_engine.calls) == 1 and win.table_model.is_ai_mode)
        assert fake_engine.calls[0][1]["root_id"] == root.id
        assert fake_engine.calls[0][1]["folder_rel_path"] == "PackA"

        # 4. Toggling scope while AI results are shown refines the active AI search
        win.scope_btn.click()
        _wait_until(lambda: len(fake_engine.calls) == 2)
        assert fake_engine.calls[1][1]["root_id"] is None

        # 5. Clicking another sidebar folder afterwards browses it cleanly without triggering AI search
        win.scope_btn.click()
        _wait_until(lambda: len(fake_engine.calls) == 3)
        win.folder_tree.setCurrentItem(pack_b)
        _wait_until(
            lambda: (
                not win.table_model.is_ai_mode
                and win.table_model.rowCount() == 1
                and win.table_model.record_at(0) is not None
                and win.table_model.record_at(0).filename == "short_roar.wav"
            )
        )
        assert len(fake_engine.calls) == 3
        assert "Enter" in win.ai_banner_label.text()

        # 6. All AI worker calls ran on one persistent worker thread, never on the main thread
        main_tid = threading.main_thread().ident
        assert fake_engine.thread_ids
        assert all(tid != main_tid for tid in fake_engine.thread_ids)
        assert len(set(fake_engine.thread_ids)) == 1
    finally:
        win.close()


def test_remove_root_deactivates_vectors_and_updates_status_tooltip(tmp_path):
    from audiosearch.ai.vector_store import LibraryVectorIndex

    create_app([])
    sfx1 = tmp_path / "sfx1"
    sfx2 = tmp_path / "sfx2"
    _make_wav(sfx1 / "a.wav", seconds=0.4)
    _make_wav(sfx2 / "b.wav", seconds=0.4)

    db_path = tmp_path / "prune_ui.sqlite3"
    store = LibraryStore(db_path)
    r1 = store.add_root(sfx1)
    r2 = store.add_root(sfx2)
    scanner = LibraryScanner(db_path)
    scanner.scan_root(r1.id)
    scanner.scan_root(r2.id)
    all_recs = {r.filename: r for r in next(store.iter_files_for_indexing())}
    store.close()

    vec_idx = LibraryVectorIndex(tmp_path / "vec", profile="small")
    vec_idx.append_text_batch(
        [
            (all_recs["a.wav"].id, all_recs["a.wav"].mtime_ns, "sha256:" + "a" * 64),
            (all_recs["b.wav"].id, all_recs["b.wav"].mtime_ns, "sha256:" + "b" * 64),
        ],
        np.ones((2, 512), dtype=np.float32),
    )
    vec_idx.append_audio_batch(
        [(all_recs["a.wav"].id, all_recs["a.wav"].mtime_ns)],
        [(all_recs["a.wav"].id, all_recs["a.wav"].mtime_ns, 0.0, 0.4)],
        np.ones((1, 512), dtype=np.float32),
    )

    win = BrowserWindow(db_path, audition=_FakeAudition(), ai_index=vec_idx)
    win.show()
    _wait_until(lambda: win.table_model.rowCount() == 2)
    try:
        # 2 files total, text=2, audio=1 -> 75% and tooltip explains diff=1
        assert "75%" in win.status_ai.text()
        assert "文本 2 · 声波 1" in win.status_ai.text()
        tip = win.status_ai.toolTip()
        assert "当前素材库共 2 条" in tip
        assert "差值 1 条" in tip

        # Removing r1 deactivates a.wav vectors immediately and refreshes status_ai
        win._remove_root(r1.id)
        _wait_until(lambda: win.table_model.rowCount() == 1)
        assert "50%" in win.status_ai.text()
        assert "文本 1 · 声波 0" in win.status_ai.text()
        assert vec_idx.get_file_audio_vector(all_recs["a.wav"].id) is None
    finally:
        win.close()
        vec_idx.close()


def test_table_disables_row_tooltips_and_ai_limit_load_more_works(tmp_path):
    create_app([])
    sfx = tmp_path / "sfx_ui"
    _make_wav(sfx / "PackA" / "long_monster_26s.wav", seconds=26.0)
    _make_wav(sfx / "PackB" / "short_roar.wav", seconds=0.6)

    db_path = tmp_path / "ui_006_008.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(sfx)
    LibraryScanner(db_path).scan_root(root.id)
    recs = {r.filename: r for r in next(store.iter_files_for_indexing())}
    store.close()

    class _PagedFakeEngine(_FakeAiEngine):
        def search_semantic(self, **kwargs) -> AiSearchResponse:
            base = super().search_semantic(**kwargs)
            lim = int(kwargs.get("limit", 500))
            return AiSearchResponse(
                mode=base.mode,
                query_text=base.query_text,
                hits=base.hits[:1] if lim <= 200 else base.hits,
                records=base.records[:1] if lim <= 200 else base.records,
                total_indexed_text=base.total_indexed_text,
                total_indexed_audio=base.total_indexed_audio,
                elapsed_ms=base.elapsed_ms,
                has_more=(lim <= 200),
                result_limit=lim,
            )

    fake_engine = _PagedFakeEngine(recs["long_monster_26s.wav"], recs["short_roar.wav"])
    win = BrowserWindow(
        db_path,
        audition=_FakeAudition(),
        ai_engine=fake_engine,
        ai_index=_FakeVectorIndex(),
    )
    win.show()
    _wait_until(lambda: win.table_model.rowCount() == 2)
    try:
        # #006: Row tooltips must be disabled on both file rows and folder rows
        for r in range(win.table_model.rowCount()):
            assert win.table_model.data(win.table_model.index(r, 0), Qt.ItemDataRole.ToolTipRole) is None
        mark_col = [c[0] for c in COLUMNS].index("mark")
        assert win.table_model.headerData(
            mark_col, Qt.Orientation.Horizontal, Qt.ItemDataRole.ToolTipRole
        )

        # #008: Limit combo defaults to 500; choosing 200 triggers truncated banner + load-more button
        assert win.ai_limit_combo.currentData() == 500
        win.mode_ai_btn.click()
        win.ai_limit_combo.setCurrentIndex(win.ai_limit_combo.findData(200))
        win.search_input.setText("怪物")
        QTest.keyClick(win.search_input, Qt.Key.Key_Return)
        _wait_until(lambda: len(fake_engine.calls) == 1 and win.table_model.rowCount() == 1)
        assert fake_engine.calls[-1][1]["limit"] == 200
        assert win.ai_load_more_btn.isVisible() is True

        win.ai_load_more_btn.click()
        _wait_until(lambda: len(fake_engine.calls) == 2 and win.table_model.rowCount() == 2)
        assert fake_engine.calls[-1][1]["limit"] == 500
        assert win.ai_load_more_btn.isVisible() is False
    finally:
        win.close()
