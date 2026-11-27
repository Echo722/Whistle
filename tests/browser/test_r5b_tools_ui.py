"""Offscreen PySide6 behavioral tests for R5b Batch Rename, CSV Tools, and AI Enrichment Dialogs."""

from __future__ import annotations

import os
import threading
import wave
from pathlib import Path

import numpy as np

from audiosearch.browser.main_window import BrowserWindow, create_app
from audiosearch.browser.metadata_tools_dialogs import (
    AiEnrichDialog,
    BatchRenameDialog,
    CsvMetadataDialog,
)
from audiosearch.library.scanner import LibraryScanner
from audiosearch.library.store import LibraryStore
from tests.ai.test_browser_ai_ui import _FakeAudition, _FakeVectorIndex, _wait_until
from tests.ai.test_metadata_enrich import _FakeAiSearchEngineStub, _FakeClap2AudioEncoder

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _make_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(np.zeros(4000, dtype="<i2").tobytes())


def test_r5b_dialogs_rename_csv_and_ai_enrich_flow(tmp_path):
    create_app([])
    sfx = tmp_path / "sfx"
    _make_wav(sfx / "raw_magic_1.wav")
    _make_wav(sfx / "raw_magic_2.wav")

    db_path = tmp_path / "r5b_ui.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(sfx)
    LibraryScanner(db_path).scan_root(root.id)
    store.close()

    fake_enc = _FakeClap2AudioEncoder("MAGSpel")
    fake_eng = _FakeAiSearchEngineStub(fake_enc)

    win = BrowserWindow(
        db_path,
        audition=_FakeAudition(),
        ai_engine=fake_eng,
        ai_index=_FakeVectorIndex(),
    )
    win.show()
    _wait_until(lambda: win.table_model.rowCount() == 2)

    try:
        ids = [win.table_model.record_at(0).id, win.table_model.record_at(1).id]

        # 1. AiEnrichDialog runs through BrowserWindow._ai_worker (persistent worker thread!)
        worker_threads: list[int] = []
        real_run_custom = win._ai_worker.run_custom

        def _spy_run_custom(fn, on_done, on_error):
            def wrapped():
                worker_threads.append(threading.get_ident())
                return fn()

            real_run_custom(wrapped, on_done, on_error)

        win._ai_worker.run_custom = _spy_run_custom
        ai_dlg = AiEnrichDialog(
            win.store,
            ids,
            run_ai_job=win._ai_worker.run_custom,
            ai_index=win._ensure_ai_index(),
            ai_engine_provider=win._ensure_ai_engine,
            parent=win,
        )
        ai_dlg.start_analysis()
        _wait_until(lambda: ai_dlg.apply_btn.isEnabled() and ai_dlg.results_table.rowCount() == 2, timeout_ms=3000)
        assert worker_threads and worker_threads[0] != threading.main_thread().ident
        ai_dlg.metadataSaved.connect(lambda changed: win._refresh_after_metadata_change(changed))
        ai_dlg._on_apply()
        ai_dlg.close()

        assert win.store.dirty_file_count() == 2
        assert win.write_back_btn.isVisible() is True
        assert win.store.get_canonical_metadata(ids[0])["cat_id"][1] == "ai"

        # 2. BatchRenameDialog: preview -> rename -> undo
        ren_dlg = BatchRenameDialog(
            win.store, ids, before_disk_rename=win._prepare_files_for_disk_write, parent=win
        )
        ren_dlg.template_edit.setText("{CatID}_{Index:02d}")
        assert ren_dlg.execute_btn.isEnabled() is True
        ren_dlg._on_execute()
        assert (sfx / "MAGSpel_01.wav").exists()
        assert (sfx / "MAGSpel_02.wav").exists()
        assert ren_dlg.batches_table.rowCount() == 1

        ren_dlg.batches_table.setCurrentCell(0, 0)
        ren_dlg._on_undo_selected()
        assert (sfx / "raw_magic_1.wav").exists()
        assert (sfx / "raw_magic_2.wav").exists()
        ren_dlg.close()

        # 3. CsvMetadataDialog: export -> modify -> preview -> import
        csv_dlg = CsvMetadataDialog(win.store, ids, parent=win)
        exp_csv = tmp_path / "r5b_test.csv"
        from audiosearch.library.metadata_csv import export_metadata_csv

        export_metadata_csv(win.store, exp_csv, ids)
        txt = exp_csv.read_text(encoding="utf-8-sig").replace(
            win.store.get_canonical_metadata(ids[0])["description"][0],
            "UI导入测试描述",
        )
        exp_csv.write_text(txt, encoding="utf-8-sig")
        csv_dlg.load_import_file(exp_csv)
        assert csv_dlg.apply_import_btn.isEnabled() is True
        csv_dlg._on_apply_import()
        assert win.store.get_file(ids[0]).description == "UI导入测试描述"
        csv_dlg.close()
    finally:
        win.close()
