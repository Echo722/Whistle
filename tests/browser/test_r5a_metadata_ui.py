"""Offscreen PySide6 behavioral tests for R5a Metadata Editor, Inline Edit, UCS & Write-Back Dialog."""

from __future__ import annotations

import os
import wave
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLineEdit, QPlainTextEdit

from audiosearch.browser.file_table import COLUMNS
from audiosearch.browser.main_window import BrowserWindow, create_app
from audiosearch.browser.metadata_dialogs import MetadataEditorDialog, WriteBackDialog
from audiosearch.library.metadata_writer import hash_wav_data_chunk
from audiosearch.library.scanner import LibraryScanner
from audiosearch.library.store import LibraryStore
from audiosearch.library.ucs import lookup_cat_id
from tests.ai.test_browser_ai_ui import _FakeAudition, _wait_until

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _make_wav(path: Path, seconds: float = 0.25) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sr = 16000
    frames = int(sr * seconds)
    samples = (np.sin(2 * np.pi * 520.0 * np.arange(frames) / sr) * 9000).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


def test_r5a_inline_edit_metadata_dialog_ucs_and_write_back_flow(tmp_path):
    create_app([])
    sfx = tmp_path / "sfx"
    wav1 = sfx / "MAGSpel_FireBolt_Echo_MagicPack.wav"
    wav2 = sfx / "coin_hit_02.wav"
    _make_wav(wav1)
    _make_wav(wav2)
    hash1_orig = hash_wav_data_chunk(wav1)
    hash2_orig = hash_wav_data_chunk(wav2)
    bytes1_orig = wav1.read_bytes()

    db_path = tmp_path / "r5a_ui.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(sfx)
    LibraryScanner(db_path).scan_root(root.id)
    store.close()

    win = BrowserWindow(db_path, audition=_FakeAudition())
    win.show()
    _wait_until(lambda: win.table_model.rowCount() == 2)

    col_keys = [c[0] for c in COLUMNS]
    desc_col = col_keys.index("description")

    try:
        # Initially clean: status-bar dirty button is hidden
        assert win.write_back_btn.isVisible() is False

        # 1. Inline table edit (setData on Description column) saves to DB + marks dirty without touching WAV
        row_mag = next(
            r
            for r in range(win.table_model.rowCount())
            if (rec := win.table_model.record_at(r)) is not None
            and rec.filename == "MAGSpel_FireBolt_Echo_MagicPack.wav"
        )
        rec0 = win.table_model.record_at(row_mag)
        assert rec0 is not None
        idx0 = win.table_model.index(row_mag, desc_col)
        assert bool(win.table_model.flags(idx0) & Qt.ItemFlag.ItemIsEditable)
        assert win.table_model.setData(idx0, "Inline sparkle magic", Qt.ItemDataRole.EditRole)
        _wait_until(lambda: win.write_back_btn.isVisible() and "1" in win.write_back_btn.text())
        assert wav1.read_bytes() == bytes1_orig

        # Select row -> Inspector shows 待写回 badge & enabled 编辑元数据 button
        win._on_audition_row(row_mag)
        assert win.inspector.edit_meta_btn.isEnabled() is True
        assert win.inspector.dirty_pill.isVisible() is True

        # 2. Single-file MetadataEditorDialog + UCS Picker + filename reverse-parse
        dlg = MetadataEditorDialog(win.store, [rec0.id], parent=win)
        dlg._on_parse_single_ucs()
        mag = lookup_cat_id("MAGSpel")
        assert mag is not None
        dlg.ucs_picker.entrySelected.emit(mag)
        cat_w = dlg._single_inputs["cat_id"]
        assert isinstance(cat_w, QLineEdit) and cat_w.text() == "MAGSpel"
        desc_w = dlg._single_inputs["description"]
        if isinstance(desc_w, QPlainTextEdit):
            desc_w.setPlainText("Sparkle Arcane Bolt")
        dlg._on_save()
        assert wav1.read_bytes() == bytes1_orig

        # FTS5 immediately finds the edited description in Keyword mode
        win.search_input.setText("Sparkle")
        _wait_until(lambda: win.table_model.rowCount() == 1)
        win.search_input.clear()
        _wait_until(lambda: win.table_model.rowCount() == 2)

        # 3. Multi-select Batch MetadataEditorDialog (regex replace / set)
        all_ids = [win.table_model.record_at(0).id, win.table_model.record_at(1).id]
        batch_dlg = MetadataEditorDialog(win.store, all_ids, parent=win)
        batch_dlg.field_combo.setCurrentIndex(batch_dlg.field_combo.findData("designer"))
        batch_dlg.op_combo.setCurrentIndex(batch_dlg.op_combo.findData("set"))
        batch_dlg.value_edit.setText("Echo722")
        batch_dlg._on_save()
        assert win.store.dirty_file_count() == 2

        # 4. WriteBackDialog: dry-run preview -> execute -> verify disk hash -> undo batch!
        wb_dlg = WriteBackDialog(win.store, None, before_disk_write=win._prepare_files_for_disk_write, parent=win)
        assert wb_dlg.preview_table.rowCount() == 2
        assert wb_dlg.execute_btn.isEnabled() is True
        wb_dlg._on_execute_write()
        _wait_until(lambda: wb_dlg.batches_table.rowCount() == 1 and win.store.dirty_file_count() == 0, timeout_ms=4000)

        assert hash_wav_data_chunk(wav1) == hash1_orig
        assert hash_wav_data_chunk(wav2) == hash2_orig
        assert wb_dlg.batches_table.item(0, 3).text() == "已写入"

        # Undo batch via WriteBackDialog
        wb_dlg.batches_table.setCurrentCell(0, 0)
        wb_dlg.batches_table.selectRow(0)
        wb_dlg._on_undo_selected()
        _wait_until(lambda: wb_dlg.batches_table.item(0, 3).text() == "已撤销", timeout_ms=4000)
        assert hash_wav_data_chunk(wav1) == hash1_orig
        assert hash_wav_data_chunk(wav2) == hash2_orig
        wb_dlg.close()
    finally:
        win.close()
