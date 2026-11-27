"""Behavioral tests for R5b CSV metadata export, column detection, preview, and DB import."""

from __future__ import annotations

import csv
import wave
from pathlib import Path

import numpy as np

from audiosearch.library.metadata_csv import (
    apply_metadata_csv_import,
    export_metadata_csv,
    preview_metadata_csv_import,
)
from audiosearch.library.scanner import LibraryScanner
from audiosearch.library.store import LibraryStore


def _make_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(np.zeros(1600, dtype="<i2").tobytes())


def test_csv_export_preview_import_and_matching_rules(tmp_path):
    sfx = tmp_path / "sfx"
    _make_wav(sfx / "PackA" / "coin_01.wav")
    _make_wav(sfx / "PackB" / "sword_01.wav")
    # Duplicate filename in different folders to test ambiguous filename guard
    _make_wav(sfx / "PackA" / "hit.wav")
    _make_wav(sfx / "PackB" / "hit.wav")

    db_path = tmp_path / "csv.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(sfx)
    LibraryScanner(db_path).scan_root(root.id)
    disk_bytes = (sfx / "PackA" / "coin_01.wav").read_bytes()

    csv_path = tmp_path / "export.csv"
    exp = export_metadata_csv(store, csv_path)
    assert exp.row_count == 4
    assert csv_path.read_bytes().startswith(b"\xef\xbb\xbf")  # UTF-8 BOM

    # Modify exported CSV rows: row 0 by file_id, remove file_id on row 1 to test filename match
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    for row in rows:
        if row["filename"] == "coin_01.wav":
            row["description"] = "清脆金币拾取音效"
            row["cat_id"] = "GAMEGold"
            row["favorite"] = "1"
        elif row["filename"] == "sword_01.wav":
            row["file_id"] = ""
            row["rel_path"] = ""
            row["designer"] = "Echo722"
            row["user_note"] = "需剪掉尾部"

    # Append a bad ambiguous row (hit.wav with no file_id/rel_path) & an unknown file row
    rows.append({"file_id": "", "rel_path": "", "filename": "hit.wav", "description": "ambiguous"})
    rows.append({"file_id": "", "rel_path": "", "filename": "nonexistent.wav", "description": "ghost"})

    edited_csv = tmp_path / "edited.csv"
    with edited_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    prev = preview_metadata_csv_import(store, edited_csv)
    assert len(prev.matched) == 2
    assert len(prev.skipped) == 2
    assert any("重名" in sk.reason for sk in prev.skipped)

    res = apply_metadata_csv_import(store, edited_csv)
    assert res.updated_files == 2
    assert res.updated_fields >= 4
    assert store.dirty_file_count() == 2
    # Disk WAV files must remain completely untouched
    assert (sfx / "PackA" / "coin_01.wav").read_bytes() == disk_bytes

    recs = {r.filename: r for r in store.query_page(offset=0, limit=10)}
    assert recs["coin_01.wav"].favorite is True
    assert recs["coin_01.wav"].description == "清脆金币拾取音效"
    assert recs["sword_01.wav"].note == "需剪掉尾部"
    assert store.get_canonical_metadata(recs["sword_01.wav"].id)["designer"][0] == "Echo722"

    # Verify GBK encoded Chinese CSV works as well
    gbk_csv = tmp_path / "gbk.csv"
    gbk_csv.write_text("文件名,描述\ncoin_01.wav,GBK中文描述测试\n", encoding="gb18030")
    res_gbk = apply_metadata_csv_import(store, gbk_csv)
    assert res_gbk.updated_files == 1
    assert store.get_file(recs["coin_01.wav"].id).description == "GBK中文描述测试"
    store.close()
