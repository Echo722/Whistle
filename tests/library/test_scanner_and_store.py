"""Behavioral tests for LibraryScanner, LibraryStore, and PeakData."""

from __future__ import annotations

import struct
import threading
import time
import wave
from pathlib import Path

import numpy as np

from audiosearch.library.peaks import compute_file_peaks
from audiosearch.library.scanner import LibraryScanner
from audiosearch.library.store import LibraryStore


def _write_wav_with_info(path: Path, *, title: str = "", frames: int = 2400, rate: int = 48000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        samples = (np.sin(np.linspace(0, 20, frames)) * 16000).astype("<i2")
        wf.writeframes(samples.tobytes())
    if title:
        raw = path.read_bytes()
        data_bytes = title.encode("utf-8") + b"\x00"
        if len(data_bytes) % 2:
            data_bytes += b"\x00"
        icmt = b"ICMT" + struct.pack("<I", len(data_bytes)) + data_bytes
        list_chunk = b"LIST" + struct.pack("<I", 4 + len(icmt)) + b"INFO" + icmt
        new_size = len(raw) - 8 + len(list_chunk)
        patched = b"RIFF" + struct.pack("<I", new_size) + raw[8:] + list_chunk
        path.write_bytes(patched)


def test_incremental_scan_metadata_and_offline(tmp_path):
    sfx_dir = tmp_path / "sfx"
    _write_wav_with_info(sfx_dir / "Magic" / "cast_01.wav", title="magic spell cast")
    _write_wav_with_info(sfx_dir / "UI" / "btn_click.wav", title="ui button click")

    db_path = tmp_path / "lib.sqlite3"
    store = LibraryStore(db_path)
    scanner = LibraryScanner(db_path)
    try:
        root = store.add_root(sfx_dir, label="TestSFX")
        r1 = scanner.scan_root(root.id)
        assert r1.status == "completed"
        assert r1.added == 2
        assert r1.skipped == 0

        # Second scan skips unchanged files
        r2 = scanner.scan_root(root.id)
        assert r2.added == 0
        assert r2.skipped == 2

        # Folder tree and prefix query
        top_folders = store.list_folders(root.id)
        assert [f.name for f in top_folders] == ["Magic", "UI"]
        assert top_folders[0].file_count == 1

        magic_rows = store.query_page(
            offset=0,
            limit=50,
            root_id=root.id,
            folder_rel_path="Magic",
            sort_column="filename",
            sort_asc=True,
        )
        assert [r.filename for r in magic_rows] == ["cast_01.wav"]
        assert magic_rows[0].description == "magic spell cast"
        assert magic_rows[0].bit_depth == 16

        fields = store.get_metadata_fields(magic_rows[0].id)
        assert any(src == "embedded" and "magic spell cast" in val for _, val, src in fields)

        # Peaks computation + persistent waveform_cache + zoomed range
        peaks = compute_file_peaks(magic_rows[0].full_path, points=2048, mtime_ns=magic_rows[0].mtime_ns)
        assert peaks.channel_count == 1
        assert peaks.points == 2048
        assert peaks.peaks.shape == (1, 2048, 2)
        store.save_cached_peaks(
            magic_rows[0].id, magic_rows[0].mtime_ns, peaks, requested_points=2048
        )
        cached = store.get_cached_peaks(magic_rows[0].id, magic_rows[0].mtime_ns, 2048)
        assert cached is not None
        assert np.allclose(cached.peaks, peaks.peaks)

        zoom_peaks = compute_file_peaks(
            magic_rows[0].full_path,
            points=64,
            mtime_ns=magic_rows[0].mtime_ns,
            start_seconds=0.01,
            end_seconds=0.03,
        )
        assert zoom_peaks.start_seconds > 0.005
        assert zoom_peaks.end_seconds < 0.035

        # Offline detection preserves rows but marks unavailable
        sfx_dir.rename(tmp_path / "sfx_unplugged")
        roots = store.refresh_roots_online()
        assert roots[0].is_online is False
        assert store.total_file_count() == 2
        rows = store.query_page(offset=0, limit=10)
        assert len(rows) == 2
        assert all(not r.is_available for r in rows)
    finally:
        store.close()


def test_scanner_pause_and_resume(tmp_path):
    sfx_dir = tmp_path / "bulk"
    for i in range(6):
        _write_wav_with_info(sfx_dir / f"clip_{i:02d}.wav", frames=480)

    db_path = tmp_path / "pause.sqlite3"
    store = LibraryStore(db_path)
    scanner = LibraryScanner(db_path)
    root = store.add_root(sfx_dir)

    seen_paused = threading.Event()

    def on_progress(p):
        if p.scanned == 1 and p.status == "scanning":
            scanner.pause()
        elif p.status == "paused":
            seen_paused.set()

    t = threading.Thread(target=lambda: scanner.scan_root(root.id, on_progress=on_progress))
    t.start()
    assert seen_paused.wait(timeout=3.0)
    time.sleep(0.05)
    scanner.resume()
    t.join(timeout=5.0)
    assert store.total_file_count() == 6
    store.close()


def test_absolute_path_search_and_query_file_position(tmp_path):
    sfx_root = tmp_path / "Sound Effects"
    p_auto = sfx_root / "2496 Sound Effects" / "Raw - Noise OR Pack" / "Automotive Noise Collection 1.wav"
    p_auto2 = sfx_root / "2496 Sound Effects" / "Raw - Noise OR Pack" / "Automotive Noise Collection 2.wav"
    p_scifi = sfx_root / "科幻" / "激光-脉冲" / "laser_01.wav"
    p_other = sfx_root / "UI" / "click_01.wav"
    for p in (p_auto, p_auto2, p_scifi, p_other):
        _write_wav_with_info(p, frames=480)

    db_path = tmp_path / "abs_path.sqlite3"
    store = LibraryStore(db_path)
    try:
        root = store.add_root(sfx_root)
        LibraryScanner(db_path).scan_root(root.id)
        ui_folder = next(f for f in store.list_folders(root.id) if f.name == "UI")

        # 1. Exact single file (with Windows backslashes, quotes, case difference, hyphens & OR in folder)
        q_exact = f'"{str(p_auto).replace("/", "\\").lower()}"'
        hits = store.query_page(
            offset=0,
            limit=50,
            query=q_exact,
            root_id=root.id,
            folder_id=ui_folder.id,
            folder_rel_path=ui_folder.rel_path,
        )
        assert [r.filename for r in hits] == ["Automotive Noise Collection 1.wav"]
        assert (
            store.query_count(
                query=q_exact,
                root_id=root.id,
                folder_id=ui_folder.id,
                folder_rel_path=ui_folder.rel_path,
            )
            == 1
        )

        # 2. Exact directory (with and without trailing backslash)
        dir_str = str(p_auto.parent).replace("/", "\\")
        for dq in (dir_str, dir_str + "\\", str(p_scifi.parent)):
            res = store.query_page(offset=0, limit=50, query=dq, sort_column="filename", sort_asc=True)
            assert len(res) >= 1
        dir_hits = store.query_page(offset=0, limit=50, query=dir_str + "\\", sort_column="filename", sort_asc=True)
        assert [r.filename for r in dir_hits] == [
            "Automotive Noise Collection 1.wav",
            "Automotive Noise Collection 2.wav",
        ]

        # 3. Partial path prefix (mid-typing)
        partial = str(sfx_root / "2496 Sound Effects" / "Raw - Noise").replace("/", "\\")
        partial_hits = store.query_page(offset=0, limit=50, query=partial)
        assert {r.filename for r in partial_hits} == {
            "Automotive Noise Collection 1.wav",
            "Automotive Noise Collection 2.wav",
        }

        # 4. Parent prefix of root itself
        parent_prefix = str(tmp_path).replace("/", "\\")
        assert store.query_count(query=parent_prefix) == 4

        # 5. Unindexed absolute path returns 0
        unindexed = str(tmp_path / "NotIndexed" / "missing.wav").replace("/", "\\")
        assert store.query_count(query=unindexed) == 0

        # 6. query_file_position matches query_page order for both ASC and DESC
        all_asc = store.query_page(offset=0, limit=50, sort_column="filename", sort_asc=True)
        for expected_idx, rec in enumerate(all_asc):
            assert (
                store.query_file_position(rec.id, sort_column="filename", sort_asc=True)
                == expected_idx
            )
        all_desc = store.query_page(offset=0, limit=50, sort_column="filename", sort_asc=False)
        for expected_idx, rec in enumerate(all_desc):
            assert (
                store.query_file_position(rec.id, sort_column="filename", sort_asc=False)
                == expected_idx
            )
    finally:
        store.close()
