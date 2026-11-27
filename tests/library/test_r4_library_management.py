"""Behavioral tests for R4 library management: favorites, notes, recents, collections, history, filters, relocate, duplicates."""

from __future__ import annotations

import math
import shutil
import struct
import wave
from pathlib import Path

from audiosearch.library.duplicates import DuplicateScanner
from audiosearch.library.scanner import LibraryScanner
from audiosearch.library.store import LibraryStore, QueryFilter


def _write_pcm_wav(
    path: Path,
    *,
    duration_sec: float = 0.25,
    sample_rate: int = 44100,
    channels: int = 1,
    freq: float = 440.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total_frames = max(1, int(round(duration_sec * sample_rate)))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for i in range(total_frames):
            sample = int(12000 * math.sin(2.0 * math.pi * freq * (i / sample_rate)))
            packed = struct.pack("<h", sample)
            frames.extend(packed * channels)
        wf.writeframes(bytes(frames))


def test_schema_v4_and_spectral_peaks_preserved(tmp_path: Path) -> None:
    db_path = tmp_path / "lib.sqlite3"
    store = LibraryStore(db_path)
    try:
        conn = store._conn()
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        assert version == 4
        cols = {row[1] for row in conn.execute("PRAGMA table_info(waveform_thumbs)").fetchall()}
        assert "spectral_blob" in cols
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert {
            "collections",
            "collection_folders",
            "collection_files",
            "recent_plays",
            "search_history",
            "hidden_files",
            "duplicate_groups",
            "duplicate_members",
        }.issubset(tables)
    finally:
        store.close()


def test_favorites_notes_and_recent_plays(tmp_path: Path) -> None:
    lib_dir = tmp_path / "sfx"
    _write_pcm_wav(lib_dir / "coin_a.wav", duration_sec=0.2)
    _write_pcm_wav(lib_dir / "coin_b.wav", duration_sec=0.3)
    _write_pcm_wav(lib_dir / "boom_c.wav", duration_sec=0.4)
    _write_pcm_wav(lib_dir / "wind_d.wav", duration_sec=0.5)

    db_path = tmp_path / "lib.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(lib_dir)
    store.close()
    LibraryScanner(db_path).scan_root(root.id)

    store = LibraryStore(db_path)
    try:
        rows = store.query_page(offset=0, limit=10, sort_column="filename", sort_asc=True)
        by_name = {r.filename: r for r in rows}

        assert store.toggle_favorite(by_name["coin_a.wav"].id) is True
        assert store.set_favorites([by_name["coin_b.wav"].id, by_name["boom_c.wav"].id], True) == 2
        assert store.favorite_count() == 3

        fav_rows = store.query_page(offset=0, limit=10, filters=QueryFilter(favorites_only=True))
        assert {r.filename for r in fav_rows} == {"coin_a.wav", "coin_b.wav", "boom_c.wav"}

        assert store.toggle_favorite(by_name["boom_c.wav"].id) is False
        assert store.favorite_count() == 2

        store.set_note(by_name["coin_a.wav"].id, "清脆结算金币主版本")
        coin_fresh = store.get_file(by_name["coin_a.wav"].id)
        assert coin_fresh is not None
        assert coin_fresh.note == "清脆结算金币主版本"

        # Record recent plays with cap=3 and dedupe
        store.record_recent_play(by_name["wind_d.wav"].id, max_entries=3)
        store.record_recent_play(by_name["coin_b.wav"].id, max_entries=3)
        store.record_recent_play(by_name["boom_c.wav"].id, max_entries=3)
        store.record_recent_play(by_name["coin_a.wav"].id, max_entries=3)
        store.record_recent_play(by_name["coin_b.wav"].id, max_entries=3)

        assert store.recent_play_count() == 3
        recent = store.query_page(offset=0, 
            limit=10,
            sort_column="played_at",
            sort_asc=False,
            filters=QueryFilter(recent_only=True),
        )
        assert [r.filename for r in recent] == ["coin_b.wav", "coin_a.wav", "boom_c.wav"]

        store.clear_recent_plays()
        assert store.recent_play_count() == 0
    finally:
        store.close()


def test_unified_collections_folders_and_picked_files(tmp_path: Path) -> None:
    dir_a = tmp_path / "pack_a"
    dir_b = tmp_path / "pack_b"
    _write_pcm_wav(dir_a / "ui" / "click_01.wav")
    _write_pcm_wav(dir_a / "ui" / "hover_02.wav")
    _write_pcm_wav(dir_a / "impact" / "punch.wav")
    _write_pcm_wav(dir_b / "foley" / "coin_pick.wav")
    _write_pcm_wav(dir_b / "foley" / "other.wav")

    db_path = tmp_path / "lib.sqlite3"
    store = LibraryStore(db_path)
    r_a = store.add_root(dir_a)
    r_b = store.add_root(dir_b)
    store.close()
    scanner = LibraryScanner(db_path)
    scanner.scan_root(r_a.id)
    scanner.scan_root(r_b.id)

    store = LibraryStore(db_path)
    try:
        coll = store.create_collection("UI项目常用")
        ui_folder = next(f for f in store.list_folders(r_a.id, None) if f.name == "ui")
        store.add_folder_to_collection(coll.id, r_a.id, ui_folder.id, ui_folder.rel_path, label="ui")

        coin_file = next(
            r for r in store.query_page(offset=0, limit=20, query="coin_pick") if r.filename == "coin_pick.wav"
        )
        store.add_files_to_collection(coll.id, [coin_file.id])

        # Collection aggregates both the mounted 'ui' folder and individually picked 'coin_pick.wav'
        coll_filter = QueryFilter(collection_id=coll.id)
        coll_rows = store.query_page(offset=0, limit=20, filters=coll_filter)
        assert {r.filename for r in coll_rows} == {"click_01.wav", "hover_02.wav", "coin_pick.wav"}
        assert store.query_count(filters=coll_filter) == 3
        assert len(store.query_scoped_file_ids(filters=coll_filter)) == 3

        # Keyword search inside unified collection
        assert store.query_count(query="coin", filters=coll_filter) == 1

        # Deleting collection keeps original files untouched
        store.delete_collection(coll.id)
        assert store.total_file_count() == 5
    finally:
        store.close()


def test_search_history_and_quick_filters(tmp_path: Path) -> None:
    lib_dir = tmp_path / "lib"
    _write_pcm_wav(lib_dir / "short_hit.wav", duration_sec=0.3)
    _write_pcm_wav(lib_dir / "mid_hit.wav", duration_sec=1.5)

    off_dir = tmp_path / "ext_drive"
    _write_pcm_wav(off_dir / "short_ext.wav", duration_sec=0.3)

    db_path = tmp_path / "lib.sqlite3"
    store = LibraryStore(db_path)
    r1 = store.add_root(lib_dir)
    r2 = store.add_root(off_dir)
    store.close()
    scanner = LibraryScanner(db_path)
    scanner.scan_root(r1.id)
    scanner.scan_root(r2.id)

    shutil.rmtree(off_dir)
    store = LibraryStore(db_path)
    try:
        store.refresh_roots_online()
        store.record_search_history("impact", mode="keyword", result_count=3)
        store.record_search_history("清脆金属", mode="ai", result_count=5)
        store.record_search_history("impact", mode="keyword", result_count=2)

        hist = store.list_search_history()
        assert len(hist) == 2
        assert hist[0].query == "impact" and hist[0].result_count == 2

        store.set_search_history_pinned(hist[1].id, True)
        store.clear_search_history(keep_pinned=True)
        remaining = store.list_search_history()
        assert len(remaining) == 1 and remaining[0].query == "清脆金属" and remaining[0].pinned

        # Quick filter combinations: WAV + 1-5s + online_only
        f_combo = QueryFilter(ext="wav", min_duration=1.0, max_duration=5.0, online_only=True)
        rows = store.query_page(offset=0, limit=10, filters=f_combo)
        assert [r.filename for r in rows] == ["mid_hit.wav"]

        f_short_online = QueryFilter(ext="wav", max_duration=1.0, online_only=True)
        assert [r.filename for r in store.query_page(offset=0, limit=10, filters=f_short_online)] == ["short_hit.wav"]
    finally:
        store.close()


def test_relocate_root_preserves_user_data_and_peaks(tmp_path: Path) -> None:
    old_dir = tmp_path / "drive_f"
    new_dir = tmp_path / "drive_h"
    _write_pcm_wav(old_dir / "ui" / "confirm.wav", duration_sec=0.4)

    db_path = tmp_path / "lib.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(old_dir)
    store.close()
    LibraryScanner(db_path).scan_root(root.id)

    shutil.move(str(old_dir), str(new_dir))
    store = LibraryStore(db_path)
    try:
        store.refresh_roots_online()
        assert store.get_root(root.id).is_online is False

        rec = store.query_page(offset=0, limit=5)[0]
        store.set_favorites([rec.id], True)
        store.set_note(rec.id, "保持 ID 不丢")
        coll = store.create_collection("定版合集")
        store.add_files_to_collection(coll.id, [rec.id])
        store.save_cached_thumbnail_entries([(rec.id, rec.mtime_ns, b"\x01\x02\x03\x04", b"\x05\x06\x07\x08")])

        relocated = store.relocate_root(root.id, new_dir)
        assert relocated.is_online is True

        fresh = store.get_file(rec.id)
        assert fresh is not None
        assert fresh.id == rec.id
        assert fresh.favorite is True
        assert fresh.note == "保持 ID 不丢"
        assert fresh.is_available is True
        assert fresh.full_path == (new_dir / "ui" / "confirm.wav").resolve()
        assert store.query_count(filters=QueryFilter(collection_id=coll.id)) == 1
        assert store.get_cached_thumbnail_entry(rec.id, rec.mtime_ns) == (
            b"\x01\x02\x03\x04",
            b"\x05\x06\x07\x08",
        )
    finally:
        store.close()


def test_duplicate_scanner_exact_and_hide_non_destructive(tmp_path: Path) -> None:
    lib_dir = tmp_path / "dup_lib"
    src_a = lib_dir / "master" / "impact.wav"
    copy_a = lib_dir / "backup" / "impact_copy.wav"
    unique = lib_dir / "other.wav"
    _write_pcm_wav(src_a, duration_sec=0.3, freq=220.0)
    copy_a.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_a, copy_a)
    _write_pcm_wav(unique, duration_sec=0.3, freq=880.0)

    db_path = tmp_path / "lib.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(lib_dir)
    store.close()
    LibraryScanner(db_path).scan_root(root.id)

    rep_exact = DuplicateScanner(db_path).scan(mode="exact")
    assert rep_exact.groups_found == 1
    assert rep_exact.redundant_files == 1

    store = LibraryStore(db_path)
    try:
        dup_rows = store.list_duplicate_rows(mode="exact", include_hidden=True)
        assert len(dup_rows) == 2
        primary = next(d for d in dup_rows if d.is_primary)
        secondary = next(d for d in dup_rows if not d.is_primary)

        # Switch primary
        store.set_duplicate_primary(primary.group_id, secondary.file.id)
        updated_rows = store.list_duplicate_rows(mode="exact", include_hidden=True)
        assert any(d.file.id == secondary.file.id and d.is_primary for d in updated_rows)

        # Hide non-primary copies in library ONLY
        hidden_n = store.hide_all_non_primary_duplicates("exact")
        assert hidden_n == 1
        assert store.hidden_file_count() == 1
        assert store.query_count() == 2  # other.wav + 1 kept primary

        # Physical files untouched!
        assert src_a.exists() and copy_a.exists() and unique.exists()

        # Unhide restores
        assert store.unhide_files() == 1
        assert store.query_count() == 3
    finally:
        store.close()


def test_duplicate_scanner_name_duration_mode(tmp_path: Path) -> None:
    lib_dir = tmp_path / "multi_fmt"
    _write_pcm_wav(lib_dir / "wav" / "hit_8082.wav", duration_sec=0.5, sample_rate=48000, freq=300.0)
    # Create a lower-rate wav copy in another folder with exact same stem + duration (like OGG/WAV pair)
    _write_pcm_wav(lib_dir / "alt" / "hit_8082.wav", duration_sec=0.5, sample_rate=22050, freq=300.0)
    _write_pcm_wav(lib_dir / "alt" / "distinct_sfx.wav", duration_sec=0.5, sample_rate=44100, freq=500.0)

    db_path = tmp_path / "lib.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(lib_dir)
    store.close()
    LibraryScanner(db_path).scan_root(root.id)

    rep = DuplicateScanner(db_path).scan(mode="name_duration")
    assert rep.groups_found == 1
    assert rep.redundant_files == 1

    store = LibraryStore(db_path)
    try:
        rows = store.list_duplicate_rows(mode="name_duration", include_hidden=True)
        assert len(rows) == 2
        primary = next(d for d in rows if d.is_primary)
        # Higher sample rate / higher quality copy is preferred automatically
        assert primary.file.sample_rate == 48000
    finally:
        store.close()


def test_subfolder_offline_detection_and_relocate_folder(tmp_path: Path) -> None:
    root_dir = tmp_path / "pack"
    _write_pcm_wav(root_dir / "ui" / "sub" / "btn.wav", duration_sec=0.3)
    _write_pcm_wav(root_dir / "sfx" / "hit.wav", duration_sec=0.3)

    db_path = tmp_path / "lib.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(root_dir)
    store.close()
    LibraryScanner(db_path).scan_root(root.id)

    store = LibraryStore(db_path)
    try:
        ui_folder = next(f for f in store.list_folders(root.id, None) if f.name == "ui")
        btn_rec = next(
            r for r in store.query_page(offset=0, limit=10) if r.filename == "btn.wav"
        )
        store.set_favorites([btn_rec.id], True)
        store.set_note(btn_rec.id, "子目录重定位备注")
        coll = store.create_collection("合集A")
        store.add_folder_to_collection(coll.id, root.id, ui_folder.id, ui_folder.rel_path, "ui")
        store.save_cached_thumbnail_entries(
            [(btn_rec.id, btn_rec.mtime_ns, b"\x01\x02", b"\x03\x04")]
        )

        # Rename subfolder within same root -> root stays online, subfolder files marked offline!
        moved_same_root = root_dir / "ui_renamed"
        shutil.move(str(root_dir / "ui"), str(moved_same_root))
        assert store.refresh_root_folders_online(root.id) is True
        btn_off = store.get_file(btn_rec.id)
        assert btn_off is not None and btn_off.is_available is False
        assert store.query_count(filters=QueryFilter(online_only=True)) == 1  # only hit.wav online

        # Scenario A: relocate_folder within same root
        new_rid, new_fid, new_rel, _ = store.relocate_folder(root.id, ui_folder.id, moved_same_root)
        assert new_rid == root.id and new_fid == ui_folder.id and new_rel == "ui_renamed"
        btn_back = store.get_file(btn_rec.id)
        assert btn_back is not None
        assert btn_back.id == btn_rec.id
        assert btn_back.is_available is True
        assert btn_back.favorite is True
        assert btn_back.note == "子目录重定位备注"
        assert btn_back.full_path == (moved_same_root / "sub" / "btn.wav").resolve()
        assert store.query_count(filters=QueryFilter(online_only=True)) == 2
        assert store.query_count(filters=QueryFilter(collection_id=coll.id)) == 1

        # Scenario B: move subfolder completely outside original root -> relocate_folder cross-root
        moved_outside = tmp_path / "external_ui"
        shutil.move(str(moved_same_root), str(moved_outside))
        assert store.refresh_root_folders_online(root.id) is True
        assert store.get_file(btn_rec.id).is_available is False

        ext_rid, ext_fid, ext_rel, _ = store.relocate_folder(root.id, ui_folder.id, moved_outside)
        assert ext_rid != root.id and ext_fid is None and ext_rel == ""
        btn_ext = store.get_file(btn_rec.id)
        assert btn_ext is not None
        assert btn_ext.id == btn_rec.id
        assert btn_ext.is_available is True
        assert btn_ext.favorite is True
        assert btn_ext.note == "子目录重定位备注"
        assert btn_ext.full_path == (moved_outside / "sub" / "btn.wav").resolve()
        assert store.query_count(filters=QueryFilter(collection_id=coll.id)) == 1
        assert store.get_cached_thumbnail_entry(btn_rec.id, btn_rec.mtime_ns) == (
            b"\x01\x02",
            b"\x03\x04",
        )
    finally:
        store.close()


def test_single_file_missing_inside_existing_folder(tmp_path: Path) -> None:
    root_dir = tmp_path / "sfx_root"
    f1 = root_dir / "AccessDenied" / "denied_01.wav"
    f2 = root_dir / "AccessDenied" / "denied_02.wav"
    _write_pcm_wav(f1, duration_sec=0.3)
    _write_pcm_wav(f2, duration_sec=0.3)

    db_path = tmp_path / "lib.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(root_dir)
    store.close()
    LibraryScanner(db_path).scan_root(root.id)

    moved_file = tmp_path / "denied_01.wav"
    shutil.move(str(f1), str(moved_file))

    store = LibraryStore(db_path)
    try:
        rows = {
            r.filename: r for r in store.query_page(offset=0, limit=10)
        }
        r1 = rows["denied_01.wav"]
        r2 = rows["denied_02.wav"]

        # Refreshing the existing subfolder must mark only denied_01 offline (NOT reset to 1!)
        assert store.refresh_folder_online(root.id, "AccessDenied") is True
        assert store.get_file(r1.id).is_available is False
        assert store.get_file(r2.id).is_available is True
        assert store.query_count(filters=QueryFilter(online_only=True)) == 1

        # Full root refresh also preserves per-file offline status inside existing folders
        assert store.refresh_root_folders_online(root.id) is False
        assert store.get_file(r1.id).is_available is False

        # Restore file -> refresh recovers online status automatically
        shutil.move(str(moved_file), str(f1))
        assert store.refresh_folder_online(root.id, "AccessDenied") is True
        assert store.get_file(r1.id).is_available is True
        assert store.query_count(filters=QueryFilter(online_only=True)) == 2

        # Direct-children-only mode (include_subfolders=False) at root returns only root files
        root_direct_only = root_dir / "demo_root.wav"
        _write_pcm_wav(root_direct_only, duration_sec=0.2)
        store.close()
        LibraryScanner(db_path).scan_root(root.id)
        store = LibraryStore(db_path)
        assert store.query_count(root_id=root.id, include_subfolders=False) == 1
        direct_rows = store.query_page(offset=0, limit=10, root_id=root.id, include_subfolders=False)
        assert [r.filename for r in direct_rows] == ["demo_root.wav"]
        assert store.query_count(root_id=root.id, include_subfolders=True) == 3
    finally:
        store.close()


def test_sidebar_custom_sort_order_and_cross_collection_folder_move(tmp_path: Path) -> None:
    dir_a = tmp_path / "Alpha"
    dir_b = tmp_path / "Beta"
    dir_c = tmp_path / "Gamma"
    _write_pcm_wav(dir_a / "Pack1" / "a1.wav", duration_sec=0.2)
    _write_pcm_wav(dir_a / "Pack2" / "a2.wav", duration_sec=0.2)
    _write_pcm_wav(dir_b / "Pack3" / "b1.wav", duration_sec=0.2)
    _write_pcm_wav(dir_c / "c1.wav", duration_sec=0.2)

    db_path = tmp_path / "order.sqlite3"
    store = LibraryStore(db_path)
    r_a = store.add_root(dir_a)
    r_b = store.add_root(dir_b)
    r_c = store.add_root(dir_c)
    store.close()
    for rid in (r_a.id, r_b.id, r_c.id):
        LibraryScanner(db_path).scan_root(rid)

    store = LibraryStore(db_path)
    try:
        # Default roots sorted alphabetically
        assert [r.label for r in store.list_roots()] == ["Alpha", "Beta", "Gamma"]

        # Reorder roots & verify newly added root appends at end
        store.reorder_roots([r_c.id, r_a.id, r_b.id])
        assert [r.id for r in store.list_roots()] == [r_c.id, r_a.id, r_b.id]

        dir_d = tmp_path / "Delta"
        _write_pcm_wav(dir_d / "d1.wav", duration_sec=0.2)
        r_d = store.add_root(dir_d)
        assert [r.id for r in store.list_roots()] == [r_c.id, r_a.id, r_b.id, r_d.id]

        # Reorder collections
        c1 = store.create_collection("合集一")
        c2 = store.create_collection("合集二")
        c3 = store.create_collection("合集三")
        store.reorder_collections([c3.id, c1.id, c2.id])
        assert [c.id for c in store.list_collections()] == [c3.id, c1.id, c2.id]

        # Collection folder same-collection reorder & cross-collection move
        f_p1 = next(f for f in store.list_folders(r_a.id, None) if f.name == "Pack1")
        f_p2 = next(f for f in store.list_folders(r_a.id, None) if f.name == "Pack2")
        f_p3 = next(f for f in store.list_folders(r_b.id, None) if f.name == "Pack3")
        store.add_folder_to_collection(c1.id, r_a.id, f_p1.id, f_p1.rel_path, "Pack1")
        store.add_folder_to_collection(c1.id, r_a.id, f_p2.id, f_p2.rel_path, "Pack2")
        store.add_folder_to_collection(c2.id, r_b.id, f_p3.id, f_p3.rel_path, "Pack3")

        cfs_c1 = store.list_collection_folders(c1.id)
        assert [cf.label for cf in cfs_c1] == ["Pack1", "Pack2"]
        assert store.move_or_reorder_collection_folder(cfs_c1[1].id, c1.id, 0) is True
        assert [cf.label for cf in store.list_collection_folders(c1.id)] == ["Pack2", "Pack1"]

        # Move Pack2 from c1 into c2 before Pack3
        cfs_c1 = store.list_collection_folders(c1.id)
        assert store.move_or_reorder_collection_folder(cfs_c1[0].id, c2.id, 0) is True
        assert [cf.label for cf in store.list_collection_folders(c1.id)] == ["Pack1"]
        assert [cf.label for cf in store.list_collection_folders(c2.id)] == ["Pack2", "Pack3"]

        # Duplicate target merge: re-add Pack1 to c2 then move c1's Pack1 into c2 -> merges cleanly without error
        store.add_folder_to_collection(c2.id, r_a.id, f_p1.id, f_p1.rel_path, "Pack1")
        c1_pack1 = store.list_collection_folders(c1.id)[0]
        assert store.move_or_reorder_collection_folder(c1_pack1.id, c2.id, 0) is True
        assert store.list_collection_folders(c1.id) == []
        assert [cf.label for cf in store.list_collection_folders(c2.id)] == ["Pack1", "Pack2", "Pack3"]
    finally:
        store.close()

    # Reopen store cleanly confirms persistence
    store2 = LibraryStore(db_path)
    try:
        assert [r.id for r in store2.list_roots()] == [r_c.id, r_a.id, r_b.id, r_d.id]
        assert [c.id for c in store2.list_collections()] == [c3.id, c1.id, c2.id]
        assert [cf.label for cf in store2.list_collection_folders(c2.id)] == ["Pack1", "Pack2", "Pack3"]
    finally:
        store2.close()
