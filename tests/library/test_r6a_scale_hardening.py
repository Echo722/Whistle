"""Behavioral tests for R6a million-file library hardening."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from audiosearch.library.scanner import LibraryScanner
from audiosearch.library.schema import open_connection
from audiosearch.library.store import LibraryStore, QueryFilter


def _make_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(np.zeros(1600, dtype="<i2").tobytes())


def test_dirty_metadata_partial_index_used(tmp_path: Path):
    db_path = tmp_path / "idx_test.sqlite3"
    store = LibraryStore(db_path)
    try:
        conn = open_connection(db_path)
        try:
            idx_row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index' AND name = 'idx_metadata_fields_dirty'"
            ).fetchone()
            assert idx_row is not None

            plan = conn.execute(
                "EXPLAIN QUERY PLAN SELECT COUNT(DISTINCT file_id) FROM metadata_fields WHERE dirty = 1"
            ).fetchall()
            plan_text = " ".join(str(r[3]) for r in plan)
            assert "idx_metadata_fields_dirty" in plan_text
        finally:
            conn.close()
    finally:
        store.close()


def test_folder_query_count_fast_path_and_safety(tmp_path: Path):
    sfx = tmp_path / "sfx"
    _make_wav(sfx / "Magic" / "Sub" / "spell_01.wav")
    _make_wav(sfx / "Magic" / "Sub" / "spell_02.wav")
    _make_wav(sfx / "Magic" / "cast_01.wav")
    _make_wav(sfx / "UI" / "click_01.wav")

    db_path = tmp_path / "count_test.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(sfx)
    LibraryScanner(db_path).scan_root(root.id)

    try:
        top_folders = store.list_folders(root.id, None)
        magic = next(f for f in top_folders if f.name == "Magic")
        assert magic.file_count == 3

        assert store.query_count(root_id=root.id, folder_id=magic.id, folder_rel_path="Magic") == 3
        assert (
            store.query_count(
                root_id=root.id,
                folder_id=magic.id,
                folder_rel_path="Magic",
                filters=QueryFilter(ext="ogg"),
            )
            == 0
        )

        # While scanning, fast path yields to live COUNT(*)
        conn = open_connection(db_path)
        try:
            conn.execute("UPDATE roots SET status = 'scanning' WHERE id = ?", (root.id,))
            conn.commit()
        finally:
            conn.close()
        assert store.query_count(root_id=root.id, folder_id=magic.id, folder_rel_path="Magic") == 3

        # Reset to idle and hide one file -> hidden_file_count() > 0 bypasses fast path
        conn = open_connection(db_path)
        try:
            conn.execute("UPDATE roots SET status = 'idle' WHERE id = ?", (root.id,))
            fid = int(
                conn.execute("SELECT id FROM files WHERE filename = 'spell_01.wav'").fetchone()[0]
            )
            conn.commit()
        finally:
            conn.close()
        store.hide_files([fid], reason="duplicate")
        assert store.query_count(root_id=root.id, folder_id=magic.id, folder_rel_path="Magic") == 2
    finally:
        store.close()


def test_scanner_handles_corrupt_files_and_does_not_lock_db(tmp_path: Path):
    sfx = tmp_path / "sfx_corrupt"
    _make_wav(sfx / "SubA" / "valid_01.wav")
    _make_wav(sfx / "SubB" / "valid_02.wav")
    corrupt = sfx / "SubA" / "broken.wav"
    corrupt.write_bytes(b"NOT_A_RIFF_HEADER_1234")

    db_path = tmp_path / "corrupt_scan.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(sfx)
    events = []
    report = LibraryScanner(db_path).scan_root(root.id, on_progress=events.append)

    assert report.status == "completed"
    assert report.added == 2
    assert report.errors >= 1
    assert any(e.status == "completed" for e in events)

    # Concurrent read/write succeeds cleanly on same WAL DB after folder creation & scan
    assert store.total_file_count() == 2
    files = store.query_page(offset=0, limit=10)
    assert len(files) == 2
    store.set_favorites([files[0].id], True)
    assert store.get_file(files[0].id).favorite is True
    store.close()


def test_delete_folder_uses_fk_indexes_no_full_scan(tmp_path: Path):
    db_path = tmp_path / "fk_plan.sqlite3"
    store = LibraryStore(db_path)
    try:
        conn = open_connection(db_path)
        try:
            for idx_name in (
                "idx_files_folder",
                "idx_folders_parent",
                "idx_coll_folders_root",
                "idx_coll_folders_folder",
            ):
                row = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'index' AND name = ?",
                    (idx_name,),
                ).fetchone()
                assert row is not None, f"missing index {idx_name}"

            del_plan = conn.execute("EXPLAIN QUERY PLAN DELETE FROM folders WHERE id = 1").fetchall()
            details = [str(r[3]) for r in del_plan]
            for detail in details:
                assert not detail.startswith("SCAN "), f"unexpected full table scan: {details}"

            child_plan = conn.execute(
                "EXPLAIN QUERY PLAN SELECT 1 FROM folders WHERE parent_id = 1 LIMIT 1"
            ).fetchall()
            assert any("idx_folders_parent" in str(r[3]) for r in child_plan)
        finally:
            conn.close()
    finally:
        store.close()


def test_extract_inspector_summary_promotes_brand_package_ucs_and_folder_path():
    from types import SimpleNamespace

    from audiosearch.library.metadata_fields import extract_inspector_summary

    rec = SimpleNamespace(
        full_path=Path(r"F:\SFX\Rock The Speakerbox\Physics Body Parts\arm_impact.wav"),
        root_path=r"F:\SFX",
        rel_path="Rock The Speakerbox/Physics Body Parts/arm_impact.wav",
        package="SC Physics: Body Parts",
    )
    fields = [
        ("embedded:bext", "8d7b0d7ef5634e2cb3668b363ba1d8a400000000", "embedded"),
        ("embedded:info:IART", "Rock The Speakerbox - Professional SFX", "embedded"),
        ("embedded:ixml:USER:CATID", "GOREMisc", "embedded"),
        ("embedded:ixml:USER:CATEGORY", "GORE", "embedded"),
        ("embedded:ixml:USER:SUBCATEGORY", "MISC", "embedded"),
        ("embedded:ixml:USER:LIBRARY", "SC Physics: Body Parts", "embedded"),
        ("embedded:ixml:IXML_VERSION", "1.61", "embedded"),
    ]
    summary = extract_inspector_summary(rec, fields)
    assert summary.brand == "Rock The Speakerbox - Professional SFX"
    assert summary.package == "SC Physics: Body Parts"
    assert summary.ucs_label == "GOREMisc · GORE / MISC"
    assert summary.path_segments == ("Rock The Speakerbox", "Physics Body Parts")
    assert summary.root_label == "SFX"
    assert summary.folder_path == str(rec.full_path.parent)
    assert summary.formatted_fields[-1][0] == "BWF (bext)"
    assert summary.formatted_fields[0][0] == "厂牌 / 作者 (IART)"

    # Deduplicates package when identical to brand (case-insensitive)
    rec_same = SimpleNamespace(
        full_path=Path(r"F:\SFX\344 Audio\duck.wav"),
        root_path=r"F:\SFX",
        rel_path="344 Audio/duck.wav",
        package="344 Audio",
    )
    summary_same = extract_inspector_summary(
        rec_same, [("embedded:info:IART", "344 audio", "embedded")]
    )
    assert summary_same.brand == "344 audio"
    assert summary_same.package == ""

    # Never guesses brand/package from folder path or DAW/copyright noise
    rec_noisy = SimpleNamespace(
        full_path=Path(r"F:\Sound Effects\Boom Library\Anime Essentials\impact.wav"),
        root_path=r"F:\Sound Effects",
        rel_path="Boom Library/Anime Essentials/impact.wav",
        package="Pro Tools",
    )
    noisy_fields = [
        ("embedded:info:IART", "Pro Tools", "embedded"),
        ("embedded:tag:ALBUM", "©Sound IdeasAll Rights Reserved", "embedded"),
    ]
    summary_noisy = extract_inspector_summary(rec_noisy, noisy_fields)
    assert summary_noisy.brand == ""
    assert summary_noisy.package == ""
    assert summary_noisy.path_segments == ("Boom Library", "Anime Essentials")
    assert summary_noisy.root_label == "Sound Effects"
    assert summary_noisy.folder_path == str(rec_noisy.full_path.parent)


def test_query_page_deferred_join_and_reverse_offset_parity(tmp_path: Path):
    sfx = tmp_path / "sfx_paging"
    for i in range(25):
        sub = "Alpha" if i % 2 == 0 else "Beta"
        # Reuse a filename across folders (i=0 in Alpha, i=1 in Beta) to test tie-breaker parity.
        name = "clip_shared.wav" if i in (0, 1) else f"clip_{i % 7:02d}_{i:02d}.wav"
        p = sfx / sub / name
        p.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(p), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(np.zeros(800 + i * 160, dtype="<i2").tobytes())

    db_path = tmp_path / "paging_parity.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(sfx)
    LibraryScanner(db_path).scan_root(root.id)

    try:
        all_recs = store.query_page(offset=0, limit=100)
        assert len(all_recs) == 25
        store.set_favorites([all_recs[1].id, all_recs[5].id, all_recs[19].id], True)
        store.record_recent_play(all_recs[2].id)
        store.record_recent_play(all_recs[14].id)

        # Verify Phase 1 ID query uses covering index when sole root & no hidden files
        id_body, id_params = store._build_id_query_parts(
            query="",
            root_id=root.id,
            folder_id=None,
            folder_rel_path="",
            include_subfolders=True,
            sort_column="filename",
            filters=None,
        )
        plan_rows = store._conn().execute(
            f"EXPLAIN QUERY PLAN SELECT f.id {id_body} "
            "ORDER BY f.filename COLLATE NOCASE ASC, f.id ASC LIMIT 5 OFFSET 10",
            id_params,
        ).fetchall()
        assert any("COVERING INDEX idx_files_filename" in str(r[3]) for r in plan_rows)

        # Parity across sort columns, ASC/DESC, and offsets (first half, midpoint, second half, end)
        for sort_col in ("filename", "duration_sec", "favorite", "played_at", "size_bytes"):
            for sort_asc in (True, False):
                full_ids = [
                    r.id
                    for r in store.query_page(
                        offset=0, limit=100, sort_column=sort_col, sort_asc=sort_asc
                    )
                ]
                assert len(full_ids) == 25
                for off in (0, 5, 12, 13, 18, 22, 24, 25):
                    fwd = [
                        r.id
                        for r in store.query_page(
                            offset=off,
                            limit=4,
                            sort_column=sort_col,
                            sort_asc=sort_asc,
                            total_count=None,
                        )
                    ]
                    rev = [
                        r.id
                        for r in store.query_page(
                            offset=off,
                            limit=4,
                            sort_column=sort_col,
                            sort_asc=sort_asc,
                            total_count=25,
                        )
                    ]
                    assert fwd == full_ids[off : off + 4]
                    assert rev == full_ids[off : off + 4]

        # Parity with folder scope, FTS query, QueryFilter, and hidden_files active
        store.hide_files([all_recs[3].id, all_recs[17].id], reason="duplicate")
        alpha_total = store.query_count(
            root_id=root.id, folder_rel_path="Alpha", include_subfolders=True
        )
        alpha_all = [
            r.id
            for r in store.query_page(
                offset=0,
                limit=100,
                root_id=root.id,
                folder_rel_path="Alpha",
                include_subfolders=True,
                sort_column="duration_sec",
                sort_asc=False,
            )
        ]
        assert len(alpha_all) == alpha_total
        for off in range(0, alpha_total + 2, 3):
            paged = [
                r.id
                for r in store.query_page(
                    offset=off,
                    limit=3,
                    root_id=root.id,
                    folder_rel_path="Alpha",
                    include_subfolders=True,
                    sort_column="duration_sec",
                    sort_asc=False,
                    total_count=alpha_total,
                )
            ]
            assert paged == alpha_all[off : off + 3]

        fts_total = store.query_count(query="clip")
        fts_all = [r.id for r in store.query_page(offset=0, limit=100, query="clip")]
        assert len(fts_all) == fts_total
        assert [
            r.id
            for r in store.query_page(
                offset=fts_total - 4, limit=6, query="clip", total_count=fts_total
            )
        ] == fts_all[fts_total - 4 :]

        # RAM sorted-ID array + query_page_by_ids parity & cancellation & index_version behavior
        import threading

        for sort_col in ("filename", "duration_sec", "favorite", "size_bytes"):
            for sort_asc in (True, False):
                expected = [
                    r.id
                    for r in store.query_page(
                        offset=0, limit=100, sort_column=sort_col, sort_asc=sort_asc
                    )
                ]
                arr = store.query_sorted_id_array(sort_column=sort_col, sort_asc=sort_asc)
                assert arr is not None
                assert arr.tolist() == expected
                sliced = store.query_page_by_ids(arr[5:11].tolist())
                assert [r.id for r in sliced] == expected[5:11]

        fts_arr = store.query_sorted_id_array(query="clip")
        assert fts_arr is not None and fts_arr.tolist() == fts_all

        cancel_ev = threading.Event()
        cancel_ev.set()
        assert store.query_sorted_id_array(cancel_event=cancel_ev) is None

        v0 = store.index_version
        store.record_recent_play(all_recs[0].id)
        assert store.index_version == v0
        store.set_favorites([all_recs[0].id], True)
        assert store.index_version == v0 + 1
    finally:
        store.close()
