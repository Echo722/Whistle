"""Tests for REAPER Media Explorer shortcut parsing, idempotent import, UI dialog, and collection performance."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog

from audiosearch.browser.collections_tree import KIND_COLLECTION
from audiosearch.browser.folder_tree import FolderTree
from audiosearch.browser.main_window import BrowserWindow, create_app
from audiosearch.browser.reaper_dialog import ReaperSetupDialog
from audiosearch.browser.reaper_shortcuts import (
    apply_reaper_import_plan,
    build_reaper_import_plan,
    parse_reaper_ini_shortcuts,
)
from audiosearch.browser.reaper_shortcuts_dialog import ReaperShortcutsImportDialog
from audiosearch.browser.reaper_spot import ReaperSpotClient
from audiosearch.browser.settings import load_settings
from audiosearch.library.scanner import LibraryScanner
from audiosearch.library.store import LibraryStore, QueryFilter
from tests.ai.test_browser_ai_ui import _FakeAudition
from tests.audioio.wav_fixtures import write_wav

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _make_wav(path: Path, *, rate: int = 48000, frames: int = 12000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return write_wav(path, np.zeros((frames, 1), dtype=np.float32), rate=rate)


def _seed_library(tmp_path: Path) -> tuple[LibraryStore, Path, Path]:
    lib = tmp_path / "Sound Effects"
    (lib / "David Dumais Audio").mkdir(parents=True)
    (lib / "Cinetools").mkdir(parents=True)
    (lib / "Stormwave Audio" / "Footsteps Pack").mkdir(parents=True)
    (lib / "Tovusound" / "Biological Steps").mkdir(parents=True)
    outside = tmp_path / "Exported"
    outside.mkdir(parents=True)

    _make_wav(lib / "David Dumais Audio" / "dd_hit_01.wav", frames=24000)
    _make_wav(lib / "Cinetools" / "ct_boom_01.wav", frames=24000)
    _make_wav(lib / "Cinetools" / "ct_boom_02.wav", frames=24000)
    _make_wav(lib / "Stormwave Audio" / "Footsteps Pack" / "fs_snow_01.wav", frames=24000)
    _make_wav(lib / "Tovusound" / "Biological Steps" / "bio_step_01.wav", frames=24000)
    _make_wav(outside / "temp_out.wav", frames=12000)

    store = LibraryStore(tmp_path / "library.sqlite3")
    root = store.add_root(lib)
    LibraryScanner(store.db_path).scan_root(root.id)
    return store, lib, outside


def _write_sample_reaper_ini(ini_path: Path, lib: Path, outside: Path) -> None:
    lines = [
        "[REAPER]",
        "lastprojs=0",
        "[reaper_explorer]",
        "NbShortcuts=15",
        "Shortcut0=<Track Templates>",
        "ShortcutT0=<轨道模板>",
        "Shortcut1=<Project Directory>",
        "Shortcut2=Desktop",
        f"Shortcut3={outside}",
        f"Shortcut4={lib}",
        "Shortcut5=00.ReaperFileList",
        "ShortcutT5=DB: sound effects",
        r"Shortcut6=C:\Users\old\REAPER Media\------------------------名称---------------------------",
        "ShortcutT6=------------------------厂牌分类---------------------------",
        f"Shortcut7={lib / 'David Dumais Audio'}",
        f"Shortcut8={lib / 'Cinetools'}",
        f"Shortcut9={lib / 'Unindexed Brand'}",
        r"Shortcut10=C:\Users\old\Desktop\新建文件夹",
        "ShortcutT10=------------------------Footstep---------------------------",
        f"Shortcut11={str(lib / 'stormwave audio' / 'footsteps pack').upper()}",
        "ShortcutT11=Footsteps Pack【地面材质脚步】",
        f"Shortcut12={lib / 'Tovusound' / 'Biological Steps'}",
        'ShortcutT12=" 终极生物的脚步和动作"',
        f"Shortcut13={lib / 'Missing Pack'}",
        "ShortcutT14=---Footstep---",
        f"Shortcut15={lib / 'David Dumais Audio'}",
    ]
    # Write with UTF-8 BOM to match Windows editors/REAPER exports
    ini_path.write_text("\n".join(lines), encoding="utf-8-sig")


def test_parse_and_build_reaper_shortcuts_plan(tmp_path: Path) -> None:
    store, lib, outside = _seed_library(tmp_path)
    try:
        (lib / "Unindexed Brand").mkdir()
        ini_path = tmp_path / "reaper.ini"
        _write_sample_reaper_ini(ini_path, lib, outside)

        total_raw, parsed_groups, skipped_special, skipped_db = parse_reaper_ini_shortcuts(ini_path)
        assert total_raw == 16
        assert len(skipped_special) == 3
        assert len(skipped_db) == 1
        assert [g[0] for g in parsed_groups] == ["REAPER 快捷方式", "厂牌分类", "Footstep"]

        plan = build_reaper_import_plan(ini_path, store)
        assert plan.has_divider_groups is True
        groups_by_name = {g.name: g for g in plan.groups}

        # Ungrouped contains outside dir + root lib
        ungrouped = {it.raw_path: it for it in groups_by_name["REAPER 快捷方式"].items}
        assert ungrouped[str(outside)].match_status == "outside_root"
        assert ungrouped[str(lib)].match_status == "matched_root"

        # 厂牌分类 preserves non-alphabetical order (David Dumais Audio before Cinetools)
        brands = groups_by_name["厂牌分类"]
        assert [it.label for it in brands.matched_items] == ["David Dumais Audio", "Cinetools"]
        assert [it.match_status for it in brands.unmatched_items] == ["unindexed_under_root"]

        # Footstep handles case-insensitive nested path, custom title, outer quote stripping, and missing dir
        footstep = groups_by_name["Footstep"]
        assert [it.label for it in footstep.matched_items] == [
            "Footsteps Pack【地面材质脚步】",
            "终极生物的脚步和动作",
            "David Dumais Audio",
        ]
        assert [it.match_status for it in footstep.unmatched_items] == ["missing"]

        # Original ini untouched
        assert ini_path.read_text(encoding="utf-8-sig").count("NbShortcuts=15") == 1
    finally:
        store.close()


def test_apply_reaper_import_preserves_order_and_is_idempotent(tmp_path: Path) -> None:
    store, lib, outside = _seed_library(tmp_path)
    try:
        ini_path = tmp_path / "reaper.ini"
        _write_sample_reaper_ini(ini_path, lib, outside)
        store.create_collection("我的手工分组")

        plan = build_reaper_import_plan(ini_path, store)
        report = apply_reaper_import_plan(store, plan, selected_group_names=["厂牌分类", "Footstep"])
        assert report.collections_created == 2
        assert report.collections_reused == 0
        assert report.folders_added == 5
        assert report.folders_updated == 0

        colls = {c.name: c for c in store.list_collections()}
        assert "我的手工分组" in colls
        assert "REAPER 快捷方式" not in colls
        assert colls["厂牌分类"].file_count == 3  # 1 (David Dumais) + 2 (Cinetools)
        assert colls["Footstep"].file_count == 3

        # Verify custom sort_order keeps David Dumais Audio BEFORE Cinetools (not alphabetical!)
        brand_folders = store.list_collection_folders(colls["厂牌分类"].id)
        assert [cf.label for cf in brand_folders] == ["David Dumais Audio", "Cinetools"]
        assert [cf.sort_order for cf in brand_folders] == [1, 2]

        # Re-run import idempotently -> 0 new collections, 0 new folders, existing updated cleanly
        plan2 = build_reaper_import_plan(ini_path, store)
        report2 = apply_reaper_import_plan(store, plan2, selected_group_names=["厂牌分类", "Footstep"])
        assert report2.collections_created == 0
        assert report2.collections_reused == 2
        assert report2.folders_added == 0
        assert report2.folders_updated == 5
        assert len(store.list_collection_folders(colls["厂牌分类"].id)) == 2
    finally:
        store.close()


def test_collection_overlapping_mounts_and_hidden_counts(tmp_path: Path) -> None:
    lib = tmp_path / "sfx"
    (lib / "Stormwave" / "Footsteps").mkdir(parents=True)
    f1 = _make_wav(lib / "Stormwave" / "sw_main.wav")
    f2 = _make_wav(lib / "Stormwave" / "Footsteps" / "sw_step.wav")
    f3 = _make_wav(lib / "extra_pick.wav")

    store = LibraryStore(tmp_path / "library.sqlite3")
    try:
        root = store.add_root(lib)
        LibraryScanner(store.db_path).scan_root(root.id)
        parent = store.list_folders(root.id, parent_id=None)[0]
        child = store.list_folders(root.id, parent_id=parent.id)[0]
        rec_extra = store.query_page(offset=0, limit=1, query="extra_pick")[0]
        rec_step = store.query_page(offset=0, limit=1, query="sw_step")[0]

        coll = store.create_collection("OverlapTest")
        store.add_folder_to_collection(coll.id, root.id, parent.id, parent.rel_path, label="Stormwave", sort_order=1)
        store.add_folder_to_collection(coll.id, root.id, child.id, child.rel_path, label="Footsteps", sort_order=2)

        assert store.collection_item_count(coll.id) == 2
        assert store.query_count(filters=QueryFilter(collection_id=coll.id)) == 2
        rows = store.query_page(offset=0, limit=20, filters=QueryFilter(collection_id=coll.id))
        assert {Path(r.filename).name for r in rows} == {f1.name, f2.name}

        # Mixing direct collection_files + folder mounts
        store.add_files_to_collection(coll.id, [rec_extra.id])
        assert store.collection_item_count(coll.id) == 3
        rows2 = store.query_page(offset=0, limit=20, filters=QueryFilter(collection_id=coll.id))
        assert {Path(r.filename).name for r in rows2} == {f1.name, f2.name, f3.name}

        # Hiding one file disables the fast path and reflects exact visible count
        store.hide_files([rec_step.id])
        assert store.collection_item_count(coll.id) == 2
    finally:
        store.close()


def test_reaper_shortcuts_dialog_and_sidebar_collapse(tmp_path: Path) -> None:
    create_app([])
    store, lib, outside = _seed_library(tmp_path)
    try:
        ini_path = tmp_path / "reaper.ini"
        _write_sample_reaper_ini(ini_path, lib, outside)

        dlg = ReaperShortcutsImportDialog(store, ini_path=ini_path)
        assert dlg.plan is not None
        assert dlg.selected_group_names() == ["厂牌分类", "Footstep"]
        assert dlg.collapse_after_import() is True

        dlg._on_import()
        assert dlg.result() == QDialog.DialogCode.Accepted
        assert dlg.import_report is not None
        assert dlg.import_report.collections_created == 2

        tree = FolderTree(store)
        tree.reload()
        coll_items = [
            tree.topLevelItem(i)
            for i in range(tree.topLevelItemCount())
            if tree.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole) == KIND_COLLECTION
        ]
        assert len(coll_items) == 2
        # Verify FolderTree displays collections in import sort_order rather than alphabetical
        brand_item = next(it for it in coll_items if it.text(0) == "厂牌分类")
        assert [brand_item.child(j).text(0) for j in range(brand_item.childCount())] == [
            "David Dumais Audio",
            "Cinetools",
        ]

        emitted: list[bool] = []
        tree.layoutStateChanged.connect(lambda: emitted.append(True))
        tree.collapse_all_collections()
        assert len(tree.collapsed_collection_ids()) == 2
        assert emitted == [True]
        tree.expand_all_collections()
        assert tree.collapsed_collection_ids() == []
        assert len(emitted) == 2
    finally:
        store.close()


def test_browser_window_import_reaper_shortcuts_persists_collapsed(tmp_path: Path) -> None:
    create_app([])
    lib = tmp_path / "Sound Effects"
    _make_wav(lib / "David Dumais Audio" / "hit.wav")
    _make_wav(lib / "Cinetools" / "boom.wav")

    reaper_dir = tmp_path / "REAPER"
    reaper_dir.mkdir()
    _write_sample_reaper_ini(reaper_dir / "reaper.ini", lib, tmp_path)

    setup_called: list[Path] = []
    spot = ReaperSpotClient(channel_dir=tmp_path / "spot")
    setup_dlg = ReaperSetupDialog(
        spot,
        reaper_dir=reaper_dir,
        on_import_shortcuts=setup_called.append,
    )
    assert setup_dlg.import_shortcuts_btn is not None
    setup_dlg.import_shortcuts_btn.click()
    assert setup_called == [reaper_dir]
    setup_dlg.close()

    db_path = tmp_path / "library.sqlite3"
    win = BrowserWindow(db_path=db_path, audition=_FakeAudition())
    try:
        root = win.store.add_root(lib)
        LibraryScanner(win.store.db_path).scan_root(root.id)
        win.folder_tree.reload()

        def _auto_accept_import(dlg: ReaperShortcutsImportDialog) -> int:
            dlg._on_import()
            return dlg.result()

        with patch.object(ReaperShortcutsImportDialog, "exec", _auto_accept_import):
            win._on_import_reaper_shortcuts(reaper_dir=reaper_dir)

        colls = win.store.list_collections()
        expected_ids = {c.id for c in colls}
        assert {c.name for c in colls} == {"厂牌分类", "Footstep"}
        assert set(win.folder_tree.collapsed_collection_ids()) == expected_ids
        assert set(win._settings.collapsed_collections) == expected_ids
        assert set(load_settings(win._settings_path).collapsed_collections) == expected_ids
    finally:
        win.close()
