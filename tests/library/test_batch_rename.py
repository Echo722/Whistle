"""Behavioral tests for R5b template batch rename, collision handling, swaps, and Undo."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from audiosearch.library.batch_rename import (
    build_rename_preview,
    execute_batch_rename,
    render_rename_template,
    sanitize_filename_stem,
    undo_rename_batch,
)
from audiosearch.library.peaks import compute_file_peaks
from audiosearch.library.scanner import LibraryScanner
from audiosearch.library.store import LibraryStore


def _make_wav(path: Path, freq: float = 440.0) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    sr = 16000
    frames = int(sr * 0.15)
    pcm = (np.sin(2 * np.pi * freq * np.arange(frames) / sr) * 9000).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return path.read_bytes()


def test_sanitize_and_template_rendering(tmp_path):
    assert sanitize_filename_stem('CON<>:"/\\|?*  test..') == "CON_test"
    assert sanitize_filename_stem("CON") == "_CON"
    assert sanitize_filename_stem("Arcane Bolt Spell", space_mode="underscore") == "Arcane_Bolt_Spell"
    assert sanitize_filename_stem("arcane bolt spell", space_mode="camel") == "ArcaneBoltSpell"

    sfx = tmp_path / "sfx"
    wav = sfx / "Magic" / "raw_spell.wav"
    _make_wav(wav)
    db_path = tmp_path / "tpl.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(sfx)
    LibraryScanner(db_path).scan_root(root.id)
    rec = store.query_page(offset=0, limit=5)[0]
    rendered = render_rename_template(
        "{CatID}_{FXName}_{Designer}_{Index:02d}",
        canonical={"cat_id": "MAGSpel", "fx_name": "Fire Bolt", "designer": "Echo"},
        rec=rec,
        index=3,
        space_mode="underscore",
    )
    assert rendered == "MAGSpel_Fire_Bolt_Echo_03.wav"
    store.close()


def test_batch_rename_collisions_preserves_ids_and_supports_undo(tmp_path):
    sfx = tmp_path / "sfx"
    wav_a = sfx / "raw_1.wav"
    wav_b = sfx / "raw_2.wav"
    bytes_a = _make_wav(wav_a, 330.0)
    bytes_b = _make_wav(wav_b, 660.0)

    db_path = tmp_path / "ren.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(sfx)
    LibraryScanner(db_path).scan_root(root.id)
    recs = {r.filename: r for r in store.query_page(offset=0, limit=10)}
    id_a = recs["raw_1.wav"].id
    id_b = recs["raw_2.wav"].id

    # Seed cached peaks & metadata
    peak_a = compute_file_peaks(wav_a, points=16, mtime_ns=recs["raw_1.wav"].mtime_ns)
    store.save_cached_peaks(id_a, recs["raw_1.wav"].mtime_ns, peak_a, requested_points=16)
    store.update_file_metadata(id_a, {"cat_id": "UIClick", "fx_name": "Tap"})
    store.update_file_metadata(id_b, {"cat_id": "UIClick", "fx_name": "Tap"})

    # Strict error mode detects collision when template omits {Index}
    conflict_prev = build_rename_preview(
        store, [id_a, id_b], "{CatID}_{FXName}", collision_policy="error"
    )
    assert conflict_prev.conflict_count == 1
    assert execute_batch_rename(store, [id_a, id_b], "{CatID}_{FXName}", collision_policy="error").status == "conflict"
    assert wav_a.exists() and wav_b.exists()

    # auto_suffix resolves collision cleanly
    res = execute_batch_rename(
        store,
        [id_a, id_b],
        "{CatID}_{FXName}",
        collision_policy="auto_suffix",
    )
    assert res.status == "committed"
    assert len(res.renamed) == 2
    assert (sfx / "UIClick_Tap.wav").exists()
    assert (sfx / "UIClick_Tap_01.wav").exists()
    assert not wav_a.exists() and not wav_b.exists()

    # file_id, cached peaks, and dirty metadata stay intact + FTS5 finds new filename!
    rec_a_after = store.get_file(id_a)
    assert rec_a_after is not None
    assert rec_a_after.filename == "UIClick_Tap.wav"
    assert store.get_cached_peaks(id_a, rec_a_after.mtime_ns, 16) is not None
    assert store.query_count(query="UIClick_Tap") == 2

    # Undo batch restores exact filenames and disk bytes
    undo_res = undo_rename_batch(store, res.batch_id)
    assert undo_res.status == "reverted"
    assert wav_a.read_bytes() == bytes_a
    assert wav_b.read_bytes() == bytes_b
    assert store.get_file(id_a).filename == "raw_1.wav"
    assert store.get_file(id_b).filename == "raw_2.wav"
    store.close()


def test_batch_rename_cyclic_swap_and_undo(tmp_path):
    sfx = tmp_path / "swap"
    p1 = sfx / "alpha.wav"
    p2 = sfx / "beta.wav"
    bytes_1 = _make_wav(p1, 220.0)
    bytes_2 = _make_wav(p2, 880.0)

    db_path = tmp_path / "swap.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(sfx)
    LibraryScanner(db_path).scan_root(root.id)
    recs = {r.filename: r for r in store.query_page(offset=0, limit=10)}
    id1 = recs["alpha.wav"].id
    id2 = recs["beta.wav"].id

    # Force swap: id1 -> beta.wav, id2 -> alpha.wav via FXName
    store.update_file_metadata(id1, {"fx_name": "beta"})
    store.update_file_metadata(id2, {"fx_name": "alpha"})

    res = execute_batch_rename(store, [id1, id2], "{FXName}", collision_policy="error")
    assert res.status == "committed"
    assert p2.read_bytes() == bytes_1
    assert p1.read_bytes() == bytes_2
    assert store.get_file(id1).filename == "beta.wav"
    assert store.get_file(id2).filename == "alpha.wav"

    undo = undo_rename_batch(store, res.batch_id)
    assert undo.status == "reverted"
    assert p1.read_bytes() == bytes_1
    assert p2.read_bytes() == bytes_2
    store.close()
