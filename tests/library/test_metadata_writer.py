"""Safety & round-trip behavioral tests for R5a metadata writer and batch Undo."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import audiosearch.library.metadata_writer as writer_mod
from audiosearch.catalog.embedded import read_embedded_metadata
from audiosearch.library.metadata_writer import (
    AudioIntegrityError,
    build_write_back_preview,
    compute_audio_hash,
    execute_write_back,
    hash_wav_data_chunk,
    undo_write_batch,
)
from audiosearch.library.peaks import compute_file_peaks
from audiosearch.library.scanner import LibraryScanner
from audiosearch.library.store import LibraryStore


def _chunk(cid: bytes, body: bytes) -> bytes:
    pad = b"\x00" if (len(body) & 1) else b""
    return cid + struct.pack("<I", len(body)) + body + pad


def _write_wav_with_unknown_and_adtl(path: Path) -> tuple[bytes, bytes]:
    path.parent.mkdir(parents=True, exist_ok=True)
    sr = 16000
    n_samples = 1600
    pcm = (np.sin(2 * np.pi * 440.0 * np.arange(n_samples) / sr) * 12000).astype("<i2").tobytes()
    fmt_body = struct.pack("<HHIIHH", 1, 1, sr, sr * 2, 2, 16)
    unknown_body = b"CUSTOM_CHUNK_PAYLOAD_ODD!"  # odd length tests RIFF padding preservation
    adtl_body = b"adtl" + _chunk(b"labl", struct.pack("<I", 1) + b"MarkerA\x00")
    info_body = b"INFO" + _chunk(b"ISFT", b"LegacyTool\x00")

    payload = (
        b"WAVE"
        + _chunk(b"fmt ", fmt_body)
        + _chunk(b"FOOB", unknown_body)
        + _chunk(b"LIST", info_body)
        + _chunk(b"data", pcm)
        + _chunk(b"LIST", adtl_body)
    )
    path.write_bytes(b"RIFF" + struct.pack("<I", len(payload)) + payload)
    return unknown_body, adtl_body


def _walk_top_chunks(path: Path) -> list[tuple[bytes, bytes]]:
    raw = path.read_bytes()
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"
    pos = 12
    out: list[tuple[bytes, bytes]] = []
    while pos + 8 <= len(raw):
        cid = raw[pos : pos + 4]
        sz = struct.unpack("<I", raw[pos + 4 : pos + 8])[0]
        body = raw[pos + 8 : pos + 8 + sz]
        out.append((cid, body))
        pos = pos + 8 + sz + (sz & 1)
    return out


def _make_minimal_mp3(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Valid MPEG-1 Layer III 128kbps 44.1kHz mono frames (no padding, 417B? no: 144*128000/44100 = 417B!)
    frame_hdr = b"\xff\xfb\x90\x00"
    frame = frame_hdr + (b"\x00" * (417 - len(frame_hdr)))
    path.write_bytes(frame * 6)


def test_wav_safe_write_preserves_data_hash_unknown_chunks_and_supports_undo(tmp_path):
    sfx = tmp_path / "sfx"
    wav_path = sfx / "MAGSpel_ArcaneCast.wav"
    unknown_body, adtl_body = _write_wav_with_unknown_and_adtl(wav_path)
    orig_bytes = wav_path.read_bytes()
    pre_data_hash = hash_wav_data_chunk(wav_path)

    db_path = tmp_path / "lib.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(sfx)
    LibraryScanner(db_path).scan_root(root.id)
    rec = store.query_page(offset=0, limit=10)[0]

    # Seed waveform cache to verify mtime_ns stays valid after write-back
    dummy_peak = compute_file_peaks(wav_path, points=16, mtime_ns=rec.mtime_ns)
    store.save_cached_peaks(rec.id, rec.mtime_ns, dummy_peak, requested_points=16)

    # Re-scanning clean or dirty must never wipe dirty user edits
    long_desc = "清脆的魔法火焰发射音效 · " + ("A" * 280)
    store.update_file_metadata(
        rec.id,
        {
            "description": long_desc,
            "keywords": "magic, fire, spell",
            "cat_id": "MAGSpel",
            "category": "MAGIC",
            "subcategory": "SPELL",
            "fx_name": "ArcaneCast",
            "library": "MagicPack",
            "designer": "Echo",
            "notes": "Layer 1",
        },
        source="user",
    )
    assert store.dirty_file_count() == 1
    assert store.get_file(rec.id).is_dirty is True

    # Rescan preserves dirty edits
    LibraryScanner(db_path).scan_root(root.id)
    assert store.dirty_file_count() == 1
    assert store.get_canonical_metadata(rec.id)["cat_id"][0] == "MAGSpel"

    # Dry-run preview shows diffs
    preview = build_write_back_preview(store)
    assert len(preview.items) == 1
    assert len(preview.skipped) == 0
    assert {d.field for d in preview.items[0].diffs} >= {"description", "cat_id", "designer"}

    # Execute safe write-back
    res = execute_write_back(store)
    assert res.status == "committed"
    assert len(res.succeeded) == 1
    assert res.succeeded[0].audio_sha256_before == pre_data_hash
    assert res.succeeded[0].audio_sha256_after == pre_data_hash
    assert hash_wav_data_chunk(wav_path) == pre_data_hash

    # Verify unknown chunk & LIST/adtl preserved byte-for-byte in exact order
    chunks = _walk_top_chunks(wav_path)
    cids = [cid for cid, _ in chunks]
    assert b"FOOB" in cids
    assert cids.index(b"FOOB") < cids.index(b"data") < len(cids) - 1 or cids[-1] == b"LIST"
    assert dict((cid, body) for cid, body in chunks if cid == b"FOOB")[b"FOOB"] == unknown_body
    adtl_matches = [body for cid, body in chunks if cid == b"LIST" and body.startswith(b"adtl")]
    assert adtl_matches == [adtl_body]

    # Verify round-trip read via read_embedded_metadata (both BWF bext, LIST/INFO, iXML USER)
    em = read_embedded_metadata(wav_path)
    by_src = {v.source: v.value for v in em.values}
    assert by_src["embedded:info:ICMT"] == long_desc
    assert by_src["embedded:ixml:USER:CATID"] == "MAGSpel"
    assert by_src["embedded:ixml:USER:DESIGNER"] == "Echo"
    assert by_src["embedded:ixml:USER:FXNAME"] == "ArcaneCast"
    assert "embedded:bext" in by_src

    # Verify DB cleaned dirty state & updated waveform cache mtime_ns
    rec_after = store.get_file(rec.id)
    assert rec_after is not None
    assert rec_after.is_dirty is False
    assert store.dirty_file_count() == 0
    assert store.get_cached_peaks(rec.id, rec_after.mtime_ns, 16) is not None

    # Undo batch restores exact pre-write chunks + keeps data hash identical + re-marks dirty
    undo_res = undo_write_batch(store, res.batch_id)
    assert undo_res.status == "reverted"
    assert len(undo_res.restored) == 1
    assert hash_wav_data_chunk(wav_path) == pre_data_hash
    restored_chunks = _walk_top_chunks(wav_path)
    assert [c for c, _ in restored_chunks] == [c for c, _ in _walk_top_chunks_bytes(orig_bytes)]
    assert read_embedded_metadata(wav_path).description is None
    assert store.get_file(rec.id).is_dirty is True
    assert store.dirty_file_count() == 1
    store.close()


def _walk_top_chunks_bytes(raw: bytes) -> list[tuple[bytes, bytes]]:
    pos = 12
    out: list[tuple[bytes, bytes]] = []
    while pos + 8 <= len(raw):
        cid = raw[pos : pos + 4]
        sz = struct.unpack("<I", raw[pos + 4 : pos + 8])[0]
        out.append((cid, raw[pos + 8 : pos + 8 + sz]))
        pos = pos + 8 + sz + (sz & 1)
    return out


def test_write_back_aborts_and_preserves_original_on_hash_mismatch_and_skips_unwritable(tmp_path, monkeypatch):
    sfx = tmp_path / "sfx"
    ok_wav = sfx / "corrupt_guard.wav"
    missing_wav = sfx / "will_disappear.wav"
    _write_wav_with_unknown_and_adtl(ok_wav)
    _write_wav_with_unknown_and_adtl(missing_wav)
    orig_bytes = ok_wav.read_bytes()

    db_path = tmp_path / "guard.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(sfx)
    LibraryScanner(db_path).scan_root(root.id)
    recs = {r.filename: r for r in store.query_page(offset=0, limit=10)}
    store.update_file_metadata(recs["corrupt_guard.wav"].id, {"description": "Should rollback"})
    store.update_file_metadata(recs["will_disappear.wav"].id, {"description": "Missing"})

    missing_wav.unlink()
    preview = build_write_back_preview(store)
    assert [item.filename for item in preview.items] == ["corrupt_guard.wav"]
    assert [sk.filename for sk in preview.skipped] == ["will_disappear.wav"]

    # Simulate post-write audio hash corruption on tmp file
    real_hash = writer_mod.hash_wav_data_chunk

    def _tampered_hash(p: Path) -> str:
        if ".audiosearch_tmp_" in p.name:
            return "0" * 64
        return real_hash(p)

    monkeypatch.setattr(writer_mod, "hash_wav_data_chunk", _tampered_hash)
    res = execute_write_back(store)
    assert res.status == "failed"
    assert len(res.failed) == 1
    assert "哈希不一致" in res.failed[0].message
    # Original file bytes intact & no leftover temp files
    assert ok_wav.read_bytes() == orig_bytes
    assert not any(sfx.glob(".audiosearch_tmp_*"))
    store.close()


def test_multiformat_mp3_flac_ogg_write_and_undo_preserve_audio_bitstream(tmp_path):
    sfx = tmp_path / "multi"
    sfx.mkdir(parents=True)
    sr = 16000
    t = np.arange(1600) / sr
    samples = (0.35 * np.sin(2 * np.pi * 330.0 * t)).astype(np.float32)

    flac_path = sfx / "magic_chime.flac"
    ogg_path = sfx / "magic_chime.ogg"
    mp3_path = sfx / "magic_chime.mp3"
    sf.write(str(flac_path), samples, sr)
    sf.write(str(ogg_path), samples, sr)
    _make_minimal_mp3(mp3_path)

    hashes_before = {
        "flac": compute_audio_hash(flac_path, "flac"),
        "ogg": compute_audio_hash(ogg_path, "ogg"),
        "mp3": compute_audio_hash(mp3_path, "mp3"),
    }

    db_path = tmp_path / "multi.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(sfx)
    LibraryScanner(db_path).scan_root(root.id)
    recs = {r.ext: r for r in store.query_page(offset=0, limit=10)}
    assert set(recs.keys()) == {"flac", "ogg", "mp3"}

    for ext, rec in recs.items():
        store.update_file_metadata(
            rec.id,
            {
                "description": f"水晶提示 {ext.upper()}",
                "cat_id": "UINotif",
                "category": "USER INTERFACE",
                "subcategory": "NOTIFICATION",
                "designer": "Echo",
                "library": "CasualUI",
            },
        )

    res = execute_write_back(store)
    assert res.status == "committed"
    assert len(res.succeeded) == 3

    for ext, p in (("flac", flac_path), ("ogg", ogg_path), ("mp3", mp3_path)):
        assert compute_audio_hash(p, ext) == hashes_before[ext]

    # Rescan from scratch into a fresh DB to verify tags actually live on disk & round-trip cleanly
    verify_db = tmp_path / "verify_rescan.sqlite3"
    v_store = LibraryStore(verify_db)
    v_root = v_store.add_root(sfx)
    LibraryScanner(verify_db).scan_root(v_root.id)
    for r in v_store.query_page(offset=0, limit=10):
        canon = v_store.get_canonical_metadata(r.id)
        assert canon["cat_id"][0] == "UINotif"
        assert canon["designer"][0] == "Echo"
        assert "水晶提示" in canon["description"][0]
    v_store.close()

    # Undo restores original clean audio hashes too
    undo_res = undo_write_batch(store, res.batch_id)
    assert undo_res.status == "reverted"
    assert len(undo_res.restored) == 3
    for ext, p in (("flac", flac_path), ("ogg", ogg_path), ("mp3", mp3_path)):
        assert compute_audio_hash(p, ext) == hashes_before[ext]
    store.close()
