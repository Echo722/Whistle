"""Behavioral tests for incremental float16 LibraryVectorIndex & VectorIndexer."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from audiosearch.ai.indexer import (
    VectorIndexer,
    build_file_meta_text,
    plan_audio_windows,
    text_input_hash,
)
from audiosearch.ai.vector_store import IndexCoverage, LibraryVectorIndex
from audiosearch.library.scanner import LibraryScanner
from audiosearch.library.store import LibraryStore


def _unit_vec(dim: int, hot_idx: int, val: float = 1.0) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[hot_idx] = val
    return v


def test_vector_index_append_update_collapse_and_tail_reconcile(tmp_path):
    idx = LibraryVectorIndex(tmp_path / "vec", profile="small")
    try:
        # Small profile: text 512d, audio 512d
        t1 = _unit_vec(512, 0)
        t2 = _unit_vec(512, 1)
        idx.append_text_batch([(1, 100, "sha256:" + "a" * 64), (2, 200, "sha256:" + "b" * 64)], np.stack([t1, t2]))

        # Update file 1 with a new vector: old row must become inactive and not match
        t1_new = _unit_vec(512, 2)
        idx.append_text_batch([(1, 101, "sha256:" + "c" * 64)], t1_new[None, :])

        # Simulate crashed trailing bytes in text.f16 before SQLite commit
        with idx.text_vec_path.open("ab") as fh:
            fh.write(b"\x00" * 512)
        t3 = _unit_vec(512, 3)
        idx.append_text_batch([(3, 300, "sha256:" + "d" * 64)], t3[None, :])

        # Query along axis 0 (inactive file 1 old vec) must return nothing
        assert idx.search_text(_unit_vec(512, 0), min_score=0.2) == []
        # Query along axis 2 hits updated file 1 cleanly
        hits = idx.search_text(_unit_vec(512, 2), min_score=0.5, allowed_file_ids={1, 2})
        assert len(hits) == 1 and hits[0][0] == 1 and hits[0][1] == pytest.approx(1.0, abs=1e-2)

        # Audio windows: file 10 has 2 windows, best-window collapse picks [15, 25)
        a_vecs = np.stack([_unit_vec(512, 4), _unit_vec(512, 5), _unit_vec(512, 5) * 0.5 + _unit_vec(512, 6) * 0.866])
        idx.append_audio_batch(
            [(10, 1000), (11, 2000)],
            [(10, 1000, 0.0, 10.0), (10, 1000, 15.0, 25.0), (11, 2000, 0.0, 5.0)],
            a_vecs,
        )
        q_a = _unit_vec(512, 5)
        audio_hits = idx.search_audio(q_a, min_score=0.3)
        assert [m.file_id for m in audio_hits] == [10, 11]
        assert audio_hits[0].start_sec == pytest.approx(15.0)
        assert audio_hits[0].end_sec == pytest.approx(25.0)

        # Exclusion hides seed
        audio_excl = idx.search_audio(q_a, min_score=0.3, exclude_file_ids={10})
        assert [m.file_id for m in audio_excl] == [11]

        # Seed audio vector helper returns normalized vector
        seed_v = idx.get_file_audio_vector(10)
        assert seed_v is not None and seed_v.shape == (512,)
        assert np.linalg.norm(seed_v) == pytest.approx(1.0, abs=1e-4)

        cov = idx.coverage(total_files=10)
        assert cov.text_files == 3
        assert cov.audio_files == 2
        assert cov.audio_windows == 3
    finally:
        idx.close()


def _write_wav(path: Path, seconds: float = 0.5, freq: float = 440.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sr = 16000
    frames = int(sr * seconds)
    t = np.arange(frames, dtype=np.float32) / sr
    pcm = (np.sin(2.0 * np.pi * freq * t) * 12000.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


class _FakeTextEncoder:
    def load(self) -> None:
        return None

    def encode_documents(self, texts, *, batch_size=None):
        out = np.zeros((len(texts), 512), dtype=np.float32)
        for i, txt in enumerate(texts):
            out[i, hash(txt) % 16] = 1.0
        return out


class _FakeAudioEncoder:
    def load(self) -> None:
        return None

    def encode_audio_items(self, items, *, batch_size=None):
        out = np.zeros((len(items), 512), dtype=np.float32)
        for i, (wav, _sr) in enumerate(items):
            out[i, 0] = float(np.mean(np.abs(wav))) + 0.5
            out[i, 1] = 0.5
        return out


def test_vector_indexer_incremental_reuse_and_long_windows(tmp_path):
    sfx = tmp_path / "sfx"
    _write_wav(sfx / "ui_coin.wav", seconds=0.4, freq=880.0)
    _write_wav(sfx / "amb_forest.wav", seconds=12.0, freq=220.0)

    db_path = tmp_path / "lib.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(sfx)
    LibraryScanner(db_path).scan_root(root.id)
    store.close()

    wins_short = plan_audio_windows(0.4)
    wins_12s = plan_audio_windows(12.0)
    wins_long = plan_audio_windows(300.0)
    assert wins_short == [(0.0, 0.4)]
    assert wins_12s == [(0.0, 10.0), (2.0, 12.0)]
    assert len(wins_long) == 24 and wins_long[-1][1] <= 300.0

    indexer = VectorIndexer(
        db_path,
        profile="small",
        text_encoder=_FakeTextEncoder(),
        audio_encoder=_FakeAudioEncoder(),
    )
    r1 = indexer.run()
    assert r1.status == "done"
    assert r1.total_files == 2
    assert r1.encoded_text == 2
    assert r1.encoded_audio == 2
    assert r1.audio_windows_added == 3  # 1 (0.4s) + 2 (12s)

    # Second run skips everything completely
    r2 = indexer.run()
    assert r2.encoded_text == 0
    assert r2.encoded_audio == 0
    assert r2.reused_text == 2
    assert r2.reused_audio == 2

    # Build meta text & hash helpers stay deterministic
    s = LibraryStore(db_path)
    try:
        recs = list(s.iter_files_for_indexing())[0]
        h = text_input_hash(build_file_meta_text(recs[0]))
        assert h.startswith("sha256:")
    finally:
        s.close()


class _PreparedAudioEncoder:
    """Fake encoder exposing the prefetch-capable surface (prepare + encode_prepared)."""

    def __init__(self) -> None:
        self.prepare_calls = 0
        self.encode_calls = 0

    def load(self) -> None:
        return None

    def prepare_audio_item(self, wav, sr):
        self.prepare_calls += 1
        return {"patches": np.ones((max(1, len(wav) // 160), 4), dtype=np.float32), "sr": sr}

    def encode_prepared(self, prepared, *, batch_size=None):
        self.encode_calls += 1
        out = np.zeros((len(prepared), 512), dtype=np.float32)
        for i, item in enumerate(prepared):
            out[i, 0] = float(item["patches"].shape[0]) + 1.0
            out[i, 1] = 0.5
        return out


def _build_lib(tmp_path):
    sfx = tmp_path / "sfx"
    _write_wav(sfx / "a_coin.wav", seconds=0.4, freq=880.0)
    _write_wav(sfx / "b_amb.wav", seconds=12.0, freq=220.0)
    _write_wav(sfx / "c_long.wav", seconds=30.0, freq=110.0)
    db_path = tmp_path / "lib.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(sfx)
    LibraryScanner(db_path).scan_root(root.id)
    store.close()
    return db_path


def test_vector_indexer_prefetch_pipeline_keeps_order_and_reuses(tmp_path):
    db_path = _build_lib(tmp_path)
    enc = _PreparedAudioEncoder()
    indexer = VectorIndexer(
        db_path, profile="small", text_encoder=_FakeTextEncoder(), audio_encoder=enc
    )
    r1 = indexer.run(prefetch_workers=3, audio_window_batch_size=4)
    assert r1.status == "done"
    assert r1.encoded_audio == 3
    expected = len(plan_audio_windows(0.4)) + len(plan_audio_windows(12.0)) + len(plan_audio_windows(30.0))
    assert r1.audio_windows_added == expected
    assert enc.prepare_calls == expected

    index = LibraryVectorIndex(indexer.vector_dir, profile="small")
    try:
        rows = index._conn().execute(
            "SELECT file_id, start_sec FROM audio_rows WHERE active = 1 ORDER BY row_idx"
        ).fetchall()
    finally:
        index.close()
    file_ids = [r[0] for r in rows]
    assert file_ids == sorted(file_ids)  # files stay in submission order
    assert len(rows) == expected

    r2 = indexer.run(prefetch_workers=3)
    assert r2.encoded_audio == 0 and r2.reused_audio == 3


def test_vector_indexer_prefetch_cancel_returns(tmp_path):
    db_path = _build_lib(tmp_path)
    indexer = VectorIndexer(
        db_path,
        profile="small",
        text_encoder=_FakeTextEncoder(),
        audio_encoder=_PreparedAudioEncoder(),
    )

    def on_progress(p):
        if p.phase == "audio":
            indexer.cancel()

    r = indexer.run(prefetch_workers=2, audio_window_batch_size=1, on_progress=on_progress)
    assert r.status == "cancelled"


def test_encode_in_length_buckets_order_oom_split_and_reraise():
    from audiosearch.inference.clap2_encoder import encode_in_length_buckets

    lengths = [50, 5, 30, 5, 100, 1, 70]
    seen_sizes: list[int] = []

    def run_ok(idxs):
        seen_sizes.append(len(idxs))
        return np.array([[float(i), float(lengths[i])] for i in idxs], dtype=np.float32)

    out = encode_in_length_buckets(lengths, 3, run_ok, lambda e: False)
    assert [int(r[0]) for r in out] == list(range(len(lengths)))
    assert max(seen_sizes) <= 3

    class Oom(RuntimeError):
        pass

    def run_oom_big(idxs):
        if len(idxs) > 2:
            raise Oom("oom")
        return run_ok(idxs)

    out2 = encode_in_length_buckets(lengths, 4, run_oom_big, lambda e: isinstance(e, Oom))
    assert [int(r[0]) for r in out2] == list(range(len(lengths)))

    def always_oom(idxs):
        raise Oom("oom")

    with pytest.raises(Oom):
        encode_in_length_buckets(lengths, 4, always_oom, lambda e: isinstance(e, Oom))


def test_index_coverage_caps_each_tower_independently():
    cov = IndexCoverage(
        profile="large",
        total_files=10,
        text_files=15,
        audio_files=5,
        audio_windows=8,
    )
    # Text overflow (15 > 10) must be capped at 10, so (10 + 5) / 20 = 75%
    assert cov.overall_percent == pytest.approx(75.0)
    assert IndexCoverage("large", 0, 3, 3, 3).overall_percent == 0.0


def test_vector_index_deactivate_and_prune_preserves_binary_append(tmp_path):
    idx = LibraryVectorIndex(tmp_path / "vec", profile="small")
    try:
        idx.append_text_batch(
            [
                (1, 100, "sha256:" + "1" * 64),
                (2, 200, "sha256:" + "2" * 64),
                (3, 300, "sha256:" + "3" * 64),
            ],
            np.stack([_unit_vec(512, 0), _unit_vec(512, 1), _unit_vec(512, 2)]),
        )
        idx.append_audio_batch(
            [(1, 100), (2, 200), (3, 300)],
            [
                (1, 100, 0.0, 1.0),
                (2, 200, 0.0, 1.0),
                (3, 300, 0.0, 1.0),
            ],
            np.stack([_unit_vec(512, 0), _unit_vec(512, 1), _unit_vec(512, 2)]),
        )

        # Deactivate file 1 explicitly
        dt, da = idx.deactivate_file_ids([1, 999])
        assert (dt, da) == (1, 1)
        assert idx.get_file_audio_vector(1) is None
        assert idx.search_text(_unit_vec(512, 0), min_score=0.5) == []
        assert idx.search_audio(_unit_vec(512, 0), min_score=0.5) == []

        # Prune against valid_file_ids={2}: file 3 becomes orphan and deactivated
        pt, pa = idx.prune_missing_file_ids({2})
        assert (pt, pa) == (1, 1)
        cov = idx.coverage(total_files=1)
        assert cov.text_files == 1
        assert cov.audio_files == 1
        assert cov.overall_percent == pytest.approx(100.0)

        # Appending new vectors after pruning keeps binary offsets intact
        idx.append_text_batch(
            [(4, 400, "sha256:" + "4" * 64)],
            _unit_vec(512, 4)[None, :],
        )
        idx.append_audio_batch(
            [(4, 400)],
            [(4, 400, 0.0, 2.0)],
            _unit_vec(512, 4)[None, :],
        )
        assert [fid for fid, _ in idx.search_text(_unit_vec(512, 4), min_score=0.5)] == [4]
        assert [m.file_id for m in idx.search_audio(_unit_vec(512, 4), min_score=0.5)] == [4]
        assert idx.text_vec_path.stat().st_size == 4 * 512 * 2
        assert idx.audio_vec_path.stat().st_size == 4 * 512 * 2
    finally:
        idx.close()


def test_vector_indexer_prunes_deleted_root_but_keeps_offline_root(tmp_path):
    root_online = tmp_path / "online"
    root_offline = tmp_path / "offline"
    root_deleted = tmp_path / "deleted"
    _write_wav(root_online / "keep_online.wav", seconds=0.4)
    _write_wav(root_offline / "keep_offline.wav", seconds=0.4)
    _write_wav(root_deleted / "to_delete.wav", seconds=0.4)

    db_path = tmp_path / "lib.sqlite3"
    store = LibraryStore(db_path)
    r_on = store.add_root(root_online)
    r_off = store.add_root(root_offline)
    r_del = store.add_root(root_deleted)
    scanner = LibraryScanner(db_path)
    scanner.scan_root(r_on.id)
    scanner.scan_root(r_off.id)
    scanner.scan_root(r_del.id)
    store.close()

    indexer = VectorIndexer(
        db_path,
        profile="small",
        text_encoder=_FakeTextEncoder(),
        audio_encoder=_FakeAudioEncoder(),
    )
    first = indexer.run()
    assert first.encoded_text == 3
    assert first.encoded_audio == 3

    # Mark r_off offline in DB and remove r_del completely
    store = LibraryStore(db_path)
    conn = store._conn()
    conn.execute("UPDATE roots SET is_online = 0, status = 'offline' WHERE id = ?", (r_off.id,))
    conn.execute("UPDATE files SET is_available = 0 WHERE root_id = ?", (r_off.id,))
    conn.commit()
    offline_ids = store.file_ids_for_root(r_off.id)
    store.remove_root(r_del.id)
    store.close()

    second = indexer.run()
    assert second.pruned_text == 1
    assert second.pruned_audio == 1
    assert second.reused_text == 1
    assert second.reused_audio == 1

    idx = LibraryVectorIndex(indexer.vector_dir, profile="small")
    try:
        cov = idx.coverage(total_files=2)
        assert cov.text_files == 2
        assert cov.audio_files == 2
        assert idx.get_file_audio_vector(offline_ids[0]) is not None
    finally:
        idx.close()


def test_vector_index_pruned_topk_parity_and_npz_meta_cache(tmp_path):
    vec_dir = tmp_path / "vec_parity"
    idx = LibraryVectorIndex(vec_dir, profile="small")
    try:
        rng = np.random.default_rng(7)
        n = 60
        vecs = rng.standard_normal((n, 512)).astype(np.float32)
        # Force explicit tied vectors at rows 10..13 (fids 11..14) to verify smallest-fid tiebreak
        vecs[10:14] = _unit_vec(512, 0) * 0.7 + _unit_vec(512, 1) * 0.7
        items = [(i + 1, 1000 + i, f"sha256:{i:064x}") for i in range(n)]
        idx.append_text_batch(items, vecs)

        a_spans = [(i + 1, 1000 + i, 0.0, 2.0) for i in range(n)]
        idx.append_audio_batch([(i + 1, 1000 + i) for i in range(n)], a_spans, vecs)

        q = _unit_vec(512, 0) * 0.7 + _unit_vec(512, 1) * 0.7
        q_norm = q / np.linalg.norm(q)
        # Naive exact scores across all rows
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norm_vecs = (vecs / norms).astype(np.float16).astype(np.float32)
        naive_scores = norm_vecs @ (q_norm / np.linalg.norm(q_norm)).astype(np.float32)
        expected = sorted(
            [(i + 1, float(naive_scores[i])) for i in range(n) if naive_scores[i] >= 0.1],
            key=lambda kv: (-kv[1], kv[0]),
        )[:12]

        stats_t: dict[str, int] = {}
        got_t = idx.search_text(q, top_k=12, min_score=0.1, stats_out=stats_t)
        assert [fid for fid, _ in got_t] == [fid for fid, _ in expected]
        assert stats_t["matched_rows"] >= len(expected)

        stats_a: dict[str, int] = {}
        got_a = idx.search_audio(q, top_k=12, min_score=0.1, exclude_file_ids={11}, stats_out=stats_a)
        expected_excl = [fid for fid, _ in expected if fid != 11][:12]
        assert [m.file_id for m in got_a] == expected_excl
        assert idx.text_meta_cache_path.is_file()
        assert idx.audio_meta_cache_path.is_file()
    finally:
        idx.close()

    # Reopen from .npz sidecar and confirm identical results + invalidation after deactivation
    idx2 = LibraryVectorIndex(vec_dir, profile="small")
    try:
        q = _unit_vec(512, 0) * 0.7 + _unit_vec(512, 1) * 0.7
        got2 = idx2.search_text(q, top_k=12, min_score=0.1)
        assert [fid for fid, _ in got2] == [fid for fid, _ in got_t]
        idx2.deactivate_file_ids([11])
        got3 = idx2.search_text(q, top_k=12, min_score=0.1)
        assert 11 not in [fid for fid, _ in got3]
        assert idx2.coverage(total_files=n).text_files == n - 1
    finally:
        idx2.close()
