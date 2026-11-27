"""Behavioral tests for R3 UCS query rules, full-sentence prompts, and fusion ranking."""

from __future__ import annotations

import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from audiosearch.ai.query_rules import analyze_ai_query, build_clap2_prompt
from audiosearch.ai.search_engine import AiSearchEngine
from audiosearch.ai.vector_store import LibraryVectorIndex
from audiosearch.library.scanner import LibraryScanner
from audiosearch.library.store import LibraryStore, QueryFilter


@pytest.mark.parametrize(
    ("query", "expected_tokens"),
    [
        ("清脆的金币拾取或星星奖励音效", ["coin", "star", "pickup"]),
        ("火焰呼啸掠过", ["flame", "whoosh", "flyby"]),
        ("火焰持续燃烧噼啪声", ["flame", "burning", "crackling"]),
        ("重金属武器挥砍碰撞", ["heavy metal", "weapon", "sword slash"]),
        ("科幻激光枪射击", ["sci-fi", "laser gun", "shooting"]),
        ("巨大的爆炸冲击波", ["explosion", "shockwave"]),
        ("怪物低吼咆哮", ["monster", "growl", "roar"]),
        ("复古8位像素游戏跳跃升级", ["retro", "eight-bit", "jump"]),
        ("机械机器人运转变形", ["mechanical", "robot", "transform"]),
        ("电流噼啪放电魔法", ["electric", "crackling", "magic"]),
    ],
)
def test_benchmark_chinese_queries_cover_ucs_terms(query, expected_tokens):
    plan = analyze_ai_query(query)
    assert plan.is_chinese is True
    assert plan.translation_coverage == 1.0
    assert plan.text_weight == 1.0
    assert plan.audio_weight == 1.0
    assert plan.audio_prompt.endswith(".")
    lowered = plan.audio_prompt.casefold()
    for tok in expected_tokens:
        assert tok in lowered


def test_dynamic_weighting_and_hard_exclusions():
    untranslated = analyze_ai_query("彻底未收录的拟音词测试")
    assert untranslated.translation_coverage == 0.0
    assert untranslated.text_weight == 1.5
    assert untranslated.audio_weight == 0.25
    assert untranslated.audio_prompt == ""

    partial = analyze_ai_query("火焰呼啸 稀奇古怪")
    assert 0.25 <= partial.translation_coverage < 0.85
    assert partial.text_weight == 1.25
    assert partial.audio_weight == 0.75

    me_excl = analyze_ai_query("火焰呼啸 -爆炸 -dragon")
    assert "火焰呼啸" in me_excl.positive_text_query
    assert "explosion" in me_excl.hard_exclusions
    assert "爆炸" in me_excl.hard_exclusions
    assert "dragon" in me_excl.hard_exclusions
    assert "explosion" not in me_excl.audio_prompt.casefold()

    cn_excl = analyze_ai_query("玻璃碎裂，不要爆炸")
    assert "explosion" in cn_excl.hard_exclusions
    assert "explosion" not in cn_excl.audio_prompt.casefold()

    assert build_clap2_prompt("metal sword hit") == "A clear sound effect of metal sword hit."
    assert (
        build_clap2_prompt("a crisp coin chime sound in a casual game")
        == "A crisp coin chime sound in a casual game."
    )


def _touch_wav(path: Path, seconds: float = 0.5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sr = 16000
    frames = int(sr * seconds)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(np.zeros(frames, dtype="<i2").tobytes())


def _axis(idx: int, weight: float = 1.0) -> np.ndarray:
    v = np.zeros(512, dtype=np.float32)
    v[idx] = weight
    return v


class _MockDualEncoder:
    """Deterministic query encoder driving specific semantic axes."""

    def load(self) -> None:
        return None

    def encode_queries(self, texts):
        out = []
        for t in texts:
            if "怪" in t or "roar" in t:
                out.append(_axis(0))
            else:
                out.append(_axis(1))
        return np.stack(out, axis=0)

    def encode_texts(self, texts):
        out = []
        for t in texts:
            if "monster" in t.casefold() or "roar" in t.casefold():
                out.append(_axis(10))
            else:
                out.append(_axis(11))
        return np.stack(out, axis=0)

    def encode_audio_items(self, items, *, batch_size=None):
        return np.stack([_axis(10) for _ in items], axis=0)


def test_fusion_compensation_exclusion_long_window_and_find_similar(tmp_path):
    sfx = tmp_path / "sfx"
    _touch_wav(sfx / "PackA" / "Bonus_LargeMonster_2.wav", seconds=0.5)  # no desc!
    _touch_wav(sfx / "PackA" / "Monster_Roar_Explosion.wav", seconds=0.5)  # excluded by -爆炸
    _touch_wav(sfx / "PackB" / "Long_Dungeon_Ambience.wav", seconds=25.0)  # long file >10s
    _touch_wav(sfx / "PackB" / "Unrelated_UI_Click.wav", seconds=0.5)

    db_path = tmp_path / "ai.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(sfx)
    LibraryScanner(db_path).scan_root(root.id)
    # Give Monster_Roar_Explosion & Long_Dungeon_Ambience descriptions; Bonus_LargeMonster_2 stays empty
    recs = {r.filename: r for r in next(store.iter_files_for_indexing())}
    conn = store._conn()
    with conn:
        conn.execute("UPDATE files SET description = 'Monster roar explosion' WHERE id = ?", (recs["Monster_Roar_Explosion.wav"].id,))
        conn.execute("UPDATE files SET description = 'Dark dungeon ambience monster' WHERE id = ?", (recs["Long_Dungeon_Ambience.wav"].id,))
        conn.execute("UPDATE files SET package = 'PackA' WHERE id IN (?, ?)", (recs["Bonus_LargeMonster_2.wav"].id, recs["Monster_Roar_Explosion.wav"].id))
        conn.execute("UPDATE files SET package = 'PackB' WHERE id IN (?, ?)", (recs["Long_Dungeon_Ambience.wav"].id, recs["Unrelated_UI_Click.wav"].id))

    vec_idx = LibraryVectorIndex(tmp_path / "vectors" / "small", profile="small")
    # Text vectors: PackB Long_Dungeon_Ambience matches axis 0 strongly; Bonus_LargeMonster_2 barely matches (0.30)
    vec_idx.append_text_batch(
        [
            (recs["Bonus_LargeMonster_2.wav"].id, recs["Bonus_LargeMonster_2.wav"].mtime_ns, "sha256:" + "1" * 64),
            (recs["Monster_Roar_Explosion.wav"].id, recs["Monster_Roar_Explosion.wav"].mtime_ns, "sha256:" + "2" * 64),
            (recs["Long_Dungeon_Ambience.wav"].id, recs["Long_Dungeon_Ambience.wav"].mtime_ns, "sha256:" + "3" * 64),
            (recs["Unrelated_UI_Click.wav"].id, recs["Unrelated_UI_Click.wav"].mtime_ns, "sha256:" + "4" * 64),
        ],
        np.stack([
            _axis(0, 0.30) + _axis(2, 0.95),  # weak text (0.30)
            _axis(0, 0.99) + _axis(2, 0.14),  # excluded by -爆炸
            _axis(0, 0.90) + _axis(2, 0.43),  # strong text (0.90)
            _axis(5, 1.0),                    # orthogonal noise (filtered by min_score)
        ]),
    )
    # Audio vectors: Bonus_LargeMonster_2 is top-1 audio (0.95) -> triggers no-desc high-conf compensation!
    # Long_Dungeon_Ambience has window [15, 25) matching 0.85 > [0, 10) matching 0.20
    vec_idx.append_audio_batch(
        [
            (recs["Bonus_LargeMonster_2.wav"].id, recs["Bonus_LargeMonster_2.wav"].mtime_ns),
            (recs["Monster_Roar_Explosion.wav"].id, recs["Monster_Roar_Explosion.wav"].mtime_ns),
            (recs["Long_Dungeon_Ambience.wav"].id, recs["Long_Dungeon_Ambience.wav"].mtime_ns),
            (recs["Unrelated_UI_Click.wav"].id, recs["Unrelated_UI_Click.wav"].mtime_ns),
        ],
        [
            (recs["Bonus_LargeMonster_2.wav"].id, recs["Bonus_LargeMonster_2.wav"].mtime_ns, 0.0, 0.5),
            (recs["Monster_Roar_Explosion.wav"].id, recs["Monster_Roar_Explosion.wav"].mtime_ns, 0.0, 0.5),
            (recs["Long_Dungeon_Ambience.wav"].id, recs["Long_Dungeon_Ambience.wav"].mtime_ns, 0.0, 10.0),
            (recs["Long_Dungeon_Ambience.wav"].id, recs["Long_Dungeon_Ambience.wav"].mtime_ns, 15.0, 25.0),
            (recs["Unrelated_UI_Click.wav"].id, recs["Unrelated_UI_Click.wav"].mtime_ns, 0.0, 0.5),
        ],
        np.stack([
            _axis(10, 0.95) + _axis(12, 0.31),
            _axis(10, 0.99) + _axis(12, 0.10),
            _axis(10, 0.20) + _axis(12, 0.98),
            _axis(10, 0.85) + _axis(12, 0.52),
            _axis(15, 1.0),
        ]),
    )

    mock = _MockDualEncoder()
    engine = AiSearchEngine(store, vec_idx, text_encoder=mock, audio_encoder=mock)
    try:
        resp = engine.search_semantic("怪物低吼咆哮 -爆炸")
        names = [r.filename for r in resp.records]
        assert "Monster_Roar_Explosion.wav" not in names  # hard excluded!
        assert "Unrelated_UI_Click.wav" not in names      # denoised!
        assert names == ["Bonus_LargeMonster_2.wav", "Long_Dungeon_Ambience.wav"]
        assert "声波高置信(无描述)" in resp.hits[0].hit_reason
        assert resp.hits[1].hit_start_sec == pytest.approx(15.0)
        assert resp.hits[1].hit_end_sec == pytest.approx(25.0)
        assert "0:15-0:25" in resp.hits[1].hit_reason

        # Find Similar from Bonus_LargeMonster_2 (in PackA): excluding same pkg hides PackA, keeps PackB
        sim_all = engine.find_similar_file(
            recs["Bonus_LargeMonster_2.wav"].id,
            exclude_same_folder_or_package=False,
            min_score=0.5,
        )
        assert [r.filename for r in sim_all.records] == [
            "Monster_Roar_Explosion.wav",
            "Long_Dungeon_Ambience.wav",
        ]
        sim_cross = engine.find_similar_file(
            recs["Bonus_LargeMonster_2.wav"].id,
            exclude_same_folder_or_package=True,
            min_score=0.5,
        )
        assert [r.filename for r in sim_cross.records] == ["Long_Dungeon_Ambience.wav"]

        # Range-based Find Similar works directly via audio encoder
        sim_range = engine.find_similar_range(
            recs["Long_Dungeon_Ambience.wav"].id,
            15.0,
            25.0,
            exclude_same_folder_or_package=True,
            min_score=0.5,
        )
        assert sim_range.seed_range == (15.0, 25.0)
        assert "Monster_Roar_Explosion.wav" in [r.filename for r in sim_range.records]
    finally:
        vec_idx.close()
        store.close()


class _CountingConcurrentEncoder:
    def __init__(self) -> None:
        self.text_calls = 0
        self.audio_calls = 0
        self.max_concurrent = 0
        self._active = 0
        self._guard = threading.Lock()

    def load(self) -> None:
        return None

    def _enter(self) -> None:
        with self._guard:
            self._active += 1
            if self._active > self.max_concurrent:
                self.max_concurrent = self._active

    def _leave(self) -> None:
        with self._guard:
            self._active -= 1

    def encode_queries(self, texts):
        self._enter()
        try:
            self.text_calls += 1
            time.sleep(0.02)
            return np.stack([_axis(0) for _ in texts], axis=0)
        finally:
            self._leave()

    def encode_texts(self, texts):
        self._enter()
        try:
            self.audio_calls += 1
            time.sleep(0.02)
            return np.stack([_axis(10) for _ in texts], axis=0)
        finally:
            self._leave()


def test_ai_search_engine_caches_embeddings_short_circuits_empty_scope_and_serializes(tmp_path):
    sfx = tmp_path / "sfx"
    _touch_wav(sfx / "PackA" / "Monster_A.wav", seconds=0.5)
    _touch_wav(sfx / "PackB" / "Monster_B.wav", seconds=0.5)

    db_path = tmp_path / "cache.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(sfx)
    LibraryScanner(db_path).scan_root(root.id)
    recs = {r.filename: r for r in next(store.iter_files_for_indexing())}

    vec_idx = LibraryVectorIndex(tmp_path / "vectors" / "small", profile="small")
    vec_idx.append_text_batch(
        [
            (recs["Monster_A.wav"].id, recs["Monster_A.wav"].mtime_ns, "sha256:" + "a" * 64),
            (recs["Monster_B.wav"].id, recs["Monster_B.wav"].mtime_ns, "sha256:" + "b" * 64),
        ],
        np.stack([_axis(0), _axis(0)]),
    )
    vec_idx.append_audio_batch(
        [
            (recs["Monster_A.wav"].id, recs["Monster_A.wav"].mtime_ns),
            (recs["Monster_B.wav"].id, recs["Monster_B.wav"].mtime_ns),
        ],
        [
            (recs["Monster_A.wav"].id, recs["Monster_A.wav"].mtime_ns, 0.0, 0.5),
            (recs["Monster_B.wav"].id, recs["Monster_B.wav"].mtime_ns, 0.0, 0.5),
        ],
        np.stack([_axis(10), _axis(10)]),
    )

    enc = _CountingConcurrentEncoder()
    engine = AiSearchEngine(store, vec_idx, text_encoder=enc, audio_encoder=enc)
    try:
        # 1. Empty scope (favorites_only with 0 favorites) short-circuits with 0 encoder calls
        empty_resp = engine.search_semantic("怪物低吼咆哮", filters=QueryFilter(favorites_only=True))
        assert empty_resp.hits == ()
        assert enc.text_calls == 0
        assert enc.audio_calls == 0

        # 2. First real search encodes once; second search across another folder reuses cache
        r1 = engine.search_semantic("怪物低吼咆哮", root_id=root.id, folder_rel_path="PackA")
        assert [r.filename for r in r1.records] == ["Monster_A.wav"]
        assert enc.text_calls == 1
        assert enc.audio_calls == 1

        r2 = engine.search_semantic("怪物低吼咆哮", root_id=root.id, folder_rel_path="PackB")
        assert [r.filename for r in r2.records] == ["Monster_B.wav"]
        assert enc.text_calls == 1
        assert enc.audio_calls == 1

        # 3. Concurrent searches serialize encoder access (max_concurrent == 1)
        errors: list[Exception] = []

        def run_q(q_text: str) -> None:
            try:
                engine.search_semantic(q_text)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        t1 = threading.Thread(target=run_q, args=("火焰呼啸",))
        t2 = threading.Thread(target=run_q, args=("电流放电",))
        t1.start()
        t2.start()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)
        assert not errors
        assert enc.max_concurrent == 1
    finally:
        engine.close()
        vec_idx.close()
        store.close()


def test_ai_search_limit_has_more_and_sole_root_scope(tmp_path):
    sfx = tmp_path / "sfx_limits"
    for i in range(6):
        _touch_wav(sfx / f"Monster_Hit_{i:02d}.wav", seconds=0.4)

    db_path = tmp_path / "limits.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(sfx)
    LibraryScanner(db_path).scan_root(root.id)
    recs = sorted(next(store.iter_files_for_indexing()), key=lambda r: r.id)

    vec_idx = LibraryVectorIndex(tmp_path / "vectors" / "small", profile="small")
    vec_idx.append_text_batch(
        [(r.id, r.mtime_ns, f"sha256:{i:064x}") for i, r in enumerate(recs)],
        np.stack([_axis(0) for _ in recs]),
    )
    vec_idx.append_audio_batch(
        [(r.id, r.mtime_ns) for r in recs],
        [(r.id, r.mtime_ns, 0.0, 0.4) for r in recs],
        np.stack([_axis(10) for _ in recs]),
    )
    mock = _MockDualEncoder()
    engine = AiSearchEngine(store, vec_idx, text_encoder=mock, audio_encoder=mock)
    try:
        # Limited to 3 of 6 -> has_more=True
        resp3 = engine.search_semantic("怪物低吼咆哮", root_id=root.id, limit=3)
        assert len(resp3.records) == 3
        assert resp3.has_more is True
        assert resp3.result_limit == 3

        # Limit 10 >= 6 -> has_more=False, matches global scope
        resp_all = engine.search_semantic("怪物低吼咆哮", root_id=root.id, limit=10)
        resp_global = engine.search_semantic("怪物低吼咆哮", root_id=None, limit=10)
        assert len(resp_all.records) == 6
        assert resp_all.has_more is False
        assert [r.id for r in resp_all.records] == [r.id for r in resp_global.records]
    finally:
        engine.close()
        vec_idx.close()
        store.close()
