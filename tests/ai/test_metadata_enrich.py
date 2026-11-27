"""Behavioral tests for R5b AI zero-shot UCS classification, prompt caching, and CN<->EN translation."""

from __future__ import annotations

import threading
import wave
from pathlib import Path

import numpy as np

from audiosearch.ai.metadata_enrich import AiMetadataEnricher, translate_sound_text
from audiosearch.ai.vector_store import LibraryVectorIndex
from audiosearch.library.scanner import LibraryScanner
from audiosearch.library.store import LibraryStore
from audiosearch.library.ucs import UCS_CATALOG


def _touch_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(np.zeros(8000, dtype="<i2").tobytes())


class _FakeClap2AudioEncoder:
    def __init__(self, target_cat_id: str = "MAGSpel") -> None:
        self.target_cat_id = target_cat_id
        self.encode_text_calls = 0
        self.encode_audio_calls = 0

    def encode_texts(self, prompts):
        self.encode_text_calls += 1
        vecs = np.zeros((len(prompts), 512), dtype=np.float32)
        for i, entry in enumerate(UCS_CATALOG):
            if entry.cat_id == self.target_cat_id:
                vecs[i, 0] = 1.0
            else:
                vecs[i, 1] = 1.0
        return vecs

    def encode_audio_items(self, items, *, batch_size=None):
        self.encode_audio_calls += 1
        out = np.zeros((len(items), 512), dtype=np.float32)
        out[:, 0] = 1.0
        return out


class _FakeAiSearchEngineStub:
    def __init__(self, encoder: _FakeClap2AudioEncoder) -> None:
        self._lock = threading.RLock()
        self._encoder = encoder

    def _get_audio_encoder(self):
        return self._encoder


def test_bilingual_sound_translation():
    en = translate_sound_text("清脆金币拾取", "en")
    assert "coin" in en.casefold() and "pickup" in en.casefold()

    zh = translate_sound_text("Magic Spell Cast", "zh")
    assert ("魔法" in zh or "法术" in zh) and "施法" in zh


def test_clap2_zero_shot_ucs_caching_and_fallback(tmp_path):
    sfx = tmp_path / "sfx"
    _touch_wav(sfx / "arc_01.wav")
    _touch_wav(sfx / "unindexed_02.wav")

    db_path = tmp_path / "enrich.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(sfx)
    LibraryScanner(db_path).scan_root(root.id)
    recs = {r.filename: r for r in store.query_page(offset=0, limit=10)}

    vec_idx = LibraryVectorIndex(tmp_path / "vectors" / "small", profile="small")
    audio_vec = np.zeros(512, dtype=np.float32)
    audio_vec[0] = 1.0
    vec_idx.append_audio_batch(
        [(recs["arc_01.wav"].id, recs["arc_01.wav"].mtime_ns)],
        [(recs["arc_01.wav"].id, recs["arc_01.wav"].mtime_ns, 0.0, 0.5)],
        audio_vec.reshape(1, 512),
    )

    fake_enc = _FakeClap2AudioEncoder("MAGSpel")
    fake_eng = _FakeAiSearchEngineStub(fake_enc)
    enricher = AiMetadataEnricher(store, vec_idx, fake_eng)

    # First call encodes UCS catalog once & uses indexed vector (0 audio decode calls)
    s1 = enricher.enrich_file(recs["arc_01.wav"].id, translate_mode="zh")
    assert s1.top_ucs[0].entry.cat_id == "MAGSpel"
    assert s1.suggested_fields["cat_id"] == "MAGSpel"
    assert s1.suggested_fields["category"] == "MAGIC"
    assert fake_enc.encode_text_calls == 1
    assert fake_enc.encode_audio_calls == 0

    # Second call on unindexed file decodes & encodes audio on the fly, REUSING cached UCS prompt matrix
    s2 = enricher.enrich_file(recs["unindexed_02.wav"].id)
    assert s2.top_ucs[0].entry.cat_id == "MAGSpel"
    assert fake_enc.encode_text_calls == 1
    assert fake_enc.encode_audio_calls == 1

    # Graceful heuristic fallback when AI engine/vectors unavailable
    offline_enricher = AiMetadataEnricher(store, None, None)
    s3 = offline_enricher.enrich_file(recs["arc_01.wav"].id)
    assert s3.file_id == recs["arc_01.wav"].id

    vec_idx.close()
    store.close()
