# ruff: noqa: E402  (the numpy import guard must run before project imports)
"""Audio results: best window per asset first, then take N assets.

The earlier implementation asked for ``limit * 2`` *windows* and de-duplicated
afterwards, so one long file with many good windows could fill the whole page and
hide other qualifying assets. These tests pin the corrected semantics: collapse
to the best window per asset inside the scan, merge across descriptions, then
truncate by asset count.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip(
    "numpy",
    reason="Agent B numeric tests need numpy; it is an optional extra, not a shared dev dependency",
)
from helpers import make_asset, make_catalog, make_space

from audiosearch.contracts.models import QueryPlan
from audiosearch.indexing.encode import audio_input_fingerprint, plan_asset_windows
from audiosearch.indexing.store import EmbeddingBatch, EmbeddingRow, EmbeddingStore
from audiosearch.retrieval.vector.retrievers import AudioVectorRetriever


class ListEncoder:
    """Returns one preset query vector per description, in order."""

    def __init__(self, space, vectors):
        self.space = space
        self.config = type("C", (), {"batch_size": 4})()
        self._vectors = [np.asarray(v, dtype=np.float32) for v in vectors]

    def load(self):
        return self.space

    def encode_texts(self, texts, batch_size=None):
        return np.stack([self._vectors[0]] * len(texts))

    def encode_audio_mono(self, windows, sample_rate_hz, *, batch_size=None):  # pragma: no cover
        raise AssertionError("audio encoding is not used in these tests")

    def close(self):
        pass


class PerTextEncoder(ListEncoder):
    """Returns a different vector for each description text."""

    def __init__(self, space, mapping):
        super().__init__(space, list(mapping.values()))
        self._mapping = {key: np.asarray(value, np.float32) for key, value in mapping.items()}

    def encode_texts(self, texts, batch_size=None):
        return np.stack([self._mapping[text] for text in texts])


def _space(dimension: int = 4):
    return make_space(tower="audio", dimension=dimension)


def _store_with_windows(directory, space, catalog, spec):
    """``spec`` maps asset index -> list of (start, end, unit_vector)."""
    rows, vectors = [], []
    for index, windows in spec.items():
        asset = catalog.assets[index]
        for start, end, vector in windows:
            rows.append(EmbeddingRow(str(asset.asset_id), "segment", start, end, coverage=1.0))
            raw = np.asarray(vector, dtype=np.float32)
            vectors.append(raw / np.linalg.norm(raw))
    revisions = {str(asset.asset_id) for asset in catalog.assets}
    row_assets = {row.asset_id for row in rows}
    fingerprints = {
        str(asset.asset_id): audio_input_fingerprint(
            plan_asset_windows(asset, window_seconds=10.0, hop_seconds=5.0)
        )
        for asset in catalog.assets
        if str(asset.asset_id) in row_assets
    }
    store = EmbeddingStore(directory)
    store.write(
        EmbeddingBatch(
            space=space,
            vectors=np.stack(vectors),
            rows=rows,
            revisions={
                str(asset.asset_id): asset.content_revision
                for asset in catalog.assets
                if str(asset.asset_id) in row_assets
            },
            input_fingerprints=fingerprints,
        ),
        manifest_id="sha256:" + "2" * 64,
        library_id="test-library",
        fingerprint={},
        params={"window_seconds": 10.0, "hop_seconds": 5.0},
    )
    assert revisions  # keep the intent of the helper obvious
    return store


def test_one_long_asset_cannot_starve_the_page(tmp_path):
    space = _space()
    assets = [
        make_asset(0, duration=300.0),
        make_asset(1, duration=12.0),
        make_asset(2, duration=12.0),
    ]
    catalog = make_catalog(assets)

    # Asset 0 has 30 windows, all slightly better than the others.
    long_windows = [(float(i * 5), float(i * 5 + 10), [0.99, 0.01, 0.0, 0.0]) for i in range(30)]
    spec = {
        0: long_windows,
        1: [(0.0, 10.0, [0.5, 0.5, 0.0, 0.0])],
        2: [(0.0, 10.0, [0.4, 0.6, 0.0, 0.0])],
    }
    store = _store_with_windows(tmp_path / "audio", space, catalog, spec)
    encoder = ListEncoder(space, [[1.0, 0.0, 0.0, 0.0]])
    retriever = AudioVectorRetriever(catalog, store, encoder)

    plan = QueryPlan(original_text="whoosh", audio_descriptions=["whoosh"])
    outcome = retriever.search_text(plan, limit=3)
    names = [item.asset.name for item in outcome.results]
    assert len(names) == 3, f"expected three assets, got {names}"
    assert names[0] == "asset-0.wav"
    assert set(names) == {"asset-0.wav", "asset-1.wav", "asset-2.wav"}


def test_limit_counts_assets_not_windows(tmp_path):
    space = _space()
    assets = [make_asset(0, duration=300.0), make_asset(1, duration=12.0)]
    catalog = make_catalog(assets)
    spec = {
        0: [(float(i * 5), float(i * 5 + 10), [1.0, 0.0, 0.0, 0.0]) for i in range(20)],
        1: [(0.0, 10.0, [0.2, 0.98, 0.0, 0.0])],
    }
    store = _store_with_windows(tmp_path / "audio", space, catalog, spec)
    retriever = AudioVectorRetriever(catalog, store, ListEncoder(space, [[1.0, 0.0, 0.0, 0.0]]))
    plan = QueryPlan(original_text="whoosh", audio_descriptions=["whoosh"])
    for limit in (1, 2):
        outcome = retriever.search_text(plan, limit=limit)
        assert len(outcome.results) == limit
        assert len({item.asset.name for item in outcome.results}) == limit


def test_best_window_per_asset_is_kept(tmp_path):
    space = _space()
    assets = [make_asset(0, duration=100.0)]
    catalog = make_catalog(assets)
    spec = {
        0: [
            (0.0, 10.0, [0.3, 0.95, 0.0, 0.0]),
            (40.0, 50.0, [1.0, 0.0, 0.0, 0.0]),
            (70.0, 80.0, [0.5, 0.5, 0.0, 0.0]),
        ]
    }
    store = _store_with_windows(tmp_path / "audio", space, catalog, spec)
    retriever = AudioVectorRetriever(catalog, store, ListEncoder(space, [[1.0, 0.0, 0.0, 0.0]]))
    outcome = retriever.search_text(QueryPlan(original_text="x", audio_descriptions=["x"]), limit=5)
    assert len(outcome.results) == 1
    matched = outcome.results[0].matched_range
    assert (matched.start_seconds, matched.end_seconds) == (40.0, 50.0)


def test_multiple_descriptions_merge_per_asset_then_truncate(tmp_path):
    space = _space()
    assets = [make_asset(0, duration=60.0), make_asset(1, duration=60.0)]
    catalog = make_catalog(assets)
    spec = {
        0: [
            (0.0, 10.0, [0.8, 0.6, 0.0, 0.0]),
            (50.0, 60.0, [0.0, 0.0, 1.0, 0.0]),
        ],
        1: [(0.0, 10.0, [0.0, 1.0, 0.0, 0.0])],
    }
    store = _store_with_windows(tmp_path / "audio", space, catalog, spec)
    encoder = PerTextEncoder(
        space,
        {
            "first": [1.0, 0.0, 0.0, 0.0],
            "second": [0.0, 0.0, 1.0, 0.0],
        },
    )
    retriever = AudioVectorRetriever(catalog, store, encoder)
    outcome = retriever.search_text(
        QueryPlan(original_text="x", audio_descriptions=["first", "second"]), limit=2
    )
    # Asset 0 wins from the second description with its 50-60 s window.
    assert [item.asset.name for item in outcome.results][0] == "asset-0.wav"
    matched = outcome.results[0].matched_range
    assert (matched.start_seconds, matched.end_seconds) == (50.0, 60.0)
    assert len(outcome.results) == 2


def test_hard_filter_still_applies_before_the_page_is_filled(tmp_path):
    space = _space()
    assets = [make_asset(0, duration=6.0), make_asset(1, duration=60.0)]
    catalog = make_catalog(assets)
    spec = {
        0: [(0.0, 6.0, [1.0, 0.0, 0.0, 0.0])],
        1: [(0.0, 10.0, [0.5, 0.86, 0.0, 0.0])],
    }
    store = _store_with_windows(tmp_path / "audio", space, catalog, spec)
    retriever = AudioVectorRetriever(catalog, store, ListEncoder(space, [[1.0, 0.0, 0.0, 0.0]]))
    plan = QueryPlan(
        original_text="x",
        audio_descriptions=["x"],
        filters={"duration_min_seconds": 30.0},
    )
    outcome = retriever.search_text(plan, limit=5)
    assert [item.asset.name for item in outcome.results] == ["asset-1.wav"]
