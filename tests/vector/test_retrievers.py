# ruff: noqa: E402  (the numpy import guard must run before project imports)
"""Retriever behaviour with stub encoders.

Real weights are not loaded here: the retrievers' job is to join a manifest, a
vector store and a query, and to keep original-file ranges and route state
honest. That logic is exercised with deterministic fake encoders, so the test
never depends on a download, a GPU or a particular checkpoint revision.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip(
    "numpy",
    reason="Agent B numeric tests need numpy; it is an optional extra, not a shared dev dependency",
)

from audiosearch.audioio.catalog import ManifestCatalog
from audiosearch.contracts.models import (
    AssetLocation,
    AssetRecord,
    QueryPlan,
    SampleManifest,
    SourceRange,
)
from audiosearch.indexing.encode import (
    audio_input_fingerprint,
    plan_asset_windows,
    text_input_fingerprint,
    window_policy,
)
from audiosearch.indexing.store import EmbeddingBatch, EmbeddingRow, EmbeddingStore
from audiosearch.inference.spaces import EmbeddingSpace, ModelUnavailableError
from audiosearch.retrieval.vector.retrievers import AudioVectorRetriever, TextVectorRetriever

ASSETS = [
    ("11111111-1111-5111-8111-111111111111", "whoosh magic air.wav", 21.3, "rev-a"),
    ("22222222-2222-5222-8222-222222222222", "metal gong hit.wav", 28.8, "rev-b"),
    ("33333333-3333-5333-8333-333333333333", "water drop small.wav", 0.4, "rev-c"),
]


def build_catalog() -> ManifestCatalog:
    records = []
    for asset_id, name, duration, revision in ASSETS:
        records.append(
            AssetRecord(
                asset_id=asset_id,
                content_revision=revision,
                name=name,
                library_id="test-library",
                package="pkg",
                duration_seconds=duration,
                sample_rate_hz=96000,
                channel_count=2,
                bit_depth=24,
                locations=[
                    AssetLocation(
                        location_id=str(asset_id).replace("1111", "9999"),
                        root_id="root",
                        relative_path=f"{name}",
                        availability="online",
                        size_bytes=1,
                        mtime_ns=1,
                    )
                ],
            )
        )
    manifest = SampleManifest(
        manifest_id="sha256:" + "2" * 64,
        generated_at="2026-11-10T00:00:00Z",
        selection_method="sha256-relative-path-v1",
        selection_limit=8,
        library_id="test-library",
        root_id="root",
        assets=records,
    )
    return ManifestCatalog(manifest, {"root": __import__("pathlib").Path(".")})


def text_space() -> EmbeddingSpace:
    return EmbeddingSpace("stub-text", "a" * 40, "text", "v1", 4, True)


def audio_space() -> EmbeddingSpace:
    return EmbeddingSpace("stub-clap", "b" * 40, "audio", "v1", 4, True)


class StubTextEncoder:
    def __init__(self, space: EmbeddingSpace):
        self.space = space
        self.config = type("C", (), {"max_tokens": 32, "query_instruction": ""})()
        self.fingerprint = None

    def load(self):
        return self.space

    def encode_query(self, text: str) -> np.ndarray:
        # Query "whoosh" matches the first asset's stored vector exactly.
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def close(self) -> None:
        pass


class StubAudioEncoder:
    def __init__(self, space: EmbeddingSpace):
        self.space = space
        self.config = type("C", (), {"batch_size": 4})()

    def load(self):
        return self.space

    def encode_texts(self, texts, batch_size=None) -> np.ndarray:
        return np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)

    def close(self) -> None:
        pass


def write_text_store(directory, space) -> EmbeddingStore:
    store = EmbeddingStore(directory)
    vectors = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    rows = [
        EmbeddingRow(asset_id=asset_id, kind="asset", start_seconds=0.0, end_seconds=duration)
        for asset_id, _name, duration, _rev in ASSETS
    ]
    revisions = {asset_id: rev for asset_id, _name, _dur, rev in ASSETS}
    fingerprints = {
        str(asset.asset_id): text_input_fingerprint(asset) for asset in build_catalog().assets
    }
    store.write(
        EmbeddingBatch(
            space=space,
            vectors=vectors,
            rows=rows,
            revisions=revisions,
            input_fingerprints=fingerprints,
        ),
        manifest_id="sha256:" + "2" * 64,
        library_id="test-library",
        fingerprint={},
        params={},
    )
    return store


def write_audio_store(directory, space, *, window_seconds=10.0, hop_seconds=5.0):
    store = EmbeddingStore(directory)
    vectors = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    rows = [
        EmbeddingRow(asset_id=ASSETS[0][0], kind="segment", start_seconds=0.0, end_seconds=10.0),
        EmbeddingRow(asset_id=ASSETS[0][0], kind="segment", start_seconds=5.0, end_seconds=15.0),
        EmbeddingRow(asset_id=ASSETS[1][0], kind="segment", start_seconds=0.0, end_seconds=10.0),
    ]
    revisions = {asset_id: rev for asset_id, _name, _dur, rev in ASSETS[:2]}
    fingerprints = audio_fingerprints(window_seconds=window_seconds, hop_seconds=hop_seconds)
    store.write(
        EmbeddingBatch(
            space=space,
            vectors=vectors,
            rows=rows,
            revisions=revisions,
            input_fingerprints=fingerprints,
        ),
        manifest_id="sha256:" + "2" * 64,
        library_id="test-library",
        fingerprint={},
        params={
            "window_seconds": window_seconds,
            "hop_seconds": hop_seconds,
            "window_policy": window_policy(window_seconds, hop_seconds),
        },
    )
    return store


def audio_fingerprints(*, window_seconds=10.0, hop_seconds=5.0) -> dict[str, str]:
    return {
        str(asset.asset_id): audio_input_fingerprint(
            plan_asset_windows(asset, window_seconds=window_seconds, hop_seconds=hop_seconds)
        )
        for asset in build_catalog().assets
    }


def test_text_retriever_ranks_by_cosine_and_keeps_asset_range(tmp_path):
    catalog = build_catalog()
    space = text_space()
    store = write_text_store(tmp_path / "text", space)
    retriever = TextVectorRetriever(catalog, store, StubTextEncoder(space))
    plan = QueryPlan(original_text="whoosh", text_query="whoosh")
    outcome = retriever.search(plan, limit=2)
    assert outcome.state == "ready"
    assert outcome.results[0].asset.name == "whoosh magic air.wav"
    assert outcome.results[0].score == pytest.approx(1.0)
    assert outcome.results[0].matched_range is None or isinstance(
        outcome.results[0].matched_range, SourceRange
    )


def test_text_retriever_applies_hard_filters_before_ranking(tmp_path):
    catalog = build_catalog()
    space = text_space()
    store = write_text_store(tmp_path / "text", space)
    retriever = TextVectorRetriever(catalog, store, StubTextEncoder(space))
    plan = QueryPlan(
        original_text="whoosh",
        text_query="whoosh",
        filters={"duration_min_seconds": 5.0},
    )
    outcome = retriever.search(plan, limit=5)
    assert "water drop small.wav" not in [item.asset.name for item in outcome.results]


def test_audio_retriever_collapses_windows_to_one_row_per_asset(tmp_path):
    catalog = build_catalog()
    space = audio_space()
    store = write_audio_store(tmp_path / "audio", space)
    retriever = AudioVectorRetriever(catalog, store, StubAudioEncoder(space))
    plan = QueryPlan(original_text="metal gong", audio_descriptions=["metal gong"])
    outcome = retriever.search_text(plan, limit=10)
    names = [item.asset.name for item in outcome.results]
    assert names.count("whoosh magic air.wav") == 1
    top = outcome.results[0]
    assert top.matched_range is not None
    assert top.matched_range.start_seconds == pytest.approx(0.0)
    assert top.matched_range.end_seconds == pytest.approx(10.0)


def test_audio_retriever_uses_first_window_for_the_tie(tmp_path):
    catalog = build_catalog()
    space = audio_space()
    store = write_audio_store(tmp_path / "audio", space)
    retriever = AudioVectorRetriever(catalog, store, StubAudioEncoder(space))
    plan = QueryPlan(original_text="metal gong", audio_descriptions=["metal gong"])
    outcome = retriever.search_text(plan, limit=1)
    assert outcome.results[0].asset.name == "whoosh magic air.wav"
    assert outcome.results[0].matched_range.start_seconds == pytest.approx(0.0)


def test_audio_retriever_filters_before_truncating(tmp_path):
    """A narrow filter must not be starved by high-scoring excluded windows.

    Eight assets, one window each, with scores descending by index; the filter
    keeps only the three worst. Truncating first would return one hit, so this
    pins "filter inside the scan, then limit".
    """
    space = EmbeddingSpace("stub-clap", "b" * 40, "audio", "v1", 8, True)
    dim = space.dimension
    catalog = _catalog_of_eight()
    directory = tmp_path / "eight"
    store = EmbeddingStore(directory)
    vectors = np.eye(dim, dtype=np.float32)
    rows = [
        EmbeddingRow(
            asset_id=f"{index:08d}-0000-5000-8000-000000000000",
            kind="segment",
            start_seconds=0.0,
            end_seconds=5.0,
        )
        for index in range(dim)
    ]
    revisions = {row.asset_id: f"rev-{row.asset_id}" for row in rows}
    fingerprints = {
        str(asset.asset_id): audio_input_fingerprint(
            plan_asset_windows(asset, window_seconds=10.0, hop_seconds=5.0)
        )
        for asset in catalog.assets
    }
    store.write(
        EmbeddingBatch(
            space=space,
            vectors=vectors,
            rows=rows,
            revisions=revisions,
            input_fingerprints=fingerprints,
        ),
        manifest_id="sha256:" + "2" * 64,
        library_id="test-library",
        fingerprint={},
        params={
            "window_seconds": 10.0,
            "hop_seconds": 5.0,
            "window_policy": window_policy(10.0, 5.0),
        },
    )

    class OrderedEncoder(StubAudioEncoder):
        def encode_texts(self, texts, batch_size=None):
            query = np.zeros(dim, dtype=np.float32)
            for index in range(dim):
                query[index] = (dim - index) / dim
            return query[None, :]

    retriever = AudioVectorRetriever(catalog, store, OrderedEncoder(space))
    plan = QueryPlan(
        original_text="only short files",
        audio_descriptions=["only short files"],
        filters={"duration_max_seconds": 6.0},
    )
    outcome = retriever.search_text(plan, limit=3)
    # Assets 5, 6, 7 are the only short ones; all three must survive.
    assert len(outcome.results) == 3
    assert {item.asset.name for item in outcome.results} == {
        "asset-5.wav",
        "asset-6.wav",
        "asset-7.wav",
    }


def _catalog_of_eight() -> ManifestCatalog:
    records = []
    for index in range(8):
        records.append(
            AssetRecord(
                asset_id=f"{index:08d}-0000-5000-8000-000000000000",
                content_revision=f"rev-{index:08d}-0000-5000-8000-000000000000",
                name=f"asset-{index}.wav",
                library_id="test-library",
                package="pkg",
                # 60 s for the loud ones, 5 s for the three the filter keeps.
                duration_seconds=60.0 if index < 5 else 5.0,
                sample_rate_hz=96000,
                channel_count=2,
                bit_depth=24,
                locations=[
                    AssetLocation(
                        location_id=f"{index:08d}-1111-5000-8000-000000000000",
                        root_id="root",
                        relative_path=f"asset-{index}.wav",
                        availability="online",
                        size_bytes=1,
                        mtime_ns=1,
                    )
                ],
            )
        )
    manifest = SampleManifest(
        manifest_id="sha256:" + "3" * 64,
        generated_at="2026-11-10T00:00:00Z",
        selection_method="sha256-relative-path-v1",
        selection_limit=16,
        library_id="test-library",
        root_id="root",
        assets=records,
    )
    return ManifestCatalog(manifest, {"root": __import__("pathlib").Path(".")})


def test_audio_retriever_without_vectors_reports_unavailable_assets(tmp_path):
    catalog = build_catalog()
    space = audio_space()
    store = EmbeddingStore(tmp_path / "empty")
    retriever = AudioVectorRetriever(catalog, store, StubAudioEncoder(space))
    with pytest.raises(ModelUnavailableError):
        retriever.index()


def test_text_retriever_without_vectors_raises_model_unavailable(tmp_path):
    catalog = build_catalog()
    space = text_space()
    retriever = TextVectorRetriever(
        catalog, EmbeddingStore(tmp_path / "nope"), StubTextEncoder(space)
    )
    with pytest.raises(ModelUnavailableError):
        retriever.index()


def test_mismatched_space_between_store_and_encoder_is_refused(tmp_path):
    catalog = build_catalog()
    stored = text_space()
    store = write_text_store(tmp_path / "text", stored)
    other = EmbeddingSpace("stub-text", "a" * 40, "text", "v2", 4, True)
    retriever = TextVectorRetriever(catalog, store, StubTextEncoder(other))
    # The encoder declares v2 while the store holds v1: refuse, do not mix.
    with pytest.raises(Exception, match="space"):
        retriever.search(QueryPlan(original_text="whoosh", text_query="whoosh"), limit=3)
