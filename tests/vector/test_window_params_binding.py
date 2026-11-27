# ruff: noqa: E402  (the numpy import guard must run before project imports)
"""Window parameters must come from the loaded generation, not from defaults.

Audio fingerprints depend on the window policy recorded in the artifact header. If
the catalog snapshot is taken before the artifact is loaded, the first query after
opening a store (or after republishing with another hop) would compare the new
artifact against default parameters and mark every row stale.

Both cases below fail on the pre-fix ordering and pass on the loaded-first one.
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


class DirectionEncoder:
    """Query vector fixed per instance; the audio tower is never invoked."""

    def __init__(self, space, vector):
        self.space = space
        self.config = type("C", (), {"batch_size": 4})()
        self._vector = np.asarray(vector, dtype=np.float32)

    def load(self):
        return self.space

    def encode_texts(self, texts, batch_size=None):
        return np.stack([self._vector] * len(texts))

    def encode_audio_mono(self, windows, sample_rate_hz, *, batch_size=None):  # pragma: no cover
        raise AssertionError("audio encoding is not used in these tests")

    def close(self):
        pass


def _space(dimension: int = 2):
    return make_space(tower="audio", dimension=dimension)


def _write_audio_store(directory, space, assets, *, window, hop, vectors):
    """Publish one generation whose rows follow ``window``/``hop`` exactly."""
    rows, fingerprints = [], {}
    for asset in assets:
        windows = plan_asset_windows(asset, window_seconds=window, hop_seconds=hop)
        rows.extend(
            EmbeddingRow(str(asset.asset_id), "segment", w.start_seconds, w.end_seconds)
            for w in windows
        )
        fingerprints[str(asset.asset_id)] = audio_input_fingerprint(windows)
    matrix = np.stack([np.asarray(vector, np.float32) for vector in vectors])
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    store = EmbeddingStore(directory)
    store.write(
        EmbeddingBatch(
            space=space,
            vectors=matrix,
            rows=rows,
            revisions={str(asset.asset_id): asset.content_revision for asset in assets},
            input_fingerprints=fingerprints,
        ),
        manifest_id="sha256:" + "2" * 64,
        library_id="test-library",
        fingerprint={},
        params={
            "window_seconds": window,
            "hop_seconds": hop,
            "window_policy": f"win{window:g}s-hop{hop:g}s-onset0",
        },
    )
    return store


def _plan() -> QueryPlan:
    return QueryPlan(original_text="whoosh", audio_descriptions=["whoosh"])


def test_first_search_after_opening_a_non_default_hop_artifact(tmp_path):
    """A never-loaded Store must use the artifact's hop on the very first query."""
    space = _space()
    assets = [make_asset(0, duration=40.0), make_asset(1, duration=40.0)]
    catalog = make_catalog(assets)
    directory = tmp_path / "audio"

    # hop=2.0 produces many more windows than the default 10/5 policy.
    windows = plan_asset_windows(assets[0], window_seconds=4.0, hop_seconds=2.0)
    vectors = [[0.0, 1.0]] * len(windows)  # asset 0 wins
    vectors += [[1.0, 0.0]] * len(
        plan_asset_windows(assets[1], window_seconds=4.0, hop_seconds=2.0)
    )
    _write_audio_store(directory, space, assets, window=4.0, hop=2.0, vectors=vectors)

    # Deliberately a fresh Store object: nothing has been loaded yet.
    store = EmbeddingStore(directory)
    assert store.header is None
    retriever = AudioVectorRetriever(catalog, store, DirectionEncoder(space, [0.0, 1.0]))

    outcome = retriever.search_text(_plan(), limit=5)
    assert outcome.state == "ready"
    assert outcome.stale_assets == 0
    assert [item.asset.name for item in outcome.results][0] == "asset-0.wav"


def test_first_search_after_republishing_with_another_hop(tmp_path):
    """The query right after a policy change must already use the new parameters."""
    space = _space()
    assets = [make_asset(0, duration=40.0), make_asset(1, duration=40.0)]
    catalog = make_catalog(assets)
    directory = tmp_path / "audio"

    def publish(window, hop, winner: int):
        first = plan_asset_windows(assets[0], window_seconds=window, hop_seconds=hop)
        second = plan_asset_windows(assets[1], window_seconds=window, hop_seconds=hop)
        vectors = [[0.0, 1.0]] * len(first) if winner == 0 else [[1.0, 0.0]] * len(first)
        vectors += [[1.0, 0.0]] * len(second) if winner == 0 else [[0.0, 1.0]] * len(second)
        _write_audio_store(directory, space, assets, window=window, hop=hop, vectors=vectors)

    publish(10.0, 5.0, winner=0)
    store = EmbeddingStore(directory)
    retriever = AudioVectorRetriever(catalog, store, DirectionEncoder(space, [0.0, 1.0]))

    first = retriever.search_text(_plan(), limit=5)
    assert first.stale_assets == 0
    assert first.results[0].asset.name == "asset-0.wav"

    # Republish with a different window policy and the opposite ranking.
    publish(4.0, 2.0, winner=1)
    second = retriever.search_text(_plan(), limit=5)
    assert second.state == "ready"
    assert second.stale_assets == 0, "first query after republish must not be all-stale"
    assert second.results[0].asset.name == "asset-1.wav", "must use the new generation's vectors"


def test_cache_records_the_parameters_of_the_loaded_generation(tmp_path):
    space = _space()
    assets = [make_asset(0, duration=20.0)]
    catalog = make_catalog(assets)
    directory = tmp_path / "audio"
    windows = plan_asset_windows(assets[0], window_seconds=10.0, hop_seconds=5.0)
    _write_audio_store(
        directory, space, assets, window=10.0, hop=5.0, vectors=[[1.0, 0.0]] * len(windows)
    )
    retriever = AudioVectorRetriever(
        catalog, EmbeddingStore(directory), DirectionEncoder(space, [1.0, 0.0])
    )
    state = retriever.index_state()
    assert state.artifact is not None
    assert state.params.startswith("sha256:")
    # Re-reading the same generation keeps the cached state, params included.
    assert retriever.index_state() is state


def test_default_policy_artifact_still_works(tmp_path):
    """The default 10/5 policy is just one case; it must not be special-cased."""
    space = _space()
    assets = [make_asset(0, duration=30.0)]
    catalog = make_catalog(assets)
    directory = tmp_path / "audio"
    windows = plan_asset_windows(assets[0], window_seconds=10.0, hop_seconds=5.0)
    _write_audio_store(
        directory, space, assets, window=10.0, hop=5.0, vectors=[[1.0, 0.0]] * len(windows)
    )
    retriever = AudioVectorRetriever(
        catalog, EmbeddingStore(directory), DirectionEncoder(space, [1.0, 0.0])
    )
    outcome = retriever.search_text(_plan(), limit=5)
    assert outcome.stale_assets == 0
    assert outcome.results[0].matched_range is not None


def test_snapshot_before_load_would_be_wrong(tmp_path):
    """Documents the hazard: derived before loading, the fingerprint is invalid.

    This guards against reintroducing the ordering: the token and fingerprints
    computed on an unloaded store must differ from the loaded ones, and using the
    former would reject every row.
    """
    space = _space()
    assets = [make_asset(0, duration=40.0)]
    catalog = make_catalog(assets)
    directory = tmp_path / "audio"
    windows = plan_asset_windows(assets[0], window_seconds=4.0, hop_seconds=2.0)
    _write_audio_store(
        directory, space, assets, window=4.0, hop=2.0, vectors=[[1.0, 0.0]] * len(windows)
    )

    store = EmbeddingStore(directory)
    retriever = AudioVectorRetriever(catalog, store, DirectionEncoder(space, [1.0, 0.0]))
    token_before = retriever._catalog_token()
    fingerprints_before = retriever._catalog_fingerprints()

    store.load()
    assert retriever._catalog_token() != token_before
    stale = store.validity(space, list(catalog.assets), fingerprints_before)
    assert stale.valid == frozenset()
    fresh = store.validity(space, list(catalog.assets), retriever._catalog_fingerprints())
    assert len(fresh.valid) == 1
