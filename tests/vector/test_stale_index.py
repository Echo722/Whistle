# ruff: noqa: E402  (the numpy import guard must run before project imports)
"""Stale vectors must never be returned as if they described the current asset.

The catalog can move on while an artifact stays put: a file is re-exported
(``content_revision`` changes) or only its metadata changes (revision stays, the
text input changes). Both must remove the asset from the candidate list, before
any truncation, and the retriever has to notice a re-published artifact instead
of serving a cached matrix forever.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip(
    "numpy",
    reason="Agent B numeric tests need numpy; it is an optional extra, not a shared dev dependency",
)

from helpers import make_asset, make_catalog, make_space

from audiosearch.contracts.models import QueryPlan
from audiosearch.indexing.encode import text_input_fingerprint
from audiosearch.indexing.store import EmbeddingBatch, EmbeddingRow, EmbeddingStore
from audiosearch.inference.spaces import EmbeddingSpace, SpaceMismatchError
from audiosearch.retrieval.vector.retrievers import TextVectorRetriever


class StubEncoder:
    def __init__(self, space: EmbeddingSpace, query: np.ndarray) -> None:
        self.space = space
        self._query = query

    def load(self):
        return self.space

    def encode_query(self, text: str) -> np.ndarray:
        return self._query

    def close(self) -> None:
        pass


def _space(dimension: int = 4) -> EmbeddingSpace:
    return make_space(tower="text", dimension=dimension)


def _write_text_store(
    directory,
    space,
    assets,
    *,
    fingerprints=None,
    revision_override=None,
) -> EmbeddingStore:
    """One unit vector per asset: asset i -> e_i, so ordering is predictable."""
    count = len(assets)
    vectors = np.eye(space.dimension, dtype=np.float32)[:count]
    rows = [
        EmbeddingRow(
            asset_id=str(asset.asset_id),
            kind="asset",
            start_seconds=0.0,
            end_seconds=float(asset.duration_seconds or 1.0),
        )
        for asset in assets
    ]
    revisions = {
        str(asset.asset_id): (revision_override or {}).get(
            str(asset.asset_id), asset.content_revision
        )
        for asset in assets
    }
    if fingerprints is None:
        fingerprints = {str(asset.asset_id): text_input_fingerprint(asset) for asset in assets}
    store = EmbeddingStore(directory)
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


def _plan() -> QueryPlan:
    return QueryPlan(original_text="whoosh", text_query="whoosh")


def test_stale_revision_is_excluded_and_reported(tmp_path):
    space = _space()
    old_assets = [make_asset(0), make_asset(1)]
    store = _write_text_store(tmp_path / "text", space, old_assets)

    # The catalog now says asset 0 has a new content revision.
    current = [
        make_asset(0, revision="rev-0-updated"),
        make_asset(1),
    ]
    catalog = make_catalog(current)
    retriever = TextVectorRetriever(
        catalog, store, StubEncoder(space, np.array([1.0, 0.0, 0.0, 0.0], np.float32))
    )

    outcome = retriever.search(_plan(), limit=5)
    names = [item.asset.name for item in outcome.results]
    assert "asset-0.wav" not in names, "an R1 vector must not be returned as the R2 asset"
    assert names == ["asset-1.wav"]
    assert outcome.stale_assets == 1
    assert any("re-run the encode step" in note for note in outcome.warnings)


def test_all_stale_vectors_report_unavailable_instead_of_empty_success(tmp_path):
    space = _space()
    old_assets = [make_asset(0)]
    store = _write_text_store(tmp_path / "text", space, old_assets)

    catalog = make_catalog([make_asset(0, revision="rev-0-updated")])
    retriever = TextVectorRetriever(
        catalog, store, StubEncoder(space, np.array([1.0, 0.0, 0.0, 0.0], np.float32))
    )
    outcome = retriever.search(_plan(), limit=5)
    assert outcome.results == []
    assert outcome.state == "unavailable"
    assert "stale=1" in (outcome.reason or "")


def test_metadata_change_with_same_revision_is_excluded(tmp_path):
    space = _space()
    base = make_asset(0)
    store = _write_text_store(tmp_path / "text", space, [base])

    # Same revision, different name: the text vector no longer describes the asset.
    renamed = make_asset(0, name="totally different.wav")
    assert renamed.content_revision == base.content_revision
    catalog = make_catalog([renamed])
    retriever = TextVectorRetriever(
        catalog, store, StubEncoder(space, np.array([1.0, 0.0, 0.0, 0.0], np.float32))
    )
    outcome = retriever.search(_plan(), limit=5)
    assert outcome.results == []
    assert outcome.stale_assets == 1


def test_catalog_missing_vectors_are_reported_separately(tmp_path):
    space = _space()
    store = _write_text_store(tmp_path / "text", space, [make_asset(0)])
    catalog = make_catalog([make_asset(0), make_asset(1)])
    retriever = TextVectorRetriever(
        catalog, store, StubEncoder(space, np.array([1.0, 0.0, 0.0, 0.0], np.float32))
    )
    outcome = retriever.search(_plan(), limit=5)
    assert len(outcome.results) == 1
    assert outcome.missing_assets == 1
    assert any("no stored vectors" in note for note in outcome.warnings)


def test_republished_artifact_is_picked_up_without_a_new_retriever(tmp_path):
    space = _space()
    assets = [make_asset(0), make_asset(1)]
    directory = tmp_path / "text"
    store = _write_text_store(directory, space, assets)
    catalog = make_catalog(assets)
    retriever = TextVectorRetriever(
        catalog, store, StubEncoder(space, np.array([1.0, 0.0, 0.0, 0.0], np.float32))
    )

    first = retriever.search(_plan(), limit=1)
    assert first.results[0].asset.name == "asset-0.wav"

    # Rewrite the artifact with the columns swapped: the same query must now
    # return asset 1. A generation-blind cache would keep returning asset 0.
    swapped = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    rows = [
        EmbeddingRow(str(assets[0].asset_id), "asset", 0.0, 12.0),
        EmbeddingRow(str(assets[1].asset_id), "asset", 0.0, 12.0),
    ]
    EmbeddingStore(directory).write(
        EmbeddingBatch(
            space=space,
            vectors=swapped,
            rows=rows,
            revisions={str(a.asset_id): a.content_revision for a in assets},
            input_fingerprints={str(a.asset_id): text_input_fingerprint(a) for a in assets},
        ),
        manifest_id="sha256:" + "2" * 64,
        library_id="test-library",
        fingerprint={},
        params={},
    )

    second = retriever.search(_plan(), limit=1)
    assert second.results[0].asset.name == "asset-1.wav"


def test_cached_index_is_reused_when_the_artifact_did_not_change(tmp_path):
    space = _space()
    assets = [make_asset(0)]
    store = _write_text_store(tmp_path / "text", space, assets)
    catalog = make_catalog(assets)
    retriever = TextVectorRetriever(
        catalog, store, StubEncoder(space, np.array([1.0, 0.0, 0.0, 0.0], np.float32))
    )
    retriever.search(_plan(), limit=1)
    first_index = retriever.index_state().index
    retriever.search(_plan(), limit=1)
    assert retriever.index_state().index is first_index


def test_artifact_from_another_space_is_refused(tmp_path):
    stored_space = make_space(tower="text", version="v1")
    store = _write_text_store(tmp_path / "text", stored_space, [make_asset(0)])
    other = make_space(tower="text", version="v2")
    catalog = make_catalog([make_asset(0)])
    retriever = TextVectorRetriever(
        catalog, store, StubEncoder(other, np.array([1.0, 0.0, 0.0, 0.0], np.float32))
    )
    with pytest.raises(SpaceMismatchError):
        retriever.search(_plan(), limit=1)
