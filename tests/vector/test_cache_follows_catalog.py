# ruff: noqa: E402  (the numpy import guard must run before project imports)
"""The index cache must follow the catalog, not only the artifact.

A single retriever instance is reused across all cases here, because the bug only
appears then: the artifact generation stays the same while an asset's revision or
metadata changes, so a generation-only cache keeps an outdated ``validity`` and
can label old vectors with the new record.

``MutableCatalog`` stands in for a catalog whose rows are updated in place (the
service reloads records without re-encoding).
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
from audiosearch.retrieval.vector.retrievers import TextVectorRetriever


class MutableCatalog:
    """Catalog double whose records can be swapped while a retriever holds it."""

    def __init__(self, catalog) -> None:
        self._catalog = catalog
        self._by_id = {asset.asset_id: asset for asset in catalog.assets}

    @property
    def assets(self):
        return list(self._by_id.values())

    @property
    def manifest(self):
        return self._catalog.manifest

    def get_asset(self, asset_id):
        return self._by_id.get(asset_id)

    def replace(self, asset) -> None:
        self._by_id[asset.asset_id] = asset

    def restore(self, asset) -> None:
        self._by_id[asset.asset_id] = asset


class StubEncoder:
    def __init__(self, space, query):
        self.space = space
        self._query = query

    def load(self):
        return self.space

    def encode_query(self, text):
        return self._query

    def close(self):
        pass


def _space():
    return make_space(tower="text", dimension=2)


def _store_with(tmp_path, space, assets):
    directory = tmp_path / "text"
    store = EmbeddingStore(directory)
    # One unit vector per asset so the ranking stays trivially predictable.
    vectors = np.eye(space.dimension, dtype=np.float32)[: len(assets)]
    rows = [EmbeddingRow(str(asset.asset_id), "asset", 0.0, 12.0) for asset in assets]
    store.write(
        EmbeddingBatch(
            space=space,
            vectors=vectors,
            rows=rows,
            revisions={str(a.asset_id): a.content_revision for a in assets},
            input_fingerprints={str(a.asset_id): text_input_fingerprint(a) for a in assets},
        ),
        manifest_id="sha256:" + "2" * 64,
        library_id="test-library",
        fingerprint={},
        params={},
    )
    return store


def _plan():
    return QueryPlan(original_text="whoosh", text_query="whoosh")


def _names(outcome):
    return [item.asset.name for item in outcome.results]


def test_revision_change_is_seen_by_the_same_retriever(tmp_path):
    space = _space()
    base = [make_asset(0), make_asset(1)]
    store = _store_with(tmp_path, space, base)
    catalog = MutableCatalog(make_catalog(base))
    retriever = TextVectorRetriever(
        catalog, store, StubEncoder(space, np.array([1.0, 0.0], np.float32))
    )

    first = retriever.search(_plan(), limit=5)
    assert set(_names(first)) == {"asset-0.wav", "asset-1.wav"}

    catalog.replace(make_asset(0, revision="rev-0-updated"))
    second = retriever.search(_plan(), limit=5)
    assert _names(second) == ["asset-1.wav"]
    assert second.stale_assets == 1

    catalog.restore(make_asset(0))
    third = retriever.search(_plan(), limit=5)
    assert set(_names(third)) == {"asset-0.wav", "asset-1.wav"}
    assert third.stale_assets == 0


def test_metadata_only_change_is_seen_by_the_same_retriever(tmp_path):
    space = _space()
    base = [make_asset(0), make_asset(1)]
    store = _store_with(tmp_path, space, base)
    catalog = MutableCatalog(make_catalog(base))
    retriever = TextVectorRetriever(
        catalog, store, StubEncoder(space, np.array([1.0, 0.0], np.float32))
    )
    assert len(retriever.search(_plan(), limit=5).results) == 2

    # Same revision, new description: only the input fingerprint can catch this.
    described = make_asset(0).model_copy(
        update={"description": "a different sound", "description_source": "user"}
    )
    catalog.replace(described)
    second = retriever.search(_plan(), limit=5)
    assert _names(second) == ["asset-1.wav"]
    assert second.stale_assets == 1

    catalog.restore(make_asset(0))
    third = retriever.search(_plan(), limit=5)
    assert len(third.results) == 2


def test_rename_is_seen_by_the_same_retriever(tmp_path):
    space = _space()
    base = [make_asset(0)]
    store = _store_with(tmp_path, space, base)
    catalog = MutableCatalog(make_catalog(base))
    retriever = TextVectorRetriever(
        catalog, store, StubEncoder(space, np.array([1.0, 0.0], np.float32))
    )
    assert retriever.search(_plan(), limit=5).results

    catalog.replace(make_asset(0, name="renamed.wav"))
    outcome = retriever.search(_plan(), limit=5)
    assert outcome.results == []
    assert outcome.state == "unavailable"
    assert outcome.stale_assets == 1


def test_records_returned_carry_the_current_revision(tmp_path):
    """A result must never pair an old vector with a newer AssetRecord."""
    space = _space()
    base = [make_asset(0), make_asset(1)]
    store = _store_with(tmp_path, space, base)
    catalog = MutableCatalog(make_catalog(base))
    retriever = TextVectorRetriever(
        catalog, store, StubEncoder(space, np.array([1.0, 0.0], np.float32))
    )
    retriever.search(_plan(), limit=5)

    catalog.replace(make_asset(0, revision="rev-0-updated"))
    outcome = retriever.search(_plan(), limit=5)
    returned = {str(item.asset.asset_id): item.asset.content_revision for item in outcome.results}
    assert (
        returned.get(str(make_asset(0).asset_id)) is None
        or returned[str(make_asset(0).asset_id)] == "rev-0-updated"
    )
    # The excluded asset is simply absent, never present with an old revision.
    assert str(make_asset(0).asset_id) not in returned


def test_unchanged_catalog_and_artifact_keep_using_the_cache(tmp_path):
    space = _space()
    base = [make_asset(0)]
    store = _store_with(tmp_path, space, base)
    catalog = MutableCatalog(make_catalog(base))
    retriever = TextVectorRetriever(
        catalog, store, StubEncoder(space, np.array([1.0, 0.0], np.float32))
    )
    retriever.search(_plan(), limit=5)
    first = retriever.index_state().index
    retriever.search(_plan(), limit=5)
    assert retriever.index_state().index is first


def test_new_artifact_generation_still_invalidates_the_cache(tmp_path):
    space = _space()
    base = [make_asset(0)]
    store = _store_with(tmp_path, space, base)
    catalog = MutableCatalog(make_catalog(base))
    retriever = TextVectorRetriever(
        catalog, store, StubEncoder(space, np.array([1.0, 0.0], np.float32))
    )
    retriever.search(_plan(), limit=5)
    old_index = retriever.index_state().index

    store.write(
        EmbeddingBatch(
            space=space,
            vectors=np.array([[0.0, 1.0]], dtype=np.float32),
            rows=[EmbeddingRow(str(base[0].asset_id), "asset", 0.0, 12.0)],
            revisions={str(base[0].asset_id): base[0].content_revision},
            input_fingerprints={str(base[0].asset_id): text_input_fingerprint(base[0])},
        ),
        manifest_id="sha256:" + "2" * 64,
        library_id="test-library",
        fingerprint={},
        params={},
    )
    retriever.search(_plan(), limit=5)
    assert retriever.index_state().index is not old_index


@pytest.mark.parametrize(
    "change",
    [
        lambda asset: asset.model_copy(update={"keywords": ["new-keyword"]}),
        lambda asset: asset.model_copy(update={"package": "another/package"}),
        lambda asset: asset.model_copy(
            update={"description": "text", "description_source": "user"}
        ),
    ],
)
def test_every_encoded_field_invalidates_the_cached_validity(tmp_path, change):
    space = _space()
    base = [make_asset(0)]
    store = _store_with(tmp_path, space, base)
    catalog = MutableCatalog(make_catalog(base))
    retriever = TextVectorRetriever(
        catalog, store, StubEncoder(space, np.array([1.0, 0.0], np.float32))
    )
    assert retriever.search(_plan(), limit=5).results

    catalog.replace(change(base[0]))
    outcome = retriever.search(_plan(), limit=5)
    assert outcome.results == []
    assert outcome.stale_assets == 1
