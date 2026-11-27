# ruff: noqa: E402  (the numpy import guard must run before project imports)
"""Fingerprint strictness: unverifiable rows are excluded, not merely warned about.

A vector is only usable when the artifact can prove which inputs produced it. A
legacy ``embeddings-v1`` artifact has no fingerprints, so it stays readable for
diagnostics but never feeds a normal search. A v2 artifact must carry a
well-formed fingerprint for the asset; missing or malformed values on either side
must not compare equal by accident.
"""

from __future__ import annotations

import json

import pytest

np = pytest.importorskip(
    "numpy",
    reason="Agent B numeric tests need numpy; it is an optional extra, not a shared dev dependency",
)
from helpers import make_asset, make_catalog, make_space

from audiosearch.contracts.models import QueryPlan
from audiosearch.indexing.encode import text_input_fingerprint
from audiosearch.indexing.store import (
    POINTER_NAME,
    EmbeddingBatch,
    EmbeddingRow,
    EmbeddingStore,
    is_valid_fingerprint,
)
from audiosearch.retrieval.vector.retrievers import TextVectorRetriever


class StubEncoder:
    def __init__(self, space):
        self.space = space
        self.query = np.array([1.0, 0.0], dtype=np.float32)

    def load(self):
        return self.space

    def encode_query(self, text):
        return self.query

    def close(self):
        pass


def _space():
    return make_space(tower="text", dimension=2)


def _plan():
    return QueryPlan(original_text="whoosh", text_query="whoosh")


def _batch(space, asset, fingerprint=None):
    return EmbeddingBatch(
        space=space,
        vectors=np.array([[1.0, 0.0]], dtype=np.float32),
        rows=[EmbeddingRow(str(asset.asset_id), "asset", 0.0, 12.0)],
        revisions={str(asset.asset_id): asset.content_revision},
        input_fingerprints=(
            {str(asset.asset_id): text_input_fingerprint(asset)}
            if fingerprint is None
            else {str(asset.asset_id): fingerprint}
        ),
    )


def _write_legacy_flat(directory, space, asset) -> None:
    """An embeddings-v1 artifact: no fingerprints, old checksum formula."""
    from audiosearch.indexing.store import _checksum_v1

    directory.mkdir(parents=True, exist_ok=True)
    row = EmbeddingRow(str(asset.asset_id), "asset", 0.0, 12.0)
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)
    np.save(directory / "vectors.npy", vectors)
    (directory / "rows.jsonl").write_text(json.dumps(row.as_dict()) + "\n", encoding="utf-8")
    header = {
        "schema_version": "embeddings-v1",
        "space": space.as_dict(),
        "manifest_id": "sha256:" + "2" * 64,
        "library_id": "test-library",
        "model": {},
        "params": {},
        "count": 1,
        "dimension": space.dimension,
        "asset_count": 1,
        "asset_revisions": {str(asset.asset_id): asset.content_revision},
        "checksum": _checksum_v1(vectors, [row]),
    }
    (directory / "header.json").write_text(json.dumps(header), encoding="utf-8")


# -- legacy artifacts --------------------------------------------------------
def test_legacy_artifact_is_inspectable_but_not_searchable(tmp_path):
    space = _space()
    asset = make_asset(0)
    directory = tmp_path / "legacy"
    _write_legacy_flat(directory, space, asset)

    store = EmbeddingStore(directory).load()
    assert store.schema_version == "embeddings-v1"
    assert store.vectors.shape == (1, 2)  # still readable for diagnostics

    # Diagnostics can see the rows, but they are flagged as unverified.
    diagnostic = store.validity(
        space,
        [asset],
        {str(asset.asset_id): text_input_fingerprint(asset)},
        allow_unfingerprinted=True,
    )
    assert diagnostic.valid == frozenset({str(asset.asset_id)})
    assert diagnostic.unverified == frozenset({str(asset.asset_id)})

    # A normal search refuses them and says why.
    retriever = TextVectorRetriever(make_catalog([asset]), store, StubEncoder(space))
    outcome = retriever.search(_plan(), limit=5)
    assert outcome.results == []
    assert outcome.state == "unavailable"
    assert "legacy" in (outcome.reason or "")
    assert outcome.unverified_assets == 1


def test_legacy_artifact_is_not_reused(tmp_path):
    space = _space()
    asset = make_asset(0)
    directory = tmp_path / "legacy"
    _write_legacy_flat(directory, space, asset)
    store = EmbeddingStore(directory).load()
    fingerprints = {str(asset.asset_id): text_input_fingerprint(asset)}
    assert store.reusable_assets(space, [asset], fingerprints) == set()
    assert store.validity(space, [asset], fingerprints).needs_rebuild


def test_migrating_a_legacy_artifact_makes_it_searchable(tmp_path):
    space = _space()
    asset = make_asset(0)
    directory = tmp_path / "store"
    _write_legacy_flat(directory, space, asset)
    store = EmbeddingStore(directory)
    store.write(
        _batch(space, asset),
        manifest_id="sha256:" + "2" * 64,
        library_id="test-library",
        fingerprint={},
        params={},
    )
    retriever = TextVectorRetriever(make_catalog([asset]), store, StubEncoder(space))
    outcome = retriever.search(_plan(), limit=5)
    assert outcome.state == "ready"
    assert len(outcome.results) == 1
    assert outcome.unverified_assets == 0


# -- v2 fingerprint requirements --------------------------------------------
def test_missing_stored_fingerprint_is_excluded(tmp_path):
    space = _space()
    asset = make_asset(0)
    store = EmbeddingStore(tmp_path / "store")
    store.write(
        _batch(space, asset),
        manifest_id="sha256:" + "2" * 64,
        library_id="test-library",
        fingerprint={},
        params={},
    )
    # Simulate an artifact edited to drop one fingerprint (re-signed so the
    # checksum is not the thing that rejects it).
    from audiosearch.indexing.store import _checksum_v2

    path = store.directory / (store.directory / POINTER_NAME).read_text().strip() / "header.json"
    header = json.loads(path.read_text(encoding="utf-8"))
    header["input_fingerprints"] = {}
    header["checksum"] = _checksum_v2(header, store.vectors, store.rows)
    path.write_text(json.dumps(header), encoding="utf-8")

    loaded = EmbeddingStore(store.directory).load()
    fingerprints = {str(asset.asset_id): text_input_fingerprint(asset)}
    report = loaded.validity(space, [asset], fingerprints)
    assert report.valid == frozenset()
    assert str(asset.asset_id) in report.stale
    assert loaded.reusable_assets(space, [asset], fingerprints) == set()


def test_malformed_stored_fingerprint_is_excluded(tmp_path):
    space = _space()
    asset = make_asset(0)
    # Writing refuses a malformed fingerprint, which is the first line of defence.
    with pytest.raises(ValueError, match="fingerprint"):
        EmbeddingBatch(
            space=space,
            vectors=np.array([[1.0, 0.0]], dtype=np.float32),
            rows=[EmbeddingRow(str(asset.asset_id), "asset", 0.0, 12.0)],
            revisions={str(asset.asset_id): asset.content_revision},
            input_fingerprints={str(asset.asset_id): "not-a-fingerprint"},
        )


def test_both_sides_missing_cannot_be_treated_as_equal(tmp_path):
    space = _space()
    asset = make_asset(0)
    store = EmbeddingStore(tmp_path / "store")
    store.write(
        _batch(space, asset),
        manifest_id="sha256:" + "2" * 64,
        library_id="test-library",
        fingerprint={},
        params={},
    )
    from audiosearch.indexing.store import _checksum_v2

    path = store.directory / (store.directory / POINTER_NAME).read_text().strip() / "header.json"
    header = json.loads(path.read_text(encoding="utf-8"))
    header["input_fingerprints"] = {}
    header["checksum"] = _checksum_v2(header, store.vectors, store.rows)
    path.write_text(json.dumps(header), encoding="utf-8")

    loaded = EmbeddingStore(store.directory).load()
    # Empty on both sides must not compare equal into "valid".
    report = loaded.validity(space, [asset], {})
    assert report.valid == frozenset()
    assert report.stale == frozenset({str(asset.asset_id)})


def test_expected_fingerprint_missing_is_excluded(tmp_path):
    space = _space()
    asset = make_asset(0)
    store = EmbeddingStore(tmp_path / "store")
    store.write(
        _batch(space, asset),
        manifest_id="sha256:" + "2" * 64,
        library_id="test-library",
        fingerprint={},
        params={},
    )
    loaded = EmbeddingStore(store.directory).load()
    report = loaded.validity(space, [asset], {})
    assert report.valid == frozenset()
    assert str(asset.asset_id) in report.stale


def test_complete_matching_v2_artifact_is_reused_and_searchable(tmp_path):
    space = _space()
    asset = make_asset(0)
    store = EmbeddingStore(tmp_path / "store")
    store.write(
        _batch(space, asset),
        manifest_id="sha256:" + "2" * 64,
        library_id="test-library",
        fingerprint={},
        params={},
    )
    loaded = EmbeddingStore(store.directory).load()
    fingerprints = {str(asset.asset_id): text_input_fingerprint(asset)}
    assert loaded.validity(space, [asset], fingerprints).valid == frozenset({str(asset.asset_id)})
    assert loaded.reusable_assets(space, [asset], fingerprints) == {str(asset.asset_id)}

    retriever = TextVectorRetriever(make_catalog([asset]), store, StubEncoder(space))
    outcome = retriever.search(_plan(), limit=5)
    assert outcome.state == "ready"
    assert len(outcome.results) == 1
    assert outcome.unverified_assets == 0


def test_fingerprint_format_is_enforced():
    good = "sha256:" + "a" * 64
    assert is_valid_fingerprint(good)
    assert not is_valid_fingerprint(None)
    assert not is_valid_fingerprint("")
    assert not is_valid_fingerprint("sha256:" + "a" * 63)
    assert not is_valid_fingerprint("md5:" + "a" * 64)
    assert not is_valid_fingerprint("sha256:" + "A" * 64)


def test_edited_legacy_header_stays_unsearchable(tmp_path):
    """A v1 header is not covered by its checksum; it must still be unusable.

    This is the honest limit of the old format: editing ``asset_revisions`` cannot
    be detected, which is exactly why v1 rows never feed a normal search.
    """
    space = _space()
    asset = make_asset(0)
    directory = tmp_path / "store"
    _write_legacy_flat(directory, space, asset)
    path = directory / "header.json"
    header = json.loads(path.read_text(encoding="utf-8"))
    header["asset_revisions"][str(asset.asset_id)] = "forged"
    path.write_text(json.dumps(header), encoding="utf-8")

    loaded = EmbeddingStore(directory).load()
    assert loaded.schema_version == "embeddings-v1"
    fingerprints = {str(asset.asset_id): text_input_fingerprint(asset)}
    report = loaded.validity(space, [asset], fingerprints)
    assert report.valid == frozenset()
    # The forged revision no longer matches, so the row is stale rather than
    # merely unverified; either way it is unusable.
    assert report.stale == frozenset({str(asset.asset_id)})

    retriever = TextVectorRetriever(make_catalog([asset]), loaded, StubEncoder(space))
    assert retriever.search(_plan(), limit=5).results == []
