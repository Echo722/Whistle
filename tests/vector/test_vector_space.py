# ruff: noqa: E402  (the numpy import guard must run before project imports)
"""Agent B tests for vector spaces, stores, exact search and RRF fusion.

Everything here uses synthetic vectors with a declared space id: the point is to
prove the guard rails (no cross-space comparison, no stale reuse, no silent
dimension mismatch), not to measure retrieval quality.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip(
    "numpy",
    reason="Agent B numeric tests need numpy; it is an optional extra, not a shared dev dependency",
)

from audiosearch.contracts.models import RetrievalSource
from audiosearch.indexing.store import EmbeddingBatch, EmbeddingRow, EmbeddingStore
from audiosearch.inference.spaces import EmbeddingSpace, SpaceMismatchError, l2_normalize
from audiosearch.retrieval.fusion.rrf import RouteHit, RouteResult, fuse
from audiosearch.retrieval.vector.exact import DimensionError, ExactVectorIndex


def _fingerprint(label: str) -> str:
    """A syntactically valid input fingerprint for synthetic batches."""
    import hashlib as _hashlib

    return "sha256:" + _hashlib.sha256(label.encode()).hexdigest()


def make_space(tower="text", dimension=8, model="test-model", version="v1") -> EmbeddingSpace:
    return EmbeddingSpace(
        model_id=model,
        revision="0" * 40,
        tower=tower,
        preprocessing_version=version,
        dimension=dimension,
        normalized=True,
    )


def make_batch(space: EmbeddingSpace, count: int = 4, seed: int = 0) -> EmbeddingBatch:
    rng = np.random.default_rng(seed)
    vectors = l2_normalize(rng.normal(size=(count, space.dimension)).astype(np.float32))
    rows = [
        EmbeddingRow(
            asset_id=f"asset-{index}",
            kind="segment",
            start_seconds=float(index),
            end_seconds=float(index) + 10.0,
        )
        for index in range(count)
    ]
    return EmbeddingBatch(
        space=space,
        vectors=vectors,
        rows=rows,
        revisions={f"asset-{index}": f"rev-{index}" for index in range(count)},
        input_fingerprints={
            f"asset-{index}": _fingerprint(f"asset-{index}") for index in range(count)
        },
    )


def test_normalized_vectors_have_unit_norm():
    space = make_space()
    batch = make_batch(space)
    norms = np.linalg.norm(batch.vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_batch_rejects_non_finite_and_wrong_width():
    space = make_space(dimension=4)
    bad = np.zeros((2, 4), dtype=np.float32)
    bad[0, 0] = np.nan
    rows = [EmbeddingRow("a", "asset", 0.0, 1.0), EmbeddingRow("b", "asset", 0.0, 1.0)]
    with pytest.raises(ValueError, match="finite"):
        EmbeddingBatch(space=space, vectors=bad, rows=rows)
    with pytest.raises(ValueError, match="width"):
        EmbeddingBatch(space=space, vectors=np.zeros((2, 3), dtype=np.float32), rows=rows)


def test_exact_search_returns_query_itself_first():
    space = make_space()
    batch = make_batch(space)
    index = ExactVectorIndex.from_batch(batch)
    for row_index in range(len(batch.rows)):
        hits = index.search(batch.vectors[row_index], space=space, limit=2)
        assert hits[0].row_index == row_index
        assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_exact_search_keeps_the_row_time_range():
    space = make_space()
    batch = make_batch(space)
    index = ExactVectorIndex.from_batch(batch)
    hits = index.search(batch.vectors[2], space=space, limit=1)
    assert hits[0].start_seconds == pytest.approx(2.0)
    assert hits[0].end_seconds == pytest.approx(12.0)


def test_cross_space_search_is_refused_even_at_equal_dimension():
    audio_space = make_space(tower="audio", dimension=8)
    text_space = make_space(tower="text", dimension=8)
    index = ExactVectorIndex.from_batch(make_batch(audio_space))
    query = np.ones(8, dtype=np.float32)
    with pytest.raises(SpaceMismatchError):
        index.search(query, space=text_space, limit=1)


def test_dimension_mismatch_is_reported_not_truncated():
    space = make_space(dimension=8)
    index = ExactVectorIndex.from_batch(make_batch(space))
    with pytest.raises(DimensionError):
        index.search(np.ones(7, dtype=np.float32), space=space, limit=1)


def test_search_collapses_long_files_to_one_row():
    space = make_space(dimension=4)
    rng = np.random.default_rng(7)
    rows = [
        EmbeddingRow("long", "segment", 0.0, 10.0),
        EmbeddingRow("long", "segment", 5.0, 15.0),
        EmbeddingRow("short", "asset", 0.0, 2.0),
    ]
    vectors = l2_normalize(rng.normal(size=(3, 4)).astype(np.float32))
    index = ExactVectorIndex(space=space, vectors=vectors, rows=rows)
    collapsed = index.search(vectors[0], space=space, limit=10)
    assert len(collapsed) == 2
    uncollapsed = index.search(vectors[0], space=space, limit=10, collapse_per_asset=False)
    assert len(uncollapsed) == 3


def test_store_roundtrip_and_revision_aware_reuse(tmp_path):
    space = make_space()
    store = EmbeddingStore(tmp_path / "text-space")
    batch = make_batch(space, count=3)
    store.write(
        batch,
        manifest_id="sha256:" + "1" * 64,
        library_id="lib",
        fingerprint={"m": 1},
        params={"p": 2},
    )
    reloaded = EmbeddingStore(tmp_path / "text-space").load()
    assert reloaded.vectors.shape == (3, space.dimension)
    assert np.allclose(reloaded.vectors, batch.vectors)
    assert reloaded.header["manifest_id"] == "sha256:" + "1" * 64

    class _Asset:
        def __init__(self, index: int, revision: str):
            self.asset_id = f"asset-{index}"
            self.content_revision = revision

    keep = reloaded.reusable_assets(
        space,
        [_Asset(0, "rev-0"), _Asset(1, "changed")],
        {"asset-0": _fingerprint("asset-0"), "asset-1": _fingerprint("asset-1")},
    )
    assert keep == {"asset-0"}
    # A changed input with an unchanged revision invalidates the stored vector.
    assert (
        reloaded.reusable_assets(
            space, [_Asset(0, "rev-0")], {"asset-0": _fingerprint("different-text")}
        )
        == set()
    )


def test_store_refuses_reuse_from_another_space(tmp_path):
    space = make_space(version="v1")
    store = EmbeddingStore(tmp_path / "s")
    store.write(make_batch(space), manifest_id="m", library_id="l", fingerprint={}, params={})
    other = make_space(version="v2")
    store.load()

    class _Asset:
        asset_id = "asset-0"
        content_revision = "rev-0"

    assert store.reusable_assets(other, [_Asset()], {"asset-0": _fingerprint("asset-0")}) == set()


def test_store_write_is_atomic_and_replaces_content(tmp_path):
    space = make_space()
    directory = tmp_path / "s"
    store = EmbeddingStore(directory)
    store.write(
        make_batch(space, count=4), manifest_id="m", library_id="l", fingerprint={}, params={}
    )
    store.write(
        make_batch(space, count=2, seed=5),
        manifest_id="m",
        library_id="l",
        fingerprint={},
        params={},
    )
    assert not any(p.name.startswith(".s.") for p in directory.parent.iterdir())
    reloaded = EmbeddingStore(directory).load()
    assert reloaded.vectors.shape[0] == 2


def test_rrf_rank_fusion_uses_ranks_not_raw_scores():
    routes = [
        RouteResult(
            source=RetrievalSource.TEXT,
            state="ready",
            hits=(RouteHit("a", 1, score=0.9), RouteHit("b", 2, score=0.1)),
        ),
        RouteResult(
            source=RetrievalSource.AUDIO,
            state="ready",
            hits=(RouteHit("b", 1, score=0.2, start_seconds=1.0, end_seconds=2.0),),
        ),
    ]
    fused = fuse(routes, k=60, limit=5)
    assert [hit.asset_id for hit in fused] == ["b", "a"]
    assert fused[0].sources == (RetrievalSource.AUDIO, RetrievalSource.TEXT)
    assert fused[0].start_seconds == pytest.approx(1.0)
    assert fused[0].rrf_score == pytest.approx(1 / 61 + 1 / 62, rel=1e-9)


def test_unavailable_route_contributes_nothing():
    routes = [
        RouteResult(
            source=RetrievalSource.TEXT,
            state="ready",
            hits=(RouteHit("a", 1, score=0.4),),
        ),
        RouteResult(
            source=RetrievalSource.AUDIO,
            state="unavailable",
            reason="model not installed",
            hits=(RouteHit("z", 1, score=0.99),),
        ),
    ]
    fused = fuse(routes, limit=5)
    assert [hit.asset_id for hit in fused] == ["a"]


def test_fusion_rejects_zero_based_ranks():
    routes = [
        RouteResult(
            source=RetrievalSource.KEYWORD, state="ready", hits=(RouteHit("a", 0, score=None),)
        )
    ]
    with pytest.raises(ValueError, match="1-based"):
        fuse(routes, limit=5)
