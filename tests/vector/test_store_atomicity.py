# ruff: noqa: E402  (the numpy import guard must run before project imports)
"""Artifact integrity: staged publishes, interruption recovery, load validation.

Every case uses temporary directories and synthetic vectors. The point is the
commit protocol, not retrieval quality:

* a write publishes by renaming ``CURRENT``, so a failure before that rename
  leaves the previous generation loadable;
* a corrupted or incompatible artifact is refused with a clear error instead of
  being reused.
"""

from __future__ import annotations

import json
import shutil

import pytest

np = pytest.importorskip(
    "numpy",
    reason="Agent B numeric tests need numpy; it is an optional extra, not a shared dev dependency",
)

from audiosearch.indexing.store import (
    GENERATIONS_DIR,
    POINTER_NAME,
    SCHEMA_VERSION,
    ArtifactCorruptError,
    ArtifactIncompatibleError,
    EmbeddingBatch,
    EmbeddingRow,
    EmbeddingStore,
    _checksum_v1,
)
from audiosearch.inference.spaces import EmbeddingSpace, l2_normalize
from audiosearch.inference.spaces import EmbeddingSpace as Space


def _fingerprint(label: str) -> str:
    """A syntactically valid input fingerprint for synthetic batches."""
    import hashlib as _hashlib

    return "sha256:" + _hashlib.sha256(label.encode()).hexdigest()


def make_space(dimension: int = 6, version: str = "v1") -> EmbeddingSpace:
    return Space("test-model", "a" * 40, "text", version, dimension, True)


def make_batch(space: EmbeddingSpace, count: int = 3, seed: int = 0) -> EmbeddingBatch:
    rng = np.random.default_rng(seed)
    return EmbeddingBatch(
        space=space,
        vectors=l2_normalize(rng.normal(size=(count, space.dimension)).astype(np.float32)),
        rows=[
            EmbeddingRow(f"asset-{index}", "segment", float(index), float(index) + 4.0)
            for index in range(count)
        ],
        revisions={f"asset-{index}": f"rev-{index}" for index in range(count)},
        input_fingerprints={
            f"asset-{index}": _fingerprint(f"asset-{index}") for index in range(count)
        },
    )


def write(store: EmbeddingStore, batch: EmbeddingBatch) -> None:
    store.write(
        batch, manifest_id="sha256:" + "1" * 64, library_id="lib", fingerprint={}, params={}
    )


def active_generation(directory) -> str:
    return (directory / POINTER_NAME).read_text(encoding="utf-8").strip()


# -- publish protocol --------------------------------------------------------
def test_successful_write_publishes_a_pointer_not_a_flat_layout(tmp_path):
    space = make_space()
    directory = tmp_path / "store"
    write(EmbeddingStore(directory), make_batch(space))

    assert (directory / POINTER_NAME).is_file()
    assert not (directory / "header.json").exists()
    pointer = active_generation(directory)
    assert pointer.startswith(f"{GENERATIONS_DIR}/gen-")
    assert (directory / pointer / "header.json").is_file()
    assert len(list((directory / GENERATIONS_DIR).iterdir())) == 1


def test_previous_generation_is_kept_for_rollback(tmp_path):
    space = make_space()
    directory = tmp_path / "store"
    store = EmbeddingStore(directory)
    write(store, make_batch(space, count=3))
    first = active_generation(directory)
    write(EmbeddingStore(directory), make_batch(space, count=2, seed=7))
    second = active_generation(directory)
    assert first != second
    generations = {path.name for path in (directory / GENERATIONS_DIR).iterdir()}
    # Live + the immediate predecessor; anything older is pruned.
    assert generations == {first.split("/")[-1], second.split("/")[-1]}


def test_failure_while_writing_vectors_keeps_the_old_generation(tmp_path, monkeypatch):
    space = make_space()
    directory = tmp_path / "store"
    write(EmbeddingStore(directory), make_batch(space, count=3))
    before = active_generation(directory)

    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(np, "save", boom)
    with pytest.raises(OSError):
        write(EmbeddingStore(directory), make_batch(space, count=2, seed=9))

    assert active_generation(directory) == before
    reloaded = EmbeddingStore(directory).load()
    assert reloaded.vectors.shape[0] == 3
    # The failed attempt did not leave a half-written generation behind.
    assert len(list((directory / GENERATIONS_DIR).iterdir())) == 1


def test_failure_during_publish_keeps_the_old_generation(tmp_path, monkeypatch):
    space = make_space()
    directory = tmp_path / "store"
    write(EmbeddingStore(directory), make_batch(space, count=3))
    before = active_generation(directory)

    def boom(_self, _staged):
        raise PermissionError("pointer is locked")

    monkeypatch.setattr(EmbeddingStore, "_publish", boom)
    with pytest.raises(PermissionError):
        write(EmbeddingStore(directory), make_batch(space, count=2, seed=11))

    assert active_generation(directory) == before
    assert EmbeddingStore(directory).load().vectors.shape[0] == 3
    assert len(list((directory / GENERATIONS_DIR).iterdir())) == 1


def test_crash_after_staging_before_publish_recovers_the_old_data(tmp_path):
    """Simulate a process dying between staging and the pointer swap."""
    space = make_space()
    directory = tmp_path / "store"
    write(EmbeddingStore(directory), make_batch(space, count=3))
    before = active_generation(directory)

    store = EmbeddingStore(directory)
    store._stage(  # noqa: SLF001 - deliberately stopping before publish
        make_batch(space, count=2, seed=13),
        manifest_id="m",
        library_id="l",
        fingerprint={},
        params={},
        extra=None,
    )
    # Nothing was published, so readers still see the committed generation.
    assert active_generation(directory) == before
    assert EmbeddingStore(directory).load().vectors.shape[0] == 3

    # A later successful write prunes the orphaned staging directory.
    write(EmbeddingStore(directory), make_batch(space, count=2, seed=17))
    assert len(list((directory / GENERATIONS_DIR).iterdir())) == 2


def test_pointer_to_a_missing_generation_is_an_error_not_an_empty_store(tmp_path):
    space = make_space()
    directory = tmp_path / "store"
    write(EmbeddingStore(directory), make_batch(space))
    shutil.rmtree(directory / active_generation(directory))
    with pytest.raises(ArtifactCorruptError, match="missing generation"):
        EmbeddingStore(directory).load()


def test_pointer_escaping_the_store_is_refused(tmp_path):
    directory = tmp_path / "store"
    directory.mkdir()
    (directory / POINTER_NAME).write_text("../elsewhere\n", encoding="utf-8")
    with pytest.raises(ArtifactCorruptError, match="escapes"):
        EmbeddingStore(directory).load()


# -- legacy layout -----------------------------------------------------------
def _write_legacy_flat(directory, space, count: int = 3) -> None:
    """Write an embeddings-v1 store the way the previous writer did."""
    directory.mkdir(parents=True, exist_ok=True)
    batch = make_batch(space, count=count)
    np.save(directory / "vectors.npy", batch.vectors)
    rows_text = "".join(json.dumps(row.as_dict()) + "\n" for row in batch.rows)
    (directory / "rows.jsonl").write_text(rows_text, encoding="utf-8")
    header = {
        "schema_version": "embeddings-v1",
        "space": batch.space.as_dict(),
        "manifest_id": "sha256:" + "1" * 64,
        "library_id": "lib",
        "model": {},
        "params": {},
        "count": int(batch.vectors.shape[0]),
        "dimension": int(batch.vectors.shape[1]),
        "asset_count": len(batch.revisions),
        "asset_revisions": batch.revisions,
        "checksum": _checksum_v1(batch.vectors, batch.rows),
    }
    (directory / "header.json").write_text(json.dumps(header), encoding="utf-8")


def test_legacy_flat_artifact_is_readable_then_migrated(tmp_path):
    space = make_space()
    directory = tmp_path / "store"
    _write_legacy_flat(directory, space, count=3)

    legacy = EmbeddingStore(directory).load()
    assert legacy.schema_version == "embeddings-v1"
    assert legacy.hashed_ok if hasattr(legacy, "hashed_ok") else True
    assert legacy.vectors.shape[0] == 3

    # A v1 artifact has no input fingerprints, so nothing in it is reusable.
    class _Asset:
        asset_id = "asset-0"
        content_revision = "rev-0"

    assert legacy.reusable_assets(space, [_Asset()], {"asset-0": _fingerprint("asset-0")}) == set()

    write(EmbeddingStore(directory), make_batch(space, count=2, seed=3))
    assert (directory / POINTER_NAME).is_file()
    assert not (directory / "header.json").exists()
    migrated = EmbeddingStore(directory).load()
    assert migrated.schema_version == SCHEMA_VERSION
    assert migrated.vectors.shape[0] == 2


# -- load validation ---------------------------------------------------------
def _stored(tmp_path, space=None, count: int = 3):
    space = space or make_space()
    directory = tmp_path / "store"
    write(EmbeddingStore(directory), make_batch(space, count=count))
    return directory, EmbeddingStore(directory).load()


def _rewrite_header_checksum(directory, store, mutate) -> None:
    """Edit the header and re-sign it, to prove non-checksum guards also fire."""
    from audiosearch.indexing.store import _checksum_v2

    path = directory / active_generation(directory) / "header.json"
    header = json.loads(path.read_text(encoding="utf-8"))
    mutate(header)
    header["checksum"] = _checksum_v2(header, store.vectors, store.rows)
    path.write_text(json.dumps(header), encoding="utf-8")


def test_tampered_vectors_are_detected(tmp_path):
    directory, store = _stored(tmp_path)
    path = directory / active_generation(directory) / "vectors.npy"
    # Swapping rows keeps every norm at 1, so only the checksum can catch it.
    tampered = store.vectors.copy()
    tampered[[0, 1]] = tampered[[1, 0]]
    np.save(path, tampered)
    with pytest.raises(ArtifactCorruptError, match="checksum"):
        EmbeddingStore(directory).load()


def test_tampered_rows_are_detected(tmp_path):
    directory, store = _stored(tmp_path)
    path = directory / active_generation(directory) / "rows.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    # Still a valid coverage value; only the checksum can notice the edit.
    first["coverage"] = 0.5
    lines[0] = json.dumps(first)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ArtifactCorruptError, match="checksum"):
        EmbeddingStore(directory).load()


def test_tampered_header_is_detected(tmp_path):
    directory, store = _stored(tmp_path)
    path = directory / active_generation(directory) / "header.json"
    header = json.loads(path.read_text(encoding="utf-8"))
    header["asset_revisions"]["asset-0"] = "forged-revision"
    path.write_text(json.dumps(header), encoding="utf-8")
    with pytest.raises(ArtifactCorruptError, match="checksum"):
        EmbeddingStore(directory).load()


def test_checksum_covers_header_fields_not_just_the_data(tmp_path):
    """Editing a header field is caught by the checksum that now covers the header."""
    directory, store = _stored(tmp_path)
    path = directory / active_generation(directory) / "header.json"
    header = json.loads(path.read_text(encoding="utf-8"))
    header["space"]["dimension"] = 8
    path.write_text(json.dumps(header), encoding="utf-8")
    with pytest.raises((ArtifactCorruptError, ArtifactIncompatibleError)):
        EmbeddingStore(directory).load()


def test_forged_space_id_is_detected(tmp_path):
    directory, store = _stored(tmp_path)

    def forge(header):
        header["space"]["space_id"] = "test-model/audio/v1"

    _rewrite_header_checksum(directory, store, forge)
    # Even with a matching checksum the derived id disagrees with its components.
    with pytest.raises(ArtifactCorruptError, match="space_id"):
        EmbeddingStore(directory).load()


def test_resigned_but_inconsistent_asset_count_is_detected(tmp_path):
    directory, store = _stored(tmp_path)

    def forge(header):
        header["count"] = 2

    _rewrite_header_checksum(directory, store, forge)
    with pytest.raises(ArtifactCorruptError, match="count"):
        EmbeddingStore(directory).load()


def test_non_finite_vectors_are_refused(tmp_path):
    directory, store = _stored(tmp_path)
    path = directory / active_generation(directory) / "vectors.npy"
    broken = store.vectors.copy()
    broken[1, 1] = np.nan
    np.save(path, broken)
    with pytest.raises(ArtifactCorruptError):
        EmbeddingStore(directory).load()


def test_unnormalized_vectors_are_refused(tmp_path):
    directory, store = _stored(tmp_path)
    path = directory / active_generation(directory) / "vectors.npy"
    scaled = (store.vectors * 2.0).astype(np.float32)
    np.save(path, scaled)
    with pytest.raises(ArtifactCorruptError, match="norm"):
        EmbeddingStore(directory).load()


def test_wrong_dimension_is_refused(tmp_path):
    directory, store = _stored(tmp_path)
    path = directory / active_generation(directory) / "vectors.npy"
    wide = np.zeros((store.vectors.shape[0], store.vectors.shape[1] + 2), dtype=np.float32)
    np.save(path, wide)
    with pytest.raises((ArtifactCorruptError, ArtifactIncompatibleError)):
        EmbeddingStore(directory).load()


def test_unknown_schema_is_incompatible(tmp_path):
    directory, store = _stored(tmp_path)
    path = directory / active_generation(directory) / "header.json"
    header = json.loads(path.read_text(encoding="utf-8"))
    header["schema_version"] = "embeddings-v99"
    path.write_text(json.dumps(header), encoding="utf-8")
    with pytest.raises(ArtifactIncompatibleError, match="schema"):
        EmbeddingStore(directory).load()


def test_row_count_mismatch_is_detected(tmp_path):
    directory, store = _stored(tmp_path)
    path = directory / active_generation(directory) / "rows.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ArtifactCorruptError, match="rows"):
        EmbeddingStore(directory).load()


def test_row_range_must_be_non_empty(tmp_path):
    directory, store = _stored(tmp_path)
    path = directory / active_generation(directory) / "rows.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["end_seconds"] = first["start_seconds"]
    lines[0] = json.dumps(first)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ArtifactCorruptError):
        EmbeddingStore(directory).load()


def test_empty_directory_loads_as_empty_not_as_error(tmp_path):
    store = EmbeddingStore(tmp_path / "nothing").load()
    assert store.is_empty
    assert store.generation is None
