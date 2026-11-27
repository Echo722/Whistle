# ruff: noqa: E402  (the numpy import guard must run before project imports)
"""Reuse must track the *inputs*, not only the audio file version.

Two cases that previously reused stale vectors:

* renaming an asset (or editing its description/keywords/package) leaves the
  audio untouched, so ``content_revision`` is unchanged while the text vector is
  wrong;
* changing ``window_seconds``/``hop_seconds`` changes every audio window while
  the file is unchanged, and the old header recorded a fixed policy string.

Both now have to match an input fingerprint, and the audio header records the
policy that was actually used.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip(
    "numpy",
    reason="Agent B numeric tests need numpy; it is an optional extra, not a shared dev dependency",
)

from helpers import (
    StubClapEncoder,
    StubTextEncoder,
    make_asset,
    make_asset_for_file,
    make_catalog,
    make_space,
    write_wav,
)

from audiosearch.contracts.models import AssetRecord
from audiosearch.indexing.encode import (
    audio_input_fingerprint,
    encode_audio_segments,
    encode_text_assets,
    plan_asset_windows,
    text_input_fingerprint,
    window_policy,
)
from audiosearch.indexing.store import EmbeddingStore


def _text_run(catalog, store, space):
    encoder = StubTextEncoder(space)
    report = encode_text_assets(catalog, store, encoder, batch_size=2)
    return report, encoder


def test_identical_inputs_are_reused(tmp_path):
    space = make_space(tower="text")
    catalog = make_catalog([make_asset(0), make_asset(1)])
    store = EmbeddingStore(tmp_path / "text")

    first, _ = _text_run(catalog, store, space)
    assert (first.files_encoded, first.files_reused) == (2, 0)

    second, encoder = _text_run(catalog, store, space)
    assert (second.files_encoded, second.files_reused) == (0, 2)
    assert encoder.encode_calls == 0


def test_rename_without_audio_change_forces_reencode(tmp_path):
    space = make_space(tower="text")
    store = EmbeddingStore(tmp_path / "text")
    _text_run(make_catalog([make_asset(0)]), store, space)

    renamed = make_asset(0, name="renamed whoosh.wav")
    assert renamed.content_revision == make_asset(0).content_revision
    report, encoder = _text_run(make_catalog([renamed]), store, space)
    assert report.files_reused == 0
    assert report.files_encoded == 1
    assert encoder.encode_calls == 1


def test_description_change_forces_reencode(tmp_path):
    space = make_space(tower="text")
    store = EmbeddingStore(tmp_path / "text")
    base = make_asset(0)
    _text_run(make_catalog([base]), store, space)

    described = base.model_copy(
        update={
            "description": "synthetic whoosh",
            "description_source": "machine",
            "keywords": ["air"],
        }
    )
    assert described.content_revision == base.content_revision
    report, _ = _text_run(make_catalog([described]), store, space)
    assert report.files_reused == 0


def test_text_fingerprint_depends_on_every_encoded_field():
    base = make_asset(0)
    described = base.model_copy(
        update={"description": "something", "description_source": "machine"}
    )
    packaged = base.model_copy(update={"package": "another/package"})
    fingerprints = {
        text_input_fingerprint(base),
        text_input_fingerprint(described),
        text_input_fingerprint(packaged),
    }
    assert len(fingerprints) == 3


def test_text_fingerprint_is_stable_for_the_same_record():
    asset = make_asset(0)
    assert text_input_fingerprint(asset) == text_input_fingerprint(asset.model_copy())


def test_changed_window_policy_invalidates_audio_reuse(tmp_path):
    space = make_space(tower="audio")
    root = tmp_path / "library"
    path = write_wav(root / "asset-0.wav", 12.0)
    asset = make_asset_for_file(0, path, duration=12.0)
    catalog = make_catalog([asset], root=root)
    store = EmbeddingStore(tmp_path / "audio")

    encoder = StubClapEncoder(space)
    first = encode_audio_segments(catalog, store, encoder, window_seconds=10.0, hop_seconds=5.0)
    assert first.files_encoded == 1
    default_fp = audio_input_fingerprint(
        plan_asset_windows(asset, window_seconds=10.0, hop_seconds=5.0)
    )

    # Same files, different hop: the stored windows are no longer the right ones.
    second_encoder = StubClapEncoder(space)
    second = encode_audio_segments(
        catalog, store, second_encoder, window_seconds=10.0, hop_seconds=2.5
    )
    assert second.files_reused == 0
    assert second.files_encoded == 1
    new_fp = audio_input_fingerprint(
        plan_asset_windows(asset, window_seconds=10.0, hop_seconds=2.5)
    )
    assert new_fp != default_fp

    stored = EmbeddingStore(store.directory).load()
    assert stored.input_fingerprints[str(asset.asset_id)] == new_fp


def test_identical_window_policy_is_reused(tmp_path):
    space = make_space(tower="audio")
    root = tmp_path / "library"
    path = write_wav(root / "asset-0.wav", 12.0)
    asset = make_asset_for_file(0, path, duration=12.0)
    catalog = make_catalog([asset], root=root)
    store = EmbeddingStore(tmp_path / "audio")

    encode_audio_segments(
        catalog, store, StubClapEncoder(space), window_seconds=6.0, hop_seconds=3.0
    )
    again = encode_audio_segments(
        catalog, store, StubClapEncoder(space), window_seconds=6.0, hop_seconds=3.0
    )
    assert (again.files_encoded, again.files_reused) == (0, 1)


def test_header_records_the_window_policy_actually_used(tmp_path):
    space = make_space(tower="audio")
    root = tmp_path / "library"
    path = write_wav(root / "asset-0.wav", 8.0)
    asset = make_asset_for_file(0, path, duration=8.0)
    catalog = make_catalog([asset], root=root)
    store = EmbeddingStore(tmp_path / "audio")
    encode_audio_segments(
        catalog, store, StubClapEncoder(space), window_seconds=4.0, hop_seconds=2.0
    )

    params = EmbeddingStore(store.directory).load().params
    assert params["window_seconds"] == 4.0
    assert params["hop_seconds"] == 2.0
    assert params["window_policy"] == window_policy(4.0, 2.0) == "win4s-hop2s-onset0"


def test_window_policy_changes_with_parameters():
    assert window_policy(10.0, 5.0) != window_policy(10.0, 2.5)


def test_audio_fingerprint_follows_the_file_duration():
    short = make_asset(0, duration=5.0)
    long = make_asset(0, duration=30.0)
    short_fp = audio_input_fingerprint(plan_asset_windows(short, window_seconds=10, hop_seconds=5))
    long_fp = audio_input_fingerprint(plan_asset_windows(long, window_seconds=10, hop_seconds=5))
    assert short_fp != long_fp


def test_asset_without_duration_has_no_windows():
    assert plan_asset_windows(make_asset(0, duration=0.0), window_seconds=10, hop_seconds=5) == []


def test_reuse_does_not_require_a_loadable_model(tmp_path):
    """Reuse is decided from stored metadata, before any weights are touched."""
    space = make_space(tower="text")
    store = EmbeddingStore(tmp_path / "text")
    catalog = make_catalog([make_asset(0)])
    _text_run(catalog, store, space)

    stored = EmbeddingStore(store.directory).load()
    fingerprints = {str(asset.asset_id): text_input_fingerprint(asset) for asset in catalog.assets}
    assert stored.reusable_assets(space, catalog.assets, fingerprints) == {
        str(catalog.assets[0].asset_id)
    }
    # A different space never reuses, even with identical inputs.
    other = make_space(tower="text", version="v2")
    assert stored.reusable_assets(other, catalog.assets, fingerprints) == set()


def test_revisions_and_fingerprints_are_written_for_every_row(tmp_path):
    space = make_space(tower="text")
    store = EmbeddingStore(tmp_path / "text")
    catalog = make_catalog([make_asset(0), make_asset(1)])
    _text_run(catalog, store, space)

    stored = EmbeddingStore(store.directory).load()
    row_assets = {row.asset_id for row in stored.rows}
    assert set(stored.revisions) == row_assets
    assert set(stored.input_fingerprints) == row_assets


def test_missing_vectors_for_an_asset_are_reported_as_missing(tmp_path):
    space = make_space(tower="text")
    store = EmbeddingStore(tmp_path / "text")
    _text_run(make_catalog([make_asset(0)]), store, space)

    bigger = make_catalog([make_asset(0), make_asset(1)])
    stored = EmbeddingStore(store.directory).load()
    fingerprints = {str(asset.asset_id): text_input_fingerprint(asset) for asset in bigger.assets}
    report = stored.validity(space, bigger.assets, fingerprints)
    assert len(report.valid) == 1
    assert len(report.missing) == 1
    assert report.needs_rebuild


def test_vectors_are_not_reused_when_the_model_revision_changes(tmp_path):
    space = make_space(tower="text")
    store = EmbeddingStore(tmp_path / "text")
    _text_run(make_catalog([make_asset(0)]), store, space)

    bumped = make_space(tower="text", version="v2")
    encoder = StubTextEncoder(bumped)
    report = encode_text_assets(make_catalog([make_asset(0)]), store, encoder)
    assert report.files_reused == 0
    assert report.space_id != space.space_id


@pytest.mark.parametrize("window,hop", [(10.0, 5.0), (4.0, 2.0), (2.0, 1.0)])
def test_fingerprint_is_stable_across_planner_calls(window, hop):
    asset: AssetRecord = make_asset(3, duration=37.3)
    first = audio_input_fingerprint(
        plan_asset_windows(asset, window_seconds=window, hop_seconds=hop)
    )
    second = audio_input_fingerprint(
        plan_asset_windows(asset, window_seconds=window, hop_seconds=hop)
    )
    assert first == second
    assert isinstance(np.zeros(1), np.ndarray)
