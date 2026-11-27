# ruff: noqa: E402  (the numpy import guard must run before project imports)
"""Agent B tests for model-facing contracts that must hold without a GPU.

These cover the parts that decide *what* gets encoded and *how* a file is cut
into windows: template text, window planning, coverage accounting, and the
resampling that sits between the source file and CLAP's 48 kHz input. No model
weights are loaded here.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip(
    "numpy",
    reason="Agent B numeric tests need numpy; it is an optional extra, not a shared dev dependency",
)

from audiosearch.audioio.resample import resample
from audiosearch.contracts.models import AssetLocation, AssetRecord
from audiosearch.inference.clap_encoder import AudioWindow, coverage, plan_windows
from audiosearch.inference.spaces import EmbeddingSpace, SpaceMismatchError, l2_normalize
from audiosearch.inference.text_encoder import plan_asset_text


def make_asset(**overrides) -> AssetRecord:
    payload = {
        "asset_id": "11111111-1111-5111-8111-111111111111",
        "content_revision": "stat-v1:1:1",
        "name": "MACK Whoosh Synth Superfast.wav",
        "library_id": "lib",
        "package": "Magic - Bundle/boom35_5",
        "duration_seconds": 21.3,
        "sample_rate_hz": 96000,
        "channel_count": 2,
        "bit_depth": 24,
        "locations": [
            AssetLocation(
                location_id="22222222-2222-5222-8222-222222222222",
                root_id="root",
                relative_path="a/b.wav",
                availability="online",
                size_bytes=1,
                mtime_ns=1,
            )
        ],
    }
    payload.update(overrides)
    return AssetRecord(**payload)


def test_asset_text_uses_only_existing_fields():
    asset = make_asset()
    plan = plan_asset_text(asset)
    assert plan.text == "MACK Whoosh Synth Superfast.wav | Magic - Bundle/boom35_5"
    assert plan.fields_used == ("name", "package")
    assert plan.is_sparse is True


def test_asset_text_marks_described_assets_as_not_sparse():
    asset = make_asset(
        description="synthetic whoosh", description_source="machine", keywords=["air"]
    )
    plan = plan_asset_text(asset)
    assert "synthetic whoosh" in plan.text and "air" in plan.text
    assert plan.is_sparse is False


def test_short_file_is_one_window_covering_everything():
    windows = plan_windows(3.2, window_seconds=10.0, hop_seconds=5.0)
    assert windows == [AudioWindow(0.0, 3.2)]
    assert coverage(windows, 3.2) == pytest.approx(1.0)


def test_windows_are_anchored_at_zero_and_never_exceed_the_file():
    windows = plan_windows(21.34, window_seconds=10.0, hop_seconds=5.0)
    assert windows[0].start_seconds == pytest.approx(0.0)
    assert windows[-1].end_seconds == pytest.approx(21.34)
    assert all(window.end_seconds > window.start_seconds for window in windows)
    assert coverage(windows, 21.34) == pytest.approx(1.0)


def test_windows_overlap_by_the_hop():
    windows = plan_windows(30.0, window_seconds=10.0, hop_seconds=5.0)
    starts = [window.start_seconds for window in windows]
    assert starts == [0.0, 5.0, 10.0, 15.0, 20.0]
    assert windows[-1].end_seconds == pytest.approx(30.0)


def test_max_windows_reports_partial_coverage_instead_of_claiming_complete():
    windows = plan_windows(300.0, window_seconds=10.0, hop_seconds=5.0, max_windows=4)
    assert len(windows) == 4
    assert coverage(windows, 300.0) < 0.2


def test_coverage_merges_overlaps_without_double_counting():
    windows = [AudioWindow(0.0, 10.0), AudioWindow(5.0, 15.0)]
    assert coverage(windows, 20.0) == pytest.approx(0.75)


def test_coverage_of_no_windows_is_zero():
    assert coverage([], 10.0) == 0.0


def test_invalid_window_parameters_are_rejected():
    with pytest.raises(ValueError):
        plan_windows(0.0)
    with pytest.raises(ValueError):
        plan_windows(10.0, window_seconds=0.0)
    with pytest.raises(ValueError):
        plan_windows(10.0, window_seconds=10.0, hop_seconds=0.0)


def test_resample_preserves_a_sine_and_its_amplitude():
    src = 96000
    seconds = 1.0
    t = np.arange(int(src * seconds)) / src
    signal = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    out = resample(signal, src, 48000)
    assert out.shape[0] == 48000
    mid = out[1000:-1000]
    expected_rms = 0.5 / np.sqrt(2.0)
    assert float(np.sqrt((mid**2).mean())) == pytest.approx(expected_rms, rel=0.02)


def test_resample_keeps_channels_independent():
    src = 48000
    t = np.arange(src) / src
    left = (0.4 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    right = (0.7 * np.sin(2 * np.pi * 660.0 * t)).astype(np.float32)
    out = resample(np.stack([left, right], axis=1), src, 24000)
    assert out.shape == (24000, 2)
    assert float(np.sqrt((out[:, 0][500:-500] ** 2).mean())) < float(
        np.sqrt((out[:, 1][500:-500] ** 2).mean())
    )


def test_resample_identity_when_rates_match():
    signal = np.arange(100, dtype=np.float32)
    assert np.array_equal(resample(signal, 48000, 48000), signal)


def test_resample_rejects_bad_rates():
    with pytest.raises(ValueError):
        resample(np.zeros(10, dtype=np.float32), 0, 48000)


def test_normalize_raises_on_zero_vector_instead_of_returning_nan():
    with pytest.raises(ValueError, match="zero"):
        l2_normalize(np.zeros((2, 4), dtype=np.float32))


def test_space_id_changes_with_revision_tower_or_preprocessing():
    base = EmbeddingSpace("m", "a" * 40, "text", "v1", 8, True)
    same = EmbeddingSpace("m", "a" * 40, "text", "v1", 8, True)
    assert base.space_id == same.space_id
    assert base.space_id != EmbeddingSpace("m", "b" * 40, "text", "v1", 8, True).space_id
    assert base.space_id != EmbeddingSpace("m", "a" * 40, "audio", "v1", 8, True).space_id
    assert base.space_id != EmbeddingSpace("m", "a" * 40, "text", "v2", 8, True).space_id
    with pytest.raises(SpaceMismatchError):
        base.require_same(EmbeddingSpace("m", "a" * 40, "audio", "v1", 8, True))
