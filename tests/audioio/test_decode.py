# ruff: noqa: E402  (the numpy import guard must run before project imports)
"""Behavioural checks for bounded decoding, source-range mapping and true peaks.

These use synthetic WAVE files written during the test run; no real library
sample is read and nothing is written outside ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip(
    "numpy",
    reason="Agent B numeric tests need numpy; it is an optional extra, not a shared dev dependency",
)

from wav_fixtures import tone, write_extensible_wav, write_wav

from audiosearch.audioio import (
    AssetChangedError,
    AudioError,
    ManifestCatalog,
    SourceRangeError,
    WavWaveformProvider,
    compute_peaks,
    open_wav,
    parse_wav,
    probe,
)
from audiosearch.contracts.models import (
    AssetLocation,
    AssetRecord,
    AssetRef,
    SampleManifest,
    SourceRange,
    WaveformData,
    WaveformRequest,
)


def test_probe_reports_original_geometry(tmp_path):
    path = write_wav(tmp_path / "stereo24.wav", np.zeros((4800, 2)), rate=96000, bit_depth=24)
    with open_wav(path) as reader:
        info = reader.info
        assert info.sample_rate_hz == 96000
        assert info.channel_count == 2
        assert info.bit_depth == 24
        assert info.frames == 4800
        assert info.sample_format == "pcm_s24"
        assert info.duration_seconds == pytest.approx(0.05)


def test_probe_returns_header_and_leaves_no_open_handle(tmp_path):
    path = write_wav(tmp_path / "probe.wav", np.zeros((960, 2)), rate=48000, bit_depth=24)
    info = probe(path)
    assert info.frames == 960
    assert info.channel_count == 2

    # The probe is header-only and must close what it opened: on Windows an
    # open handle blocks deletion, so a successful unlink proves the release.
    path.unlink()
    assert not path.exists()


def test_probe_reports_missing_file_as_offline(tmp_path):
    from audiosearch.audioio import AssetOfflineError

    with pytest.raises(AssetOfflineError):
        probe(tmp_path / "absent.wav")


def test_extra_chunks_after_data_do_not_break_the_header_scan(tmp_path):
    path = write_wav(tmp_path / "tagged.wav", np.zeros((96, 1)), rate=48000)
    # Append a LIST chunk after the audio payload, as real BWF/INFO files do.
    raw = path.read_bytes()
    body = b"INFOISFT" + b"\x08\x00\x00\x00" + b"test\x00\x00\x00\x00"
    extra = b"LIST" + (len(body)).to_bytes(4, "little") + body
    path.write_bytes(raw + extra)
    with open_wav(path) as reader:
        assert reader.info.frames == 96


def test_missing_format_is_reported_not_guessed(tmp_path):
    path = tmp_path / "broken.wav"
    path.write_bytes(
        b"RIFF"
        + (16).to_bytes(4, "little")
        + b"WAVE"
        + b"junk"
        + (4).to_bytes(4, "little")
        + b"abcd"
    )
    with pytest.raises(AudioError):
        with path.open("rb") as stream:
            parse_wav(stream)


def test_extensible_float_is_decoded(tmp_path):
    samples = tone(4800, 48000, 440.0, amplitude=0.25)[:, None]
    path = write_extensible_wav(tmp_path / "ext.wav", samples)
    with open_wav(path) as reader:
        assert reader.info.sample_format == "float32"
        decoded, _ = reader.read_range(0.0, 0.05)
    assert decoded.shape == (2400, 1)
    assert np.allclose(decoded[:, 0], samples[:2400, 0], atol=1e-6)


def test_range_maps_back_to_the_same_source_seconds(tmp_path):
    rate = 8000
    samples = tone(rate * 2, rate, 100.0, amplitude=0.5)[:, None]
    path = write_wav(tmp_path / "tone.wav", samples, rate=rate)
    with open_wav(path) as reader:
        trimmed, first_frame = reader.read_range(0.5, 1.0)
        assert first_frame == rate // 2
        assert trimmed.shape[0] == rate // 2
        # The decoded window must be the same waveform as the source slice.
        assert np.allclose(trimmed[:, 0], samples[rate // 2 : rate, 0], atol=1e-4)
        assert reader.frame_to_seconds(first_frame) == pytest.approx(0.5)


def test_range_beyond_the_payload_is_rejected_not_silently_padded(tmp_path):
    path = write_wav(tmp_path / "short.wav", np.zeros((100, 1)), rate=48000)
    with open_wav(path) as reader:
        with pytest.raises(SourceRangeError):
            reader.frame_bounds(10.0, 10.5)


def test_peaks_are_true_min_max_per_bucket(tmp_path):
    rate = 48000
    # Ramp from -0.9 to +0.9 so every bucket's extremes are known analytically.
    ramp = np.linspace(-0.9, 0.9, rate, dtype=np.float32)[:, None]
    path = write_wav(tmp_path / "ramp.wav", ramp, rate=rate)
    with open_wav(path) as reader:
        result = compute_peaks(reader, start_seconds=0.0, end_seconds=1.0, points=10)
    assert result.frames == rate
    expected_min = np.linspace(-0.9, 0.9, rate, dtype=np.float32).reshape(10, -1).min(axis=1)
    expected_max = np.linspace(-0.9, 0.9, rate, dtype=np.float32).reshape(10, -1).max(axis=1)
    assert np.allclose(result.minima[:, 0], expected_min, atol=2e-4)
    assert np.allclose(result.maxima[:, 0], expected_max, atol=2e-4)


def test_peaks_keep_original_channel_order_and_are_bounded(tmp_path):
    rate = 48000
    left = tone(rate, rate, 220.0, amplitude=0.2)
    right = tone(rate, rate, 440.0, amplitude=1.4)  # deliberately out of range
    path = write_wav(tmp_path / "lr.wav", np.stack([left, right], axis=1), rate=rate, bit_depth=32)
    with open_wav(path) as reader:
        result = compute_peaks(reader, start_seconds=0.0, end_seconds=1.0, points=32)
        selected = compute_peaks(
            reader, start_seconds=0.0, end_seconds=1.0, points=32, channels=[1]
        )
    assert result.channel_indices == [0, 1]
    assert result.maxima[:, 1].max() == pytest.approx(1.0)
    assert result.maxima[:, 1].max() > result.maxima[:, 0].max()
    assert result.clamped_frames > 0
    assert selected.channel_indices == [1]


def test_peaks_for_a_window_only_use_that_window(tmp_path):
    rate = 8000
    silence = np.zeros((rate, 1), dtype=np.float32)
    loud = tone(rate, rate, 300.0, amplitude=0.8)[:, None]
    path = write_wav(tmp_path / "two.wav", np.concatenate([silence, loud]), rate=rate)
    with open_wav(path) as reader:
        quiet = compute_peaks(reader, start_seconds=0.0, end_seconds=1.0, points=16)
        noisy = compute_peaks(reader, start_seconds=1.0, end_seconds=2.0, points=16)
    assert np.allclose(quiet.maxima, 0.0)
    assert noisy.maxima.max() > 0.7


def _manifest_for(tmp_path, samples, *, rate=48000, bit_depth=24) -> tuple[ManifestCatalog, Path]:
    root = tmp_path / "library"
    root.mkdir()
    path = write_wav(root / "one.wav", samples, rate=rate, bit_depth=bit_depth)
    stat = path.stat()
    record = AssetRecord(
        asset_id="11111111-1111-5111-8111-111111111111",
        content_revision=f"stat-v1:{stat.st_size}:{stat.st_mtime_ns}",
        name="one.wav",
        library_id="test-library",
        locations=[
            AssetLocation(
                location_id="22222222-2222-5222-8222-222222222222",
                root_id="test-root",
                relative_path="one.wav",
                availability="online",
                size_bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
            )
        ],
        duration_seconds=samples.shape[0] / rate,
        sample_rate_hz=rate,
        channel_count=samples.shape[1],
        bit_depth=bit_depth,
    )
    manifest = SampleManifest(
        manifest_id="sha256:" + "0" * 64,
        generated_at="2026-11-10T00:00:00Z",
        selection_method="sha256-relative-path-v1",
        selection_limit=8,
        library_id="test-library",
        root_id="test-root",
        assets=[record],
    )
    return ManifestCatalog(manifest, {"test-root": root}), path


def test_provider_returns_contract_waveform_for_real_decode(tmp_path):
    rate = 48000
    signal = tone(rate, rate, 500.0, amplitude=0.6)[:, None]
    catalog, _ = _manifest_for(tmp_path, signal, rate=rate)
    provider = WavWaveformProvider(catalog)
    asset = catalog.assets[0]
    request = WaveformRequest(
        asset=AssetRef(asset_id=asset.asset_id, content_revision=asset.content_revision),
        source_range=SourceRange(start_seconds=0.0, end_seconds=1.0),
        points=32,
    )
    data = provider.waveform(request)
    assert isinstance(data, WaveformData)
    assert data.data_kind == "local"
    assert data.status == "ready"
    assert data.channel_indices == [0]
    assert len(data.peaks) == 1 and len(data.peaks[0]) == 32
    assert max(peak.maximum for peak in data.peaks[0]) > 0.5
    assert all(peak.minimum <= peak.maximum for peak in data.peaks[0])


def test_provider_rejects_stale_content_revision(tmp_path):
    catalog, _ = _manifest_for(tmp_path, np.zeros((480, 1)))
    asset = catalog.assets[0]
    provider = WavWaveformProvider(catalog)
    with pytest.raises(AssetChangedError):
        provider.waveform(
            WaveformRequest(
                asset=AssetRef(asset_id=asset.asset_id, content_revision="stat-v1:0:0"),
                source_range=SourceRange(start_seconds=0.0, end_seconds=0.01),
                points=16,
            )
        )


def test_long_file_is_streamed_in_bounded_blocks(tmp_path, monkeypatch):
    rate = 8000
    seconds = 60
    path = write_wav(tmp_path / "long.wav", np.zeros((rate * seconds, 2)), rate=rate)
    with open_wav(path) as reader:
        sizes = [
            block.shape[0] for block in reader.iter_frames(0, rate * seconds, block_frames=1000)
        ]
    assert max(sizes) <= 1000
    assert sum(sizes) == rate * seconds
    # Blocks stay small and independent of the total file length.
    assert len(sizes) == (rate * seconds) // 1000


def test_short_read_is_a_decode_error_not_a_silent_pad(tmp_path):
    path = write_wav(tmp_path / "trunc.wav", np.zeros((4800, 1)), rate=48000)
    raw = path.read_bytes()
    path.write_bytes(raw[: len(raw) // 2])
    with open_wav(path) as reader:
        with pytest.raises(AudioError):
            reader.read_frames(0, reader.info.frames)
