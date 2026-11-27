"""Behavioral tests for Spectral Peaks analysis, coloring, v2 serialization, and v3 migration."""

from __future__ import annotations

import sqlite3
import wave
from pathlib import Path

import numpy as np

from audiosearch.library.peaks import (
    THUMBNAIL_POINTS,
    PeakData,
    analyze_spectral_frames,
    compute_file_peaks,
    compute_thumbnail_peaks,
    compute_thumbnail_with_spectral,
    decode_thumbnail_spectral,
    deserialize_peaks,
    encode_thumbnail_amplitudes,
    peaks_to_thumbnail,
    peaks_to_thumbnail_spectral,
    resample_spectral_columns,
    serialize_peaks,
    spectral_color_rgb,
    spectral_fft_size,
)
from audiosearch.library.schema import ensure_schema
from audiosearch.library.store import LibraryStore


def _write_pcm16_wav(path: Path, samples: np.ndarray, *, rate: int = 48000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    channels = arr.shape[1]
    pcm = np.clip(np.rint(arr * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())


def test_spectral_analysis_frequency_tonality_and_dc_rejection():
    sr = 48000
    n_fft = spectral_fft_size(sr)
    assert n_fft == 1024
    t = np.arange(n_fft, dtype=np.float32) / sr

    sine_low = 0.6 * np.sin(2.0 * np.pi * 80.0 * t) + 0.35  # include strong DC offset
    sine_mid = 0.6 * np.sin(2.0 * np.pi * 1000.0 * t)
    sine_high = 0.6 * np.sin(2.0 * np.pi * 8000.0 * t)

    rng = np.random.default_rng(7)
    white = 0.3 * rng.standard_normal(n_fft).astype(np.float32)
    wfft = np.fft.rfft(white)
    wfft[1:] /= np.sqrt(np.arange(1, len(wfft), dtype=np.float32))
    pink = np.fft.irfft(wfft, n_fft).astype(np.float32)
    click = np.zeros(n_fft, dtype=np.float32)
    click[n_fft // 2] = 0.9
    silence = np.zeros(n_fft, dtype=np.float32)

    frames = np.stack([sine_low, sine_mid, sine_high, white, pink, click, silence], axis=0)
    tags = analyze_spectral_frames(frames, sr)
    assert tags.shape == (7, 2)
    assert tags.dtype == np.uint8

    f_low, ton_low = int(tags[0, 0]), int(tags[0, 1])
    f_mid, ton_mid = int(tags[1, 0]), int(tags[1, 1])
    f_high, ton_high = int(tags[2, 0]), int(tags[2, 1])

    assert f_low < f_mid < f_high
    assert f_low < 65
    assert 100 < f_mid < 165
    assert f_high > 200
    assert ton_low > 180
    assert ton_mid > 180
    assert ton_high > 180

    # Broadband noise and single-sample click must register as low-tonality noise.
    assert int(tags[3, 1]) < 40
    assert int(tags[4, 1]) < 40
    assert int(tags[5, 1]) < 40

    # Silence yields (0, 0) without NaN/garbage.
    assert tuple(int(x) for x in tags[6]) == (0, 0)
    assert analyze_spectral_frames(np.zeros((0, n_fft), dtype=np.float32), sr).shape == (0, 2)


def test_spectral_color_rgb_mapping_and_played_dimming():
    # Low ~80 Hz pure tone -> red dominant
    r_low, g_low, b_low = spectral_color_rgb(38, 255, played=False)
    assert r_low > g_low + 40 and r_low > b_low + 120

    # Mid ~1.2 kHz pure tone -> green dominant
    r_mid, g_mid, b_mid = spectral_color_rgb(142, 255, played=False)
    assert g_mid > r_mid and g_mid > b_mid

    # High ~9 kHz pure tone -> blue/violet dominant
    r_hi, g_hi, b_hi = spectral_color_rgb(228, 255, played=False)
    assert b_hi > r_hi and b_hi >= g_hi

    # Pure noise -> neutral readable gray on dark background
    r_n, g_n, b_n = spectral_color_rgb(180, 0, played=False)
    assert max(r_n, g_n, b_n) - min(r_n, g_n, b_n) <= 25
    assert (r_n + g_n + b_n) / 3.0 > 120.0

    # Played dimming reduces brightness without collapsing to black
    rp, gp, bp = spectral_color_rgb(38, 255, played=True)
    assert rp < r_low and rp >= 28 and gp >= 28 and bp >= 28


def test_streaming_chunk_boundary_long_bucket_and_stride_thumbnail(tmp_path):
    sr = 48000
    n = 96000  # 2.0s > 65536 chunk boundary
    t = np.arange(n, dtype=np.float32) / sr
    # Left channel: low tone first half, high tone second half; right channel: white noise burst
    ch0 = np.where(
        t < 1.0,
        0.5 * np.sin(2.0 * np.pi * 100.0 * t),
        0.5 * np.sin(2.0 * np.pi * 6000.0 * t),
    )
    rng = np.random.default_rng(11)
    ch1 = 0.25 * rng.standard_normal(n).astype(np.float32)
    stereo = np.stack([ch0, ch1], axis=1)

    wav_path = tmp_path / "cross_chunk.wav"
    _write_pcm16_wav(wav_path, stereo, rate=sr)

    # 1. Dense streaming mode (points=2048 -> ~47 frames/bucket)
    data = compute_file_peaks(wav_path, points=2048, mtime_ns=123)
    assert data.spectral is not None
    assert data.spectral.shape == (2, 2048, 2)
    assert int(data.spectral[0, 200, 0]) < 70
    assert int(data.spectral[0, 1800, 0]) > 180
    assert int(data.spectral[0, 200, 1]) > 170
    assert int(data.spectral[1, :, 1].mean()) < 60

    # 2. Sparse-bucket peak-tracking mode (points=32 -> 3000 frames/bucket > n_fft)
    sparse = compute_file_peaks(wav_path, points=32)
    assert sparse.spectral is not None
    assert sparse.spectral.shape == (2, 32, 2)
    assert int(sparse.spectral[0, 2, 0]) < 70
    assert int(sparse.spectral[0, 28, 0]) > 180

    # 3. Thumbnail + spectral one-pass (both streaming and large-WAV stride path)
    thumb_stream, spec_stream = compute_thumbnail_with_spectral(wav_path)
    assert len(thumb_stream) == THUMBNAIL_POINTS
    assert len(spec_stream) == THUMBNAIL_POINTS * 2
    assert thumb_stream == compute_thumbnail_peaks(wav_path)

    thumb_stride, spec_stride = compute_thumbnail_with_spectral(
        wav_path, stride_threshold_bytes=1024
    )
    assert len(thumb_stride) == THUMBNAIL_POINTS
    assert len(spec_stride) == THUMBNAIL_POINTS * 2
    arr_stride = decode_thumbnail_spectral(spec_stride)
    assert arr_stride.shape == (THUMBNAIL_POINTS, 2)
    assert int(arr_stride[20, 0]) < int(arr_stride[480, 0])


def test_v1_v2_peak_serialization_and_spectral_resampling():
    rng = np.random.default_rng(3)
    peaks = rng.uniform(-0.8, 0.8, size=(2, 256, 2)).astype(np.float32)
    spectral = np.zeros((2, 256, 2), dtype=np.uint8)
    spectral[0, :, 0] = 40
    spectral[0, :, 1] = 240
    spectral[1, :, 0] = 210
    spectral[1, :, 1] = 190

    # v1 roundtrip
    v1_data = PeakData(
        duration_seconds=1.5,
        sample_rate_hz=48000,
        channel_count=2,
        points=256,
        peaks=peaks,
        spectral=None,
    )
    v1_blob = serialize_peaks(v1_data)
    assert not v1_blob.startswith(b"SPK2")
    v1_restored = deserialize_peaks(v1_blob)
    assert v1_restored.spectral is None
    assert np.allclose(v1_restored.peaks, peaks)

    # v2 roundtrip
    v2_data = PeakData(
        duration_seconds=1.5,
        sample_rate_hz=48000,
        channel_count=2,
        points=256,
        peaks=peaks,
        spectral=spectral,
    )
    v2_blob = serialize_peaks(v2_data)
    assert v2_blob.startswith(b"SPK2")
    v2_restored = deserialize_peaks(v2_blob)
    assert v2_restored.spectral is not None
    assert np.array_equal(v2_restored.spectral, spectral)

    # peaks_to_thumbnail_spectral & resample_spectral_columns pick peak tag, not mean hue
    env = np.array([0.05, 0.95, 0.05, 0.05], dtype=np.float32)
    spec = np.array([[220, 250], [20, 250], [220, 250], [220, 250]], dtype=np.uint8)
    down_env, down_spec = resample_spectral_columns(env, spec, 2)
    assert down_env.shape == (2,)
    assert down_spec.shape == (2, 2)
    assert float(down_env[0]) > 0.9
    assert int(down_spec[0, 0]) == 20  # loudest bucket wins, not averaged to ~120
    assert int(down_spec[1, 0]) == 220

    up_env, up_spec = resample_spectral_columns(env, spec, 8)
    assert up_env.shape == (8,) and up_spec.shape == (8, 2)

    spec_thumb = peaks_to_thumbnail_spectral(v2_data, points=128)
    assert len(spec_thumb) == 128 * 2
    assert decode_thumbnail_spectral(b"\x01\x02\x03").shape == (0, 2)


def test_store_v2_to_v3_migration_preserves_thumbs_and_upgrades(tmp_path):
    sfx = tmp_path / "sfx"
    sr = 48000
    t = np.arange(4800, dtype=np.float32) / sr
    wav_path = sfx / "tone.wav"
    _write_pcm16_wav(wav_path, 0.5 * np.sin(2.0 * np.pi * 440.0 * t), rate=sr)

    db_path = tmp_path / "legacy_v2.sqlite3"
    # Simulate an existing user_version=2 DB lacking spectral_blob column.
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE roots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL UNIQUE,
            label TEXT NOT NULL,
            is_online INTEGER NOT NULL DEFAULT 1,
            file_count INTEGER NOT NULL DEFAULT 0,
            last_scan_at REAL,
            scan_cursor TEXT,
            status TEXT NOT NULL DEFAULT 'idle'
        );
        CREATE TABLE folders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            root_id INTEGER NOT NULL REFERENCES roots(id) ON DELETE CASCADE,
            parent_id INTEGER REFERENCES folders(id) ON DELETE CASCADE,
            rel_path TEXT NOT NULL,
            name TEXT NOT NULL,
            file_count INTEGER NOT NULL DEFAULT 0,
            UNIQUE(root_id, rel_path)
        );
        CREATE TABLE files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            root_id INTEGER NOT NULL REFERENCES roots(id) ON DELETE CASCADE,
            folder_id INTEGER REFERENCES folders(id) ON DELETE SET NULL,
            rel_path TEXT NOT NULL,
            filename TEXT NOT NULL,
            ext TEXT NOT NULL,
            size_bytes INTEGER NOT NULL DEFAULT 0,
            mtime_ns INTEGER NOT NULL DEFAULT 0,
            duration_sec REAL NOT NULL DEFAULT 0.0,
            sample_rate INTEGER NOT NULL DEFAULT 0,
            channels INTEGER NOT NULL DEFAULT 0,
            bit_depth INTEGER NOT NULL DEFAULT 0,
            package TEXT NOT NULL DEFAULT '',
            description TEXT NOT NULL DEFAULT '',
            embedded_text TEXT NOT NULL DEFAULT '',
            bext_json TEXT NOT NULL DEFAULT '{}',
            ixml_json TEXT NOT NULL DEFAULT '{}',
            is_available INTEGER NOT NULL DEFAULT 1,
            indexed_at REAL NOT NULL DEFAULT 0.0,
            UNIQUE(root_id, rel_path)
        );
        CREATE TABLE waveform_thumbs (
            file_id INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
            mtime_ns INTEGER NOT NULL,
            thumb_blob BLOB NOT NULL
        );
        PRAGMA user_version = 2;
        """
    )
    conn.close()

    store = LibraryStore(db_path)
    try:
        assert int(store._conn().execute("PRAGMA user_version").fetchone()[0]) >= 3
        root = store.add_root(sfx)
        st = wav_path.stat()
        with store._conn():
            store._conn().execute(
                """
                INSERT INTO files(
                    root_id, folder_id, rel_path, filename, ext, size_bytes, mtime_ns,
                    duration_sec, sample_rate, channels, bit_depth, is_available
                ) VALUES (?, NULL, ?, ?, 'wav', ?, ?, 0.1, 48000, 1, 16, 1)
                """,
                (root.id, "tone.wav", "tone.wav", st.st_size, st.st_mtime_ns),
            )
        rec = store.query_page(offset=0, limit=10)[0]

        legacy_thumb = encode_thumbnail_amplitudes(
            np.full(THUMBNAIL_POINTS, 0.4, dtype=np.float32)
        )
        store.save_cached_thumbnails([(rec.id, rec.mtime_ns, legacy_thumb)])

        # Re-run ensure_schema: legacy v2 thumb must NOT be wiped, and spectral_thumb_blob is None.
        ensure_schema(store._conn())
        rows = store.query_page(offset=0, limit=10)
        assert rows[0].thumb_blob == legacy_thumb
        assert rows[0].spectral_thumb_blob is None
        assert store.get_cached_thumbnail_entry(rec.id, rec.mtime_ns) == (legacy_thumb, None)

        # Backfill via save_cached_peaks populates both waveform_cache v2 and spectral_blob.
        peak_data = compute_file_peaks(rec.full_path, points=2048, mtime_ns=rec.mtime_ns)
        store.save_cached_peaks(rec.id, rec.mtime_ns, peak_data, requested_points=2048)

        upgraded_rows = store.query_page(offset=0, limit=10)
        assert upgraded_rows[0].thumb_blob == peaks_to_thumbnail(peak_data)
        assert upgraded_rows[0].spectral_thumb_blob == peaks_to_thumbnail_spectral(peak_data)
        assert upgraded_rows[0].spectral_thumb_blob is not None
    finally:
        store.close()
