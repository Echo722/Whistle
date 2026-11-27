"""Behavioral tests for thumbnail peak computation, envelope shaping, and SQLite caching."""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import numpy as np

from audiosearch.library.peaks import (
    THUMBNAIL_POINTS,
    PeakData,
    compute_thumbnail_peaks,
    encode_thumbnail_amplitudes,
    peaks_to_thumbnail,
    shape_thumbnail_envelope,
    thumbnail_width_fraction,
)
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


def _write_pcm24_mono_wav(path: Path, samples: np.ndarray, *, rate: int = 48000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 0.999999)
    ints = np.rint(arr * 8388607.0).astype("<i4")
    raw_u8 = ints.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    data_size = len(raw_u8)
    fmt_body = struct.pack("<HHIIHH", 1, 1, rate, rate * 3, 3, 24)
    riff_size = 4 + (8 + len(fmt_body)) + (8 + data_size)
    header = (
        b"RIFF"
        + struct.pack("<I", riff_size)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt_body))
        + fmt_body
        + b"data"
        + struct.pack("<I", data_size)
    )
    path.write_bytes(header + raw_u8)


def test_thumbnail_peaks_short_stereo_24bit_and_silence(tmp_path):
    # Stereo 16-bit: first half loud on right channel, second half quiet on left channel.
    n = 4800
    stereo = np.zeros((n, 2), dtype=np.float32)
    stereo[: n // 2, 1] = 0.8
    stereo[n // 2 :, 0] = 0.1
    stereo_path = tmp_path / "stereo.wav"
    _write_pcm16_wav(stereo_path, stereo)

    blob = compute_thumbnail_peaks(stereo_path)
    assert len(blob) == THUMBNAIL_POINTS
    env = shape_thumbnail_envelope(blob)
    assert env.shape == (THUMBNAIL_POINTS,)
    assert float(env[: THUMBNAIL_POINTS // 4].mean()) > float(env[(THUMBNAIL_POINTS * 3) // 4 :].mean()) * 2.0

    # 24-bit mono wav
    mono24 = np.zeros(4800, dtype=np.float32)
    mono24[1200:2400] = -0.6
    wav24_path = tmp_path / "mono24.wav"
    _write_pcm24_mono_wav(wav24_path, mono24)
    blob24 = compute_thumbnail_peaks(wav24_path)
    assert len(blob24) == THUMBNAIL_POINTS
    env24 = shape_thumbnail_envelope(blob24)
    assert 0.55 < float(env24[THUMBNAIL_POINTS // 4 : THUMBNAIL_POINTS // 2].max()) < 0.65
    assert float(env24[: THUMBNAIL_POINTS // 8].max()) == 0.0

    # Pure digital silence stays flat zero
    silence_path = tmp_path / "silence.wav"
    _write_pcm16_wav(silence_path, np.zeros(2400, dtype=np.float32))
    sil_blob = compute_thumbnail_peaks(silence_path)
    assert len(sil_blob) == THUMBNAIL_POINTS
    assert np.all(shape_thumbnail_envelope(sil_blob) == 0.0)


def test_thumbnail_sparse_stride_for_large_wav(tmp_path):
    n = 48000
    samples = np.full(n, 0.02, dtype=np.float32)
    samples[22000:26000] = 0.9
    wav_path = tmp_path / "long_like.wav"
    _write_pcm16_wav(wav_path, samples)

    # Force sparse stride path via small byte threshold
    blob = compute_thumbnail_peaks(wav_path, stride_threshold_bytes=1024)
    assert len(blob) == THUMBNAIL_POINTS
    env = shape_thumbnail_envelope(blob)
    lo = (THUMBNAIL_POINTS * 45) // 100
    hi = (THUMBNAIL_POINTS * 55) // 100
    assert float(env[lo:hi].max()) > float(env[: THUMBNAIL_POINTS // 6].max()) * 2.0


def test_peaks_to_thumbnail_and_envelope_shaping():
    arr = np.zeros((2, 2048, 2), dtype=np.float32)
    arr[1, 1024, 0] = -0.95
    data = PeakData(
        duration_seconds=2.0,
        sample_rate_hz=48000,
        channel_count=2,
        points=2048,
        peaks=arr,
    )
    blob = peaks_to_thumbnail(data)
    assert len(blob) == THUMBNAIL_POINTS
    env = shape_thumbnail_envelope(blob)
    assert float(env[THUMBNAIL_POINTS // 2]) > 0.9
    assert float(env[0]) == 0.0

    # True linear absolute level: silence=0, -6dBFS~0.5, -24dBFS~0.063, 0dBFS~1.0
    assert np.all(shape_thumbnail_envelope(b"\x00" * 32) == 0.0)
    minus_6_db = encode_thumbnail_amplitudes(np.full(32, 0.5, dtype=np.float32))
    minus_24_db = encode_thumbnail_amplitudes(np.full(32, 10 ** (-24 / 20), dtype=np.float32))
    full_scale = encode_thumbnail_amplitudes(np.full(32, 1.0, dtype=np.float32))

    env_6 = shape_thumbnail_envelope(minus_6_db)
    env_24 = shape_thumbnail_envelope(minus_24_db)
    env_full = shape_thumbnail_envelope(full_scale)
    assert 0.47 < float(env_6[0]) < 0.53
    assert 0.05 < float(env_24[0]) < 0.08
    assert float(env_full[0]) > 0.99


def test_thumbnail_width_fraction_log_scale():
    assert thumbnail_width_fraction(0.0) == 1.0
    assert thumbnail_width_fraction(-1.0) == 1.0
    assert thumbnail_width_fraction(0.05) == 0.20
    assert thumbnail_width_fraction(0.08) == 0.20
    assert thumbnail_width_fraction(15.0) == 1.0
    assert thumbnail_width_fraction(120.0) == 1.0

    f_01 = thumbnail_width_fraction(0.1)
    f_05 = thumbnail_width_fraction(0.5)
    f_15 = thumbnail_width_fraction(1.5)
    f_50 = thumbnail_width_fraction(5.0)
    assert 0.20 < f_01 < f_05 < f_15 < f_50 < 1.0


def test_store_waveform_thumbs_lifecycle_and_peak_backfill(tmp_path):
    sfx = tmp_path / "sfx"
    p_one = sfx / "one.wav"
    p_two = sfx / "two.wav"
    _write_pcm16_wav(p_one, np.linspace(-0.5, 0.5, 4800, dtype=np.float32))
    _write_pcm16_wav(p_two, np.linspace(-0.2, 0.2, 4800, dtype=np.float32))

    db_path = tmp_path / "thumbs.sqlite3"
    store = LibraryStore(db_path)
    try:
        root = store.add_root(sfx)
        conn = store._conn()
        with conn:
            for name, p in [("one.wav", p_one), ("two.wav", p_two)]:
                st = p.stat()
                conn.execute(
                    """
                    INSERT INTO files(
                        root_id, folder_id, rel_path, filename, ext, size_bytes, mtime_ns,
                        duration_sec, sample_rate, channels, bit_depth, is_available
                    ) VALUES (?, NULL, ?, ?, 'wav', ?, ?, 0.1, 48000, 1, 16, 1)
                    """,
                    (root.id, name, name, st.st_size, st.st_mtime_ns),
                )
        rows = store.query_page(offset=0, limit=10)
        assert len(rows) == 2
        assert all(r.thumb_blob is None for r in rows)

        r0, r1 = rows[0], rows[1]
        thumb0 = compute_thumbnail_peaks(r0.full_path)
        store.save_cached_thumbnails([(r0.id, r0.mtime_ns, thumb0)])

        # query_page returns cached thumb_blob, and stale mtime_ns misses cleanly
        rows_after = store.query_page(offset=0, limit=10)
        by_id = {r.id: r for r in rows_after}
        assert by_id[r0.id].thumb_blob == thumb0
        assert by_id[r1.id].thumb_blob is None
        assert store.get_cached_thumbnail(r0.id, r0.mtime_ns + 1) is None

        # save_cached_peaks on r1 also populates waveform_thumbs automatically
        arr = np.zeros((1, 2048, 2), dtype=np.float32)
        arr[0, 500, 1] = 0.75
        peak_data = PeakData(
            duration_seconds=0.1,
            sample_rate_hz=48000,
            channel_count=1,
            points=2048,
            peaks=arr,
        )
        store.save_cached_peaks(r1.id, r1.mtime_ns, peak_data, requested_points=2048)
        assert store.get_cached_thumbnail(r1.id, r1.mtime_ns) == peaks_to_thumbnail(peak_data)

        # Removing root cascades waveform_thumbs
        store.remove_root(root.id)
        assert store.get_cached_thumbnails([(r0.id, r0.mtime_ns), (r1.id, r1.mtime_ns)]) == {}
    finally:
        store.close()


def test_512pt_upsample_interpolation_and_schema_migration(tmp_path):
    from audiosearch.browser.waveform_delegate import _sample_columns
    from audiosearch.library.scanner import LibraryScanner
    from audiosearch.library.schema import ensure_schema

    assert THUMBNAIL_POINTS == 512

    # Upsampling coarse envelope produces strictly monotonic ramp (no stair-step duplicates).
    coarse = np.linspace(0.0, 1.0, 128, dtype=np.float32)
    upsampled = _sample_columns(coarse, 320)
    assert upsampled.shape == (320,)
    assert np.all(np.diff(upsampled) > 0.0)

    # Downsampling preserves sharp transient spikes.
    dense = np.zeros(512, dtype=np.float32)
    dense[255] = 0.95
    downsampled = _sample_columns(dense, 128)
    assert downsampled.shape == (128,)
    assert float(downsampled.max()) >= 0.94

    # Legacy user_version == 0 DB migrates cleanly: old 128-pt waveform_thumbs cleared, files kept.
    sfx = tmp_path / "sfx"
    _write_pcm16_wav(sfx / "legacy.wav", np.full(4800, 0.3, dtype=np.float32))
    db_path = tmp_path / "migrate.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(sfx)
    LibraryScanner(db_path).scan_root(root.id)
    rec = store.query_page(offset=0, limit=10)[0]
    store.save_cached_thumbnails([(rec.id, rec.mtime_ns, b"\x55" * 128)])
    conn = store._conn()
    conn.execute("PRAGMA user_version = 0")
    conn.commit()
    ensure_schema(conn)
    assert int(conn.execute("PRAGMA user_version").fetchone()[0]) >= 2
    assert store.get_cached_thumbnails([(rec.id, rec.mtime_ns)]) == {}
    assert store.total_file_count() == 1
    store.close()
