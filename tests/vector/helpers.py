"""Shared synthetic fixtures for the vector/retriever tests.

Everything here is temporary: catalogs are built in memory and audio is written
to a pytest ``tmp_path`` at 8 kHz mono, so the tests never touch the real library
and never need a model checkpoint.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from audiosearch.audioio.catalog import ManifestCatalog
from audiosearch.contracts.models import (
    AssetLocation,
    AssetRecord,
    SampleManifest,
)
from audiosearch.indexing.encode import plan_asset_text
from audiosearch.inference.spaces import EmbeddingSpace


def make_space(tower: str = "audio", dimension: int = 4, version: str = "v1") -> EmbeddingSpace:
    return EmbeddingSpace("stub-model", "b" * 40, tower, version, dimension, True)


def make_asset(
    index: int,
    *,
    name: str | None = None,
    duration: float = 12.0,
    revision: str | None = None,
) -> AssetRecord:
    asset_id = f"{index:08d}-0000-5000-8000-000000000000"
    return AssetRecord(
        asset_id=asset_id,
        content_revision=revision or f"rev-{index}",
        name=name or f"asset-{index}.wav",
        library_id="test-library",
        package="pkg",
        duration_seconds=duration,
        sample_rate_hz=8000,
        channel_count=1,
        bit_depth=16,
        locations=[
            AssetLocation(
                location_id=f"{index:08d}-1111-5000-8000-000000000000",
                root_id="root",
                relative_path=f"asset-{index}.wav",
                availability="online",
                size_bytes=1,
                mtime_ns=1,
            )
        ],
    )


def make_catalog(assets: list[AssetRecord], root: Path | None = None) -> ManifestCatalog:
    manifest = SampleManifest(
        manifest_id="sha256:" + "4" * 64,
        generated_at="2026-11-10T00:00:00Z",
        selection_method="sha256-relative-path-v1",
        selection_limit=64,
        library_id="test-library",
        root_id="root",
        assets=assets,
    )
    return ManifestCatalog(manifest, {"root": root or Path(".")})


def write_wav(path: Path, seconds: float, *, rate: int = 8000, tone_hz: float = 220.0) -> Path:
    """Write a small mono 16-bit WAV so the encoder path has something to decode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(seconds * rate)
    t = np.arange(frames, dtype=np.float64) / rate
    samples = (0.4 * np.sin(2 * np.pi * tone_hz * t) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())
    return path


def make_asset_for_file(index: int, path: Path, *, duration: float) -> AssetRecord:
    """An asset whose location matches the file on disk (size/mtime verified)."""
    stat = path.stat()
    asset_id = f"{index:08d}-0000-5000-8000-000000000000"
    return AssetRecord(
        asset_id=asset_id,
        content_revision=f"stat-v1:{stat.st_size}:{stat.st_mtime_ns}",
        name=path.name,
        library_id="test-library",
        package="pkg",
        duration_seconds=duration,
        sample_rate_hz=8000,
        channel_count=1,
        bit_depth=16,
        locations=[
            AssetLocation(
                location_id=f"{index:08d}-1111-5000-8000-000000000000",
                root_id="root",
                relative_path=path.name,
                availability="online",
                size_bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
            )
        ],
    )


class StubTextEncoder:
    """Deterministic stand-in for BGE-M3: hashed text -> unit vector."""

    def __init__(self, space: EmbeddingSpace) -> None:
        self.space = space
        self.config = type("C", (), {"max_tokens": 32, "query_instruction": ""})()
        self.fingerprint = type("F", (), {"as_dict": lambda _self: {"model_id": "stub-model"}})()
        self.encode_calls = 0

    def load(self) -> EmbeddingSpace:
        return self.space

    def encode_query(self, text: str) -> np.ndarray:
        return self._vector(text)[0]

    def _vector(self, text: str) -> np.ndarray:
        digest = sum(ord(char) * (index + 1) for index, char in enumerate(text))
        rng = np.random.default_rng(digest % (2**32))
        vector = rng.normal(size=(1, self.space.dimension)).astype(np.float32)
        vector /= np.linalg.norm(vector, axis=1, keepdims=True)
        return vector

    def encode_assets(self, assets):
        self.encode_calls += 1
        vectors = np.concatenate([self._vector(plan_asset_text(asset).text) for asset in assets])
        return vectors, [plan_asset_text(asset) for asset in assets]

    def close(self) -> None:
        pass


class StubClapEncoder:
    """Deterministic stand-in for CLAP: mean amplitude -> unit vector."""

    def __init__(self, space: EmbeddingSpace, batch_size: int = 4) -> None:
        self.space = space
        self.config = type("C", (), {"batch_size": batch_size})()
        self.fingerprint = type("F", (), {"as_dict": lambda _self: {"model_id": "stub-model"}})()
        self.batch_calls: list[int] = []

    def load(self) -> EmbeddingSpace:
        return self.space

    def encode_audio_mono(self, windows, sample_rate_hz, *, batch_size=None):
        rows = list(windows)
        self.batch_calls.append(len(rows))
        vectors = []
        for row in rows:
            energy = float(np.mean(np.abs(row)) + 1e-6)
            vector = np.full((self.space.dimension,), energy, dtype=np.float32)
            vector[0] = 1.0
            vector /= np.linalg.norm(vector)
            vectors.append(vector)
        return np.stack(vectors)

    def encode_texts(self, texts, batch_size=None):
        return np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (len(texts), 1))

    def close(self) -> None:
        pass
