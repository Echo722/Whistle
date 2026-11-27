"""Indexing jobs: turn the private manifest into reproducible vector artifacts.

Two steps, kept separate on purpose:

``encode_text_assets``    BGE-M3 over name/description/keywords/package.
``encode_audio_segments`` CLAP over overlapping windows of each file, keeping the
                          original-file start/end of every window.

Both take an ``EmbeddingStore`` so a second run reuses rows only when the
embedding space, the manifest ``content_revision`` *and* the exact encoded input
(text template, or window list) all match. Each asset therefore carries an input
fingerprint: renaming a file invalidates its text vector even though the audio is
unchanged, and changing the window policy invalidates its audio windows.

Both record per-asset failures instead of throwing away the healthy files.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from audiosearch.audioio.catalog import ManifestCatalog
from audiosearch.audioio.decode import open_wav
from audiosearch.audioio.errors import AudioError
from audiosearch.contracts.models import AssetRecord
from audiosearch.indexing.store import EmbeddingBatch, EmbeddingRow, EmbeddingStore
from audiosearch.inference import runtime
from audiosearch.inference.clap_encoder import (
    AudioWindow,
    ClapEncoder,
    coverage,
    plan_windows,
)
from audiosearch.inference.spaces import EmbeddingSpace, ModelUnavailableError
from audiosearch.inference.text_encoder import BgeM3TextEncoder, plan_asset_text

# Default window policy; the values actually used are recorded per artifact.
AUDIO_WINDOW_SECONDS = 10.0
AUDIO_HOP_SECONDS = 5.0
AUDIO_WINDOW_POLICY = f"win{AUDIO_WINDOW_SECONDS:g}s-hop{AUDIO_HOP_SECONDS:g}s-onset0"


def window_policy(window_seconds: float, hop_seconds: float) -> str:
    """Version string for the window layout that was really used."""
    return f"win{window_seconds:g}s-hop{hop_seconds:g}s-onset0"


def text_input_fingerprint(asset: AssetRecord) -> str:
    """Hash of the exact text the text encoder would see for this asset."""
    payload = plan_asset_text(asset).text
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def audio_input_fingerprint(windows: Sequence[AudioWindow]) -> str:
    """Hash of the exact window list that was encoded."""
    payload = json.dumps(
        [[round(window.start_seconds, 6), round(window.end_seconds, 6)] for window in windows],
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def plan_asset_windows(
    asset: AssetRecord, *, window_seconds: float, hop_seconds: float
) -> list[AudioWindow]:
    duration = float(asset.duration_seconds or 0.0)
    if duration <= 0:
        return []
    return plan_windows(duration, window_seconds=window_seconds, hop_seconds=hop_seconds)


@dataclass(slots=True)
class AssetFailure:
    asset_id: str
    name: str
    error_code: str
    message: str


@dataclass(slots=True)
class EncodeReport:
    """What an encode pass actually did; never estimates."""

    route: Literal["text", "audio"]
    space_id: str
    files_total: int = 0
    files_encoded: int = 0
    files_reused: int = 0
    windows: int = 0
    audio_seconds: float = 0.0
    covered_seconds: float = 0.0
    files_sampled: int = 0
    elapsed_seconds: float = 0.0
    model_load_seconds: float = 0.0
    failures: list[AssetFailure] = field(default_factory=list)
    peak_rss_bytes: int | None = None
    peak_gpu_bytes: int | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def files_failed(self) -> int:
        return len(self.failures)

    def as_dict(self, *, include_failures: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "route": self.route,
            "space_id": self.space_id,
            "files_total": self.files_total,
            "files_encoded": self.files_encoded,
            "files_reused": self.files_reused,
            "files_failed": self.files_failed,
            "windows": self.windows,
            "audio_seconds": round(self.audio_seconds, 3),
            "covered_seconds": round(self.covered_seconds, 3),
            "files_sampled": self.files_sampled,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "model_load_seconds": round(self.model_load_seconds, 3),
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_gpu_bytes": self.peak_gpu_bytes,
            "notes": self.notes,
        }
        if include_failures:
            payload["failures"] = [
                {
                    "asset_id": item.asset_id,
                    "name": item.name,
                    "error_code": item.error_code,
                    "message": item.message,
                }
                for item in self.failures
            ]
        return payload


class _GpuPeak:
    """Peak CUDA allocation across a block; no-ops on CPU-only hosts."""

    def __init__(self) -> None:
        try:
            import torch
        except ImportError:  # pragma: no cover
            torch = None
        self._torch = torch if torch is not None and torch.cuda.is_available() else None

    def reset(self) -> None:
        if self._torch is not None:
            self._torch.cuda.reset_peak_memory_stats()

    def read(self) -> int | None:
        if self._torch is None:
            return None
        return int(self._torch.cuda.max_memory_allocated())


def _reuse_mask(
    store: EmbeddingStore,
    space: EmbeddingSpace,
    assets: list[AssetRecord],
    fingerprints: dict[str, str],
) -> tuple[set[str], EmbeddingBatch | None]:
    """Load whatever is on disk and return the rows still valid for this run."""
    if store.generation is not None:
        store.load()
    elif (store.directory / "header.json").is_file():
        store.load()
    reusable = store.reusable_assets(space, assets, fingerprints)
    if not reusable:
        return set(), None
    return reusable, store.subset(reusable)


def _finish(report: EncodeReport, started: float, load_seconds: float, gpu: _GpuPeak) -> None:
    report.elapsed_seconds = time.perf_counter() - started
    report.model_load_seconds = load_seconds
    report.peak_gpu_bytes = gpu.read()
    report.peak_rss_bytes = runtime.peak_rss_bytes()


def encode_text_assets(
    catalog: ManifestCatalog,
    store: EmbeddingStore,
    encoder: BgeM3TextEncoder,
    *,
    batch_size: int = 8,
    limit: int | None = None,
    reuse: bool = True,
) -> EncodeReport:
    """Encode catalog text into ``store``; reuse unchanged assets."""
    assets = catalog.assets[:limit] if limit else list(catalog.assets)
    started = time.perf_counter()
    load_started = time.perf_counter()
    space = encoder.load()
    load_seconds = time.perf_counter() - load_started
    gpu = _GpuPeak()
    gpu.reset()
    report = EncodeReport(route="text", space_id=space.space_id, files_total=len(assets))

    fingerprints = {str(asset.asset_id): text_input_fingerprint(asset) for asset in assets}
    reused_ids: set[str] = set()
    previous: EmbeddingBatch | None = None
    if reuse:
        reused_ids, previous = _reuse_mask(store, space, assets, fingerprints)
    report.files_reused = len(reused_ids)

    todo = [asset for asset in assets if str(asset.asset_id) not in reused_ids]
    # A WAV whose header this reader cannot parse (e.g. IEEE float) has no duration.
    # An asset row needs a non-empty range, so encoding it would fail the whole batch
    # long after the useful work was done. Report it and encode the rest.
    encodable: list[AssetRecord] = []
    for asset in todo:
        if not asset.duration_seconds or asset.duration_seconds <= 0:
            report.failures.append(
                AssetFailure(
                    str(asset.asset_id), asset.name, "invalid_request", "unknown duration"
                )
            )
            continue
        encodable.append(asset)
    todo = encodable
    vectors: list[np.ndarray] = []
    rows: list[EmbeddingRow] = []
    revisions: dict[str, str] = {}
    sparse = 0
    try:
        if todo:
            matrix, plans = encoder.encode_assets(todo)
            sparse = sum(1 for plan in plans if plan.is_sparse)
            vectors.append(matrix)
            for asset in todo:
                rows.append(
                    EmbeddingRow(
                        asset_id=str(asset.asset_id),
                        kind="asset",
                        start_seconds=0.0,
                        end_seconds=float(asset.duration_seconds or 0.0),
                    )
                )
                revisions[str(asset.asset_id)] = asset.content_revision
        if previous is not None:
            vectors.insert(0, previous.vectors)
            rows = previous.rows + rows
            revisions.update(previous.revisions)
    finally:
        encoder.close()

    if sparse:
        report.notes.append(f"{sparse} assets had only name/package fields (no description)")
    report.files_encoded = len(todo)
    matrix = (
        np.concatenate(vectors, axis=0) if vectors else np.zeros((0, space.dimension), np.float32)
    )
    batch = EmbeddingBatch(
        space=space,
        vectors=matrix,
        rows=rows,
        revisions=revisions,
        input_fingerprints=fingerprints,
    )
    store.write(
        batch,
        manifest_id=catalog.manifest.manifest_id,
        library_id=catalog.manifest.library_id,
        fingerprint=(encoder.fingerprint.as_dict() if encoder.fingerprint else {}),
        params={
            "batch_size": batch_size,
            "max_tokens": encoder.config.max_tokens,
            "template_module": plan_asset_text.__module__,
            "text_template": "bge-m3-fields-v1",
            "sparse_assets": sparse,
        },
        extra={"reuse": True},
    )
    _finish(report, started, load_seconds, gpu)
    return report


def encode_audio_segments(
    catalog: ManifestCatalog,
    store: EmbeddingStore,
    encoder: ClapEncoder,
    *,
    limit: int | None = None,
    window_seconds: float = AUDIO_WINDOW_SECONDS,
    hop_seconds: float = AUDIO_HOP_SECONDS,
    batch_size: int = 8,
    reuse: bool = True,
    target_sample_rate: int = 48000,
) -> EncodeReport:
    """Decode → window → CLAP-encode every asset; keep original-file ranges."""
    assets = catalog.assets[:limit] if limit else list(catalog.assets)
    started = time.perf_counter()
    load_started = time.perf_counter()
    space = encoder.load()
    load_seconds = time.perf_counter() - load_started
    gpu = _GpuPeak()
    gpu.reset()
    report = EncodeReport(route="audio", space_id=space.space_id, files_total=len(assets))

    planned: dict[str, list[AudioWindow]] = {}
    fingerprints: dict[str, str] = {}
    for asset in assets:
        windows = plan_asset_windows(asset, window_seconds=window_seconds, hop_seconds=hop_seconds)
        planned[str(asset.asset_id)] = windows
        fingerprints[str(asset.asset_id)] = audio_input_fingerprint(windows)

    reused_ids: set[str] = set()
    previous: EmbeddingBatch | None = None
    if reuse:
        reused_ids, previous = _reuse_mask(store, space, assets, fingerprints)
    report.files_reused = len(reused_ids)

    vectors: list[np.ndarray] = []
    rows: list[EmbeddingRow] = []
    revisions: dict[str, str] = {}
    try:
        for asset in assets:
            asset_id = str(asset.asset_id)
            if asset_id in reused_ids:
                continue
            duration = float(asset.duration_seconds or 0.0)
            if duration <= 0:
                report.failures.append(
                    AssetFailure(asset_id, asset.name, "invalid_request", "unknown duration")
                )
                continue
            windows = planned[asset_id]
            try:
                chunk_vectors = _encode_asset(
                    encoder,
                    catalog,
                    asset,
                    windows,
                    windows_per_batch=max(1, encoder.config.batch_size),
                )
            except (AudioError, ModelUnavailableError) as exc:
                report.failures.append(
                    AssetFailure(asset_id, asset.name, exc.code.value, exc.message)
                )
                continue
            except (OSError, ValueError, RuntimeError) as exc:
                report.failures.append(
                    AssetFailure(asset_id, asset.name, "internal_error", str(exc)[:300])
                )
                continue
            vectors.append(chunk_vectors)
            covered = coverage(windows, duration)
            rows.extend(
                EmbeddingRow(
                    asset_id=asset_id,
                    kind="segment",
                    start_seconds=window.start_seconds,
                    end_seconds=window.end_seconds,
                    coverage=covered,
                )
                for window in windows
            )
            revisions[asset_id] = asset.content_revision
            report.files_encoded += 1
            report.windows += len(windows)
            report.audio_seconds += duration
            report.covered_seconds += covered * duration
            if covered < 1.0 - 1e-9:
                report.files_sampled += 1
        if previous is not None:
            vectors.insert(0, previous.vectors)
            rows = previous.rows + rows
            revisions.update(previous.revisions)
    finally:
        encoder.close()

    matrix = (
        np.concatenate(vectors, axis=0) if vectors else np.zeros((0, space.dimension), np.float32)
    )
    batch = EmbeddingBatch(
        space=space,
        vectors=matrix,
        rows=rows,
        revisions=revisions,
        input_fingerprints=fingerprints,
    )
    store.write(
        batch,
        manifest_id=catalog.manifest.manifest_id,
        library_id=catalog.manifest.library_id,
        fingerprint=(encoder.fingerprint.as_dict() if encoder.fingerprint else {}),
        params={
            "batch_size": batch_size,
            "window_seconds": window_seconds,
            "hop_seconds": hop_seconds,
            "window_policy": window_policy(window_seconds, hop_seconds),
            "target_sample_rate": target_sample_rate,
        },
        extra={"reuse": True},
    )
    _finish(report, started, load_seconds, gpu)
    return report


def _encode_asset(
    encoder: ClapEncoder,
    catalog: ManifestCatalog,
    asset: AssetRecord,
    windows: list[Any],
    *,
    windows_per_batch: int,
) -> np.ndarray:
    """Decode and encode a file window-group by window-group.

    Only ``windows_per_batch`` decoded windows are alive at once, so peak memory
    is set by the batch size and the window length, not by the file duration.
    A 322 s file and a 7 s file therefore need the same working set. Encoded
    vectors are 512 float32 values each, which is what gets accumulated.
    """
    path = catalog.path_for(asset)
    vectors: list[np.ndarray] = []
    pending: list[np.ndarray] = []
    with open_wav(path) as reader:
        sample_rate = reader.sample_rate_hz
        for window in windows:
            mono = reader.read_range_mono(window.start_seconds, window.end_seconds)
            if mono.shape[0] == 0:
                raise ValueError(f"empty decode for window {window.start_seconds:.3f}")
            pending.append(mono)
            if len(pending) >= windows_per_batch:
                vectors.append(
                    encoder.encode_audio_mono(pending, sample_rate, batch_size=windows_per_batch)
                )
                pending = []
        if pending:
            vectors.append(
                encoder.encode_audio_mono(pending, sample_rate, batch_size=windows_per_batch)
            )
    if not vectors:
        raise ValueError("no windows could be decoded")
    return np.concatenate(vectors, axis=0)


__all__ = [
    "AUDIO_HOP_SECONDS",
    "AUDIO_WINDOW_POLICY",
    "AUDIO_WINDOW_SECONDS",
    "AssetFailure",
    "EncodeReport",
    "audio_input_fingerprint",
    "encode_audio_segments",
    "encode_text_assets",
    "plan_asset_windows",
    "text_input_fingerprint",
    "window_policy",
]
