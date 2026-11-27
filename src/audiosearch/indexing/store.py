"""Vector artifacts with an auditable header, atomic publishes and asset reuse.

Layout of one store directory (schema ``embeddings-v2``)::

    <dir>/CURRENT                          pointer: "generations/gen-<uuid>"
    <dir>/generations/gen-<uuid>/header.json
    <dir>/generations/gen-<uuid>/vectors.npy
    <dir>/generations/gen-<uuid>/rows.jsonl

A write never deletes the live generation. It stages a complete new generation,
re-loads and verifies it from disk, atomically swaps the ``CURRENT`` pointer, and
only then removes generations that are neither live nor the immediate
predecessor. An interrupted write therefore leaves the previous generation
loadable: the pointer is the single commit point.

``embeddings-v1`` stores (flat ``header.json`` beside the data files, no input
fingerprints) are still readable so artifacts produced before this change remain
inspectable; they are never reused, because a v1 header cannot prove which inputs
produced its vectors.

Reuse is per asset and requires all three of:

* the same embedding space (model, revision, tower, preprocessing),
* the same ``content_revision`` from the manifest,
* the same *input fingerprint* — a hash of the exact text or window list that was
  encoded. Renaming an asset without touching its audio changes the text
  fingerprint, so the stale text vector is not reused.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np

from audiosearch.contracts.models import AssetRecord
from audiosearch.inference.spaces import EmbeddingSpace

SCHEMA_VERSION = "embeddings-v2"
LEGACY_SCHEMA_VERSIONS = frozenset({"embeddings-v1"})
POINTER_NAME = "CURRENT"
GENERATIONS_DIR = "generations"
DATA_FILES = ("header.json", "vectors.npy", "rows.jsonl")
# Normalized rows must be unit length; 1e-3 leaves room for float32 accumulation.
NORM_TOLERANCE = 1e-3
RowKind = Literal["asset", "segment"]


class ArtifactError(Exception):
    """Base class for unreadable or untrustworthy vector artifacts."""


class ArtifactCorruptError(ArtifactError):
    """Checksum, shape or content disagree with what the header claims."""


class ArtifactIncompatibleError(ArtifactError):
    """The artifact is structurally valid but cannot be used by this code."""


_FINGERPRINT_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def is_valid_fingerprint(value: object) -> bool:
    """A fingerprint is ``sha256:`` plus 64 lowercase hex characters."""
    return isinstance(value, str) and _FINGERPRINT_RE.match(value) is not None


@dataclass(frozen=True, slots=True)
class EmbeddingRow:
    asset_id: str
    kind: RowKind
    start_seconds: float
    end_seconds: float
    # For a segment: fraction of the file the window covers; 1.0 for asset rows.
    coverage: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "kind": self.kind,
            "start_seconds": round(self.start_seconds, 6),
            "end_seconds": round(self.end_seconds, 6),
            "coverage": round(self.coverage, 6),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EmbeddingRow:
        return cls(
            asset_id=payload["asset_id"],
            kind=payload["kind"],
            start_seconds=payload["start_seconds"],
            end_seconds=payload["end_seconds"],
            coverage=payload.get("coverage", 1.0),
        )

    def validate(self) -> None:
        if self.kind not in ("asset", "segment"):
            raise ArtifactCorruptError(f"unknown row kind: {self.kind!r}")
        if not (self.start_seconds >= 0.0):
            raise ArtifactCorruptError(f"row starts before zero: {self.start_seconds}")
        if not (self.end_seconds > self.start_seconds):
            raise ArtifactCorruptError(
                f"row range is empty or inverted: [{self.start_seconds}, {self.end_seconds})"
            )
        if not (0.0 <= self.coverage <= 1.0):
            raise ArtifactCorruptError(f"row coverage outside [0, 1]: {self.coverage}")


@dataclass
class EmbeddingBatch:
    """Vectors plus their rows, before they are written anywhere."""

    space: EmbeddingSpace
    vectors: np.ndarray
    rows: list[EmbeddingRow]
    revisions: dict[str, str] = field(default_factory=dict)
    input_fingerprints: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2 or self.vectors.shape[0] != len(self.rows):
            raise ValueError("vectors and rows disagree")
        if self.vectors.shape[1] != self.space.dimension:
            raise ValueError(
                f"vector width {self.vectors.shape[1]} != space dimension {self.space.dimension}"
            )
        if not np.all(np.isfinite(self.vectors)):
            raise ValueError("vectors contain non-finite values")
        for row in self.rows:
            row.validate()
        row_assets = {row.asset_id for row in self.rows}
        if row_assets - set(self.revisions):
            missing = sorted(row_assets - set(self.revisions))[:3]
            raise ValueError(f"rows reference assets without a revision: {missing}")
        if set(self.revisions) - row_assets:
            raise ValueError(
                "revisions listed for assets with no rows: "
                f"{sorted(set(self.revisions) - row_assets)[:3]}"
            )
        # A v2 artifact must be able to prove which inputs produced each vector.
        for asset_id in row_assets:
            fingerprint = self.input_fingerprints.get(asset_id)
            if not is_valid_fingerprint(fingerprint):
                raise ValueError(
                    f"asset {asset_id} needs a valid input fingerprint before it can be written"
                )


@dataclass(frozen=True, slots=True)
class ValidityReport:
    """Which stored assets may be used for a given catalog and input set."""

    valid: frozenset[str]
    stale: frozenset[str]
    missing: frozenset[str]
    # Rows that pass the revision check but whose artifact cannot prove which
    # inputs produced them (a legacy embeddings-v1 artifact). Never returned by a
    # normal search; kept so a diagnostic run can still inspect the artifact.
    unverified: frozenset[str] = frozenset()

    @property
    def needs_rebuild(self) -> bool:
        return bool(self.stale or self.missing or self.unverified)

    def as_dict(self) -> dict[str, int]:
        return {
            "valid": len(self.valid),
            "stale": len(self.stale),
            "missing": len(self.missing),
            "unverified": len(self.unverified),
        }


class EmbeddingStore:
    """One on-disk embedding space: versioned generations behind a pointer."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.header: dict[str, Any] | None = None
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        self.rows: list[EmbeddingRow] = []
        self._active: Path | None = None

    # -- reading ------------------------------------------------------------
    @property
    def is_empty(self) -> bool:
        return self.header is None

    @property
    def schema_version(self) -> str | None:
        return (self.header or {}).get("schema_version")

    @property
    def revisions(self) -> dict[str, str]:
        return dict((self.header or {}).get("asset_revisions", {}))

    @property
    def input_fingerprints(self) -> dict[str, str]:
        return dict((self.header or {}).get("input_fingerprints", {}))

    @property
    def has_input_fingerprints(self) -> bool:
        return bool(self.input_fingerprints)

    @property
    def space_id(self) -> str | None:
        return (self.header or {}).get("space", {}).get("space_id")

    @property
    def params(self) -> dict[str, Any]:
        return dict((self.header or {}).get("params", {}))

    @property
    def generation(self) -> str | None:
        """Stable token for the currently published data, cheap to re-read."""
        pointer = self.directory / POINTER_NAME
        if pointer.is_file():
            try:
                value = pointer.read_text(encoding="utf-8").strip()
            except OSError:
                return None
            return value or None
        flat_header = self.directory / "header.json"
        if flat_header.is_file():
            try:
                stat = flat_header.stat()
            except OSError:
                return None
            return f"flat:{stat.st_size}:{stat.st_mtime_ns}"
        return None

    def current_generation(self) -> str | None:
        """Backwards-compatible alias used by retrievers' cache checks."""
        return self.generation

    def _active_directory(self) -> Path | None:
        pointer = self.directory / POINTER_NAME
        if pointer.is_file():
            value = pointer.read_text(encoding="utf-8").strip()
            if not value:
                raise ArtifactCorruptError(f"empty pointer file: {pointer}")
            target = Path(value)
            if target.is_absolute():
                raise ArtifactCorruptError(f"pointer must be relative: {value!r}")
            target = (self.directory / target).resolve()
            root = self.directory.resolve()
            if not target.is_relative_to(root):
                raise ArtifactCorruptError(f"pointer escapes the store: {value!r}")
            if not (target / "header.json").is_file():
                raise ArtifactCorruptError(f"pointer references a missing generation: {value!r}")
            return target
        if (self.directory / "header.json").is_file():
            return self.directory
        return None

    def load(self) -> EmbeddingStore:
        """Read and fully validate the published generation."""
        active = self._active_directory()
        if active is None:
            self._active = None
            self.header = None
            self.vectors = np.zeros((0, 0), dtype=np.float32)
            self.rows = []
            return self

        try:
            header = json.loads((active / "header.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactCorruptError(f"unreadable header at {active}: {exc}") from exc
        if not isinstance(header, dict):
            raise ArtifactCorruptError("header is not a JSON object")

        schema = header.get("schema_version")
        if schema != SCHEMA_VERSION and schema not in LEGACY_SCHEMA_VERSIONS:
            raise ArtifactIncompatibleError(
                f"unsupported artifact schema {schema!r}; this build reads {SCHEMA_VERSION} "
                f"and {sorted(LEGACY_SCHEMA_VERSIONS)}"
            )
        space = self._space_from_header(header)

        try:
            vectors = np.load(active / "vectors.npy")
        except (OSError, ValueError) as exc:
            raise ArtifactCorruptError(f"unreadable vectors at {active}: {exc}") from exc
        if vectors.ndim != 2:
            raise ArtifactCorruptError(f"vectors must be 2-D, got shape {vectors.shape}")
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.shape[1] != space.dimension:
            raise ArtifactIncompatibleError(
                f"vector width {vectors.shape[1]} != space dimension {space.dimension}"
            )
        if _int_field(header, "dimension") != space.dimension:
            raise ArtifactCorruptError(
                f"header dimension {header.get('dimension')} != space dimension {space.dimension}"
            )
        if not np.all(np.isfinite(vectors)):
            raise ArtifactCorruptError("vectors contain non-finite values")

        try:
            lines = (active / "rows.jsonl").read_text(encoding="utf-8").splitlines()
            rows = [EmbeddingRow.from_dict(json.loads(line)) for line in lines if line.strip()]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ArtifactCorruptError(f"unreadable rows at {active}: {exc}") from exc

        if _int_field(header, "count") != vectors.shape[0]:
            raise ArtifactCorruptError(
                f"header count {header.get('count')} != {vectors.shape[0]} vectors"
            )
        if len(rows) != vectors.shape[0]:
            raise ArtifactCorruptError(f"{vectors.shape[0]} vectors but {len(rows)} rows")

        revisions = header.get("asset_revisions")
        if not isinstance(revisions, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in revisions.items()
        ):
            raise ArtifactCorruptError("asset_revisions is missing or malformed")
        if _int_field(header, "asset_count") != len(revisions):
            raise ArtifactCorruptError(
                f"header asset_count {header.get('asset_count')} != {len(revisions)} revisions"
            )
        for row in rows:
            row.validate()
            if row.asset_id not in revisions:
                raise ArtifactCorruptError(
                    f"row references an asset without a revision: {row.asset_id}"
                )
        row_assets = {row.asset_id for row in rows}
        if row_assets != set(revisions):
            raise ArtifactCorruptError(
                "asset_revisions does not match the assets present in rows.jsonl"
            )

        if space.normalized:
            norms = np.linalg.norm(vectors, axis=1)
            if norms.size:
                low, high = float(norms.min()), float(norms.max())
                if low < 1.0 - NORM_TOLERANCE or high > 1.0 + NORM_TOLERANCE:
                    raise ArtifactCorruptError(
                        f"space declares normalized vectors but norms span [{low:.6f}, {high:.6f}]"
                    )

        self._verify_checksum(header, vectors, rows, schema)

        self._active = active
        self.header = header
        self.vectors = vectors
        self.rows = rows
        return self

    @staticmethod
    def _space_from_header(header: dict[str, Any]) -> EmbeddingSpace:
        space = header.get("space")
        if not isinstance(space, dict):
            raise ArtifactCorruptError("header has no space block")
        required = (
            "model_id",
            "revision",
            "tower",
            "preprocessing_version",
            "dimension",
            "normalized",
        )
        missing = [key for key in required if key not in space]
        if missing:
            raise ArtifactCorruptError(f"space block is missing {missing}")
        if space["tower"] not in ("text", "audio"):
            raise ArtifactIncompatibleError(f"unknown tower {space['tower']!r}")
        if not isinstance(space["dimension"], int) or space["dimension"] <= 0:
            raise ArtifactCorruptError(f"invalid dimension {space['dimension']!r}")
        if not isinstance(space["normalized"], bool):
            raise ArtifactCorruptError("space.normalized must be a boolean")
        try:
            rebuilt = EmbeddingSpace(
                model_id=space["model_id"],
                revision=space["revision"],
                tower=space["tower"],
                preprocessing_version=space["preprocessing_version"],
                dimension=space["dimension"],
                normalized=space["normalized"],
            )
        except (TypeError, ValueError) as exc:
            raise ArtifactCorruptError(f"invalid space block: {exc}") from exc
        recorded = space.get("space_id")
        if recorded is not None and recorded != rebuilt.space_id:
            raise ArtifactCorruptError(
                f"recorded space_id {recorded!r} does not match its components ({rebuilt.space_id})"
            )
        return rebuilt

    @staticmethod
    def _verify_checksum(
        header: dict[str, Any], vectors: np.ndarray, rows: list[EmbeddingRow], schema: str
    ) -> None:
        recorded = header.get("checksum")
        if not isinstance(recorded, str):
            raise ArtifactCorruptError("header has no checksum")
        actual = (
            _checksum_v2(header, vectors, rows)
            if schema == SCHEMA_VERSION
            else _checksum_v1(vectors, rows)
        )
        if actual != recorded:
            raise ArtifactCorruptError(
                "artifact checksum mismatch: data or header was modified after writing"
            )

    # -- reuse --------------------------------------------------------------
    def validity(
        self,
        space: EmbeddingSpace,
        assets: list[AssetRecord],
        fingerprints: Mapping[str, str],
        *,
        allow_unfingerprinted: bool = False,
    ) -> ValidityReport:
        """Classify stored rows against the catalog and the current inputs.

        By default nothing is usable unless its inputs can be proven: an asset is
        ``valid`` only when its revision matches and both the stored and the
        expected input fingerprint are present, well formed and equal.

        ``allow_unfingerprinted`` is a diagnostics-only escape hatch for legacy
        ``embeddings-v1`` rows: those are reported as ``valid`` *and*
        ``unverified`` so a tool can inspect them. Normal search never sets it.
        """
        if self.header is None or self.space_id != space.space_id:
            return ValidityReport(
                frozenset(), frozenset(), frozenset(str(a.asset_id) for a in assets)
            )
        stored_rev = self.revisions
        stored_fp = self.input_fingerprints
        legacy = self.schema_version in LEGACY_SCHEMA_VERSIONS
        valid: set[str] = set()
        unverified: set[str] = set()
        stale: set[str] = set()
        missing: set[str] = set()
        for asset in assets:
            asset_id = str(asset.asset_id)
            if asset_id not in stored_rev:
                missing.add(asset_id)
                continue
            if stored_rev[asset_id] != asset.content_revision:
                stale.add(asset_id)
                continue
            stored_value = stored_fp.get(asset_id)
            expected_value = fingerprints.get(asset_id)
            if legacy:
                # No fingerprint was ever written; only a diagnostic caller may
                # use these rows, and it must still see them as unverified.
                unverified.add(asset_id)
                if allow_unfingerprinted:
                    valid.add(asset_id)
                continue
            if not (
                is_valid_fingerprint(stored_value)
                and is_valid_fingerprint(expected_value)
                and stored_value == expected_value
            ):
                # Missing on either side, malformed, or simply different: the
                # vector cannot be tied to the current inputs.
                stale.add(asset_id)
                continue
            valid.add(asset_id)
        return ValidityReport(
            frozenset(valid), frozenset(stale), frozenset(missing), frozenset(unverified)
        )

    def reusable_assets(
        self,
        space: EmbeddingSpace,
        assets: list[AssetRecord],
        fingerprints: Mapping[str, str],
    ) -> set[str]:
        """Assets whose stored rows were built from exactly these inputs."""
        return set(self.validity(space, assets, fingerprints).valid)

    def subset(self, asset_ids: set[str]) -> EmbeddingBatch:
        keep = np.array([row.asset_id in asset_ids for row in self.rows], dtype=bool)
        header = self.header or {}
        return EmbeddingBatch(
            space=self._space_from_header(header),
            vectors=self.vectors[keep],
            rows=[row for row, taken in zip(self.rows, keep, strict=True) if taken],
            revisions={key: value for key, value in self.revisions.items() if key in asset_ids},
            input_fingerprints={
                key: value for key, value in self.input_fingerprints.items() if key in asset_ids
            },
        )

    # -- writing ------------------------------------------------------------
    def write(
        self,
        batch: EmbeddingBatch,
        *,
        manifest_id: str,
        library_id: str,
        fingerprint: dict[str, Any],
        params: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Stage, verify and publish a new generation; the old one stays until then."""
        batch.__post_init__()  # re-validate before touching the disk
        root = self.directory
        root.mkdir(parents=True, exist_ok=True)
        (root / GENERATIONS_DIR).mkdir(exist_ok=True)
        previous = self._active_directory()

        staged = self._stage(
            batch,
            manifest_id=manifest_id,
            library_id=library_id,
            fingerprint=fingerprint,
            params=params,
            extra=extra,
        )
        try:
            # Verify from disk, exactly as a later reader would.
            probe = EmbeddingStore(staged)
            probe.load()
            if probe.vectors.shape != batch.vectors.shape:
                raise ArtifactCorruptError("staged generation does not round-trip")
            self._publish(staged)
        except Exception:
            # The old generation is still live: the pointer was never replaced.
            shutil.rmtree(staged, ignore_errors=True)
            raise

        self._prune(keep={staged, previous} if previous is not None else {staged})
        self.load()
        return root

    def _stage(
        self,
        batch: EmbeddingBatch,
        *,
        manifest_id: str,
        library_id: str,
        fingerprint: dict[str, Any],
        params: dict[str, Any],
        extra: dict[str, Any] | None,
    ) -> Path:
        generation = f"gen-{uuid4().hex}"
        staged = self.directory / GENERATIONS_DIR / generation
        if staged.exists():  # pragma: no cover - uuid collision
            shutil.rmtree(staged)
        staged.mkdir(parents=True)
        vectors = np.ascontiguousarray(batch.vectors, dtype=np.float32)
        rows_text = "".join(
            json.dumps(row.as_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            for row in batch.rows
        )
        header: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "generation": generation,
            "space": batch.space.as_dict(),
            "manifest_id": manifest_id,
            "library_id": library_id,
            "model": fingerprint,
            "params": params,
            "count": int(vectors.shape[0]),
            "dimension": int(vectors.shape[1]),
            "asset_count": len({row.asset_id for row in batch.rows}),
            "asset_revisions": dict(sorted(batch.revisions.items())),
            "input_fingerprints": dict(sorted(batch.input_fingerprints.items())),
        }
        if extra:
            header.update(extra)
        header["checksum"] = _checksum_v2(header, vectors, batch.rows)
        try:
            np.save(staged / "vectors.npy", vectors)
            (staged / "rows.jsonl").write_text(rows_text, encoding="utf-8", newline="\n")
            (staged / "header.json").write_text(
                json.dumps(header, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        except BaseException:
            shutil.rmtree(staged, ignore_errors=True)
            raise
        return staged

    def _publish(self, staged: Path) -> None:
        """Atomically repoint CURRENT; this single rename is the commit point."""
        pointer = self.directory / POINTER_NAME
        relative = f"{GENERATIONS_DIR}/{staged.name}"
        temporary = self.directory / f".{POINTER_NAME}.{uuid4().hex}.tmp"
        try:
            temporary.write_text(relative + "\n", encoding="utf-8")
            _retry(lambda: os.replace(temporary, pointer))
        finally:
            temporary.unlink(missing_ok=True)

    def _prune(self, keep: set[Path]) -> None:
        """Remove generations that are neither live nor the rollback predecessor."""
        keep_names = {path.name for path in keep}
        generations = self.directory / GENERATIONS_DIR
        if generations.is_dir():
            for child in generations.iterdir():
                if child.name in keep_names or not child.name.startswith("gen-"):
                    continue
                shutil.rmtree(child, ignore_errors=True)
        # A flat legacy layout is superseded by the new generation; drop its files
        # only after the pointer swap succeeded, so a failed write keeps them.
        for name in DATA_FILES:
            legacy = self.directory / name
            if legacy.is_file():
                legacy.unlink(missing_ok=True)


def _int_field(header: dict[str, Any], key: str) -> int:
    """Read an integer header field, rejecting bools and non-integers."""
    value = header.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactCorruptError(f"header field {key!r} must be an integer, got {value!r}")
    return value


def _retry(action, *, attempts: int = 5, delay: float = 0.05) -> None:
    """Retry a filesystem step that a scanner or indexer may briefly lock.

    Windows rejects ``rmtree``/``os.replace`` with ``PermissionError`` while
    another process holds a handle; the store is only ever written by its owner,
    so a short bounded retry is safe and avoids a spurious failed encode.
    """
    for attempt in range(attempts):
        try:
            action()
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay * (attempt + 1))


def _checksum_v2(header: dict[str, Any], vectors: np.ndarray, rows: list[EmbeddingRow]) -> str:
    """Cover the header (minus the checksum itself), the vectors and the rows."""
    payload = {key: value for key, value in header.items() if key != "checksum"}
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            payload, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
    )
    digest.update(np.ascontiguousarray(vectors, dtype=np.float32).tobytes())
    for row in rows:
        digest.update(json.dumps(row.as_dict(), sort_keys=True).encode("utf-8"))
    return "sha256:" + digest.hexdigest()


def _checksum_v1(vectors: np.ndarray, rows: list[EmbeddingRow]) -> str:
    """Legacy formula: vectors and rows only, header not covered."""
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(vectors, dtype=np.float32).tobytes())
    for row in rows:
        digest.update(json.dumps(row.as_dict(), sort_keys=True).encode("utf-8"))
    return "sha256:" + digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


__all__ = [
    "DATA_FILES",
    "GENERATIONS_DIR",
    "LEGACY_SCHEMA_VERSIONS",
    "POINTER_NAME",
    "SCHEMA_VERSION",
    "ArtifactCorruptError",
    "ArtifactError",
    "ArtifactIncompatibleError",
    "EmbeddingBatch",
    "EmbeddingRow",
    "EmbeddingStore",
    "ValidityReport",
    "write_json_atomic",
]
