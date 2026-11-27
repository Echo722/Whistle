"""Catalog + root-map resolution so B never builds a second identity system.

The only source of truth for asset identity is the private sample manifest and
its root map (``.local/bootstrap/manifest.json`` + ``roots.local.json``). This
module resolves ``root_id`` → absolute directory and exposes a read-only
``AssetCatalog``-shaped view plus the on-disk path for an ``AssetRef``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from audiosearch.contracts.models import (
    AssetLocation,
    AssetRecord,
    AssetRef,
    SampleManifest,
)

from .errors import AssetChangedError, AssetNotFoundError, AssetOfflineError


@dataclass(frozen=True, slots=True)
class ResolvedAsset:
    """A catalog record together with the concrete readable path."""

    asset: AssetRecord
    location: AssetLocation
    path: Path

    @property
    def asset_id(self) -> UUID:
        return self.asset.asset_id

    @property
    def content_revision(self) -> str:
        return self.asset.content_revision


def load_manifest(manifest_path: Path, roots_path: Path) -> tuple[SampleManifest, dict[str, Path]]:
    """Load the private manifest and its root map, rejecting a mismatched pair.

    Root values must be absolute directories on this machine. They are resolved
    when an asset is opened, and every asset path is required to stay inside its
    resolved root, so a junction or symlink cannot widen the read scope.
    """
    manifest = SampleManifest.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))
    roots_payload = json.loads(roots_path.read_text(encoding="utf-8"))
    if roots_payload.get("manifest_id") != manifest.manifest_id:
        raise ValueError(
            "root map does not belong to this manifest: "
            f"{roots_payload.get('manifest_id')} != {manifest.manifest_id}"
        )
    roots: dict[str, Path] = {}
    for key, value in roots_payload["roots"].items():
        candidate = Path(str(value))
        if not candidate.is_absolute():
            raise ValueError(f"root {key!r} must be an absolute path, got {value!r}")
        roots[str(key)] = candidate
    return manifest, roots


class ManifestCatalog:
    """Read-only catalog over a fixed manifest; no database, no writes."""

    def __init__(self, manifest: SampleManifest, roots: dict[str, Path]) -> None:
        self.manifest = manifest
        self.roots = roots
        self._by_id = {asset.asset_id: asset for asset in manifest.assets}

    @classmethod
    def from_files(cls, manifest_path: Path, roots_path: Path) -> ManifestCatalog:
        manifest, roots = load_manifest(manifest_path, roots_path)
        return cls(manifest, roots)

    @property
    def assets(self) -> list[AssetRecord]:
        return self.manifest.assets

    def get_asset(self, asset_id: UUID) -> AssetRecord | None:
        return self._by_id.get(asset_id)

    def require_asset(self, asset_id: UUID) -> AssetRecord:
        asset = self._by_id.get(asset_id)
        if asset is None:
            raise AssetNotFoundError(f"asset {asset_id} is not in this manifest")
        return asset

    def path_for(self, asset: AssetRecord, *, verify: bool = True) -> Path:
        """Resolve a readable absolute path, checking availability, version and root containment."""
        location = self._primary_location(asset)
        root = self.roots.get(location.root_id)
        if root is None:
            raise AssetOfflineError(f"root {location.root_id!r} is not mapped on this machine")
        if location.availability != "online":
            raise AssetOfflineError(f"asset {asset.name} is marked {location.availability}")
        resolved_root, path = self._contained_path(location, root, asset)
        if verify:
            self._verify_on_disk(asset, location, path)
        return path

    @staticmethod
    def _contained_path(
        location: AssetLocation, root: Path, asset: AssetRecord
    ) -> tuple[Path, Path]:
        return contained_path(location, root, asset)

    def resolve(self, ref: AssetRef, *, verify: bool = True) -> ResolvedAsset:
        """Resolve an ``AssetRef``: identity, content version and readable path."""
        asset = self.require_asset(ref.asset_id)
        if asset.content_revision != ref.content_revision:
            raise AssetChangedError(
                f"asset {asset.name} changed: {ref.content_revision} != {asset.content_revision}"
            )
        location = self._primary_location(asset)
        return ResolvedAsset(
            asset=asset, location=location, path=self.path_for(asset, verify=verify)
        )

    def resolve_by_id(self, asset_id: UUID, *, verify: bool = True) -> ResolvedAsset:
        asset = self.require_asset(asset_id)
        location = self._primary_location(asset)
        return ResolvedAsset(
            asset=asset, location=location, path=self.path_for(asset, verify=verify)
        )

    @staticmethod
    def _primary_location(asset: AssetRecord) -> AssetLocation:
        return asset.locations[0]

    @staticmethod
    def _verify_on_disk(asset: AssetRecord, location: AssetLocation, path: Path) -> None:
        verify_on_disk(asset, location, path)


def contained_path(location: AssetLocation, root: Path, asset: AssetRecord) -> tuple[Path, Path]:
    """Join and resolve, then require the result to stay under the resolved root.

    ``relative_path`` is already validated by the contract (no ``..``, no
    drive), but a junction or symlink inside the root can still point
    elsewhere. Resolution happens once and the real path is what gets checked
    and returned, so callers cannot be handed an escaping path.

    Shared so every adapter (manifest-backed or database-backed) applies the same
    containment rule instead of re-deriving it.
    """
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise AssetOfflineError(
            f"root {location.root_id!r} is not available on this machine ({type(exc).__name__})"
        ) from exc
    if not resolved_root.is_dir():
        raise AssetOfflineError(f"root {location.root_id!r} is not a directory")
    candidate = resolved_root.joinpath(*location.relative_path.split("/"))
    try:
        resolved = candidate.resolve(strict=False)
    except OSError as exc:  # pragma: no cover - unusual filesystem errors
        raise AssetOfflineError(
            f"asset {asset.name} cannot be resolved ({type(exc).__name__})"
        ) from exc
    if not resolved.is_relative_to(resolved_root):
        raise AssetOfflineError(
            f"asset {asset.name} resolves outside its root "
            f"({location.root_id}); refusing to read it"
        )
    return resolved_root, resolved


def verify_on_disk(asset: AssetRecord, location: AssetLocation, path: Path) -> None:
    """Refuse a file whose size/mtime no longer match the recorded version."""
    try:
        stat = path.stat()
    except OSError as exc:
        raise AssetOfflineError(
            f"asset {asset.name} is not readable at its mapped root ({type(exc).__name__})"
        ) from exc
    if stat.st_size != location.size_bytes or stat.st_mtime_ns != location.mtime_ns:
        raise AssetChangedError(
            f"asset {asset.name} on disk differs from the manifest (size/mtime)"
        )


__all__ = [
    "ManifestCatalog",
    "ResolvedAsset",
    "contained_path",
    "load_manifest",
    "verify_on_disk",
]
