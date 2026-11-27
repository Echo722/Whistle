"""Source-path resolution injected from the trusted catalog.

The bridge never accepts a path from the desktop. Commands carry an
``AssetRef`` (asset_id + content_revision); the adapter asks a resolver supplied
by the service (backed by A's catalog / the fixed manifest) for the record and
the verified absolute path. There is exactly one asset identity: the catalog's.

A resolver signals "cannot use this asset" by raising any exception with a
``code`` attribute that is an ``ErrorCode`` (B's ``AudioError`` family already
does: ``AssetNotFoundError`` / ``AssetOfflineError`` / ``AssetChangedError``), or
``SourceUnavailableError``. Other exceptions become ``internal_error``.

This module imports only the contracts; it does not import ``audioio`` or
``integration`` (those pull numpy) — catalogs are used by duck typing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from audiosearch.contracts.models import AssetRecord, AssetRef, ErrorCode

_SOURCE_CODES = frozenset(
    {
        ErrorCode.ASSET_NOT_FOUND,
        ErrorCode.ASSET_OFFLINE,
        ErrorCode.ASSET_CHANGED,
        ErrorCode.INVALID_REQUEST,
    }
)


@dataclass(frozen=True, slots=True)
class ResolvedSource:
    """Catalog record plus the verified, absolute, readable path of its file."""

    asset: AssetRecord
    path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise ValueError("ResolvedSource.path must be an absolute pathlib.Path")

    @property
    def ref(self) -> AssetRef:
        return AssetRef(asset_id=self.asset.asset_id, content_revision=self.asset.content_revision)


class SourceResolver(Protocol):
    def __call__(self, ref: AssetRef, /) -> ResolvedSource: ...


class SourceUnavailableError(Exception):  # noqa: N818 - mirrors audioio naming
    """Raised by a resolver; ``code`` is one of not_found/offline/changed/invalid."""

    def __init__(self, code: ErrorCode | str, message: str) -> None:
        code = ErrorCode(code)
        if code not in _SOURCE_CODES:
            raise ValueError(f"{code} is not a source error code")
        super().__init__(message)
        self.code = code
        self.message = message


def source_error_code(exc: BaseException) -> ErrorCode | None:
    """The public code an exception maps to, if it declares a usable one."""
    code = getattr(exc, "code", None)
    if isinstance(code, ErrorCode) and code in _SOURCE_CODES:
        return code
    return None


class _CatalogLike(Protocol):
    def resolve(self, ref: AssetRef, *, verify: bool = True) -> Any: ...


def catalog_source_resolver(catalog: _CatalogLike) -> SourceResolver:
    """Wrap ``CatalogView`` / ``ManifestCatalog`` (anything with ``resolve``).

    ``resolve(ref, verify=True)`` checks identity, content revision, availability,
    root containment and on-disk size/mtime, and returns an object with ``asset``
    and ``path``. The wrapper re-checks the identity it got back so a resolver bug
    cannot silently substitute another asset.
    """

    def resolve(ref: AssetRef, /) -> ResolvedSource:
        resolved = catalog.resolve(ref, verify=True)
        asset = resolved.asset
        if asset.asset_id != ref.asset_id or asset.content_revision != ref.content_revision:
            raise SourceUnavailableError(
                ErrorCode.ASSET_CHANGED, "目录返回的素材身份或版本与请求不一致"
            )
        return ResolvedSource(asset=asset, path=Path(resolved.path))

    return resolve


__all__ = [
    "ResolvedSource",
    "SourceResolver",
    "SourceUnavailableError",
    "catalog_source_resolver",
    "source_error_code",
]
