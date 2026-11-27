"""Typed audio failures that map onto the public error codes.

Adapters convert these into ``ErrorResponse``; the codes are kept identical to
``audiosearch.contracts.models.ErrorCode`` so a service layer does not have to
guess which HTTP state a decode failure means.
"""

from audiosearch.contracts.models import ErrorCode


class AudioError(Exception):
    """Base class for audio read failures with a stable public error code."""

    code: ErrorCode = ErrorCode.INTERNAL_ERROR
    retryable: bool = False

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class AssetNotFoundError(AudioError):
    code = ErrorCode.ASSET_NOT_FOUND


class AssetOfflineError(AudioError):
    """The source file is missing, unreadable, or marked offline by the catalog."""

    code = ErrorCode.ASSET_OFFLINE


class AssetChangedError(AudioError):
    """Metadata and the file on disk no longer agree on the content version."""

    code = ErrorCode.ASSET_CHANGED


class SourceRangeError(AudioError):
    """The requested original-file range or channel set cannot be served."""

    code = ErrorCode.INVALID_REQUEST


class UnsupportedFormatError(AudioError):
    code = ErrorCode.INVALID_REQUEST


class DecodeError(AudioError):
    """The container was readable but the audio frames could not be decoded."""

    code = ErrorCode.INTERNAL_ERROR
