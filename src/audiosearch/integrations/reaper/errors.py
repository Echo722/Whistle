"""Error model for bridge adapters: three separate kinds of failure.

1. **Business rejection** — the command was checked and NOT executed (wrong
   session, expired, asset offline/changed, target gone, unsupported option...).
   Returned as ``CommandReceipt(status="rejected", error=...)``; never raised.
   ``error.retryable`` is True only when the same request may be sent again
   unchanged and nothing has been executed (e.g. bridge not connected yet).
2. **Uncertain execution** — the command may or may not have taken effect (sent
   but no reply before the timeout, link lost after dispatch). Returned as
   ``CommandReceipt(status="uncertain", error.code="execution_uncertain",
   retryable=False)``; the caller must query ``receipt(request_id)`` or inspect the
   project, and must NOT resend automatically.
3. **Structural / usage errors** — programming mistakes (wrong request type,
   non-positive timeout, connect after close). Raised as ``BridgeUsageError`` /
   ``BridgeClosedError``; they never reach REAPER.
"""

from __future__ import annotations

from collections.abc import Iterable
from uuid import UUID

from audiosearch.contracts.models import CommandReceipt, ErrorCode, ErrorDetail

# Codes a rejected receipt may carry. EXECUTION_UNCERTAIN is reserved for
# "uncertain" receipts and CURSOR_INVALID / MODEL_UNAVAILABLE never apply.
REJECTION_CODES = frozenset(
    {
        ErrorCode.INVALID_REQUEST,
        ErrorCode.ASSET_NOT_FOUND,
        ErrorCode.ASSET_OFFLINE,
        ErrorCode.ASSET_CHANGED,
        ErrorCode.BRIDGE_UNAVAILABLE,
        ErrorCode.REQUEST_EXPIRED,
        ErrorCode.TARGET_CHANGED,
        ErrorCode.INTERNAL_ERROR,
    }
)


class BridgeError(Exception):
    """Base class for errors that are raised (structural), never for receipts."""


class BridgeUsageError(BridgeError, TypeError):
    """The caller used the adapter incorrectly; nothing was sent."""


class BridgeClosedError(BridgeError, RuntimeError):
    """``connect()`` on an adapter that was closed. Create a new adapter instead."""


class CommandRejected(Exception):  # noqa: N818 - control flow, converted to a receipt
    """Internal: raised by validation helpers, converted into a rejected receipt."""

    def __init__(self, code: ErrorCode, message: str, *, retryable: bool = False) -> None:
        if code not in REJECTION_CODES:
            raise ValueError(f"{code} is not a rejection code")
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable

    def receipt(self, request_id: UUID) -> CommandReceipt:
        return rejected(request_id, self.code, self.message, retryable=self.retryable)


def rejected(
    request_id: UUID, code: ErrorCode, message: str, *, retryable: bool = False
) -> CommandReceipt:
    if code not in REJECTION_CODES:
        raise ValueError(f"{code} is not a rejection code")
    return CommandReceipt(
        request_id=request_id,
        status="rejected",
        error=ErrorDetail(code=code, message=message, retryable=retryable),
    )


def uncertain(request_id: UUID, message: str) -> CommandReceipt:
    return CommandReceipt(
        request_id=request_id,
        status="uncertain",
        error=ErrorDetail(code=ErrorCode.EXECUTION_UNCERTAIN, message=message, retryable=False),
    )


def completed(request_id: UUID, created_item_ids: Iterable[str] = ()) -> CommandReceipt:
    return CommandReceipt(
        request_id=request_id, status="completed", created_item_ids=list(created_item_ids)
    )


__all__ = [
    "REJECTION_CODES",
    "BridgeClosedError",
    "BridgeError",
    "BridgeUsageError",
    "CommandRejected",
    "completed",
    "rejected",
    "uncertain",
]
