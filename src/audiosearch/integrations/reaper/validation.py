"""Pure pre-execution checks shared by the fake and the real bridge.

Each check raises ``CommandRejected`` (converted to a rejected receipt) and does
no I/O except calling the injected resolver. The real bridge runs the same
checks before dispatch and the REAPER side re-checks session/expiry/target
right before executing.
"""

from __future__ import annotations

import hashlib
import math
from datetime import datetime, timedelta

from pydantic import BaseModel

from audiosearch.contracts.models import (
    AssetRef,
    CommandContext,
    ErrorCode,
    PlaybackOptions,
    SourceRange,
    validate_source_range,
)

from .adapter import MAX_CLOCK_SKEW_SECONDS
from .errors import BridgeUsageError, CommandRejected
from .sources import ResolvedSource, SourceResolver, source_error_code
from .status import BridgeCommand, BridgeStatus, PlaybackOptionName

_DEFAULT_OPTIONS = PlaybackOptions()
_OPTION_FIELDS: tuple[PlaybackOptionName, ...] = (
    "gain_db",
    "play_rate",
    "pitch_semitones",
    "preserve_pitch",
    "loop",
)


def request_digest(request: BaseModel) -> str:
    """Stable digest of a request, for request_id de-duplication."""
    return hashlib.sha256(request.model_dump_json().encode("utf-8")).hexdigest()


def check_timeout(timeout: float) -> float:
    if isinstance(timeout, bool) or not isinstance(timeout, int | float):
        raise BridgeUsageError("timeout must be a number of seconds")
    if not math.isfinite(timeout) or timeout <= 0:
        raise BridgeUsageError("timeout must be a positive finite number of seconds")
    return float(timeout)


def check_ready(status: BridgeStatus, command: BridgeCommand) -> None:
    if not status.is_ready:
        reason = status.reason or f"桥状态为 {status.state.value}"
        raise CommandRejected(
            ErrorCode.BRIDGE_UNAVAILABLE, f"REAPER 桥不可用，未执行：{reason}", retryable=True
        )
    if not status.supports(command):
        raise CommandRejected(
            ErrorCode.INVALID_REQUEST, f"当前桥没有经过验证的 {command.value} 能力"
        )


def check_session(context: CommandContext, status: BridgeStatus) -> None:
    if context.reaper_instance_id != status.reaper_instance_id:
        raise CommandRejected(
            ErrorCode.TARGET_CHANGED, "命令指向的 REAPER 实例不是当前连接的实例；请刷新桥状态"
        )
    if context.session_id != status.session_id:
        raise CommandRejected(
            ErrorCode.TARGET_CHANGED, "命令属于旧的桥会话（已断线或重启）；请刷新桥状态后重发"
        )


def check_deadline(context: CommandContext, now: datetime) -> None:
    if now >= context.expires_at:
        raise CommandRejected(ErrorCode.REQUEST_EXPIRED, "命令已过期，未执行")
    if context.issued_at > now + timedelta(seconds=MAX_CLOCK_SKEW_SECONDS):
        raise CommandRejected(ErrorCode.INVALID_REQUEST, "命令的发出时间晚于本机当前时间过多")


def remaining_seconds(context: CommandContext, now: datetime, timeout: float) -> float:
    """Wait budget: the smaller of ``timeout`` and the time left before expiry."""
    return max(0.0, min(timeout, (context.expires_at - now).total_seconds()))


def check_options(
    options: PlaybackOptions | None, supported: list[PlaybackOptionName], what: str
) -> None:
    if options is None:
        return
    for name in _OPTION_FIELDS:
        if getattr(options, name) != getattr(_DEFAULT_OPTIONS, name) and name not in supported:
            raise CommandRejected(
                ErrorCode.INVALID_REQUEST, f"{what}：参数 {name} 尚未经过真实验证，不支持非默认值"
            )


def resolve_source(resolver: SourceResolver, ref: AssetRef) -> ResolvedSource:
    try:
        resolved = resolver(ref)
    except CommandRejected:
        raise
    except Exception as exc:  # noqa: BLE001 - mapped to a receipt, nothing executed
        code = source_error_code(exc)
        if code is None:
            raise CommandRejected(
                ErrorCode.INTERNAL_ERROR, f"解析素材源失败（{type(exc).__name__}），未执行"
            ) from exc
        message = getattr(exc, "message", None) or str(exc) or code.value
        raise CommandRejected(code, f"素材不可用，未执行：{message}") from exc
    if not isinstance(resolved, ResolvedSource):
        raise CommandRejected(ErrorCode.INTERNAL_ERROR, "素材解析器返回了无效结果，未执行")
    if (
        resolved.asset.asset_id != ref.asset_id
        or resolved.asset.content_revision != ref.content_revision
    ):
        raise CommandRejected(ErrorCode.ASSET_CHANGED, "素材版本已经变化，未执行")
    return resolved


def effective_range(
    source: ResolvedSource, source_range: SourceRange | None, *, channel_selection: bool
) -> SourceRange | None:
    """Validate the original-file range; None means the whole file."""
    if source_range is None:
        return None
    if source_range.channels is not None and not channel_selection:
        raise CommandRejected(
            ErrorCode.INVALID_REQUEST, "当前桥不支持选择声道；channels 必须为 null（全部声道）"
        )
    try:
        validate_source_range(source.asset, source_range)
    except ValueError as exc:
        raise CommandRejected(ErrorCode.INVALID_REQUEST, f"源区间无效：{exc}") from exc
    return source_range


def check_target(status: BridgeStatus, project_id: str, track_id: str) -> None:
    project = status.project(project_id)
    if project is None:
        raise CommandRejected(
            ErrorCode.TARGET_CHANGED, "目标工程已不存在或已切换内容；请刷新工程/轨道后重选"
        )
    if not track_id:
        # Empty is "no track selected": the bridge will create one, so there is
        # nothing to look up. Only the project still has to exist.
        return
    if project.track(track_id) is None:
        raise CommandRejected(
            ErrorCode.TARGET_CHANGED, "目标轨道已不在该工程中；请刷新工程/轨道后重选"
        )


__all__ = [
    "check_deadline",
    "check_options",
    "check_ready",
    "check_session",
    "check_target",
    "check_timeout",
    "effective_range",
    "remaining_seconds",
    "request_digest",
    "resolve_source",
]
