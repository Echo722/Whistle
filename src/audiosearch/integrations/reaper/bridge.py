"""Real REAPER bridge adapter: ``PlaybackBridge`` over the asb1 file channel.

The adapter never starts REAPER and never runs anything on import or
construction. ``connect()`` reads the bridge config (token) and the status the
Lua side publishes; commands are validated here first (same checks as the fake
bridge), then queued for the Lua script, which checks token, instance, session,
expiry and target again right before executing.

Timeout semantics (see docs/contracts/reaper-adapter-v0.md):

* wait for a receipt until ``min(timeout, expires_at - now)``;
* still queued at that point -> the command is withdrawn (deleted) and the
  result is a rejection: ``bridge_unavailable`` (retryable) or, if the command
  expired, ``request_expired``. A withdrawn command can never run;
* already claimed by REAPER -> wait up to ``claim_grace`` more; then
  ``uncertain``. ``receipt(request_id)`` later reads the real outcome; nothing
  is resent automatically.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel

from audiosearch.contracts.models import (
    AssetRef,
    CommandReceipt,
    ErrorCode,
    InsertRequest,
    PlaybackSeekRequest,
    PlaybackStartRequest,
    PlaybackStopRequest,
    SourceRange,
)
from .adapter import (
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_HEARTBEAT_TIMEOUT,
    DEFAULT_INSERT_TIMEOUT,
    DEFAULT_PREVIEW_TIMEOUT,
)
from .channel import CONFIG_NAME, PROTOCOL, FileChannel, read_config
from .errors import (
    REJECTION_CODES,
    BridgeClosedError,
    BridgeUsageError,
    CommandRejected,
    completed,
    rejected,
    uncertain,
)
from .source_guard import SourceGuard, guard_source
from .sources import SourceResolver, SourceUnavailableError
from .status import (
    BridgeCapabilities,
    BridgeCommand,
    BridgeStatus,
    ConnectionState,
    PlaybackSnapshot,
    ProjectInfo,
    TrackInfo,
)
from .validation import (
    check_deadline,
    check_options,
    check_ready,
    check_session,
    check_target,
    check_timeout,
    effective_range,
    request_digest,
    resolve_source,
)

_COMMAND_NAMES = {
    BridgeCommand.PLAYBACK_START: "preview.start",
    BridgeCommand.PLAYBACK_STOP: "preview.stop",
    BridgeCommand.PLAYBACK_SEEK: "preview.seek",
    BridgeCommand.INSERT: "insert",
}


@dataclass(frozen=True, slots=True)
class ReaperBridgeConfig:
    """Where the bridge lives. Building one performs no I/O."""

    config_path: Path  # <test resource dir>/audiosearch-bridge.cfg (holds the token)
    heartbeat_timeout: float = DEFAULT_HEARTBEAT_TIMEOUT
    poll_interval: float = 0.01
    claim_grace: float = 0.5

    @classmethod
    def for_resource_dir(cls, resource_dir: Path, **kwargs: Any) -> ReaperBridgeConfig:
        return cls(config_path=Path(resource_dir) / CONFIG_NAME, **kwargs)


@dataclass
class _Entry:
    digest: str
    receipt: CommandReceipt
    pending: bool = False  # uncertain: may still be resolved from the outbox


# At most this many claimed-but-unresolved commands may hold a source guard. When
# the limit is reached, new preview/insert commands are rejected BEFORE sending
# (retryable); a guard of a command that may still run is never released early.
MAX_PENDING_GUARDS = 8
_PENDING_POLL_SECONDS = 0.05


@dataclass
class _Pending:
    """Source guard owned by a claimed command that has no terminal result yet."""

    command: BridgeCommand
    guard: SourceGuard
    session: str
    fields: dict[str, object]
    after: Callable[[CommandReceipt], None]


class ReaperBridge:
    """``PlaybackBridge`` for a REAPER instance running ``audiosearch_bridge.lua``."""

    def __init__(
        self,
        resolver: SourceResolver,
        config: ReaperBridgeConfig,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not callable(resolver):
            raise BridgeUsageError("resolver must be callable: AssetRef -> ResolvedSource")
        check_timeout(config.heartbeat_timeout)
        check_timeout(config.poll_interval)
        self._resolver = resolver
        self._config = config
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lock = threading.RLock()
        self._closed = False
        self._connected = False
        self._channel: FileChannel | None = None
        self._token: str | None = None
        self._instance_id: str | None = None
        self._reason: str | None = "尚未连接"
        self._raw: dict[str, str] | None = None
        self._journal: dict[UUID, _Entry] = {}
        self._inflight: dict[UUID, threading.Event] = {}
        self._ranges: dict[tuple[str, int], SourceRange | None] = {}
        self._max_generation: dict[str, int] = {}
        self._details: dict[UUID, dict[str, str]] = {}
        self._drop_replies = 0
        self._preview_guard: SourceGuard | None = None
        self._preview_guard_key: tuple[str, int] | None = None  # (session, generation)
        self._preview_guard_timer: threading.Timer | None = None
        self._pending: dict[UUID, _Pending] = {}
        self._watcher: threading.Thread | None = None

    # ------------------------------------------------------------- lifecycle
    @property
    def kind(self) -> Literal["reaper"]:
        return "reaper"

    def __enter__(self) -> ReaperBridge:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def connect(self, *, timeout: float = DEFAULT_CONNECT_TIMEOUT) -> BridgeStatus:
        budget = check_timeout(timeout)
        with self._lock:
            if self._closed:
                raise BridgeClosedError("adapter is closed; create a new one")
            try:
                cfg = read_config(self._config.config_path)
            except (OSError, ValueError):
                self._connected = False
                self._reason = "未找到 REAPER 桥配置（测试实例尚未准备或路径错误）"
                return self._snapshot(None)
            if cfg.get("protocol") != PROTOCOL or not cfg.get("token") or not cfg.get("channel"):
                self._connected = False
                self._reason = "REAPER 桥配置无效"
                return self._snapshot(None)
            self._token = cfg["token"]
            self._instance_id = cfg.get("instance_id")
            self._channel = FileChannel(Path(cfg["channel"]))
        deadline = time.monotonic() + budget
        while True:
            raw = self._channel.read_status()
            with self._lock:
                problem = self._raw_problem(raw)
                if problem is None:
                    self._connected = True
                    self._reason = None
                    self._raw = raw
                    return self._snapshot(raw)
                if time.monotonic() >= deadline:
                    self._connected = False
                    self._reason = problem
                    self._raw = raw
                    return self._snapshot(raw)
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    def close(self) -> None:
        """Idempotent. Best effort: stops the preview THIS adapter started (bounded
        0.5 s, no retry); other scripts' previews and project playback are untouched.

        Source guards of claimed commands without a terminal result are NOT released
        here: a background watcher keeps them until terminal evidence arrives (see
        ``_reap_pending``) or the process exits."""
        with self._lock:
            if self._closed:
                return
            stop = self._stop_own_preview_fields()
            channel = self._channel
            self._closed = True
            self._connected = False
            self._reason = "适配器已关闭"
        if stop is not None and channel is not None:
            request_id, fields = stop
            try:
                self._exchange(channel, request_id, fields, 0.5, False)
            except OSError:
                pass
        with self._lock:
            self._set_preview_guard(None, 0.0)

    def pending_guards(self) -> list[UUID]:
        """Request IDs whose source guard is held until a terminal result (diagnostic)."""
        with self._lock:
            return list(self._pending)

    # ---------------------------------------------------------- source guards
    def _set_preview_guard(
        self, guard: SourceGuard | None, seconds: float, key: tuple[str, int] | None = None
    ) -> None:
        """Keep ``guard`` while its preview is current; release the previous one."""
        old, self._preview_guard = self._preview_guard, guard
        self._preview_guard_key = key if guard is not None else None
        timer, self._preview_guard_timer = self._preview_guard_timer, None
        if timer is not None:
            timer.cancel()
        if old is not None and old is not guard:
            old.release()
        if guard is not None:
            timer = threading.Timer(seconds, self._expire_preview_guard, args=(guard,))
            timer.daemon = True
            self._preview_guard_timer = timer
            timer.start()

    def _expire_preview_guard(self, guard: SourceGuard) -> None:
        with self._lock:
            if self._preview_guard is guard:
                self._preview_guard = None
                self._preview_guard_key = None
                self._preview_guard_timer = None
        guard.release()

    def _settle(
        self,
        command: BridgeCommand,
        receipt: CommandReceipt,
        guard: SourceGuard | None,
        session: str,
        fields: dict[str, object],
        after: Callable[[CommandReceipt], None],
    ) -> None:
        """Apply a TERMINAL receipt (sync or late) and decide what happens to ``guard``.

        Called with the lock held. A completed preview start becomes the current
        preview only if its generation is newer than every generation already
        accepted in that session; a late older one neither rolls the generation
        back nor touches the newer preview's guard (its own guard is released:
        REAPER has already replaced that preview).
        """
        if receipt.status == "completed":
            if command is BridgeCommand.PLAYBACK_START:
                generation = int(fields["generation"])
                current = self._max_generation.get(session)
                if current is None or generation > current:
                    after(receipt)
                    if guard is not None:
                        self._set_preview_guard(
                            guard, _preview_guard_seconds(fields), (session, generation)
                        )
                        guard = None
            else:
                after(receipt)
                if command is BridgeCommand.PLAYBACK_STOP and self._preview_guard_key == (
                    session,
                    int(fields["generation"]),
                ):
                    self._set_preview_guard(None, 0.0)
        if guard is not None:
            guard.release()

    def _hold_pending(self, request_id: UUID, pending: _Pending) -> None:
        """Called with the lock held: ``request_id`` now owns ``pending.guard``."""
        self._pending[request_id] = pending
        if self._watcher is None or not self._watcher.is_alive():
            self._watcher = threading.Thread(
                target=self._watch_pending, name="asb-pending-guards", daemon=True
            )
            self._watcher.start()

    def _watch_pending(self) -> None:
        while True:
            time.sleep(_PENDING_POLL_SECONDS)
            with self._lock:
                self._reap_pending()
                if not self._pending:
                    self._watcher = None
                    return

    def _reap_pending(self, only: UUID | None = None) -> None:
        """Release guards of pending commands that have terminal evidence. Lock held.

        Terminal evidence (anything else, including elapsed time, is not):
        * a receipt for that request_id in ``outbox`` - ``completed`` / ``rejected``,
          or the script's own ``uncertain`` (it has finished with the command and
          never runs that request_id again);
        * a fresh, valid heartbeat from a DIFFERENT bridge session: a claimed command
          lives only in the script instance that claimed it, so after that instance
          is gone it can no longer be executed.
        """
        channel = self._channel
        if channel is None or not self._pending:
            return
        raw_status = None
        for request_id in [only] if only is not None else list(self._pending):
            pending = self._pending.get(request_id)
            if pending is None:
                continue
            raw = channel.read_receipt(request_id)
            if (
                raw is not None
                and raw.get("request_id") == str(request_id)
                and raw.get("status") in {"completed", "rejected", "uncertain"}
            ):
                receipt = self._to_receipt(request_id, raw)
                del self._pending[request_id]
                entry = self._journal.get(request_id)
                if entry is not None:
                    entry.receipt = receipt
                    entry.pending = receipt.status == "uncertain"
                self._settle(
                    pending.command,
                    receipt,
                    pending.guard,
                    pending.session,
                    pending.fields,
                    pending.after,
                )
                continue
            if raw_status is None:
                raw_status = channel.read_status(attempts=1) or {}
            session = raw_status.get("session_id")
            if session and session != pending.session and self._raw_problem(raw_status) is None:
                del self._pending[request_id]  # journal entry stays uncertain
                pending.guard.release()

    def _stop_own_preview_fields(self) -> tuple[UUID, dict[str, object]] | None:
        raw = self._raw
        if not self._connected or raw is None or self._token is None:
            return None
        session = raw.get("session_id", "")
        generation = self._max_generation.get(session)
        if generation is None:  # this adapter never started a preview in this session
            return None
        request_id = uuid4()
        now = time.time()
        return request_id, {
            "protocol": PROTOCOL,
            "token": self._token,
            "command": _COMMAND_NAMES[BridgeCommand.PLAYBACK_STOP],
            "request_id": str(request_id),
            "instance_id": raw.get("instance_id", ""),
            "session_id": session,
            "issued_epoch": now,
            "expires_epoch": now + 1.0,
            "generation": generation,
        }

    def status(self) -> BridgeStatus:
        with self._lock:
            if self._channel is not None and not self._closed:
                raw = self._channel.read_status(attempts=1)
                if raw is not None:
                    self._raw = raw
            return self._snapshot(self._raw)

    def refresh(self, *, timeout: float = DEFAULT_CONNECT_TIMEOUT) -> BridgeStatus:
        budget = check_timeout(timeout)
        with self._lock:
            channel = self._channel
        if channel is None or self._closed:
            return self.status()
        raw = channel.read_status(attempts=max(1, int(min(budget, 0.2) / 0.01)))
        with self._lock:
            if raw is not None:
                self._raw = raw
            return self._snapshot(self._raw)

    # -------------------------------------------------------------- commands
    def start_playback(
        self, request: PlaybackStartRequest, *, timeout: float = DEFAULT_PREVIEW_TIMEOUT
    ) -> CommandReceipt:
        return self._run(BridgeCommand.PLAYBACK_START, PlaybackStartRequest, request, timeout)

    def stop_playback(
        self, request: PlaybackStopRequest, *, timeout: float = DEFAULT_PREVIEW_TIMEOUT
    ) -> CommandReceipt:
        if isinstance(request, PlaybackSeekRequest):
            raise BridgeUsageError("stop_playback needs a PlaybackStopRequest, not a seek")
        return self._run(BridgeCommand.PLAYBACK_STOP, PlaybackStopRequest, request, timeout)

    def seek_playback(
        self, request: PlaybackSeekRequest, *, timeout: float = DEFAULT_PREVIEW_TIMEOUT
    ) -> CommandReceipt:
        return self._run(BridgeCommand.PLAYBACK_SEEK, PlaybackSeekRequest, request, timeout)

    def insert(
        self, request: InsertRequest, *, timeout: float = DEFAULT_INSERT_TIMEOUT
    ) -> CommandReceipt:
        return self._run(BridgeCommand.INSERT, InsertRequest, request, timeout)

    def receipt(self, request_id: UUID) -> CommandReceipt | None:
        with self._lock:
            entry = self._journal.get(request_id)
            channel = self._channel
        if entry is None:
            return None
        with self._lock:
            if request_id in self._pending:
                self._reap_pending(request_id)
        if entry.pending and channel is not None:
            raw = channel.read_receipt(request_id)
            if raw is not None:
                resolved = self._to_receipt(request_id, raw)
                with self._lock:
                    entry.receipt = resolved
                    entry.pending = resolved.status == "uncertain"
        return entry.receipt

    def receipt_details(self, request_id: UUID) -> dict[str, str] | None:
        """Diagnostic read-back from REAPER for one request (not part of the protocol).

        For a completed insert this includes item position/length, take start
        offset, source file and track GUID as REAPER reports them.
        """
        with self._lock:
            details = self._details.get(request_id)
            return dict(details) if details is not None else None

    def simulate_lost_reply_once(self) -> None:
        """TEST ONLY: ignore the next receipt as if the reply were lost in transit.

        The command still runs in REAPER; the caller gets ``uncertain`` and must
        resolve it with ``receipt()``. Used by the real-instance test tool.
        """
        with self._lock:
            self._drop_replies += 1

    # --------------------------------------------------------------- status
    def _raw_problem(self, raw: dict[str, str] | None) -> str | None:
        if raw is None:
            return "未收到 REAPER 桥状态（测试实例未启动或脚本未加载）"
        if raw.get("protocol") != PROTOCOL:
            return "REAPER 桥协议版本不符"
        if raw.get("instance_id") != self._instance_id:
            return "状态来自另一个 REAPER 实例"
        if raw.get("state") != "running":
            return "REAPER 桥脚本已退出"
        age = time.time() - _float(raw.get("epoch"), 0.0)
        if age > self._config.heartbeat_timeout:
            return "REAPER 桥心跳超时"
        return None

    def _snapshot(self, raw: dict[str, str] | None) -> BridgeStatus:
        now = self._clock()
        if self._closed:
            state = ConnectionState.CLOSED
        elif not self._connected or raw is None:
            state = ConnectionState.DISCONNECTED
        else:
            problem = self._raw_problem(raw)
            if problem is None:
                state = ConnectionState.CONNECTED
            elif problem == "REAPER 桥心跳超时":
                state = ConnectionState.STALE
            else:
                state = ConnectionState.DISCONNECTED
                self._reason = problem
        live = state in {ConnectionState.CONNECTED, ConnectionState.STALE} and raw is not None
        reason = None if state is ConnectionState.CONNECTED else self._reason
        if state is ConnectionState.STALE:
            reason = "REAPER 桥心跳超时，等待刷新"
        heartbeat = _epoch_dt(raw.get("epoch")) if raw else None
        session = _uuid(raw.get("session_id")) if live and raw else None
        return BridgeStatus(
            kind="reaper",
            state=state,
            reaper_instance_id=raw.get("instance_id") if live and raw else None,
            session_id=session,
            connected_at=_epoch_dt(raw.get("started_epoch")) if live and raw else None,
            last_heartbeat_at=heartbeat if live else None,
            heartbeat_timeout_seconds=self._config.heartbeat_timeout,
            capabilities=_capabilities(raw) if live and raw else None,
            projects=_projects(raw) if live and raw else [],
            discovery_version=int(_float(raw.get("discovery_version"), 0)) if raw else 0,
            playback=_playback(raw) if live and raw else PlaybackSnapshot(),
            reason=reason,
            observed_at=now,
        )

    # -------------------------------------------------------------- execution
    def _run(
        self, command: BridgeCommand, model: type[BaseModel], request: Any, timeout: float
    ) -> CommandReceipt:
        if not isinstance(request, model):
            raise BridgeUsageError(f"{command.value} needs {model.__name__}")
        budget = check_timeout(timeout)
        request_id: UUID = request.context.request_id
        digest = request_digest(request)
        while True:
            with self._lock:
                entry = self._journal.get(request_id)
                if entry is not None:
                    if entry.digest != digest:
                        return rejected(
                            request_id,
                            ErrorCode.INVALID_REQUEST,
                            "request_id 已用于另一条内容不同的命令，未执行",
                        )
                    break
                waiter = self._inflight.get(request_id)
                if waiter is None:
                    self._inflight[request_id] = threading.Event()
                    return self._execute(command, request, request_id, digest, budget)
            waiter.wait()
        known = self.receipt(request_id)
        assert known is not None
        return known

    def _execute(
        self,
        command: BridgeCommand,
        request: Any,
        request_id: UUID,
        digest: str,
        budget: float,
    ) -> CommandReceipt:
        # Called with self._lock held; releases it while waiting on REAPER.
        receipt: CommandReceipt | None = None
        journal = True
        pending = False
        guard: SourceGuard | None = None
        try:
            try:
                status = self._snapshot(self._fresh_raw())
                check_ready(status, command)
                check_session(request.context, status)
                check_deadline(request.context, status.observed_at)
                fields, after, guard = self._prepare(command, request, status)
            except CommandRejected as exc:
                journal = not exc.retryable
                receipt = exc.receipt(request_id)
                return receipt
            assert self._channel is not None and self._token is not None
            channel = self._channel
            expires = request.context.expires_at
            fields = {
                "protocol": PROTOCOL,
                "token": self._token,
                "command": _COMMAND_NAMES[command],
                "request_id": str(request_id),
                "instance_id": request.context.reaper_instance_id,
                "session_id": str(request.context.session_id),
                "issued_epoch": request.context.issued_at.timestamp(),
                "expires_epoch": expires.timestamp(),
                **fields,
            }
            drop = self._drop_replies > 0
            if drop:
                self._drop_replies -= 1
            now = self._clock()
            remaining = (expires - now).total_seconds()
            wait = max(0.0, min(budget, remaining))
            expiry_bound = remaining <= budget
            self._lock.release()
            try:
                raw, outcome = self._exchange(channel, request_id, fields, wait, drop)
            finally:
                self._lock.acquire()
            session = str(request.context.session_id)
            if raw is not None and raw.get("request_id") == str(request_id):
                receipt = self._to_receipt(request_id, raw)
                # Terminal answer from the script: settle (keeps or releases the guard).
                self._settle(command, receipt, guard, session, fields, after)
                guard = None
                pending = receipt.status == "uncertain"
            elif raw is not None:
                receipt = self._to_receipt(request_id, raw)  # mismatched: uncertain
                pending = True
                if guard is not None:
                    self._hold_pending(request_id, _Pending(command, guard, session, fields, after))
                    guard = None
            elif outcome == "withdrawn":
                if expiry_bound:
                    receipt = rejected(
                        request_id,
                        ErrorCode.REQUEST_EXPIRED,
                        "命令到期前 REAPER 未取走，已撤回，未执行",
                    )
                else:
                    journal = False
                    receipt = rejected(
                        request_id,
                        ErrorCode.BRIDGE_UNAVAILABLE,
                        "REAPER 未在时限内取走命令，已撤回，未执行",
                        retryable=True,
                    )
            else:
                pending = True
                receipt = uncertain(
                    request_id,
                    "REAPER 已取走命令但未在时限内回执，无法确定是否执行；请查询回执或核对工程",
                )
                if guard is not None:
                    # It may still run: the source stays pinned until terminal evidence.
                    self._hold_pending(request_id, _Pending(command, guard, session, fields, after))
                    guard = None
            return receipt
        finally:
            if guard is not None:
                guard.release()
            if receipt is not None and journal:
                self._journal[request_id] = _Entry(digest, receipt, pending)
            event = self._inflight.pop(request_id, None)
            if event is not None:
                event.set()

    def _exchange(
        self,
        channel: FileChannel,
        request_id: UUID,
        fields: dict[str, object],
        wait: float,
        drop: bool,
    ) -> tuple[dict[str, str] | None, str]:
        path = channel.submit(request_id, fields)
        poll = self._config.poll_interval
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            if not drop:
                raw = channel.read_receipt(request_id)
                if raw is not None:
                    return raw, "answered"
            time.sleep(min(poll, max(0.0, deadline - time.monotonic())))
        if channel.withdraw(path) == "withdrawn":
            return None, "withdrawn"
        if not drop:
            grace = time.monotonic() + self._config.claim_grace
            while time.monotonic() < grace:
                raw = channel.read_receipt(request_id)
                if raw is not None:
                    return raw, "answered"
                time.sleep(poll)
        return None, "claimed"

    def _fresh_raw(self) -> dict[str, str] | None:
        if self._closed or not self._connected or self._channel is None:
            return self._raw
        raw = self._channel.read_status()
        if raw is not None:
            self._raw = raw
        return self._raw

    def _to_receipt(self, request_id: UUID, raw: dict[str, str]) -> CommandReceipt:
        self._details[request_id] = dict(raw)
        status = raw.get("status")
        if raw.get("request_id") != str(request_id):
            return uncertain(request_id, "回执与请求不匹配，需核对工程现场")
        if status == "completed":
            item = raw.get("item_id", "")
            return completed(request_id, [item] if item else [])
        if status == "rejected":
            try:
                code = ErrorCode(raw.get("code", ""))
            except ValueError:
                code = ErrorCode.INTERNAL_ERROR
            if code not in REJECTION_CODES:
                code = ErrorCode.INTERNAL_ERROR
            return rejected(request_id, code, raw.get("message") or "REAPER 拒绝执行")
        return uncertain(request_id, raw.get("message") or "REAPER 报告执行状态不确定")

    # ------------------------------------------------------------- preparing
    def _check_pending_capacity(self) -> None:
        self._reap_pending()
        if len(self._pending) >= MAX_PENDING_GUARDS:
            raise CommandRejected(
                ErrorCode.BRIDGE_UNAVAILABLE,
                f"已有 {len(self._pending)} 条命令结果未知、其源文件仍受保护；"
                "请先查询这些命令的回执，暂不发送新命令",
                retryable=True,
            )

    def _generation_floor(self, session: str, status: BridgeStatus) -> int | None:
        known = [
            g
            for g in (self._max_generation.get(session), status.playback.generation)
            if g is not None
        ]
        return max(known) if known else None

    def _prepare(
        self, command: BridgeCommand, request: Any, status: BridgeStatus
    ) -> tuple[dict[str, object], Callable[[CommandReceipt], None], SourceGuard | None]:
        caps = status.capabilities or BridgeCapabilities()
        session = str(request.context.session_id)
        if command is BridgeCommand.PLAYBACK_START:
            current = self._generation_floor(session, status)
            if current is not None and request.generation <= current:
                raise CommandRejected(
                    ErrorCode.REQUEST_EXPIRED, "预览代次已被更新的预览取代，未执行"
                )
            if request.output_project_id is not None and not caps.output_to_track:
                raise CommandRejected(
                    ErrorCode.INVALID_REQUEST, "当前桥不支持通过工程轨道试听；请不指定输出工程/轨道"
                )
            check_options(request.options, caps.playback_options, "试听")
            source = resolve_source(self._resolver, request.asset)
            source_range = effective_range(
                source, request.source_range, channel_selection=caps.channel_selection
            )
            self._check_pending_capacity()
            guard = _guard(source.path, source.asset)
            fields = _source_fields(source.path, request.asset, source.asset, guard)
            fields.update(
                generation=request.generation,
                start=source_range.start_seconds if source_range else 0.0,
                stop_at=source_range.end_seconds if source_range else "",
            )

            def after(_receipt: CommandReceipt) -> None:
                self._max_generation[session] = request.generation
                self._ranges[(session, request.generation)] = source_range

            return fields, after, guard
        if command in (BridgeCommand.PLAYBACK_STOP, BridgeCommand.PLAYBACK_SEEK):
            current = self._generation_floor(session, status)
            if current is None or request.generation > current:
                raise CommandRejected(ErrorCode.INVALID_REQUEST, "未知的预览代次，未执行")
            if request.generation < current:
                raise CommandRejected(
                    ErrorCode.REQUEST_EXPIRED, "该预览代次已被更新的预览取代，未影响当前预览"
                )
            fields: dict[str, object] = {"generation": request.generation}
            if command is BridgeCommand.PLAYBACK_SEEK:
                position = request.source_position_seconds
                known_range = self._ranges.get((session, request.generation))
                if known_range is not None and not (
                    known_range.start_seconds <= position < known_range.end_seconds
                ):
                    raise CommandRejected(
                        ErrorCode.INVALID_REQUEST,
                        "定位位置不在当前预览的原文件区间 [start, end) 内",
                    )
                fields["position"] = position
            return fields, lambda _receipt: None, None
        # insert
        target = request.target
        check_target(status, target.project_id, target.track_id)
        check_options(request.apply_playback_options, caps.insert_playback_options, "插入")
        source = resolve_source(self._resolver, request.asset)
        source_range = effective_range(
            source, request.source_range, channel_selection=caps.channel_selection
        )
        self._check_pending_capacity()
        guard = _guard(source.path, source.asset)
        fields = _source_fields(source.path, request.asset, source.asset, guard)
        fields.update(
            project_id=target.project_id,
            track_id=target.track_id,
            position=target.timeline_position_seconds,
            offset=source_range.start_seconds if source_range else 0.0,
            length=(source_range.end_seconds - source_range.start_seconds) if source_range else "",
            expect_channels=source.asset.channel_count or "",
            take_name=source.asset.name,
        )
        return fields, lambda _receipt: None, guard


# ---------------------------------------------------------------- parsing
def _guard(path: Path, asset: Any) -> SourceGuard:
    """Pin + re-verify the resolved version right before it is queued for REAPER."""
    try:
        return guard_source(path, asset)
    except SourceUnavailableError as exc:
        raise CommandRejected(exc.code, exc.message) from exc


def _source_fields(path: Path, ref: AssetRef, asset: Any, guard: SourceGuard) -> dict[str, object]:
    return {
        "path": str(path),
        "asset_id": str(ref.asset_id),
        "revision": ref.content_revision,
        "expect_duration": asset.duration_seconds if asset.duration_seconds is not None else "",
        "expect_size": guard.size,  # re-checked by the Lua side before opening the file
    }


def _preview_guard_seconds(fields: dict[str, object]) -> float:
    """How long a preview's source stays pinned at most (range length + 2 s, capped)."""
    start = float(fields.get("start") or 0.0)
    stop_at = fields.get("stop_at")
    duration = fields.get("expect_duration")
    end = (
        float(stop_at)
        if stop_at not in (None, "")
        else (float(duration) if duration not in (None, "") else start + 600.0)
    )
    return min(max(end - start, 0.0) + 2.0, 600.0)


def _float(value: str | None, default: float) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except ValueError:
        return default


def _uuid(value: str | None) -> UUID | None:
    try:
        return UUID(value) if value else None
    except ValueError:
        return None


def _epoch_dt(value: str | None) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        return datetime.fromtimestamp(float(value), tz=UTC)
    except (ValueError, OverflowError, OSError):
        return None


def _capabilities(raw: dict[str, str]) -> BridgeCapabilities:
    commands = [BridgeCommand.INSERT]
    if raw.get("cf_preview") == "1":
        commands[:0] = [
            BridgeCommand.PLAYBACK_START,
            BridgeCommand.PLAYBACK_STOP,
            BridgeCommand.PLAYBACK_SEEK,
        ]
    return BridgeCapabilities(
        commands=commands,
        playback_options=[],
        insert_playback_options=[],
        channel_selection=False,
        output_to_track=False,
        reaper_version=raw.get("reaper_version") or None,
        extensions={
            "SWS": raw.get("sws_version") or None,
            "js_ReaScriptAPI": raw.get("js_version") or None,
        },
    )


def _projects(raw: dict[str, str]) -> list[ProjectInfo]:
    projects = []
    for p in range(1, int(_float(raw.get("nprojects"), 0)) + 1):
        pre = f"p{p}_"
        project_id = raw.get(pre + "id")
        if not project_id:
            continue
        tracks = []
        for t in range(1, int(_float(raw.get(pre + "ntracks"), 0)) + 1):
            track_id = raw.get(f"{pre}t{t}_id")
            if track_id:
                tracks.append(
                    TrackInfo(track_id=track_id, name=raw.get(f"{pre}t{t}_name", ""), index=t - 1)
                )
        projects.append(
            ProjectInfo(
                project_id=project_id,
                project_version=int(_float(raw.get(pre + "state"), 0)),
                name=raw.get(pre + "name", ""),
                is_current=raw.get(pre + "current") == "1",
                edit_cursor_seconds=(
                    _float(raw.get(pre + "ecursor"), None)
                    if raw.get(pre + "ecursor") not in (None, "")
                    else None
                ),
                selected_track_id=raw.get(pre + "sel_track") or None,
                tracks=tracks,
            )
        )
    return projects


def _playback(raw: dict[str, str]) -> PlaybackSnapshot:
    generation = raw.get("pb_generation")
    state = raw.get("pb_state", "idle")
    asset_id = _uuid(raw.get("pb_asset_id"))
    start = raw.get("pb_start")
    end = raw.get("pb_end")
    source_range = None
    if start not in (None, "") and end not in (None, "") and float(end) > float(start):
        source_range = SourceRange(start_seconds=float(start), end_seconds=float(end))
    position = raw.get("pb_position")
    return PlaybackSnapshot(
        generation=int(float(generation)) if generation not in (None, "") else None,
        state=state if state in {"idle", "playing", "stopped"} else "idle",
        request_id=_uuid(raw.get("pb_request_id")),
        asset=AssetRef(asset_id=asset_id, content_revision=raw["pb_revision"])
        if asset_id and raw.get("pb_revision")
        else None,
        source_range=source_range,
        source_position_seconds=max(0.0, float(position)) if position not in (None, "") else None,
    )


__all__ = ["ReaperBridge", "ReaperBridgeConfig"]
