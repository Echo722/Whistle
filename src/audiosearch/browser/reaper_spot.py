"""Lightweight client for non-destructive Spot-to-REAPER insertion via audiosearch_spot_bridge.lua."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from audiosearch.integrations.reaper.channel import atomic_write, decode_message, encode_message

SPOT_PROTOCOL = "spot1"


@dataclass(frozen=True, slots=True)
class SpotBridgeStatus:
    connected: bool
    project_name: str = ""
    track_name: str = ""
    cursor_sec: float = 0.0


@dataclass(frozen=True, slots=True)
class SpotResult:
    ok: bool
    message: str
    track_name: str = ""
    position: float = 0.0
    length: float = 0.0


def default_spot_channel_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / "AudioSearchTool" / "reaper-spot"


class ReaperSpotClient:
    """File-channel client talking to REAPER's deferred ``audiosearch_spot_bridge.lua``."""

    def __init__(
        self,
        channel_dir: Path | None = None,
        *,
        heartbeat_ttl: float = 3.5,
    ) -> None:
        self._dir = Path(channel_dir) if channel_dir is not None else default_spot_channel_dir()
        self._heartbeat_ttl = max(0.5, float(heartbeat_ttl))

    @property
    def channel_dir(self) -> Path:
        return self._dir

    @property
    def heartbeat_path(self) -> Path:
        return self._dir / "heartbeat.txt"

    def status(self) -> SpotBridgeStatus:
        hb = self.heartbeat_path
        if not hb.exists():
            return SpotBridgeStatus(connected=False)
        try:
            stat = hb.stat()
            fields = decode_message(hb.read_bytes())
        except OSError:
            return SpotBridgeStatus(connected=False)
        if not fields or fields.get("protocol") != SPOT_PROTOCOL:
            return SpotBridgeStatus(connected=False)

        now = time.time()
        try:
            hb_epoch = float(fields.get("epoch", "0"))
        except ValueError:
            hb_epoch = 0.0
        fresh = abs(now - hb_epoch) <= self._heartbeat_ttl or abs(now - stat.st_mtime) <= self._heartbeat_ttl
        if not fresh:
            return SpotBridgeStatus(connected=False)

        try:
            cursor = float(fields.get("cursor", "0"))
        except ValueError:
            cursor = 0.0
        return SpotBridgeStatus(
            connected=True,
            project_name=fields.get("project", ""),
            track_name=fields.get("track_name", ""),
            cursor_sec=cursor,
        )

    def spot(
        self,
        path: Path,
        *,
        start_sec: float = 0.0,
        end_sec: float | None = None,
        pitch_semitones: float = 0.0,
        rate: float = 1.0,
        preserve_pitch: bool = False,
        apply_tone: bool = True,
        move_cursor: bool = True,
        take_name: str | None = None,
        timeout: float = 1.5,
    ) -> SpotResult:
        src = Path(path)
        if not src.exists():
            return SpotResult(ok=False, message=f"文件不存在：{src}")

        inbox = self._dir / "inbox"
        outbox = self._dir / "outbox"
        req_id = uuid4().hex
        cmd_path = inbox / f"{int(time.time() * 1000):013d}-{req_id}.cmd"
        rcpt_path = outbox / f"{req_id}.rcpt"

        fields: dict[str, object] = {
            "protocol": SPOT_PROTOCOL,
            "command": "spot",
            "request_id": req_id,
            "path": str(src),
            "start_sec": max(0.0, float(start_sec)),
            "end_sec": "" if end_sec is None else max(0.0, float(end_sec)),
            "pitch_semitones": max(-24.0, min(24.0, float(pitch_semitones))),
            "rate": max(0.25, min(4.0, float(rate))),
            "preserve_pitch": bool(preserve_pitch),
            "apply_tone": bool(apply_tone),
            "move_cursor": bool(move_cursor),
            "take_name": take_name or src.name,
        }
        try:
            inbox.mkdir(parents=True, exist_ok=True)
            outbox.mkdir(parents=True, exist_ok=True)
            atomic_write(cmd_path, encode_message(fields))
        except PermissionError as exc:
            return SpotResult(
                ok=False,
                message=f"无权限写入 REAPER 通道目录（可能是当前进程处于低完整性/沙箱权限）：{exc.filename or self._dir}",
            )
        except OSError as exc:
            return SpotResult(ok=False, message=f"写入 REAPER 通道失败：{exc}")

        deadline = time.monotonic() + max(0.05, float(timeout))
        while time.monotonic() < deadline:
            if rcpt_path.exists():
                try:
                    data = decode_message(rcpt_path.read_bytes())
                except OSError:
                    data = None
                if data is not None:
                    try:
                        rcpt_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    ok = data.get("ok") == "1"
                    try:
                        pos = float(data.get("position", "0"))
                    except ValueError:
                        pos = 0.0
                    try:
                        length = float(data.get("length", "0"))
                    except ValueError:
                        length = 0.0
                    return SpotResult(
                        ok=ok,
                        message=data.get("message", ""),
                        track_name=data.get("track_name", ""),
                        position=pos,
                        length=length,
                    )
            time.sleep(0.015)

        try:
            cmd_path.unlink(missing_ok=True)
        except OSError:
            pass
        return SpotResult(
            ok=False,
            message="未检测到 REAPER 响应（请确认 REAPER 已启动并加载 audiosearch_spot_bridge.lua）",
        )


__all__ = [
    "ReaperSpotClient",
    "SPOT_PROTOCOL",
    "SpotBridgeStatus",
    "SpotResult",
    "default_spot_channel_dir",
]
