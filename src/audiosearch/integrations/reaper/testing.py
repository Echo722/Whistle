"""Test-only controls for an isolated instance (``test_commands=1`` in its config).

These commands exist so the real-instance test can inspect items, undo exactly
the block it created, change tracks/projects, and send raw commands that skip
the Python-side checks (to prove the Lua side re-checks). They are refused by
the Lua script unless the test profile enabled them, and they are not part of
the ``PlaybackBridge`` protocol.
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

from .bridge import ReaperBridge
from .channel import PROTOCOL


class ReaperTestControl:
    def __init__(self, bridge: ReaperBridge) -> None:
        self.bridge = bridge

    def raw(
        self, fields: dict[str, object], *, timeout: float = 5.0, ttl: float = 10.0
    ) -> dict[str, str] | None:
        """Send one message as-is (defaults filled in) and wait for its receipt."""
        bridge = self.bridge
        channel = bridge._channel  # noqa: SLF001 - same package, test helper
        if channel is None:
            raise RuntimeError("bridge not connected")
        status = bridge.status()
        now = datetime.now(UTC)
        request_id = fields.get("request_id") or str(uuid.uuid4())
        message: dict[str, object] = {
            "protocol": PROTOCOL,
            "token": bridge._token,  # noqa: SLF001
            "request_id": request_id,
            "instance_id": status.reaper_instance_id or "",
            "session_id": str(status.session_id or ""),
            "issued_epoch": now.timestamp(),
            "expires_epoch": (now + timedelta(seconds=ttl)).timestamp(),
        }
        message.update(fields)
        path = channel.submit(UUID(str(request_id)), message)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            receipt = channel.read_receipt(UUID(str(request_id)))
            if receipt is not None:
                return receipt
            time.sleep(0.01)
        channel.withdraw(path)
        return channel.read_receipt(UUID(str(request_id)))

    def inspect(self, project_id: str, item_id: str = "", track_id: str = "") -> dict[str, str]:
        return self._ok(
            {
                "command": "test.inspect",
                "project_id": project_id,
                "item_id": item_id,
                "track_id": track_id,
            }
        )

    def undo(self, project_id: str, expect_desc: str) -> dict[str, str] | None:
        return self.raw(
            {"command": "test.undo", "project_id": project_id, "expect_desc": expect_desc}
        )

    def add_track(self, project_id: str, name: str) -> str:
        return self._ok({"command": "test.add_track", "project_id": project_id, "name": name})[
            "track_id"
        ]

    def delete_track(self, project_id: str, track_id: str) -> dict[str, str]:
        return self._ok(
            {"command": "test.delete_track", "project_id": project_id, "track_id": track_id}
        )

    def open_project(self, path: Path) -> dict[str, str]:
        return self._ok({"command": "test.open_project", "path": str(path)})

    def _ok(self, fields: dict[str, object]) -> dict[str, str]:
        receipt = self.raw(fields)
        if receipt is None or receipt.get("status") != "completed":
            raise RuntimeError(f"test command failed: {fields.get('command')}: {receipt}")
        return receipt


__all__ = ["ReaperTestControl"]
