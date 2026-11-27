"""File channel shared with ``reaper/audiosearch_bridge.lua`` (protocol ``asb1``).

Layout under the channel directory (inside the task's ``.local`` output):

* ``status.txt`` — written by the Lua side every heartbeat (temp file + rename).
* ``inbox/<seq>-<request_id>.cmd`` — commands, written here atomically
  (``os.replace``). The Lua side *claims* a command by deleting it; the service
  *withdraws* one by deleting it. Exactly one delete can succeed, so a withdrawn
  command can never run afterwards.
* ``outbox/<request_id>.rcpt`` — one receipt per request, never overwritten.
* ``journal/<request_id>.begin`` — written by Lua right before a project write.

Messages are UTF-8 ``key=value`` lines; ``%``, CR and LF in values are
percent-encoded; every message ends with ``end=1`` so a partial read is detected.
Standard library only.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Literal
from uuid import UUID

PROTOCOL = "asb1"
CONFIG_NAME = "audiosearch-bridge.cfg"


def _encode_value(value: object) -> str:
    if isinstance(value, bool):
        text = "1" if value else "0"
    elif isinstance(value, float):
        text = repr(value)
    else:
        text = str(value)
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


_ESCAPE = re.compile(r"%([0-9A-Fa-f]{2})")


def _decode_value(text: str) -> str:
    # Only %, CR and LF are ever escaped (ASCII), so chr() is exact.
    return _ESCAPE.sub(lambda match: chr(int(match.group(1), 16)), text)


def encode_message(fields: Mapping[str, object]) -> bytes:
    lines = []
    for key, value in fields.items():
        if not key or "=" in key or "\n" in key or key == "end":
            raise ValueError(f"invalid message key {key!r}")
        lines.append(f"{key}={_encode_value(value)}\n")
    lines.append("end=1\n")
    return "".join(lines).encode("utf-8")


def decode_message(data: bytes) -> dict[str, str] | None:
    """Parse a message; ``None`` when it is incomplete (no ``end=1``)."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    fields: dict[str, str] = {}
    for line in text.split("\n"):
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key] = _decode_value(value)
    if fields.pop("end", None) != "1":
        return None
    return fields


def atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def read_config(path: Path) -> dict[str, str]:
    data = decode_message(path.read_bytes())
    if data is None:
        raise ValueError(f"bridge config {path.name} is incomplete")
    return data


class FileChannel:
    """Python end of the channel. Cheap to construct: touches nothing."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.inbox = self.root / "inbox"
        self.outbox = self.root / "outbox"
        self.journal = self.root / "journal"
        self.status_path = self.root / "status.txt"
        self._counter = 0

    def ensure(self) -> None:
        for directory in (self.inbox, self.outbox, self.journal):
            directory.mkdir(parents=True, exist_ok=True)

    def read_status(self, *, attempts: int = 3, pause: float = 0.01) -> dict[str, str] | None:
        """Latest complete status; tolerates the Lua side's remove+rename window."""
        for attempt in range(max(1, attempts)):
            try:
                data = self.status_path.read_bytes()
            except OSError:
                data = b""
            parsed = decode_message(data) if data else None
            if parsed is not None:
                return parsed
            if attempt + 1 < attempts:
                time.sleep(pause)
        return None

    def submit(self, request_id: UUID, fields: Mapping[str, object]) -> Path:
        """Queue one command; names sort in submission order."""
        self._counter += 1
        name = f"{time.time_ns():020d}{self._counter % 1000:03d}-{request_id}.cmd"
        path = self.inbox / name
        atomic_write(path, encode_message(fields))
        return path

    @staticmethod
    def withdraw(path: Path) -> Literal["withdrawn", "claimed"]:
        """Delete a queued command. ``claimed``: the bridge took it first."""
        try:
            os.remove(path)
        except FileNotFoundError:
            return "claimed"
        except PermissionError:
            # The bridge has it open right now; treat as claimed (conservative).
            return "claimed"
        return "withdrawn"

    def read_receipt(self, request_id: UUID) -> dict[str, str] | None:
        try:
            data = (self.outbox / f"{request_id}.rcpt").read_bytes()
        except OSError:
            return None
        return decode_message(data)

    def began(self, request_id: UUID) -> bool:
        return (self.journal / f"{request_id}.begin").exists()


__all__ = [
    "CONFIG_NAME",
    "PROTOCOL",
    "FileChannel",
    "atomic_write",
    "decode_message",
    "encode_message",
    "read_config",
]
