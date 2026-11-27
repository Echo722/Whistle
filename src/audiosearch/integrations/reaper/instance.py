"""Prepare, launch and stop an ISOLATED REAPER test instance.

Nothing here runs on import. Isolation rules (GROK-R-001):

* REAPER is started with ``-newinst -cfgfile <test dir>/reaper.ini``; per the
  REAPER command-line documentation ``-cfgfile`` selects an alternate resource
  directory, so the test instance reads/writes its own ini files, UserPlugins
  and Scripts there instead of the production profile.
* The production installation is only *read*: the SWS DLL is copied into the
  test resource directory. ``REAPER.ini``, ``UserPlugins`` and scripts of the
  production profile are never written.
* Launch is refused while any REAPER process is running (it could be the
  user's production session or would compete for the audio device), and also
  when the process list cannot be obtained or trusted.
* ``stop()`` only terminates the process this module started, through its own
  ``Popen`` handle.
* The automated profile forces Dummy Audio. A separate, user-run listening
  profile (``audio="user"``) leaves the device choice to the user.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .channel import CONFIG_NAME, PROTOCOL, encode_message

DEFAULT_REAPER_EXE = Path(r"C:\REAPER\reaper.exe")
SWS_DLL = "reaper_sws-x64.dll"
_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


@dataclass(frozen=True)
class ReaperProcess:
    pid: int
    path: str
    command_line: str | None  # None: not readable (e.g. another user's/elevated process)


class ProcessQueryError(RuntimeError):
    """The process list could not be obtained or trusted; nothing can be concluded."""


_SENTINEL = "ASB-PROC-OK"
_QUERY = (
    "$ErrorActionPreference='Stop';"
    "$p=@(Get-CimInstance Win32_Process -Filter \"Name='reaper.exe'\" | "
    "Select-Object ProcessId,ExecutablePath,CommandLine);"
    f"Write-Output ('{_SENTINEL} ' + $p.Count);"
    "if($p.Count){$p | ConvertTo-Json -Compress}"
)


def _run_query(timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", _QUERY],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        creationflags=_NO_WINDOW,
    )


def list_reaper_processes(
    *,
    runner: Callable[[float], subprocess.CompletedProcess[str]] = _run_query,
    timeout: float = 60.0,
) -> list[ReaperProcess]:
    """Running ``reaper.exe`` processes (Windows, via CIM; read-only).

    An empty list means the query *succeeded* and found none. Any failure
    (non-zero exit, timeout, missing success marker, malformed or inconsistent
    data, non-Windows host) raises ``ProcessQueryError`` instead.
    """
    if sys.platform != "win32" and runner is _run_query:
        raise ProcessQueryError("只能在 Windows 上枚举 REAPER 进程")
    try:
        result = runner(timeout)
    except subprocess.TimeoutExpired as exc:
        raise ProcessQueryError(f"进程查询超时（{timeout:g} s）") from exc
    except OSError as exc:
        raise ProcessQueryError(f"无法运行进程查询：{exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or "").strip()[:200]
        raise ProcessQueryError(f"进程查询失败（返回码 {result.returncode}）：{detail}")
    lines = (result.stdout or "").strip().splitlines()
    if not lines or not lines[0].startswith(_SENTINEL + " "):
        raise ProcessQueryError("进程查询没有返回成功标记，无法确认")
    try:
        count = int(lines[0].split()[1])
    except (IndexError, ValueError) as exc:
        raise ProcessQueryError("进程查询返回的数量无效") from exc
    payload = "\n".join(lines[1:]).strip()
    if count == 0:
        if payload:
            raise ProcessQueryError("进程查询结果自相矛盾（数量为 0 却有数据）")
        return []
    try:
        data = json.loads(payload)
    except ValueError as exc:
        raise ProcessQueryError("进程查询返回的数据不是有效 JSON") from exc
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list) or len(data) != count:
        raise ProcessQueryError("进程查询返回的条目数与数量不符")
    processes = []
    for item in data:
        if not isinstance(item, dict) or not isinstance(item.get("ProcessId"), int):
            raise ProcessQueryError("进程查询条目缺少有效的 ProcessId")
        command_line = item.get("CommandLine")
        processes.append(
            ReaperProcess(
                pid=item["ProcessId"],
                path=item.get("ExecutablePath") or "",
                command_line=command_line if isinstance(command_line, str) else None,
            )
        )
    return processes


# Settings forced into the TEST profile's own reaper.ini (never the production one).
# [audioconfig] mode=4 is REAPER's "Dummy Audio" system (0 Kernel Streaming,
# 1 DirectSound, 2 WaveOut, 3 ASIO, 4 Dummy Audio, 5 WASAPI): the engine runs in
# real time without opening any sound card, so it cannot grab the production
# device (and nothing is audible). Without a device REAPER shows a modal
# first-run prompt that would block startup.
TEST_INI_SETTINGS: dict[str, dict[str, str]] = {
    "reaper": {"audiocloseinactive": "0", "audioclosestop": "0", "faultyproject": ""},
    "audioconfig": {"mode": "4", "dummy_srate": "48000", "dummy_blocksize": "512"},
}
# A user-run LISTENING profile (separate directory) seeds no [audioconfig] at all:
# REAPER's own first-run prompt / Preferences let the user pick the device
# explicitly, and the listening tool verifies at runtime that the engine is
# running on a non-Dummy device before anything is played. No mode number is
# guessed or written for a real device.
LISTEN_INI_SETTINGS: dict[str, dict[str, str]] = {
    "reaper": {"audiocloseinactive": "0", "audioclosestop": "0", "faultyproject": ""},
}
TEST_PROFILE = "reaper-test"
LISTEN_PROFILE = "reaper-listen"


def seed_ini(path: Path, settings: dict[str, dict[str, str]]) -> None:
    """Set ``section/key=value`` lines in an ini file, keeping everything else."""
    lines = (
        path.read_text(encoding="utf-8", errors="surrogateescape").splitlines()
        if path.exists()
        else []
    )
    for section, values in settings.items():
        header = f"[{section}]"
        try:
            start = next(
                i for i, line in enumerate(lines) if line.strip().lower() == header.lower()
            )
        except StopIteration:
            lines += [header]
            start = len(lines) - 1
        end = next(
            (i for i in range(start + 1, len(lines)) if lines[i].startswith("[")), len(lines)
        )
        for key, value in values.items():
            for i in range(start + 1, end):
                if lines[i].split("=", 1)[0] == key:
                    lines[i] = f"{key}={value}"
                    break
            else:
                lines.insert(end, f"{key}={value}")
                end += 1
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", errors="surrogateescape")


def _blank_project(name: str, tracks: list[str]) -> str:
    del name  # display only; REAPER shows the file name
    lines = ['<REAPER_PROJECT 0.1 "7.84/x64" 0']
    for track in tracks:
        guid = "{" + str(uuid.uuid4()).upper() + "}"
        lines += [f"  <TRACK {guid}", f"    NAME {json.dumps(track)}", "  >"]
    lines.append(">")
    return "\n".join(lines) + "\n"


@dataclass
class TestInstance:
    """One isolated REAPER test profile under ``root`` (e.g. ``.local/bridge-001``)."""

    __test__ = False  # not a pytest class

    root: Path
    reaper_exe: Path = DEFAULT_REAPER_EXE
    script: Path = field(
        default_factory=lambda: (
            Path(__file__).resolve().parents[4] / "reaper" / "audiosearch_bridge.lua"
        )
    )
    process_lister: Callable[[], list[ReaperProcess]] = list_reaper_processes
    process: subprocess.Popen | None = None
    launch_id: str = ""
    # "dummy": automated tests, Dummy Audio forced, never a sound card.
    # "user": user-run listening only, in its own profile; the user picks the device.
    audio: Literal["dummy", "user"] = "dummy"
    profile: str = TEST_PROFILE
    minimized: bool = True

    def __post_init__(self) -> None:
        if self.audio not in ("dummy", "user"):
            raise ValueError(f"unknown audio mode {self.audio!r}")
        if self.audio == "user" and self.profile == TEST_PROFILE:
            raise ValueError("真人试听必须使用独立的 profile，不得复用自动化测试 profile")

    @property
    def resource_dir(self) -> Path:
        return self.root / self.profile

    @property
    def ini_path(self) -> Path:
        return self.resource_dir / "reaper.ini"

    @property
    def config_path(self) -> Path:
        return self.resource_dir / CONFIG_NAME

    @property
    def project_path(self) -> Path:
        return self.resource_dir / "projects" / "bridge-test.rpp"

    @property
    def alt_project_path(self) -> Path:
        return self.resource_dir / "projects" / "bridge-test-other.rpp"

    def channel_dir(self) -> Path:
        return self.root / "channel" / self.launch_id

    def prepare(self, *, test_commands: bool = True, copy_sws: bool = True) -> Path:
        """Write a fresh config (new token + channel) and blank projects. No launch."""
        self.root = self.root.resolve()
        self.launch_id = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
        (self.resource_dir / "projects").mkdir(parents=True, exist_ok=True)
        plugins = self.resource_dir / "UserPlugins"
        plugins.mkdir(parents=True, exist_ok=True)
        if copy_sws:
            source = self.reaper_exe.parent / "UserPlugins" / SWS_DLL
            if source.exists() and not (plugins / SWS_DLL).exists():
                shutil.copy2(source, plugins / SWS_DLL)
        seed_ini(self.ini_path, TEST_INI_SETTINGS if self.audio == "dummy" else LISTEN_INI_SETTINGS)
        channel = self.channel_dir()
        for sub in ("inbox", "outbox", "journal"):
            (channel / sub).mkdir(parents=True, exist_ok=True)
        self.project_path.write_text(
            _blank_project("AudioSearch bridge test", ["AS Test 1", "AS Test 2"]), encoding="utf-8"
        )
        self.alt_project_path.write_text(
            _blank_project("AudioSearch bridge test (other)", ["AS Other 1"]), encoding="utf-8"
        )
        instance_id = "asb-test-" + uuid.uuid5(uuid.NAMESPACE_URL, str(self.resource_dir)).hex[:12]
        cfg = {
            "protocol": PROTOCOL,
            "instance_id": instance_id,
            "token": secrets.token_hex(32),
            "channel": str(channel),
            "resource_path": str(self.resource_dir),
            "test_root": str(self.root),
            "test_commands": test_commands,
            "poll_interval": 0.03,
            "heartbeat_interval": 0.25,
            "max_per_tick": 4,
        }
        tmp = self.config_path.with_suffix(".tmp")
        tmp.write_bytes(encode_message(cfg))
        os.replace(tmp, self.config_path)
        return self.config_path

    def command(self) -> list[str]:
        return [
            str(self.reaper_exe),
            "-newinst",
            "-nosplash",
            "-cfgfile",
            str(self.ini_path),
            str(self.project_path),
            str(self.script),
        ]

    def launch(self) -> int:
        try:
            running = self.process_lister()
        except ProcessQueryError as exc:
            raise RuntimeError(
                f"无法确认是否有 REAPER 在运行（{exc}），按规则不启动测试实例"
            ) from exc
        if running:
            pids = "; ".join(
                f"{p.pid} {(p.command_line or '<命令行不可读>')[:160]}" for p in running
            )
            raise RuntimeError(f"已有 REAPER 进程在运行（PID {pids}），按规则不启动测试实例")
        if not self.config_path.exists():
            raise RuntimeError("测试实例尚未 prepare()")
        log = (self.root / f"{self.profile}-launch.log").open("ab")
        startup = None
        if sys.platform == "win32" and self.minimized:
            # Start minimised without taking focus so the user's work is not interrupted.
            startup = subprocess.STARTUPINFO()
            startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup.wShowWindow = 7  # SW_SHOWMINNOACTIVE
        self.process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            self.command(), stdout=log, stderr=log, cwd=str(self.resource_dir), startupinfo=startup
        )
        return self.process.pid

    def stop(self, *, timeout: float = 10.0) -> bool:
        """Terminate only the process this object started. Idempotent.

        Ownership is proven by the ``Popen`` handle itself: while it is open the
        OS cannot reuse the PID, and ``terminate()`` acts on that handle, never on
        a PID looked up again. No process enumeration is needed (so a failing
        enumeration cannot make us touch another REAPER). An instance that has
        already exited counts as stopped.
        """
        if self.process is None:
            return False
        if self.process.poll() is not None:
            return True
        self.process.terminate()
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=timeout)
        return True


__all__ = [
    "DEFAULT_REAPER_EXE",
    "LISTEN_PROFILE",
    "TEST_PROFILE",
    "ProcessQueryError",
    "ReaperProcess",
    "TestInstance",
    "list_reaper_processes",
]
