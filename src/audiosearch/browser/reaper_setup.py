"""One-click installer and health inspector for REAPER Spot bridge script."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

SCRIPT_FILENAME = "audiosearch_spot_bridge.lua"
SCRIPT_SUBDIR = "AudioSearchTool"
STARTUP_MARKER_BEGIN = "-- >>> AudioSearchTool Spot Bridge (auto-managed) >>>"
STARTUP_MARKER_END = "-- <<< AudioSearchTool Spot Bridge <<<"

STARTUP_SNIPPET = "\n".join(
    [
        STARTUP_MARKER_BEGIN,
        "do",
        '  local _as_script = reaper.GetResourcePath() .. "/Scripts/audiosearch_spot_bridge.lua"',
        "  if reaper.file_exists(_as_script) then",
        "    pcall(dofile, _as_script)",
        "  end",
        "end",
        STARTUP_MARKER_END,
    ]
)

_MARKER_BLOCK_RE = re.compile(
    rf"(?:\r?\n)*{re.escape(STARTUP_MARKER_BEGIN)}.*?{re.escape(STARTUP_MARKER_END)}(?:\r?\n)*",
    re.DOTALL,
)
_LEGACY_STARTUP_LINE_RE = re.compile(
    r"^[ \t]*(?!--)[^\r\n]*audiosearch_spot_bridge\.lua[^\r\n]*$",
    re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class ReaperBridgeInstallState:
    reaper_dir: Path
    reaper_dir_exists: bool
    script_dest: Path
    script_installed: bool
    script_up_to_date: bool
    startup_lua_path: Path
    startup_registered: bool


def find_bundled_spot_script() -> Path | None:
    """Locate bundled ``audiosearch_spot_bridge.lua`` across patch, frozen, package, and repo paths."""
    candidates: list[Path] = []
    patch_dir = os.environ.get("WHISTLE_PATCH_DIR", "").strip()
    if patch_dir:
        base = Path(patch_dir)
        candidates.append(base / "reaper" / SCRIPT_FILENAME)
        candidates.append(base / "audiosearch" / "browser" / "resources" / SCRIPT_FILENAME)
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        base = Path(str(meipass))
        candidates.append(base / "reaper" / SCRIPT_FILENAME)
        candidates.append(base / "audiosearch" / "browser" / "resources" / SCRIPT_FILENAME)
    candidates.append(Path(__file__).resolve().parent / "resources" / SCRIPT_FILENAME)
    candidates.append(Path(__file__).resolve().parents[3] / "reaper" / SCRIPT_FILENAME)
    for cand in candidates:
        try:
            if cand.is_file():
                return cand
        except OSError:
            continue
    return None


def default_reaper_resource_dir() -> Path:
    """Return the active REAPER resource directory (checking standard and portable installs)."""
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        standard = Path(appdata) / "REAPER" if appdata else Path.home() / "AppData" / "Roaming" / "REAPER"
        candidates = [standard, Path("C:/REAPER"), Path("D:/REAPER"), Path("E:/REAPER")]
        for cand in candidates:
            try:
                if (cand / "reaper.ini").is_file():
                    return cand
            except OSError:
                continue
        return standard
    return Path.home() / ".config" / "REAPER"


def _has_startup_registration(text: str) -> bool:
    if STARTUP_MARKER_BEGIN in text and STARTUP_MARKER_END in text:
        return True
    return bool(_LEGACY_STARTUP_LINE_RE.search(text))


def _upsert_startup_block(existing_text: str, enable: bool) -> str:
    """Remove any managed/legacy AudioSearchTool entry from ``__startup.lua`` and optionally append it."""
    stripped = _MARKER_BLOCK_RE.sub("\n\n", existing_text)
    stripped = _LEGACY_STARTUP_LINE_RE.sub("", stripped)
    stripped = re.sub(r"(?:\r?\n){3,}", "\n\n", stripped)
    cleaned = stripped.strip("\r\n")
    if not enable:
        return f"{cleaned}\n" if cleaned else ""
    if not cleaned:
        return f"{STARTUP_SNIPPET}\n"
    return f"{cleaned}\n\n{STARTUP_SNIPPET}\n"


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.name}.tmp.{os.getpid()}"
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def inspect_reaper_install(
    reaper_dir: Path | None = None,
    *,
    source_script: Path | None = None,
) -> ReaperBridgeInstallState:
    """Inspect current installation and ``__startup.lua`` state for REAPER Spot bridge."""
    resolved_dir = Path(reaper_dir) if reaper_dir is not None else default_reaper_resource_dir()
    src = source_script or find_bundled_spot_script()
    primary_dest = resolved_dir / "Scripts" / SCRIPT_FILENAME
    legacy_dest = resolved_dir / "Scripts" / SCRIPT_SUBDIR / SCRIPT_FILENAME
    script_dest = primary_dest if primary_dest.is_file() or not legacy_dest.is_file() else legacy_dest
    startup_lua_path = resolved_dir / "Scripts" / "__startup.lua"

    reaper_dir_exists = resolved_dir.is_dir()
    script_installed = script_dest.is_file()
    script_up_to_date = False
    if script_installed and src is not None and src.is_file():
        try:
            script_up_to_date = script_dest.read_bytes() == src.read_bytes()
        except OSError:
            script_up_to_date = False

    startup_registered = False
    if startup_lua_path.is_file():
        try:
            startup_text = startup_lua_path.read_text(encoding="utf-8", errors="replace")
            startup_registered = _has_startup_registration(startup_text)
        except OSError:
            startup_registered = False

    return ReaperBridgeInstallState(
        reaper_dir=resolved_dir,
        reaper_dir_exists=reaper_dir_exists,
        script_dest=script_dest,
        script_installed=script_installed,
        script_up_to_date=script_up_to_date,
        startup_lua_path=startup_lua_path,
        startup_registered=startup_registered,
    )


def install_reaper_bridge(
    reaper_dir: Path,
    *,
    enable_startup: bool = True,
    source_script: Path | None = None,
) -> ReaperBridgeInstallState:
    """Install or update ``audiosearch_spot_bridge.lua`` and manage ``__startup.lua`` safely."""
    src = source_script or find_bundled_spot_script()
    if src is None or not src.is_file():
        raise FileNotFoundError("未找到内置 REAPER 桥接脚本 audiosearch_spot_bridge.lua")

    resolved_dir = Path(reaper_dir)
    current = inspect_reaper_install(resolved_dir, source_script=src)
    if (
        current.script_installed
        and current.script_up_to_date
        and current.startup_registered == bool(enable_startup)
    ):
        return current

    primary_dest = resolved_dir / "Scripts" / SCRIPT_FILENAME
    startup_lua_path = resolved_dir / "Scripts" / "__startup.lua"
    src_bytes = src.read_bytes()

    need_write_script = True
    if primary_dest.is_file():
        try:
            need_write_script = primary_dest.read_bytes() != src_bytes
        except OSError:
            need_write_script = True
    if need_write_script:
        _atomic_write_bytes(primary_dest, src_bytes)

    existing = ""
    if startup_lua_path.is_file():
        existing = startup_lua_path.read_text(encoding="utf-8", errors="replace")
    updated = _upsert_startup_block(existing, enable=enable_startup)
    if updated != existing:
        if updated:
            _atomic_write_bytes(startup_lua_path, updated.encode("utf-8"))
        elif startup_lua_path.is_file():
            startup_lua_path.unlink(missing_ok=True)

    return inspect_reaper_install(resolved_dir, source_script=src)


def uninstall_reaper_bridge(
    reaper_dir: Path,
    *,
    source_script: Path | None = None,
) -> ReaperBridgeInstallState:
    """Uninstall ``audiosearch_spot_bridge.lua`` and remove managed/legacy ``__startup.lua`` entries."""
    resolved_dir = Path(reaper_dir)
    primary_dest = resolved_dir / "Scripts" / SCRIPT_FILENAME
    legacy_dest = resolved_dir / "Scripts" / SCRIPT_SUBDIR / SCRIPT_FILENAME
    startup_lua_path = resolved_dir / "Scripts" / "__startup.lua"

    if startup_lua_path.is_file():
        existing = startup_lua_path.read_text(encoding="utf-8", errors="replace")
        cleaned = _upsert_startup_block(existing, enable=False)
        if cleaned != existing:
            if cleaned:
                _atomic_write_bytes(startup_lua_path, cleaned.encode("utf-8"))
            else:
                startup_lua_path.unlink(missing_ok=True)

    for dest in (primary_dest, legacy_dest):
        if dest.is_file():
            dest.unlink(missing_ok=True)

    legacy_dir = legacy_dest.parent
    if legacy_dir.is_dir():
        try:
            next(legacy_dir.iterdir())
        except StopIteration:
            try:
                legacy_dir.rmdir()
            except OSError:
                pass

    return inspect_reaper_install(resolved_dir, source_script=source_script)


__all__ = [
    "ReaperBridgeInstallState",
    "SCRIPT_FILENAME",
    "SCRIPT_SUBDIR",
    "STARTUP_MARKER_BEGIN",
    "STARTUP_MARKER_END",
    "STARTUP_SNIPPET",
    "default_reaper_resource_dir",
    "find_bundled_spot_script",
    "inspect_reaper_install",
    "install_reaper_bridge",
    "uninstall_reaper_bridge",
]
