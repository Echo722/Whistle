"""Portable vs. development runtime paths for data, icons, and offline AI models."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def is_frozen() -> bool:
    """Return True when running from a PyInstaller frozen bundle."""
    return bool(getattr(sys, "frozen", False))


def executable_dir() -> Path:
    """Directory containing the running executable (or CWD during source runs)."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path.cwd()


def _dir_is_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f".audiosearch_write_probe.{os.getpid()}"
        probe.write_bytes(b"1")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def default_data_dir() -> Path:
    """Resolve data directory for SQLite library, settings, renders, logs, and vector indexes."""
    env_dir = os.environ.get("AUDIOSEARCH_DATA_DIR", "").strip()
    if env_dir:
        return Path(env_dir).expanduser()

    if is_frozen():
        exe_dir = executable_dir()
        if (
            (exe_dir / "portable.flag").is_file()
            or (exe_dir / "data").is_dir()
            or _dir_is_writable(exe_dir)
        ):
            return exe_dir / "data"
        if os.name == "nt":
            base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
            return Path(base) / "AudioSearchTool" / "data"
        return Path.home() / ".local" / "share" / "AudioSearchTool"

    return Path(".local")


def default_db_path() -> Path:
    """Default SQLite library path for dev and packaged builds."""
    return default_data_dir() / "library.sqlite3"


def resolve_app_icon_path() -> Path | None:
    """Locate the application icon (.ico preferred, fallback .png) in patch, frozen, or dev trees."""
    candidates: list[Path] = []
    patch_dir = os.environ.get("WHISTLE_PATCH_DIR", "").strip()
    if patch_dir:
        base = Path(patch_dir)
        candidates.append(base / "audiosearch" / "browser" / "resources" / "audiosearch.ico")
        candidates.append(base / "audiosearch" / "browser" / "resources" / "audiosearch.png")
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        base = Path(str(meipass))
        candidates.append(base / "audiosearch" / "browser" / "resources" / "audiosearch.ico")
        candidates.append(base / "audiosearch" / "browser" / "resources" / "audiosearch.png")
    pkg_res = Path(__file__).resolve().parent / "resources"
    candidates.append(pkg_res / "audiosearch.ico")
    candidates.append(pkg_res / "audiosearch.png")
    for cand in candidates:
        try:
            if cand.is_file():
                return cand
        except OSError:
            continue
    return None


def resolve_packaged_models_dir() -> Path | None:
    """Locate an offline HuggingFace model cache next to a frozen executable."""
    if not is_frozen():
        return None
    exe_dir = executable_dir()
    for cand in (exe_dir / "models" / "hf", exe_dir / "data" / "models" / "hf"):
        try:
            if (cand / "hub").is_dir():
                return cand
        except OSError:
            continue
    return None


__all__ = [
    "default_data_dir",
    "default_db_path",
    "executable_dir",
    "is_frozen",
    "resolve_app_icon_path",
    "resolve_packaged_models_dir",
]
