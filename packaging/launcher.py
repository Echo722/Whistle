"""PyInstaller frozen entry point for Whistle desktop client with hot-patch loader & rollback."""

from __future__ import annotations

import ctypes
import json
import multiprocessing
import os
import shutil
import sys
import time
from pathlib import Path

_RUNTIME_ABI = 2
_PATCH_DIR_NAME = "app_patch"
_PATCH_PREV_DIR_NAME = "app_patch.prev"
_PATCH_FAILED_DIR_NAME = "app_patch.failed"
_MANIFEST_NAME = "patch_manifest.json"
_SENTINEL_NAME = ".boot_attempt"


def _set_windows_app_id() -> None:
    if os.name != "nt":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Whistle.Desktop.1")
    except Exception:
        pass


def _rename_dir_with_retry(
    src: Path,
    dst: Path,
    *,
    attempts: int = 12,
    base_delay: float = 0.05,
) -> None:
    """Retry transient Windows directory locks during launcher rollback swaps."""
    for idx in range(attempts):
        try:
            src.rename(dst)
            return
        except PermissionError:
            if idx + 1 >= attempts:
                raise
            time.sleep(base_delay * (idx + 1))


def _rollback_patch_dirs(exe_dir: Path) -> None:
    patch_dir = exe_dir / _PATCH_DIR_NAME
    prev_dir = exe_dir / _PATCH_PREV_DIR_NAME
    failed_dir = exe_dir / _PATCH_FAILED_DIR_NAME
    if failed_dir.exists():
        shutil.rmtree(failed_dir, ignore_errors=True)
    if patch_dir.exists():
        try:
            _rename_dir_with_retry(patch_dir, failed_dir)
        except OSError:
            shutil.rmtree(patch_dir, ignore_errors=True)
    if prev_dir.exists():
        try:
            _rename_dir_with_retry(prev_dir, patch_dir)
        except OSError:
            pass


def _prepare_patch_sys_path(exe_dir: Path, *, runtime_abi: int = _RUNTIME_ABI) -> Path | None:
    """Activate ``exe_dir/app_patch`` on ``sys.path`` if valid and auto-rollback on failed boot."""
    patch_dir = exe_dir / _PATCH_DIR_NAME
    if not patch_dir.is_dir():
        return None

    sentinel = patch_dir / _SENTINEL_NAME
    if sentinel.is_file():
        try:
            state = sentinel.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            state = "1"
        if state == "1":
            # Previous startup with this patch crashed before marking healthy -> rollback once.
            _rollback_patch_dirs(exe_dir)
            if not patch_dir.is_dir():
                return None
        else:
            try:
                sentinel.write_text("1", encoding="utf-8")
            except OSError:
                pass

    manifest_path = patch_dir / _MANIFEST_NAME
    if not (patch_dir / "audiosearch" / "__init__.py").is_file() or not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if int(manifest.get("min_runtime_abi") or 1) > runtime_abi:
            return None
    except (OSError, ValueError, TypeError):
        return None

    patch_str = str(patch_dir)
    if patch_str not in sys.path:
        sys.path.insert(0, patch_str)
    os.environ["WHISTLE_PATCH_DIR"] = patch_str
    return patch_dir


def _run_with_patch_fallback() -> int:
    exe_dir = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path.cwd()
    active_patch = _prepare_patch_sys_path(exe_dir) if getattr(sys, "frozen", False) else None
    try:
        from audiosearch.browser.__main__ import main

        return main(sys.argv[1:])
    except Exception:
        if active_patch is not None:
            patch_str = str(active_patch)
            while patch_str in sys.path:
                sys.path.remove(patch_str)
            for mod_name in [k for k in sys.modules if k == "audiosearch" or k.startswith("audiosearch.")]:
                sys.modules.pop(mod_name, None)
            os.environ.pop("WHISTLE_PATCH_DIR", None)
            _rollback_patch_dirs(exe_dir)
            from audiosearch.browser.__main__ import main

            return main(sys.argv[1:])
        raise


if __name__ == "__main__":
    multiprocessing.freeze_support()
    _set_windows_app_id()
    raise SystemExit(_run_with_patch_fallback())
