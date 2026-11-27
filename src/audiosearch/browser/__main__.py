"""Entry point for ``python -m audiosearch.browser`` and frozen executable."""

from __future__ import annotations

import argparse
import faulthandler
import importlib
import json
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from audiosearch.browser.main_window import BrowserWindow, create_app
from audiosearch.browser.reaper_setup import find_bundled_spot_script
from audiosearch.browser.runtime_paths import (
    default_db_path,
    is_frozen,
    resolve_app_icon_path,
)


def _install_crash_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    crash_file = log_path.open("a", encoding="utf-8", buffering=1)
    faulthandler.enable(file=crash_file, all_threads=True)

    prev_sys_hook = sys.excepthook
    prev_thread_hook = threading.excepthook

    def _log_exc(prefix: str, exc_type, exc_value, exc_tb) -> None:
        ts = datetime.now().isoformat(timespec="seconds")
        try:
            crash_file.write(f"\n[{ts}] {prefix}\n")
            traceback.print_exception(exc_type, exc_value, exc_tb, file=crash_file)
            crash_file.flush()
        except OSError:
            pass

    def _sys_hook(exc_type, exc_value, exc_tb) -> None:
        _log_exc("Uncaught exception", exc_type, exc_value, exc_tb)
        prev_sys_hook(exc_type, exc_value, exc_tb)

    def _thread_hook(args: threading.ExceptHookArgs) -> None:
        name = args.thread.name if args.thread is not None else "unknown"
        _log_exc(f"Uncaught thread exception ({name})", args.exc_type, args.exc_value, args.exc_traceback)
        prev_thread_hook(args)

    sys.excepthook = _sys_hook
    threading.excepthook = _thread_hook

    def _cleanup() -> None:
        sys.excepthook = prev_sys_hook
        threading.excepthook = prev_thread_hook
        try:
            faulthandler.disable()
        except Exception:
            pass
        try:
            crash_file.close()
        except OSError:
            pass

    return _cleanup


def run_smoke_test(db_path: Path | None = None) -> tuple[int, dict[str, Any]]:
    """Verify bundled resources, SQLite store, audio stack, and optional AI runtime."""
    from audiosearch.library.store import LibraryStore

    resolved_db = Path(db_path) if db_path is not None else default_db_path()
    report: dict[str, Any] = {
        "frozen": is_frozen(),
        "db_path": str(resolved_db),
        "store_ok": False,
        "reaper_script": False,
        "app_icon": False,
        "core": {},
        "ai": {},
    }

    try:
        store = LibraryStore(resolved_db)
        _ = store.total_file_count()
        store.close()
        report["store_ok"] = True
    except Exception as exc:
        report["store_error"] = f"{type(exc).__name__}: {exc}"

    spot_script = find_bundled_spot_script()
    report["reaper_script"] = bool(spot_script is not None and spot_script.is_file())

    icon_path = resolve_app_icon_path()
    report["app_icon"] = bool(icon_path is not None and icon_path.is_file())

    core_mods = ("PySide6", "sounddevice", "soundfile", "pedalboard", "mutagen")
    for mod in core_mods:
        try:
            importlib.import_module(mod)
            report["core"][mod] = True
        except Exception as exc:
            report["core"][mod] = f"{type(exc).__name__}: {exc}"

    for mod in ("torch", "transformers", "open_clip", "soxr"):
        try:
            imported = importlib.import_module(mod)
            if mod == "torch":
                report["ai"]["torch"] = str(getattr(imported, "__version__", True))
                try:
                    report["ai"]["cuda_available"] = bool(imported.cuda.is_available())
                except Exception:
                    report["ai"]["cuda_available"] = False
                try:
                    from audiosearch.inference.runtime import probe_cuda

                    cuda_usable, probe_reason = probe_cuda(imported, force=True)
                    report["ai"]["cuda_usable"] = bool(cuda_usable)
                    if not cuda_usable:
                        report["ai"]["cuda_probe_reason"] = probe_reason
                except Exception as exc:
                    report["ai"]["cuda_usable"] = False
                    report["ai"]["cuda_probe_reason"] = f"{type(exc).__name__}: {exc}"
                try:
                    report["ai"]["cuda_arch_list"] = list(imported.cuda.get_arch_list())
                except Exception:
                    report["ai"]["cuda_arch_list"] = []
            else:
                report["ai"][mod] = True
        except Exception as exc:
            report["ai"][mod] = f"{type(exc).__name__}: {exc}"

    core_all_ok = (
        report["store_ok"]
        and report["reaper_script"]
        and all(v is True for v in report["core"].values())
    )
    return (0 if core_all_ok else 1), report


def launch_browser(
    *,
    db_path: Path | None = None,
    add_root: Path | None = None,
    argv: list[str] | None = None,
) -> int:
    resolved_db = Path(db_path) if db_path is not None else default_db_path()
    log_path = resolved_db.parent / "logs" / "browser-crash.log"
    cleanup = _install_crash_logging(log_path)
    try:
        app = create_app(argv)
        window = BrowserWindow(db_path=resolved_db)
        window.show()
        if add_root is not None:
            window.add_and_scan_root(add_root)
        return app.exec()
    finally:
        cleanup()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="audiosearch-browser")
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--add-root", type=Path, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(argv)

    if args.db is not None:
        db_path = args.db
    elif args.data_dir is not None:
        db_path = args.data_dir / "library.sqlite3"
    else:
        db_path = default_db_path()

    if args.smoke_test:
        code, report = run_smoke_test(db_path)
        print(json.dumps(report, ensure_ascii=False))
        return code

    return launch_browser(db_path=db_path, add_root=args.add_root, argv=[])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
