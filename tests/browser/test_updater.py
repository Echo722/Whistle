"""Behavioral tests for Whistle <1 MB GitHub hot-updater, launcher rollback, dialog, and patch script."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

from audiosearch.browser.main_window import BrowserWindow, create_app
from audiosearch.browser.reaper_setup import find_bundled_spot_script
from audiosearch.browser.runtime_paths import resolve_app_icon_path
from audiosearch.browser.update_dialog import UpdateDialog
from audiosearch.browser.updater import (
    APP_VERSION,
    UpdateCheckResult,
    UpdateManifest,
    active_patch_info,
    apply_patch_zip,
    can_rollback,
    check_for_updates,
    download_and_stage_patch,
    effective_version,
    is_newer_version,
    mark_patch_boot_healthy,
    parse_version,
    rollback_patch,
)
from tests.ai.test_browser_ai_ui import _FakeAudition

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _build_patch_zip(
    out_zip: Path,
    *,
    version: str = "0.2.0",
    min_abi: int = 1,
    marker: str = "v2",
    bad_member: str | None = None,
) -> tuple[str, int]:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("audiosearch/__init__.py", f'__version__ = "{version}"\n')
        zf.writestr("audiosearch/marker.txt", marker)
        zf.writestr(
            "patch_manifest.json",
            json.dumps({"version": version, "min_runtime_abi": min_abi}),
        )
        if bad_member is not None:
            zf.writestr(bad_member, "evil")
    raw = out_zip.read_bytes()
    return hashlib.sha256(raw).hexdigest().lower(), len(raw)


def test_version_compare_and_check_for_updates_states(tmp_path: Path) -> None:
    assert parse_version("v0.1.10") == (0, 1, 10)
    assert is_newer_version("v0.1.10", "0.1.2") is True
    assert is_newer_version("0.1.0", "0.1.0") is False

    patch_zip = tmp_path / "whistle-patch-v0.2.0.zip"
    sha, size = _build_patch_zip(patch_zip, version="0.2.0")
    manifest_file = tmp_path / "latest.json"

    def _write_manifest(obj: dict) -> str:
        manifest_file.write_text(json.dumps(obj), encoding="utf-8")
        return manifest_file.as_uri()

    # 1. Up to date
    url = _write_manifest({"version": "0.1.0", "min_runtime_abi": 1})
    res_same = check_for_updates(manifest_url=url, current_version="0.1.0")
    assert res_same.status == "up_to_date"

    # 2. Patch available
    url = _write_manifest(
        {
            "version": "0.2.0",
            "min_runtime_abi": 1,
            "patch_url": patch_zip.as_uri(),
            "patch_sha256": sha,
            "patch_size": size,
            "release_url": "https://github.com/Echo722/Whistle/releases/tag/v0.2.0",
            "notes": "修复若干问题",
        }
    )
    res_patch = check_for_updates(manifest_url=url, current_version="0.1.0")
    assert res_patch.status == "patch_available"
    assert res_patch.manifest is not None
    assert res_patch.manifest.version == "0.2.0"

    # 3. Full update required when min_runtime_abi > current
    url = _write_manifest(
        {
            "version": "0.3.0",
            "min_runtime_abi": 2,
            "patch_url": patch_zip.as_uri(),
            "patch_sha256": sha,
            "release_url": "https://github.com/Echo722/Whistle/releases/tag/v0.3.0",
        }
    )
    res_full = check_for_updates(manifest_url=url, current_version="0.1.0", current_runtime_abi=1)
    assert res_full.status == "full_update_required"

    # 4. Network/missing file handled gracefully
    res_err = check_for_updates(manifest_url=(tmp_path / "missing.json").as_uri())
    assert res_err.status == "error"


def test_patch_download_security_and_rollback_preserves_data(tmp_path: Path) -> None:
    app_root = tmp_path / "Whistle"
    (app_root / "data").mkdir(parents=True)
    (app_root / "data" / "library.sqlite3").write_text("KEEP_DB", encoding="utf-8")
    (app_root / "models" / "hf").mkdir(parents=True)
    (app_root / "models" / "hf" / "keep.txt").write_text("KEEP_MODELS", encoding="utf-8")

    # 1. Zip-Slip protection
    evil_zip = tmp_path / "evil.zip"
    evil_sha, evil_sz = _build_patch_zip(evil_zip, bad_member="../escape.txt")
    evil_manifest = UpdateManifest("0.2.0", 1, evil_zip.as_uri(), evil_sha, evil_sz, "", "", "")
    try:
        apply_patch_zip(evil_zip, evil_manifest, app_root=app_root)
        raise AssertionError("Expected ValueError for path traversal")
    except ValueError:
        pass
    assert not (tmp_path / "escape.txt").exists()

    # 2. Bad SHA-256 rejected
    good_zip = tmp_path / "good_v2.zip"
    good_sha, good_sz = _build_patch_zip(good_zip, version="0.2.0", marker="v2")
    bad_hash_manifest = UpdateManifest("0.2.0", 1, good_zip.as_uri(), "0" * 64, good_sz, "", "", "")
    try:
        apply_patch_zip(good_zip, bad_hash_manifest, app_root=app_root)
        raise AssertionError("Expected ValueError for bad SHA-256")
    except ValueError:
        pass

    # 3. Valid download_and_stage_patch v0.2.0 -> v0.2.1 preserving data/ and models/
    m2 = UpdateManifest("0.2.0", 1, good_zip.as_uri(), good_sha, good_sz, "", "v2", "")
    progress_calls: list[tuple[int, int]] = []
    install2 = download_and_stage_patch(
        m2, app_root=app_root, progress_cb=lambda a, b: progress_calls.append((a, b))
    )
    assert install2.version == "0.2.0"
    assert progress_calls
    assert effective_version(app_root) == "0.2.0"
    assert can_rollback(app_root) is True
    assert (app_root / "app_patch" / "audiosearch" / "marker.txt").read_text() == "v2"
    mark_patch_boot_healthy(app_root)
    assert not (app_root / "app_patch" / ".boot_attempt").exists()

    good_v3 = tmp_path / "good_v3.zip"
    sha3, sz3 = _build_patch_zip(good_v3, version="0.2.1", marker="v3")
    m3 = UpdateManifest("0.2.1", 1, good_v3.as_uri(), sha3, sz3, "", "v3", "")
    download_and_stage_patch(m3, app_root=app_root)
    assert effective_version(app_root) == "0.2.1"
    assert (app_root / "app_patch" / "audiosearch" / "marker.txt").read_text() == "v3"

    # Rollback v0.2.1 -> v0.2.0 -> built-in APP_VERSION
    assert rollback_patch(app_root) is True
    assert effective_version(app_root) == "0.2.0"
    assert rollback_patch(app_root) is True
    assert active_patch_info(app_root) is None
    assert effective_version(app_root) == APP_VERSION

    # data/ and models/ untouched across updates & rollbacks
    assert (app_root / "data" / "library.sqlite3").read_text(encoding="utf-8") == "KEEP_DB"
    assert (app_root / "models" / "hf" / "keep.txt").read_text(encoding="utf-8") == "KEEP_MODELS"


def test_launcher_boot_attempt_auto_rollback(tmp_path: Path, monkeypatch) -> None:
    launcher_path = Path(__file__).resolve().parents[2] / "packaging" / "launcher.py"
    spec = importlib.util.spec_from_file_location("whistle_launcher_test", str(launcher_path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    app_root = tmp_path / "Whistle"
    z2 = tmp_path / "v2.zip"
    sha2, sz2 = _build_patch_zip(z2, version="0.2.0", marker="v2")
    apply_patch_zip(z2, UpdateManifest("0.2.0", 1, "", sha2, sz2, "", "", ""), app_root=app_root)
    mark_patch_boot_healthy(app_root)

    z3 = tmp_path / "v3.zip"
    sha3, sz3 = _build_patch_zip(z3, version="0.2.1", marker="v3")
    apply_patch_zip(z3, UpdateManifest("0.2.1", 1, "", sha3, sz3, "", "", ""), app_root=app_root)

    # First launch writes "1" into .boot_attempt and activates v0.2.1
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delenv("WHISTLE_PATCH_DIR", raising=False)
    active = mod._prepare_patch_sys_path(app_root)
    assert active == app_root / "app_patch"
    assert (app_root / "app_patch" / ".boot_attempt").read_text(encoding="utf-8") == "1"

    # Simulate crash BEFORE mark_patch_boot_healthy: second launch rolls back v0.2.1 -> v0.2.0!
    monkeypatch.delenv("WHISTLE_PATCH_DIR", raising=False)
    active2 = mod._prepare_patch_sys_path(app_root)
    assert active2 == app_root / "app_patch"
    assert effective_version(app_root) == "0.2.0"


def test_update_dialog_states_and_build_patch_script(tmp_path: Path) -> None:
    create_app([])
    patch_out = tmp_path / "patches"
    script = Path(__file__).resolve().parents[2] / "packaging" / "build_patch.ps1"
    proc = subprocess.run(
        [
            "powershell",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-Version",
            "0.1.9",
            "-Notes",
            "自动打包补丁测试",
            "-OutDir",
            str(patch_out),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr

    zip_file = patch_out / "whistle-patch-v0.1.9.zip"
    latest_file = patch_out / "latest.json"
    assert zip_file.is_file()
    assert latest_file.is_file()
    assert zip_file.stat().st_size < 1024 * 1024  # Strictly < 1 MB!

    latest = json.loads(latest_file.read_text(encoding="utf-8"))
    assert latest["version"] == "0.1.9"
    assert latest["patch_url"].startswith("https://github.com/Echo722/Whistle/releases/download/v0.1.9/")

    # Override patch_url to local file URI and test end-to-end check + UpdateDialog UI
    latest["patch_url"] = zip_file.as_uri()
    latest_file.write_text(json.dumps(latest, ensure_ascii=False), encoding="utf-8")

    app_root = tmp_path / "AppRoot"
    app_root.mkdir()
    check_res = check_for_updates(manifest_url=latest_file.as_uri(), current_version="0.1.0")
    assert check_res.status == "patch_available"

    dlg = UpdateDialog(
        manifest_url=latest_file.as_uri(),
        app_root=app_root,
        initial_result=check_res,
    )
    assert dlg.action_btn.isEnabled() is True
    assert "0.1.9" in dlg.status_label.text()
    assert "自动打包补丁测试" in dlg.notes_edit.toPlainText()
    dlg.close()

    # Verify extracted patch contains real audiosearch package
    assert check_res.manifest is not None
    download_and_stage_patch(check_res.manifest, app_root=app_root)
    assert (app_root / "app_patch" / "audiosearch" / "browser" / "updater.py").is_file()
    assert (app_root / "app_patch" / "reaper" / "audiosearch_spot_bridge.lua").is_file()

    # Check UpdateDialog for full_update_required
    full_res = UpdateCheckResult(
        status="full_update_required",
        current_version="0.1.0",
        current_runtime_abi=1,
        manifest=UpdateManifest("0.9.0", 2, "", "", 0, "https://github.com/Echo722/Whistle/releases", "PySide6 upgrade", ""),
        message="发现新版本 v0.9.0（含底层依赖升级）",
    )
    dlg2 = UpdateDialog(initial_result=full_res)
    assert dlg2.action_btn.isEnabled() is False
    dlg2.show()
    assert dlg2.open_release_btn.isVisible() is True
    dlg2.close()


def test_patch_resource_overrides_and_main_window_version_button(tmp_path: Path, monkeypatch) -> None:
    create_app([])
    patch_dir = tmp_path / "custom_patch"
    res_dir = patch_dir / "audiosearch" / "browser" / "resources"
    res_dir.mkdir(parents=True)
    ico = res_dir / "audiosearch.ico"
    ico.write_bytes(b"ICO")
    lua_dir = patch_dir / "reaper"
    lua_dir.mkdir(parents=True)
    lua = lua_dir / "audiosearch_spot_bridge.lua"
    lua.write_text("-- patched", encoding="utf-8")

    monkeypatch.setenv("WHISTLE_PATCH_DIR", str(patch_dir))
    assert resolve_app_icon_path() == ico
    assert find_bundled_spot_script() == lua
    monkeypatch.delenv("WHISTLE_PATCH_DIR", raising=False)

    win = BrowserWindow(db_path=tmp_path / "lib.sqlite3", audition=_FakeAudition())
    try:
        assert win.status_version.text() == f"v{APP_VERSION}"
        res = UpdateCheckResult(
            status="patch_available",
            current_version="0.1.0",
            current_runtime_abi=1,
            manifest=UpdateManifest("0.2.0", 1, "https://example.com/p.zip", "a" * 64, 1024, "", "ok", ""),
            message="发现新版本 v0.2.0",
        )
        win._on_silent_update_checked(res)
        assert win.status_version.text() == "↑ v0.2.0"
        assert win.status_version.property("statusState") == "ok"
    finally:
        win.close()


def test_build_patch_py_output_passes_updater_check_and_apply(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "packaging" / "build_patch.py"
    out_dir = tmp_path / "patches"

    proc = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--version",
            "v0.2.0",
            "--notes",
            "CI patch test",
            "--out-dir",
            str(out_dir),
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout

    patch_zip = out_dir / "whistle-patch-v0.2.0.zip"
    latest_json = out_dir / "latest.json"
    assert patch_zip.is_file()
    assert latest_json.is_file()

    payload = json.loads(latest_json.read_text(encoding="utf-8"))
    raw = patch_zip.read_bytes()
    assert payload["version"] == "0.2.0"
    assert payload["min_runtime_abi"] >= 1
    assert payload["patch_sha256"] == hashlib.sha256(raw).hexdigest().lower()
    assert payload["patch_size"] == len(raw)
    assert payload["notes"] == "CI patch test"

    local_payload = dict(payload)
    local_payload["patch_url"] = patch_zip.as_uri()
    local_manifest = tmp_path / "local-latest.json"
    local_manifest.write_text(json.dumps(local_payload), encoding="utf-8")

    check_res = check_for_updates(manifest_url=local_manifest.as_uri(), current_version="0.1.0")
    assert check_res.status == "patch_available"
    assert check_res.manifest is not None

    app_root = tmp_path / "WhistleApp"
    app_root.mkdir(parents=True)
    install_res = download_and_stage_patch(check_res.manifest, app_root=app_root)
    assert effective_version(app_root) == "0.2.0"
    assert (install_res.patch_dir / "audiosearch" / "__init__.py").is_file()
    assert (install_res.patch_dir / "reaper" / "audiosearch_spot_bridge.lua").is_file()
