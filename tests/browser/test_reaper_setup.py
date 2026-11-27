"""Behavioral and UI tests for REAPER one-click bridge setup and multi-select Spot."""

from __future__ import annotations

import os
import wave
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QPushButton

from audiosearch.browser import reaper_dialog
from audiosearch.browser.main_window import BrowserWindow, create_app
from audiosearch.browser.reaper_dialog import ReaperSetupDialog
from audiosearch.browser.reaper_setup import (
    STARTUP_MARKER_BEGIN,
    STARTUP_MARKER_END,
    find_bundled_spot_script,
    inspect_reaper_install,
    install_reaper_bridge,
    uninstall_reaper_bridge,
)
from audiosearch.browser.reaper_spot import SpotBridgeStatus, SpotResult
from audiosearch.library.scanner import LibraryScanner
from audiosearch.library.store import LibraryStore
from tests.ai.test_browser_ai_ui import _FakeAudition, _wait_until

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _make_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(np.zeros(2400, dtype="<i2").tobytes())


class _SwitchableFakeSpotClient:
    def __init__(self, *, connected: bool = False) -> None:
        self.connected = connected
        self.calls: list[dict[str, object]] = []
        self._cursor = 4.0

    def status(self) -> SpotBridgeStatus:
        if self.connected:
            return SpotBridgeStatus(
                connected=True,
                project_name="Demo.rpp",
                track_name="SFX_Main",
                cursor_sec=self._cursor,
            )
        return SpotBridgeStatus(connected=False)

    def spot(self, path: Path, **kwargs) -> SpotResult:
        self.calls.append({"path": Path(path), **kwargs})
        pos = self._cursor
        self._cursor += 0.5
        return SpotResult(
            ok=True,
            message="ok",
            track_name="SFX_Main",
            position=pos,
            length=0.5,
        )


def test_bundled_spot_script_matches_repo_script():
    bundled = find_bundled_spot_script()
    assert bundled is not None and bundled.is_file()
    repo_script = Path(__file__).resolve().parents[2] / "reaper" / "audiosearch_spot_bridge.lua"
    assert repo_script.is_file()
    assert bundled.read_bytes() == repo_script.read_bytes()


def test_reaper_install_preserves_existing_startup_and_uninstalls_cleanly(tmp_path: Path):
    reaper_dir = tmp_path / "REAPER"
    scripts_dir = reaper_dir / "Scripts"
    scripts_dir.mkdir(parents=True)
    startup_path = scripts_dir / "__startup.lua"
    existing_startup = (
        "-- User custom scripts\n"
        'dofile(reaper.GetResourcePath() .. "/Scripts/reaper_mcp_bridge.lua")\n'
    )
    startup_path.write_text(existing_startup, encoding="utf-8")

    state = install_reaper_bridge(reaper_dir, enable_startup=True)
    assert state.script_installed is True
    assert state.script_up_to_date is True
    assert state.startup_registered is True
    assert state.script_dest == scripts_dir / "audiosearch_spot_bridge.lua"
    assert state.script_dest.is_file()

    startup_after = startup_path.read_text(encoding="utf-8")
    assert "reaper_mcp_bridge.lua" in startup_after
    assert STARTUP_MARKER_BEGIN in startup_after
    assert STARTUP_MARKER_END in startup_after

    # Idempotent re-install must not duplicate marker block
    state2 = install_reaper_bridge(reaper_dir, enable_startup=True)
    assert state2.startup_registered is True
    assert startup_path.read_text(encoding="utf-8").count(STARTUP_MARKER_BEGIN) == 1

    # Disable auto-startup while keeping script
    state_no_auto = install_reaper_bridge(reaper_dir, enable_startup=False)
    assert state_no_auto.script_installed is True
    assert state_no_auto.startup_registered is False
    startup_clean = startup_path.read_text(encoding="utf-8")
    assert "reaper_mcp_bridge.lua" in startup_clean
    assert STARTUP_MARKER_BEGIN not in startup_clean

    # Re-enable then uninstall completely: user lines survive
    install_reaper_bridge(reaper_dir, enable_startup=True)
    state_un = uninstall_reaper_bridge(reaper_dir)
    assert state_un.script_installed is False
    assert state_un.startup_registered is False
    assert not state_un.script_dest.exists()
    assert startup_path.is_file()
    assert "reaper_mcp_bridge.lua" in startup_path.read_text(encoding="utf-8")

    # Fresh REAPER dir with no prior __startup.lua cleans up __startup.lua too
    reaper_fresh = tmp_path / "REAPER_FRESH"
    install_reaper_bridge(reaper_fresh, enable_startup=True)
    assert (reaper_fresh / "Scripts" / "__startup.lua").is_file()
    uninstall_reaper_bridge(reaper_fresh)
    assert inspect_reaper_install(reaper_fresh).script_installed is False
    assert not (reaper_fresh / "Scripts" / "__startup.lua").exists()


def test_reaper_detects_legacy_startup_and_skips_redundant_write(tmp_path: Path):
    bundled = find_bundled_spot_script()
    assert bundled is not None
    reaper_dir = tmp_path / "REAPER_PORTABLE"
    scripts_dir = reaper_dir / "Scripts"
    scripts_dir.mkdir(parents=True)
    dest = scripts_dir / "audiosearch_spot_bridge.lua"
    dest.write_bytes(bundled.read_bytes())

    legacy_startup = (
        'dofile(reaper.GetResourcePath() .. "/Scripts/reaper_mcp_bridge.lua")  -- reaper-mcp\n'
        'pcall(dofile, reaper.GetResourcePath() .. "/Scripts/audiosearch_spot_bridge.lua")  -- audiosearch-spot\n'
    )
    startup_path = scripts_dir / "__startup.lua"
    startup_path.write_text(legacy_startup, encoding="utf-8")

    state = inspect_reaper_install(reaper_dir)
    assert state.script_installed is True
    assert state.script_up_to_date is True
    assert state.startup_registered is True

    # No-op install leaves legacy __startup.lua byte-identical without rewriting
    after = install_reaper_bridge(reaper_dir, enable_startup=True)
    assert after.startup_registered is True
    assert startup_path.read_text(encoding="utf-8") == legacy_startup

    # Uninstall cleanly strips legacy audiosearch line while keeping reaper_mcp_bridge
    uninstalled = uninstall_reaper_bridge(reaper_dir)
    assert uninstalled.script_installed is False
    assert uninstalled.startup_registered is False
    remaining = startup_path.read_text(encoding="utf-8")
    assert "reaper_mcp_bridge.lua" in remaining
    assert "audiosearch_spot_bridge.lua" not in remaining


def test_reaper_setup_dialog_ui_and_one_click_install(tmp_path: Path):
    create_app([])
    reaper_dir = tmp_path / "REAPER_PORTABLE"
    fake_spot = _SwitchableFakeSpotClient(connected=False)

    dlg = ReaperSetupDialog(fake_spot, reaper_dir=reaper_dir)
    try:
        assert "未连接" in dlg.conn_status_label.text()
        assert "未安装" in dlg.script_status_label.text()
        assert dlg.uninstall_btn.isEnabled() is False

        dlg.install_btn.click()
        assert (reaper_dir / "Scripts" / "audiosearch_spot_bridge.lua").is_file()
        assert (reaper_dir / "Scripts" / "__startup.lua").is_file()
        assert "已安装" in dlg.script_status_label.text()
        assert dlg.uninstall_btn.isEnabled() is True

        # Clicking again when already up-to-date gives explicit ready message
        dlg.install_btn.click()
        assert "已是最新版本" in dlg.feedback_label.text()

        fake_spot.connected = True
        dlg.refresh_btn.click()
        assert "已连接" in dlg.conn_status_label.text()
        assert "SFX_Main" in dlg.conn_status_label.text()

        dlg.uninstall_btn.click()
        assert not (reaper_dir / "Scripts" / "audiosearch_spot_bridge.lua").exists()
        assert dlg.uninstall_btn.isEnabled() is False
    finally:
        dlg.close()


def test_reaper_setup_dialog_handles_permission_error_gracefully(tmp_path: Path, monkeypatch):
    create_app([])
    reaper_dir = tmp_path / "LOCKED_REAPER"
    fake_spot = _SwitchableFakeSpotClient(connected=False)

    def _deny(*_args, **_kwargs):
        raise PermissionError(5, "拒绝访问")

    monkeypatch.setattr(reaper_dialog, "install_reaper_bridge", _deny)
    dlg = ReaperSetupDialog(fake_spot, reaper_dir=reaper_dir)
    try:
        dlg.install_btn.click()
        assert "无权限写入" in dlg.feedback_label.text()
        assert "audiosearch_spot_bridge.lua" in dlg.feedback_label.text()
    finally:
        dlg.close()


def test_multi_select_sequential_spot_and_status_button(tmp_path: Path):
    create_app([])
    sfx = tmp_path / "sfx"
    _make_wav(sfx / "a_hit_01.wav")
    _make_wav(sfx / "b_hit_02.wav")

    db_path = tmp_path / "spot_multi.sqlite3"
    store = LibraryStore(db_path)
    root = store.add_root(sfx)
    LibraryScanner(db_path).scan_root(root.id)
    store.close()

    fake_spot = _SwitchableFakeSpotClient(connected=False)
    win = BrowserWindow(db_path, audition=_FakeAudition(), spot_client=fake_spot)
    win.show()
    _wait_until(lambda: win.table_model.rowCount() == 2)

    try:
        assert isinstance(win.status_mode, QPushButton)
        assert "REAPER 未连接" in win.status_mode.text()

        fake_spot.connected = True
        win._refresh_reaper_status()
        assert "REAPER 已连接" in win.status_mode.text()

        rec0 = win.table_model.record_at(0)
        rec1 = win.table_model.record_at(1)
        assert rec0 is not None and rec1 is not None

        win.table_view.spotFilesRequested.emit([rec0.id, rec1.id])
        assert len(fake_spot.calls) == 2
        assert fake_spot.calls[0]["path"] == rec0.full_path
        assert fake_spot.calls[1]["path"] == rec1.full_path
        assert fake_spot.calls[0]["move_cursor"] is True
        assert fake_spot.calls[1]["move_cursor"] is True
        assert "连续插入 REAPER：2 条" in win.statusBar().currentMessage()
    finally:
        win.close()
