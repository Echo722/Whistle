"""Tests for R6b frozen/portable runtime paths, app icon, smoke test, and PyInstaller files."""

from __future__ import annotations

import ast
import os
from pathlib import Path

from PySide6.QtGui import QIcon

from audiosearch.ai.indexer import resolve_hf_cache_root
from audiosearch.browser.__main__ import run_smoke_test
from audiosearch.browser.main_window import create_app
from audiosearch.browser.runtime_paths import (
    default_data_dir,
    default_db_path,
    resolve_app_icon_path,
)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_runtime_paths_dev_vs_frozen_and_env(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("AUDIOSEARCH_DATA_DIR", raising=False)
    monkeypatch.delattr("sys.frozen", raising=False)

    assert default_data_dir() == Path(".local")
    assert default_db_path() == Path(".local") / "library.sqlite3"

    custom = tmp_path / "custom-data"
    monkeypatch.setenv("AUDIOSEARCH_DATA_DIR", str(custom))
    assert default_data_dir() == custom
    assert default_db_path() == custom / "library.sqlite3"

    # Simulated PyInstaller frozen writable portable directory
    monkeypatch.delenv("AUDIOSEARCH_DATA_DIR", raising=False)
    exe_dir = tmp_path / "PortableBundle"
    exe_dir.mkdir(parents=True)
    fake_exe = exe_dir / "AudioSearchTool.exe"
    fake_exe.write_bytes(b"")
    (exe_dir / "portable.flag").write_text("portable=1\n", encoding="utf-8")

    monkeypatch.setattr("sys.frozen", True, raising=False)
    monkeypatch.setattr("sys.executable", str(fake_exe))
    assert default_data_dir() == exe_dir / "data"
    assert default_db_path() == exe_dir / "data" / "library.sqlite3"

    # Simulated read-only Program Files directory (even with portable.flag) falls back to LOCALAPPDATA on Windows
    ro_dir = tmp_path / "ProgramFilesBundle"
    ro_dir.mkdir(parents=True)
    ro_exe = ro_dir / "AudioSearchTool.exe"
    ro_exe.write_bytes(b"")
    (ro_dir / "portable.flag").write_text("portable=1\n", encoding="utf-8")
    local_app = tmp_path / "LocalAppData"
    monkeypatch.setenv("LOCALAPPDATA", str(local_app))
    monkeypatch.setattr("sys.executable", str(ro_exe))
    monkeypatch.setattr("audiosearch.browser.runtime_paths._dir_is_writable", lambda _p: False)
    assert default_data_dir() == local_app / "AudioSearchTool" / "data"


def test_resolve_hf_cache_root_env_and_frozen(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AUDIOSEARCH_MODELS_DIR", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)

    direct = tmp_path / "shared_direct"
    (direct / "hub").mkdir(parents=True)
    monkeypatch.setenv("AUDIOSEARCH_MODELS_DIR", str(direct))
    assert resolve_hf_cache_root() == direct

    nested = tmp_path / "shared_nested"
    (nested / "hf" / "hub").mkdir(parents=True)
    monkeypatch.setenv("AUDIOSEARCH_MODELS_DIR", str(nested))
    assert resolve_hf_cache_root() == nested / "hf"

    monkeypatch.delenv("AUDIOSEARCH_MODELS_DIR", raising=False)
    exe_dir = tmp_path / "Bundle"
    (exe_dir / "models" / "hf" / "hub").mkdir(parents=True)
    fake_exe = exe_dir / "AudioSearchTool.exe"
    fake_exe.write_bytes(b"")
    monkeypatch.setattr("sys.frozen", True, raising=False)
    monkeypatch.setattr("sys.executable", str(fake_exe))

    # Unrelated HF_HOME must not shadow bundled models/hf/hub
    unrelated_hf = tmp_path / "other_hf"
    (unrelated_hf / "hub").mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(unrelated_hf))
    assert resolve_hf_cache_root(tmp_path / "elsewhere" / "library.sqlite3") == exe_dir / "models" / "hf"

    # HF_HOME containing actual Whistle snapshots is preferred over empty bundled hub
    whistle_hf = tmp_path / "whistle_hf"
    (whistle_hf / "hub" / "models--Qwen--Qwen3-Embedding-0.6B" / "snapshots").mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(whistle_hf))
    assert resolve_hf_cache_root(tmp_path / "elsewhere" / "library.sqlite3") == whistle_hf


def test_app_icon_exists_and_loads_in_qt():
    create_app([])
    icon_path = resolve_app_icon_path()
    assert icon_path is not None and icon_path.is_file()
    png_path = icon_path.with_suffix(".png")
    assert png_path.is_file()

    res_dir = icon_path.parent
    assert (res_dir / "whistle.svg").is_file()
    assert (res_dir / "whistle.png").is_file()

    ico_icon = QIcon(str(icon_path))
    png_icon = QIcon(str(png_path))
    assert not ico_icon.isNull()
    assert not png_icon.isNull()


def test_run_smoke_test_reports_core_ok(tmp_path: Path):
    code, report = run_smoke_test(tmp_path / "smoke.sqlite3")
    assert code == 0
    assert report["store_ok"] is True
    assert report["reaper_script"] is True
    assert report["app_icon"] is True
    assert report["core"]["PySide6"] is True
    assert report["core"]["sounddevice"] is True
    assert report["core"]["soundfile"] is True
    assert report["core"]["pedalboard"] is True
    assert report["core"]["mutagen"] is True
    if "cuda_available" in report["ai"]:
        assert isinstance(report["ai"].get("cuda_usable"), bool)


def test_packaging_spec_and_launcher_are_valid():
    root = Path(__file__).resolve().parents[2]
    spec_path = root / "packaging" / "audiosearch.spec"
    launcher_path = root / "packaging" / "launcher.py"
    gen_icon_path = root / "packaging" / "generate_icon.py"
    build_ps1 = root / "packaging" / "build_windows.ps1"

    assert spec_path.is_file()
    assert launcher_path.is_file()
    assert gen_icon_path.is_file()
    assert build_ps1.is_file()

    spec_src = spec_path.read_text(encoding="utf-8")
    ast.parse(spec_src, filename=str(spec_path))
    ast.parse(launcher_path.read_text(encoding="utf-8"), filename=str(launcher_path))
    ast.parse(gen_icon_path.read_text(encoding="utf-8"), filename=str(gen_icon_path))

    for required in (
        "_sounddevice_data",
        "_soundfile_data",
        "pedalboard",
        "open_clip",
        "torch",
        "audiosearch_spot_bridge.lua",
        "audiosearch.ico",
        'name="Whistle"',
    ):
        assert required in spec_src

    ps1_src = build_ps1.read_text(encoding="utf-8")
    assert "models--Qwen--Qwen3-Embedding-0.6B" in ps1_src
    assert "models--laion--clap2-large-audioxl-s1B" in ps1_src
    assert "tar.exe" in ps1_src
    assert "Whistle.exe" in ps1_src
    assert "Whistle-win64.zip" in ps1_src
