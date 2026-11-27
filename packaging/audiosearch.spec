# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for AudioSearchTool standalone Windows desktop distribution."""

from __future__ import annotations

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules, copy_metadata

ROOT = Path(SPECPATH).resolve().parent  # noqa: F821 - injected by PyInstaller
WITH_AI = os.environ.get("AUDIOSEARCH_BUILD_AI", "1").strip().lower() not in ("0", "false", "no")

datas: list[tuple[str, str]] = [
    (str(ROOT / "reaper" / "audiosearch_spot_bridge.lua"), "reaper"),
    (str(ROOT / "src" / "audiosearch" / "browser" / "resources"), "audiosearch/browser/resources"),
]
binaries: list[tuple[str, str]] = []
hiddenimports: list[str] = [
    *collect_submodules("audiosearch"),
    *collect_submodules("mutagen"),
]

for pkg in ("_sounddevice_data", "_soundfile_data", "pedalboard"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

if WITH_AI:
    for pkg in ("torch", "soxr", "tokenizers", "safetensors"):
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h

    hiddenimports += ["httpx2", "httpcore2", "huggingface_hub"]
    datas += collect_data_files("open_clip")
    hiddenimports += collect_submodules("open_clip")

    datas += collect_data_files("transformers")
    hiddenimports += collect_submodules("transformers.models.qwen3")
    hiddenimports += collect_submodules("transformers.models.qwen2")

    for meta_pkg in (
        "transformers",
        "tokenizers",
        "huggingface-hub",
        "httpx2",
        "httpcore2",
        "safetensors",
        "torch",
        "open-clip-torch",
    ):
        datas += copy_metadata(meta_pkg)

excludes = [
    "fastapi",
    "uvicorn",
    "pytest",
    "ruff",
    "tkinter",
    "matplotlib",
    "IPython",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.Qt3DCore",
    "PySide6.QtQuick",
    "PySide6.QtQml",
]
if not WITH_AI:
    excludes += ["torch", "torchvision", "transformers", "open_clip", "soxr"]

a = Analysis(  # noqa: F821
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)  # noqa: F821

icon_file = ROOT / "src" / "audiosearch" / "browser" / "resources" / "audiosearch.ico"

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Whistle",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(icon_file) if icon_file.is_file() else None,
)

coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Whistle",
)
