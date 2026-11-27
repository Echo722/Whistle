"""Behavioral tests for in-app CUDA 12.8 runtime inspection, staging, swap script, and UI."""

from __future__ import annotations

import hashlib
import os
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from audiosearch.browser.cuda_updater import (
    CudaRuntimeInfo,
    CudaWheelSpec,
    download_and_stage_cuda_upgrade,
    inspect_cuda_runtime,
)
from audiosearch.browser.main_window import create_app
from audiosearch.browser.update_dialog import UpdateDialog
from audiosearch.browser.updater import UpdateCheckResult

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _make_fake_torch(*, arch_list: list[str], gpu_name: str, capability: tuple[int, int], works: bool, version: str):
    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_arch_list() -> list[str]:
            return arch_list

        @staticmethod
        def get_device_name(_idx: int = 0) -> str:
            return gpu_name

        @staticmethod
        def get_device_capability(_idx: int = 0) -> tuple[int, int]:
            return capability

    class _FakeVal:
        def __init__(self, v: float) -> None:
            self._v = v

        def item(self) -> float:
            return self._v

    class _FakeTensor:
        def __add__(self, _other):
            return self

        def sum(self):
            return _FakeVal(2.0)

    def _ones(*_args, **_kwargs):
        if not works:
            raise RuntimeError("CUDA error: no kernel image is available for execution on the device")
        return _FakeTensor()

    return SimpleNamespace(
        __version__=version,
        float32="float32",
        cuda=_FakeCuda(),
        ones=_ones,
    )


def _build_fake_wheel(out_path: Path, entries: dict[str, bytes]) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return hashlib.sha256(out_path.read_bytes()).hexdigest().lower()


def test_inspect_cuda_runtime_states(tmp_path: Path) -> None:
    app_root = tmp_path / "Whistle"
    internal = app_root / "_internal"
    (internal / "torch").mkdir(parents=True)
    (internal / "torch-2.9.1+cu126.dist-info").mkdir(parents=True)

    # 1. RTX 5070 Ti on cu126 -> kernel fails -> upgrade_recommended=True
    t5070 = _make_fake_torch(
        arch_list=["sm_50", "sm_86", "sm_90"],
        gpu_name="NVIDIA GeForce RTX 5070 Ti",
        capability=(12, 0),
        works=False,
        version="2.9.1+cu126",
    )
    info_5070 = inspect_cuda_runtime(app_root=app_root, frozen_override=True, torch_mod=t5070)
    assert info_5070.can_upgrade_in_place is True
    assert info_5070.supports_sm120 is False
    assert info_5070.upgrade_available is True
    assert info_5070.upgrade_recommended is True
    assert info_5070.gpu_capability == "sm_120"

    # 2. RTX 3060 on cu126 -> kernel works -> optional upgrade_available=True, recommended=False
    t3060 = _make_fake_torch(
        arch_list=["sm_50", "sm_86", "sm_90"],
        gpu_name="NVIDIA GeForce RTX 3060",
        capability=(8, 6),
        works=True,
        version="2.9.1+cu126",
    )
    info_3060 = inspect_cuda_runtime(app_root=app_root, frozen_override=True, torch_mod=t3060)
    assert info_3060.upgrade_available is True
    assert info_3060.upgrade_recommended is False

    # 3. cu128 already installed -> supports_sm120=True, upgrade_available=False
    t_cu128 = _make_fake_torch(
        arch_list=["sm_70", "sm_86", "sm_120"],
        gpu_name="NVIDIA GeForce RTX 5070 Ti",
        capability=(12, 0),
        works=True,
        version="2.9.1+cu128",
    )
    info_128 = inspect_cuda_runtime(app_root=app_root, frozen_override=True, torch_mod=t_cu128)
    assert info_128.supports_sm120 is True
    assert info_128.upgrade_available is False
    assert info_128.upgrade_recommended is False


def test_download_stage_security_and_real_cmd_swap(tmp_path: Path) -> None:
    app_root = tmp_path / "Whistle"
    internal = app_root / "_internal"
    (internal / "torch" / "lib").mkdir(parents=True)
    (internal / "torch" / "lib" / "torch_cuda.dll").write_bytes(b"OLD_CU126")
    (internal / "torch-2.9.1+cu126.dist-info").mkdir(parents=True)
    (internal / "torch-2.9.1+cu126.dist-info" / "METADATA").write_text("cu126", encoding="utf-8")
    (internal / "torchvision").mkdir(parents=True)
    (internal / "torchvision" / "_C.pyd").write_bytes(b"OLD_TV")
    (internal / "torchvision-0.24.1+cu126.dist-info").mkdir(parents=True)
    (internal / "torchvision-0.24.1+cu126.dist-info" / "METADATA").write_text("cu126", encoding="utf-8")

    # Preserve user data and models
    (app_root / "data").mkdir(parents=True)
    (app_root / "data" / "library.sqlite3").write_text("KEEP_DB", encoding="utf-8")
    (app_root / "models" / "hf").mkdir(parents=True)
    (app_root / "models" / "hf" / "weights.bin").write_text("KEEP_MODELS", encoding="utf-8")

    wheel_repo = tmp_path / "wheel_repo"
    torch_whl = wheel_repo / "torch-2.9.1+cu128-cp312-cp312-win_amd64.whl"
    tv_whl = wheel_repo / "torchvision-0.24.1+cu128-cp312-cp312-win_amd64.whl"

    sha_torch = _build_fake_wheel(
        torch_whl,
        {
            "torch/__init__.py": b"# cu128\n",
            "torch/lib/torch_cuda.dll": b"NEW_CU128",
            "torch-2.9.1+cu128.dist-info/METADATA": b"Version: 2.9.1+cu128\n",
        },
    )
    sha_tv = _build_fake_wheel(
        tv_whl,
        {
            "torchvision/__init__.py": b"# tv cu128\n",
            "torchvision/_C.pyd": b"NEW_TV",
            "torchvision-0.24.1+cu128.dist-info/METADATA": b"Version: 0.24.1+cu128\n",
        },
    )

    good_specs = (
        CudaWheelSpec("torch", "torch-2.9.1+cu128-cp312-cp312-win_amd64.whl", "torch-2.9.1+cu128-cp312-cp312-win_amd64.whl", sha_torch, 1024, "torch", "torch-"),
        CudaWheelSpec("torchvision", "torchvision-0.24.1+cu128-cp312-cp312-win_amd64.whl", "torchvision-0.24.1+cu128-cp312-cp312-win_amd64.whl", sha_tv, 512, "torchvision", "torchvision-"),
    )

    # 1. SHA-256 mismatch is rejected
    bad_sha_specs = (
        CudaWheelSpec("torch", "torch-2.9.1+cu128-cp312-cp312-win_amd64.whl", "torch-2.9.1+cu128-cp312-cp312-win_amd64.whl", "0" * 64, 1024, "torch", "torch-"),
    )
    with pytest.raises(ValueError, match="SHA-256"):
        download_and_stage_cuda_upgrade(
            app_root=app_root,
            base_url=wheel_repo.as_uri(),
            specs=bad_sha_specs,
            min_free_bytes=1,
            target_pid=0,
        )

    # 2. Zip-slip inside wheel is rejected
    evil_whl = wheel_repo / "evil.whl"
    sha_evil = _build_fake_wheel(evil_whl, {"../../escape.txt": b"boom"})
    evil_specs = (
        CudaWheelSpec("torch", "evil.whl", "evil.whl", sha_evil, 128, "torch", "torch-"),
    )
    with pytest.raises(ValueError, match="越界路径"):
        download_and_stage_cuda_upgrade(
            app_root=app_root,
            base_url=wheel_repo.as_uri(),
            specs=evil_specs,
            min_free_bytes=1,
            target_pid=0,
        )

    # 3. Valid download + staging + real cmd.exe swap
    staged = download_and_stage_cuda_upgrade(
        app_root=app_root,
        base_url=wheel_repo.as_uri(),
        specs=good_specs,
        min_free_bytes=1,
        target_pid=0,
        relaunch_exe=None,
    )
    assert staged.apply_script_path.is_file()
    assert b"\r\n" in staged.apply_script_path.read_bytes()

    proc = subprocess.run(
        ["cmd.exe", "/c", str(staged.apply_script_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr

    # Verify cu128 replaced cu126 and user data/models remain untouched
    assert (internal / "torch" / "lib" / "torch_cuda.dll").read_bytes() == b"NEW_CU128"
    assert (internal / "torchvision" / "_C.pyd").read_bytes() == b"NEW_TV"
    assert (internal / "torch-2.9.1+cu128.dist-info" / "METADATA").is_file()
    assert not (internal / "torch-2.9.1+cu126.dist-info").exists()
    assert not (internal / "_cuda_backup").exists()
    assert (app_root / "data" / "library.sqlite3").read_text(encoding="utf-8") == "KEEP_DB"
    assert (app_root / "models" / "hf" / "weights.bin").read_text(encoding="utf-8") == "KEEP_MODELS"


def test_update_dialog_cuda_runtime_ui_states(tmp_path: Path) -> None:
    create_app([])
    up_to_date = UpdateCheckResult(status="up_to_date", current_version="0.1.1", current_runtime_abi=1, message="当前已是最新版本 v0.1.1")

    rec_info = CudaRuntimeInfo(
        is_frozen=True,
        internal_dir=tmp_path / "_internal",
        can_upgrade_in_place=True,
        torch_dist_version="2.9.1+cu126",
        supports_sm120=False,
        cuda_available=True,
        cuda_usable=False,
        gpu_name="NVIDIA GeForce RTX 5070 Ti",
        gpu_capability="sm_120",
        probe_reason="no kernel image",
        upgrade_recommended=True,
        upgrade_available=True,
    )
    dlg_rec = UpdateDialog(initial_result=up_to_date, initial_cuda_info=rec_info)
    assert dlg_rec.cuda_action_btn.isHidden() is False
    assert dlg_rec.cuda_action_btn.text() == "升级 CUDA 12.8 运行库"
    assert "RTX 5070 Ti" in dlg_rec.cuda_status_label.text()
    dlg_rec.close()

    opt_info = CudaRuntimeInfo(
        is_frozen=True,
        internal_dir=tmp_path / "_internal",
        can_upgrade_in_place=True,
        torch_dist_version="2.9.1+cu126",
        supports_sm120=False,
        cuda_available=True,
        cuda_usable=True,
        gpu_name="NVIDIA GeForce RTX 3060",
        gpu_capability="sm_86",
        probe_reason="CUDA device available",
        upgrade_recommended=False,
        upgrade_available=True,
    )
    dlg_opt = UpdateDialog(initial_result=up_to_date, initial_cuda_info=opt_info)
    assert dlg_opt.cuda_action_btn.isHidden() is False
    assert dlg_opt.cuda_action_btn.text() == "按需升级 CUDA 12.8"
    dlg_opt.close()

    ready_info = CudaRuntimeInfo(
        is_frozen=True,
        internal_dir=tmp_path / "_internal",
        can_upgrade_in_place=True,
        torch_dist_version="2.9.1+cu128",
        supports_sm120=True,
        cuda_available=True,
        cuda_usable=True,
        gpu_name="NVIDIA GeForce RTX 5070 Ti",
        gpu_capability="sm_120",
        probe_reason="CUDA device available",
        upgrade_recommended=False,
        upgrade_available=False,
    )
    dlg_ready = UpdateDialog(initial_result=up_to_date, initial_cuda_info=ready_info)
    assert dlg_ready.cuda_action_btn.isHidden() is True
    assert "已就绪" in dlg_ready.cuda_status_label.text()
    dlg_ready.close()
