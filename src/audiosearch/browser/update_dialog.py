"""Background check, <1 MB patch download, and rollback dialog for Whistle."""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from audiosearch.browser.cuda_updater import (
    CUDA_UPGRADE_MIN_DRIVER,
    CudaRuntimeInfo,
    CudaStageResult,
    download_and_stage_cuda_upgrade,
    inspect_cuda_runtime,
    resolve_cuda_wheel_base_url,
)
from audiosearch.browser.runtime_paths import is_frozen
from audiosearch.browser.updater import (
    DEFAULT_GITHUB_REPO,
    PatchInstallResult,
    RUNTIME_ABI,
    UpdateCheckResult,
    UpdateManifest,
    can_rollback,
    check_for_updates,
    download_and_stage_patch,
    effective_version,
    is_running_from_patch,
    rollback_patch,
)


class _CheckSignals(QObject):
    finished = Signal(object)  # UpdateCheckResult


class _CheckWorker(QRunnable):
    def __init__(
        self,
        signals: _CheckSignals,
        *,
        repo: str,
        manifest_url: str | None,
        app_root: Path | None,
    ) -> None:
        super().__init__()
        self._signals = signals
        self._repo = repo
        self._manifest_url = manifest_url
        self._app_root = app_root

    @Slot()
    def run(self) -> None:
        res = check_for_updates(
            repo=self._repo,
            manifest_url=self._manifest_url,
            app_root=self._app_root,
        )
        try:
            self._signals.finished.emit(res)
        except RuntimeError:
            pass


class _DownloadSignals(QObject):
    progress = Signal(int, int)
    finished = Signal(object)  # PatchInstallResult
    failed = Signal(str)


class _DownloadWorker(QRunnable):
    def __init__(
        self,
        signals: _DownloadSignals,
        manifest: UpdateManifest,
        *,
        app_root: Path | None,
        current_version: str,
    ) -> None:
        super().__init__()
        self._signals = signals
        self._manifest = manifest
        self._app_root = app_root
        self._current_version = current_version

    @Slot()
    def run(self) -> None:
        try:
            res = download_and_stage_patch(
                self._manifest,
                app_root=self._app_root,
                current_version=self._current_version,
                progress_cb=self._on_progress,
            )
        except Exception as exc:
            try:
                self._signals.failed.emit(str(exc))
            except RuntimeError:
                pass
            return
        try:
            self._signals.finished.emit(res)
        except RuntimeError:
            pass

    def _on_progress(self, downloaded: int, total: int) -> None:
        try:
            self._signals.progress.emit(int(downloaded), int(total))
        except RuntimeError:
            pass


class _CudaInspectSignals(QObject):
    finished = Signal(object)  # CudaRuntimeInfo


class _CudaInspectWorker(QRunnable):
    def __init__(self, signals: _CudaInspectSignals, *, app_root: Path | None) -> None:
        super().__init__()
        self._signals = signals
        self._app_root = app_root

    @Slot()
    def run(self) -> None:
        info = inspect_cuda_runtime(app_root=self._app_root)
        try:
            self._signals.finished.emit(info)
        except RuntimeError:
            pass


class _CudaUpgradeSignals(QObject):
    progress = Signal(int, int, str)
    finished = Signal(object)  # CudaStageResult
    failed = Signal(str)


class _CudaUpgradeWorker(QRunnable):
    def __init__(
        self,
        signals: _CudaUpgradeSignals,
        *,
        app_root: Path | None,
        cancel_event: threading.Event,
        relaunch_exe: Path | None,
    ) -> None:
        super().__init__()
        self._signals = signals
        self._app_root = app_root
        self._cancel_event = cancel_event
        self._relaunch_exe = relaunch_exe

    @Slot()
    def run(self) -> None:
        try:
            res = download_and_stage_cuda_upgrade(
                app_root=self._app_root,
                cancel_event=self._cancel_event,
                progress_cb=self._on_progress,
                target_pid=os.getpid(),
                relaunch_exe=self._relaunch_exe,
            )
        except Exception as exc:
            try:
                self._signals.failed.emit(str(exc))
            except RuntimeError:
                pass
            return
        try:
            self._signals.finished.emit(res)
        except RuntimeError:
            pass

    def _on_progress(self, done: int, total: int, stage: str) -> None:
        try:
            self._signals.progress.emit(int(done), int(total), str(stage))
        except RuntimeError:
            pass


class UpdateDialog(QDialog):
    """Show Whistle version, release notes, and one-click <1 MB GitHub patch updates."""

    restartRequested = Signal()
    cudaUpgradeRestartRequested = Signal(str)

    def __init__(
        self,
        *,
        repo: str = DEFAULT_GITHUB_REPO,
        manifest_url: str | None = None,
        app_root: Path | None = None,
        initial_result: UpdateCheckResult | None = None,
        initial_cuda_info: CudaRuntimeInfo | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("检查更新 · Whistle")
        self.resize(580, 480)
        self._repo = repo
        self._manifest_url = manifest_url
        self._app_root = app_root
        self._pool = QThreadPool.globalInstance()
        self._latest_result: UpdateCheckResult | None = initial_result
        self._staged_restart = False
        self._cuda_info: CudaRuntimeInfo | None = initial_cuda_info
        self._staged_cuda_script: Path | None = None
        self._cuda_cancel_event: threading.Event | None = None
        self._cuda_downloading = False

        self._check_signals = _CheckSignals(self)
        self._check_signals.finished.connect(self._on_check_finished)
        self._dl_signals = _DownloadSignals(self)
        self._dl_signals.progress.connect(self._on_download_progress)
        self._dl_signals.finished.connect(self._on_download_finished)
        self._dl_signals.failed.connect(self._on_download_failed)
        self._cuda_inspect_signals = _CudaInspectSignals(self)
        self._cuda_inspect_signals.finished.connect(self._on_cuda_inspect_finished)
        self._cuda_dl_signals = _CudaUpgradeSignals(self)
        self._cuda_dl_signals.progress.connect(self._on_cuda_progress)
        self._cuda_dl_signals.finished.connect(self._on_cuda_finished)
        self._cuda_dl_signals.failed.connect(self._on_cuda_failed)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        ver_card = QFrame()
        ver_card.setObjectName("panel")
        ver_card.setStyleSheet(
            "QFrame#panel { border: 1px solid #34322E; border-radius: 8px; padding: 8px; }"
        )
        ver_layout = QVBoxLayout(ver_card)
        ver_layout.setContentsMargins(10, 8, 10, 8)
        ver_layout.setSpacing(4)

        cur_ver = effective_version(self._app_root)
        mode_note = "热更新补丁运行中" if is_running_from_patch() else "内置基础版运行中"
        self.current_label = QLabel(
            f"当前版本：Whistle v{cur_ver}（底座 ABI {RUNTIME_ABI} · {mode_note}）\n"
            f"更新仓库：github.com/{self._repo}"
        )
        self.current_label.setWordWrap(True)
        self.current_label.setStyleSheet("background: transparent; color: #ECE9E2; font-size: 12px;")
        self.status_label = QLabel("准备检查更新…")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("background: transparent; color: #A5A096; font-size: 12px;")
        ver_layout.addWidget(self.current_label)
        ver_layout.addWidget(self.status_label)
        root.addWidget(ver_card)

        self.cuda_card = QFrame()
        self.cuda_card.setObjectName("panel")
        self.cuda_card.setStyleSheet(
            "QFrame#panel { border: 1px solid #34322E; border-radius: 8px; padding: 8px; }"
        )
        cuda_layout = QHBoxLayout(self.cuda_card)
        cuda_layout.setContentsMargins(10, 8, 10, 8)
        cuda_layout.setSpacing(10)

        cuda_text_col = QVBoxLayout()
        cuda_text_col.setSpacing(2)
        cuda_title = QLabel("GPU 运行库（CUDA 12.8 · RTX 50 系支持）")
        cuda_title.setStyleSheet("background: transparent; color: #ECE9E2; font-size: 12px; font-weight: 600;")
        self.cuda_status_label = QLabel("正在检测本地 PyTorch CUDA 运行库…")
        self.cuda_status_label.setWordWrap(True)
        self.cuda_status_label.setStyleSheet("background: transparent; color: #A5A096; font-size: 11px;")
        cuda_text_col.addWidget(cuda_title)
        cuda_text_col.addWidget(self.cuda_status_label)
        cuda_layout.addLayout(cuda_text_col, 1)

        self.cuda_cancel_btn = QPushButton("取消")
        self.cuda_cancel_btn.setProperty("variant", "secondary")
        self.cuda_cancel_btn.setVisible(False)
        self.cuda_cancel_btn.clicked.connect(self._on_cuda_cancel)

        self.cuda_action_btn = QPushButton("升级 CUDA 12.8")
        self.cuda_action_btn.setProperty("variant", "chip")
        self.cuda_action_btn.setVisible(False)
        self.cuda_action_btn.clicked.connect(self._on_cuda_action_clicked)

        cuda_layout.addWidget(self.cuda_cancel_btn)
        cuda_layout.addWidget(self.cuda_action_btn)
        self.cuda_card.setVisible(initial_cuda_info is not None or is_frozen())
        root.addWidget(self.cuda_card)

        notes_lbl = QLabel("更新说明（仅更新代码与脚本，不改动 data/ 曲库、向量索引与 models/）")
        notes_lbl.setStyleSheet("background: transparent; color: #A5A096; font-size: 11px;")
        root.addWidget(notes_lbl)

        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setReadOnly(True)
        self.notes_edit.setPlaceholderText("暂无更新说明。")
        root.addWidget(self.notes_edit, 1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        root.addWidget(self.progress_bar)

        footer = QHBoxLayout()
        footer.setSpacing(8)
        self.recheck_btn = QPushButton("重新检查")
        self.recheck_btn.setProperty("variant", "secondary")
        self.recheck_btn.clicked.connect(self.start_check)

        self.rollback_btn = QPushButton("回退上一版")
        self.rollback_btn.setProperty("variant", "chip")
        self.rollback_btn.setEnabled(can_rollback(self._app_root))
        self.rollback_btn.clicked.connect(self._on_rollback)

        self.open_release_btn = QPushButton("打开发布页")
        self.open_release_btn.setProperty("variant", "secondary")
        self.open_release_btn.setVisible(False)
        self.open_release_btn.clicked.connect(self._on_open_release)

        self.action_btn = QPushButton("立即下载并更新")
        self.action_btn.setProperty("variant", "primary")
        self.action_btn.setEnabled(False)
        self.action_btn.clicked.connect(self._on_action_clicked)

        self.close_btn = QPushButton("关闭")
        self.close_btn.setProperty("variant", "secondary")
        self.close_btn.clicked.connect(self.reject)

        footer.addWidget(self.recheck_btn)
        footer.addWidget(self.rollback_btn)
        footer.addStretch(1)
        footer.addWidget(self.open_release_btn)
        footer.addWidget(self.action_btn)
        footer.addWidget(self.close_btn)
        root.addLayout(footer)

        if initial_result is not None:
            self._apply_result(initial_result)
        else:
            self.start_check()

        if initial_cuda_info is not None:
            self._apply_cuda_info(initial_cuda_info)
        elif is_frozen():
            self._pool.start(_CudaInspectWorker(self._cuda_inspect_signals, app_root=self._app_root))

    @Slot()
    def start_check(self) -> None:
        self.recheck_btn.setEnabled(False)
        self.action_btn.setEnabled(False)
        self.status_label.setText("正在检查 GitHub 最新版本…")
        self.status_label.setStyleSheet("background: transparent; color: #A5A096; font-size: 12px;")
        self._pool.start(
            _CheckWorker(
                self._check_signals,
                repo=self._repo,
                manifest_url=self._manifest_url,
                app_root=self._app_root,
            )
        )

    @Slot(object)
    def _on_check_finished(self, result: UpdateCheckResult) -> None:
        self.recheck_btn.setEnabled(True)
        self._apply_result(result)

    def _apply_result(self, result: UpdateCheckResult) -> None:
        self._latest_result = result
        self._staged_restart = False
        self.open_release_btn.setVisible(False)
        self.action_btn.setText("立即下载并更新")

        manifest = result.manifest
        if manifest is not None and manifest.notes:
            self.notes_edit.setPlainText(manifest.notes)
        elif result.status == "up_to_date":
            self.notes_edit.setPlainText("当前版本已包含最新代码与修复。")

        if result.status == "patch_available" and manifest is not None:
            size_kb = max(1, manifest.patch_size // 1024) if manifest.patch_size else 0
            size_note = f"（约 {size_kb} KB）" if size_kb else ""
            self.status_label.setText(f"● 发现增量热更新 v{manifest.version}{size_note}")
            self.status_label.setStyleSheet(
                "background: transparent; color: #7EC89B; font-weight: 600; font-size: 12px;"
            )
            self.action_btn.setEnabled(True)
            self.open_release_btn.setVisible(bool(manifest.release_url))
        elif result.status == "full_update_required" and manifest is not None:
            self.status_label.setText(result.message)
            self.status_label.setStyleSheet(
                "background: transparent; color: #D99B57; font-weight: 600; font-size: 12px;"
            )
            self.action_btn.setEnabled(False)
            self.open_release_btn.setVisible(bool(manifest.release_url))
        elif result.status == "up_to_date":
            self.status_label.setText(f"✓ {result.message}")
            self.status_label.setStyleSheet(
                "background: transparent; color: #7EC89B; font-size: 12px;"
            )
            self.action_btn.setEnabled(False)
        else:
            self.status_label.setText(f"! {result.message}")
            self.status_label.setStyleSheet(
                "background: transparent; color: #E38566; font-size: 12px;"
            )
            self.action_btn.setEnabled(False)

    @Slot()
    def _on_action_clicked(self) -> None:
        if self._staged_restart:
            self.restartRequested.emit()
            self.accept()
            return
        res = self._latest_result
        if res is None or res.manifest is None or res.status != "patch_available":
            return
        self.recheck_btn.setEnabled(False)
        self.rollback_btn.setEnabled(False)
        self.action_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"正在下载并校验补丁 v{res.manifest.version}…")
        self.status_label.setStyleSheet("background: transparent; color: #ECE9E2; font-size: 12px;")
        self._pool.start(
            _DownloadWorker(
                self._dl_signals,
                res.manifest,
                app_root=self._app_root,
                current_version=res.current_version,
            )
        )

    @Slot(int, int)
    def _on_download_progress(self, downloaded: int, total: int) -> None:
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(min(downloaded, total))
        else:
            self.progress_bar.setRange(0, 0)

    @Slot(object)
    def _on_download_finished(self, result: PatchInstallResult) -> None:
        self.progress_bar.setVisible(False)
        self.recheck_btn.setEnabled(True)
        self.rollback_btn.setEnabled(can_rollback(self._app_root))
        self._staged_restart = True
        self.action_btn.setText("立即重启生效")
        self.action_btn.setEnabled(True)
        self.status_label.setText(f"✓ 补丁 v{result.version} 已就绪，重启后立即生效。")
        self.status_label.setStyleSheet(
            "background: transparent; color: #7EC89B; font-weight: 600; font-size: 12px;"
        )

    @Slot(str)
    def _on_download_failed(self, msg: str) -> None:
        self.progress_bar.setVisible(False)
        self.recheck_btn.setEnabled(True)
        self.rollback_btn.setEnabled(can_rollback(self._app_root))
        self.action_btn.setEnabled(
            bool(self._latest_result and self._latest_result.status == "patch_available")
        )
        self.status_label.setText(f"! 更新失败：{msg}")
        self.status_label.setStyleSheet("background: transparent; color: #E38566; font-size: 12px;")

    @Slot()
    def _on_open_release(self) -> None:
        if self._latest_result and self._latest_result.manifest and self._latest_result.manifest.release_url:
            QDesktopServices.openUrl(QUrl(self._latest_result.manifest.release_url))

    @Slot()
    def _on_rollback(self) -> None:
        if rollback_patch(self._app_root):
            self._staged_restart = True
            self.rollback_btn.setEnabled(can_rollback(self._app_root))
            self.action_btn.setText("立即重启生效")
            self.action_btn.setEnabled(True)
            self.status_label.setText("✓ 已回退到上一版本，请点击「立即重启生效」。")
            self.status_label.setStyleSheet(
                "background: transparent; color: #7EC89B; font-weight: 600; font-size: 12px;"
            )

    @Slot(object)
    def _on_cuda_inspect_finished(self, info: CudaRuntimeInfo) -> None:
        self._apply_cuda_info(info)

    def _apply_cuda_info(self, info: CudaRuntimeInfo) -> None:
        self._cuda_info = info
        if not info.is_frozen and not info.can_upgrade_in_place and self._staged_cuda_script is None:
            if self.cuda_card.isVisible() is False:
                return
        self.cuda_card.setVisible(True)
        ver_tag = info.torch_dist_version or "未知版本"

        if info.supports_sm120:
            self.cuda_status_label.setText(
                f"✓ 已就绪（{ver_tag} · 原生支持 RTX 50/40/30/20 系 GPU）"
            )
            self.cuda_status_label.setStyleSheet(
                "background: transparent; color: #7EC89B; font-size: 11px;"
            )
            self.cuda_action_btn.setVisible(False)
            self.cuda_cancel_btn.setVisible(False)
            return

        if info.upgrade_recommended:
            gpu_desc = info.gpu_name or "NVIDIA GPU"
            if info.gpu_capability:
                gpu_desc = f"{gpu_desc} / {info.gpu_capability}"
            self.cuda_status_label.setText(
                f"⚠ 检测到不兼容的 GPU（{gpu_desc}），当前运行库为 {ver_tag}（已临时回退 CPU）；"
                "可在线原地升级 CUDA 12.8 运行库（约 2.7 GB，不改动 data/ 与 models/）。"
            )
            self.cuda_status_label.setStyleSheet(
                "background: transparent; color: #D99B57; font-size: 11px;"
            )
            self.cuda_action_btn.setText("升级 CUDA 12.8 运行库")
            self.cuda_action_btn.setProperty("variant", "primary")
            self.cuda_action_btn.setVisible(True)
            self.cuda_action_btn.setEnabled(True)
            return

        if info.upgrade_available:
            self.cuda_status_label.setText(
                f"当前运行库为 {ver_tag}（RTX 20/30/40 系与 A 卡/CPU 无需升级；如需升级 CUDA 12.8 可按需操作）。"
            )
            self.cuda_status_label.setStyleSheet(
                "background: transparent; color: #A5A096; font-size: 11px;"
            )
            self.cuda_action_btn.setText("按需升级 CUDA 12.8")
            self.cuda_action_btn.setProperty("variant", "chip")
            self.cuda_action_btn.setVisible(True)
            self.cuda_action_btn.setEnabled(True)
            return

        self.cuda_status_label.setText(f"当前运行库：{ver_tag}（当前目录或环境不支持原地替换）")
        self.cuda_status_label.setStyleSheet(
            "background: transparent; color: #A5A096; font-size: 11px;"
        )
        self.cuda_action_btn.setVisible(False)

    @Slot()
    def _on_cuda_action_clicked(self) -> None:
        if self._staged_cuda_script is not None:
            self.cudaUpgradeRestartRequested.emit(str(self._staged_cuda_script))
            self.accept()
            return
        if self._cuda_downloading:
            return

        base_url = resolve_cuda_wheel_base_url()
        reply = QMessageBox.question(
            self,
            "升级 CUDA 12.8 运行库",
            f"将从 PyTorch 官方源（{base_url}）下载约 2.7 GB 运行库组件（torch 2.9.1+cu128）：\n\n"
            "• 全程不改动 data/ 曲库、向量索引与 models/ 模型；\n"
            f"• 需保持磁盘至少 8 GB 可用空间，且 NVIDIA 驱动版本 >= {CUDA_UPGRADE_MIN_DRIVER}；\n"
            "• 下载校验完成后将重启 Whistle 并原地完成替换。\n\n"
            "是否立即开始下载？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._cuda_downloading = True
        self._cuda_cancel_event = threading.Event()
        self.recheck_btn.setEnabled(False)
        self.rollback_btn.setEnabled(False)
        self.action_btn.setEnabled(False)
        self.cuda_action_btn.setEnabled(False)
        self.cuda_cancel_btn.setVisible(True)
        self.cuda_cancel_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.cuda_status_label.setText("正在连接 PyTorch 下载源…")
        self.cuda_status_label.setStyleSheet(
            "background: transparent; color: #ECE9E2; font-size: 11px;"
        )
        relaunch_exe = Path(sys.executable).resolve() if (is_frozen() and self._app_root is None) else None
        self._pool.start(
            _CudaUpgradeWorker(
                self._cuda_dl_signals,
                app_root=self._app_root,
                cancel_event=self._cuda_cancel_event,
                relaunch_exe=relaunch_exe,
            )
        )

    @Slot()
    def _on_cuda_cancel(self) -> None:
        if self._cuda_cancel_event is not None:
            self._cuda_cancel_event.set()
            self.cuda_cancel_btn.setEnabled(False)
            self.cuda_status_label.setText("正在取消下载…")

    @Slot(int, int, str)
    def _on_cuda_progress(self, done: int, total: int, stage: str) -> None:
        if total > 0:
            pct = min(100, int((done * 100) / total))
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(pct)
            done_gb = done / (1024**3)
            total_gb = total / (1024**3)
            self.cuda_status_label.setText(f"{stage} {done_gb:.2f} / {total_gb:.2f} GB ({pct}%)")
        else:
            self.progress_bar.setRange(0, 0)
            self.cuda_status_label.setText(stage)

    @Slot(object)
    def _on_cuda_finished(self, result: CudaStageResult) -> None:
        self._cuda_downloading = False
        self._cuda_cancel_event = None
        self._staged_cuda_script = result.apply_script_path
        self.progress_bar.setVisible(False)
        self.cuda_cancel_btn.setVisible(False)
        self.recheck_btn.setEnabled(True)
        self.rollback_btn.setEnabled(can_rollback(self._app_root))
        if self._latest_result and self._latest_result.status == "patch_available":
            self.action_btn.setEnabled(True)
        self.cuda_action_btn.setText("立即重启并完成升级")
        self.cuda_action_btn.setProperty("variant", "primary")
        self.cuda_action_btn.setVisible(True)
        self.cuda_action_btn.setEnabled(True)
        self.cuda_status_label.setText(
            "✓ CUDA 12.8 运行库已下载校验就绪，点击右侧按钮后将重启并自动完成替换。"
        )
        self.cuda_status_label.setStyleSheet(
            "background: transparent; color: #7EC89B; font-weight: 600; font-size: 11px;"
        )

    @Slot(str)
    def _on_cuda_failed(self, msg: str) -> None:
        self._cuda_downloading = False
        self._cuda_cancel_event = None
        self.progress_bar.setVisible(False)
        self.cuda_cancel_btn.setVisible(False)
        self.recheck_btn.setEnabled(True)
        self.rollback_btn.setEnabled(can_rollback(self._app_root))
        if self._latest_result and self._latest_result.status == "patch_available":
            self.action_btn.setEnabled(True)
        if self._cuda_info is not None:
            self._apply_cuda_info(self._cuda_info)
        self.cuda_status_label.setText(f"! CUDA 升级未完成：{msg}")
        self.cuda_status_label.setStyleSheet(
            "background: transparent; color: #E38566; font-size: 11px;"
        )

    def reject(self) -> None:
        if self._cuda_cancel_event is not None:
            self._cuda_cancel_event.set()
        super().reject()

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._cuda_cancel_event is not None:
            self._cuda_cancel_event.set()
        super().closeEvent(event)


__all__ = ["UpdateDialog"]
