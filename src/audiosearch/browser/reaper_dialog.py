"""One-click setup and live health dialog for REAPER Spot bridge."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QCloseEvent, QGuiApplication, QHideEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from audiosearch.browser.reaper_setup import (
    SCRIPT_FILENAME,
    ReaperBridgeInstallState,
    default_reaper_resource_dir,
    find_bundled_spot_script,
    inspect_reaper_install,
    install_reaper_bridge,
    uninstall_reaper_bridge,
)
from audiosearch.browser.reaper_spot import ReaperSpotClient


class ReaperSetupDialog(QDialog):
    """One-click installer, startup registration, and live diagnostics for REAPER Spot."""

    def __init__(
        self,
        spot_client: ReaperSpotClient,
        *,
        reaper_dir: Path | None = None,
        source_script: Path | None = None,
        on_import_shortcuts: Callable[[Path], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("REAPER 桥接配置与一键安装")
        self.resize(620, 440)
        self._spot_client = spot_client
        self._source_script = source_script
        self._on_import_shortcuts = on_import_shortcuts
        self._startup_pref_initialized = False
        self._last_state: ReaperBridgeInstallState | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # Live connection card
        conn_card = QFrame()
        conn_card.setObjectName("panel")
        conn_card.setStyleSheet(
            "QFrame#panel { border: 1px solid #34322E; border-radius: 8px; padding: 8px; }"
        )
        conn_layout = QHBoxLayout(conn_card)
        conn_layout.setContentsMargins(10, 8, 10, 8)
        conn_layout.setSpacing(10)

        self.conn_status_label = QLabel("○ REAPER 未连接")
        self.conn_status_label.setWordWrap(True)
        self.conn_status_label.setStyleSheet("background: transparent; color: #A5A096;")
        self.refresh_btn = QPushButton("刷新状态")
        self.refresh_btn.setProperty("variant", "secondary")
        self.refresh_btn.clicked.connect(self.refresh_state)

        conn_layout.addWidget(self.conn_status_label, 1)
        conn_layout.addWidget(self.refresh_btn, 0)
        root.addWidget(conn_card)

        # Resource directory selector
        dir_label = QLabel("REAPER 资源目录（便携版 REAPER 可手动浏览选择根目录）")
        dir_label.setStyleSheet(
            "background: transparent; color: #ECE9E2; font-weight: 600; font-size: 12px;"
        )
        root.addWidget(dir_label)

        dir_row = QHBoxLayout()
        dir_row.setSpacing(8)
        initial_dir = Path(reaper_dir) if reaper_dir is not None else default_reaper_resource_dir()
        self.reaper_dir_edit = QLineEdit(str(initial_dir))
        self.reaper_dir_edit.textChanged.connect(lambda _t: self.refresh_state())
        self.browse_btn = QPushButton("浏览…")
        self.browse_btn.setProperty("variant", "secondary")
        self.browse_btn.clicked.connect(self._on_browse_dir)
        dir_row.addWidget(self.reaper_dir_edit, 1)
        dir_row.addWidget(self.browse_btn, 0)
        root.addLayout(dir_row)

        # Script install status + options
        info_card = QFrame()
        info_card.setObjectName("panel")
        info_card.setStyleSheet(
            "QFrame#panel { border: 1px solid #34322E; border-radius: 8px; padding: 10px; }"
        )
        info_layout = QVBoxLayout(info_card)
        info_layout.setContentsMargins(10, 8, 10, 8)
        info_layout.setSpacing(8)

        self.script_status_label = QLabel("")
        self.script_status_label.setWordWrap(True)
        self.script_status_label.setStyleSheet("background: transparent; color: #ECE9E2;")
        self.script_status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        info_layout.addWidget(self.script_status_label)

        self.startup_checkbox = QCheckBox(
            "随 REAPER 启动自动加载（写入 __startup.lua，不影响已有启动脚本）"
        )
        self.startup_checkbox.setChecked(True)
        info_layout.addWidget(self.startup_checkbox)
        root.addWidget(info_card)

        # Action buttons
        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        self.install_btn = QPushButton("一键安装 / 更新脚本")
        self.install_btn.setProperty("variant", "primary")
        self.install_btn.clicked.connect(self._on_install)

        self.copy_path_btn = QPushButton("复制脚本路径")
        self.copy_path_btn.setProperty("variant", "secondary")
        self.copy_path_btn.clicked.connect(self._on_copy_path)

        self.open_dir_btn = QPushButton("打开脚本目录")
        self.open_dir_btn.setProperty("variant", "secondary")
        self.open_dir_btn.clicked.connect(self._on_open_dir)

        self.uninstall_btn = QPushButton("卸载脚本")
        self.uninstall_btn.setProperty("variant", "chip")
        self.uninstall_btn.clicked.connect(self._on_uninstall)

        action_row.addWidget(self.install_btn)
        action_row.addWidget(self.copy_path_btn)
        action_row.addWidget(self.open_dir_btn)
        action_row.addStretch(1)
        action_row.addWidget(self.uninstall_btn)
        root.addLayout(action_row)

        self.feedback_label = QLabel("")
        self.feedback_label.setWordWrap(True)
        self.feedback_label.setStyleSheet("background: transparent; color: #A5A096; font-size: 12px;")
        root.addWidget(self.feedback_label)

        # Instructions box
        tip_box = QLabel(
            "使用说明：\n"
            "1. 点击「一键安装 / 更新脚本」，把 audiosearch_spot_bridge.lua 写入 REAPER Scripts 目录；\n"
            "2. 如果 REAPER 已经打开，要么重启一次 REAPER（通过 __startup.lua 自动加载），\n"
            "   要么在 REAPER 里按 ? 打开 Actions → New action → Load ReaScript… 选择复制的脚本路径并运行一次；\n"
            "3. 上方状态变为「● REAPER 已连接」后，在列表里双击或按 Enter 即可直接插到当前光标处（支持多选连续插入）。"
        )
        tip_box.setWordWrap(True)
        tip_box.setStyleSheet(
            "background-color: #181715; border: 1px solid #34322E; border-radius: 7px; "
            "padding: 10px; color: #A5A096; font-size: 12px;"
        )
        root.addWidget(tip_box, 1)

        footer = QHBoxLayout()
        if on_import_shortcuts is not None:
            self.import_shortcuts_btn = QPushButton("导入 REAPER 快捷方式…")
            self.import_shortcuts_btn.setProperty("variant", "secondary")
            self.import_shortcuts_btn.clicked.connect(
                lambda _c=False: on_import_shortcuts(self._current_reaper_dir())
            )
            footer.addWidget(self.import_shortcuts_btn)
        else:
            self.import_shortcuts_btn = None
        footer.addStretch(1)
        self.close_btn = QPushButton("关闭")
        self.close_btn.setProperty("variant", "secondary")
        self.close_btn.clicked.connect(self.accept)
        footer.addWidget(self.close_btn)
        root.addLayout(footer)

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(1200)
        self._poll_timer.timeout.connect(self.refresh_state)
        self._poll_timer.start()

        self.refresh_state()

    def _current_reaper_dir(self) -> Path:
        raw = self.reaper_dir_edit.text().strip()
        return Path(raw) if raw else default_reaper_resource_dir()

    @Slot()
    def refresh_state(self) -> None:
        st = self._spot_client.status()
        if st.connected:
            proj = st.project_name or "(未命名工程)"
            track = st.track_name or "(新建轨道)"
            self.conn_status_label.setText(
                f"● REAPER 已连接 · 工程：{proj} · 目标轨：{track} · 光标：{st.cursor_sec:.2f}s"
            )
            self.conn_status_label.setStyleSheet(
                "background: transparent; color: #7EC89B; font-weight: 600;"
            )
        else:
            self.conn_status_label.setText(
                "○ REAPER 未连接（可直接拖拽导入，或安装下方脚本启用双击/Enter 插入）"
            )
            self.conn_status_label.setStyleSheet("background: transparent; color: #A5A096;")

        reaper_dir = self._current_reaper_dir()
        info = inspect_reaper_install(reaper_dir, source_script=self._source_script)
        self._last_state = info

        if not self._startup_pref_initialized:
            if info.script_installed:
                self.startup_checkbox.setChecked(info.startup_registered)
            else:
                self.startup_checkbox.setChecked(True)
            self._startup_pref_initialized = True

        dir_note = "已检测到目录" if info.reaper_dir_exists else "未检测到该目录（点击安装会自动创建）"
        if info.script_installed and info.script_up_to_date:
            script_badge = "已安装（与当前内置版本一致）"
        elif info.script_installed:
            script_badge = "已安装（有新版可更新）"
        else:
            script_badge = "未安装"

        startup_badge = "已注册开机自启" if info.startup_registered else "未注册自启"
        self.script_status_label.setText(
            f"目录状态：{dir_note}\n"
            f"脚本状态：{script_badge} · {startup_badge}\n"
            f"目标路径：{info.script_dest}"
        )
        self.uninstall_btn.setEnabled(info.script_installed or info.startup_registered)
        bundled = self._source_script or find_bundled_spot_script()
        self.open_dir_btn.setEnabled(
            info.script_dest.parent.is_dir() or (bundled is not None and bundled.parent.is_dir())
        )

    @Slot()
    def _on_browse_dir(self) -> None:
        start_dir = self._current_reaper_dir()
        if not start_dir.exists():
            start_dir = Path.home()
        chosen = QFileDialog.getExistingDirectory(
            self,
            "选择 REAPER 资源目录",
            str(start_dir),
        )
        if chosen:
            self._startup_pref_initialized = False
            self.reaper_dir_edit.setText(chosen)

    @Slot()
    def _on_install(self) -> None:
        target = self._current_reaper_dir()
        enable_startup = self.startup_checkbox.isChecked()
        before = inspect_reaper_install(target, source_script=self._source_script)
        was_ready = (
            before.script_installed
            and before.script_up_to_date
            and before.startup_registered == enable_startup
        )
        try:
            state = install_reaper_bridge(
                target,
                enable_startup=enable_startup,
                source_script=self._source_script,
            )
        except PermissionError:
            self.feedback_label.setText(
                "无权限写入该目录（可能受系统或终端安全软件保护）。"
                "可点「打开脚本目录」或「复制脚本路径」，将内置 audiosearch_spot_bridge.lua "
                "手动复制到 REAPER 的 Scripts 目录，或以管理员权限运行后再试。"
            )
            self.feedback_label.setStyleSheet(
                "background: transparent; color: #E38566; font-size: 12px;"
            )
            return
        except Exception as exc:
            self.feedback_label.setText(f"安装失败：{exc}")
            self.feedback_label.setStyleSheet(
                "background: transparent; color: #E38566; font-size: 12px;"
            )
            return

        self._startup_pref_initialized = True
        if was_ready:
            msg = (
                "当前脚本已是最新版本且已配置自启，直接打开 REAPER 即可自动连接。"
                if enable_startup
                else "当前脚本已是最新版本（未启用自启）。"
            )
        else:
            startup_note = "并已写入 __startup.lua" if state.startup_registered else "（未启用自启）"
            msg = f"已安装脚本到：{state.script_dest} {startup_note}"
        self.feedback_label.setText(msg)
        self.feedback_label.setStyleSheet("background: transparent; color: #7EC89B; font-size: 12px;")
        self.refresh_state()

    @Slot()
    def _on_copy_path(self) -> None:
        if self._last_state is not None and self._last_state.script_installed:
            target_path = self._last_state.script_dest
        else:
            bundled = self._source_script or find_bundled_spot_script()
            if bundled is not None and bundled.is_file():
                target_path = bundled
            else:
                target_path = (
                    self._last_state.script_dest
                    if self._last_state is not None
                    else self._current_reaper_dir() / "Scripts" / SCRIPT_FILENAME
                )
        QGuiApplication.clipboard().setText(str(target_path))
        self.feedback_label.setText(f"已复制脚本路径：{target_path}")
        self.feedback_label.setStyleSheet("background: transparent; color: #ECE9E2; font-size: 12px;")

    @Slot()
    def _on_open_dir(self) -> None:
        dest_dir = (
            self._last_state.script_dest.parent
            if self._last_state is not None
            else self._current_reaper_dir() / "Scripts"
        )
        bundled = self._source_script or find_bundled_spot_script()
        if dest_dir.is_dir():
            open_target = dest_dir
            note = f"已打开脚本目录：{dest_dir}"
        elif bundled is not None and bundled.parent.is_dir():
            open_target = bundled.parent
            note = f"REAPER 脚本目录尚未创建，已打开内置脚本目录：{open_target}（可手动复制过去）"
        else:
            self.feedback_label.setText("脚本目录尚未创建，请先点击「一键安装 / 更新脚本」。")
            self.feedback_label.setStyleSheet(
                "background: transparent; color: #A5A096; font-size: 12px;"
            )
            return

        if os.name == "nt":
            os.startfile(str(open_target))  # noqa: S606
        self.feedback_label.setText(note)
        self.feedback_label.setStyleSheet("background: transparent; color: #ECE9E2; font-size: 12px;")

    @Slot()
    def _on_uninstall(self) -> None:
        target = self._current_reaper_dir()
        try:
            uninstall_reaper_bridge(target, source_script=self._source_script)
        except PermissionError:
            self.feedback_label.setText("无权限修改该目录，请检查目录权限或以管理员身份运行后再卸载。")
            self.feedback_label.setStyleSheet(
                "background: transparent; color: #E38566; font-size: 12px;"
            )
            return
        except Exception as exc:
            self.feedback_label.setText(f"卸载失败：{exc}")
            self.feedback_label.setStyleSheet(
                "background: transparent; color: #E38566; font-size: 12px;"
            )
            return

        self.feedback_label.setText("已移除桥接脚本与 __startup.lua 自动注册项（已保留你的其他启动脚本）。")
        self.feedback_label.setStyleSheet("background: transparent; color: #A5A096; font-size: 12px;")
        self.refresh_state()

    def hideEvent(self, event: QHideEvent) -> None:  # noqa: N802
        self._poll_timer.stop()
        super().hideEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self._poll_timer.stop()
        super().closeEvent(event)


__all__ = ["ReaperSetupDialog"]
