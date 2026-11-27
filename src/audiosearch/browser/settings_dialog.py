"""Unified Settings dialog for UI scaling, audio output selection, and search preferences."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from audiosearch.browser import theme

_TAB_ORDER: tuple[tuple[str, str], ...] = (
    ("appearance", "外观与显示"),
    ("audio", "试听与音频"),
    ("search", "检索与集成"),
)


class SettingsDialog(QDialog):
    """Categorized preferences dialog with live preview and zero-restart UI scaling."""

    def __init__(
        self,
        browser_window: Any,
        *,
        initial_tab: str = "appearance",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent or browser_window)
        self._window = browser_window
        self._syncing = False
        self._tab_keys: list[str] = [k for k, _ in _TAB_ORDER]
        self.nav_buttons: dict[str, QPushButton] = {}
        self.scale_buttons: dict[str, QPushButton] = {}

        self.setWindowTitle("设置")
        self.setMinimumSize(680, 460)
        self.resize(760, 540)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        # Left category navigation rail
        nav_rail = QFrame(self)
        nav_rail.setObjectName("settingsSidebar")
        nav_rail.setFixedWidth(176)
        nav_layout = QVBoxLayout(nav_rail)
        nav_layout.setContentsMargins(14, 18, 14, 14)
        nav_layout.setSpacing(6)

        rail_title = QLabel("偏好设置")
        rail_title.setProperty("role", "inspectorTitle")
        rail_sub = QLabel("外观、声卡与检索")
        rail_sub.setProperty("role", "muted")
        nav_layout.addWidget(rail_title)
        nav_layout.addWidget(rail_sub)
        nav_layout.addSpacing(10)

        for tab_key, tab_label in _TAB_ORDER:
            btn = QPushButton(tab_label)
            btn.setCheckable(True)
            btn.setProperty("variant", "settingsNav")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _c=False, k=tab_key: self.select_tab(k))
            self.nav_buttons[tab_key] = btn
            nav_layout.addWidget(btn)
        nav_layout.addStretch(1)

        # Right stacked pages (each wrapped in QScrollArea so large/xlarge never clips)
        self.stack = QStackedWidget(self)
        self.stack.addWidget(self._wrap_scroll(self._build_appearance_page()))
        self.stack.addWidget(self._wrap_scroll(self._build_audio_page()))
        self.stack.addWidget(self._wrap_scroll(self._build_search_page()))

        body.addWidget(nav_rail)
        body.addWidget(self.stack, 1)
        root.addLayout(body, 1)

        # Bottom action footer
        footer = QFrame(self)
        footer.setObjectName("topBar")
        footer.setStyleSheet("QFrame#topBar { border-top: 1px solid #34322E; border-bottom: none; }")
        foot_layout = QHBoxLayout(footer)
        foot_layout.setContentsMargins(16, 10, 16, 10)
        foot_layout.setSpacing(8)

        self.reset_defaults_btn = QPushButton("恢复默认")
        self.reset_defaults_btn.setProperty("variant", "ghost")
        self.reset_defaults_btn.setToolTip("恢复界面缩放为「标准」并切回「跟随系统默认」声卡")
        self.reset_defaults_btn.clicked.connect(self._on_reset_defaults)

        self.close_btn = QPushButton("完成")
        self.close_btn.setProperty("variant", "primary")
        self.close_btn.setDefault(True)
        self.close_btn.clicked.connect(self.accept)

        foot_layout.addWidget(self.reset_defaults_btn)
        foot_layout.addStretch(1)
        foot_layout.addWidget(self.close_btn)
        root.addWidget(footer)

        self.select_tab(initial_tab)
        self._sync_from_window()

    @staticmethod
    def _wrap_scroll(inner: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(inner)
        return scroll

    @staticmethod
    def _make_card(title: str, subtitle: str = "") -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("settingsCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        title_lbl = QLabel(title)
        title_lbl.setProperty("role", "inspectorTitle")
        layout.addWidget(title_lbl)
        if subtitle:
            sub_lbl = QLabel(subtitle)
            sub_lbl.setProperty("role", "muted")
            sub_lbl.setWordWrap(True)
            layout.addWidget(sub_lbl)
        return card, layout

    def _build_appearance_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)

        scale_card, scale_box = self._make_card(
            "界面字号与密度",
            "即时切换全局字体、表格行高与侧栏间距，无需重启软件（默认「标准」保持原有紧凑观感）。",
        )
        grid = QGridLayout()
        grid.setContentsMargins(0, 2, 0, 2)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        for idx, spec in enumerate(theme.UI_SCALE_PRESETS.values()):
            btn = QPushButton(f"{spec.label}\n{spec.description}")
            btn.setCheckable(True)
            btn.setProperty("variant", "scaleCard")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _c=False, k=spec.key: self._on_scale_clicked(k))
            self.scale_buttons[spec.key] = btn
            grid.addWidget(btn, idx // 2, idx % 2)
        scale_box.addLayout(grid)
        layout.addWidget(scale_card)

        view_card, view_box = self._make_card("界面视图选项")
        self.inspector_cb = QCheckBox("显示右侧检查器")
        self.spectral_cb = QCheckBox("开启 REAPER 风格频谱着色波形")
        self.hide_offline_cb = QCheckBox("列表中隐藏离线文件")

        self.inspector_cb.toggled.connect(
            lambda checked: self._sync_toggle(self._window.inspector_btn, checked)
        )
        self.spectral_cb.toggled.connect(
            lambda checked: self._sync_toggle(self._window.spectral_peaks_btn, checked)
        )
        self.hide_offline_cb.toggled.connect(
            lambda checked: self._sync_toggle(self._window.filter_online_cb, checked)
        )

        view_box.addWidget(self.inspector_cb)
        view_box.addWidget(self.spectral_cb)
        view_box.addWidget(self.hide_offline_cb)
        layout.addWidget(view_card)
        layout.addStretch(1)
        return page

    def _build_audio_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)

        dev_card, dev_box = self._make_card(
            "输出声卡设备",
            "可选择跟随系统默认输出，或固定绑定外置声卡/耳机通道；切换后即时重连且无需重启。",
        )
        dev_row = QHBoxLayout()
        dev_row.setContentsMargins(0, 0, 0, 0)
        dev_row.setSpacing(8)
        self.device_combo = QComboBox()
        self.device_combo.currentIndexChanged.connect(self._on_device_combo_changed)
        self.refresh_devices_btn = QPushButton("刷新设备")
        self.refresh_devices_btn.setProperty("variant", "secondary")
        self.refresh_devices_btn.setToolTip("重新扫描已连接或热插拔的音频输出设备")
        self.refresh_devices_btn.clicked.connect(self._on_refresh_devices_clicked)
        dev_row.addWidget(self.device_combo, 1)
        dev_row.addWidget(self.refresh_devices_btn)
        dev_box.addLayout(dev_row)

        self.device_status_label = QLabel("")
        self.device_status_label.setProperty("role", "muted")
        self.device_status_label.setWordWrap(True)
        dev_box.addWidget(self.device_status_label)

        rate_note = QLabel(
            "采样率说明：试听采用 WASAPI 共享模式自动对齐系统混音器采样率，"
            "拖拽或插入 DAW 时始终交付原始音频文件，绝不改变源文件采样率。"
        )
        rate_note.setProperty("role", "muted")
        rate_note.setWordWrap(True)
        dev_box.addWidget(rate_note)
        layout.addWidget(dev_card)

        play_card, play_box = self._make_card("试听与导出习惯")
        self.autoplay_cb = QCheckBox("点击或上下键切换文件时自动播放")
        self.reset_switch_cb = QCheckBox("切换文件试听时自动重置变调与变速")
        self.bake_drag_cb = QCheckBox("拖出到 DAW 或插入 REAPER 时应用当前变调 / 变速")

        self.autoplay_cb.toggled.connect(
            lambda checked: self._sync_toggle(self._window.autoplay_btn, checked)
        )
        self.reset_switch_cb.toggled.connect(
            lambda checked: self._sync_toggle(self._window.reset_switch_btn, checked)
        )
        self.bake_drag_cb.toggled.connect(
            lambda checked: self._sync_toggle(self._window.bake_drag_btn, checked)
        )

        play_box.addWidget(self.autoplay_cb)
        play_box.addWidget(self.reset_switch_cb)
        play_box.addWidget(self.bake_drag_cb)
        layout.addWidget(play_card)
        layout.addStretch(1)
        return page

    def _build_search_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)

        search_card, search_box = self._make_card("检索偏好")
        limit_row = QHBoxLayout()
        limit_row.setContentsMargins(0, 0, 0, 0)
        limit_row.setSpacing(10)
        limit_lbl = QLabel("AI 语义 / 找同类单次返回上限")
        limit_lbl.setProperty("role", "muted")
        self.ai_limit_combo = QComboBox()
        for opt_val in (200, 500, 1000, 2000, 5000):
            self.ai_limit_combo.addItem(f"{opt_val} 条", opt_val)
        self.ai_limit_combo.currentIndexChanged.connect(self._on_ai_limit_changed)
        limit_row.addWidget(limit_lbl)
        limit_row.addStretch(1)
        limit_row.addWidget(self.ai_limit_combo)
        search_box.addLayout(limit_row)

        self.exclude_same_pkg_cb = QCheckBox("AI 找同类默认排除同目录 / 同素材包")
        self.global_scope_cb = QCheckBox("默认使用全库范围搜索（而非仅当前目录）")
        self.exclude_same_pkg_cb.toggled.connect(
            lambda checked: self._sync_toggle(self._window.exclude_same_pkg_btn, checked)
        )
        self.global_scope_cb.toggled.connect(
            lambda checked: self._sync_toggle(self._window.scope_btn, checked)
        )
        search_box.addWidget(self.exclude_same_pkg_cb)
        search_box.addWidget(self.global_scope_cb)
        layout.addWidget(search_card)

        int_card, int_box = self._make_card("REAPER 桥接与快捷键")
        reaper_row = QHBoxLayout()
        reaper_row.setContentsMargins(0, 0, 0, 0)
        reaper_row.setSpacing(10)
        self.reaper_status_label = QLabel("")
        self.reaper_status_label.setProperty("role", "muted")
        self.reaper_status_label.setWordWrap(True)
        self.reaper_setup_btn = QPushButton("打开 REAPER 桥接配置…")
        self.reaper_setup_btn.setProperty("variant", "secondary")
        self.reaper_setup_btn.clicked.connect(self._on_open_reaper_setup)
        reaper_row.addWidget(self.reaper_status_label, 1)
        reaper_row.addWidget(self.reaper_setup_btn)
        int_box.addLayout(reaper_row)

        shortcuts_lbl = QLabel(
            "常用快捷键：Space 播放/暂停 · Esc 清除选区/停止 · Ctrl+F 搜索框 · "
            "Ctrl+, 设置 · Alt+←/→ 前进/后退 · Enter 插入 REAPER"
        )
        shortcuts_lbl.setProperty("role", "muted")
        shortcuts_lbl.setWordWrap(True)
        int_box.addWidget(shortcuts_lbl)
        layout.addWidget(int_card)
        layout.addStretch(1)
        return page

    def select_tab(self, tab_key: str) -> None:
        key = tab_key if tab_key in self._tab_keys else self._tab_keys[0]
        self.stack.setCurrentIndex(self._tab_keys.index(key))
        for k, btn in self.nav_buttons.items():
            blocked = btn.blockSignals(True)
            btn.setChecked(k == key)
            btn.blockSignals(blocked)

    def _sync_toggle(self, target_btn: Any, checked: bool) -> None:
        if self._syncing:
            return
        target_btn.setChecked(bool(checked))

    def _sync_from_window(self) -> None:
        self._syncing = True
        try:
            active_scale = getattr(self._window, "_ui_scale", theme.current_ui_scale())
            for k, btn in self.scale_buttons.items():
                btn.setChecked(k == active_scale)

            self.inspector_cb.setChecked(self._window.inspector_btn.isChecked())
            self.spectral_cb.setChecked(self._window.spectral_peaks_btn.isChecked())
            self.hide_offline_cb.setChecked(self._window.filter_online_cb.isChecked())

            self.autoplay_cb.setChecked(self._window.autoplay_btn.isChecked())
            self.reset_switch_cb.setChecked(self._window.reset_switch_btn.isChecked())
            self.bake_drag_cb.setChecked(self._window.bake_drag_btn.isChecked())

            limit_val = int(self._window.ai_limit_combo.currentData() or 500)
            limit_idx = self.ai_limit_combo.findData(limit_val)
            if limit_idx >= 0:
                self.ai_limit_combo.setCurrentIndex(limit_idx)
            self.exclude_same_pkg_cb.setChecked(self._window.exclude_same_pkg_btn.isChecked())
            self.global_scope_cb.setChecked(self._window.scope_btn.isChecked())

            self._populate_output_devices()
            self._refresh_reaper_label()
        finally:
            self._syncing = False

    def _populate_output_devices(self) -> None:
        blocked = self.device_combo.blockSignals(True)
        try:
            self.device_combo.clear()
            self.device_combo.addItem("跟随系统默认", "")
            audition = self._window.audition
            devices = []
            list_fn = getattr(audition, "list_output_devices", None)
            if callable(list_fn):
                devices = list(list_fn() or ())
            else:
                from audiosearch.audition.sink import list_output_devices

                devices = list_output_devices()

            known_names: set[str] = set()
            for dev in devices:
                rate = int(getattr(dev, "default_samplerate", 48000) or 48000)
                rate_str = f"{rate // 1000}kHz" if rate % 1000 == 0 else f"{rate / 1000:.1f}kHz"
                tag = " · 默认" if getattr(dev, "is_default", False) else ""
                self.device_combo.addItem(f"{dev.name} ({rate_str}{tag})", dev.name)
                known_names.add(dev.name)

            requested = str(getattr(self._window, "_output_device_name", "") or "").strip()
            if requested and requested not in known_names:
                # Preserve the user's saved selection even when currently unplugged.
                self.device_combo.addItem(f"{requested} (未连接，已回退默认)", requested)

            sel_idx = self.device_combo.findData(requested)
            self.device_combo.setCurrentIndex(sel_idx if sel_idx >= 0 else 0)
        finally:
            self.device_combo.blockSignals(blocked)
        self._refresh_audio_status_label()

    def _refresh_audio_status_label(self) -> None:
        audition = self._window.audition
        err = getattr(audition, "device_error", None)
        rate = int(getattr(audition, "output_rate_hz", 48000) or 48000)
        rate_str = f"{rate // 1000}kHz" if rate % 1000 == 0 else f"{rate / 1000:.1f}kHz"
        active_name = str(getattr(audition, "active_device_name", "") or "").strip()
        fallback = bool(getattr(audition, "device_fallback", False))
        if err:
            self.device_status_label.setText(f"当前状态：声卡异常 · {err}")
        else:
            dev_desc = active_name or "系统默认输出设备"
            suffix = "（原指定设备未连接，已自动回退）" if fallback else ""
            self.device_status_label.setText(
                f"当前活动设备：{dev_desc} · {rate_str} · WASAPI 共享模式{suffix}"
            )

    def _refresh_reaper_label(self) -> None:
        try:
            st = self._window._spot_client.status()
            if st.connected:
                track = st.track_name or "(新建轨道)"
                self.reaper_status_label.setText(f"当前状态：已连接 REAPER · 目标轨 {track}")
            else:
                self.reaper_status_label.setText("当前状态：REAPER 未连接（可直接拖拽音频进轨道）")
        except Exception:
            self.reaper_status_label.setText("当前状态：REAPER 未连接")

    @Slot(str)
    def _on_scale_clicked(self, scale_key: str) -> None:
        if self._syncing:
            return
        self._window.apply_ui_scale(scale_key)
        self._sync_from_window()

    @Slot(int)
    def _on_device_combo_changed(self, _index: int) -> None:
        if self._syncing:
            return
        dev_name = str(self.device_combo.currentData() or "").strip()
        self._window.set_output_device(dev_name, refresh_portaudio=False)
        self._refresh_audio_status_label()

    @Slot()
    def _on_refresh_devices_clicked(self) -> None:
        dev_name = str(self.device_combo.currentData() or "").strip()
        self._window.set_output_device(dev_name, refresh_portaudio=True)
        self._populate_output_devices()

    @Slot(int)
    def _on_ai_limit_changed(self, _index: int) -> None:
        if self._syncing:
            return
        limit_val = int(self.ai_limit_combo.currentData() or 500)
        idx = self._window.ai_limit_combo.findData(limit_val)
        if idx >= 0:
            self._window.ai_limit_combo.setCurrentIndex(idx)

    @Slot()
    def _on_open_reaper_setup(self) -> None:
        self._window._open_reaper_setup_dialog()
        self._refresh_reaper_label()

    @Slot()
    def _on_reset_defaults(self) -> None:
        self._window.apply_ui_scale(theme.DEFAULT_UI_SCALE)
        self._window.set_output_device("", refresh_portaudio=False)
        self._sync_from_window()


__all__ = ["SettingsDialog"]
