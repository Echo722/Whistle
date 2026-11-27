"""Right-side inspector displaying MediaEye cover art and embedded metadata."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QPointF, QRectF, QRunnable, Qt, QThreadPool, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from audiosearch.audioio.video import VideoProbeInfo, is_video_path
from audiosearch.audition.covers import CoverImage, CoverResolver
from audiosearch.browser import theme
from audiosearch.browser.formatting import (
    format_bit_depth,
    format_channels,
    format_duration,
    format_sample_rate,
    format_size,
)
from audiosearch.browser.theme import mono_font
from audiosearch.browser.video_preview import (
    VideoCanvasWidget,
    VideoFrameWorker,
    VideoPopoutDialog,
)
from audiosearch.library.metadata_fields import extract_inspector_summary
from audiosearch.library.store import FileRecord

COVER_SIZE = 200


def _build_cover_placeholder(size: int, caption: str) -> QPixmap:
    """Paint a subtle warm vector waveform frame when no embedded cover is present."""
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    cx = size / 2.0
    cy = size / 2.0 - 10.0

    frame = QRectF(cx - 24.0, cy - 20.0, 48.0, 40.0)
    p.setPen(QPen(QColor("#45423B"), 1.3))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(frame, 8.0, 8.0)

    pts = [
        QPointF(cx - 16.0, cy + 1.0),
        QPointF(cx - 10.0, cy + 1.0),
        QPointF(cx - 6.0, cy - 9.0),
        QPointF(cx - 1.0, cy + 10.0),
        QPointF(cx + 4.0, cy - 7.0),
        QPointF(cx + 9.0, cy + 4.0),
        QPointF(cx + 12.0, cy + 1.0),
        QPointF(cx + 16.0, cy + 1.0),
    ]
    p.setPen(QPen(QColor("#6E685E"), 1.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    for i in range(len(pts) - 1):
        p.drawLine(pts[i], pts[i + 1])

    font = p.font()
    font.setPointSize(9)
    p.setFont(font)
    p.setPen(QColor("#736E66"))
    p.drawText(QRectF(8.0, cy + 26.0, size - 16.0, 22.0), Qt.AlignmentFlag.AlignCenter, caption)
    p.end()
    return pm


class _CoverSignals(QObject):
    resolved = Signal(int, object)  # (generation, CoverImage | None)


class _CoverTask(QRunnable):
    def __init__(
        self,
        resolver: CoverResolver,
        audio_path: Path,
        generation: int,
        signals: _CoverSignals,
    ) -> None:
        super().__init__()
        self._resolver = resolver
        self._audio_path = audio_path
        self._generation = generation
        self._signals = signals

    @Slot()
    def run(self) -> None:
        try:
            cover = self._resolver.resolve(self._audio_path)
        except Exception:
            cover = None
        try:
            self._signals.resolved.emit(self._generation, cover)
        except RuntimeError:
            pass


class InspectorPanel(QFrame):
    """Fixed-width inspector showing cover art, format summary, user notes, and metadata."""

    favoriteToggled = Signal(int, bool)  # (file_id, favorite)
    noteChanged = Signal(int, str)  # (file_id, note_text)
    editMetadataRequested = Signal(int)  # file_id
    navigateToFolderRequested = Signal(int)  # file_id
    statusMessageRequested = Signal(str)  # ephemeral status message
    togglePlayRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        spec = theme.ui_scale_spec()
        self.setObjectName("inspectorPanel")
        self.setMinimumWidth(spec.inspector_min_w)
        self.setMaximumWidth(spec.inspector_max_w)
        self._key_labels: list[QLabel] = []
        self._mono_pills: list[QLabel] = []

        self._resolver = CoverResolver()
        self._pool = QThreadPool.globalInstance()
        self._cover_gen = 0
        self._video_gen = 0
        self._current_is_video = False
        self._current_title = ""
        self._current_video_meta = ""
        self._last_video_time = 0.0
        self._current_file_id: int | None = None
        self._current_full_path: Path | None = None
        self._current_folder_path: str = ""
        self._suppress_note = False
        self._note_dirty = False
        self._note_timer = QTimer(self)
        self._note_timer.setSingleShot(True)
        self._note_timer.setInterval(500)
        self._note_timer.timeout.connect(self.flush_pending_note)
        self._signals = _CoverSignals()
        self._signals.resolved.connect(self._on_cover_resolved)

        self._video_worker = VideoFrameWorker(self)
        self._video_worker.videoLoaded.connect(self._on_video_loaded)
        self._video_worker.frameReady.connect(self._on_video_frame_ready)
        self._video_worker.videoFailed.connect(self._on_video_failed)
        self._popout_dialog: VideoPopoutDialog | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(9)

        self.media_stack = QStackedWidget(self)
        self.media_stack.setFixedHeight(COVER_SIZE)

        cover_wrap = QWidget(self.media_stack)
        cover_wrap.setStyleSheet("background: transparent;")
        cover_box = QVBoxLayout(cover_wrap)
        cover_box.setContentsMargins(0, 0, 0, 0)
        cover_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cover_label = QLabel("")
        self.cover_label.setObjectName("coverCard")
        self.cover_label.setFixedSize(COVER_SIZE, COVER_SIZE)
        self.cover_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cover_label.setPixmap(_build_cover_placeholder(COVER_SIZE, "未选择素材"))
        cover_box.addWidget(self.cover_label, 0, Qt.AlignmentFlag.AlignHCenter)

        self.video_canvas = VideoCanvasWidget(show_popout_btn=True, parent=self.media_stack)
        self.video_canvas.togglePlayRequested.connect(self.togglePlayRequested.emit)
        self.video_canvas.popOutRequested.connect(self.open_video_popout)

        self.media_stack.addWidget(cover_wrap)
        self.media_stack.addWidget(self.video_canvas)
        self.media_stack.setCurrentIndex(0)
        outer.addWidget(self.media_stack)

        self.title_label = QLabel("未选择素材")
        self.title_label.setWordWrap(True)
        self.title_label.setProperty("role", "inspectorTitle")
        outer.addWidget(self.title_label)

        self.pills_container = QWidget(self)
        self.pills_container.setStyleSheet("background: transparent;")
        pills_col = QVBoxLayout(self.pills_container)
        pills_col.setContentsMargins(0, 0, 0, 0)
        pills_col.setSpacing(4)

        row1 = QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)
        row1.setSpacing(4)
        row2 = QHBoxLayout()
        row2.setContentsMargins(0, 0, 0, 0)
        row2.setSpacing(4)

        self._pill_ext = self._make_pill("spec", mono=True)
        self._pill_sr = self._make_pill("spec", mono=True)
        self._pill_bit = self._make_pill("spec", mono=True)
        self._pill_ch = self._make_pill("spec", mono=True)
        self._pill_dur = self._make_pill("spec", mono=True)
        self._pill_size = self._make_pill("spec", mono=True)
        self._pill_status = self._make_pill("statusOnline", mono=False)
        self._mono_pills = [
            self._pill_ext,
            self._pill_sr,
            self._pill_bit,
            self._pill_ch,
            self._pill_dur,
            self._pill_size,
        ]

        for w in (self._pill_ext, self._pill_sr, self._pill_bit, self._pill_ch):
            row1.addWidget(w)
        row1.addStretch(1)
        for w in (self._pill_dur, self._pill_size, self._pill_status):
            row2.addWidget(w)
        row2.addStretch(1)
        pills_col.addLayout(row1)
        pills_col.addLayout(row2)
        self.pills_container.setVisible(False)
        outer.addWidget(self.pills_container)

        self.info_card = QFrame(self)
        self.info_card.setObjectName("inspectorInfoCard")
        card_layout = QVBoxLayout(self.info_card)
        card_layout.setContentsMargins(9, 7, 9, 7)
        card_layout.setSpacing(5)

        path_key_lbl = QLabel("所在路径")
        path_key_lbl.setProperty("role", "inspectorKey")
        card_layout.addWidget(path_key_lbl)

        self.path_breadcrumb_label = QLabel("")
        self.path_breadcrumb_label.setWordWrap(True)
        self.path_breadcrumb_label.setTextFormat(Qt.TextFormat.RichText)
        self.path_breadcrumb_label.setProperty("role", "inspectorBreadcrumb")
        self.path_breadcrumb_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        card_layout.addWidget(self.path_breadcrumb_label)

        self.path_value_label = QLabel("")
        self.path_value_label.setWordWrap(True)
        self.path_value_label.setProperty("role", "inspectorPath")
        self.path_value_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        card_layout.addWidget(self.path_value_label)

        path_actions = QHBoxLayout()
        path_actions.setContentsMargins(0, 2, 0, 1)
        path_actions.setSpacing(5)
        self.jump_folder_btn = QPushButton("跳转目录")
        self.jump_folder_btn.setProperty("variant", "cardAction")
        self.jump_folder_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.jump_folder_btn.setToolTip("在左侧库中跳转到该文件所在目录并定位")
        self.jump_folder_btn.clicked.connect(self._on_jump_folder_clicked)
        self.reveal_folder_btn = QPushButton("资源管理器")
        self.reveal_folder_btn.setProperty("variant", "cardAction")
        self.reveal_folder_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.reveal_folder_btn.setToolTip("在系统资源管理器中定位该文件")
        self.reveal_folder_btn.clicked.connect(self._on_reveal_folder_clicked)
        self.copy_path_btn = QPushButton("复制路径")
        self.copy_path_btn.setProperty("variant", "cardAction")
        self.copy_path_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.copy_path_btn.setToolTip("复制当前文件完整路径")
        self.copy_path_btn.clicked.connect(self._on_copy_path_clicked)
        path_actions.addWidget(self.jump_folder_btn, 4)
        path_actions.addWidget(self.reveal_folder_btn, 5)
        path_actions.addWidget(self.copy_path_btn, 4)
        card_layout.addLayout(path_actions)

        self._info_divider = QFrame(self.info_card)
        self._info_divider.setFixedHeight(1)
        self._info_divider.setStyleSheet("background-color: #2A2824; border: none;")
        card_layout.addWidget(self._info_divider)

        self._brand_row, brand_key, self.brand_value_label = self._make_info_row(
            "设计师",
            "inspectorBrand",
            spec.inspector_key_w,
        )
        self._package_row, pkg_key, self.package_value_label = self._make_info_row(
            "素材库",
            "inspectorPackage",
            spec.inspector_key_w,
        )
        self._ucs_row, ucs_key, self.ucs_value_label = self._make_info_row(
            "分类",
            "inspectorUcs",
            spec.inspector_key_w,
        )
        self._key_labels = [brand_key, pkg_key, ucs_key]
        card_layout.addWidget(self._brand_row)
        card_layout.addWidget(self._package_row)
        card_layout.addWidget(self._ucs_row)

        self.info_card.setVisible(False)
        outer.addWidget(self.info_card)

        self.meta_label = QLabel("")
        self.meta_label.setWordWrap(True)
        self.meta_label.setVisible(False)
        outer.addWidget(self.meta_label)

        user_row = QHBoxLayout()
        user_row.setContentsMargins(0, 0, 0, 0)
        user_row.setSpacing(6)
        self.fav_btn = QPushButton("☆ 收藏")
        self.fav_btn.setCheckable(True)
        self.fav_btn.setEnabled(False)
        self.fav_btn.setProperty("variant", "chip")
        self.fav_btn.setProperty("compact", True)
        self.fav_btn.clicked.connect(self._on_fav_clicked)
        self.edit_meta_btn = QPushButton("编辑元数据")
        self.edit_meta_btn.setEnabled(False)
        self.edit_meta_btn.setProperty("variant", "chip")
        self.edit_meta_btn.setProperty("compact", True)
        self.edit_meta_btn.setToolTip("编辑本地元数据、UCS 分类或从文件名解析（不会立即修改磁盘文件）")
        self.edit_meta_btn.clicked.connect(self._on_edit_meta_clicked)
        self.dirty_pill = QLabel("待写回")
        self.dirty_pill.setProperty("pill", "dirty")
        self.dirty_pill.setVisible(False)
        user_row.addWidget(self.fav_btn)
        user_row.addWidget(self.edit_meta_btn)
        user_row.addWidget(self.dirty_pill)
        user_row.addStretch(1)
        outer.addLayout(user_row)

        self.note_edit = QPlainTextEdit()
        self.note_edit.setObjectName("noteEdit")
        self.note_edit.setPlaceholderText("添加本地备注（仅存工具库，自动保存）")
        self.note_edit.setFixedHeight(spec.inspector_note_h)
        self.note_edit.setEnabled(False)
        self.note_edit.textChanged.connect(self._on_note_text_changed)
        outer.addWidget(self.note_edit)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.detail_label = QLabel("")
        self.detail_label.setWordWrap(True)
        self.detail_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.detail_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.detail_label.setProperty("role", "inspectorDetail")
        scroll.setWidget(self.detail_label)
        outer.addWidget(scroll, 1)

    @staticmethod
    def _make_info_row(
        key_text: str, value_role: str, key_width: int
    ) -> tuple[QWidget, QLabel, QLabel]:
        row = QWidget()
        row.setStyleSheet("background: transparent;")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        key_lbl = QLabel(key_text)
        key_lbl.setFixedWidth(key_width)
        key_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        key_lbl.setProperty("role", "inspectorKey")
        val_lbl = QLabel("")
        val_lbl.setWordWrap(True)
        val_lbl.setProperty("role", value_role)
        val_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(key_lbl, 0)
        layout.addWidget(val_lbl, 1)
        return row, key_lbl, val_lbl

    def apply_ui_scale(self, scale_key: str) -> None:
        spec = theme.ui_scale_spec(scale_key)
        self.setMinimumWidth(spec.inspector_min_w)
        self.setMaximumWidth(spec.inspector_max_w)
        self.note_edit.setFixedHeight(spec.inspector_note_h)
        for key_lbl in self._key_labels:
            key_lbl.setFixedWidth(spec.inspector_key_w)
        pill_font = mono_font(max(7, spec.mono_pt - 1))
        for pill in self._mono_pills:
            pill.setFont(pill_font)
        self.updateGeometry()

    def _on_jump_folder_clicked(self) -> None:
        if self._current_file_id is not None:
            self.navigateToFolderRequested.emit(self._current_file_id)

    def _on_copy_path_clicked(self) -> None:
        if self._current_full_path is not None:
            QGuiApplication.clipboard().setText(str(self._current_full_path))

    def _on_reveal_folder_clicked(self) -> None:
        if self._current_full_path is None:
            return
        from audiosearch.browser.file_table import reveal_in_explorer

        ok, error_msg = reveal_in_explorer(self._current_full_path)
        if not ok and error_msg:
            self.statusMessageRequested.emit(error_msg)

    @staticmethod
    def _format_path_breadcrumb(segments: tuple[str, ...], root_label: str) -> str:
        if not segments:
            fallback = html.escape(root_label or "根目录")
            return f'<span style="color:#E89274; font-weight:700;">{fallback}</span>'
        styled: list[str] = []
        for idx, seg in enumerate(segments):
            escaped = html.escape(seg)
            if idx == 0:
                styled.append(f'<span style="color:#E89274; font-weight:700;">{escaped}</span>')
            elif idx == 1:
                styled.append(f'<span style="color:#F2C2AE; font-weight:600;">{escaped}</span>')
            else:
                styled.append(f'<span style="color:#CFC9BE; font-weight:500;">{escaped}</span>')
        sep = ' <span style="color:#8C6252;">›</span> '
        return sep.join(styled)

    @staticmethod
    def _make_pill(pill_kind: str, *, mono: bool) -> QLabel:
        lbl = QLabel("")
        lbl.setProperty("pill", pill_kind)
        if mono:
            spec = theme.ui_scale_spec()
            lbl.setFont(mono_font(max(7, spec.mono_pt - 1)))
        return lbl

    @staticmethod
    def _repolish(widget: QWidget) -> None:
        st = widget.style()
        if st is not None:
            st.unpolish(widget)
            st.polish(widget)
        widget.update()

    def _on_fav_clicked(self, checked: bool) -> None:
        self._update_fav_btn_visual(checked)
        if self._current_file_id is not None:
            self.favoriteToggled.emit(self._current_file_id, bool(checked))

    def _on_edit_meta_clicked(self) -> None:
        if self._current_file_id is not None:
            self.editMetadataRequested.emit(self._current_file_id)

    def _update_fav_btn_visual(self, favorite: bool) -> None:
        self.fav_btn.setChecked(bool(favorite))
        self.fav_btn.setText("★ 已收藏" if favorite else "☆ 收藏")

    def _on_note_text_changed(self) -> None:
        if self._suppress_note or self._current_file_id is None:
            return
        self._note_dirty = True
        self._note_timer.start()

    def flush_pending_note(self) -> None:
        self._note_timer.stop()
        if not self._note_dirty or self._current_file_id is None:
            return
        self._note_dirty = False
        self.noteChanged.emit(self._current_file_id, self.note_edit.toPlainText())

    def set_favorite_state(self, file_id: int, favorite: bool) -> None:
        if self._current_file_id == file_id:
            self._update_fav_btn_visual(favorite)

    def show_file(
        self,
        rec: FileRecord | None,
        fields: list[tuple[str, str, str]] | None = None,
        *,
        ai_hit: Any | None = None,
    ) -> None:
        if rec is None or rec.id != self._current_file_id:
            self.flush_pending_note()
        self._cover_gen += 1
        if rec is None:
            self._current_is_video = False
            self._current_title = ""
            self._current_video_meta = ""
            self._last_video_time = 0.0
            self._video_worker.clear()
            self.video_canvas.clear("未选择素材")
            self.media_stack.setCurrentIndex(0)
            self.media_stack.setFixedHeight(COVER_SIZE)
            self._current_file_id = None
            self._current_full_path = None
            self._current_folder_path = ""
            self.fav_btn.setEnabled(False)
            self.edit_meta_btn.setEnabled(False)
            self.dirty_pill.setVisible(False)
            self._update_fav_btn_visual(False)
            self._suppress_note = True
            self.note_edit.setPlainText("")
            self.note_edit.setEnabled(False)
            self._note_dirty = False
            self._suppress_note = False
            self.cover_label.setText("")
            self.cover_label.setPixmap(_build_cover_placeholder(COVER_SIZE, "未选择素材"))
            self.title_label.setText("未选择素材")
            self.pills_container.setVisible(False)
            self.info_card.setVisible(False)
            self.brand_value_label.setText("")
            self.package_value_label.setText("")
            self.ucs_value_label.setText("")
            self.path_breadcrumb_label.setText("")
            self.path_value_label.setText("")
            self.meta_label.setText("")
            self.detail_label.setText("")
            return

        self._current_file_id = rec.id
        self._current_full_path = rec.full_path
        self._current_title = rec.filename
        is_video = is_video_path(rec.full_path)
        self._current_is_video = is_video
        self.fav_btn.setEnabled(True)
        self.edit_meta_btn.setEnabled(not is_video)
        self.edit_meta_btn.setToolTip(
            "视频文件暂不支持本地写回编辑"
            if is_video
            else "编辑本地元数据、UCS 分类或从文件名解析（不会立即修改磁盘文件）"
        )
        self.dirty_pill.setVisible(bool(getattr(rec, "is_dirty", False)))
        self._update_fav_btn_visual(rec.favorite)
        self._suppress_note = True
        if self.note_edit.toPlainText() != rec.note:
            self.note_edit.setPlainText(rec.note)
        self.note_edit.setEnabled(True)
        self._note_dirty = False
        self._suppress_note = False

        self.title_label.setText(rec.filename)
        status = "在线" if rec.is_available else "离线"
        self._pill_ext.setText((rec.ext or "—").upper())
        if is_video and rec.sample_rate <= 0:
            self._pill_sr.setText("无音轨")
            self._pill_bit.setVisible(False)
            self._pill_ch.setVisible(False)
        else:
            self._pill_sr.setText(format_sample_rate(rec.sample_rate))
            show_bit = (not is_video) or rec.bit_depth > 0
            self._pill_bit.setVisible(show_bit)
            self._pill_bit.setText(format_bit_depth(rec.bit_depth))
            self._pill_ch.setVisible(True)
            self._pill_ch.setText(format_channels(rec.channels))
        self._pill_dur.setText(format_duration(rec.duration_sec))
        self._pill_size.setText(format_size(rec.size_bytes))
        self._pill_status.setText(status)
        self._pill_status.setProperty("pill", "statusOnline" if rec.is_available else "statusOffline")
        self._repolish(self._pill_status)
        self.pills_container.setVisible(True)

        summary = extract_inspector_summary(rec, fields or [])
        self._current_folder_path = summary.folder_path
        self.path_breadcrumb_label.setText(
            self._format_path_breadcrumb(summary.path_segments, summary.root_label)
        )
        self.path_value_label.setText(summary.folder_path or str(rec.full_path))
        self.path_value_label.setToolTip(str(rec.full_path))
        self.brand_value_label.setText(summary.brand)
        self._brand_row.setVisible(bool(summary.brand))
        self.package_value_label.setText(summary.package)
        self._package_row.setVisible(bool(summary.package))
        self.ucs_value_label.setText(summary.ucs_label)
        self._ucs_row.setVisible(bool(summary.ucs_label))
        self._info_divider.setVisible(bool(summary.brand or summary.package or summary.ucs_label))
        self.info_card.setVisible(True)

        path_badge = " / ".join(summary.path_segments) if summary.path_segments else summary.root_label
        compat_summary = (
            f"{summary.brand or summary.package or path_badge or '—'} · {status}\n{rec.full_path}"
        )
        self.meta_label.setText(compat_summary)

        lines: list[str] = []
        if ai_hit is not None:
            ai_parts = [f"相关度：{ai_hit.relevance_pct}%（{ai_hit.hit_reason}）"]
            if ai_hit.text_score is not None:
                ai_parts.append(f"文本语义分：{ai_hit.text_score:.3f}")
            if ai_hit.audio_score is not None:
                ai_parts.append(f"声波相似分：{ai_hit.audio_score:.3f}")
            lines.append("AI 匹配：\n" + "\n".join(ai_parts))
        if rec.description:
            lines.append(f"描述：\n{rec.description}")
        if summary.formatted_fields:
            lines.append("内嵌元数据：")
            for label, val, src in summary.formatted_fields:
                if src == "user":
                    lines.append(f"• {label} [已编辑]: {val}")
                elif src == "ai":
                    lines.append(f"• {label} [AI]: {val}")
                else:
                    lines.append(f"• {label}: {val}")
        self.detail_label.setText("\n\n".join(lines) if lines else "无内嵌描述")

        if is_video and rec.is_available:
            self._last_video_time = 0.0
            self._current_video_meta = self._build_video_meta_from_fields(fields, rec.sample_rate <= 0)
            self.media_stack.setMinimumHeight(COVER_SIZE)
            self.media_stack.setMaximumHeight(240)
            self.media_stack.setCurrentIndex(1)
            self.video_canvas.clear("加载视频帧…")
            self.video_canvas.set_meta_text(self._current_video_meta)
            max_w, max_h = self._target_video_decode_size()
            self._video_worker.set_active(self._should_decode_video())
            self._video_gen = self._video_worker.load_video(
                rec.full_path, max_w=max_w, max_h=max_h
            )
            if self._popout_dialog is not None and self._popout_dialog.isVisible():
                self._popout_dialog.set_header_info(rec.filename, self._current_video_meta)
                self._popout_dialog.canvas.clear("加载视频帧…")
        else:
            self._video_worker.clear()
            self.video_canvas.clear("未选择视频")
            self.media_stack.setCurrentIndex(0)
            self.media_stack.setFixedHeight(COVER_SIZE)
            if self._popout_dialog is not None and self._popout_dialog.isVisible():
                self._popout_dialog.close()
            self.cover_label.setText("")
            self.cover_label.setPixmap(
                _build_cover_placeholder(COVER_SIZE, "加载封面…" if rec.is_available else "文件离线")
            )
            if rec.is_available:
                self._pool.start(
                    _CoverTask(self._resolver, rec.full_path, self._cover_gen, self._signals)
                )

    @staticmethod
    def _build_video_meta_from_fields(
        fields: list[tuple[str, str, str]] | None, silent_audio: bool
    ) -> str:
        fmap = {str(k).lower(): str(v) for k, v, _ in (fields or ())}
        parts: list[str] = []
        if fmap.get("video_resolution"):
            parts.append(fmap["video_resolution"])
        if fmap.get("video_fps"):
            parts.append(f"{fmap['video_fps']}fps")
        if silent_audio:
            parts.append("无音轨")
        return " · ".join(parts)

    def _should_decode_video(self) -> bool:
        pop_vis = self._popout_dialog is not None and self._popout_dialog.isVisible()
        return self.isVisible() or pop_vis

    def _target_video_decode_size(self) -> tuple[int, int]:
        if self._popout_dialog is not None and self._popout_dialog.isVisible():
            cw = max(640, self._popout_dialog.canvas.width())
            ch = max(360, self._popout_dialog.canvas.height())
            return min(1280, cw), min(720, ch)
        return max(380, self.width()), 320

    def sync_video_time(self, seconds: float) -> None:
        if not self._current_is_video or self._video_gen <= 0:
            return
        self._last_video_time = max(0.0, float(seconds))
        self._video_worker.request_time(self._video_gen, self._last_video_time)

    def set_video_playing(self, playing: bool) -> None:
        self.video_canvas.set_playing(playing)
        if self._popout_dialog is not None and self._popout_dialog.isVisible():
            self._popout_dialog.set_playing(playing)

    def open_video_popout(self) -> None:
        if not self._current_is_video:
            return
        if self._popout_dialog is None:
            self._popout_dialog = VideoPopoutDialog(self)
            self._popout_dialog.togglePlayRequested.connect(self.togglePlayRequested.emit)
            self._popout_dialog.closed.connect(self._on_popout_closed)
        self._popout_dialog.set_header_info(self._current_title, self._current_video_meta)
        self._popout_dialog.canvas.set_image(self.video_canvas.current_image())
        self._popout_dialog.set_playing(self.video_canvas._playing)
        self._popout_dialog.show()
        self._popout_dialog.raise_()
        self._popout_dialog.activateWindow()
        max_w, max_h = self._target_video_decode_size()
        self._video_worker.set_max_size(max_w, max_h)
        self._video_worker.set_active(True)
        self.sync_video_time(self._last_video_time)

    @Slot()
    def _on_popout_closed(self) -> None:
        max_w, max_h = self._target_video_decode_size()
        self._video_worker.set_max_size(max_w, max_h)
        self._video_worker.set_active(self._should_decode_video())

    @Slot(int, object)
    def _on_video_loaded(self, generation: int, info: VideoProbeInfo) -> None:
        if generation != self._video_gen:
            return
        parts: list[str] = []
        if info.width > 0 and info.height > 0:
            parts.append(f"{info.width}×{info.height}")
        if info.fps > 0.0:
            parts.append(f"{info.fps:.0f}fps" if abs(info.fps - round(info.fps)) < 0.05 else f"{info.fps:.2f}fps")
        if not info.has_audio:
            parts.append("无音轨")
        self._current_video_meta = " · ".join(parts)
        self.video_canvas.set_meta_text(self._current_video_meta)
        if self._popout_dialog is not None and self._popout_dialog.isVisible():
            self._popout_dialog.set_header_info(self._current_title, self._current_video_meta)

    @Slot(int, float, object)
    def _on_video_frame_ready(self, generation: int, _pts_sec: float, image: QImage) -> None:
        if generation != self._video_gen:
            return
        self.video_canvas.set_image(image)
        if self._popout_dialog is not None and self._popout_dialog.isVisible():
            self._popout_dialog.canvas.set_image(image)

    @Slot(int, str)
    def _on_video_failed(self, generation: int, message: str) -> None:
        if generation != self._video_gen:
            return
        self.video_canvas.set_image(None)
        self.video_canvas.set_status_text(f"视频解码失败：{message}")
        if self._popout_dialog is not None and self._popout_dialog.isVisible():
            self._popout_dialog.canvas.set_image(None)
            self._popout_dialog.canvas.set_status_text(f"视频解码失败：{message}")

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._video_worker.set_active(self._should_decode_video())
        if self._current_is_video and self._video_gen > 0:
            self._video_worker.request_time(self._video_gen, self._last_video_time)

    def hideEvent(self, event) -> None:  # noqa: N802
        super().hideEvent(event)
        self._video_worker.set_active(self._should_decode_video())

    def shutdown(self) -> None:
        if self._popout_dialog is not None:
            try:
                self._popout_dialog.close()
            except Exception:
                pass
        self._video_worker.shutdown()

    @Slot(int, object)
    def _on_cover_resolved(self, generation: int, cover: CoverImage | None) -> None:
        if generation != self._cover_gen:
            return
        if cover is None:
            self.cover_label.setText("")
            self.cover_label.setPixmap(_build_cover_placeholder(COVER_SIZE, "无内嵌封面"))
            return
        image = QImage.fromData(cover.data)
        if image.isNull():
            self.cover_label.setText("")
            self.cover_label.setPixmap(_build_cover_placeholder(COVER_SIZE, "无内嵌封面"))
            return
        pix = QPixmap.fromImage(image).scaled(
            COVER_SIZE,
            COVER_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.cover_label.setText("")
        self.cover_label.setPixmap(pix)
