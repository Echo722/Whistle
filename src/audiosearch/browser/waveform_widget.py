"""Full-width waveform with selection, zoom/pan, detail peaks, and drag-out."""

from __future__ import annotations

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPolygonF,
    QWheelEvent,
)
from PySide6.QtWidgets import QApplication, QMenu, QSizePolicy, QWidget

from audiosearch.browser import theme
from audiosearch.browser.formatting import format_axis_time
from audiosearch.browser.theme import spectral_color_luts
from audiosearch.library.peaks import PeakData

BACKGROUND = QColor("#1D1C19")
LANE = QColor("#22211E")
PEAK = QColor("#878177")
PEAK_PLAYED = QColor("#D97757")
ZERO = QColor("#34322E")
TEXT = QColor("#ECE9E2")
MUTED = QColor("#736E66")
SELECTION_FILL = QColor(217, 119, 87, 52)
SELECTION_EDGE = QColor("#D97757")
SELECTION_HANDLE = QColor("#F0A387")
HIGHLIGHT_FILL = QColor(217, 119, 87, 36)
HIGHLIGHT_EDGE = QColor("#D97757")
PLAYHEAD = QColor("#A5A096")
PLAYBACK_HEAD = QColor("#E38566")
RULER_HEIGHT = 20
MIN_DRAG_PIXELS = 4
EDGE_HIT_PX = 6
MIN_VIEW_SPAN = 0.005


class WaveformWidget(QWidget):
    """Multi-channel waveform supporting ME selection, edge trim, wheel zoom/pan, and drag-out."""

    seekRequested = Signal(float)  # original-file seconds
    selectionChanged = Signal(object)  # tuple[float, float] | None
    selectionCommitted = Signal(float, float)  # emitted when draw or edge-trim completes
    selectionDragRequested = Signal(float, float, bool)  # (start, end, fade_5ms)
    zoomRangeRequested = Signal(float, float)  # (view_start, view_end)
    findSimilarRangeRequested = Signal(object)  # tuple[float, float] | None
    spectralPeaksToggled = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(96)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)

        self._overview_peaks: PeakData | None = None
        self._detail_peaks: PeakData | None = None
        self._duration: float = 0.0
        self._view: tuple[float, float] = (0.0, 1.0)
        self._selection: tuple[float, float] | None = None
        self._highlight_range: tuple[float, float] | None = None
        self._highlight_label: str = ""
        self._state_text = "尚未选择素材"
        self._cursor_sec: float | None = None
        self._playback_sec: float | None = None
        self._spectral_peaks: bool = False

        self._press_pos: QPoint | None = None
        self._drag_origin_sec: float | None = None
        self._mode: str | None = None
        self._resize_anchor_sec: float = 0.0
        self._pan_start_view: tuple[float, float] | None = None

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def view(self) -> tuple[float, float]:
        return self._view

    @property
    def selection(self) -> tuple[float, float] | None:
        return self._selection

    @property
    def highlight_range(self) -> tuple[float, float] | None:
        return self._highlight_range

    @property
    def spectral_peaks(self) -> bool:
        return self._spectral_peaks

    def set_spectral_peaks(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if self._spectral_peaks == enabled:
            return
        self._spectral_peaks = enabled
        self.update()

    def needs_spectral_data(self) -> bool:
        if not self._spectral_peaks:
            return False
        if self._overview_peaks is not None and self._overview_peaks.spectral is None:
            return True
        if self._detail_peaks is not None and self._detail_peaks.spectral is None:
            return True
        return False

    def clear(self, message: str = "尚未选择素材") -> None:
        self._overview_peaks = None
        self._detail_peaks = None
        self._duration = 0.0
        self._view = (0.0, 1.0)
        self._selection = None
        self._highlight_range = None
        self._highlight_label = ""
        self._cursor_sec = None
        self._playback_sec = None
        self._state_text = message
        self._reset_mouse_state()
        self.update()

    def set_loading(self, duration: float, message: str = "正在读取波形…") -> None:
        self._overview_peaks = None
        self._detail_peaks = None
        self._duration = max(0.0, duration)
        self._view = (0.0, self._duration) if self._duration > 0 else (0.0, 1.0)
        self._selection = None
        self._highlight_range = None
        self._highlight_label = ""
        self._cursor_sec = None
        self._playback_sec = None
        self._state_text = message
        self._reset_mouse_state()
        self.update()

    def set_peaks(self, peak_data: PeakData | None, *, message: str = "") -> None:
        if peak_data is None:
            self._overview_peaks = None
            self._detail_peaks = None
            if message:
                self._state_text = message
            self.update()
            return

        is_full = peak_data.start_seconds <= 1e-4 and (
            peak_data.duration_seconds <= 0
            or abs(peak_data.end_seconds - peak_data.duration_seconds) <= 0.05
        )
        if is_full:
            self._overview_peaks = peak_data
            self._detail_peaks = None
            self._duration = peak_data.duration_seconds
            if self._view == (0.0, 1.0) or self._view[1] <= 0 or self._view[1] > self._duration:
                self._view = (0.0, self._duration)
            rate_str = (
                f"{peak_data.sample_rate_hz // 1000}k"
                if peak_data.sample_rate_hz % 1000 == 0
                else f"{peak_data.sample_rate_hz / 1000:.1f}k"
            )
            self._state_text = f"{peak_data.points} 点 · {peak_data.channel_count}ch · {rate_str}"
        else:
            self._detail_peaks = peak_data
        self.update()

    def set_selection(self, selection: tuple[float, float] | None) -> None:
        if selection is not None and self._duration > 0:
            s0 = max(0.0, min(selection[0], self._duration))
            s1 = max(0.0, min(selection[1], self._duration))
            selection = (s0, s1) if s1 - s0 > 1e-4 else None
        if self._selection == selection:
            return
        self._selection = selection
        self.update()

    def set_highlight_range(
        self, highlight_range: tuple[float, float] | None, label: str = ""
    ) -> None:
        if highlight_range is not None and self._duration > 0:
            h0 = max(0.0, min(float(highlight_range[0]), self._duration))
            h1 = max(0.0, min(float(highlight_range[1]), self._duration))
            highlight_range = (h0, h1) if h1 - h0 > 1e-4 else None
        self._highlight_range = highlight_range
        self._highlight_label = label if highlight_range is not None else ""
        self.update()

    def reset_view(self) -> None:
        if self._duration > 0:
            self._view = (0.0, self._duration)
            self._detail_peaks = None
            self.update()

    def set_cursor(self, seconds: float | None) -> None:
        self._cursor_sec = seconds
        self.update()

    def set_playback_head(self, seconds: float | None) -> None:
        if self._playback_sec == seconds:
            return
        self._playback_sec = seconds
        self.update()

    def _reset_mouse_state(self) -> None:
        self._press_pos = None
        self._drag_origin_sec = None
        self._mode = None
        self._resize_anchor_sec = 0.0
        self._pan_start_view = None

    def _hit_selection(self, x: float, rect: QRectF) -> str:
        """Classify pointer ``x`` relative to current selection: left_edge|right_edge|inside|outside."""
        if self._selection is None or self._duration <= 0:
            return "outside"
        s0, s1 = self._selection
        x_left = self._time_to_x(s0, rect)
        x_right = self._time_to_x(s1, rect)
        d_left = abs(x - x_left)
        d_right = abs(x - x_right)
        if d_left <= EDGE_HIT_PX or d_right <= EDGE_HIT_PX:
            return "left_edge" if d_left <= d_right else "right_edge"
        if x_left < x < x_right:
            return "inside"
        return "outside"

    def _update_hover_cursor(self, x: float, y: float) -> None:
        if self._duration <= 0:
            self.setCursor(Qt.CursorShape.CrossCursor)
            return
        rect = self._plot_rect()
        if not rect.contains(QPointF(x, y)):
            self.setCursor(Qt.CursorShape.CrossCursor)
            return
        hit = self._hit_selection(x, rect)
        if hit in ("left_edge", "right_edge"):
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        elif hit == "inside":
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.setCursor(Qt.CursorShape.CrossCursor)

    def _set_view(self, start: float, end: float) -> None:
        if self._duration <= 0:
            return
        span = max(min(end - start, self._duration), min(MIN_VIEW_SPAN, self._duration))
        start = max(0.0, min(start, max(0.0, self._duration - span)))
        self._view = (start, start + span)
        if span < self._duration * 0.25:
            self.zoomRangeRequested.emit(self._view[0], self._view[1])
        else:
            self._detail_peaks = None
        self.update()

    def _active_peaks(self) -> PeakData | None:
        if self._detail_peaks is not None:
            v0, v1 = self._view
            if self._detail_peaks.start_seconds <= v0 + 1e-3 and self._detail_peaks.end_seconds >= v1 - 1e-3:
                return self._detail_peaks
        return self._overview_peaks

    def _plot_rect(self) -> QRectF:
        return QRectF(
            12,
            RULER_HEIGHT,
            max(self.width() - 24, 10),
            max(self.height() - RULER_HEIGHT - 6, 10),
        )

    def _time_to_x(self, seconds: float, rect: QRectF) -> float:
        v0, v1 = self._view
        span = max(v1 - v0, 1e-9)
        return rect.left() + ((seconds - v0) / span) * rect.width()

    def _x_to_time(self, x: float, rect: QRectF) -> float:
        if self._duration <= 0:
            return 0.0
        v0, v1 = self._view
        span = max(v1 - v0, 1e-9)
        ratio = (x - rect.left()) / max(rect.width(), 1e-9)
        return max(0.0, min(v0 + ratio * span, self._duration))

    # -- painting ----------------------------------------------------------
    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.fillRect(self.rect(), BACKGROUND)
        rect = self._plot_rect()
        if rect.width() <= 0 or rect.height() <= 0:
            return
        painter.fillRect(rect, LANE)
        self._paint_ruler(painter, rect)

        peaks = self._active_peaks()
        if peaks is not None:
            self._paint_peaks(painter, rect, peaks)
        else:
            painter.setPen(QPen(TEXT if self._duration > 0 else MUTED))
            painter.drawText(
                rect.adjusted(10, 10, -10, -10),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                self._state_text,
            )

        self._paint_highlight_range(painter, rect)
        self._paint_selection(painter, rect)

        if self._cursor_sec is not None and self._duration > 0:
            cx = self._time_to_x(self._cursor_sec, rect)
            if rect.left() <= cx <= rect.right():
                painter.setPen(QPen(PLAYHEAD, 1))
                painter.drawLine(QPointF(cx, rect.top()), QPointF(cx, rect.bottom()))

        if self._playback_sec is not None and self._duration > 0:
            px = self._time_to_x(self._playback_sec, rect)
            if rect.left() <= px <= rect.right():
                painter.setPen(QPen(PLAYBACK_HEAD, 2))
                painter.drawLine(QPointF(px, rect.top()), QPointF(px, rect.bottom()))
                painter.setBrush(PLAYBACK_HEAD)
                painter.drawPolygon(
                    QPolygonF(
                        [
                            QPointF(px - 4, rect.top()),
                            QPointF(px + 4, rect.top()),
                            QPointF(px, rect.top() + 6),
                        ]
                    )
                )
                painter.setBrush(Qt.BrushStyle.NoBrush)

        if peaks is not None:
            painter.setPen(QPen(MUTED))
            font = theme.derived_font(painter.font(), -2, min_px=9)
            painter.setFont(font)
            hint_h = max(14.0, float(painter.fontMetrics().height() + 2))
            hint = self._state_text
            if self._selection is not None:
                s0, s1 = self._selection
                hint += (
                    f"   选区: {s0:.3f}s - {s1:.3f}s ({s1 - s0:.3f}s)"
                    " · 拖边缘微调 · 拖中部导出(Alt加5ms淡变)"
                )
            show_legend = self._spectral_peaks and rect.width() >= 480
            hint_w = rect.width() - (154.0 if show_legend else 12.0)
            painter.drawText(
                QRectF(rect.left() + 6, rect.bottom() - hint_h - 2.0, max(40.0, hint_w), hint_h),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                hint,
            )
            if show_legend:
                self._paint_spectral_legend(painter, rect)

    def _paint_spectral_legend(self, painter: QPainter, rect: QRectF) -> None:
        normal_lut, _ = spectral_color_luts()
        right = rect.right() - 6.0
        y_text = rect.bottom() - 16.0
        painter.setPen(QPen(MUTED))
        painter.drawText(
            QRectF(right - 56.0, y_text, 56.0, 14.0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "高 · 灰=噪",
        )
        bar_left = right - 116.0
        bar_y = rect.bottom() - 11.5
        seg_count = 14
        seg_w = 56.0 / seg_count
        for idx in range(seg_count):
            h_idx = min(63, int(round(idx * 63 / (seg_count - 1))))
            painter.fillRect(
                QRectF(bar_left + idx * seg_w, bar_y, seg_w + 0.5, 5.0),
                normal_lut[h_idx][7],
            )
        painter.setPen(QPen(MUTED))
        painter.drawText(
            QRectF(bar_left - 18.0, y_text, 16.0, 14.0),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            "低",
        )

    def _paint_ruler(self, painter: QPainter, rect: QRectF) -> None:
        painter.setPen(QPen(MUTED))
        font = theme.derived_font(painter.font(), -2, min_px=9)
        painter.setFont(font)
        v0, v1 = self._view if self._duration > 0 else (0.0, 1.0)
        span = max(v1 - v0, 1e-9)
        ticks = max(2, int(rect.width() / 120))
        for idx in range(ticks + 1):
            pos = v0 + span * idx / ticks
            x = rect.left() + rect.width() * idx / ticks
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.top() + 4))
            label = format_axis_time(pos, span)
            align_x = x - 38 if idx == ticks else x + 3
            painter.drawText(QPointF(align_x, rect.top() - 4), label)
        painter.drawLine(QPointF(rect.left(), rect.top()), QPointF(rect.right(), rect.top()))

    def _paint_peaks(self, painter: QPainter, rect: QRectF, data: PeakData) -> None:
        channels = data.channel_count
        lane_height = rect.height() / max(channels, 1)
        buckets = data.points
        v0, v1 = self._view
        d0 = data.start_seconds
        d1 = data.end_seconds if data.end_seconds > d0 else self._duration
        d_span = max(d1 - d0, 1e-9)
        played_x = (
            self._time_to_x(self._playback_sec, rect)
            if self._playback_sec is not None and self._duration > 0
            else -1.0
        )

        # Map visible view slice directly to bucket range for fast zoomed rendering.
        i_start = max(0, int(((v0 - d0) / d_span) * buckets) - 1)
        i_end = min(buckets, int(((v1 - d0) / d_span) * buckets) + 2)
        if i_end <= i_start:
            return

        use_spectral = (
            self._spectral_peaks
            and data.spectral is not None
            and data.spectral.shape == (channels, buckets, 2)
        )
        normal_lut, played_lut = spectral_color_luts() if use_spectral else (None, None)

        for ch in range(channels):
            top = rect.top() + lane_height * ch
            mid = top + lane_height / 2.0
            amp = max(lane_height / 2.0 - 6.0, 4.0)
            painter.setPen(QPen(ZERO, 1))
            painter.drawLine(QPointF(rect.left(), mid), QPointF(rect.right(), mid))

            ch_peaks = data.peaks[ch]
            if use_spectral and data.spectral is not None and normal_lut is not None and played_lut is not None:
                ch_spec = data.spectral[ch]
                last_spec_key: tuple[int, int, bool] | None = None
                cur_color = PEAK
                for i in range(i_start, i_end):
                    b_start_sec = d0 + (d_span * i) / buckets
                    b_end_sec = d0 + (d_span * (i + 1)) / buckets
                    x0 = self._time_to_x(b_start_sec, rect)
                    x1 = self._time_to_x(b_end_sec, rect)
                    if x1 < rect.left() or x0 > rect.right():
                        continue
                    x0 = max(rect.left(), x0)
                    x1 = min(rect.right(), x1)
                    is_played = played_x > rect.left() and x0 <= played_x
                    h_idx = int(ch_spec[i, 0]) >> 2
                    t_idx = int(ch_spec[i, 1]) >> 5
                    spec_key = (h_idx, t_idx, is_played)
                    if spec_key != last_spec_key:
                        last_spec_key = spec_key
                        cur_color = (played_lut if is_played else normal_lut)[h_idx][t_idx]
                        painter.setPen(QPen(cur_color, 1))
                    pmin = float(ch_peaks[i, 0])
                    pmax = float(ch_peaks[i, 1])
                    y_max = mid - pmax * amp
                    y_min = mid - pmin * amp
                    w = x1 - x0
                    if abs(y_min - y_max) < 1.0:
                        painter.drawLine(QPointF(x0, mid), QPointF(max(x0 + 1.0, x1), mid))
                    elif w >= 1.5:
                        painter.fillRect(QRectF(x0, y_max, w, max(1.0, y_min - y_max)), cur_color)
                    else:
                        painter.drawLine(QPointF(x0, y_max), QPointF(x0, y_min))
            else:
                last_played = False
                painter.setPen(QPen(PEAK, 1))
                for i in range(i_start, i_end):
                    b_start_sec = d0 + (d_span * i) / buckets
                    b_end_sec = d0 + (d_span * (i + 1)) / buckets
                    x0 = self._time_to_x(b_start_sec, rect)
                    x1 = self._time_to_x(b_end_sec, rect)
                    if x1 < rect.left() or x0 > rect.right():
                        continue
                    x0 = max(rect.left(), x0)
                    x1 = min(rect.right(), x1)
                    is_played = played_x > rect.left() and x0 <= played_x
                    if is_played != last_played:
                        last_played = is_played
                        painter.setPen(QPen(PEAK_PLAYED if is_played else PEAK, 1))
                    pmin = float(ch_peaks[i, 0])
                    pmax = float(ch_peaks[i, 1])
                    y_max = mid - pmax * amp
                    y_min = mid - pmin * amp
                    w = x1 - x0
                    if abs(y_min - y_max) < 1.0:
                        painter.drawLine(QPointF(x0, mid), QPointF(max(x0 + 1.0, x1), mid))
                    elif w >= 2.0:
                        painter.drawLine(QPointF(x0, y_max), QPointF(x0, y_min))
                        painter.drawLine(QPointF(x0, y_max), QPointF(x1, y_max))
                        painter.drawLine(QPointF(x0, y_min), QPointF(x1, y_min))
                    else:
                        painter.drawLine(QPointF(x0, y_max), QPointF(x0, y_min))

    def _paint_highlight_range(self, painter: QPainter, rect: QRectF) -> None:
        if self._highlight_range is None or self._duration <= 0:
            return
        h0, h1 = self._highlight_range
        left = max(rect.left(), self._time_to_x(h0, rect))
        right = min(rect.right(), self._time_to_x(h1, rect))
        if right <= left:
            return
        band = QRectF(left, rect.top(), max(right - left, 2.0), rect.height())
        painter.fillRect(band, HIGHLIGHT_FILL)
        pen = QPen(HIGHLIGHT_EDGE, 1, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(QPointF(band.left(), band.top()), QPointF(band.left(), band.bottom()))
        painter.drawLine(QPointF(band.right(), band.top()), QPointF(band.right(), band.bottom()))
        if self._highlight_label:
            font = theme.derived_font(painter.font(), -2, min_px=9)
            painter.setFont(font)
            label_rect = QRectF(band.left() + 4, rect.top() + 3, max(band.width() - 8, 90.0), 15)
            painter.setPen(QPen(HIGHLIGHT_EDGE))
            painter.drawText(
                label_rect,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                self._highlight_label,
            )

    def _paint_selection(self, painter: QPainter, rect: QRectF) -> None:
        if self._selection is None or self._duration <= 0:
            return
        s0, s1 = self._selection
        left = max(rect.left(), self._time_to_x(s0, rect))
        right = min(rect.right(), self._time_to_x(s1, rect))
        if right <= left:
            return
        band = QRectF(left, rect.top(), max(right - left, 1.0), rect.height())
        painter.fillRect(band, SELECTION_FILL)
        painter.setPen(QPen(SELECTION_EDGE, 1))
        painter.drawLine(QPointF(band.left(), band.top()), QPointF(band.left(), band.bottom()))
        painter.drawLine(QPointF(band.right(), band.top()), QPointF(band.right(), band.bottom()))
        # Subtle top/bottom trim handles for left/right edges.
        painter.setPen(QPen(SELECTION_HANDLE, 2))
        h_cap = min(8.0, band.height() * 0.25)
        for edge_x in (band.left(), band.right()):
            painter.drawLine(QPointF(edge_x, band.top()), QPointF(edge_x, band.top() + h_cap))
            painter.drawLine(QPointF(edge_x, band.bottom() - h_cap), QPointF(edge_x, band.bottom()))

    # -- mouse & wheel interaction -----------------------------------------
    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._duration <= 0:
            super().mousePressEvent(event)
            return
        rect = self._plot_rect()
        if event.button() == Qt.MouseButton.MiddleButton and rect.contains(event.position()):
            self._press_pos = event.position().toPoint()
            self._pan_start_view = self._view
            self._mode = "pan"
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and rect.contains(event.position()):
            self.setFocus(Qt.FocusReason.MouseFocusReason)
            pos = event.position().toPoint()
            x = float(event.position().x())
            sec = self._x_to_time(x, rect)
            self._press_pos = pos
            self._drag_origin_sec = sec
            hit = self._hit_selection(x, rect)
            if hit == "left_edge" and self._selection is not None:
                self._mode = "resize_left"
                self._resize_anchor_sec = self._selection[1]
            elif hit == "right_edge" and self._selection is not None:
                self._mode = "resize_right"
                self._resize_anchor_sec = self._selection[0]
            elif hit == "inside" and self._selection is not None:
                self._mode = "inside_selection"
            else:
                self._mode = "draw_or_seek"
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._mode == "pan" and (event.buttons() & Qt.MouseButton.MiddleButton):
            if self._press_pos is not None and self._pan_start_view is not None:
                rect = self._plot_rect()
                v0, v1 = self._pan_start_view
                span = max(v1 - v0, 1e-9)
                dx = float(event.position().x() - self._press_pos.x())
                dt = -(dx / max(rect.width(), 1.0)) * span
                self._set_view(v0 + dt, v1 + dt)
            event.accept()
            return

        if self._press_pos is None or self._duration <= 0 or not (event.buttons() & Qt.MouseButton.LeftButton):
            self._update_hover_cursor(float(event.position().x()), float(event.position().y()))
            super().mouseMoveEvent(event)
            return

        rect = self._plot_rect()
        x = float(event.position().x())
        cur_sec = self._x_to_time(x, rect)

        if self._mode in ("resize_left", "resize_right"):
            s0, s1 = sorted((self._resize_anchor_sec, cur_sec))
            if s1 - s0 > 1e-4:
                self._selection = (s0, s1)
                self.selectionChanged.emit(self._selection)
                self.update()
            event.accept()
            return

        if self._mode == "inside_selection":
            dist = (event.position().toPoint() - self._press_pos).manhattanLength()
            if dist >= QApplication.startDragDistance() and self._selection is not None:
                s0, s1 = self._selection
                fade = bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
                self._reset_mouse_state()
                self._update_hover_cursor(x, float(event.position().y()))
                self.selectionDragRequested.emit(s0, s1, fade)
            event.accept()
            return

        if self._mode == "draw_or_seek":
            if abs(x - self._press_pos.x()) >= MIN_DRAG_PIXELS:
                self._mode = "drawing"

        if self._mode == "drawing" and self._drag_origin_sec is not None:
            s0, s1 = sorted((self._drag_origin_sec, cur_sec))
            if s1 - s0 > 1e-4:
                self._selection = (s0, s1)
                self.selectionChanged.emit(self._selection)
                self.update()
            event.accept()
            return

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.MiddleButton and self._mode == "pan":
            self._reset_mouse_state()
            self._update_hover_cursor(float(event.position().x()), float(event.position().y()))
            event.accept()
            return

        if event.button() == Qt.MouseButton.LeftButton and self._press_pos is not None:
            rect = self._plot_rect()
            x = float(event.position().x())
            sec = self._x_to_time(x, rect)
            mode = self._mode
            origin_sec = self._drag_origin_sec if self._drag_origin_sec is not None else sec
            self._reset_mouse_state()
            self._update_hover_cursor(x, float(event.position().y()))

            if mode in ("drawing", "resize_left", "resize_right"):
                if self._selection is not None:
                    s0, s1 = self._selection
                    self.selectionChanged.emit((s0, s1))
                    self.selectionCommitted.emit(s0, s1)
            elif mode == "inside_selection":
                self._cursor_sec = origin_sec
                self.update()
                self.seekRequested.emit(origin_sec)
            elif mode == "draw_or_seek":
                if self._selection is not None:
                    self._selection = None
                    self.selectionChanged.emit(None)
                self._cursor_sec = origin_sec
                self.update()
                self.seekRequested.emit(origin_sec)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        has_audio = self._duration > 0
        menu = QMenu(self)
        act_similar_sel = menu.addAction("以此选区找同类 (AI)")
        act_similar_sel.setEnabled(has_audio and self._selection is not None)
        act_similar_all = menu.addAction("以当前音频找同类 (AI)")
        act_similar_all.setEnabled(has_audio)
        menu.addSeparator()
        act_clear = menu.addAction("清除选区 (Esc)")
        act_clear.setEnabled(has_audio and self._selection is not None)
        act_zoom = menu.addAction("缩放到选区")
        act_zoom.setEnabled(has_audio and self._selection is not None)
        act_reset = menu.addAction("重置视图（显示全长）")
        act_reset.setEnabled(has_audio)
        menu.addSeparator()
        act_spectral = menu.addAction("频谱波形着色 (Spectral Peaks)")
        act_spectral.setCheckable(True)
        act_spectral.setChecked(self._spectral_peaks)
        chosen = menu.exec(event.globalPos())
        if chosen is act_similar_sel and self._selection is not None:
            self.findSimilarRangeRequested.emit(self._selection)
        elif chosen is act_similar_all and has_audio:
            self.findSimilarRangeRequested.emit(None)
        elif chosen is act_clear and self._selection is not None:
            self._selection = None
            self.selectionChanged.emit(None)
            self.update()
        elif chosen is act_zoom and self._selection is not None:
            self._set_view(self._selection[0], self._selection[1])
        elif chosen is act_reset and has_audio:
            self.reset_view()
        elif chosen is act_spectral:
            self.spectralPeaksToggled.emit(not self._spectral_peaks)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._duration > 0:
            self._reset_mouse_state()
            self.reset_view()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        if self._duration <= 0:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        v0, v1 = self._view
        span = max(v1 - v0, 1e-9)
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            step = span * 0.15 * (-1.0 if delta > 0 else 1.0)
            self._set_view(v0 + step, v1 + step)
            event.accept()
            return

        anchor = self._x_to_time(event.position().x(), self._plot_rect())
        factor = 0.78 if delta > 0 else 1.28
        new_span = max(min(span * factor, self._duration), min(MIN_VIEW_SPAN, self._duration))
        ratio = (anchor - v0) / span
        self._set_view(anchor - ratio * new_span, anchor + (1.0 - ratio) * new_span)
        event.accept()
