"""DAW-style vertical mouse-drag rotary value control for Volume, Pitch, and Rate."""

from __future__ import annotations

import math
from collections.abc import Callable

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QWheelEvent,
)
from PySide6.QtWidgets import QInputDialog, QMenu, QSizePolicy, QWidget

from audiosearch.browser import theme


class DragValueControl(QWidget):
    """Compact rotary knob + numeric readout supporting vertical drag, fine mode, wheel & reset."""

    valueChanged = Signal(float)

    def __init__(
        self,
        label: str,
        *,
        minimum: float,
        maximum: float,
        default: float,
        value_format: Callable[[float], str],
        drag_range_px: float = 150.0,
        wheel_step: float = 1.0,
        decimals: int = 2,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._label = label
        self._minimum = float(minimum)
        self._maximum = float(maximum)
        self._default = max(self._minimum, min(float(default), self._maximum))
        self._value_format = value_format
        self._drag_range_px = max(20.0, float(drag_range_px))
        self._wheel_step = float(wheel_step)
        self._decimals = max(0, int(decimals))

        self._value = self._round_value(self._default)
        self._acc_value = self._value
        self._dragging = False
        self._hovered = False
        self._last_y = 0.0

        spec = theme.ui_scale_spec()
        self._scale = spec.scale
        self._hint_size = QSize(spec.knob_w, spec.knob_h)
        self._mono_pt = spec.mono_pt

        self.setFocusPolicy(Qt.NoFocus)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setMinimumSize(spec.knob_w - 6, spec.knob_h)
        self.setToolTip(f"{label}：上下拖动无极调节，按住 Shift/Ctrl 微调，双击重置，右键精确输入")

    def sizeHint(self) -> QSize:
        return QSize(self._hint_size)

    def minimumSizeHint(self) -> QSize:
        return QSize(max(84, self._hint_size.width() - 8), self._hint_size.height())

    def apply_ui_scale(self, scale_key: str) -> None:
        spec = theme.ui_scale_spec(scale_key)
        self._scale = spec.scale
        self._hint_size = QSize(spec.knob_w, spec.knob_h)
        self._mono_pt = spec.mono_pt
        self.setMinimumSize(spec.knob_w - 6, spec.knob_h)
        self.updateGeometry()
        self.update()

    def value(self) -> float:
        return self._value

    def setValue(self, val: float) -> None:
        self._set_value_internal(val, sync_acc=True)

    def reset(self) -> None:
        self.setValue(self._default)

    def _round_value(self, val: float) -> float:
        clamped = max(self._minimum, min(float(val), self._maximum))
        rounded = round(clamped, self._decimals)
        # Snap tiny -0.0 to 0.0 cleanly.
        return 0.0 if abs(rounded) < (10.0 ** (-self._decimals - 2)) else rounded

    def _set_value_internal(self, val: float, *, sync_acc: bool) -> None:
        clamped_raw = max(self._minimum, min(float(val), self._maximum))
        if sync_acc:
            self._acc_value = clamped_raw
        rounded = self._round_value(clamped_raw)
        if abs(rounded - self._value) <= 1e-9:
            self.update()
            return
        self._value = rounded
        self.update()
        self.valueChanged.emit(self._value)

    def _is_modified(self) -> bool:
        return abs(self._value - self._default) > 1e-4

    def enterEvent(self, event) -> None:
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._last_y = float(event.position().y())
            self._acc_value = self._value
            self.setCursor(Qt.SizeVerCursor)
            self.update()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging and (event.buttons() & Qt.LeftButton):
            cur_y = float(event.position().y())
            dy = self._last_y - cur_y
            self._last_y = cur_y
            scale = (self._maximum - self._minimum) / self._drag_range_px
            if event.modifiers() & (Qt.ShiftModifier | Qt.ControlModifier):
                scale *= 0.1
            self._acc_value = max(self._minimum, min(self._acc_value + dy * scale, self._maximum))
            self._set_value_internal(self._acc_value, sync_acc=False)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            self.unsetCursor()
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self._dragging = False
            self.unsetCursor()
            self.reset()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        steps = delta / 120.0
        step = self._wheel_step
        if event.modifiers() & (Qt.ShiftModifier | Qt.ControlModifier):
            step *= 0.1
        self.setValue(self._value + steps * step)
        event.accept()

    def contextMenuEvent(self, event) -> None:
        menu = QMenu(self)
        act_reset = menu.addAction(f"重置 {self._label} ({self._value_format(self._default)})")
        act_input = menu.addAction("精确输入...")
        chosen = menu.exec(event.globalPos())
        if chosen is act_reset:
            self.reset()
        elif chosen is act_input:
            val, ok = QInputDialog.getDouble(
                self,
                f"设置{self._label}",
                f"{self._label} ({self._minimum:g} ~ {self._maximum:g}):",
                self._value,
                self._minimum,
                self._maximum,
                self._decimals,
            )
            if ok:
                self.setValue(val)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        rect = QRectF(0.5, 0.5, self.width() - 1.0, self.height() - 1.0)
        modified = self._is_modified()
        if self._dragging:
            border = QColor("#D97757")
            bg = QColor("#382822")
        elif modified:
            border = QColor("#8A4C38")
            bg = QColor("#382822")
        elif self._hovered:
            border = QColor("#524E47")
            bg = QColor("#33312C")
        else:
            border = QColor("#3D3A34")
            bg = QColor("#2B2A26")

        painter.setPen(QPen(border, 1.0))
        painter.setBrush(bg)
        painter.drawRoundedRect(rect, 7.0, 7.0)

        # Rotary arc on left (270 deg sweep starting at 225 deg).
        knob_size = max(16.0, min(22.0, 18.0 * self._scale))
        kx = 5.0
        ky = (self.height() - knob_size) / 2.0
        arc_rect = QRectF(kx, ky, knob_size, knob_size)
        norm = (self._value - self._minimum) / max(self._maximum - self._minimum, 1e-9)
        norm = max(0.0, min(norm, 1.0))

        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor("#45423B"), 2.2, Qt.SolidLine, Qt.RoundCap))
        painter.drawArc(arc_rect, 225 * 16, -270 * 16)

        accent = QColor("#D97757") if (modified or self._dragging) else QColor("#A5A096")
        if self._minimum < self._default < self._maximum:
            default_norm = (self._default - self._minimum) / (self._maximum - self._minimum)
            start_deg = 225.0 - default_norm * 270.0
            span_deg = -(norm - default_norm) * 270.0
            if abs(span_deg) >= 1.0:
                painter.setPen(QPen(accent, 2.4, Qt.SolidLine, Qt.RoundCap))
                painter.drawArc(arc_rect, int(round(start_deg * 16)), int(round(span_deg * 16)))
        else:
            span_deg = -norm * 270.0
            if abs(span_deg) >= 1.0:
                painter.setPen(QPen(accent, 2.4, Qt.SolidLine, Qt.RoundCap))
                painter.drawArc(arc_rect, 225 * 16, int(round(span_deg * 16)))

        # Needle dot/line from knob center.
        angle_rad = math.radians(225.0 - norm * 270.0)
        cx, cy = arc_rect.center().x(), arc_rect.center().y()
        r_inner = 2.0
        r_outer = knob_size * 0.42
        painter.setPen(QPen(QColor("#ECE9E2"), 1.6, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(
            QPointF(cx + r_inner * math.cos(angle_rad), cy - r_inner * math.sin(angle_rad)),
            QPointF(cx + r_outer * math.cos(angle_rad), cy - r_outer * math.sin(angle_rad)),
        )

        # Text readout on right.
        h = float(self.height())
        top_h = max(10.0, h * 0.40)
        bot_y = max(10.0, h * 0.39)
        bot_h = max(12.0, h - bot_y - 1.0)
        text_x = int(kx + knob_size + 6)
        text_w = max(10, self.width() - text_x - 6)
        label_font = QFont(self.font())
        label_font.setPixelSize(max(9, min(12, int(round(10 * self._scale)))))
        painter.setFont(label_font)
        painter.setPen(QColor("#736E66"))
        painter.drawText(
            QRectF(text_x, 1.5, text_w, top_h),
            Qt.AlignLeft | Qt.AlignVCenter,
            self._label,
        )

        val_font = QFont("Consolas", self._mono_pt)
        if not val_font.exactMatch():
            val_font = QFont(self.font())
            val_font.setPointSize(self._mono_pt)
        painter.setFont(val_font)
        painter.setPen(QColor("#F0A387") if (modified or self._dragging) else QColor("#ECE9E2"))
        painter.drawText(
            QRectF(text_x, bot_y, text_w, bot_h),
            Qt.AlignLeft | Qt.AlignVCenter,
            self._value_format(self._value),
        )
        painter.end()


__all__ = ["DragValueControl"]
