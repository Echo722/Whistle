"""Integrated dark title bar with bespoke vector Whistle wordmark and Win32 frameless support."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path
from typing import Any

from PySide6.QtCore import QPoint, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QCursor,
    QIcon,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import QFrame, QHBoxLayout, QPushButton, QWidget

from audiosearch.browser.runtime_paths import resolve_app_icon_path

TITLE_BAR_HEIGHT = 36

if sys.platform == "win32":
    import ctypes.wintypes as _wintypes

    class _NCCALCSIZE_PARAMS(ctypes.Structure):
        _fields_ = [
            ("rgrc", _wintypes.RECT * 3),
            ("lppos", ctypes.c_void_p),
        ]

    class _MARGINS(ctypes.Structure):
        _fields_ = [
            ("cxLeftWidth", ctypes.c_int),
            ("cxRightWidth", ctypes.c_int),
            ("cyTopHeight", ctypes.c_int),
            ("cyBottomHeight", ctypes.c_int),
        ]

    _USER32 = ctypes.WinDLL("user32")
    _DWMAPI = ctypes.WinDLL("dwmapi")

    _GET_LONG_PTR = getattr(_USER32, "GetWindowLongPtrW", _USER32.GetWindowLongW)
    _GET_LONG_PTR.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _GET_LONG_PTR.restype = ctypes.c_ssize_t

    _SET_LONG_PTR = getattr(_USER32, "SetWindowLongPtrW", _USER32.SetWindowLongW)
    _SET_LONG_PTR.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ssize_t]
    _SET_LONG_PTR.restype = ctypes.c_ssize_t

    _USER32.SetWindowPos.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    _USER32.SetWindowPos.restype = _wintypes.BOOL

    _USER32.IsZoomed.argtypes = [ctypes.c_void_p]
    _USER32.IsZoomed.restype = _wintypes.BOOL

    _USER32.GetWindowRect.argtypes = [ctypes.c_void_p, ctypes.POINTER(_wintypes.RECT)]
    _USER32.GetWindowRect.restype = _wintypes.BOOL

    _USER32.GetSystemMetrics.argtypes = [ctypes.c_int]
    _USER32.GetSystemMetrics.restype = ctypes.c_int

    _DWMAPI.DwmExtendFrameIntoClientArea.argtypes = [ctypes.c_void_p, ctypes.POINTER(_MARGINS)]
    _DWMAPI.DwmExtendFrameIntoClientArea.restype = ctypes.c_long

    _DWMAPI.DwmSetWindowAttribute.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    _DWMAPI.DwmSetWindowAttribute.restype = ctypes.c_long


def apply_win32_frameless_style(window: QWidget) -> None:
    """Enable native resize, Aero Snap, shadow, and Win11 rounded corners on a frameless window."""
    if sys.platform != "win32":
        return
    try:
        hwnd = int(window.winId())
        if not hwnd:
            return
        gwl_style = -16
        ws_caption = 0x00C00000
        ws_thickframe = 0x00040000
        ws_minimizebox = 0x00020000
        ws_maximizebox = 0x00010000
        swp_flags = 0x0020 | 0x0002 | 0x0001 | 0x0004  # FRAMECHANGED | NOMOVE | NOSIZE | NOZORDER

        style = int(_GET_LONG_PTR(hwnd, gwl_style))
        new_style = style | ws_thickframe | ws_caption | ws_minimizebox | ws_maximizebox
        _SET_LONG_PTR(hwnd, gwl_style, new_style)
        _USER32.SetWindowPos(hwnd, None, 0, 0, 0, 0, swp_flags)

        margins = _MARGINS(0, 0, 0, 1)
        _DWMAPI.DwmExtendFrameIntoClientArea(hwnd, ctypes.byref(margins))

        # Windows 11 rounded corners (DWMWA_WINDOW_CORNER_PREFERENCE = 33, DWMWCP_ROUND = 2).
        pref = ctypes.c_int(2)
        _DWMAPI.DwmSetWindowAttribute(hwnd, 33, ctypes.byref(pref), ctypes.sizeof(pref))
    except Exception:
        pass


def handle_win32_native_event(
    window: QWidget,
    title_bar: CustomTitleBar,
    event_type: Any,
    message: Any,
) -> tuple[bool, int] | None:
    """Handle WM_NCCALCSIZE and WM_NCHITTEST for native Win32 drag, resize, and snap."""
    if sys.platform != "win32":
        return None
    try:
        if bytes(event_type) != b"windows_generic_MSG":
            return None
        msg = _wintypes.MSG.from_address(int(message))
        hwnd = msg.hWnd

        if msg.message == 0x0083:  # WM_NCCALCSIZE
            if msg.wParam:
                if _USER32.IsZoomed(hwnd):
                    params = _NCCALCSIZE_PARAMS.from_address(msg.lParam)
                    pad = _USER32.GetSystemMetrics(92)  # SM_CXPADDEDBORDER
                    fx = _USER32.GetSystemMetrics(32) + pad  # SM_CXSIZEFRAME
                    fy = _USER32.GetSystemMetrics(33) + pad  # SM_CYSIZEFRAME
                    params.rgrc[0].left += fx
                    params.rgrc[0].top += fy
                    params.rgrc[0].right -= fx
                    params.rgrc[0].bottom -= fy
            return True, 0

        if msg.message == 0x0084:  # WM_NCHITTEST
            x = ctypes.c_short(msg.lParam & 0xFFFF).value
            y = ctypes.c_short((msg.lParam >> 16) & 0xFFFF).value
            rect = _wintypes.RECT()
            if not _USER32.GetWindowRect(hwnd, ctypes.byref(rect)):
                return None

            if not _USER32.IsZoomed(hwnd):
                border = max(4, int(round(5 * float(window.devicePixelRatioF()))))
                on_left = x < rect.left + border
                on_right = x >= rect.right - border
                on_top = y < rect.top + border
                on_bottom = y >= rect.bottom - border
                if on_top and on_left:
                    return True, 13  # HTTOPLEFT
                if on_top and on_right:
                    return True, 14  # HTTOPRIGHT
                if on_bottom and on_left:
                    return True, 16  # HTBOTTOMLEFT
                if on_bottom and on_right:
                    return True, 17  # HTBOTTOMRIGHT
                if on_left:
                    return True, 10  # HTLEFT
                if on_right:
                    return True, 11  # HTRIGHT
                if on_top:
                    return True, 12  # HTTOP
                if on_bottom:
                    return True, 15  # HTBOTTOM

            if title_bar.isVisible():
                local_pos = title_bar.mapFromGlobal(QCursor.pos())
                if title_bar.rect().contains(local_pos):
                    if title_bar.is_interactive_child(local_pos):
                        return True, 1  # HTCLIENT
                    return True, 2  # HTCAPTION
    except Exception:
        return None
    return None


def _resolve_brand_icon_pixmap(logical_size: int = 20) -> QPixmap | None:
    candidates: list[Path] = [
        Path(__file__).resolve().parent / "resources" / "whistle.png",
    ]
    app_icon = resolve_app_icon_path()
    if app_icon is not None:
        candidates.append(app_icon)
    for cand in candidates:
        try:
            if cand.is_file():
                pm = QPixmap(str(cand))
                if not pm.isNull():
                    scale = 2
                    scaled = pm.scaled(
                        logical_size * scale,
                        logical_size * scale,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                    scaled.setDevicePixelRatio(float(scale))
                    return scaled
        except OSError:
            continue
    return None


def _make_win_ctrl_icon(kind: str) -> QIcon:
    import math

    logical = 13 if kind == "settings" else 12
    scale = 2
    px = logical * scale
    colors = (
        (QIcon.Mode.Normal, QColor("#A5A096")),
        (QIcon.Mode.Active, QColor("#ECE9E2")),
        (QIcon.Mode.Disabled, QColor("#5C5851")),
    )
    icon = QIcon()
    for mode, color in colors:
        pm = QPixmap(px, px)
        pm.fill(Qt.GlobalColor.transparent)
        pm.setDevicePixelRatio(float(scale))
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, kind in ("close", "settings"))
        pen = QPen(color, 1.15)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        if kind == "min":
            p.drawLine(QPointF(1.5, 6.5), QPointF(10.5, 6.5))
        elif kind == "max":
            p.drawRect(QRectF(2.0, 2.0, 8.0, 8.0))
        elif kind == "restore":
            p.drawRect(QRectF(1.8, 3.8, 6.4, 6.4))
            p.drawPolyline(
                QPolygonF(
                    [
                        QPointF(3.8, 3.8),
                        QPointF(3.8, 1.8),
                        QPointF(10.2, 1.8),
                        QPointF(10.2, 8.2),
                        QPointF(8.2, 8.2),
                    ]
                )
            )
        elif kind == "close":
            p.drawLine(QPointF(2.2, 2.2), QPointF(9.8, 9.8))
            p.drawLine(QPointF(9.8, 2.2), QPointF(2.2, 9.8))
        elif kind == "settings":
            cx, cy = 6.5, 6.5
            r_in, r_out = 3.55, 5.15
            gear_pts: list[QPointF] = []
            for i in range(8):
                ang = i * (math.pi / 4.0)
                for d_ang, rad in (
                    (-0.22, r_in),
                    (-0.11, r_out),
                    (0.11, r_out),
                    (0.22, r_in),
                ):
                    a = ang + d_ang
                    gear_pts.append(QPointF(cx + rad * math.cos(a), cy + rad * math.sin(a)))
            gear_pen = QPen(color, 1.1)
            gear_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.setPen(gear_pen)
            p.drawPolygon(QPolygonF(gear_pts))
            p.drawEllipse(QPointF(cx, cy), 1.65, 1.65)
        p.end()
        icon.addPixmap(pm, mode)
    return icon


class WhistleBrandWidget(QWidget):
    """Paints the Whistle owl emblem and custom geometric vector wordmark."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(128, TITLE_BAR_HEIGHT)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._icon_pm = _resolve_brand_icon_pixmap(20)

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        x_off = 2.0
        if self._icon_pm is not None:
            y_icon = (self.height() - 20.0) / 2.0
            p.drawPixmap(QPointF(x_off, y_icon), self._icon_pm)
            x_off += 28.0

        y_base = (self.height() - 32.0) / 2.0
        p.translate(x_off, y_base)

        stroke_pen = QPen(
            QColor("#F2EFE9"),
            1.85,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
        p.setPen(stroke_pen)
        p.setBrush(Qt.BrushStyle.NoBrush)

        # W
        w_path = QPainterPath()
        w_path.moveTo(1.5, 9.5)
        w_path.lineTo(5.1, 22.5)
        w_path.lineTo(9.7, 13.4)
        w_path.lineTo(14.3, 22.5)
        w_path.lineTo(17.9, 9.5)
        p.drawPath(w_path)

        # h
        h_path = QPainterPath()
        h_path.moveTo(22.5, 9.2)
        h_path.lineTo(22.5, 22.5)
        h_path.moveTo(22.5, 16.6)
        h_path.cubicTo(23.9, 14.3, 26.5, 13.6, 28.5, 14.6)
        h_path.cubicTo(30.0, 15.3, 30.7, 16.7, 30.7, 18.6)
        h_path.lineTo(30.7, 22.5)
        p.drawPath(h_path)

        # i stem
        p.drawLine(QPointF(35.6, 14.2), QPointF(35.6, 22.5))

        # s
        s_path = QPainterPath()
        s_path.moveTo(46.8, 15.1)
        s_path.cubicTo(45.4, 13.8, 42.2, 13.7, 40.9, 15.3)
        s_path.cubicTo(39.8, 16.7, 41.3, 17.7, 43.8, 18.2)
        s_path.cubicTo(46.3, 18.7, 47.6, 19.9, 46.4, 21.5)
        s_path.cubicTo(45.1, 22.9, 41.8, 22.8, 40.2, 21.2)
        p.drawPath(s_path)

        # t
        t_path = QPainterPath()
        t_path.moveTo(52.1, 10.6)
        t_path.lineTo(52.1, 20.0)
        t_path.cubicTo(52.1, 21.7, 53.2, 22.5, 55.5, 22.1)
        t_path.moveTo(50.1, 14.2)
        t_path.lineTo(55.1, 14.2)
        p.drawPath(t_path)

        # l
        p.drawLine(QPointF(59.5, 9.2), QPointF(59.5, 22.5))

        # e
        e_path = QPainterPath()
        e_path.moveTo(64.3, 18.0)
        e_path.lineTo(72.1, 18.0)
        e_path.cubicTo(72.1, 15.3, 70.4, 13.8, 68.2, 13.8)
        e_path.cubicTo(65.8, 13.8, 64.2, 15.6, 64.2, 18.2)
        e_path.cubicTo(64.2, 20.9, 65.9, 22.5, 68.5, 22.5)
        e_path.cubicTo(69.9, 22.5, 71.0, 22.0, 71.7, 21.2)
        p.drawPath(e_path)

        # Terracotta acoustic dot on i
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#D97757"))
        p.drawEllipse(QPointF(35.6, 10.2), 1.45, 1.45)

        # Minimal twin-arc soundwave flourish
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor("#D97757"), 1.55, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        wave1 = QPainterPath()
        wave1.moveTo(77.2, 12.6)
        wave1.cubicTo(79.1, 14.3, 79.1, 17.7, 77.2, 19.4)
        p.drawPath(wave1)

        p.setPen(
            QPen(QColor(217, 119, 87, 148), 1.45, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        )
        wave2 = QPainterPath()
        wave2.moveTo(80.4, 10.4)
        wave2.cubicTo(83.1, 13.2, 83.1, 18.8, 80.4, 21.6)
        p.drawPath(wave2)
        p.end()


class CustomTitleBar(QFrame):
    """Unified dark top title bar housing the Whistle wordmark, settings trigger, and window controls."""

    settingsRequested = Signal()

    def __init__(self, target_window: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._target_window = target_window
        self.setObjectName("titleBar")
        self.setFixedHeight(TITLE_BAR_HEIGHT)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 0, 0)
        layout.setSpacing(0)

        self.brand_widget = WhistleBrandWidget(self)
        layout.addWidget(self.brand_widget)
        layout.addStretch(1)

        self._icon_settings = _make_win_ctrl_icon("settings")
        self._icon_min = _make_win_ctrl_icon("min")
        self._icon_max = _make_win_ctrl_icon("max")
        self._icon_restore = _make_win_ctrl_icon("restore")
        self._icon_close = _make_win_ctrl_icon("close")

        self.settings_btn = QPushButton("")
        self.settings_btn.setProperty("variant", "winCtrl")
        self.settings_btn.setFixedSize(40, TITLE_BAR_HEIGHT)
        self.settings_btn.setIcon(self._icon_settings)
        self.settings_btn.setIconSize(QSize(13, 13))
        self.settings_btn.setToolTip("设置 (Ctrl+,)")
        self.settings_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.settings_btn.clicked.connect(lambda _c=False: self.settingsRequested.emit())

        sep = QFrame(self)
        sep.setFixedSize(1, 14)
        sep.setStyleSheet("background-color: #2A2824; border: none;")

        self.min_btn = QPushButton("")
        self.min_btn.setProperty("variant", "winCtrl")
        self.min_btn.setFixedSize(46, TITLE_BAR_HEIGHT)
        self.min_btn.setIcon(self._icon_min)
        self.min_btn.setIconSize(QSize(12, 12))
        self.min_btn.setToolTip("最小化")
        self.min_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.min_btn.clicked.connect(self._target_window.showMinimized)

        self.max_btn = QPushButton("")
        self.max_btn.setProperty("variant", "winCtrl")
        self.max_btn.setFixedSize(46, TITLE_BAR_HEIGHT)
        self.max_btn.setIcon(self._icon_max)
        self.max_btn.setIconSize(QSize(12, 12))
        self.max_btn.setToolTip("最大化")
        self.max_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.max_btn.clicked.connect(self._toggle_maximize)

        self.close_btn = QPushButton("")
        self.close_btn.setProperty("variant", "winClose")
        self.close_btn.setFixedSize(46, TITLE_BAR_HEIGHT)
        self.close_btn.setIcon(self._icon_close)
        self.close_btn.setIconSize(QSize(12, 12))
        self.close_btn.setToolTip("关闭")
        self.close_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.close_btn.clicked.connect(self._target_window.close)

        layout.addWidget(self.settings_btn)
        layout.addWidget(sep)
        layout.addWidget(self.min_btn)
        layout.addWidget(self.max_btn)
        layout.addWidget(self.close_btn)

    def _toggle_maximize(self) -> None:
        if self._target_window.isMaximized():
            self._target_window.showNormal()
        else:
            self._target_window.showMaximized()

    def set_maximized(self, maximized: bool) -> None:
        self.max_btn.setIcon(self._icon_restore if maximized else self._icon_max)
        self.max_btn.setToolTip("还原" if maximized else "最大化")

    def is_interactive_child(self, pos_in_bar: QPoint) -> bool:
        for btn in (self.settings_btn, self.min_btn, self.max_btn, self.close_btn):
            if btn.isVisible() and btn.geometry().contains(pos_in_bar):
                return True
        return False

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position().toPoint()
            if not self.is_interactive_child(pos):
                handle = self._target_window.windowHandle()
                if handle is not None:
                    handle.startSystemMove()
                    event.accept()
                    return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position().toPoint()
            if not self.is_interactive_child(pos):
                self._toggle_maximize()
                event.accept()
                return
        super().mouseDoubleClickEvent(event)
