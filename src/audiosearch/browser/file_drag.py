"""Native OS file drag-and-drop (CF_HDROP via QUrl.fromLocalFile)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from PySide6.QtCore import QMimeData, QPoint, Qt, QUrl
from PySide6.QtGui import QColor, QDrag, QFont, QFontMetrics, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

FILE_IDS_MIME = "application/x-audiosearch-file-ids"


def _build_drag_pixmap(label: str) -> QPixmap:
    font = QFont("Segoe UI", 9)
    metrics = QFontMetrics(font)
    text = label if len(label) <= 44 else f"{label[:41]}..."
    width = min(max(metrics.horizontalAdvance(text) + 24, 96), 340)
    height = 26
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setBrush(QColor(43, 42, 38, 240))
    painter.setPen(QPen(QColor("#D97757"), 1))
    painter.drawRoundedRect(0, 0, width - 1, height - 1, 6, 6)
    painter.setFont(font)
    painter.setPen(QColor("#ECE9E2"))
    painter.drawText(10, 0, width - 20, height, Qt.AlignmentFlag.AlignVCenter, text)
    painter.end()
    return pixmap


def start_files_drag(
    widget: QWidget,
    paths: Sequence[str | Path],
    *,
    file_ids: Sequence[int] | None = None,
) -> bool:
    """Start a native CF_HDROP drag for one or more local files (plus optional internal file IDs)."""
    targets: list[Path] = []
    for raw in paths:
        if not raw:
            continue
        p = Path(raw).resolve()
        if p.exists():
            targets.append(p)
    if not targets:
        return False
    drag = QDrag(widget)
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(p)) for p in targets])
    if file_ids:
        payload = json.dumps([int(fid) for fid in file_ids]).encode("utf-8")
        mime.setData(FILE_IDS_MIME, payload)
    drag.setMimeData(mime)
    badge = targets[0].name if len(targets) == 1 else f"{targets[0].name} 等 {len(targets)} 个文件"
    pixmap = _build_drag_pixmap(badge)
    drag.setPixmap(pixmap)
    drag.setHotSpot(QPoint(12, pixmap.height() // 2))
    drag.exec(
        Qt.DropAction.CopyAction | Qt.DropAction.MoveAction | Qt.DropAction.LinkAction,
        Qt.DropAction.CopyAction,
    )
    return True


def start_file_drag(
    widget: QWidget,
    path: str | Path | None,
    *,
    file_ids: Sequence[int] | None = None,
) -> bool:
    """Start a Windows CF_HDROP drag for an existing local audio file."""
    if not path:
        return False
    return start_files_drag(widget, [path], file_ids=file_ids)
