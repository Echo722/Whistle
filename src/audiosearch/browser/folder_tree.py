"""REAPER ME-inspired unified sidebar tree: quick views, collection dividers, and local folders."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QModelIndex, QPoint, QPointF, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QAction,
    QColor,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QFont,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QMenu,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
)

from audiosearch.browser import theme
from audiosearch.browser.collections_tree import (
    KIND_COLLECTION,
    KIND_COLLECTION_FOLDER,
    KIND_DUPLICATES,
    KIND_FAVORITES,
    KIND_RECENT,
    SidebarViewTarget,
)
from audiosearch.browser.file_drag import FILE_IDS_MIME
from audiosearch.library.store import FolderRecord, LibraryStore, RootRecord

_KIND_ROLE = Qt.ItemDataRole.UserRole
_ROOT_ID_ROLE = Qt.ItemDataRole.UserRole + 1
_FOLDER_ID_ROLE = Qt.ItemDataRole.UserRole + 2
_REL_PATH_ROLE = Qt.ItemDataRole.UserRole + 3
_LOADED_ROLE = Qt.ItemDataRole.UserRole + 4
_FULL_PATH_ROLE = Qt.ItemDataRole.UserRole + 5
_COUNT_ROLE = Qt.ItemDataRole.UserRole + 6
_OFFLINE_ROLE = Qt.ItemDataRole.UserRole + 7
_COLL_ID_ROLE = Qt.ItemDataRole.UserRole + 8
_CF_ID_ROLE = Qt.ItemDataRole.UserRole + 9
_ACTIVE_PROXY_ROLE = Qt.ItemDataRole.UserRole + 10

_KIND_ALL = "all"
_KIND_ROOT = "root"
_KIND_FOLDER = "folder"
_KIND_DUMMY = "dummy"
KIND_SECTION_DIVIDER = "section_divider"
KIND_ADD_COLLECTION = "add_collection"

_DIVIDER_KINDS = frozenset({KIND_COLLECTION, KIND_SECTION_DIVIDER, KIND_ADD_COLLECTION})
_QUICK_ICON_KINDS = frozenset({_KIND_ALL, KIND_FAVORITES, KIND_RECENT, KIND_DUPLICATES})

_LINE_DIVIDER = QColor("#34322E")
_DIVIDER_TEXT = QColor("#8F8A80")
_DIVIDER_ACTIVE = QColor("#ECE9E2")
_ADD_TEXT = QColor("#D97757")
_ICON_MUTED = QColor("#8E897F")
_COUNT_COLOR = QColor("#736E66")
_HOVER_BG = QColor("#272522")


@dataclass(frozen=True, slots=True)
class TreeLocation:
    root_id: int | None
    folder_id: int | None
    folder_rel_path: str
    display_path: str


class _SidebarItemDelegate(QStyledItemDelegate):
    """Paints REAPER ME-style dense rows, centered divider headers, mono counts, and crisp icons."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        spec = theme.ui_scale_spec()
        self._row_height = spec.tree_row_h
        self._divider_height = spec.tree_divider_h
        self._scale = spec.scale

    def apply_ui_scale(self, spec: theme.UiScaleSpec) -> None:
        self._row_height = spec.tree_row_h
        self._divider_height = spec.tree_divider_h
        self._scale = spec.scale

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:  # noqa: N802
        sz = super().sizeHint(option, index)
        kind = index.data(_KIND_ROLE)
        h = self._divider_height if kind in _DIVIDER_KINDS else self._row_height
        return QSize(sz.width(), h)

    def paint(
        self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex
    ) -> None:
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)

        kind = str(index.data(_KIND_ROLE) or "")
        text = str(index.data(Qt.ItemDataRole.DisplayRole) or "")
        count_text = str(index.data(_COUNT_ROLE) or "")
        is_offline = bool(index.data(_OFFLINE_ROLE))
        is_selected = bool(opt.state & QStyle.StateFlag.State_Selected)
        is_proxy_active = bool(index.data(_ACTIVE_PROXY_ROLE))
        effective_selected = is_selected or (kind == KIND_SECTION_DIVIDER and is_proxy_active)
        is_hovered = bool(opt.state & QStyle.StateFlag.State_MouseOver)

        rect = opt.rect
        painter.save()
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            if effective_selected:
                pill = QRectF(rect.adjusted(2, 1, -4, -1))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(theme.SELECT)
                painter.drawRoundedRect(pill, 6.0, 6.0)
                tick = QRectF(
                    rect.left() + 2.0,
                    rect.top() + 5.0,
                    2.5,
                    max(4.0, rect.height() - 10.0),
                )
                painter.setBrush(theme.SELECT_MARK)
                painter.drawRoundedRect(tick, 1.2, 1.2)
            elif is_hovered:
                pill = QRectF(rect.adjusted(2, 1, -4, -1))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(_HOVER_BG)
                painter.drawRoundedRect(pill, 6.0, 6.0)

            if kind in _DIVIDER_KINDS:
                self._paint_divider_row(
                    painter,
                    rect,
                    opt.font,
                    kind=kind,
                    text=text,
                    count_text=count_text,
                    is_selected=effective_selected,
                    is_hovered=is_hovered,
                )
            else:
                self._paint_standard_row(
                    painter,
                    rect,
                    opt.font,
                    kind=kind,
                    text=text,
                    count_text=count_text,
                    is_offline=is_offline,
                    is_selected=is_selected,
                )
        finally:
            painter.restore()

    def _paint_divider_row(
        self,
        painter: QPainter,
        rect: QRect,
        base_font: QFont,
        *,
        kind: str,
        text: str,
        count_text: str,
        is_selected: bool,
        is_hovered: bool,
    ) -> None:
        label_font = theme.derived_font(base_font, -1, min_px=10)
        if kind == KIND_COLLECTION:
            label_font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(label_font)
        fm = painter.fontMetrics()

        count_w = 0
        count_font = theme.derived_font(base_font, -2, min_px=9)
        if count_text:
            painter.setFont(count_font)
            count_w = painter.fontMetrics().horizontalAdvance(count_text) + 10
            painter.setFont(label_font)

        left_pad = rect.left() + 6
        right_limit = rect.right() - 6 - count_w
        avail = max(24, right_limit - left_pad)
        elided = fm.elidedText(text, Qt.TextElideMode.ElideRight, avail)
        tw = fm.horizontalAdvance(elided)

        mid_x = rect.left() + max(0, (rect.width() - count_w) // 2)
        text_x = max(left_pad, mid_x - tw // 2)
        text_rect = QRect(text_x, rect.top(), min(tw + 2, avail), rect.height())
        mid_y = rect.center().y() + 1

        line_pen = QPen(theme.SELECT_MARK if is_selected else _LINE_DIVIDER, 1)
        painter.setPen(line_pen)
        if text_x - 6 > left_pad:
            painter.drawLine(left_pad, mid_y, text_x - 6, mid_y)
        line2_start = text_rect.right() + 6
        if right_limit > line2_start:
            painter.drawLine(line2_start, mid_y, right_limit, mid_y)

        if is_selected:
            t_color = QColor("#F3F0EA")
        elif kind == KIND_ADD_COLLECTION:
            t_color = QColor("#F0A387") if is_hovered else _ADD_TEXT
        elif kind in (KIND_COLLECTION, KIND_SECTION_DIVIDER):
            t_color = _DIVIDER_ACTIVE if is_hovered else _DIVIDER_TEXT
        else:
            t_color = _DIVIDER_TEXT
        painter.setPen(t_color)
        painter.drawText(
            text_rect,
            int(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter),
            elided,
        )

        if count_text:
            painter.setFont(count_font)
            painter.setPen(QColor("#C8C2B8") if is_selected else _COUNT_COLOR)
            c_rect = QRect(rect.right() - count_w - 4, rect.top(), count_w, rect.height())
            painter.drawText(
                c_rect,
                int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                count_text,
            )

    def _paint_standard_row(
        self,
        painter: QPainter,
        rect: QRect,
        base_font: QFont,
        *,
        kind: str,
        text: str,
        count_text: str,
        is_offline: bool,
        is_selected: bool,
    ) -> None:
        text_x = rect.left() + 4
        if kind in _QUICK_ICON_KINDS:
            icon_sz = max(12, min(15, int(round(12 * self._scale))))
            icon_rect = QRect(
                rect.left() + 5, rect.top() + (rect.height() - icon_sz) // 2, icon_sz, icon_sz
            )
            self._paint_quick_icon(
                painter,
                icon_rect,
                kind=kind,
                count_text=count_text,
                is_selected=is_selected,
            )
            text_x = icon_rect.right() + 6

        right_badge = ""
        if is_offline and count_text:
            right_badge = f"离线 · {count_text}"
        elif is_offline:
            right_badge = "离线"
        elif count_text:
            right_badge = count_text

        badge_w = 0
        if right_badge:
            count_font = theme.derived_font(base_font, -2, min_px=9)
            painter.setFont(count_font)
            cfm = painter.fontMetrics()
            badge_w = cfm.horizontalAdvance(right_badge) + 8
            badge_rect = QRect(rect.right() - badge_w - 4, rect.top(), badge_w, rect.height())
            painter.setPen(
                QColor("#C8C2B8")
                if is_selected
                else (theme.OFFLINE if is_offline else _COUNT_COLOR)
            )
            painter.drawText(
                badge_rect,
                int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                right_badge,
            )

        painter.setFont(base_font)
        fm = painter.fontMetrics()
        avail_w = max(20, rect.right() - text_x - badge_w - 6)
        elided = fm.elidedText(text, Qt.TextElideMode.ElideRight, avail_w)

        if is_selected:
            t_color = QColor("#F3F0EA")
        elif is_offline:
            t_color = theme.OFFLINE
        else:
            t_color = theme.TEXT
        painter.setPen(t_color)
        painter.drawText(
            QRect(text_x, rect.top(), avail_w, rect.height()),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            elided,
        )

    @staticmethod
    def _paint_quick_icon(
        painter: QPainter,
        r: QRect,
        *,
        kind: str,
        count_text: str,
        is_selected: bool,
    ) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        has_items = count_text not in ("", "0", "0 组")
        if kind == KIND_FAVORITES and (has_items or is_selected):
            color = theme.PLAY
        elif kind == KIND_DUPLICATES and has_items and not is_selected:
            color = QColor("#D99B57")
        elif is_selected:
            color = QColor("#F0A387")
        else:
            color = _ICON_MUTED

        pen = QPen(color, 1.2)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        if kind == _KIND_ALL:
            s = 4
            painter.drawRect(QRectF(r.left() + 1, r.top() + 1, s, s))
            painter.drawRect(QRectF(r.left() + 7, r.top() + 1, s, s))
            painter.drawRect(QRectF(r.left() + 1, r.top() + 7, s, s))
            painter.drawRect(QRectF(r.left() + 7, r.top() + 7, s, s))
        elif kind == KIND_FAVORITES:
            cx, cy = r.center().x() + 0.5, r.center().y() + 0.5
            pts: list[QPointF] = []
            for i in range(10):
                ang = math.radians(-90 + i * 36)
                rad = 5.2 if i % 2 == 0 else 2.2
                pts.append(QPointF(cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
            painter.drawPolygon(QPolygonF(pts))
        elif kind == KIND_RECENT:
            painter.drawEllipse(QRectF(r.left() + 1, r.top() + 1, 10, 10))
            cx, cy = r.left() + 6.0, r.top() + 6.0
            painter.drawLine(QPointF(cx, cy), QPointF(cx, cy - 3.0))
            painter.drawLine(QPointF(cx, cy), QPointF(cx + 2.4, cy + 1.4))
        elif kind == KIND_DUPLICATES:
            painter.drawRect(QRectF(r.left() + 1.2, r.top() + 3.2, 6.6, 7.2))
            painter.drawLine(QPointF(r.left() + 3.8, r.top() + 1.4), QPointF(r.left() + 10.4, r.top() + 1.4))
            painter.drawLine(QPointF(r.left() + 10.4, r.top() + 1.4), QPointF(r.left() + 10.4, r.top() + 8.6))


class FolderTree(QTreeWidget):
    """Unified REAPER ME-style sidebar: quick views, collections dividers, and physical folders."""

    locationSelected = Signal(object)  # TreeLocation
    viewSelected = Signal(object)  # SidebarViewTarget
    rescanRootRequested = Signal(int)  # root_id
    removeRootRequested = Signal(int)  # root_id
    relocateRootRequested = Signal(int)  # root_id
    relocateFolderRequested = Signal(int, int)  # (root_id, folder_id)
    refreshOnlineRequested = Signal(object)  # root_id | None
    addFolderToCollectionRequested = Signal(int, int, object, str, str)  # (collection_id, root_id, folder_id, rel_path, label)
    createCollectionRequested = Signal()
    renameCollectionRequested = Signal(int)
    deleteCollectionRequested = Signal(int)
    removeCollectionFolderRequested = Signal(int)  # collection_folders.id
    filesDroppedOnCollection = Signal(int, list)  # (collection_id, list[int])
    urlsDroppedOnCollection = Signal(int, list)  # (collection_id, list[str])
    importReaperShortcutsRequested = Signal()
    layoutStateChanged = Signal()

    def __init__(self, store: LibraryStore, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._store = store
        self._suppress_signal = False
        self._all_item: QTreeWidgetItem | None = None
        self._fav_item: QTreeWidgetItem | None = None
        self._recent_item: QTreeWidgetItem | None = None
        self._dup_item: QTreeWidgetItem | None = None
        self._local_div: QTreeWidgetItem | None = None
        self._local_roots_expanded: bool = False
        self._selection_is_ancestor_proxy: bool = False
        self._press_prev_item: QTreeWidgetItem | None = None
        self._collapsed_collections: set[int] = set()
        self._hover_branch_index: QModelIndex = QModelIndex()

        spec = theme.ui_scale_spec()
        self.setObjectName("sidebarTree")
        self.setHeaderHidden(True)
        self.setIndentation(spec.tree_indent)
        self.setUniformRowHeights(True)
        self._delegate = _SidebarItemDelegate(self)
        self.setItemDelegate(self._delegate)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        self.itemExpanded.connect(self._on_expanded)
        self.itemCollapsed.connect(self._on_collapsed)
        self.itemClicked.connect(self._on_item_clicked)
        self.currentItemChanged.connect(self._on_current_changed)
        self.customContextMenuRequested.connect(self._on_context_menu)

    def reload(
        self,
        *,
        keep_location: TreeLocation | None = None,
        keep_target: SidebarViewTarget | None = None,
    ) -> None:
        expanded_folders = self._snapshot_expanded_folders()
        self._suppress_signal = True
        try:
            self.clear()
            total = self._store.total_file_count()
            self._all_item = QTreeWidgetItem(["全部素材"])
            self._all_item.setData(0, _KIND_ROLE, _KIND_ALL)
            self._all_item.setData(0, _ROOT_ID_ROLE, None)
            self._all_item.setData(0, _FOLDER_ID_ROLE, None)
            self._all_item.setData(0, _REL_PATH_ROLE, "")
            self._all_item.setData(0, _FULL_PATH_ROLE, "全部素材")
            self._all_item.setData(0, _COUNT_ROLE, f"{total:,}")
            self.addTopLevelItem(self._all_item)

            fav_n = self._store.favorite_count()
            self._fav_item = QTreeWidgetItem(["我的收藏"])
            self._fav_item.setData(0, _KIND_ROLE, KIND_FAVORITES)
            self._fav_item.setData(0, _FULL_PATH_ROLE, "我的收藏")
            self._fav_item.setData(0, _COUNT_ROLE, f"{fav_n:,}")
            self.addTopLevelItem(self._fav_item)

            rec_n = self._store.recent_play_count()
            self._recent_item = QTreeWidgetItem(["最近试听"])
            self._recent_item.setData(0, _KIND_ROLE, KIND_RECENT)
            self._recent_item.setData(0, _FULL_PATH_ROLE, "最近试听")
            self._recent_item.setData(0, _COUNT_ROLE, f"{rec_n:,}")
            self.addTopLevelItem(self._recent_item)

            dup_groups, _, _, _ = self._store.duplicate_summary()
            self._dup_item = QTreeWidgetItem(["重复文件"])
            self._dup_item.setData(0, _KIND_ROLE, KIND_DUPLICATES)
            self._dup_item.setData(0, _FULL_PATH_ROLE, "重复文件")
            self._dup_item.setData(0, _COUNT_ROLE, f"{dup_groups:,} 组")
            self.addTopLevelItem(self._dup_item)

            roots = self._store.list_roots()
            roots_by_id = {r.id: r for r in roots}
            roots_with_sub = self._store.root_ids_with_folders()
            local_div = QTreeWidgetItem(["本地目录"])
            local_div.setData(0, _KIND_ROLE, KIND_SECTION_DIVIDER)
            local_div.setFlags(Qt.ItemFlag.ItemIsEnabled)
            if roots:
                local_div.setData(0, _COUNT_ROLE, f"{len(roots)} 个")
            local_div.setToolTip(0, "点击展开或收纳所有本地目录")
            self._local_div = local_div
            self.addTopLevelItem(local_div)

            for root in roots:
                local_div.addChild(
                    self._make_root_item(root, has_children=(root.id in roots_with_sub))
                )
            local_div.setExpanded(self._local_roots_expanded)

            collections = self._store.list_collections()
            if collections:
                all_cf_map = self._store.list_all_collection_folders()
                all_cf_flat = [cf for cfs in all_cf_map.values() for cf in cfs]
                folders_with_sub = self._store.folder_ids_with_children(
                    [cf.folder_id for cf in all_cf_flat if cf.folder_id is not None]
                )
                verify_disk_each = len(all_cf_flat) <= 50
                for coll in collections:
                    c_item = QTreeWidgetItem([coll.name])
                    c_item.setData(0, _KIND_ROLE, KIND_COLLECTION)
                    c_item.setData(0, _COLL_ID_ROLE, coll.id)
                    c_item.setData(0, _FULL_PATH_ROLE, coll.name)
                    c_item.setData(0, _COUNT_ROLE, f"{coll.file_count:,}")
                    c_item.setToolTip(
                        0,
                        f"合集「{coll.name}」：包含 {coll.folder_count} 个目录 / 共 {coll.file_count:,} 条（支持直接拖入音频）",
                    )
                    self.addTopLevelItem(c_item)
                    for cf in all_cf_map.get(coll.id, ()):
                        root_obj = roots_by_id.get(cf.root_id) or self._store.get_root(cf.root_id)
                        root_path = cf.root_path
                        full = (
                            str(Path(root_path) / Path(cf.folder_rel_path))
                            if cf.folder_rel_path
                            else root_path
                        )
                        root_online = (
                            root_obj.is_online
                            if root_obj is not None
                            else Path(root_path).exists()
                        )
                        folder_online = (
                            root_online and Path(full).exists()
                            if verify_disk_each
                            else root_online
                        )
                        cf_item = QTreeWidgetItem([cf.label])
                        cf_item.setData(0, _KIND_ROLE, KIND_COLLECTION_FOLDER)
                        cf_item.setData(0, _COLL_ID_ROLE, coll.id)
                        cf_item.setData(0, _CF_ID_ROLE, cf.id)
                        cf_item.setData(0, _ROOT_ID_ROLE, cf.root_id)
                        cf_item.setData(0, _FOLDER_ID_ROLE, cf.folder_id)
                        cf_item.setData(0, _REL_PATH_ROLE, cf.folder_rel_path)
                        cf_item.setData(0, _FULL_PATH_ROLE, full)
                        cf_item.setData(0, _OFFLINE_ROLE, not folder_online)
                        cf_item.setData(0, _LOADED_ROLE, False)
                        cf_item.setToolTip(0, full)
                        has_sub = (
                            cf.folder_id in folders_with_sub
                            if cf.folder_id is not None
                            else (cf.root_id in roots_with_sub)
                        )
                        if has_sub:
                            self._add_dummy(cf_item)
                        c_item.addChild(cf_item)
                    c_item.setExpanded(coll.id not in self._collapsed_collections)
            else:
                add_item = QTreeWidgetItem(["+ 新建合集分组"])
                add_item.setData(0, _KIND_ROLE, KIND_ADD_COLLECTION)
                add_item.setToolTip(0, "点击创建虚拟合集分组（像 REAPER ME 分割线一样收纳常用目录与音频）")
                self.addTopLevelItem(add_item)

            for rid, rp in sorted(expanded_folders, key=lambda x: (x[0], len(x[1]))):
                self._expand_folder_path(rid, rp)
            if self._local_div is not None:
                self._local_div.setExpanded(self._local_roots_expanded)

            if keep_target is not None:
                self._select_target_silent(keep_target)
            elif keep_location is not None and keep_location.root_id is not None:
                if not self.select_location(
                    keep_location.root_id, keep_location.folder_id, expand=False
                ):
                    self.setCurrentItem(self._all_item)
            else:
                self.setCurrentItem(self._all_item)
        finally:
            self._suppress_signal = False

    def refresh_quick_counts(self) -> None:
        if self._all_item is not None:
            self._all_item.setData(0, _COUNT_ROLE, f"{self._store.total_file_count():,}")
        if self._fav_item is not None:
            self._fav_item.setData(0, _COUNT_ROLE, f"{self._store.favorite_count():,}")
        if self._recent_item is not None:
            self._recent_item.setData(0, _COUNT_ROLE, f"{self._store.recent_play_count():,}")
        if self._dup_item is not None:
            dup_groups, _, _, _ = self._store.duplicate_summary()
            self._dup_item.setData(0, _COUNT_ROLE, f"{dup_groups:,} 组")
        self.viewport().update()

    def find_local_divider_item(self) -> QTreeWidgetItem | None:
        return self._local_div

    def is_local_roots_expanded(self) -> bool:
        return bool(self._local_roots_expanded)

    def set_local_roots_expanded(self, expanded: bool, *, emit: bool = False) -> None:
        changed = self._local_roots_expanded != bool(expanded)
        self._local_roots_expanded = bool(expanded)
        if self._local_div is not None:
            prev = self._suppress_signal
            if not emit:
                self._suppress_signal = True
            try:
                self._local_div.setExpanded(self._local_roots_expanded)
            finally:
                self._suppress_signal = prev
        if changed and emit and not self._suppress_signal:
            self.layoutStateChanged.emit()

    def collapsed_collection_ids(self) -> list[int]:
        return sorted(self._collapsed_collections)

    def set_collapsed_collection_ids(self, ids: list[int] | set[int] | tuple[int, ...]) -> None:
        self._collapsed_collections = {int(cid) for cid in ids}
        prev = self._suppress_signal
        self._suppress_signal = True
        try:
            for idx in range(self.topLevelItemCount()):
                item = self.topLevelItem(idx)
                if item.data(0, _KIND_ROLE) == KIND_COLLECTION:
                    cid = item.data(0, _COLL_ID_ROLE)
                    if cid is not None:
                        item.setExpanded(int(cid) not in self._collapsed_collections)
        finally:
            self._suppress_signal = prev

    def collapse_all_collections(self) -> None:
        target_ids: set[int] = set()
        for idx in range(self.topLevelItemCount()):
            item = self.topLevelItem(idx)
            if item.data(0, _KIND_ROLE) == KIND_COLLECTION:
                cid = item.data(0, _COLL_ID_ROLE)
                if cid is not None:
                    target_ids.add(int(cid))
        if not target_ids:
            return
        changed = target_ids != self._collapsed_collections
        self.set_collapsed_collection_ids(self._collapsed_collections | target_ids)
        if changed and not self._suppress_signal:
            self.layoutStateChanged.emit()

    def expand_all_collections(self) -> None:
        if not self._collapsed_collections:
            return
        self.set_collapsed_collection_ids(())
        if not self._suppress_signal:
            self.layoutStateChanged.emit()

    def find_root_item(self, root_id: int) -> QTreeWidgetItem | None:
        if self._local_div is not None:
            for idx in range(self._local_div.childCount()):
                cand = self._local_div.child(idx)
                if cand.data(0, _KIND_ROLE) == _KIND_ROOT and cand.data(0, _ROOT_ID_ROLE) == root_id:
                    return cand
        return None

    def find_quick_item(self, kind: str) -> QTreeWidgetItem | None:
        for idx in range(self.topLevelItemCount()):
            cand = self.topLevelItem(idx)
            if cand.data(0, _KIND_ROLE) == kind:
                return cand
        return None

    @staticmethod
    def is_item_offline(item: QTreeWidgetItem | None) -> bool:
        return False if item is None else bool(item.data(0, _OFFLINE_ROLE))

    def _snapshot_expanded_folders(self) -> set[tuple[int, str]]:
        out: set[tuple[int, str]] = set()

        def _walk(item: QTreeWidgetItem) -> None:
            if item.isExpanded():
                kind = item.data(0, _KIND_ROLE)
                if kind in (_KIND_ROOT, _KIND_FOLDER):
                    rid = item.data(0, _ROOT_ID_ROLE)
                    rp = str(item.data(0, _REL_PATH_ROLE) or "")
                    if rid is not None:
                        out.add((int(rid), rp))
            for i in range(item.childCount()):
                _walk(item.child(i))

        for idx in range(self.topLevelItemCount()):
            _walk(self.topLevelItem(idx))
        return out

    def _expand_folder_path(self, root_id: int, rel_path: str) -> None:
        root_item = self.find_root_item(root_id)
        if root_item is None:
            return
        if not root_item.data(0, _LOADED_ROLE):
            self._on_expanded(root_item)
        root_item.setExpanded(True)
        if not rel_path:
            return
        cur = root_item
        parts = [p for p in rel_path.split("/") if p]
        acc: list[str] = []
        for part in parts:
            acc.append(part)
            want = "/".join(acc)
            matched = None
            for i in range(cur.childCount()):
                ch = cur.child(i)
                if str(ch.data(0, _REL_PATH_ROLE) or "") == want:
                    matched = ch
                    break
            if matched is None:
                return
            if not matched.data(0, _LOADED_ROLE):
                self._on_expanded(matched)
            matched.setExpanded(True)
            cur = matched

    def _make_root_item(
        self, root: RootRecord, *, has_children: bool | None = None
    ) -> QTreeWidgetItem:
        item = QTreeWidgetItem([root.label])
        item.setData(0, _KIND_ROLE, _KIND_ROOT)
        item.setData(0, _ROOT_ID_ROLE, root.id)
        item.setData(0, _FOLDER_ID_ROLE, None)
        item.setData(0, _REL_PATH_ROLE, "")
        item.setData(0, _FULL_PATH_ROLE, root.path)
        item.setData(0, _LOADED_ROLE, False)
        item.setData(0, _COUNT_ROLE, f"{root.file_count:,}")
        item.setData(0, _OFFLINE_ROLE, not root.is_online)
        item.setToolTip(0, root.path)
        if not root.is_online:
            item.setForeground(0, theme.OFFLINE)
        if (
            bool(self._store.list_folders(root.id, parent_id=None))
            if has_children is None
            else has_children
        ):
            self._add_dummy(item)
        return item

    def _make_folder_item(
        self, folder: FolderRecord, root_path: str, *, root_online: bool = True
    ) -> QTreeWidgetItem:
        full = str(Path(root_path) / Path(folder.rel_path))
        folder_online = root_online and Path(full).exists()
        item = QTreeWidgetItem([folder.name])
        item.setData(0, _KIND_ROLE, _KIND_FOLDER)
        item.setData(0, _ROOT_ID_ROLE, folder.root_id)
        item.setData(0, _FOLDER_ID_ROLE, folder.id)
        item.setData(0, _REL_PATH_ROLE, folder.rel_path)
        item.setData(0, _FULL_PATH_ROLE, full)
        item.setData(0, _LOADED_ROLE, False)
        item.setData(0, _COUNT_ROLE, f"{folder.file_count:,}")
        item.setData(0, _OFFLINE_ROLE, not folder_online)
        item.setToolTip(0, full)
        if not folder_online:
            item.setForeground(0, theme.OFFLINE)
        if self._store.has_child_folders(folder.id):
            self._add_dummy(item)
        return item

    @staticmethod
    def _add_dummy(parent: QTreeWidgetItem) -> None:
        dummy = QTreeWidgetItem(["..."])
        dummy.setData(0, _KIND_ROLE, _KIND_DUMMY)
        parent.addChild(dummy)

    def _on_expanded(self, item: QTreeWidgetItem) -> None:
        kind = item.data(0, _KIND_ROLE)
        if kind == KIND_SECTION_DIVIDER:
            changed = not self._local_roots_expanded
            self._local_roots_expanded = True
            if changed and not self._suppress_signal:
                self.layoutStateChanged.emit()
            return
        if kind == KIND_COLLECTION:
            cid = item.data(0, _COLL_ID_ROLE)
            if cid is not None:
                cid_int = int(cid)
                changed = cid_int in self._collapsed_collections
                self._collapsed_collections.discard(cid_int)
                if changed and not self._suppress_signal:
                    self.layoutStateChanged.emit()
            return
        if kind not in (_KIND_ROOT, _KIND_FOLDER, KIND_COLLECTION_FOLDER):
            return
        if item.data(0, _LOADED_ROLE):
            return
        item.takeChildren()
        root_id = int(item.data(0, _ROOT_ID_ROLE))
        folder_id = item.data(0, _FOLDER_ID_ROLE)
        root = self._store.get_root(root_id)
        root_path = root.path if root is not None else ""
        root_online = root.is_online if root is not None else False
        children = self._store.list_folders(root_id, parent_id=folder_id)
        for child in children:
            child_item = self._make_folder_item(child, root_path, root_online=root_online)
            item.addChild(child_item)
        item.setData(0, _LOADED_ROLE, True)

    def _on_collapsed(self, item: QTreeWidgetItem) -> None:
        kind = item.data(0, _KIND_ROLE)
        if kind == KIND_SECTION_DIVIDER:
            changed = self._local_roots_expanded
            self._local_roots_expanded = False
            if changed and not self._suppress_signal:
                self.layoutStateChanged.emit()
            return
        if kind == KIND_COLLECTION:
            cid = item.data(0, _COLL_ID_ROLE)
            if cid is not None:
                cid_int = int(cid)
                changed = cid_int not in self._collapsed_collections
                self._collapsed_collections.add(cid_int)
                if changed and not self._suppress_signal:
                    self.layoutStateChanged.emit()

    def apply_ui_scale(self, scale_key: str) -> None:
        spec = theme.ui_scale_spec(scale_key)
        self.setIndentation(spec.tree_indent)
        self._delegate.apply_ui_scale(spec)
        self.scheduleDelayedItemsLayout()
        self.viewport().update()

    def _branch_hit_rect(self, index: QModelIndex) -> QRect:
        if not index.isValid():
            return QRect()
        model = self.model()
        if model is None or not model.hasChildren(index):
            return QRect()
        vrect = self.visualRect(index)
        if not vrect.isValid():
            return QRect()
        indent = max(16, self.indentation())
        left = max(0, vrect.left() - indent - 4)
        right = vrect.left() + 4
        return QRect(left, vrect.top(), max(20, right - left + 1), vrect.height())

    def drawBranches(self, painter: QPainter, rect: QRect, index: QModelIndex) -> None:  # noqa: N802
        if not index.isValid():
            return
        model = self.model()
        if model is None or not model.hasChildren(index):
            return
        indent = max(16, self.indentation())
        slot_w = min(rect.width(), indent)
        slot = QRectF(rect.right() - slot_w + 1, rect.top(), slot_w, rect.height())
        is_expanded = self.isExpanded(index)
        is_hovered = self._hover_branch_index.isValid() and self._hover_branch_index == index
        item = self.itemFromIndex(index)
        is_selected = item is not None and (item.isSelected() or item is self.currentItem())
        spec = theme.ui_scale_spec()
        half = max(3.4, min(5.2, 3.8 * spec.scale))

        painter.save()
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            if is_hovered:
                pill = QRectF(
                    slot.center().x() - (half + 4.0),
                    slot.center().y() - (half + 4.0),
                    (half + 4.0) * 2.0,
                    (half + 4.0) * 2.0,
                )
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor("#2D2A26"))
                painter.drawRoundedRect(pill, 4.0, 4.0)

            color = (
                QColor("#F0A387")
                if is_hovered
                else (QColor("#D9CFC0") if is_selected else QColor("#A5A096"))
            )
            pen = QPen(
                color,
                1.7,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            cx = slot.center().x()
            cy = slot.center().y() + 0.5
            path = QPainterPath()
            if is_expanded:
                path.moveTo(cx - half, cy - half * 0.45)
                path.lineTo(cx, cy + half * 0.55)
                path.lineTo(cx + half, cy - half * 0.45)
            else:
                path.moveTo(cx - half * 0.45, cy - half)
                path.lineTo(cx + half * 0.55, cy)
                path.lineTo(cx - half * 0.45, cy + half)
            painter.drawPath(path)
        finally:
            painter.restore()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        pos = event.position().toPoint()
        idx = self.indexAt(pos)
        if not idx.isValid():
            idx = self.indexAt(QPoint(max(0, self.indentation()) + 2, pos.y()))
        hover_idx = idx if (idx.isValid() and self._branch_hit_rect(idx).contains(pos)) else QModelIndex()
        if hover_idx != self._hover_branch_index:
            self._hover_branch_index = hover_idx
            self.viewport().update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        if self._hover_branch_index.isValid():
            self._hover_branch_index = QModelIndex()
            self.viewport().update()
        super().leaveEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self._press_prev_item = self.currentItem()
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position().toPoint()
            idx = self.indexAt(pos)
            if not idx.isValid():
                idx = self.indexAt(QPoint(max(0, self.indentation()) + 2, pos.y()))
            if idx.isValid() and self._branch_hit_rect(idx).contains(pos):
                self.setExpanded(idx, not self.isExpanded(idx))
                event.accept()
                return
        super().mousePressEvent(event)

    def _on_item_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        if self._suppress_signal or item is None:
            return
        kind = item.data(0, _KIND_ROLE)
        if kind == KIND_SECTION_DIVIDER:
            self._set_local_divider_highlighted(False)
            if item.childCount() > 0:
                item.setExpanded(not item.isExpanded())
            return
        same_as_before = item is self._press_prev_item
        self._press_prev_item = None
        if item is self.currentItem() and (self._selection_is_ancestor_proxy or same_as_before):
            self._selection_is_ancestor_proxy = False
            if kind in (_KIND_ALL, _KIND_ROOT, _KIND_FOLDER):
                self.locationSelected.emit(self._location_for_item(item))
            elif kind in (
                KIND_FAVORITES,
                KIND_RECENT,
                KIND_DUPLICATES,
                KIND_COLLECTION,
                KIND_COLLECTION_FOLDER,
            ):
                self.viewSelected.emit(
                    SidebarViewTarget(
                        kind=str(kind),
                        collection_id=item.data(0, _COLL_ID_ROLE),
                        root_id=item.data(0, _ROOT_ID_ROLE),
                        folder_id=item.data(0, _FOLDER_ID_ROLE),
                        folder_rel_path=str(item.data(0, _REL_PATH_ROLE) or ""),
                        display_path=str(item.data(0, _FULL_PATH_ROLE) or ""),
                    )
                )

    def _set_local_divider_highlighted(self, active: bool) -> None:
        if self._local_div is None:
            return
        cur = bool(self._local_div.data(0, _ACTIVE_PROXY_ROLE))
        new_val = bool(active)
        if cur != new_val:
            self._local_div.setData(0, _ACTIVE_PROXY_ROLE, new_val)
            self.viewport().update()

    def is_local_divider_highlighted(self) -> bool:
        return bool(
            self._local_div is not None and self._local_div.data(0, _ACTIVE_PROXY_ROLE)
        )

    def _on_current_changed(self, current: QTreeWidgetItem | None, _prev) -> None:
        if self._suppress_signal or current is None:
            return
        self._set_local_divider_highlighted(False)
        self._selection_is_ancestor_proxy = False
        kind = current.data(0, _KIND_ROLE)
        if kind in (_KIND_DUMMY, KIND_SECTION_DIVIDER):
            return
        if kind == KIND_ADD_COLLECTION:
            self.clear_selection_silent()
            self.createCollectionRequested.emit()
            return
        if kind in (
            KIND_FAVORITES,
            KIND_RECENT,
            KIND_DUPLICATES,
            KIND_COLLECTION,
            KIND_COLLECTION_FOLDER,
        ):
            target = SidebarViewTarget(
                kind=str(kind),
                collection_id=current.data(0, _COLL_ID_ROLE),
                root_id=current.data(0, _ROOT_ID_ROLE),
                folder_id=current.data(0, _FOLDER_ID_ROLE),
                folder_rel_path=str(current.data(0, _REL_PATH_ROLE) or ""),
                display_path=str(current.data(0, _FULL_PATH_ROLE) or ""),
            )
            self.viewSelected.emit(target)
            return
        self.locationSelected.emit(self._location_for_item(current))

    def _location_for_item(self, item: QTreeWidgetItem) -> TreeLocation:
        return TreeLocation(
            root_id=item.data(0, _ROOT_ID_ROLE),
            folder_id=item.data(0, _FOLDER_ID_ROLE),
            folder_rel_path=str(item.data(0, _REL_PATH_ROLE) or ""),
            display_path=str(item.data(0, _FULL_PATH_ROLE) or "全部素材"),
        )

    def clear_selection_silent(self) -> None:
        self._suppress_signal = True
        try:
            self._set_local_divider_highlighted(False)
            self._selection_is_ancestor_proxy = False
            self.setCurrentItem(None)
            self.clearSelection()
        finally:
            self._suppress_signal = False

    def _select_target_silent(self, target: SidebarViewTarget) -> None:
        self._suppress_signal = True
        try:
            self._set_local_divider_highlighted(False)
            self._selection_is_ancestor_proxy = False
            if target.kind in (KIND_FAVORITES, KIND_RECENT, KIND_DUPLICATES):
                item = self.find_quick_item(target.kind)
                if item is not None:
                    self.setCurrentItem(item)
                return
            for idx in range(self.topLevelItemCount()):
                c_item = self.topLevelItem(idx)
                if c_item.data(0, _KIND_ROLE) != KIND_COLLECTION:
                    continue
                if c_item.data(0, _COLL_ID_ROLE) != target.collection_id:
                    continue
                if target.kind == KIND_COLLECTION:
                    self.setCurrentItem(c_item)
                    return
                for j in range(c_item.childCount()):
                    cf_item = c_item.child(j)
                    if (
                        cf_item.data(0, _ROOT_ID_ROLE) == target.root_id
                        and str(cf_item.data(0, _REL_PATH_ROLE) or "") == target.folder_rel_path
                    ):
                        self.setCurrentItem(cf_item)
                        return
                self.setCurrentItem(c_item)
                return
        finally:
            self._suppress_signal = False

    def _find_best_shortcut_item(
        self, root_id: int, folder_id: int | None, target_rel: str
    ) -> tuple[QTreeWidgetItem, QTreeWidgetItem, bool] | None:
        target_clean = target_rel.strip("/")
        target_cf = target_clean.casefold()
        best: tuple[tuple[int, int, int, int], QTreeWidgetItem, QTreeWidgetItem, bool] | None = None
        order = 0
        for idx in range(self.topLevelItemCount()):
            c_item = self.topLevelItem(idx)
            if c_item.data(0, _KIND_ROLE) != KIND_COLLECTION:
                continue
            for j in range(c_item.childCount()):
                cf_item = c_item.child(j)
                if cf_item.data(0, _KIND_ROLE) != KIND_COLLECTION_FOLDER:
                    continue
                if cf_item.data(0, _ROOT_ID_ROLE) != root_id:
                    continue
                order += 1
                cf_fid = cf_item.data(0, _FOLDER_ID_ROLE)
                cf_rel = str(cf_item.data(0, _REL_PATH_ROLE) or "").strip("/")
                cf_cf = cf_rel.casefold()
                if folder_id is not None and cf_fid is not None:
                    is_exact = cf_fid == folder_id
                else:
                    is_exact = cf_cf == target_cf and (folder_id is None) == (cf_fid is None)
                is_ancestor = (not is_exact) and (
                    (not cf_cf and bool(target_cf))
                    or (bool(cf_cf) and target_cf.startswith(cf_cf + "/"))
                )
                if not is_exact and not is_ancestor:
                    continue
                score = (
                    1 if is_exact else 0,
                    len(cf_rel),
                    1 if c_item.isExpanded() else 0,
                    -order,
                )
                if best is None or score > best[0]:
                    best = (score, c_item, cf_item, is_exact)
        return None if best is None else (best[1], best[2], best[3])

    def _folder_ancestor_chain(self, folder_id: int | None) -> list[int]:
        chain: list[int] = []
        cur_id: int | None = folder_id
        while cur_id is not None:
            rec = self._store.get_folder(cur_id)
            if rec is None:
                break
            chain.append(rec.id)
            cur_id = rec.parent_id
        chain.reverse()
        return chain

    def select_location(
        self,
        root_id: int | None,
        folder_id: int | None,
        *,
        expand: bool = False,
        prefer_shortcut: bool = False,
        scroll: bool = False,
    ) -> bool:
        """Highlight matching node without emitting duplicate locationSelected or unwanted auto-expansion."""
        div_was_expanded = self._local_div.isExpanded() if self._local_div is not None else False
        did_change_layout = False
        self._suppress_signal = True
        try:
            self._set_local_divider_highlighted(False)
            if root_id is None:
                self._selection_is_ancestor_proxy = False
                if self._all_item is not None:
                    self.setCurrentItem(self._all_item)
                    if scroll or expand:
                        self.scrollToItem(
                            self._all_item, QAbstractItemView.ScrollHint.PositionAtCenter
                        )
                    return True
                return False

            target_rel = ""
            if folder_id is not None:
                target_rec = self._store.get_folder(folder_id)
                if target_rec is not None:
                    target_rel = target_rec.rel_path

            if prefer_shortcut:
                shortcut_hit = self._find_best_shortcut_item(root_id, folder_id, target_rel)
                if shortcut_hit is not None:
                    c_item, cf_item, is_exact = shortcut_hit
                    if not c_item.isExpanded():
                        c_item.setExpanded(True)
                        cid = c_item.data(0, _COLL_ID_ROLE)
                        if cid is not None:
                            self._collapsed_collections.discard(int(cid))
                        did_change_layout = True
                    target_item = cf_item
                    reached_exact = is_exact
                    if not is_exact and cf_item.isExpanded() and cf_item.data(0, _LOADED_ROLE):
                        chain = self._folder_ancestor_chain(folder_id)
                        cf_fid = cf_item.data(0, _FOLDER_ID_ROLE)
                        if cf_fid is None:
                            sub_chain = chain
                        elif cf_fid in chain:
                            sub_chain = chain[chain.index(cf_fid) + 1 :]
                        else:
                            sub_chain = []
                        cur = cf_item
                        reached_exact = bool(sub_chain)
                        for target_fid in sub_chain:
                            if not cur.isExpanded() or not cur.data(0, _LOADED_ROLE):
                                reached_exact = False
                                break
                            matched = None
                            for c_idx in range(cur.childCount()):
                                ch = cur.child(c_idx)
                                if ch.data(0, _FOLDER_ID_ROLE) == target_fid:
                                    matched = ch
                                    break
                            if matched is None:
                                reached_exact = False
                                break
                            cur = matched
                        target_item = cur
                    self._selection_is_ancestor_proxy = not reached_exact
                    self.setCurrentItem(target_item)
                    if scroll or expand:
                        self.scrollToItem(
                            target_item, QAbstractItemView.ScrollHint.PositionAtCenter
                        )
                    return True

            if not expand and not prefer_shortcut:
                cur_item = self.currentItem()
                if (
                    cur_item is not None
                    and cur_item.data(0, _KIND_ROLE) == KIND_COLLECTION_FOLDER
                    and cur_item.data(0, _ROOT_ID_ROLE) == root_id
                ):
                    cur_fid = cur_item.data(0, _FOLDER_ID_ROLE)
                    cur_rel = str(cur_item.data(0, _REL_PATH_ROLE) or "")
                    is_exact = cur_fid == folder_id
                    is_descendant = folder_id is not None and (
                        (not cur_rel and bool(target_rel))
                        or (bool(cur_rel) and target_rel.startswith(cur_rel + "/"))
                    )
                    if is_exact or is_descendant:
                        self._selection_is_ancestor_proxy = not is_exact
                        return True

            root_item = self.find_root_item(root_id)
            if root_item is None:
                return False

            if self._local_div is not None and not self._local_div.isExpanded():
                if expand:
                    self._local_div.setExpanded(True)
                    self._local_roots_expanded = True
                    did_change_layout = not div_was_expanded
                else:
                    self.setCurrentItem(None)
                    self.clearSelection()
                    self._selection_is_ancestor_proxy = False
                    if prefer_shortcut or scroll:
                        self._set_local_divider_highlighted(True)
                        self.scrollToItem(
                            self._local_div, QAbstractItemView.ScrollHint.PositionAtCenter
                        )
                    return True

            if folder_id is None:
                self._selection_is_ancestor_proxy = False
                self.setCurrentItem(root_item)
                if scroll or expand:
                    self.scrollToItem(root_item, QAbstractItemView.ScrollHint.PositionAtCenter)
                return True

            chain = self._folder_ancestor_chain(folder_id)
            parent_item = root_item
            reached_exact = True
            for target_fid in chain:
                if not expand:
                    if not parent_item.isExpanded() or not parent_item.data(0, _LOADED_ROLE):
                        reached_exact = False
                        break
                else:
                    if not parent_item.data(0, _LOADED_ROLE):
                        self._on_expanded(parent_item)
                    parent_item.setExpanded(True)
                matched = None
                for c_idx in range(parent_item.childCount()):
                    c_item = parent_item.child(c_idx)
                    if c_item.data(0, _FOLDER_ID_ROLE) == target_fid:
                        matched = c_item
                        break
                if matched is None:
                    if not expand:
                        reached_exact = False
                        break
                    return False
                parent_item = matched
            self._selection_is_ancestor_proxy = not reached_exact
            self.setCurrentItem(parent_item)
            if scroll or expand:
                self.scrollToItem(parent_item, QAbstractItemView.ScrollHint.PositionAtCenter)
            return True
        finally:
            self._suppress_signal = False
            if did_change_layout:
                self.layoutStateChanged.emit()

    # -- Drag & Drop onto Collection dividers / child nodes ----------------
    @staticmethod
    def _has_droppable_payload(event) -> bool:
        mime = event.mimeData()
        return bool(mime.hasFormat(FILE_IDS_MIME) or mime.hasUrls())

    @staticmethod
    def _collection_id_for_item(item: QTreeWidgetItem | None) -> int | None:
        cur = item
        while cur is not None:
            if cur.data(0, _KIND_ROLE) in (KIND_COLLECTION, KIND_COLLECTION_FOLDER):
                cid = cur.data(0, _COLL_ID_ROLE)
                return None if cid is None else int(cid)
            cur = cur.parent()
        return None

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if self._has_droppable_payload(event):
            event.acceptProposedAction()
            return
        event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:  # noqa: N802
        if not self._has_droppable_payload(event):
            event.ignore()
            return
        item = self.itemAt(event.position().toPoint())
        if self._collection_id_for_item(item) is not None:
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        if not self._has_droppable_payload(event):
            event.ignore()
            return
        coll_id = self._collection_id_for_item(self.itemAt(event.position().toPoint()))
        if coll_id is None:
            event.ignore()
            return
        mime = event.mimeData()
        if mime.hasFormat(FILE_IDS_MIME):
            try:
                raw = bytes(mime.data(FILE_IDS_MIME)).decode("utf-8")
                file_ids = [int(x) for x in json.loads(raw)]
            except Exception:
                event.ignore()
                return
            if file_ids:
                self.filesDroppedOnCollection.emit(coll_id, file_ids)
                event.acceptProposedAction()
                return
        if mime.hasUrls():
            paths = [u.toLocalFile() for u in mime.urls() if u.isLocalFile() and u.toLocalFile()]
            if paths:
                self.urlsDroppedOnCollection.emit(coll_id, paths)
                event.acceptProposedAction()

    def _add_collection_menu_helpers(self, menu: QMenu) -> None:
        new_act = QAction("+ 新建合集分组…", menu)
        new_act.triggered.connect(lambda _c=False: self.createCollectionRequested.emit())
        menu.addAction(new_act)

        imp_act = QAction("从 REAPER 导入快捷方式…", menu)
        imp_act.triggered.connect(lambda _c=False: self.importReaperShortcutsRequested.emit())
        menu.addAction(imp_act)

    def _add_collapse_expand_actions(self, menu: QMenu) -> None:
        has_collections = any(
            self.topLevelItem(i).data(0, _KIND_ROLE) == KIND_COLLECTION
            for i in range(self.topLevelItemCount())
        )
        if not has_collections:
            return
        menu.addSeparator()
        collapse_all_act = QAction("全部折叠分组", menu)
        collapse_all_act.triggered.connect(lambda _c=False: self.collapse_all_collections())
        menu.addAction(collapse_all_act)

        expand_all_act = QAction("全部展开分组", menu)
        expand_all_act.triggered.connect(lambda _c=False: self.expand_all_collections())
        menu.addAction(expand_all_act)

    # -- Context menu ------------------------------------------------------
    def _on_context_menu(self, pos) -> None:
        item = self.itemAt(pos)
        menu = QMenu(self)
        if item is None:
            self._add_collection_menu_helpers(menu)
            self._add_collapse_expand_actions(menu)
            menu.exec(self.viewport().mapToGlobal(pos))
            return

        kind = item.data(0, _KIND_ROLE)
        if kind in (KIND_SECTION_DIVIDER, KIND_ADD_COLLECTION, KIND_COLLECTION):
            self._add_collection_menu_helpers(menu)
            if kind == KIND_COLLECTION:
                cid = int(item.data(0, _COLL_ID_ROLE))
                menu.addSeparator()
                ren_act = QAction("重命名分组…", menu)
                ren_act.triggered.connect(
                    lambda _c=False, x=cid: self.renameCollectionRequested.emit(x)
                )
                menu.addAction(ren_act)

                del_act = QAction("删除分组（不删原文件）", menu)
                del_act.triggered.connect(
                    lambda _c=False, x=cid: self.deleteCollectionRequested.emit(x)
                )
                menu.addAction(del_act)
                self._add_collapse_expand_actions(menu)
            menu.exec(self.viewport().mapToGlobal(pos))
            return

        if kind == KIND_COLLECTION_FOLDER:
            cf_id = int(item.data(0, _CF_ID_ROLE))
            full_path = str(item.data(0, _FULL_PATH_ROLE) or "")
            if full_path and Path(full_path).exists():
                open_act = QAction("在资源管理器中打开", menu)
                open_act.triggered.connect(lambda _c=False, p=full_path: os.startfile(p))  # noqa: S606
                menu.addAction(open_act)
            rm_act = QAction("从合集移出此目录", menu)
            rm_act.triggered.connect(
                lambda _c=False, fid=cf_id: self.removeCollectionFolderRequested.emit(fid)
            )
            menu.addAction(rm_act)
            menu.exec(self.viewport().mapToGlobal(pos))
            return

        if kind not in (_KIND_ROOT, _KIND_FOLDER):
            return

        full_path = str(item.data(0, _FULL_PATH_ROLE) or "")
        root_id = int(item.data(0, _ROOT_ID_ROLE))
        folder_id = item.data(0, _FOLDER_ID_ROLE)
        rel_path = str(item.data(0, _REL_PATH_ROLE) or "")
        folder_label = str(item.data(0, Qt.ItemDataRole.DisplayRole) or "")

        if full_path and Path(full_path).exists():
            open_act = QAction("在资源管理器中打开", menu)
            open_act.triggered.connect(lambda _c=False, p=full_path: os.startfile(p))  # noqa: S606
            menu.addAction(open_act)

        add_coll_menu = menu.addMenu("加入合集分组")
        for coll in self._store.list_collections():
            act = QAction(coll.name, add_coll_menu)
            act.triggered.connect(
                lambda _c=False, cid=coll.id, rid=root_id, fid=folder_id, rp=rel_path, lbl=folder_label: self.addFolderToCollectionRequested.emit(
                    cid, rid, fid, rp, lbl
                )
            )
            add_coll_menu.addAction(act)
        if add_coll_menu.actions():
            add_coll_menu.addSeparator()
        new_coll_act = QAction("+ 新建合集并加入…", add_coll_menu)
        new_coll_act.triggered.connect(
            lambda _c=False, rid=root_id, fid=folder_id, rp=rel_path, lbl=folder_label: self.addFolderToCollectionRequested.emit(
                0, rid, fid, rp, lbl
            )
        )
        add_coll_menu.addAction(new_coll_act)

        menu.addSeparator()
        refresh_act = QAction("刷新在线状态", menu)
        refresh_act.triggered.connect(
            lambda _c=False, rid=root_id: self.refreshOnlineRequested.emit(rid)
        )
        menu.addAction(refresh_act)

        if kind == _KIND_FOLDER and folder_id is not None:
            relocate_sub_act = QAction("重定位此目录路径…", menu)
            relocate_sub_act.triggered.connect(
                lambda _c=False, rid=root_id, fid=int(folder_id): self.relocateFolderRequested.emit(
                    rid, fid
                )
            )
            menu.addAction(relocate_sub_act)

        if kind == _KIND_ROOT:
            relocate_act = QAction("重定位根目录路径…", menu)
            relocate_act.triggered.connect(
                lambda _c=False, rid=root_id: self.relocateRootRequested.emit(rid)
            )
            menu.addAction(relocate_act)

            rescan_act = QAction("重新扫描此目录", menu)
            rescan_act.triggered.connect(
                lambda _c=False, rid=root_id: self.rescanRootRequested.emit(rid)
            )
            menu.addAction(rescan_act)

            remove_act = QAction("从库中移除", menu)
            remove_act.triggered.connect(
                lambda _c=False, rid=root_id: self.removeRootRequested.emit(rid)
            )
            menu.addAction(remove_act)

        if not menu.isEmpty():
            menu.exec(self.viewport().mapToGlobal(pos))
