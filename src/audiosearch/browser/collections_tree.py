"""Pinned quick-views & unified 'My Collections' (folders + files) sidebar tree."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QDragEnterEvent, QDragMoveEvent, QDropEvent
from PySide6.QtWidgets import QAbstractItemView, QMenu, QTreeWidget, QTreeWidgetItem, QWidget

from audiosearch.browser.file_drag import FILE_IDS_MIME
from audiosearch.library.store import CollectionRecord, LibraryStore

_KIND_ROLE = Qt.ItemDataRole.UserRole
_COLL_ID_ROLE = Qt.ItemDataRole.UserRole + 1
_CF_ID_ROLE = Qt.ItemDataRole.UserRole + 2
_ROOT_ID_ROLE = Qt.ItemDataRole.UserRole + 3
_FOLDER_ID_ROLE = Qt.ItemDataRole.UserRole + 4
_REL_PATH_ROLE = Qt.ItemDataRole.UserRole + 5
_FULL_PATH_ROLE = Qt.ItemDataRole.UserRole + 6

KIND_FAVORITES = "favorites"
KIND_RECENT = "recent"
KIND_DUPLICATES = "duplicates"
KIND_COLLECTIONS_HEADER = "collections_header"
KIND_COLLECTION = "collection"
KIND_COLLECTION_FOLDER = "collection_folder"


@dataclass(frozen=True, slots=True)
class SidebarViewTarget:
    kind: str  # "favorites" | "recent" | "duplicates" | "collection" | "collection_folder"
    collection_id: int | None = None
    root_id: int | None = None
    folder_id: int | None = None
    folder_rel_path: str = ""
    display_path: str = ""


class CollectionsTree(QTreeWidget):
    """Quick views (★ Favorites, Recent, Duplicates) + Unified Collections (folders + files)."""

    viewSelected = Signal(object)  # SidebarViewTarget
    createCollectionRequested = Signal()
    renameCollectionRequested = Signal(int)
    deleteCollectionRequested = Signal(int)
    removeCollectionFolderRequested = Signal(int)  # collection_folders.id
    filesDroppedOnCollection = Signal(int, list)  # (collection_id, list[int])
    urlsDroppedOnCollection = Signal(int, list)  # (collection_id, list[str])

    def __init__(self, store: LibraryStore, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._store = store
        self._suppress_signal = False
        self._fav_item: QTreeWidgetItem | None = None
        self._recent_item: QTreeWidgetItem | None = None
        self._dup_item: QTreeWidgetItem | None = None
        self._header_item: QTreeWidgetItem | None = None

        self.setHeaderHidden(True)
        self.setIndentation(14)
        self.setUniformRowHeights(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        self.currentItemChanged.connect(self._on_current_changed)
        self.customContextMenuRequested.connect(self._on_context_menu)

    def reload(self, *, keep_target: SidebarViewTarget | None = None) -> None:
        self._suppress_signal = True
        try:
            self.clear()
            fav_n = self._store.favorite_count()
            rec_n = self._store.recent_play_count()
            dup_groups, _total, _red, _bytes = self._store.duplicate_summary()

            self._fav_item = QTreeWidgetItem([f"★ 我的收藏 ({fav_n:,})"])
            self._fav_item.setData(0, _KIND_ROLE, KIND_FAVORITES)
            self._fav_item.setData(0, _FULL_PATH_ROLE, "★ 我的收藏")
            self.addTopLevelItem(self._fav_item)

            self._recent_item = QTreeWidgetItem([f"🕒 最近试听 ({rec_n:,})"])
            self._recent_item.setData(0, _KIND_ROLE, KIND_RECENT)
            self._recent_item.setData(0, _FULL_PATH_ROLE, "🕒 最近试听")
            self.addTopLevelItem(self._recent_item)

            self._dup_item = QTreeWidgetItem([f"⚠ 重复文件 ({dup_groups:,} 组)"])
            self._dup_item.setData(0, _KIND_ROLE, KIND_DUPLICATES)
            self._dup_item.setData(0, _FULL_PATH_ROLE, "⚠ 重复文件")
            self.addTopLevelItem(self._dup_item)

            collections: list[CollectionRecord] = self._store.list_collections()
            self._header_item = QTreeWidgetItem([f"我的合集 ({len(collections)})"])
            self._header_item.setData(0, _KIND_ROLE, KIND_COLLECTIONS_HEADER)
            self._header_item.setToolTip(0, "右键可新建合集；可将目录或列表素材拖入合集分组")
            self.addTopLevelItem(self._header_item)

            for coll in collections:
                c_item = QTreeWidgetItem([f"📁 {coll.name} ({coll.file_count:,})"])
                c_item.setData(0, _KIND_ROLE, KIND_COLLECTION)
                c_item.setData(0, _COLL_ID_ROLE, coll.id)
                c_item.setData(0, _FULL_PATH_ROLE, f"合集：{coll.name}")
                self._header_item.addChild(c_item)
                for cf in self._store.list_collection_folders(coll.id):
                    full = (
                        str(Path(cf.root_path) / Path(cf.folder_rel_path))
                        if cf.folder_rel_path
                        else cf.root_path
                    )
                    f_item = QTreeWidgetItem([f"📂 {cf.label}"])
                    f_item.setData(0, _KIND_ROLE, KIND_COLLECTION_FOLDER)
                    f_item.setData(0, _COLL_ID_ROLE, coll.id)
                    f_item.setData(0, _CF_ID_ROLE, cf.id)
                    f_item.setData(0, _ROOT_ID_ROLE, cf.root_id)
                    f_item.setData(0, _FOLDER_ID_ROLE, cf.folder_id)
                    f_item.setData(0, _REL_PATH_ROLE, cf.folder_rel_path)
                    f_item.setData(0, _FULL_PATH_ROLE, full)
                    f_item.setToolTip(0, full)
                    c_item.addChild(f_item)

            self._header_item.setExpanded(True)
            if keep_target is not None:
                self._select_target_silent(keep_target)
        finally:
            self._suppress_signal = False

    def refresh_quick_counts(self) -> None:
        if self._fav_item is not None:
            self._fav_item.setText(0, f"★ 我的收藏 ({self._store.favorite_count():,})")
        if self._recent_item is not None:
            self._recent_item.setText(0, f"🕒 最近试听 ({self._store.recent_play_count():,})")
        if self._dup_item is not None:
            dup_groups, _, _, _ = self._store.duplicate_summary()
            self._dup_item.setText(0, f"⚠ 重复文件 ({dup_groups:,} 组)")

    def clear_selection_silent(self) -> None:
        self._suppress_signal = True
        try:
            self.setCurrentItem(None)
            self.clearSelection()
        finally:
            self._suppress_signal = False

    def _select_target_silent(self, target: SidebarViewTarget) -> None:
        if target.kind == KIND_FAVORITES and self._fav_item is not None:
            self.setCurrentItem(self._fav_item)
        elif target.kind == KIND_RECENT and self._recent_item is not None:
            self.setCurrentItem(self._recent_item)
        elif target.kind == KIND_DUPLICATES and self._dup_item is not None:
            self.setCurrentItem(self._dup_item)
        elif target.kind == KIND_COLLECTION and target.collection_id is not None and self._header_item is not None:
            for idx in range(self._header_item.childCount()):
                ch = self._header_item.child(idx)
                if ch.data(0, _COLL_ID_ROLE) == target.collection_id:
                    self.setCurrentItem(ch)
                    break

    def _on_current_changed(self, current: QTreeWidgetItem | None, _prev) -> None:
        if self._suppress_signal or current is None:
            return
        kind = current.data(0, _KIND_ROLE)
        if kind == KIND_COLLECTIONS_HEADER:
            return
        target = SidebarViewTarget(
            kind=str(kind),
            collection_id=current.data(0, _COLL_ID_ROLE),
            root_id=current.data(0, _ROOT_ID_ROLE),
            folder_id=current.data(0, _FOLDER_ID_ROLE),
            folder_rel_path=str(current.data(0, _REL_PATH_ROLE) or ""),
            display_path=str(current.data(0, _FULL_PATH_ROLE) or ""),
        )
        self.viewSelected.emit(target)

    # -- Drag & Drop onto Collection nodes ---------------------------------
    @staticmethod
    def _has_droppable_payload(event) -> bool:
        mime = event.mimeData()
        return bool(mime.hasFormat(FILE_IDS_MIME) or mime.hasUrls())

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
        if item is not None and item.data(0, _KIND_ROLE) in (KIND_COLLECTION, KIND_COLLECTION_FOLDER):
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        if not self._has_droppable_payload(event):
            event.ignore()
            return
        item = self.itemAt(event.position().toPoint())
        if item is None or item.data(0, _KIND_ROLE) not in (KIND_COLLECTION, KIND_COLLECTION_FOLDER):
            event.ignore()
            return
        coll_id = item.data(0, _COLL_ID_ROLE)
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
                self.filesDroppedOnCollection.emit(int(coll_id), file_ids)
                event.acceptProposedAction()
                return
        if mime.hasUrls():
            paths = [u.toLocalFile() for u in mime.urls() if u.isLocalFile() and u.toLocalFile()]
            if paths:
                self.urlsDroppedOnCollection.emit(int(coll_id), paths)
                event.acceptProposedAction()

    # -- Context menu ------------------------------------------------------
    def _on_context_menu(self, pos) -> None:
        item = self.itemAt(pos)
        menu = QMenu(self)
        kind = item.data(0, _KIND_ROLE) if item is not None else KIND_COLLECTIONS_HEADER

        if kind in (KIND_COLLECTIONS_HEADER, KIND_COLLECTION):
            new_act = QAction("+ 新建合集…", menu)
            new_act.triggered.connect(lambda _c=False: self.createCollectionRequested.emit())
            menu.addAction(new_act)

        if kind == KIND_COLLECTION and item is not None:
            cid = int(item.data(0, _COLL_ID_ROLE))
            ren_act = QAction("重命名合集…", menu)
            ren_act.triggered.connect(lambda _c=False, x=cid: self.renameCollectionRequested.emit(x))
            menu.addAction(ren_act)

            del_act = QAction("删除合集（不删原文件）", menu)
            del_act.triggered.connect(lambda _c=False, x=cid: self.deleteCollectionRequested.emit(x))
            menu.addAction(del_act)

        if kind == KIND_COLLECTION_FOLDER and item is not None:
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

        if not menu.isEmpty():
            menu.exec(self.viewport().mapToGlobal(pos))


__all__ = [
    "KIND_COLLECTION",
    "KIND_COLLECTION_FOLDER",
    "KIND_COLLECTIONS_HEADER",
    "KIND_DUPLICATES",
    "KIND_FAVORITES",
    "KIND_RECENT",
    "CollectionsTree",
    "SidebarViewTarget",
]
