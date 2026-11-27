"""Preview and selective import dialog for REAPER Media Explorer shortcuts."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from audiosearch.browser.reaper_shortcuts import (
    ReaperImportReport,
    ReaperShortcutGroup,
    ReaperShortcutsPlan,
    apply_reaper_import_plan,
    build_reaper_import_plan,
    default_reaper_ini_path,
)
from audiosearch.library.store import LibraryStore

_GROUP_NAME_ROLE = Qt.ItemDataRole.UserRole + 1
_GROUP_MATCHABLE_ROLE = Qt.ItemDataRole.UserRole + 2

_STATUS_TEXT = {
    "matched_root": ("新增导入（根目录）", QColor("#7EC89B")),
    "matched_folder": ("新增导入", QColor("#7EC89B")),
    "unindexed_under_root": ("尚未扫描（跳过）", QColor("#8F8A80")),
    "outside_root": ("不在已扫目录中（跳过）", QColor("#8F8A80")),
    "missing": ("目录不存在（跳过）", QColor("#736E66")),
}


class ReaperShortcutsImportDialog(QDialog):
    """Inspect REAPER ``reaper.ini`` shortcuts and import them into sidebar collections."""

    def __init__(
        self,
        store: LibraryStore,
        *,
        ini_path: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("从 REAPER 导入快捷方式")
        self.resize(720, 560)
        self._store = store
        self._plan: ReaperShortcutsPlan | None = None
        self._import_report: ReaperImportReport | None = None
        self._updating_tree = False

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        title = QLabel("REAPER 配置文件（仅只读解析，不会修改 REAPER 设置或原音频）")
        title.setStyleSheet("background: transparent; color: #ECE9E2; font-weight: 600; font-size: 12px;")
        root.addWidget(title)

        ini_row = QHBoxLayout()
        ini_row.setSpacing(8)
        initial_ini = Path(ini_path) if ini_path is not None else default_reaper_ini_path()
        self.ini_path_edit = QLineEdit(str(initial_ini))
        self.browse_btn = QPushButton("浏览…")
        self.browse_btn.setProperty("variant", "secondary")
        self.browse_btn.clicked.connect(self._on_browse_ini)
        self.reload_btn = QPushButton("重新解析")
        self.reload_btn.setProperty("variant", "secondary")
        self.reload_btn.clicked.connect(self.reload_plan)
        ini_row.addWidget(self.ini_path_edit, 1)
        ini_row.addWidget(self.browse_btn, 0)
        ini_row.addWidget(self.reload_btn, 0)
        root.addLayout(ini_row)

        summary_card = QFrame()
        summary_card.setObjectName("panel")
        summary_card.setStyleSheet(
            "QFrame#panel { border: 1px solid #34322E; border-radius: 8px; padding: 8px; }"
        )
        summary_layout = QVBoxLayout(summary_card)
        summary_layout.setContentsMargins(10, 8, 10, 8)
        summary_layout.setSpacing(4)
        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet("background: transparent; color: #ECE9E2; font-size: 12px;")
        summary_layout.addWidget(self.summary_label)
        root.addWidget(summary_card)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["分组 / 快捷目录", "状态"])
        self.tree.setUniformRowHeights(True)
        self.tree.setAlternatingRowColors(False)
        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.itemChanged.connect(self._on_tree_item_changed)
        root.addWidget(self.tree, 1)

        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(8)
        self.select_all_btn = QPushButton("全选可用分组")
        self.select_all_btn.setProperty("variant", "secondary")
        self.select_all_btn.clicked.connect(lambda _c=False: self._set_all_groups_checked(True))
        self.clear_all_btn = QPushButton("取消全选")
        self.clear_all_btn.setProperty("variant", "secondary")
        self.clear_all_btn.clicked.connect(lambda _c=False: self._set_all_groups_checked(False))
        self.collapse_checkbox = QCheckBox("导入后默认折叠这些分组")
        self.collapse_checkbox.setChecked(True)
        ctrl_row.addWidget(self.select_all_btn)
        ctrl_row.addWidget(self.clear_all_btn)
        ctrl_row.addStretch(1)
        ctrl_row.addWidget(self.collapse_checkbox)
        root.addLayout(ctrl_row)

        footer = QHBoxLayout()
        footer.setSpacing(8)
        footer.addStretch(1)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setProperty("variant", "secondary")
        self.cancel_btn.clicked.connect(self.reject)
        self.import_btn = QPushButton("导入所选分组")
        self.import_btn.setProperty("variant", "primary")
        self.import_btn.clicked.connect(self._on_import)
        footer.addWidget(self.cancel_btn)
        footer.addWidget(self.import_btn)
        root.addLayout(footer)

        self.reload_plan()

    @property
    def plan(self) -> ReaperShortcutsPlan | None:
        return self._plan

    @property
    def import_report(self) -> ReaperImportReport | None:
        return self._import_report

    def collapse_after_import(self) -> bool:
        return bool(self.collapse_checkbox.isChecked())

    def selected_group_names(self) -> list[str]:
        names: list[str] = []
        for idx in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(idx)
            if item.checkState(0) == Qt.CheckState.Checked and item.data(0, _GROUP_MATCHABLE_ROLE):
                gname = str(item.data(0, _GROUP_NAME_ROLE) or "")
                if gname:
                    names.append(gname)
        return names

    def _current_ini_path(self) -> Path:
        raw = self.ini_path_edit.text().strip()
        return Path(raw) if raw else default_reaper_ini_path()

    @Slot()
    def _on_browse_ini(self) -> None:
        cur = self._current_ini_path()
        start_dir = cur.parent if cur.parent.is_dir() else Path.home()
        chosen, _ = QFileDialog.getOpenFileName(
            self,
            "选择 REAPER reaper.ini",
            str(start_dir),
            "reaper.ini (*.ini);;All Files (*.*)",
        )
        if chosen:
            self.ini_path_edit.setText(chosen)
            self.reload_plan()

    @Slot()
    def reload_plan(self) -> None:
        ini_path = self._current_ini_path()
        self._updating_tree = True
        try:
            self.tree.clear()
            if not ini_path.is_file():
                self._plan = ReaperShortcutsPlan(ini_path, 0, [], [], [])
                self.summary_label.setText(
                    f"未找到 REAPER 配置文件：{ini_path}\n"
                    "请点击上方「浏览…」手动选择你的 REAPER 目录下的 reaper.ini（便携版通常在安装根目录）。"
                )
                self.summary_label.setStyleSheet(
                    "background: transparent; color: #E38566; font-size: 12px;"
                )
                self._update_import_button()
                return

            plan = build_reaper_import_plan(ini_path, self._store)
            self._plan = plan

            if not plan.groups:
                skipped_note = len(plan.skipped_special) + len(plan.skipped_databases)
                self.summary_label.setText(
                    f"已解析 {ini_path.name}（共 {plan.total_raw_entries} 条原始项），未发现可导入的文件夹快捷方式"
                    + (f"（已跳过 {skipped_note} 条内置项/DB）。" if skipped_note else "。")
                )
                self.summary_label.setStyleSheet(
                    "background: transparent; color: #A5A096; font-size: 12px;"
                )
                self._update_import_button()
                return

            existing_total = sum(
                sum(1 for x in g.matched_items if x.already_in_collection) for g in plan.groups
            )
            skipped_builtin = len(plan.skipped_special) + len(plan.skipped_databases)
            parts = [
                f"发现 <b>{len(plan.groups)}</b> 个分组",
                f"命中已入库目录 <b>{plan.total_matched}</b> 个",
            ]
            if existing_total:
                parts.append(f"其中 <b>{existing_total}</b> 个已在合集内（导入时更新别名与顺序）")
            if plan.total_unmatched:
                parts.append(f"未入库/失效跳过 <b>{plan.total_unmatched}</b> 个")
            if skipped_builtin:
                parts.append(f"已忽略内置/DB 项 <b>{skipped_builtin}</b> 个")
            self.summary_label.setText(" · ".join(parts))
            self.summary_label.setStyleSheet(
                "background: transparent; color: #ECE9E2; font-size: 12px;"
            )

            for group in plan.groups:
                self._add_group_item(group, has_divider_groups=plan.has_divider_groups)
        finally:
            self._updating_tree = False
        self._update_import_button()

    def _add_group_item(
        self, group: ReaperShortcutGroup, *, has_divider_groups: bool
    ) -> None:
        matched_n = len(group.matched_items)
        unmatched_n = len(group.unmatched_items)
        status_parts: list[str] = []
        if matched_n:
            new_n = group.new_matched_count
            if group.already_exists and new_n < matched_n:
                status_parts.append(f"新增 {new_n} / 更新 {matched_n - new_n}")
            else:
                status_parts.append(f"可导入 {matched_n}")
        if unmatched_n:
            status_parts.append(f"跳过 {unmatched_n}")

        group_item = QTreeWidgetItem([f"{group.name}（{len(group.items)}）", " · ".join(status_parts)])
        group_item.setData(0, _GROUP_NAME_ROLE, group.name)
        group_item.setData(0, _GROUP_MATCHABLE_ROLE, matched_n > 0)

        if matched_n > 0:
            default_check = (not group.is_ungrouped) or (not has_divider_groups)
            group_item.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsUserCheckable
            )
            group_item.setCheckState(
                0, Qt.CheckState.Checked if default_check else Qt.CheckState.Unchecked
            )
            group_item.setForeground(1, QColor("#7EC89B"))
        else:
            group_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            group_item.setForeground(0, QColor("#736E66"))
            group_item.setForeground(1, QColor("#736E66"))

        for entry in group.items:
            if entry.already_in_collection and entry.match_status in ("matched_root", "matched_folder"):
                st_text = "已存在（更新别名/排序）"
                st_color = QColor("#A5A096")
            else:
                st_text, st_color = _STATUS_TEXT.get(
                    entry.match_status, ("未知状态（跳过）", QColor("#8F8A80"))
                )
            child = QTreeWidgetItem([entry.label, st_text])
            child.setToolTip(0, entry.raw_path)
            child.setToolTip(1, entry.raw_path)
            child.setForeground(1, st_color)
            if entry.match_status not in ("matched_root", "matched_folder"):
                child.setForeground(0, QColor("#736E66"))
            group_item.addChild(child)

        self.tree.addTopLevelItem(group_item)
        group_item.setExpanded(False)

    @Slot(QTreeWidgetItem, int)
    def _on_tree_item_changed(self, _item: QTreeWidgetItem, column: int) -> None:
        if self._updating_tree or column != 0:
            return
        self._update_import_button()

    def _set_all_groups_checked(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        self._updating_tree = True
        try:
            for idx in range(self.tree.topLevelItemCount()):
                item = self.tree.topLevelItem(idx)
                if item.data(0, _GROUP_MATCHABLE_ROLE):
                    item.setCheckState(0, state)
        finally:
            self._updating_tree = False
        self._update_import_button()

    def _update_import_button(self) -> None:
        if self._plan is None:
            self.import_btn.setEnabled(False)
            self.import_btn.setText("导入所选分组")
            return
        selected = set(self.selected_group_names())
        folder_total = sum(
            len(g.matched_items) for g in self._plan.groups if g.name in selected
        )
        has_any_matchable = any(bool(g.matched_items) for g in self._plan.groups)
        self.select_all_btn.setEnabled(has_any_matchable)
        self.clear_all_btn.setEnabled(bool(selected))
        self.import_btn.setEnabled(bool(selected) and folder_total > 0)
        if selected:
            self.import_btn.setText(f"导入所选 {len(selected)} 个分组（{folder_total} 个目录）")
        else:
            self.import_btn.setText("导入所选分组")

    @Slot()
    def _on_import(self) -> None:
        if self._plan is None:
            return
        selected = self.selected_group_names()
        if not selected:
            return
        self._import_report = apply_reaper_import_plan(
            self._store,
            self._plan,
            selected_group_names=selected,
        )
        self.accept()


__all__ = ["ReaperShortcutsImportDialog"]
