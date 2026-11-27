"""R5b dialogs: Template Batch Rename (+ Undo), CSV Export/Import, and AI Metadata Enrichment."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from audiosearch.ai.metadata_enrich import AiMetadataEnricher, AiMetadataSuggestion
from audiosearch.library.batch_rename import (
    PLACEHOLDER_HELP,
    RENAME_PRESETS,
    build_rename_preview,
    execute_batch_rename,
    undo_rename_batch,
)
from audiosearch.library.metadata_csv import (
    CsvImportPreview,
    apply_metadata_csv_import,
    export_metadata_csv,
    preview_metadata_csv_import,
)
from audiosearch.library.metadata_fields import FIELD_SPEC_MAP
from audiosearch.library.store import LibraryStore


class BatchRenameDialog(QDialog):
    """Template-based batch rename with live preview, collision control, and per-batch Undo."""

    libraryChanged = Signal()

    def __init__(
        self,
        store: LibraryStore,
        file_ids: list[int],
        *,
        before_disk_rename: Callable[[list[int]], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.store = store
        self.file_ids = list(dict.fromkeys(int(fid) for fid in file_ids))
        self._before_disk_rename = before_disk_rename
        self._preview = None

        self.setWindowTitle(f"模板批量改名 ({len(self.file_ids):,} 条)")
        self.resize(940, 620)
        self._build_ui()
        self.refresh_all()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        self.tabs = QTabWidget()
        # Tab 1: Rename setup + live preview
        ren_tab = QWidget()
        r_layout = QVBoxLayout(ren_tab)
        r_layout.setContentsMargins(10, 10, 10, 10)
        r_layout.setSpacing(8)

        form = QFormLayout()
        form.setSpacing(8)

        self.preset_combo = QComboBox()
        for label, tpl in RENAME_PRESETS:
            self.preset_combo.addItem(f"{label}  ({tpl})", tpl)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        form.addRow("预设模板", self.preset_combo)

        self.template_edit = QLineEdit(RENAME_PRESETS[0][1])
        self.template_edit.setPlaceholderText("例如：{CatID}_{FXName}_{Designer}_{Library}_{Index:02d}")
        form.addRow("命名模板", self.template_edit)

        chips_row = QHBoxLayout()
        chips_row.setContentsMargins(0, 0, 0, 0)
        chips_row.setSpacing(4)
        for token, desc in PLACEHOLDER_HELP[:7]:
            btn = QPushButton(token)
            btn.setProperty("variant", "chip")
            btn.setProperty("compact", True)
            btn.setToolTip(f"插入 {desc}")
            btn.clicked.connect(lambda _c=False, t=token: self._insert_token(t))
            chips_row.addWidget(btn)
        for token, desc in PLACEHOLDER_HELP[8:12]:
            btn = QPushButton(token)
            btn.setProperty("variant", "chip")
            btn.setProperty("compact", True)
            btn.setToolTip(f"插入 {desc}")
            btn.clicked.connect(lambda _c=False, t=token: self._insert_token(t))
            chips_row.addWidget(btn)
        chips_row.addStretch(1)
        chips_w = QWidget()
        chips_w.setLayout(chips_row)
        form.addRow("快捷字段", chips_w)

        opts_row = QHBoxLayout()
        opts_row.setContentsMargins(0, 0, 0, 0)
        opts_row.setSpacing(10)

        self.space_combo = QComboBox()
        self.space_combo.addItem("空格处理：保留空格", "keep")
        self.space_combo.addItem("空格处理：转下划线 (_)", "underscore")
        self.space_combo.addItem("空格处理：驼峰拼接", "camel")

        self.collision_combo = QComboBox()
        self.collision_combo.addItem("重名策略：自动加序号 (_01)", "auto_suffix")
        self.collision_combo.addItem("重名策略：严格拦截冲突", "error")

        self.start_spin = QSpinBox()
        self.start_spin.setRange(0, 9999)
        self.start_spin.setValue(1)
        self.start_spin.setPrefix("起始序号 ")

        opts_row.addWidget(self.space_combo)
        opts_row.addWidget(self.collision_combo)
        opts_row.addWidget(self.start_spin)
        opts_row.addStretch(1)
        opts_w = QWidget()
        opts_w.setLayout(opts_row)
        form.addRow("规则选项", opts_w)
        r_layout.addLayout(form)

        self.summary_label = QLabel("")
        self.summary_label.setStyleSheet("color: #ECE9E2; font-weight: 600;")
        r_layout.addWidget(self.summary_label)

        self.preview_table = QTableWidget(0, 3)
        self.preview_table.setHorizontalHeaderLabels(["原文件名", "新文件名预览", "状态"])
        self.preview_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.preview_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.preview_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.preview_table.verticalHeader().setVisible(False)
        self.preview_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        r_layout.addWidget(self.preview_table, 1)

        ren_btns = QHBoxLayout()
        ren_btns.addStretch(1)
        self.execute_btn = QPushButton("执行批量改名")
        self.execute_btn.setProperty("variant", "primary")
        self.execute_btn.clicked.connect(self._on_execute)
        ren_btns.addWidget(self.execute_btn)
        r_layout.addLayout(ren_btns)
        self.tabs.addTab(ren_tab, "模板预览与改名")

        # Tab 2: Rename history & Undo
        hist_tab = QWidget()
        h_layout = QVBoxLayout(hist_tab)
        h_layout.setContentsMargins(10, 10, 10, 10)
        h_layout.setSpacing(8)

        self.batches_table = QTableWidget(0, 4)
        self.batches_table.setHorizontalHeaderLabels(["批次 ID", "时间", "文件数", "状态"])
        self.batches_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.batches_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.batches_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.batches_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.batches_table.verticalHeader().setVisible(False)
        self.batches_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.batches_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.batches_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        h_layout.addWidget(self.batches_table, 1)

        undo_row = QHBoxLayout()
        self.undo_btn = QPushButton("撤销选中改名批次")
        self.undo_btn.setProperty("variant", "secondary")
        self.undo_btn.clicked.connect(self._on_undo_selected)
        undo_row.addStretch(1)
        undo_row.addWidget(self.undo_btn)
        h_layout.addLayout(undo_row)
        self.tabs.addTab(hist_tab, "改名批次与撤销")

        root.addWidget(self.tabs, 1)

        footer = QHBoxLayout()
        self.status_label = QLabel("改名仅变更文件名并同步更新本地库记录，不重编码音频。")
        self.status_label.setStyleSheet("color: #A5A096; font-size: 12px;")
        self.close_btn = QPushButton("关闭")
        self.close_btn.setProperty("variant", "ghost")
        self.close_btn.clicked.connect(self.accept)
        footer.addWidget(self.status_label, 1)
        footer.addWidget(self.close_btn)
        root.addLayout(footer)

        self.template_edit.textChanged.connect(self.refresh_preview)
        self.space_combo.currentIndexChanged.connect(self.refresh_preview)
        self.collision_combo.currentIndexChanged.connect(self.refresh_preview)
        self.start_spin.valueChanged.connect(self.refresh_preview)

    def _insert_token(self, token: str) -> None:
        cur = self.template_edit.text()
        if cur and not cur.endswith(("_", "-", " ")):
            self.template_edit.setText(f"{cur}_{token}")
        else:
            self.template_edit.setText(f"{cur}{token}")

    def _on_preset_changed(self) -> None:
        tpl = str(self.preset_combo.currentData() or "")
        if tpl:
            self.template_edit.setText(tpl)

    def refresh_all(self) -> None:
        self.refresh_preview()
        self.refresh_batches()

    def refresh_preview(self) -> None:
        self._preview = build_rename_preview(
            self.store,
            self.file_ids,
            self.template_edit.text(),
            space_mode=str(self.space_combo.currentData() or "keep"),
            collision_policy=str(self.collision_combo.currentData() or "auto_suffix"),
            start_index=int(self.start_spin.value()),
        )
        p = self._preview
        self.summary_label.setText(
            f"可改名 {p.ready_count:,} 条 · 冲突 {p.conflict_count:,} 条 · 跳过/不变 {p.skipped_count:,} 条"
        )
        self.preview_table.setRowCount(len(p.items))
        status_zh = {
            "ready": "可改名",
            "unchanged": "无需改名",
            "conflict": "冲突",
            "skipped": "跳过",
        }
        for row_idx, item in enumerate(p.items):
            st_txt = status_zh.get(item.status, item.status)
            if item.reason:
                st_txt = f"{st_txt}: {item.reason}"
            self.preview_table.setItem(row_idx, 0, QTableWidgetItem(item.old_filename))
            self.preview_table.setItem(row_idx, 1, QTableWidgetItem(item.new_filename))
            self.preview_table.setItem(row_idx, 2, QTableWidgetItem(st_txt))
        self.execute_btn.setEnabled(p.ready_count > 0 and p.conflict_count == 0)

    def refresh_batches(self) -> None:
        batches = self.store.list_rename_batches(limit=30)
        self.batches_table.setRowCount(len(batches))
        status_map = {"committed": "已改名", "reverted": "已撤销", "failed": "失败", "pending": "进行中"}
        for idx, b in enumerate(batches):
            item = QTableWidgetItem(b.batch_id)
            item.setData(Qt.ItemDataRole.UserRole, b)
            t_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(b.created_at))
            self.batches_table.setItem(idx, 0, item)
            self.batches_table.setItem(idx, 1, QTableWidgetItem(t_str))
            self.batches_table.setItem(idx, 2, QTableWidgetItem(f"{b.file_count:,}"))
            self.batches_table.setItem(idx, 3, QTableWidgetItem(status_map.get(b.status, b.status)))

    def _on_execute(self) -> None:
        if self._preview is None or self._preview.ready_count == 0:
            return
        if self._before_disk_rename is not None:
            self._before_disk_rename([x.file_id for x in self._preview.items if x.status == "ready"])
        res = execute_batch_rename(
            self.store,
            self.file_ids,
            self.template_edit.text(),
            space_mode=str(self.space_combo.currentData() or "keep"),
            collision_policy=str(self.collision_combo.currentData() or "auto_suffix"),
            start_index=int(self.start_spin.value()),
        )
        if res.status == "committed":
            self.status_label.setText(f"已安全重命名 {len(res.renamed):,} 个文件（批次 {res.batch_id}，支持撤销）")
            self.libraryChanged.emit()
            self.refresh_all()
        else:
            msg = res.failed[0].message if res.failed else "未执行改名"
            self.status_label.setText(f"批量改名未完成：{msg}")
            self.refresh_all()

    def _on_undo_selected(self) -> None:
        row = self.batches_table.currentRow()
        if row < 0 and self.batches_table.selectionModel() is not None:
            sel = self.batches_table.selectionModel().selectedRows()
            if sel:
                row = sel[0].row()
        if row < 0:
            self.status_label.setText("请先在「改名批次与撤销」中选中一条批次。")
            return
        item = self.batches_table.item(row, 0)
        batch = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        if batch is None or batch.status == "reverted":
            self.status_label.setText("所选改名批次已撤销或不可撤销。")
            return
        if self._before_disk_rename is not None:
            self._before_disk_rename([])
        try:
            res = undo_rename_batch(self.store, batch.batch_id)
        except Exception as exc:
            self.status_label.setText(f"撤销失败：{exc}")
            return
        self.status_label.setText(f"已撤销批次 {res.batch_id}：恢复 {len(res.restored):,} 个原文件名")
        self.libraryChanged.emit()
        self.refresh_all()


class CsvMetadataDialog(QDialog):
    """Export library/selected metadata to CSV and preview/import edits back into SQLite."""

    libraryChanged = Signal()

    def __init__(
        self,
        store: LibraryStore,
        selected_file_ids: list[int],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.store = store
        self.selected_ids = list(dict.fromkeys(int(fid) for fid in selected_file_ids))
        self._import_path: Path | None = None
        self._import_preview: CsvImportPreview | None = None

        self.setWindowTitle("CSV 元数据导入 / 导出")
        self.resize(900, 580)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        self.tabs = QTabWidget()

        # Tab 1: Export
        exp_tab = QWidget()
        e_layout = QVBoxLayout(exp_tab)
        e_layout.setContentsMargins(12, 12, 12, 12)
        e_layout.setSpacing(10)

        exp_info = QLabel(
            "导出带 UTF-8 BOM 的 CSV 表格（Excel 直接打开不乱码），含 file_id / 相对路径 / 9 项标准元数据 / 收藏 / 本地备注。"
        )
        exp_info.setWordWrap(True)
        exp_info.setStyleSheet("color: #A5A096; font-size: 12px;")
        e_layout.addWidget(exp_info)

        scope_row = QHBoxLayout()
        self.export_scope_combo = QComboBox()
        if self.selected_ids:
            self.export_scope_combo.addItem(f"仅导出当前选中 ({len(self.selected_ids):,} 条)", "selected")
        self.export_scope_combo.addItem(f"导出全库全部素材 ({self.store.total_file_count():,} 条)", "all")
        self.export_btn = QPushButton("选择位置并导出 CSV…")
        self.export_btn.setProperty("variant", "primary")
        self.export_btn.clicked.connect(self._on_export_clicked)
        scope_row.addWidget(self.export_scope_combo, 1)
        scope_row.addWidget(self.export_btn)
        e_layout.addLayout(scope_row)

        self.export_result_label = QLabel("")
        self.export_result_label.setWordWrap(True)
        self.export_result_label.setStyleSheet("color: #ECE9E2; font-size: 12px;")
        e_layout.addWidget(self.export_result_label)
        e_layout.addStretch(1)
        self.tabs.addTab(exp_tab, "导出 CSV")

        # Tab 2: Import
        imp_tab = QWidget()
        i_layout = QVBoxLayout(imp_tab)
        i_layout.setContentsMargins(12, 12, 12, 12)
        i_layout.setSpacing(8)

        pick_row = QHBoxLayout()
        self.import_path_edit = QLineEdit()
        self.import_path_edit.setReadOnly(True)
        self.import_path_edit.setPlaceholderText("请选择待导入的 CSV 文件（支持 UTF-8 / GBK）")
        self.pick_csv_btn = QPushButton("浏览 CSV…")
        self.pick_csv_btn.setProperty("variant", "secondary")
        self.pick_csv_btn.clicked.connect(self._on_pick_import_csv)
        self.overwrite_empty_cb = QCheckBox("允许空单元格覆盖清空已有字段")
        self.overwrite_empty_cb.toggled.connect(self._refresh_import_preview)
        pick_row.addWidget(self.import_path_edit, 1)
        pick_row.addWidget(self.pick_csv_btn)
        pick_row.addWidget(self.overwrite_empty_cb)
        i_layout.addLayout(pick_row)

        self.import_summary_label = QLabel("请先选择 CSV 文件，系统会自动按 file_id / 相对路径 / 文件名匹配并预览差异。")
        self.import_summary_label.setWordWrap(True)
        self.import_summary_label.setStyleSheet("color: #ECE9E2; font-size: 12px;")
        i_layout.addWidget(self.import_summary_label)

        self.import_table = QTableWidget(0, 3)
        self.import_table.setHorizontalHeaderLabels(["行号 / 文件", "待导入变更 (改前 → 改后)", "状态"])
        self.import_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.import_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.import_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.import_table.verticalHeader().setVisible(False)
        self.import_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        i_layout.addWidget(self.import_table, 1)

        imp_btns = QHBoxLayout()
        imp_note = QLabel("导回后仅保存到工具库并标记「待写回」，不会直接改动音频源文件。")
        imp_note.setStyleSheet("color: #A5A096; font-size: 12px;")
        self.apply_import_btn = QPushButton("导入到工具库（不改文件）")
        self.apply_import_btn.setProperty("variant", "primary")
        self.apply_import_btn.setEnabled(False)
        self.apply_import_btn.clicked.connect(self._on_apply_import)
        imp_btns.addWidget(imp_note, 1)
        imp_btns.addWidget(self.apply_import_btn)
        i_layout.addLayout(imp_btns)
        self.tabs.addTab(imp_tab, "导入 CSV")

        root.addWidget(self.tabs, 1)

        footer = QHBoxLayout()
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #A5A096; font-size: 12px;")
        self.close_btn = QPushButton("关闭")
        self.close_btn.setProperty("variant", "ghost")
        self.close_btn.clicked.connect(self.accept)
        footer.addWidget(self.status_label, 1)
        footer.addWidget(self.close_btn)
        root.addLayout(footer)

    def _on_export_clicked(self) -> None:
        chosen, _ = QFileDialog.getSaveFileName(
            self,
            "导出元数据 CSV",
            "audiosearch_metadata.csv",
            "CSV 文件 (*.csv)",
        )
        if not chosen:
            return
        scope = str(self.export_scope_combo.currentData() or "all")
        fids = self.selected_ids if scope == "selected" and self.selected_ids else None
        res = export_metadata_csv(self.store, Path(chosen), fids)
        self.export_result_label.setText(f"导出完成：共写入 {res.row_count:,} 行 → {res.path}")
        self.status_label.setText(f"已导出 {res.row_count:,} 条到 {res.path.name}")

    def _on_pick_import_csv(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(self, "选择元数据 CSV", "", "CSV 文件 (*.csv);;所有文件 (*.*)")
        if not chosen:
            return
        self.load_import_file(Path(chosen))

    def load_import_file(self, path: Path) -> None:
        self._import_path = path
        self.import_path_edit.setText(str(path))
        self._refresh_import_preview()

    def _refresh_import_preview(self) -> None:
        if self._import_path is None or not self._import_path.exists():
            self._import_preview = None
            self.apply_import_btn.setEnabled(False)
            return
        try:
            prev = preview_metadata_csv_import(
                self.store,
                self._import_path,
                overwrite_empty=self.overwrite_empty_cb.isChecked(),
            )
        except Exception as exc:
            self.import_summary_label.setText(f"CSV 解析失败：{exc}")
            self.apply_import_btn.setEnabled(False)
            return

        self._import_preview = prev
        self.import_summary_label.setText(
            f"共 {prev.total_rows:,} 行 · 有变更 {len(prev.matched):,} 条 · 跳过 {len(prev.skipped):,} 条"
        )
        self.import_table.setRowCount(len(prev.matched) + len(prev.skipped))
        row = 0
        for item in prev.matched:
            diff_txt = "；".join(
                f"{c.label}: {c.before or '(空)'} → {c.after or '(空)'}" for c in item.changes
            )
            self.import_table.setItem(row, 0, QTableWidgetItem(f"#{item.row_num} {item.filename}"))
            self.import_table.setItem(row, 1, QTableWidgetItem(diff_txt))
            self.import_table.setItem(row, 2, QTableWidgetItem("待导入"))
            row += 1
        for sk in prev.skipped:
            self.import_table.setItem(row, 0, QTableWidgetItem(f"#{sk.row_num} {sk.identifier}"))
            self.import_table.setItem(row, 1, QTableWidgetItem(sk.reason))
            self.import_table.setItem(row, 2, QTableWidgetItem("跳过"))
            row += 1
        self.apply_import_btn.setEnabled(len(prev.matched) > 0)

    def _on_apply_import(self) -> None:
        if self._import_path is None:
            return
        res = apply_metadata_csv_import(
            self.store,
            self._import_path,
            overwrite_empty=self.overwrite_empty_cb.isChecked(),
        )
        self.status_label.setText(
            f"CSV 导入完成：更新 {res.updated_files:,} 个文件（共 {res.updated_fields:,} 项字段，仅存工具库）"
        )
        self.libraryChanged.emit()
        self._refresh_import_preview()


class AiEnrichDialog(QDialog):
    """CLAP2 zero-shot UCS classification, description generation & CN/EN translation."""

    metadataSaved = Signal(list)  # list[int]
    _progressSig = Signal(int, int, str)

    def __init__(
        self,
        store: LibraryStore,
        file_ids: list[int],
        *,
        run_ai_job: Callable[[Callable[[], Any], Callable[[Any], None], Callable[[str], None]], None],
        ai_index: Any | None,
        ai_engine_provider: Callable[[], Any | None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.store = store
        self.file_ids = list(dict.fromkeys(int(fid) for fid in file_ids))
        self._run_ai_job = run_ai_job
        self._ai_index = ai_index
        self._ai_engine_provider = ai_engine_provider
        self._suggestions: list[AiMetadataSuggestion] = []
        self._row_checks: list[QCheckBox] = []
        self._row_ucs_combos: list[QComboBox] = []

        self.setWindowTitle(f"AI 元数据自动补全 ({len(self.file_ids):,} 条)")
        self.resize(980, 620)
        self._progressSig.connect(self._on_progress)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        ctrl = QHBoxLayout()
        ctrl.setSpacing(10)
        self.fill_ucs_cb = QCheckBox("建议 UCS 分类 (CLAP2)")
        self.fill_ucs_cb.setChecked(True)
        self.fill_desc_cb = QCheckBox("生成描述与关键词")
        self.fill_desc_cb.setChecked(True)
        self.only_empty_cb = QCheckBox("仅补全空字段")
        self.only_empty_cb.setChecked(True)

        self.translate_combo = QComboBox()
        self.translate_combo.addItem("语言：中英合并", "none")
        self.translate_combo.addItem("语言：翻译为中文", "zh")
        self.translate_combo.addItem("语言：翻译为英文", "en")

        self.analyze_btn = QPushButton("开始 AI 分析")
        self.analyze_btn.setProperty("variant", "primary")
        self.analyze_btn.clicked.connect(self.start_analysis)

        ctrl.addWidget(self.fill_ucs_cb)
        ctrl.addWidget(self.fill_desc_cb)
        ctrl.addWidget(self.only_empty_cb)
        ctrl.addWidget(self.translate_combo)
        ctrl.addStretch(1)
        ctrl.addWidget(self.analyze_btn)
        root.addLayout(ctrl)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        root.addWidget(self.progress_bar)

        self.results_table = QTableWidget(0, 6)
        self.results_table.setHorizontalHeaderLabels(
            ["采纳", "文件名", "推荐 UCS (可选 Top-3)", "建议描述", "建议关键词", "依据 / 置信度"]
        )
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.results_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.results_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self.results_table.verticalHeader().setVisible(False)
        root.addWidget(self.results_table, 1)

        footer = QHBoxLayout()
        self.status_label = QLabel("点击「开始 AI 分析」生成建议，确认后再采纳到工具库（不会改盘）。")
        self.status_label.setStyleSheet("color: #A5A096; font-size: 12px;")

        self.close_btn = QPushButton("关闭")
        self.close_btn.setProperty("variant", "ghost")
        self.close_btn.clicked.connect(self.reject)

        self.apply_btn = QPushButton("采纳勾选建议到工具库（不改文件）")
        self.apply_btn.setProperty("variant", "primary")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self._on_apply)

        footer.addWidget(self.status_label, 1)
        footer.addWidget(self.close_btn)
        footer.addWidget(self.apply_btn)
        root.addLayout(footer)

    @Slot()
    def start_analysis(self) -> None:
        self.analyze_btn.setEnabled(False)
        self.apply_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, max(1, len(self.file_ids)))
        self.progress_bar.setValue(0)
        self.status_label.setText("正在运行 AI 语义/声波分析…")

        fill_ucs = self.fill_ucs_cb.isChecked()
        fill_desc = self.fill_desc_cb.isChecked()
        only_empty = self.only_empty_cb.isChecked()
        translate_mode = str(self.translate_combo.currentData() or "none")
        engine = self._ai_engine_provider()
        enricher = AiMetadataEnricher(self.store, self._ai_index, engine)
        ids = list(self.file_ids)

        def _work() -> list[AiMetadataSuggestion]:
            return enricher.enrich_files(
                ids,
                fill_ucs=fill_ucs,
                fill_desc=fill_desc,
                translate_mode=translate_mode,
                only_empty=only_empty,
                on_progress=lambda i, t, fn: self._progressSig.emit(i, t, fn),
            )

        self._run_ai_job(_work, self._on_analysis_done, self._on_analysis_failed)

    @Slot(int, int, str)
    def _on_progress(self, idx: int, total: int, filename: str) -> None:
        self.progress_bar.setMaximum(max(1, total))
        self.progress_bar.setValue(idx)
        self.status_label.setText(f"分析中 ({idx}/{total})：{filename}")

    def _on_analysis_done(self, suggestions: Any) -> None:
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self._suggestions = list(suggestions or [])
        self._row_checks.clear()
        self._row_ucs_combos.clear()

        self.results_table.setRowCount(len(self._suggestions))
        has_any_change = False
        for row_idx, sug in enumerate(self._suggestions):
            cb = QCheckBox()
            cb.setChecked(bool(sug.suggested_fields or sug.top_ucs))
            self._row_checks.append(cb)
            self.results_table.setCellWidget(row_idx, 0, cb)

            self.results_table.setItem(row_idx, 1, QTableWidgetItem(sug.filename))

            ucs_combo = QComboBox()
            for cand in sug.top_ucs:
                ucs_combo.addItem(
                    f"{cand.entry.cat_id} · {cand.entry.category_zh}/{cand.entry.subcategory_zh} ({cand.score:.2f})",
                    cand.entry,
                )
            if ucs_combo.count() == 0:
                ucs_combo.addItem("（无推荐 UCS）", None)
                ucs_combo.setEnabled(False)
            self._row_ucs_combos.append(ucs_combo)
            self.results_table.setCellWidget(row_idx, 2, ucs_combo)

            desc_val = sug.suggested_fields.get("description", sug.current_fields.get("description", ""))
            kw_val = sug.suggested_fields.get("keywords", sug.current_fields.get("keywords", ""))
            self.results_table.setItem(row_idx, 3, QTableWidgetItem(desc_val or "—"))
            self.results_table.setItem(row_idx, 4, QTableWidgetItem(kw_val or "—"))
            self.results_table.setItem(row_idx, 5, QTableWidgetItem(f"{sug.confidence:.2f} · {sug.note}"))
            if sug.suggested_fields or sug.top_ucs:
                has_any_change = True

        self.apply_btn.setEnabled(has_any_change)
        self.status_label.setText(
            f"分析完成：{len(self._suggestions):,} 条（请勾选后采纳到工具库，可随时编辑或写回）"
        )

    def _on_analysis_failed(self, msg: str) -> None:
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.status_label.setText(f"AI 分析异常：{msg}")

    def _on_apply(self) -> None:
        applied_ids: list[int] = []
        only_empty = self.only_empty_cb.isChecked()
        fill_ucs = self.fill_ucs_cb.isChecked()

        for row_idx, sug in enumerate(self._suggestions):
            if row_idx >= len(self._row_checks) or not self._row_checks[row_idx].isChecked():
                continue
            updates = dict(sug.suggested_fields)
            # Respect user's selected UCS candidate from Top-3 dropdown
            if fill_ucs and row_idx < len(self._row_ucs_combos):
                chosen_entry = self._row_ucs_combos[row_idx].currentData()
                if chosen_entry is not None:
                    for k, v in (
                        ("cat_id", chosen_entry.cat_id),
                        ("category", chosen_entry.category),
                        ("subcategory", chosen_entry.subcategory),
                    ):
                        if not only_empty or not sug.current_fields.get(k, "").strip():
                            updates[k] = v

            # Read any direct edits in Desc/Keywords cells
            desc_item = self.results_table.item(row_idx, 3)
            kw_item = self.results_table.item(row_idx, 4)
            if desc_item is not None:
                d_txt = desc_item.text().strip()
                if d_txt and d_txt != "—":
                    if not only_empty or not sug.current_fields.get("description", "").strip() or "description" in updates:
                        updates["description"] = d_txt
            if kw_item is not None:
                k_txt = kw_item.text().strip()
                if k_txt and k_txt != "—":
                    if not only_empty or not sug.current_fields.get("keywords", "").strip() or "keywords" in updates:
                        updates["keywords"] = k_txt

            if updates:
                self.store.update_file_metadata(sug.file_id, updates, source="ai")
                applied_ids.append(sug.file_id)

        if applied_ids:
            self.metadataSaved.emit(applied_ids)
        self.accept()


__all__ = [
    "AiEnrichDialog",
    "BatchRenameDialog",
    "CsvMetadataDialog",
]
