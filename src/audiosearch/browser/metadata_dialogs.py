"""R5a dialogs: UCS category picker, Single/Batch Metadata Editor, and Safe Write-Back + Undo."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from audiosearch.library.metadata_fields import (
    BATCH_OPS,
    EDITABLE_FIELDS,
    EDITABLE_FIELD_KEYS,
    FIELD_SPEC_MAP,
    apply_field_transform,
)
from audiosearch.library.metadata_writer import (
    WriteBackPreview,
    build_write_back_preview,
    execute_write_back,
    undo_write_batch,
)
from audiosearch.library.store import LibraryStore
from audiosearch.library.ucs import UcsEntry, parse_ucs_filename, search_ucs, validate_cat_id


class UcsPickerWidget(QWidget):
    """Searchable bilingual UCS v8.2 Category/SubCategory/CatID selector."""

    entrySelected = Signal(object)  # UcsEntry

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        title = QLabel("UCS v8.2 分类速查（点击填入）")
        title.setStyleSheet("font-weight: 600; color: #ECE9E2; font-size: 12px;")
        layout.addWidget(title)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜索 UCS：如 MAGSpel / 魔法 / 金币 / explosion")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.textChanged.connect(self._reload_list)
        layout.addWidget(self.search_edit)

        self.list_widget = QListWidget()
        self.list_widget.itemClicked.connect(self._on_item_activated)
        self.list_widget.itemDoubleClicked.connect(self._on_item_activated)
        layout.addWidget(self.list_widget, 1)

        self._reload_list("")

    def _reload_list(self, text: str) -> None:
        self.list_widget.clear()
        for entry in search_ucs(text, limit=120):
            item = QListWidgetItem(entry.display_label)
            item.setData(Qt.ItemDataRole.UserRole, entry)
            item.setToolTip(", ".join(entry.synonyms) if entry.synonyms else entry.display_label)
            self.list_widget.addItem(item)

    def _on_item_activated(self, item: QListWidgetItem | None) -> None:
        if item is None:
            return
        entry = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(entry, UcsEntry):
            self.entrySelected.emit(entry)


class MetadataEditorDialog(QDialog):
    """Unified single-file or multi-file batch metadata editor (local DB only, non-destructive)."""

    metadataSaved = Signal(list)  # list[int] affected file IDs

    def __init__(
        self,
        store: LibraryStore,
        file_ids: list[int],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.store = store
        self.file_ids = list(dict.fromkeys(int(fid) for fid in file_ids))
        self._is_batch = len(self.file_ids) > 1
        self._single_inputs: dict[str, QWidget] = {}
        self._single_initial: dict[str, str] = {}

        self.setWindowTitle(
            f"批量编辑元数据 ({len(self.file_ids)} 条)"
            if self._is_batch
            else "编辑元数据与 UCS 分类"
        )
        self.resize(920 if not self._is_batch else 940, 620)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        if self._is_batch:
            self._build_batch_ui(root)
        else:
            self._build_single_ui(root)

        footer = QHBoxLayout()
        footer.setSpacing(8)
        info = QLabel("修改先保存在本地工具库并标记「待写回」，不会直接改动磁盘源文件。")
        info.setStyleSheet("color: #A5A096; font-size: 12px;")
        footer.addWidget(info, 1)

        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setProperty("variant", "ghost")
        self.cancel_btn.clicked.connect(self.reject)

        self.save_btn = QPushButton("保存到工具库（不改文件）")
        self.save_btn.setProperty("variant", "primary")
        self.save_btn.clicked.connect(self._on_save)
        footer.addWidget(self.cancel_btn)
        footer.addWidget(self.save_btn)
        root.addLayout(footer)

    def _build_single_ui(self, root: QVBoxLayout) -> None:
        fid = self.file_ids[0]
        rec = self.store.get_file(fid)
        canonical = self.store.get_canonical_metadata(fid)

        header = QHBoxLayout()
        title_text = rec.filename if rec is not None else f"文件 #{fid}"
        self.header_label = QLabel(title_text)
        self.header_label.setStyleSheet("font-size: 14px; font-weight: 600; color: #ECE9E2;")
        header.addWidget(self.header_label, 1)

        self.parse_ucs_btn = QPushButton("从文件名解析 UCS")
        self.parse_ucs_btn.setProperty("variant", "secondary")
        self.parse_ucs_btn.clicked.connect(self._on_parse_single_ucs)
        header.addWidget(self.parse_ucs_btn)

        self.toggle_ucs_btn = QPushButton("UCS 分类表")
        self.toggle_ucs_btn.setCheckable(True)
        self.toggle_ucs_btn.setChecked(True)
        self.toggle_ucs_btn.setProperty("variant", "chip")
        header.addWidget(self.toggle_ucs_btn)
        root.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        form_frame = QFrame()
        form_layout = QFormLayout(form_frame)
        form_layout.setContentsMargins(0, 4, 8, 4)
        form_layout.setSpacing(8)

        for spec in EDITABLE_FIELDS:
            cur_val, _src, _dirty = canonical.get(spec.key, ("", "embedded", False))
            self._single_initial[spec.key] = cur_val
            if spec.multiline:
                w: QWidget = QPlainTextEdit()
                w.setPlainText(cur_val)
                w.setFixedHeight(68)
            else:
                w = QLineEdit(cur_val)
                w.setClearButtonEnabled(True)
            self._single_inputs[spec.key] = w
            lbl = QLabel(f"{spec.label_zh} ({spec.label_en})")
            lbl.setStyleSheet("color: #A5A096; font-size: 12px;")
            form_layout.addRow(lbl, w)

        self.cat_status_label = QLabel("")
        self.cat_status_label.setStyleSheet("color: #A5A096; font-size: 11px;")
        form_layout.addRow(QLabel("UCS 校验"), self.cat_status_label)

        cat_input = self._single_inputs["cat_id"]
        if isinstance(cat_input, QLineEdit):
            cat_input.textChanged.connect(self._refresh_cat_validation)
        self._refresh_cat_validation()

        self.ucs_picker = UcsPickerWidget()
        self.ucs_picker.entrySelected.connect(self._apply_ucs_entry_single)
        self.toggle_ucs_btn.toggled.connect(self.ucs_picker.setVisible)

        splitter.addWidget(form_frame)
        splitter.addWidget(self.ucs_picker)
        splitter.setSizes([540, 340])
        root.addWidget(splitter, 1)

    def _refresh_cat_validation(self) -> None:
        w = self._single_inputs.get("cat_id")
        if not isinstance(w, QLineEdit):
            return
        ok, _entry, msg = validate_cat_id(w.text())
        self.cat_status_label.setText(msg)
        self.cat_status_label.setStyleSheet(
            f"color: {'#A5A096' if ok else '#F0A387'}; font-size: 11px;"
        )

    def _apply_ucs_entry_single(self, entry: UcsEntry) -> None:
        for key, val in (
            ("cat_id", entry.cat_id),
            ("category", entry.category),
            ("subcategory", entry.subcategory),
        ):
            w = self._single_inputs.get(key)
            if isinstance(w, QLineEdit):
                w.setText(val)

    def _on_parse_single_ucs(self) -> None:
        rec = self.store.get_file(self.file_ids[0])
        if rec is None:
            return
        res = parse_ucs_filename(rec.filename)
        if not res.matched:
            self.cat_status_label.setText(res.note)
            self.cat_status_label.setStyleSheet("color: #F0A387; font-size: 11px;")
            return
        for key, val in res.fields.items():
            w = self._single_inputs.get(key)
            if isinstance(w, QLineEdit):
                w.setText(val)
            elif isinstance(w, QPlainTextEdit):
                w.setPlainText(val)
        self.cat_status_label.setText(res.note)
        self.cat_status_label.setStyleSheet("color: #A5A096; font-size: 11px;")

    # -- Batch mode --------------------------------------------------------
    def _build_batch_ui(self, root: QVBoxLayout) -> None:
        top = QHBoxLayout()
        top.setSpacing(8)
        lbl = QLabel(f"已选中 {len(self.file_ids):,} 条音效进行批量编辑")
        lbl.setStyleSheet("font-size: 14px; font-weight: 600; color: #ECE9E2;")
        top.addWidget(lbl, 1)

        self.batch_parse_ucs_btn = QPushButton("批量从文件名解析 UCS 并保存")
        self.batch_parse_ucs_btn.setProperty("variant", "secondary")
        self.batch_parse_ucs_btn.clicked.connect(self._on_batch_parse_ucs)
        top.addWidget(self.batch_parse_ucs_btn)
        root.addLayout(top)

        controls = QFrame()
        c_layout = QFormLayout(controls)
        c_layout.setContentsMargins(0, 4, 0, 4)
        c_layout.setSpacing(8)

        self.field_combo = QComboBox()
        for spec in EDITABLE_FIELDS:
            self.field_combo.addItem(f"{spec.label_zh} ({spec.label_en})", spec.key)
        c_layout.addRow("目标字段", self.field_combo)

        self.op_combo = QComboBox()
        for op_key, op_label in BATCH_OPS:
            self.op_combo.addItem(op_label, op_key)
        c_layout.addRow("编辑方式", self.op_combo)

        self.value_edit = QLineEdit()
        self.value_edit.setPlaceholderText("输入新值或追加/前置文本")
        c_layout.addRow("内容", self.value_edit)

        find_row = QHBoxLayout()
        find_row.setContentsMargins(0, 0, 0, 0)
        find_row.setSpacing(6)
        self.find_edit = QLineEdit()
        self.find_edit.setPlaceholderText("查找文本或正则表达式")
        self.replace_edit = QLineEdit()
        self.replace_edit.setPlaceholderText("替换为（正则支持 \\1 引用）")
        self.case_cb = QCheckBox("区分大小写")
        find_row.addWidget(self.find_edit, 1)
        find_row.addWidget(self.replace_edit, 1)
        find_row.addWidget(self.case_cb)
        self.find_row_widget = QWidget()
        self.find_row_widget.setLayout(find_row)
        c_layout.addRow("查找/替换", self.find_row_widget)

        ucs_row = QHBoxLayout()
        ucs_row.setContentsMargins(0, 0, 0, 0)
        ucs_row.setSpacing(6)
        self.batch_ucs_combo = QComboBox()
        self.batch_ucs_combo.setEditable(True)
        for entry in search_ucs("", limit=260):
            self.batch_ucs_combo.addItem(entry.display_label, entry)
        self.apply_ucs_batch_btn = QPushButton("应用所选 UCS 到全部选中文件")
        self.apply_ucs_batch_btn.setProperty("variant", "secondary")
        self.apply_ucs_batch_btn.clicked.connect(self._on_apply_ucs_batch)
        ucs_row.addWidget(self.batch_ucs_combo, 1)
        ucs_row.addWidget(self.apply_ucs_batch_btn)
        c_layout.addRow("快捷 UCS", ucs_row)
        root.addWidget(controls)

        self.batch_error_label = QLabel("")
        self.batch_error_label.setStyleSheet("color: #F0A387; font-size: 12px;")
        root.addWidget(self.batch_error_label)

        self.preview_table = QTableWidget(0, 3)
        self.preview_table.setHorizontalHeaderLabels(["文件名", "当前值", "修改后预览"])
        self.preview_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.preview_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.preview_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.preview_table.verticalHeader().setVisible(False)
        self.preview_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        root.addWidget(self.preview_table, 1)

        self._batch_cache: list[tuple[int, str, dict[str, str]]] = []
        recs = self.store.get_files_by_ids(self.file_ids[:50])
        for fid in self.file_ids[:50]:
            rec = recs.get(fid)
            if rec is None:
                continue
            canon = self.store.get_canonical_metadata(fid)
            self._batch_cache.append(
                (fid, rec.filename, {k: v for k, (v, _s, _d) in canon.items()})
            )

        self.field_combo.currentIndexChanged.connect(self._refresh_batch_preview)
        self.op_combo.currentIndexChanged.connect(self._refresh_batch_preview)
        self.value_edit.textChanged.connect(self._refresh_batch_preview)
        self.find_edit.textChanged.connect(self._refresh_batch_preview)
        self.replace_edit.textChanged.connect(self._refresh_batch_preview)
        self.case_cb.toggled.connect(self._refresh_batch_preview)
        self._refresh_batch_preview()

    def _refresh_batch_preview(self) -> None:
        op = str(self.op_combo.currentData() or "set")
        is_replace = op in ("replace", "regex_replace")
        is_clear = op == "clear"
        self.value_edit.setEnabled(not is_replace and not is_clear)
        self.find_row_widget.setVisible(is_replace)

        field_key = str(self.field_combo.currentData() or EDITABLE_FIELD_KEYS[0])
        val = self.value_edit.text()
        find = self.find_edit.text()
        repl = self.replace_edit.text()
        cs = self.case_cb.isChecked()

        self.preview_table.setRowCount(len(self._batch_cache))
        self.batch_error_label.setText("")
        for row_idx, (_fid, fname, cmap) in enumerate(self._batch_cache):
            cur = cmap.get(field_key, "")
            try:
                nxt = apply_field_transform(
                    cur, op, val, find=find, replace_with=repl, case_sensitive=cs
                )
            except ValueError as exc:
                self.batch_error_label.setText(str(exc))
                nxt = cur
            self.preview_table.setItem(row_idx, 0, QTableWidgetItem(fname))
            self.preview_table.setItem(row_idx, 1, QTableWidgetItem(cur or "—"))
            self.preview_table.setItem(row_idx, 2, QTableWidgetItem(nxt or "—"))

    def _on_apply_ucs_batch(self) -> None:
        idx = self.batch_ucs_combo.currentIndex()
        entry = self.batch_ucs_combo.itemData(idx) if idx >= 0 else None
        if not isinstance(entry, UcsEntry):
            matches = search_ucs(self.batch_ucs_combo.currentText(), limit=1)
            entry = matches[0] if matches else None
        if entry is None:
            QMessageBox.warning(self, "UCS 分类", "请先从下拉框选择有效 UCS 分类。")
            return
        for fid in self.file_ids:
            self.store.update_file_metadata(
                fid,
                {
                    "cat_id": entry.cat_id,
                    "category": entry.category,
                    "subcategory": entry.subcategory,
                },
                source="user",
            )
        self.metadataSaved.emit(list(self.file_ids))
        self.accept()

    def _on_batch_parse_ucs(self) -> None:
        recs = self.store.get_files_by_ids(self.file_ids)
        updated_ids: list[int] = []
        for fid in self.file_ids:
            rec = recs.get(fid)
            if rec is None:
                continue
            res = parse_ucs_filename(rec.filename)
            if res.matched and res.fields:
                self.store.update_file_metadata(fid, res.fields, source="user")
                updated_ids.append(fid)
        if not updated_ids:
            QMessageBox.information(self, "批量解析 UCS", "选中文件名中未检测到可识别的 UCS 规范前缀。")
            return
        self.metadataSaved.emit(updated_ids)
        self.accept()

    def _on_save(self) -> None:
        if not self._is_batch:
            updates: dict[str, str] = {}
            for key in EDITABLE_FIELD_KEYS:
                w = self._single_inputs[key]
                val = w.toPlainText().strip() if isinstance(w, QPlainTextEdit) else w.text().strip()
                if val != self._single_initial.get(key, ""):
                    updates[key] = val
            if updates:
                self.store.update_file_metadata(self.file_ids[0], updates, source="user")
                self.metadataSaved.emit([self.file_ids[0]])
            self.accept()
            return

        field_key = str(self.field_combo.currentData() or "description")
        op = str(self.op_combo.currentData() or "set")
        try:
            changed = self.store.batch_update_metadata(
                self.file_ids,
                field_key,
                op,
                self.value_edit.text(),
                find=self.find_edit.text(),
                replace_with=self.replace_edit.text(),
                case_sensitive=self.case_cb.isChecked(),
                source="user",
            )
        except ValueError as exc:
            self.batch_error_label.setText(str(exc))
            return
        if changed > 0:
            self.metadataSaved.emit(list(self.file_ids))
        self.accept()


class WriteBackDialog(QDialog):
    """Dry-run diff preview, background quadruple-safe write-back, and per-batch Undo."""

    libraryChanged = Signal()
    _progressSig = Signal(int, int, str)
    _writeDoneSig = Signal(object)
    _writeFailSig = Signal(str)
    _undoDoneSig = Signal(object)
    _undoFailSig = Signal(str)

    def __init__(
        self,
        store: LibraryStore,
        file_ids: list[int] | None = None,
        *,
        before_disk_write: Callable[[list[int]], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.store = store
        self.file_ids = list(dict.fromkeys(int(x) for x in file_ids)) if file_ids else None
        self._before_disk_write = before_disk_write
        self._worker_thread: threading.Thread | None = None
        self._preview: WriteBackPreview | None = None

        self.setWindowTitle("安全写回源文件与批次撤销")
        self.resize(880, 560)
        self._progressSig.connect(self._on_worker_progress)
        self._writeDoneSig.connect(self._on_write_done)
        self._writeFailSig.connect(self._on_write_failed)
        self._undoDoneSig.connect(self._on_undo_done)
        self._undoFailSig.connect(self._on_undo_failed)
        self._build_ui()
        self.refresh_all()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        safety_lbl = QLabel(
            "安全保证：音频数据永不修改 · 写前后 SHA-256 校验 · WAV 保留所有未知块 · "
            "同盘原子替换 · 写前备份至 .local/metadata-backups/ 支持整批撤销"
        )
        safety_lbl.setWordWrap(True)
        safety_lbl.setStyleSheet(
            "background-color: #272522; border: 1px solid #3D3A34; border-radius: 7px; "
            "padding: 8px 10px; color: #F0A387; font-size: 12px;"
        )
        root.addWidget(safety_lbl)

        self.tabs = QTabWidget()
        # Tab 1: Dry-run preview & execute
        preview_tab = QWidget()
        p_layout = QVBoxLayout(preview_tab)
        p_layout.setContentsMargins(10, 10, 10, 10)
        p_layout.setSpacing(8)

        self.preview_summary_label = QLabel("正在准备写回预览…")
        self.preview_summary_label.setStyleSheet("color: #ECE9E2; font-weight: 600;")
        p_layout.addWidget(self.preview_summary_label)

        self.preview_table = QTableWidget(0, 3)
        self.preview_table.setHorizontalHeaderLabels(["文件名", "字段变更预览 (改前 → 改后)", "状态"])
        self.preview_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.preview_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.preview_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.preview_table.verticalHeader().setVisible(False)
        self.preview_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        p_layout.addWidget(self.preview_table, 1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.discard_btn = QPushButton("放弃待写回修改")
        self.discard_btn.setProperty("variant", "ghost")
        self.discard_btn.clicked.connect(self._on_discard_dirty)

        self.execute_btn = QPushButton("执行安全写回")
        self.execute_btn.setProperty("variant", "primary")
        self.execute_btn.clicked.connect(self._on_execute_write)

        btn_row.addWidget(self.discard_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self.execute_btn)
        p_layout.addLayout(btn_row)
        self.tabs.addTab(preview_tab, "待写回预览")

        # Tab 2: Batches & Undo
        batch_tab = QWidget()
        b_layout = QVBoxLayout(batch_tab)
        b_layout.setContentsMargins(10, 10, 10, 10)
        b_layout.setSpacing(8)

        self.batches_table = QTableWidget(0, 4)
        self.batches_table.setHorizontalHeaderLabels(["批次 ID", "创建时间", "成功文件数", "状态"])
        self.batches_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.batches_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.batches_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.batches_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.batches_table.verticalHeader().setVisible(False)
        self.batches_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.batches_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.batches_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        b_layout.addWidget(self.batches_table, 1)

        undo_row = QHBoxLayout()
        self.undo_btn = QPushButton("撤销选中批次")
        self.undo_btn.setProperty("variant", "secondary")
        self.undo_btn.clicked.connect(self._on_undo_selected)
        undo_row.addStretch(1)
        undo_row.addWidget(self.undo_btn)
        b_layout.addLayout(undo_row)
        self.tabs.addTab(batch_tab, "写回批次与撤销")

        root.addWidget(self.tabs, 1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        root.addWidget(self.progress_bar)

        bottom = QHBoxLayout()
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #A5A096; font-size: 12px;")
        self.close_btn = QPushButton("关闭")
        self.close_btn.setProperty("variant", "ghost")
        self.close_btn.clicked.connect(self.accept)
        bottom.addWidget(self.status_label, 1)
        bottom.addWidget(self.close_btn)
        root.addLayout(bottom)

    def refresh_all(self) -> None:
        self._preview = build_write_back_preview(self.store, self.file_ids)
        items = self._preview.items
        skipped = self._preview.skipped
        self.preview_summary_label.setText(
            f"可写回 {len(items):,} 条 · 跳过 {len(skipped):,} 条"
        )
        self.preview_table.setRowCount(len(items) + len(skipped))
        row = 0
        for item in items:
            diff_text = "；".join(
                f"{d.label}: {d.before or '(空)'} → {d.after or '(空)'}" for d in item.diffs
            )
            self.preview_table.setItem(row, 0, QTableWidgetItem(item.filename))
            self.preview_table.setItem(row, 1, QTableWidgetItem(diff_text or "待写回"))
            self.preview_table.setItem(row, 2, QTableWidgetItem(f"就绪 ({item.ext.upper()})"))
            row += 1
        for sk in skipped:
            self.preview_table.setItem(row, 0, QTableWidgetItem(sk.filename))
            self.preview_table.setItem(row, 1, QTableWidgetItem(sk.path))
            self.preview_table.setItem(row, 2, QTableWidgetItem(f"跳过: {sk.reason}"))
            row += 1

        self.execute_btn.setEnabled(len(items) > 0)
        self.discard_btn.setEnabled(store_has_dirty := (self.store.dirty_file_count() > 0))
        _ = store_has_dirty

        batches = self.store.list_write_batches(limit=30)
        self.batches_table.setRowCount(len(batches))
        status_map = {"committed": "已写入", "reverted": "已撤销", "failed": "失败", "pending": "进行中"}
        for b_idx, b in enumerate(batches):
            id_item = QTableWidgetItem(b.batch_id)
            id_item.setData(Qt.ItemDataRole.UserRole, b)
            t_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(b.created_at))
            self.batches_table.setItem(b_idx, 0, id_item)
            self.batches_table.setItem(b_idx, 1, QTableWidgetItem(t_str))
            self.batches_table.setItem(b_idx, 2, QTableWidgetItem(f"{b.file_count:,}"))
            self.batches_table.setItem(b_idx, 3, QTableWidgetItem(status_map.get(b.status, b.status)))

    def _set_busy(self, busy: bool, total: int = 0) -> None:
        self.execute_btn.setEnabled(not busy and bool(self._preview and self._preview.items))
        self.discard_btn.setEnabled(not busy and self.store.dirty_file_count() > 0)
        self.undo_btn.setEnabled(not busy)
        self.progress_bar.setVisible(busy)
        if busy:
            self.progress_bar.setRange(0, max(1, total))
            self.progress_bar.setValue(0)

    def _on_discard_dirty(self) -> None:
        n = self.store.discard_dirty_metadata(self.file_ids)
        self.status_label.setText(f"已放弃 {n:,} 个文件的待写回修改（恢复库内原内嵌状态）")
        self.libraryChanged.emit()
        self.refresh_all()

    def _on_execute_write(self) -> None:
        if self._preview is None or not self._preview.items:
            return
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        target_ids = [item.file_id for item in self._preview.items]
        if self._before_disk_write is not None:
            self._before_disk_write(target_ids)
        self._set_busy(True, total=len(target_ids))
        self.status_label.setText("正在备份并安全写回源文件…")
        ids_copy = list(self.file_ids) if self.file_ids is not None else None

        def _work() -> None:
            try:
                res = execute_write_back(
                    self.store,
                    ids_copy,
                    on_progress=lambda i, t, fn: self._progressSig.emit(i, t, fn),
                )
                self._writeDoneSig.emit(res)
            except Exception as exc:
                self._writeFailSig.emit(str(exc))

        self._worker_thread = threading.Thread(target=_work, daemon=True)
        self._worker_thread.start()

    def _on_undo_selected(self) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        row = self.batches_table.currentRow()
        if row < 0 and self.batches_table.selectionModel() is not None:
            sel_rows = self.batches_table.selectionModel().selectedRows()
            if sel_rows:
                row = sel_rows[0].row()
        if row < 0:
            QMessageBox.information(self, "撤销批次", "请先在列表中选择一条已写入的批次。")
            return
        item = self.batches_table.item(row, 0)
        batch = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        if batch is None or batch.status == "reverted":
            QMessageBox.information(self, "撤销批次", "所选批次已撤销或不可撤销。")
            return
        if self._before_disk_write is not None:
            self._before_disk_write([])
        self._set_busy(True, total=max(1, batch.file_count))
        self.status_label.setText(f"正在撤销批次 {batch.batch_id}…")

        def _work() -> None:
            try:
                res = undo_write_batch(
                    self.store,
                    batch.batch_id,
                    on_progress=lambda i, t, fn: self._progressSig.emit(i, t, fn),
                )
                self._undoDoneSig.emit(res)
            except Exception as exc:
                self._undoFailSig.emit(str(exc))

        self._worker_thread = threading.Thread(target=_work, daemon=True)
        self._worker_thread.start()

    @Slot(int, int, str)
    def _on_worker_progress(self, idx: int, total: int, filename: str) -> None:
        self.progress_bar.setMaximum(max(1, total))
        self.progress_bar.setValue(idx)
        self.status_label.setText(f"处理中 ({idx}/{total})：{filename}")

    @Slot(object)
    def _on_write_done(self, res: Any) -> None:
        self._set_busy(False)
        msg = f"写回完成：成功 {len(res.succeeded):,} 条"
        if res.failed:
            msg += f"，失败 {len(res.failed):,} 条（{res.failed[0].message}）"
        if res.skipped:
            msg += f"，跳过 {len(res.skipped):,} 条"
        self.status_label.setText(msg)
        self.libraryChanged.emit()
        self.refresh_all()

    @Slot(str)
    def _on_write_failed(self, msg: str) -> None:
        self._set_busy(False)
        self.status_label.setText(f"写回失败：{msg}")

    @Slot(object)
    def _on_undo_done(self, res: Any) -> None:
        self._set_busy(False)
        msg = f"批次 {res.batch_id} 已撤销：恢复 {len(res.restored):,} 条"
        if res.failed:
            msg += f"，失败 {len(res.failed):,} 条"
        self.status_label.setText(msg)
        self.libraryChanged.emit()
        self.refresh_all()

    @Slot(str)
    def _on_undo_failed(self, msg: str) -> None:
        self._set_busy(False)
        self.status_label.setText(f"撤销失败：{msg}")


__all__ = [
    "MetadataEditorDialog",
    "UcsPickerWidget",
    "WriteBackDialog",
]
