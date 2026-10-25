"""
Tag Hierarchy Dialog
====================
WLPG-authentic 'All of your tags' dialog matching media_1792290433097_6cb8c319.png.
Displays:
- Title bar: 'Photo Gallery'
- Header: 'All of your tags' (11pt Segoe UI blue)
- Subtitle: 'Select tags to add to your search.'
- Clean white tree view containing 'Descriptive tags' root and hierarchical tags with tag icons
- Multi-selection support (Ctrl+click, Shift+click)
- 'Add' and 'Cancel' buttons at bottom-right
"""

from __future__ import annotations

from typing import Any, Sequence

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.ui.ribbon_toolbar import make_wlpg_icon


class TagHierarchyDialog(QDialog):
    """Dialog allowing the user to browse all descriptive tags and add selected tags to search filter."""

    tags_added = Signal(list)  # list[int]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Photo Gallery")
        self.setFixedSize(320, 480)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setStyleSheet(
            """
            QDialog {
                background-color: #FFFFFF;
            }
            """
        )
        self._tag_items: dict[int, QTreeWidgetItem] = {}

        main_vbox = QVBoxLayout(self)
        main_vbox.setContentsMargins(14, 14, 14, 12)
        main_vbox.setSpacing(6)

        # 1. Header: "All of your tags"
        lbl_header = QLabel("All of your tags", self)
        lbl_header.setStyleSheet(
            "color: #1A5296; font-family: 'Segoe UI'; font-size: 11pt; font-weight: normal;"
        )
        main_vbox.addWidget(lbl_header)

        # 2. Subtitle: "Select tags to add to your search."
        lbl_sub = QLabel("Select tags to add to your search.", self)
        lbl_sub.setStyleSheet(
            "color: #111111; font-family: 'Segoe UI'; font-size: 9pt; margin-bottom: 4px;"
        )
        main_vbox.addWidget(lbl_sub)

        # 3. Tree view of tags
        self.tree = QTreeWidget(self)
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(16)
        self.tree.setIconSize(QSize(16, 16))
        self.tree.setAnimated(True)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.setStyleSheet(
            """
            QTreeWidget {
                background: #FFFFFF;
                border: 1px solid #C4D5E6;
                font-family: 'Segoe UI';
                font-size: 9pt;
                color: #1E395B;
                outline: none;
            }
            QTreeWidget::item {
                padding: 2px 4px;
                border: 1px solid transparent;
                border-radius: 2px;
            }
            QTreeWidget::item:hover {
                background: #F4F9FE;
                border: 1px solid #B8D6F5;
            }
            QTreeWidget::item:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #DCEBFC, stop:1 #C1DCFA);
                border: 1px solid #84ACDD;
                color: #1E395B;
            }
            """
        )
        main_vbox.addWidget(self.tree, 1)

        # 4. Bottom Button Row: [Add] [Cancel]
        btn_box = QHBoxLayout()
        btn_box.setContentsMargins(0, 6, 0, 0)
        btn_box.setSpacing(8)
        btn_box.addStretch(1)

        btn_style = """
            QPushButton {
                font-family: 'Segoe UI';
                font-size: 9pt;
                min-width: 75px;
                height: 23px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F2F4F7, stop:1 #E1E6EB);
                border: 1px solid #8E9BA8;
                border-radius: 2px;
                color: #111111;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #EAF2FB, stop:1 #D2E4F7);
                border: 1px solid #3C7FB1;
            }
            QPushButton:pressed {
                background: #CBE0F5;
                border: 1px solid #2C628B;
            }
            QPushButton:default {
                border: 1px solid #3380CC;
            }
        """

        self.btn_add = QPushButton("Add", self)
        self.btn_add.setDefault(True)
        self.btn_add.setStyleSheet(btn_style)
        self.btn_add.clicked.connect(self._on_add_clicked)
        btn_box.addWidget(self.btn_add)

        self.btn_cancel = QPushButton("Cancel", self)
        self.btn_cancel.setStyleSheet(btn_style)
        self.btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(self.btn_cancel)

        main_vbox.addLayout(btn_box)

    def populate(self, tag_rows: Sequence[dict[str, Any]]) -> None:
        """Populates the tree with tags hierarchy under 'Descriptive tags' root."""
        self.tree.clear()
        self._tag_items.clear()

        tag_icon = make_wlpg_icon("tag", 16)
        italic_font = QFont("Segoe UI", 9)
        italic_font.setItalic(True)

        root = QTreeWidgetItem(self.tree, ["Descriptive tags"])
        root.setFont(0, italic_font)
        root.setIcon(0, tag_icon)
        root.setExpanded(True)

        # First pass: instantiate nodes
        for t in tag_rows:
            tid = int(t["id"])
            tname = str(t["name"])
            item = QTreeWidgetItem([tname])
            item.setIcon(0, tag_icon)
            item.setData(0, Qt.UserRole, tid)
            self._tag_items[tid] = item

        # Second pass: attach hierarchy
        for t in tag_rows:
            tid = int(t["id"])
            pid = t.get("parent_id")
            item = self._tag_items[tid]
            if pid is not None and int(pid) in self._tag_items:
                self._tag_items[int(pid)].addChild(item)
            else:
                root.addChild(item)

        root.setExpanded(True)
        # Expand all direct children of root
        for i in range(root.childCount()):
            root.child(i).setExpanded(True)

    def selected_tag_ids(self) -> list[int]:
        """Returns list of selected tag IDs."""
        ids: list[int] = []
        for item in self.tree.selectedItems():
            tid = item.data(0, Qt.UserRole)
            if tid is not None:
                ids.append(int(tid))
        return ids

    def _on_add_clicked(self) -> None:
        sel = self.selected_tag_ids()
        if sel:
            self.tags_added.emit(sel)
        self.accept()
