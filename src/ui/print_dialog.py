"""
Print Pictures Dialog matching Windows / Windows Live Photo Gallery.
Provides a fully-featured photo printing wizard with live interactive preview,
multiple layout templates, printer selection, paper size & type, quality dpi,
copies, and aspect ratio frame fitting.
"""

from __future__ import annotations

import io
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QIcon,
    QImage,
    QKeySequence,
    QPageLayout,
    QPageSize,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QShortcut,
)
from PySide6.QtPrintSupport import QPrintDialog, QPrinter, QPrinterInfo
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import src.ui.icon_provider as icon_provider


class _PrintPreviewCanvas(QWidget):
    """Draws the paper sheet with drop shadow and placed photo frames according to the active layout."""

    def __init__(self, dialog: PrintPicturesDialog, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.dialog = dialog
        self.setMinimumSize(320, 260)
        self.setStyleSheet("background: transparent;")

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        # Available preview bounds
        w, h = self.width(), self.height()
        if w < 40 or h < 40:
            return

        # Determine paper aspect ratio
        paper_w, paper_h = self.dialog.get_current_paper_aspect()
        paper_ratio = paper_w / float(paper_h)

        max_canvas_w = w - 40
        max_canvas_h = h - 30

        if max_canvas_w / float(max_canvas_h) > paper_ratio:
            sheet_h = max_canvas_h
            sheet_w = int(round(sheet_h * paper_ratio))
        else:
            sheet_w = max_canvas_w
            sheet_h = int(round(sheet_w / paper_ratio))

        sheet_x = (w - sheet_w) // 2
        sheet_y = (h - sheet_h) // 2
        sheet_rect = QRectF(sheet_x, sheet_y, sheet_w, sheet_h)

        # Draw subtle drop shadow
        shadow_rect = QRectF(sheet_x + 3, sheet_y + 3, sheet_w, sheet_h)
        painter.fillRect(shadow_rect, QColor(0, 0, 0, 25))

        # Draw white sheet of paper
        painter.fillRect(sheet_rect, QColor("#FFFFFF"))
        painter.setPen(QPen(QColor("#7F9DB9"), 1.0))
        painter.drawRect(sheet_rect)

        # Render the current page slots
        page_idx = self.dialog.current_page_index
        self.dialog.render_page_content(painter, page_idx, sheet_rect)

        painter.end()


class _LayoutCardButton(QPushButton):
    """Clickable layout card on the right column showing a thumbnail preview of the layout."""

    def __init__(
        self,
        layout_id: str,
        title: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.layout_id = layout_id
        self.title_text = title
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(140, 92)
        self.setStyleSheet(
            """
            QPushButton {
                background: #FFFFFF;
                border: 1px solid #C4D5E8;
                border-radius: 3px;
                padding: 4px;
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 8pt;
                text-align: center;
            }
            QPushButton:hover {
                background: #F0F6FE;
                border: 1px solid #70A6E8;
            }
            QPushButton:checked {
                background: #DCECFB;
                border: 2px solid #2874BA;
                font-weight: bold;
                color: #0E294B;
            }
            """
        )

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(4)

        self.lbl_thumb = QLabel(self)
        self.lbl_thumb.setAlignment(Qt.AlignCenter)
        self.lbl_thumb.setFixedSize(120, 60)
        self.lbl_thumb.setStyleSheet("border: 1px solid #D5E1EE; background: #FFFFFF;")
        vbox.addWidget(self.lbl_thumb, 0, Qt.AlignCenter)

        self.lbl_title = QLabel(title, self)
        self.lbl_title.setAlignment(Qt.AlignCenter)
        self.lbl_title.setStyleSheet("border: none; background: transparent;")
        vbox.addWidget(self.lbl_title)

    def set_thumbnail(self, pixmap: QPixmap) -> None:
        self.lbl_thumb.setPixmap(pixmap)


class PrintPicturesDialog(QDialog):
    """
    Windows Live Photo Gallery style Print Pictures Wizard.
    Supports single or batch photo printing, template layouts, paper size,
    print quality dpi, copies, fit-to-frame, and full live preview.
    """

    def __init__(
        self,
        images: list[Path | str | tuple[str, QImage | np.ndarray | QPixmap]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.images_raw = list(images)
        self._loaded_images: list[tuple[str, QImage]] = []
        self._load_image_cache()

        self.current_page_index: int = 0
        self.active_layout_id: str = "full_page"
        self.fit_to_frame: bool = True
        self.copies_per_picture: int = 1

        self.setWindowTitle("Print Pictures")
        self.setWindowIcon(icon_provider.get_icon("print", 16))
        self.resize(820, 620)
        self.setMinimumSize(740, 520)

        # Printer state
        self.printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        self.printer.setPageSize(QPageSize(QPageSize.PageSizeId.Letter))

        self._setup_ui()
        self._populate_printers()
        self._update_template_thumbnails()
        self._recalculate_pages()

    def _load_image_cache(self) -> None:
        """Normalizes all input items into (filename, QImage) pairs."""
        for item in self.images_raw:
            if isinstance(item, tuple):
                name, data = item
                if isinstance(data, QImage):
                    qimg = data
                elif isinstance(data, QPixmap):
                    qimg = data.toImage()
                elif isinstance(data, np.ndarray):
                    h, w = data.shape[:2]
                    qimg = QImage(data.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
                else:
                    continue
                self._loaded_images.append((str(name), qimg))
            else:
                p = Path(item)
                if p.is_file():
                    qimg = QImage(str(p))
                    if not qimg.isNull():
                        self._loaded_images.append((p.name, qimg))

        if not self._loaded_images:
            # Fallback placeholder image so preview always has something to display
            dummy = QImage(600, 400, QImage.Format.Format_RGB888)
            dummy.fill(QColor("#B8D0E8"))
            self._loaded_images.append(("Sample Picture", dummy))

    def _setup_ui(self) -> None:
        main_vbox = QVBoxLayout(self)
        main_vbox.setContentsMargins(0, 0, 0, 0)
        main_vbox.setSpacing(0)

        # ---------------------------------------------------------------------
        # 1. Light Blue Aero Header Banner
        # ---------------------------------------------------------------------
        header_banner = QFrame(self)
        header_banner.setFixedHeight(50)
        header_banner.setStyleSheet(
            """
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #EBF4FD, stop:0.5 #DFEDFA, stop:1 #D2E4F6);
                border-bottom: 1px solid #B4CBE2;
            }
            """
        )
        h_layout = QHBoxLayout(header_banner)
        h_layout.setContentsMargins(16, 0, 16, 0)
        lbl_header = QLabel("How do you want to print your pictures?", header_banner)
        lbl_header.setFont(QFont("Segoe UI", 12))
        lbl_header.setStyleSheet("color: #1A3E6B; font-weight: 500; border: none; background: transparent;")
        h_layout.addWidget(lbl_header)
        main_vbox.addWidget(header_banner)

        # ---------------------------------------------------------------------
        # 2. Top Controls Row (Printer, Paper size, Quality, Paper type)
        # ---------------------------------------------------------------------
        top_ctrl = QFrame(self)
        top_ctrl.setStyleSheet(
            """
            QFrame {
                background: #F7FAFD;
                border-bottom: 1px solid #D5E1EE;
            }
            QLabel {
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                color: #1E395B;
                border: none;
                background: transparent;
            }
            QComboBox {
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                color: #1E395B;
                background: #FFFFFF;
                border: 1px solid #A6B9D0;
                border-radius: 2px;
                padding: 2px 6px;
                min-height: 20px;
            }
            QComboBox:hover {
                border-color: #3399FF;
            }
            """
        )
        t_hbox = QHBoxLayout(top_ctrl)
        t_hbox.setContentsMargins(16, 8, 16, 8)
        t_hbox.setSpacing(12)

        # Printer
        v_prn = QVBoxLayout()
        v_prn.setSpacing(2)
        v_prn.addWidget(QLabel("Printer:", top_ctrl))
        self.cmb_printer = QComboBox(top_ctrl)
        self.cmb_printer.setMinimumWidth(180)
        self.cmb_printer.currentIndexChanged.connect(self._on_printer_changed)
        v_prn.addWidget(self.cmb_printer)
        t_hbox.addLayout(v_prn, 3)

        # Paper size
        v_size = QVBoxLayout()
        v_size.setSpacing(2)
        v_size.addWidget(QLabel("Paper size:", top_ctrl))
        self.cmb_paper_size = QComboBox(top_ctrl)
        self.cmb_paper_size.addItems(["Letter", "A4", "4 x 6 in", "5 x 7 in", "8 x 10 in", "Legal"])
        self.cmb_paper_size.currentIndexChanged.connect(self._on_paper_size_changed)
        v_size.addWidget(self.cmb_paper_size)
        t_hbox.addLayout(v_size, 2)

        # Quality
        v_qual = QVBoxLayout()
        v_qual.setSpacing(2)
        v_qual.addWidget(QLabel("Quality:", top_ctrl))
        self.cmb_quality = QComboBox(top_ctrl)
        self.cmb_quality.addItems(["600 dpi", "300 dpi", "1200 dpi", "Draft (150 dpi)"])
        self.cmb_quality.currentIndexChanged.connect(lambda: self.canvas.update())
        v_qual.addWidget(self.cmb_quality)
        t_hbox.addLayout(v_qual, 2)

        # Paper type
        v_type = QVBoxLayout()
        v_type.setSpacing(2)
        v_type.addWidget(QLabel("Paper type:", top_ctrl))
        self.cmb_paper_type = QComboBox(top_ctrl)
        self.cmb_paper_type.addItems(["Plain Paper", "Photo Paper - Glossy", "Photo Paper - Matte"])
        v_type.addWidget(self.cmb_paper_type)
        t_hbox.addLayout(v_type, 2)

        main_vbox.addWidget(top_ctrl)

        # ---------------------------------------------------------------------
        # 3. Center Section: Left Canvas + Right Layout Templates
        # ---------------------------------------------------------------------
        center_widget = QWidget(self)
        center_widget.setStyleSheet("background: #EBF2F9;")
        c_hbox = QHBoxLayout(center_widget)
        c_hbox.setContentsMargins(12, 12, 12, 8)
        c_hbox.setSpacing(10)

        # Left: Interactive Print Preview
        v_left = QVBoxLayout()
        v_left.setContentsMargins(0, 0, 0, 0)
        v_left.setSpacing(6)

        self.canvas = _PrintPreviewCanvas(self, center_widget)
        v_left.addWidget(self.canvas, 1)

        # Page navigation under preview
        nav_box = QHBoxLayout()
        nav_box.setAlignment(Qt.AlignCenter)
        nav_box.setSpacing(8)

        self.lbl_page_counter = QLabel("1 of 1 page", center_widget)
        self.lbl_page_counter.setStyleSheet("color: #1E395B; font-size: 8.5pt; font-family: 'Segoe UI';")

        self.btn_prev_page = QPushButton("◀", center_widget)
        self.btn_prev_page.setFixedSize(24, 22)
        self.btn_prev_page.setCursor(Qt.PointingHandCursor)
        self.btn_prev_page.setStyleSheet(
            "QPushButton { border-radius: 11px; background: #FFFFFF; border: 1px solid #A6B9D0; font-size: 8pt; color: #1E395B; }"
            "QPushButton:hover { background: #E5F1FB; border-color: #3399FF; }"
            "QPushButton:disabled { color: #A0B0C0; border-color: #D0D8E0; background: #F0F4F8; }"
        )
        self.btn_prev_page.clicked.connect(self._prev_page)

        self.btn_next_page = QPushButton("▶", center_widget)
        self.btn_next_page.setFixedSize(24, 22)
        self.btn_next_page.setCursor(Qt.PointingHandCursor)
        self.btn_next_page.setStyleSheet(self.btn_prev_page.styleSheet())
        self.btn_next_page.clicked.connect(self._next_page)

        nav_box.addWidget(self.lbl_page_counter)
        nav_box.addWidget(self.btn_prev_page)
        nav_box.addWidget(self.btn_next_page)
        v_left.addLayout(nav_box)

        c_hbox.addLayout(v_left, 1)

        # Right: Scrollable Layout Templates Column
        scroll_layouts = QScrollArea(center_widget)
        scroll_layouts.setFixedWidth(166)
        scroll_layouts.setWidgetResizable(True)
        scroll_layouts.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_layouts.setStyleSheet(
            """
            QScrollArea {
                border: 1px solid #C4D5E8;
                background: #F7FAFD;
                border-radius: 2px;
            }
            """
        )

        holder_layouts = QWidget()
        holder_layouts.setStyleSheet("background: transparent;")
        v_layouts = QVBoxLayout(holder_layouts)
        v_layouts.setContentsMargins(6, 6, 6, 6)
        v_layouts.setSpacing(6)

        self.template_btn_group = QButtonGroup(self)
        self.template_btn_group.setExclusive(True)

        self.templates_data = [
            ("full_page", "Full page photo"),
            ("two_5x7", "13 x 18 cm. (2)"),
            ("one_8x10", "20 x 25 cm. (1)"),
            ("two_4x6", "10 x 15 cm. (2)"),
            ("four_wallet", "9 x 13 cm. (4)"),
            ("contact_sheet", "Contact sheet"),
        ]

        self.template_cards: dict[str, _LayoutCardButton] = {}

        for tid, title in self.templates_data:
            card = _LayoutCardButton(tid, title, holder_layouts)
            if tid == "full_page":
                card.setChecked(True)
            card.clicked.connect(lambda _, l_id=tid: self._select_layout(l_id))
            self.template_btn_group.addButton(card)
            v_layouts.addWidget(card)
            self.template_cards[tid] = card

        v_layouts.addStretch(1)
        scroll_layouts.setWidget(holder_layouts)
        c_hbox.addWidget(scroll_layouts)

        main_vbox.addWidget(center_widget, 1)

        # ---------------------------------------------------------------------
        # 4. Bottom Action Bar
        # ---------------------------------------------------------------------
        bottom_bar = QFrame(self)
        bottom_bar.setFixedHeight(48)
        bottom_bar.setStyleSheet(
            """
            QFrame {
                background: #F0F4F9;
                border-top: 1px solid #D5E1EE;
            }
            """
        )
        b_hbox = QHBoxLayout(bottom_bar)
        b_hbox.setContentsMargins(16, 0, 16, 0)
        b_hbox.setSpacing(16)

        # Copies spinbox
        lbl_copies = QLabel("Copies of each picture:", bottom_bar)
        lbl_copies.setStyleSheet("color: #1E395B; font-size: 8.5pt; font-family: 'Segoe UI';")
        self.spn_copies = QSpinBox(bottom_bar)
        self.spn_copies.setRange(1, 99)
        self.spn_copies.setValue(1)
        self.spn_copies.setFixedWidth(54)
        self.spn_copies.setStyleSheet(
            "font-family: 'Segoe UI'; font-size: 8.5pt; border: 1px solid #A6B9D0; background: #FFFFFF; border-radius: 2px;"
        )
        self.spn_copies.valueChanged.connect(self._on_copies_changed)

        # Fit picture to frame checkbox
        self.chk_fit = QCheckBox("Fit picture to frame", bottom_bar)
        self.chk_fit.setChecked(True)
        self.chk_fit.setStyleSheet(
            "QCheckBox { color: #1E395B; font-size: 8.5pt; font-family: 'Segoe UI'; }"
            "QCheckBox::indicator { width: 13px; height: 13px; }"
        )
        self.chk_fit.toggled.connect(self._on_fit_toggled)

        b_hbox.addWidget(lbl_copies)
        b_hbox.addWidget(self.spn_copies)
        b_hbox.addWidget(self.chk_fit)

        # Options link/menu
        self.btn_options = QPushButton("Options...", bottom_bar)
        self.btn_options.setFlat(True)
        self.btn_options.setCursor(Qt.PointingHandCursor)
        self.btn_options.setStyleSheet(
            "QPushButton { color: #0066CC; border: none; background: transparent; font-family: 'Segoe UI'; font-size: 8.5pt; text-decoration: underline; }"
            "QPushButton:hover { color: #003399; }"
        )
        self.btn_options.clicked.connect(self._show_options_menu)
        b_hbox.addWidget(self.btn_options)

        b_hbox.addStretch(1)

        # Print & Cancel buttons
        self.btn_print = QPushButton("Print", bottom_bar)
        self.btn_print.setDefault(True)
        self.btn_print.setFixedSize(86, 26)
        self.btn_print.setCursor(Qt.PointingHandCursor)
        self.btn_print.setStyleSheet(
            """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F4F9FE, stop:1 #D6E8F8);
                border: 1px solid #70A6E8;
                border-radius: 2px;
                color: #0F2D59;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                font-weight: 600;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #FFFFFF, stop:1 #BCE0FD);
                border-color: #3399FF;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #CCE4F7, stop:1 #A8D3F5);
                border-color: #2468A8;
            }
            """
        )
        self.btn_print.clicked.connect(self._execute_print)

        self.btn_cancel = QPushButton("Cancel", bottom_bar)
        self.btn_cancel.setFixedSize(86, 26)
        self.btn_cancel.setCursor(Qt.PointingHandCursor)
        self.btn_cancel.setStyleSheet(
            """
            QPushButton {
                background: #FFFFFF;
                border: 1px solid #A6B9D0;
                border-radius: 2px;
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
            }
            QPushButton:hover {
                background: #F0F6FE;
                border-color: #70A6E8;
            }
            """
        )
        self.btn_cancel.clicked.connect(self.reject)

        b_hbox.addWidget(self.btn_print)
        b_hbox.addWidget(self.btn_cancel)

        main_vbox.addWidget(bottom_bar)

    def _populate_printers(self) -> None:
        """Populates installed system printers into the printer combo box."""
        printers = QPrinterInfo.availablePrinters()
        default_name = QPrinterInfo.defaultPrinterName()
        self.cmb_printer.clear()

        def_idx = 0
        for idx, prn in enumerate(printers):
            p_name = prn.printerName()
            self.cmb_printer.addItem(p_name, prn)
            if p_name == default_name:
                def_idx = idx

        if self.cmb_printer.count() > 0:
            self.cmb_printer.setCurrentIndex(def_idx)

    def _on_printer_changed(self, index: int) -> None:
        prn_info = self.cmb_printer.currentData()
        if isinstance(prn_info, QPrinterInfo):
            self.printer = QPrinter(prn_info, QPrinter.PrinterMode.HighResolution)
            self._on_paper_size_changed(self.cmb_paper_size.currentIndex())

    def _on_paper_size_changed(self, index: int) -> None:
        size_name = self.cmb_paper_size.currentText()
        size_map = {
            "Letter": QPageSize.PageSizeId.Letter,
            "A4": QPageSize.PageSizeId.A4,
            "4 x 6 in": QPageSize.PageSizeId.Custom,
            "5 x 7 in": QPageSize.PageSizeId.Custom,
            "8 x 10 in": QPageSize.PageSizeId.Custom,
            "Legal": QPageSize.PageSizeId.Legal,
        }
        pid = size_map.get(size_name, QPageSize.PageSizeId.Letter)
        if pid != QPageSize.PageSizeId.Custom:
            self.printer.setPageSize(QPageSize(pid))
        self.canvas.update()

    def get_current_paper_aspect(self) -> tuple[float, float]:
        """Returns the paper width and height ratio based on the active selection."""
        size_name = self.cmb_paper_size.currentText()
        if size_name == "A4":
            return (210.0, 297.0)
        elif size_name == "4 x 6 in":
            return (4.0, 6.0)
        elif size_name == "5 x 7 in":
            return (5.0, 7.0)
        elif size_name == "8 x 10 in":
            return (8.0, 10.0)
        elif size_name == "Legal":
            return (8.5, 14.0)
        return (8.5, 11.0)  # Letter default

    def _select_layout(self, layout_id: str) -> None:
        self.active_layout_id = layout_id
        self._recalculate_pages()
        self.canvas.update()

    def _on_fit_toggled(self, checked: bool) -> None:
        self.fit_to_frame = checked
        self.canvas.update()

    def _on_copies_changed(self, val: int) -> None:
        self.copies_per_picture = val
        self._recalculate_pages()
        self.canvas.update()

    def _get_expanded_image_list(self) -> list[tuple[str, QImage]]:
        """Returns the complete sequence of images accounting for copies."""
        out = []
        for item in self._loaded_images:
            for _ in range(self.copies_per_picture):
                out.append(item)
        return out

    def _get_layout_slots_per_page(self) -> int:
        slots_map = {
            "full_page": 1,
            "two_5x7": 2,
            "one_8x10": 1,
            "two_4x6": 2,
            "four_wallet": 4,
            "contact_sheet": 35,
        }
        return slots_map.get(self.active_layout_id, 1)

    def _recalculate_pages(self) -> None:
        total_items = len(self._get_expanded_image_list())
        slots = self._get_layout_slots_per_page()
        total_pages = max(1, math.ceil(total_items / float(slots)))

        self.current_page_index = max(0, min(self.current_page_index, total_pages - 1))
        self.lbl_page_counter.setText(f"{self.current_page_index + 1} of {total_pages} page{'s' if total_pages > 1 else ''}")
        self.btn_prev_page.setEnabled(self.current_page_index > 0)
        self.btn_next_page.setEnabled(self.current_page_index < total_pages - 1)

    def _prev_page(self) -> None:
        if self.current_page_index > 0:
            self.current_page_index -= 1
            self._recalculate_pages()
            self.canvas.update()

    def _next_page(self) -> None:
        slots = self._get_layout_slots_per_page()
        total_pages = max(1, math.ceil(len(self._get_expanded_image_list()) / float(slots)))
        if self.current_page_index < total_pages - 1:
            self.current_page_index += 1
            self._recalculate_pages()
            self.canvas.update()

    def _update_template_thumbnails(self) -> None:
        """Paints miniature representations of each template for the cards on the right."""
        sample_img = self._loaded_images[0][1] if self._loaded_images else None

        for tid, card in self.template_cards.items():
            pm = QPixmap(120, 60)
            pm.fill(QColor("#FFFFFF"))
            p = QPainter(pm)
            p.setRenderHint(QPainter.Antialiasing, True)
            p.setRenderHint(QPainter.SmoothPixmapTransform, True)

            # Miniature paper sheet
            sheet_rect = QRectF(18, 4, 84, 52)
            p.fillRect(QRectF(20, 6, 84, 52), QColor(0, 0, 0, 20))
            p.fillRect(sheet_rect, QColor("#FFFFFF"))
            p.setPen(QPen(QColor("#B8C8D8"), 1.0))
            p.drawRect(sheet_rect)

            slots = self.get_template_slots(tid, sheet_rect)
            for s_rect in slots:
                if sample_img and not sample_img.isNull():
                    self._paint_image_in_rect(p, sample_img, s_rect, fit_frame=True)
                else:
                    p.fillRect(s_rect, QColor("#6BA3D6"))
                p.setPen(QPen(QColor("#A0B8D0"), 0.5))
                p.drawRect(s_rect)

            p.end()
            card.set_thumbnail(pm)

    def get_template_slots(self, layout_id: str, page_rect: QRectF) -> list[QRectF]:
        """Calculates slot geometries on the page for any layout."""
        pw, ph = page_rect.width(), page_rect.height()
        x0, y0 = page_rect.x(), page_rect.y()

        if layout_id == "full_page":
            # 1 photo taking 94% of page
            margin = min(pw, ph) * 0.04
            return [QRectF(x0 + margin, y0 + margin, pw - 2 * margin, ph - 2 * margin)]

        elif layout_id == "two_5x7":
            # 2 photos stacked vertically (or side-by-side)
            margin = min(pw, ph) * 0.04
            slot_h = (ph - 3 * margin) / 2.0
            slot_w = pw - 2 * margin
            return [
                QRectF(x0 + margin, y0 + margin, slot_w, slot_h),
                QRectF(x0 + margin, y0 + 2 * margin + slot_h, slot_w, slot_h),
            ]

        elif layout_id == "one_8x10":
            # 1 photo centered 8:10 aspect
            margin = min(pw, ph) * 0.06
            target_aspect = 8.0 / 10.0
            avail_w = pw - 2 * margin
            avail_h = ph - 2 * margin
            if avail_w / float(avail_h) > target_aspect:
                sh = avail_h
                sw = sh * target_aspect
            else:
                sw = avail_w
                sh = sw / target_aspect
            return [QRectF(x0 + (pw - sw) / 2.0, y0 + (ph - sh) / 2.0, sw, sh)]

        elif layout_id == "two_4x6":
            # 2 photos 4:6 aspect
            margin = min(pw, ph) * 0.04
            slot_h = (ph - 3 * margin) / 2.0
            slot_w = pw - 2 * margin
            return [
                QRectF(x0 + margin, y0 + margin, slot_w, slot_h),
                QRectF(x0 + margin, y0 + 2 * margin + slot_h, slot_w, slot_h),
            ]

        elif layout_id == "four_wallet":
            # 4 photos in 2x2 grid
            margin = min(pw, ph) * 0.03
            slot_w = (pw - 3 * margin) / 2.0
            slot_h = (ph - 3 * margin) / 2.0
            return [
                QRectF(x0 + margin, y0 + margin, slot_w, slot_h),
                QRectF(x0 + 2 * margin + slot_w, y0 + margin, slot_w, slot_h),
                QRectF(x0 + margin, y0 + 2 * margin + slot_h, slot_w, slot_h),
                QRectF(x0 + 2 * margin + slot_w, y0 + 2 * margin + slot_h, slot_w, slot_h),
            ]

        elif layout_id == "contact_sheet":
            # 35 photos (5 cols x 7 rows)
            cols, rows = 5, 7
            margin = min(pw, ph) * 0.02
            slot_w = (pw - (cols + 1) * margin) / float(cols)
            slot_h = (ph - (rows + 1) * margin) / float(rows)
            out = []
            for r in range(rows):
                for c in range(cols):
                    out.append(QRectF(
                        x0 + margin + c * (slot_w + margin),
                        y0 + margin + r * (slot_h + margin),
                        slot_w,
                        slot_h,
                    ))
            return out

        return [page_rect]

    def _paint_image_in_rect(
        self,
        painter: QPainter,
        qimg: QImage,
        rect: QRectF,
        fit_frame: bool,
    ) -> None:
        """Paints an image within a slot, respecting fit-to-frame (cropping) vs fit-entire (letterbox)."""
        iw, ih = qimg.width(), qimg.height()
        rw, rh = rect.width(), rect.height()
        if iw <= 0 or ih <= 0 or rw <= 0 or rh <= 0:
            return

        img_aspect = iw / float(ih)
        rect_aspect = rw / float(rh)

        if fit_frame:
            # Crop to fill slot completely without margins
            if img_aspect > rect_aspect:
                crop_w = int(round(ih * rect_aspect))
                src_x = (iw - crop_w) // 2
                src_rect = QRectF(src_x, 0, crop_w, ih)
            else:
                crop_h = int(round(iw / rect_aspect))
                src_y = (ih - crop_h) // 2
                src_rect = QRectF(0, src_y, iw, crop_h)
            painter.drawImage(rect, qimg, src_rect)
        else:
            # Letterbox/pillarbox to show 100% of the image
            if img_aspect > rect_aspect:
                target_w = rw
                target_h = target_w / img_aspect
            else:
                target_h = rh
                target_w = target_h * img_aspect
            tx = rect.x() + (rw - target_w) / 2.0
            ty = rect.y() + (rh - target_h) / 2.0
            painter.drawImage(QRectF(tx, ty, target_w, target_h), qimg)

    def render_page_content(
        self,
        painter: QPainter,
        page_index: int,
        page_rect: QRectF,
    ) -> None:
        """Renders the specified page onto painter (used by both live preview and printer)."""
        all_items = self._get_expanded_image_list()
        slots_per_page = self._get_layout_slots_per_page()
        start_idx = page_index * slots_per_page
        slots = self.get_template_slots(self.active_layout_id, page_rect)

        for s_idx, s_rect in enumerate(slots):
            item_idx = start_idx + s_idx
            if item_idx < len(all_items):
                fname, qimg = all_items[item_idx]
                self._paint_image_in_rect(painter, qimg, s_rect, fit_frame=self.fit_to_frame)

                # For contact sheet, print filename underneath
                if self.active_layout_id == "contact_sheet":
                    painter.setPen(QColor("#334E68"))
                    f = QFont("Segoe UI", 6)
                    painter.setFont(f)
                    painter.drawText(
                        QRectF(s_rect.x(), s_rect.bottom() - 8, s_rect.width(), 8),
                        Qt.AlignCenter | Qt.TextSingleLine,
                        fname[:16],
                    )

    def _show_options_menu(self) -> None:
        """Shows the Options menu offering Printer Properties and native Windows Print Wizard."""
        menu = QMenu(self.btn_options)
        menu.setStyleSheet("QMenu { background: #FFFFFF; border: 1px solid #70A6E8; }")
        act_props = menu.addAction("Printer Properties...")
        act_wizard = menu.addAction("Print with Windows Photo Wizard...")

        act_props.triggered.connect(self._open_printer_properties)
        act_wizard.triggered.connect(self._launch_windows_wizard)
        menu.exec(self.btn_options.mapToGlobal(self.btn_options.rect().bottomLeft()))

    def _open_printer_properties(self) -> None:
        dlg = QPrintDialog(self.printer, self)
        if dlg.exec() == QDialog.Accepted:
            self._on_printer_changed(self.cmb_printer.currentIndex())

    def _launch_windows_wizard(self) -> None:
        """Launches the native Windows PhotoPrintWizard on the first disk photo."""
        for item in self.images_raw:
            path_val = item[0] if isinstance(item, tuple) else item
            fpath = Path(path_val)
            if fpath.is_file():
                try:
                    os.startfile(str(fpath), "print")
                    self.accept()
                    return
                except Exception as e:
                    QMessageBox.warning(self, "Print", f"Could not launch Windows Photo Wizard:\n{e}")
                    return

    def _execute_print(self) -> None:
        """Executes the high-resolution print job using QPrinter and QPainter."""
        total_items = len(self._get_expanded_image_list())
        slots = self._get_layout_slots_per_page()
        total_pages = max(1, math.ceil(total_items / float(slots)))

        # Resolution setting
        qual_str = self.cmb_quality.currentText()
        dpi_val = 600
        if "300" in qual_str:
            dpi_val = 300
        elif "1200" in qual_str:
            dpi_val = 1200
        elif "150" in qual_str:
            dpi_val = 150
        self.printer.setResolution(dpi_val)

        try:
            painter = QPainter(self.printer)
            if not painter.isActive():
                QMessageBox.warning(self, "Print", "Could not start print job. Please check printer connection.")
                return

            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

            for p_idx in range(total_pages):
                if p_idx > 0:
                    self.printer.newPage()
                paint_rect = QRectF(self.printer.pageLayout().paintRectPixels(self.printer.resolution()))
                self.render_page_content(painter, p_idx, paint_rect)

            painter.end()
            self.accept()
        except Exception as ex:
            QMessageBox.critical(self, "Print Error", f"An error occurred while printing:\n{ex}")
