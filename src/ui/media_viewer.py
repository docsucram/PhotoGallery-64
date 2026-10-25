"""
SingleMediaViewer (QWidget) for PhotoGallery64.
Reuses ImageGraphicsView, CropScene, EditCommand (QUndoCommand), ClickableVideoWidget,
and MouseOverlay from legacy_reference/imgBrowserv13.py.
Implements WLPG's Full-Size Viewer & 'Fix / Fine-Tune' Adjustments Drawer:
- Zoom modes ('fit', 'fill', '100%', 'custom')
- Anchored CropScene rubber-band crop
- Auto adjust, Exposure/Contrast, Saturation/Warmth, Straighten, and Preset Effect tiles
- Non-destructive '.OriginalFiles' backup stashing on save & 'Revert to original'
- Static-image-only (is_video == 0) Face Region tagging rectangle overlay
- Integrated QMediaPlayer + QAudioOutput video playback
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageOps
from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, QRectF, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QImage,
    QKeyEvent,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
    QPixmap,
    QUndoCommand,
    QUndoStack,
)
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsSimpleTextItem,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QStackedWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from legacy_reference.imgBrowserv13 import (
    ClickableVideoWidget,
    EditCommand,
    ImageGraphicsView,
    MouseOverlay,
)
from src.ui.crop_tool import CropOverlayItem, CropScene
from src.ui.markup_overlay import MarkupFloatingToolbar, MarkupElement, flatten_markup_onto_image
from src.ui.ribbon_toolbar import make_person_avatar_pixmap, make_wlpg_icon
from src.imaging.opencv_engine import (
    PhotoAdjustmentRecipe,
    analyze_auto_adjust,
    detect_auto_straighten_angle,
    render_adjusted_image,
)


def pil_to_qimage(pil_img: Image.Image) -> QImage:
    """Converts a PIL Image (RGB/RGBA) to a standalone QImage copy."""
    if pil_img.mode != "RGBA":
        rgba = pil_img.convert("RGBA")
    else:
        rgba = pil_img
    raw = rgba.tobytes("raw", "RGBA")
    qimg = QImage(raw, rgba.width, rgba.height, rgba.width * 4, QImage.Format_RGBA8888)
    return qimg.copy()


class _FaceNameBadgeItem(QGraphicsItem):
    """
    Zoom-independent WLPG face name pill rendered directly below the bottom center
    of a face rectangle (matches media_1792269887974_67460900.png).
    Uses ItemIgnoresTransformations so the text is always a crisp 9.5pt Segoe UI pill
    regardless of photo resolution or zoom level.
    """

    def __init__(self, name: str, person_id: int | None = None) -> None:
        super().__init__()
        self.name = str(name or "")
        self.person_id = person_id
        self.highlighted: bool = False
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setZValue(15)
        self._font = QFont("Segoe UI", 10, QFont.DemiBold)
        fm = QFontMetrics(self._font)
        text_w = max(32, fm.horizontalAdvance(self.name))
        self._pill_w = float(text_w + 20)
        self._pill_h = float(max(24, fm.height() + 6))

    def setText(self, text: str) -> None:
        self.prepareGeometryChange()
        self.name = str(text or "")
        fm = QFontMetrics(self._font)
        text_w = max(32, fm.horizontalAdvance(self.name))
        self._pill_w = float(text_w + 20)
        self._pill_h = float(max(24, fm.height() + 6))
        self.update()

    def text(self) -> str:
        return self.name

    def set_highlighted(self, highlighted: bool) -> None:
        if self.highlighted != highlighted:
            self.highlighted = highlighted
            self.update()

    def boundingRect(self) -> QRectF:
        return QRectF(-self._pill_w / 2.0 - 2.0, 2.0, self._pill_w + 4.0, self._pill_h + 6.0)

    def paint(self, painter: QPainter, _option, _widget=None) -> None:
        if not self.name:
            return
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(-self._pill_w / 2.0, 4.0, self._pill_w, self._pill_h)

        # Subtle drop shadow
        shadow_rect = rect.translated(0.0, 1.0)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 75))
        painter.drawRoundedRect(shadow_rect, 4.0, 4.0)

        # Pill background & border (matching WLPG screenshot media_1792269887974_67460900.png)
        if self.highlighted:
            painter.setBrush(QColor("#D8EAFD"))
            painter.setPen(QPen(QColor("#2979CC"), 1.3))
        else:
            painter.setBrush(QColor(238, 242, 247, 245))
            painter.setPen(QPen(QColor("#7A8A99"), 1.0))
        painter.drawRoundedRect(rect, 3.5, 3.5)

        painter.setFont(self._font)
        painter.setPen(QColor("#111111"))
        painter.drawText(rect, Qt.AlignCenter, self.name)
        painter.restore()


class _FaceHoverRegionItem(QGraphicsRectItem):
    """Transparent sensor that detects mouse hover over a face region to reveal face boxes and badges."""

    def __init__(
        self,
        rect: QRectF,
        viewer: "SingleMediaViewer",
        related_items: list[QGraphicsItem],
        person_id: int | None,
        region_data: dict | None = None,
    ) -> None:
        super().__init__(rect)
        self.viewer = viewer
        self.related_items = related_items
        self.person_id = person_id
        self.region_data = region_data or {}
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)
        self.setPen(QPen(Qt.NoPen))
        self.setBrush(QBrush(QColor(0, 0, 0, 1)))
        self.setZValue(3)

    def hoverEnterEvent(self, event) -> None:
        if hasattr(self.viewer, "is_in_edit_mode") and self.viewer.is_in_edit_mode():
            return
        for it in self.related_items:
            it.setVisible(True)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        if getattr(self.viewer, "_highlighted_person_id", None) != self.person_id:
            for it in self.related_items:
                it.setVisible(False)
        super().hoverLeaveEvent(event)

    def contextMenuEvent(self, event) -> None:
        if hasattr(self.viewer, "is_in_edit_mode") and self.viewer.is_in_edit_mode():
            event.ignore()
            return
        event.accept()
        self.viewer.show_face_region_context_menu(
            event.screenPos(), self.region_data, self.person_id
        )


class _FaceAdjustToolbar(QFrame):
    """Floating Aero action pill shown when interactively adjusting a face bounding box."""

    saved = Signal()
    cancelled = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("FaceAdjustToolbar")
        self.setStyleSheet(
            """
            QFrame#FaceAdjustToolbar {
                background: rgba(255, 255, 255, 0.95);
                border: 1px solid #84ACDD;
                border-radius: 4px;
            }
            QLabel {
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                color: #1E395B;
                font-weight: 600;
            }
            QPushButton#AdjustSaveBtn {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2A79D6, stop:1 #1B58B0);
                color: #FFFFFF;
                border: 1px solid #14448C;
                border-radius: 3px;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                font-weight: bold;
                padding: 2px 10px;
                min-height: 22px;
            }
            QPushButton#AdjustSaveBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3D8CE8, stop:1 #2264C4);
            }
            QPushButton#AdjustCancelBtn {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FDFDFD, stop:1 #EAEAEA);
                color: #222222;
                border: 1px solid #ACACAC;
                border-radius: 3px;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                padding: 2px 8px;
                min-height: 22px;
            }
            QPushButton#AdjustCancelBtn:hover {
                background: #E0E8F5;
                border-color: #7DA2CE;
            }
            """
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)

        lbl = QLabel("Adjust face box:", self)
        layout.addWidget(lbl)

        btn_save = QPushButton("✓ Done", self)
        btn_save.setObjectName("AdjustSaveBtn")
        btn_save.setCursor(Qt.PointingHandCursor)
        btn_save.setToolTip("Save new bounding box (Enter)")
        btn_save.clicked.connect(self.saved.emit)
        layout.addWidget(btn_save)

        btn_cancel = QPushButton("✕ Cancel", self)
        btn_cancel.setObjectName("AdjustCancelBtn")
        btn_cancel.setCursor(Qt.PointingHandCursor)
        btn_cancel.setToolTip("Cancel adjustment (Esc)")
        btn_cancel.clicked.connect(self.cancelled.emit)
        layout.addWidget(btn_cancel)
        self.adjustSize()


class _FaceTagTargetBoxItem(QGraphicsItem):
    """
    Interactive WLPG manual face tagging box with thick white corner brackets
    (matches media_1792269928233_4b31b221.png).
    Can be dragged by its interior or resized by its corners/edges.
    """

    def __init__(
        self,
        rect: QRectF,
        image_bounds: QRectF,
        on_changed_callback=None,
        on_changed=None,
    ) -> None:
        super().__init__()
        self._rect = QRectF(rect)
        self._image_bounds = QRectF(image_bounds)
        self._on_changed = on_changed if on_changed is not None else on_changed_callback
        self._drag_mode: str | None = None
        self._drag_start_scene = QPointF()
        self._drag_start_rect = QRectF()
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.setZValue(25)

    def rect(self) -> QRectF:
        return QRectF(self._rect)

    def setRect(self, rect: QRectF) -> None:
        self.prepareGeometryChange()
        self._rect = self._clamp_rect(rect)
        self.update()
        if self._on_changed is not None:
            self._on_changed()

    def _clamp_rect(self, r: QRectF) -> QRectF:
         r = r.normalized()
         ib = self._image_bounds
         if ib.isEmpty():
             return r
         min_sz = max(16.0, min(ib.width(), ib.height()) * 0.03)
         w = max(min_sz, min(r.width(), ib.width()))
         h = max(min_sz, min(r.height(), ib.height()))
         x = max(ib.left(), min(r.left(), ib.right() - w))
         y = max(ib.top(), min(r.top(), ib.bottom() - h))
         return QRectF(x, y, w, h)

    def boundingRect(self) -> QRectF:
        pad = 24.0
        return self._rect.adjusted(-pad, -pad, pad, pad)

    def _hit_mode(self, pos: QPointF) -> str | None:
        r = self._rect
        # Scale tolerance by view zoom so handles are easy to grab at any resolution
        tol = max(10.0, min(r.width(), r.height()) * 0.16)
        near_l = abs(pos.x() - r.left()) <= tol
        near_r = abs(pos.x() - r.right()) <= tol
        near_t = abs(pos.y() - r.top()) <= tol
        near_b = abs(pos.y() - r.bottom()) <= tol
        in_x = (r.left() - tol) <= pos.x() <= (r.right() + tol)
        in_y = (r.top() - tol) <= pos.y() <= (r.bottom() + tol)

        if near_t and near_l:
            return "tl"
        if near_t and near_r:
            return "tr"
        if near_b and near_l:
            return "bl"
        if near_b and near_r:
            return "br"
        if near_l and in_y:
            return "l"
        if near_r and in_y:
            return "r"
        if near_t and in_x:
            return "t"
        if near_b and in_x:
            return "b"
        if r.contains(pos):
            return "move"
        return None

    def hoverMoveEvent(self, event) -> None:
        mode = self._hit_mode(event.pos())
        cursors = {
            "tl": Qt.SizeFDiagCursor,
            "br": Qt.SizeFDiagCursor,
            "tr": Qt.SizeBDiagCursor,
            "bl": Qt.SizeBDiagCursor,
            "l": Qt.SizeHorCursor,
            "r": Qt.SizeHorCursor,
            "t": Qt.SizeVerCursor,
            "b": Qt.SizeVerCursor,
            "move": Qt.SizeAllCursor,
        }
        if mode in cursors:
            self.setCursor(cursors[mode])
        else:
            self.unsetCursor()
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            mode = self._hit_mode(event.pos())
            if mode is not None:
                self._drag_mode = mode
                self._drag_start_scene = event.scenePos()
                self._drag_start_rect = QRectF(self._rect)
                event.accept()
                return
        event.ignore()

    def mouseMoveEvent(self, event) -> None:
        if not self._drag_mode:
            super().mouseMoveEvent(event)
            return
        delta = event.scenePos() - self._drag_start_scene
        r = QRectF(self._drag_start_rect)
        min_sz = max(16.0, min(self._image_bounds.width(), self._image_bounds.height()) * 0.03)
        m = self._drag_mode
        if m == "move":
            r.translate(delta)
        else:
            if "l" in m:
                r.setLeft(min(r.right() - min_sz, r.left() + delta.x()))
            if "r" in m:
                r.setRight(max(r.left() + min_sz, r.right() + delta.x()))
            if "t" in m:
                r.setTop(min(r.bottom() - min_sz, r.top() + delta.y()))
            if "b" in m:
                r.setBottom(max(r.top() + min_sz, r.bottom() + delta.y()))
        self.setRect(r)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_mode:
            self._drag_mode = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paint(self, painter: QPainter, _option, _widget=None) -> None:
        painter.save()
        r = self._rect

        # 1. Outer dark shadow border + crisp inner white border (cosmetic so thickness is zoom-invariant)
        shadow_pen = QPen(QColor(0, 0, 0, 165), 3.0)
        shadow_pen.setCosmetic(True)
        painter.setPen(shadow_pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(r)

        thin_white = QPen(QColor(255, 255, 255, 210), 1.4)
        thin_white.setCosmetic(True)
        painter.setPen(thin_white)
        painter.drawRect(r)

        # 2. Thick white L-shaped corner brackets matching WLPG screenshot (media_1792269928233_4b31b221.png)
        arm = max(6.0, min(r.width(), r.height()) * 0.22)
        path = QPainterPath()
        # Top-Left
        path.moveTo(r.left(), r.top() + arm)
        path.lineTo(r.left(), r.top())
        path.lineTo(r.left() + arm, r.top())
        # Top-Right
        path.moveTo(r.right() - arm, r.top())
        path.lineTo(r.right(), r.top())
        path.lineTo(r.right(), r.top() + arm)
        # Bottom-Left
        path.moveTo(r.left(), r.bottom() - arm)
        path.lineTo(r.left(), r.bottom())
        path.lineTo(r.left() + arm, r.bottom())
        # Bottom-Right
        path.moveTo(r.right() - arm, r.bottom())
        path.lineTo(r.right(), r.bottom())
        path.lineTo(r.right(), r.bottom() - arm)

        corner_shadow = QPen(QColor(0, 0, 0, 190), 5.5, Qt.SolidLine, Qt.SquareCap, Qt.MiterJoin)
        corner_shadow.setCosmetic(True)
        painter.setPen(corner_shadow)
        painter.drawPath(path)

        corner_white = QPen(QColor(255, 255, 255, 255), 3.5, Qt.SolidLine, Qt.SquareCap, Qt.MiterJoin)
        corner_white.setCosmetic(True)
        painter.setPen(corner_white)
        painter.drawPath(path)
        painter.restore()


class _PersonPopupRowButton(QPushButton):
    """Clickable row in the WLPG 'Tag someone' popup with 36x36 square face avatar and name."""

    person_clicked = Signal(str)

    def __init__(self, name: str, avatar_pixmap: QPixmap, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.person_name = name
        self.setFixedHeight(42)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(
            """
            QPushButton {
                background: #FFFFFF;
                border: 1px solid transparent;
                border-radius: 2px;
                text-align: left;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F2F8FE, stop:1 #DCEBFC);
                border: 1px solid #7AC1FF;
            }
            """
        )
        hbox = QHBoxLayout(self)
        hbox.setContentsMargins(4, 3, 6, 3)
        hbox.setSpacing(8)

        lbl_icon = QLabel(self)
        lbl_icon.setFixedSize(36, 36)
        lbl_icon.setPixmap(avatar_pixmap.scaled(36, 36, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))
        lbl_icon.setStyleSheet("border: 1px solid #B8C8D8; background: #F0F4F8;")
        hbox.addWidget(lbl_icon)

        lbl_name = QLabel(name, self)
        lbl_name.setStyleSheet(
            "color: #111111; font-family: 'Segoe UI'; font-size: 9pt; background: transparent; border: none;"
        )
        hbox.addWidget(lbl_name, 1)
        self.clicked.connect(lambda: self.person_clicked.emit(self.person_name))


class _TagSomeonePopup(QFrame):
    """
    Floating WLPG 'Tag someone' card anchored beside the active face target box
    (matches media_1792269928233_4b31b221.png).
    Displays:
    - 'Tag someone' blue title + '✕' close button
    - Search QLineEdit with magnifying glass icon
    - Scrollable list of existing people with 36x36 square face avatars + names
    """

    person_selected = Signal(str)
    cancelled = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("TagSomeonePopup")
        self.setFixedSize(235, 280)
        self.setStyleSheet(
            """
            QFrame#TagSomeonePopup {
                background-color: #FFFFFF;
                border: 1px solid #7A8A99;
                border-radius: 3px;
            }
            """
        )
        self._people_data: list[dict] = []

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(10, 8, 10, 10)
        vbox.setSpacing(6)

        # 1. Header row: "Tag someone" + close "✕"
        hdr = QHBoxLayout()
        hdr.setContentsMargins(0, 0, 0, 0)
        self.lbl_title = QLabel("Tag someone", self)
        self.lbl_title.setStyleSheet(
            "color: #1B64B0; font-family: 'Segoe UI'; font-size: 10pt; border: none; background: transparent;"
        )
        hdr.addWidget(self.lbl_title)
        hdr.addStretch(1)

        self.btn_close = QToolButton(self)
        self.btn_close.setText("✕")
        self.btn_close.setFixedSize(18, 18)
        self.btn_close.setCursor(Qt.PointingHandCursor)
        self.btn_close.setStyleSheet(
            """
            QToolButton {
                border: none;
                background: transparent;
                color: #666666;
                font-family: 'Segoe UI';
                font-size: 9pt;
                font-weight: bold;
            }
            QToolButton:hover {
                color: #C50F1F;
                background: #F8D7DA;
                border-radius: 2px;
            }
            """
        )
        self.btn_close.clicked.connect(self.cancelled.emit)
        hdr.addWidget(self.btn_close)
        vbox.addLayout(hdr)

        # 2. Search input box with magnifying glass icon
        search_frame = QFrame(self)
        search_frame.setFixedHeight(24)
        search_frame.setStyleSheet(
            """
            QFrame {
                background: #FFFFFF;
                border: 1px solid #3380CC;
                border-radius: 2px;
            }
            """
        )
        s_hbox = QHBoxLayout(search_frame)
        s_hbox.setContentsMargins(4, 1, 4, 1)
        s_hbox.setSpacing(4)

        self.txt_search = QLineEdit(search_frame)
        self.txt_search.setPlaceholderText("Type a name")
        self.txt_search.setStyleSheet(
            "border: none; background: transparent; color: #111111; font-family: 'Segoe UI'; font-size: 9pt;"
        )
        self.txt_search.textChanged.connect(self._filter_list)
        self.txt_search.returnPressed.connect(self._on_return_pressed)
        s_hbox.addWidget(self.txt_search, 1)

        lbl_mag = QLabel(search_frame)
        lbl_mag.setFixedSize(16, 16)
        lbl_mag.setPixmap(make_wlpg_icon("text_search", 14).pixmap(14, 14))
        lbl_mag.setStyleSheet("border: none; background: transparent;")
        s_hbox.addWidget(lbl_mag)
        vbox.addWidget(search_frame)

        # 3. Scrollable People List
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet(
            """
            QScrollArea {
                background: #FFFFFF;
                border: 1px solid #D5E2F0;
            }
            """
        )
        self.list_container = QWidget(self.scroll_area)
        self.list_container.setStyleSheet("background: #FFFFFF;")
        self.list_layout = QVBoxLayout(self.list_container)
        self.list_layout.setContentsMargins(2, 2, 2, 2)
        self.list_layout.setSpacing(1)
        self.list_layout.addStretch(1)
        self.scroll_area.setWidget(self.list_container)
        vbox.addWidget(self.scroll_area, 1)

    def populate_people(self, people_list: list[dict]) -> None:
        self._people_data = list(people_list)
        self.txt_search.clear()
        self._rebuild_rows("")

    populate = populate_people

    def focus_input(self) -> None:
        self.txt_search.setFocus(Qt.ActiveWindowFocusReason)
        self.txt_search.selectAll()
        QTimer.singleShot(0, lambda: (self.txt_search.setFocus(Qt.ActiveWindowFocusReason), self.txt_search.selectAll()))

    def _filter_list(self, text: str) -> None:
        self._rebuild_rows(text.strip())

    def _rebuild_rows(self, query: str) -> None:
        while self.list_layout.count() > 0:
            item = self.list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        q_lower = query.lower()
        matched = [
            p for p in self._people_data
            if not q_lower or q_lower in str(p.get("name", "")).lower()
        ]
        for p in matched:
            pname = str(p.get("name", ""))
            blob = p.get("thumbnail_blob")
            pix = make_person_avatar_pixmap(pname, blob, size=36)
            row_btn = _PersonPopupRowButton(pname, pix, self.list_container)
            row_btn.person_clicked.connect(self.person_selected.emit)
            self.list_layout.addWidget(row_btn)

        if query and not any(str(p.get("name", "")).lower() == q_lower for p in matched):
            pix_new = make_person_avatar_pixmap(query, None, size=36)
            row_new = _PersonPopupRowButton(query, pix_new, self.list_container)
            row_new.person_clicked.connect(self.person_selected.emit)
            self.list_layout.addWidget(row_new)

        self.list_layout.addStretch(1)

    def _on_return_pressed(self) -> None:
        text = self.txt_search.text().strip()
        if text:
            self.person_selected.emit(text)
            return
        # If search is empty and there is at least one person row, pick the first
        if self._people_data:
            first_name = str(self._people_data[0].get("name", "")).strip()
            if first_name:
                self.person_selected.emit(first_name)


class RecipeUndoCommand(QUndoCommand):
    """A lightweight command storing before and after PhotoAdjustmentRecipe states."""

    def __init__(
        self,
        viewer: SingleMediaViewer,
        before_recipe: PhotoAdjustmentRecipe,
        after_recipe: PhotoAdjustmentRecipe,
        description: str,
        parent=None,
    ) -> None:
        super().__init__(description, parent)
        self.viewer = viewer
        self.before_recipe = before_recipe.copy()
        self.after_recipe = after_recipe.copy()

    def undo(self) -> None:
        self.viewer.current_recipe = self.before_recipe.copy()
        self.viewer._render_current_recipe()
        self.viewer.recipe_changed.emit(self.viewer.current_recipe)

    def redo(self) -> None:
        self.viewer.current_recipe = self.after_recipe.copy()
        self.viewer._render_current_recipe()
        self.viewer.recipe_changed.emit(self.viewer.current_recipe)


class PixelArrayUndoCommand(QUndoCommand):
    """Undo/redo command for destructive raster operations (Retouch inpainting, Red Eye removal)."""

    def __init__(
        self,
        viewer: SingleMediaViewer,
        before_np: np.ndarray,
        after_np: np.ndarray,
        description: str,
        parent=None,
    ) -> None:
        super().__init__(description, parent)
        self.viewer = viewer
        self.before_np = before_np.copy()
        self.after_np = after_np.copy()

    def undo(self) -> None:
        self.viewer.master_np_rgb = self.before_np.copy()
        self.viewer._update_display_proxy()
        self.viewer._render_current_recipe()

    def redo(self) -> None:
        self.viewer.master_np_rgb = self.after_np.copy()
        self.viewer._update_display_proxy()
        self.viewer._render_current_recipe()


class SingleMediaViewer(QWidget):
    """
    Single-Image / Video Viewer and WLPG 'Fix' Adjustments Editor (`docs/reference_ui/fullsize and edit_ui.png`).
    """

    closed = Signal()
    navigate_requested = Signal(int)                              # -1 (prev) or +1 (next)
    rotate_requested = Signal(bool)                               # True = CW, False = CCW
    pixel_edit_saved = Signal(int, str)                           # (media_id, backup_path)
    pixel_edit_reverted = Signal(int)                             # media_id
    face_region_tagged = Signal(int, str, float, float, float, float)  # (media_id, name, rx, ry, rw, rh)
    zoom_percent_changed = Signal(int)
    image_changed = Signal(object)                                # emitted with current PIL Image
    recipe_changed = Signal(object)                               # emitted with current PhotoAdjustmentRecipe

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("SingleMediaViewer")
        self.setFocusPolicy(Qt.StrongFocus)

        self.current_media_id: int | None = None
        self.current_file_path: Path | None = None
        self.current_is_video: bool = False
        self.master_pil_image: Image.Image | None = None
        self.master_np_rgb: np.ndarray | None = None
        self.display_proxy_rgb: np.ndarray | None = None
        self._proxy_scale: float = 1.0
        self._is_sliding: bool = False
        self.current_pil_image: Image.Image | None = None
        self.current_recipe: PhotoAdjustmentRecipe = PhotoAdjustmentRecipe()
        self._recipe_before_slider: PhotoAdjustmentRecipe | None = None
        self.current_zoom_mode: Literal["fit", "fill", "100", "custom"] = "fit"
        self.is_crop_mode: bool = False
        self.is_face_tag_mode: bool = False

        # Retouch (Magic Eraser) state
        self.is_retouch_mode: bool = False
        self.retouch_radius: int = 20
        self.main_window: Any = None
        self._retouch_active: bool = False
        self._retouch_mask: np.ndarray | None = None
        self._retouch_preview_items: list[Any] = []
        self._retouch_last_master_pt: tuple[int, int] | None = None

        # Red Eye state
        self.is_red_eye_mode: bool = False
        self._red_eye_active: bool = False
        self._red_eye_start_pos: QPointF | None = None
        self._red_eye_item: QGraphicsRectItem | None = None

        self.undo_stack = QUndoStack(self)
        self.undo_stack.canUndoChanged.connect(lambda _: self.update_revert_state())
        self.undo_stack.cleanChanged.connect(lambda _: self.update_revert_state())
        self._face_overlay_items: list[QGraphicsItem] = []
        self._current_people_regions: list[dict] = []
        self._face_target_item: _FaceTagTargetBoxItem | None = None
        self._face_tag_replace_person_id: int | None = None
        self._drawing_face_start_pos: QPointF | None = None
        self._highlighted_person_id: int | None = None

        root_hbox = QHBoxLayout(self)
        root_hbox.setContentsMargins(0, 0, 0, 0)
        root_hbox.setSpacing(0)

        # ---------------------------------------------------------------------
        # 1. Central Stack: Page 0 = ImageGraphicsView + CropScene, Page 1 = Video Player
        # ---------------------------------------------------------------------
        self.media_stack = QStackedWidget(self)
        self.media_stack.setFrameShape(QFrame.NoFrame)

        # Page 0: Image Viewer
        self.graphics_view = ImageGraphicsView(main_window=self, parent=self)
        self.graphics_view.setFrameShape(QFrame.NoFrame)
        self.graphics_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphics_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scene = CropScene(self.graphics_view)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        self.scene.pixmap_item = self.pixmap_item
        self.scene.crop_applied.connect(self.apply_crop)
        self.scene.crop_cancelled.connect(lambda: self.set_crop_mode(False))
        self.graphics_view.setScene(self.scene)
        self.graphics_view.setStyleSheet("background-color: #FFFFFF; border: none;")
        self.graphics_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.graphics_view.customContextMenuRequested.connect(self._show_viewer_context_menu)
        self.scene.setBackgroundBrush(QBrush(QColor("#FFFFFF")))
        self.graphics_view.zoomChanged.connect(self.zoom_percent_changed.emit)
        self.graphics_view.zoomChanged.connect(lambda _: self._position_tag_someone_popup())
        self.graphics_view.zoomChanged.connect(lambda _: self._position_adjust_toolbar())
        self.graphics_view.zoomModeSetToCustom.connect(self._on_custom_zoom_set)
        if self.graphics_view.horizontalScrollBar():
            self.graphics_view.horizontalScrollBar().valueChanged.connect(
                lambda _: self._position_tag_someone_popup()
            )
            self.graphics_view.horizontalScrollBar().valueChanged.connect(
                lambda _: self._position_adjust_toolbar()
            )
        if self.graphics_view.verticalScrollBar():
            self.graphics_view.verticalScrollBar().valueChanged.connect(
                lambda _: self._position_tag_someone_popup()
            )
            self.graphics_view.verticalScrollBar().valueChanged.connect(
                lambda _: self._position_adjust_toolbar()
            )
        self.media_stack.addWidget(self.graphics_view)

        # Page 1: Video Player (Authentic WLPG White Background & Transport Bar)
        self.video_container = QWidget(self)
        self.video_container.setStyleSheet("background-color: #FFFFFF; border: none;")
        vid_vbox = QVBoxLayout(self.video_container)
        vid_vbox.setContentsMargins(0, 0, 0, 0)
        vid_vbox.setSpacing(0)

        self.video_widget = ClickableVideoWidget(self.video_container)
        pal = self.video_widget.palette()
        pal.setColor(QPalette.Window, QColor("#FFFFFF"))
        self.video_widget.setPalette(pal)
        self.video_widget.setStyleSheet("background-color: #FFFFFF; border: none;")
        self.video_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.video_widget.customContextMenuRequested.connect(self._show_video_context_menu)

        self.mouse_overlay = MouseOverlay(self.video_widget)
        self.mouse_overlay.setContextMenuPolicy(Qt.CustomContextMenu)
        self.mouse_overlay.customContextMenuRequested.connect(self._show_video_context_menu)

        self.media_player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setVideoOutput(self.video_widget)
        self.media_player.playbackStateChanged.connect(self._on_playback_state_changed)
        self.video_widget.clicked.connect(self.toggle_video_playback)
        vid_vbox.addWidget(self.video_widget, 1)

        # Video Transport Bar (WLPG Aero Styling)
        transport = QFrame(self.video_container)
        transport.setFixedHeight(38)
        transport.setStyleSheet(
            """
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F5F8FC, stop:1 #E4ECF5);
                border-top: 1px solid #B8CDE3;
                color: #1E395B;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #FFFFFF, stop:1 #EBF1F8);
                border: 1px solid #B6C7DC;
                border-radius: 3px;
                padding: 2px 10px;
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
            }
            QPushButton:hover {
                background: #EBF4FD;
                border: 1px solid #7AC1FF;
            }
            QLabel {
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                background: transparent;
                border: none;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #CAD8E6;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #509BE6;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #0078D7;
                border: 1px solid #005A9E;
                width: 10px;
                margin: -5px 0;
                border-radius: 2px;
            }
            """
        )
        t_hbox = QHBoxLayout(transport)
        t_hbox.setContentsMargins(8, 2, 8, 2)

        self.btn_play_pause = QPushButton("▶ Play", transport)
        self.btn_play_pause.clicked.connect(self.toggle_video_playback)
        t_hbox.addWidget(self.btn_play_pause)

        self.slider_position = QSlider(Qt.Horizontal, transport)
        self.slider_position.setRange(0, 1000)
        self.slider_position.sliderMoved.connect(self.media_player.setPosition)
        self.media_player.positionChanged.connect(self._on_video_position_changed)
        self.media_player.durationChanged.connect(self._on_video_duration_changed)
        t_hbox.addWidget(self.slider_position, 1)

        self.lbl_video_time = QLabel("00:00 / 00:00", transport)
        t_hbox.addWidget(self.lbl_video_time)

        vid_vbox.addWidget(transport)
        self.media_stack.addWidget(self.video_container)

        root_hbox.addWidget(self.media_stack, 1)

        # ---------------------------------------------------------------------
        # 2. Right Collapsible 'Fix / Fine-Tune' Adjustments Drawer
        # ---------------------------------------------------------------------
        self.fix_pane = QFrame(self)
        self.fix_pane.setObjectName("FixAdjustmentsDrawer")
        self.fix_pane.setFixedWidth(255)
        self.fix_pane.setStyleSheet(
            """
            QFrame#FixAdjustmentsDrawer {
                background-color: #F5F7FA;
                border-left: 1px solid #C8D2E0;
            }
            """
        )
        fix_vbox = QVBoxLayout(self.fix_pane)
        fix_vbox.setContentsMargins(10, 10, 10, 10)
        fix_vbox.setSpacing(8)

        lbl_fix_title = QLabel("Fine-Tune & Fix", self.fix_pane)
        lbl_fix_title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        lbl_fix_title.setStyleSheet("color: #1E395B;")
        fix_vbox.addWidget(lbl_fix_title)

        # Auto Adjust
        self.btn_auto_adjust = QPushButton("✨ Auto adjust", self.fix_pane)
        self.btn_auto_adjust.clicked.connect(self.apply_auto_adjust)
        fix_vbox.addWidget(self.btn_auto_adjust)

        # Crop Controls
        crop_row = QHBoxLayout()
        self.btn_crop_start = QPushButton("Crop photo", self.fix_pane)
        self.btn_crop_start.setCheckable(True)
        self.btn_crop_start.toggled.connect(self.set_crop_mode)
        crop_row.addWidget(self.btn_crop_start)

        self.btn_crop_apply = QPushButton("Apply crop", self.fix_pane)
        self.btn_crop_apply.clicked.connect(self.apply_crop)
        self.btn_crop_apply.setEnabled(False)
        crop_row.addWidget(self.btn_crop_apply)
        fix_vbox.addLayout(crop_row)

        # Exposure & Contrast Sliders
        fix_vbox.addWidget(QLabel("Brightness:", self.fix_pane))
        self.slider_brightness = QSlider(Qt.Horizontal, self.fix_pane)
        self.slider_brightness.setRange(-100, 100)
        self.slider_brightness.setValue(0)
        fix_vbox.addWidget(self.slider_brightness)

        fix_vbox.addWidget(QLabel("Contrast:", self.fix_pane))
        self.slider_contrast = QSlider(Qt.Horizontal, self.fix_pane)
        self.slider_contrast.setRange(-100, 100)
        self.slider_contrast.setValue(0)
        fix_vbox.addWidget(self.slider_contrast)

        self.btn_apply_exposure = QPushButton("Apply exposure", self.fix_pane)
        self.btn_apply_exposure.clicked.connect(
            lambda: self.apply_exposure_contrast(
                self.slider_brightness.value(), self.slider_contrast.value()
            )
        )
        fix_vbox.addWidget(self.btn_apply_exposure)

        # Color Saturation Slider
        fix_vbox.addWidget(QLabel("Saturation:", self.fix_pane))
        self.slider_saturation = QSlider(Qt.Horizontal, self.fix_pane)
        self.slider_saturation.setRange(-100, 100)
        self.slider_saturation.setValue(0)
        fix_vbox.addWidget(self.slider_saturation)

        self.btn_apply_color = QPushButton("Apply color", self.fix_pane)
        self.btn_apply_color.clicked.connect(
            lambda: self.apply_color_saturation(self.slider_saturation.value())
        )
        fix_vbox.addWidget(self.btn_apply_color)

        # Straighten Slider (-15..+15 deg)
        fix_vbox.addWidget(QLabel("Straighten (-15° to +15°):", self.fix_pane))
        self.slider_straighten = QSlider(Qt.Horizontal, self.fix_pane)
        self.slider_straighten.setRange(-15, 15)
        self.slider_straighten.setValue(0)
        fix_vbox.addWidget(self.slider_straighten)

        self.btn_apply_straighten = QPushButton("Apply straighten", self.fix_pane)
        self.btn_apply_straighten.clicked.connect(
            lambda: self.apply_straighten(float(self.slider_straighten.value()))
        )
        fix_vbox.addWidget(self.btn_apply_straighten)

        # Preset Effects (B&W, Sepia, Warm, Cool)
        fix_vbox.addWidget(QLabel("Effects:", self.fix_pane))
        fx_row1 = QHBoxLayout()
        for fx_key, fx_lbl in [("bw", "B&W"), ("sepia", "Sepia"), ("warm", "Warm"), ("cool", "Cool")]:
            b_fx = QPushButton(fx_lbl, self.fix_pane)
            b_fx.clicked.connect(lambda _c=False, k=fx_key: self.apply_preset_effect(k))
            fx_row1.addWidget(b_fx)
        fix_vbox.addLayout(fx_row1)

        # Undo / Redo Row
        undo_row = QHBoxLayout()
        self.btn_undo = QPushButton("↶ Undo", self.fix_pane)
        self.btn_undo.clicked.connect(self.undo_stack.undo)
        undo_row.addWidget(self.btn_undo)

        self.btn_redo = QPushButton("↷ Redo", self.fix_pane)
        self.btn_redo.clicked.connect(self.undo_stack.redo)
        undo_row.addWidget(self.btn_redo)
        fix_vbox.addLayout(undo_row)

        # Save Pixel Edit (with .OriginalFiles backup) & Revert to Original
        self.btn_save_edit = QPushButton("💾 Save changes", self.fix_pane)
        self.btn_save_edit.setStyleSheet("font-weight: bold; color: #005A9E;")
        self.btn_save_edit.clicked.connect(self.save_current_image_with_backup)
        fix_vbox.addWidget(self.btn_save_edit)

        self.btn_revert = QPushButton("Revert to original", self.fix_pane)
        self.btn_revert.setEnabled(False)
        self.btn_revert.clicked.connect(lambda: self.revert_to_original())
        fix_vbox.addWidget(self.btn_revert)

        fix_vbox.addStretch(1)

        self.btn_close_viewer = QPushButton("✕ Close file (Esc)", self.fix_pane)
        self.btn_close_viewer.clicked.connect(self.closed.emit)
        fix_vbox.addWidget(self.btn_close_viewer)

        # (Fix pane controls are hosted exclusively in the right sidebar dock under Ctrl+F)
        # root_hbox contains only media_stack to give the photo/video the full viewport width
        self.fix_pane.hide()

        # Keyboard Navigation & Shortcut Event Filters
        self.installEventFilter(self)
        self.graphics_view.installEventFilter(self)
        if self.graphics_view.viewport():
            self.graphics_view.viewport().installEventFilter(self)
        self.video_widget.installEventFilter(self)
        self.mouse_overlay.installEventFilter(self)

        # Floating toolbars over graphics_view
        self._init_floating_toolbars()

    # -------------------------------------------------------------------------
    # Floating Contextual Toolbars (Crop Presets, Retouch Size, Red Eye, Tag Someone)
    # -------------------------------------------------------------------------
    def _init_floating_toolbars(self) -> None:
        tb_style = """
            QFrame {
                background-color: rgba(245, 248, 252, 240);
                border: 1px solid #7DA2CE;
                border-radius: 6px;
                padding: 2px 6px;
            }
            QLabel {
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 11px;
                font-weight: bold;
            }
            QComboBox {
                background: #FFFFFF;
                border: 1px solid #7DA2CE;
                border-radius: 3px;
                padding: 2px 6px;
                color: #1E395B;
                font-size: 11px;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #FBFDFF, stop:1 #E0ECF8);
                border: 1px solid #7DA2CE;
                border-radius: 3px;
                color: #1E395B;
                padding: 3px 10px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #EAF4FD, stop:1 #C6E0F8);
                border: 1px solid #3C7FB1;
            }
        """

        # 1. Crop Toolbar
        self.crop_toolbar = QFrame(self.graphics_view)
        self.crop_toolbar.setStyleSheet(tb_style)
        ct_layout = QHBoxLayout(self.crop_toolbar)
        ct_layout.setContentsMargins(6, 4, 6, 4)
        ct_layout.setSpacing(8)

        ct_lbl = QLabel("Crop Proportion:", self.crop_toolbar)
        self.combo_crop_ratio = QComboBox(self.crop_toolbar)
        self.combo_crop_ratio.addItems([
            "Custom",
            "Original",
            "1:1 (Square)",
            "3:2",
            "4:3",
            "5:4",
            "16:9",
        ])
        self.combo_crop_ratio.currentTextChanged.connect(self._on_crop_ratio_changed)

        btn_crop_apply = QPushButton("✓ Apply (Enter)", self.crop_toolbar)
        btn_crop_apply.clicked.connect(self.apply_crop)

        btn_crop_cancel = QPushButton("✕ Cancel (Esc)", self.crop_toolbar)
        btn_crop_cancel.clicked.connect(lambda: self.set_crop_mode(False))

        ct_layout.addWidget(ct_lbl)
        ct_layout.addWidget(self.combo_crop_ratio)
        ct_layout.addWidget(btn_crop_apply)
        ct_layout.addWidget(btn_crop_cancel)
        self.crop_toolbar.hide()

        # 2. Retouch Toolbar
        self.retouch_toolbar = QFrame(self.graphics_view)
        self.retouch_toolbar.setStyleSheet(tb_style)
        rt_layout = QHBoxLayout(self.retouch_toolbar)
        rt_layout.setContentsMargins(6, 4, 6, 4)
        rt_layout.setSpacing(8)

        rt_lbl = QLabel("Retouch (Magic Eraser):", self.retouch_toolbar)
        rt_hint = QLabel("Drag over blemishes to heal", self.retouch_toolbar)
        rt_hint.setStyleSheet("font-weight: normal; color: #555555;")

        lbl_brush = QLabel("Size:", self.retouch_toolbar)
        self.slider_retouch_size = QSlider(Qt.Horizontal, self.retouch_toolbar)
        self.slider_retouch_size.setRange(8, 60)
        self.slider_retouch_size.setValue(20)
        self.slider_retouch_size.setFixedWidth(80)
        self.slider_retouch_size.valueChanged.connect(self._on_retouch_size_changed)

        btn_retouch_done = QPushButton("Done (Esc)", self.retouch_toolbar)
        btn_retouch_done.clicked.connect(lambda: self.set_retouch_mode(False))

        rt_layout.addWidget(rt_lbl)
        rt_layout.addWidget(rt_hint)
        rt_layout.addWidget(lbl_brush)
        rt_layout.addWidget(self.slider_retouch_size)
        rt_layout.addWidget(btn_retouch_done)
        self.retouch_toolbar.hide()

        # 3. Red Eye Toolbar
        self.red_eye_toolbar = QFrame(self.graphics_view)
        self.red_eye_toolbar.setStyleSheet(tb_style)
        re_layout = QHBoxLayout(self.red_eye_toolbar)
        re_layout.setContentsMargins(6, 4, 6, 4)
        re_layout.setSpacing(8)

        re_lbl = QLabel("Fix Red Eye:", self.red_eye_toolbar)
        re_hint = QLabel("Drag a small box over the red pupil", self.red_eye_toolbar)
        re_hint.setStyleSheet("font-weight: normal; color: #555555;")

        btn_re_done = QPushButton("Done (Esc)", self.red_eye_toolbar)
        btn_re_done.clicked.connect(lambda: self.set_red_eye_mode(False))

        re_layout.addWidget(re_lbl)
        re_layout.addWidget(re_hint)
        re_layout.addWidget(btn_re_done)
        self.red_eye_toolbar.hide()

        # 4. WLPG Floating "Tag someone" Popup
        self._tag_someone_popup = _TagSomeonePopup(self.graphics_view)
        self._tag_someone_popup.person_selected.connect(self._on_tag_someone_selected)
        self._tag_someone_popup.cancelled.connect(self.stop_face_tagging_mode)
        self._tag_someone_popup.hide()

        # 5. Floating Face Adjust Toolbar
        self.is_face_adjust_mode: bool = False
        self._adjusting_face_info: dict = {}
        self._face_adjust_box_item: _FaceTagTargetBoxItem | None = None
        self._adjust_toolbar = _FaceAdjustToolbar(self.graphics_view)
        self._adjust_toolbar.saved.connect(self.save_face_adjust)
        self._adjust_toolbar.cancelled.connect(self.stop_face_adjust_mode)
        self._adjust_toolbar.hide()

        # 6. Markup & Privacy Redact Toolbar
        self.is_markup_mode: bool = False
        self._markup_elements: list[MarkupElement] = []
        self._markup_preview_items: list[Any] = []
        self._markup_active: bool = False
        self._markup_start_pt: QPointF | None = None
        self._markup_current_item: Any | None = None

        self.markup_toolbar = MarkupFloatingToolbar(self.graphics_view)
        self.markup_toolbar.apply_requested.connect(self._commit_markup)
        self.markup_toolbar.cancel_requested.connect(lambda: self.set_markup_mode(False))
        self.markup_toolbar.undo_requested.connect(self._undo_markup)
        self.markup_toolbar.hide()

    def _position_floating_toolbars(self) -> None:
        gw = self.graphics_view.width()
        for tb in (self.crop_toolbar, self.retouch_toolbar, self.red_eye_toolbar, self.markup_toolbar):
            if tb.isVisible():
                sh = tb.sizeHint()
                tb.setGeometry(max(10, (gw - sh.width()) // 2), 12, sh.width(), sh.height())
        self._position_tag_someone_popup()
        self._position_adjust_toolbar()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._position_floating_toolbars()

    def _update_display_proxy(self) -> None:
        if self.master_np_rgb is None:
            self.display_proxy_rgb = None
            self._proxy_scale = 1.0
            return
        h, w = self.master_np_rgb.shape[:2]
        max_dim = 1600
        if max(h, w) > max_dim:
            self._proxy_scale = max_dim / float(max(h, w))
            pw = max(1, int(round(w * self._proxy_scale)))
            ph = max(1, int(round(h * self._proxy_scale)))
            self.display_proxy_rgb = cv2.resize(self.master_np_rgb, (pw, ph), interpolation=cv2.INTER_AREA)
        else:
            self.display_proxy_rgb = self.master_np_rgb
            self._proxy_scale = 1.0

    def _scene_to_master(self, sx: float, sy: float) -> tuple[int, int]:
        if self.master_np_rgb is None:
            return int(sx), int(sy)
        mh, mw = self.master_np_rgb.shape[:2]
        if self.current_recipe.crop_rect is not None:
            l, t, r, b = self.current_recipe.crop_rect
            mx = int(round(l + sx))
            my = int(round(t + sy))
        else:
            mx = int(round(sx))
            my = int(round(sy))
        return max(0, min(mw - 1, mx)), max(0, min(mh - 1, my))

    def rotate_media(self, clockwise: bool = True) -> None:
        """Rotates the media currently viewed in the SingleMediaViewer."""
        self.rotate_requested.emit(clockwise)

    def apply_in_memory_rotation(self, clockwise: bool = True) -> None:
        """Immediately rotates current master image in memory (0ms lag, no disk reload)."""
        if self.master_np_rgb is None or self.current_is_video:
            return
        k = -1 if clockwise else 1
        self.master_np_rgb = np.ascontiguousarray(np.rot90(self.master_np_rgb, k))
        self.master_pil_image = Image.fromarray(self.master_np_rgb)
        self.current_recipe.crop_rect = None
        self._update_display_proxy()
        self._render_current_recipe()

    # -------------------------------------------------------------------------
    # Loading & Display
    # -------------------------------------------------------------------------
    def load_media(
        self,
        media_id: int,
        file_path: str | Path,
        is_video: bool = False,
        people_regions: list[dict] | None = None,
        orientation: int = 1,
    ) -> bool:
        """Loads a static image or video into the SingleMediaViewer."""
        self.current_media_id = int(media_id)
        self.current_file_path = Path(file_path)
        self.current_is_video = bool(is_video)
        self.undo_stack.clear()
        self.stop_face_tagging_mode()
        self.set_crop_mode(False)
        self.set_retouch_mode(False)
        self.set_red_eye_mode(False)
        self._clear_face_overlays()

        if self.current_is_video:
            self.media_stack.setCurrentIndex(1)
            self.fix_pane.setVisible(False)
            self.media_player.setSource(QUrl.fromLocalFile(str(self.current_file_path)))
            self.media_player.play()
            self.btn_play_pause.setText("⏸ Pause")
            return True

        self.media_player.stop()
        self.media_stack.setCurrentIndex(0)
        self.fix_pane.setVisible(False)

        if not self.current_file_path.exists():
            return False

        ext = self.current_file_path.suffix.lstrip(".").lower()
        loaded = False
        if ext in ("raf", "cr2", "cr3", "nef", "arw", "dng", "orf", "rw2"):
            try:
                import rawpy
                with rawpy.imread(str(self.current_file_path)) as raw:
                    rgb = raw.postprocess(use_camera_wb=True)
                    self.master_np_rgb = np.ascontiguousarray(rgb)
                    self.master_pil_image = Image.fromarray(self.master_np_rgb)
                    loaded = True
            except Exception:
                loaded = False

        if not loaded:
            try:
                with Image.open(self.current_file_path) as img:
                    disk_exif = img.getexif()
                    disk_orient = int(disk_exif.get(0x0112) or 1)
                    transposed = ImageOps.exif_transpose(img)
                    target_orient = int(orientation or 1)
                    orient_to_angle = {1: 0, 6: 90, 3: 180, 8: 270}
                    delta_angle = (
                        orient_to_angle.get(target_orient, 0)
                        - orient_to_angle.get(disk_orient, 0)
                    ) % 360
                    if delta_angle != 0:
                        k = {90: -1, 180: 2, 270: 1}[delta_angle]
                        transposed_np = np.ascontiguousarray(
                            np.rot90(np.array(transposed.convert("RGB")), k)
                        )
                        self.master_pil_image = Image.fromarray(transposed_np)
                        self.master_np_rgb = transposed_np
                    else:
                        self.master_pil_image = transposed.convert("RGB")
                        self.master_np_rgb = np.array(self.master_pil_image)
            except Exception:
                return False

        self._update_display_proxy()
        self.current_recipe = PhotoAdjustmentRecipe()
        self.undo_stack.clear()
        self._render_current_recipe()
        self.set_zoom_mode("fit")
        self.recipe_changed.emit(self.current_recipe)

        # Draw face regions strictly for static images (is_video == 0)
        if people_regions and not self.current_is_video:
            self.render_face_regions(people_regions)
        self.update_revert_state()
        return True

    def _render_current_recipe(self) -> None:
        """Renders self.master_np_rgb with self.current_recipe using OpenCV/NumPy float pipeline."""
        if self.master_np_rgb is None:
            return

        if self._is_sliding and self.display_proxy_rgb is not None:
            proxy_rec = self.current_recipe.copy()
            if proxy_rec.crop_rect is not None and self._proxy_scale < 0.999:
                l, t, r, b = proxy_rec.crop_rect
                proxy_rec.crop_rect = (
                    int(round(l * self._proxy_scale)),
                    int(round(t * self._proxy_scale)),
                    int(round(r * self._proxy_scale)),
                    int(round(b * self._proxy_scale)),
                )
            rendered_np = render_adjusted_image(self.display_proxy_rgb, proxy_rec)
            h, w = rendered_np.shape[:2]
            qimg = QImage(rendered_np.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            pix = QPixmap.fromImage(qimg)
            target_w = int(round(w / self._proxy_scale))
            target_h = int(round(h / self._proxy_scale))
            if target_w != w or target_h != h:
                pix = pix.scaled(target_w, target_h, Qt.IgnoreAspectRatio, Qt.FastTransformation)
            self.pixmap_item.setPixmap(pix)
            return

        rendered_np = render_adjusted_image(self.master_np_rgb, self.current_recipe)
        self.current_pil_image = Image.fromarray(rendered_np)

        h, w, c = rendered_np.shape
        bytes_per_line = 3 * w
        qimg = QImage(rendered_np.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        self._adjusted_pixmap = pix
        self.pixmap_item.setPixmap(pix)
        self.scene.setSceneRect(QRectF(0, 0, pix.width(), pix.height()))
        self.scene.crop_rect_item.set_image_rect(QRectF(0, 0, pix.width(), pix.height()))
        self.image_changed.emit(self.current_pil_image)

        # Cache unedited original pixmap for instantaneous, high-fidelity compare
        if self.current_recipe.is_default():
            self._original_pixmap = self._adjusted_pixmap
        else:
            default_rec = PhotoAdjustmentRecipe(crop_rect=self.current_recipe.crop_rect)
            orig_np = render_adjusted_image(self.master_np_rgb, default_rec)
            oh, ow = orig_np.shape[:2]
            orig_qimg = QImage(orig_np.data, ow, oh, 3 * ow, QImage.Format_RGB888).copy()
            self._original_pixmap = QPixmap.fromImage(orig_qimg)

    def update_image_display(self) -> None:
        """Legacy helper for test compatibility."""
        self._render_current_recipe()

    # -------------------------------------------------------------------------
    # Zoom & Crop Modes
    # -------------------------------------------------------------------------
    def _on_custom_zoom_set(self) -> None:
        self.current_zoom_mode = "custom"

    def set_zoom_mode(self, mode: Literal["fit", "fill", "100", "custom"]) -> None:
        self.current_zoom_mode = mode
        if mode == "fit":
            self.graphics_view.fit_in_view_with_zoom()
        elif mode == "fill":
            self.graphics_view.fill_in_view_with_zoom()
        elif mode == "100":
            self.graphics_view.set_zoom(1.0)

    def zoom_in(self) -> None:
        self.current_zoom_mode = "custom"
        self.graphics_view.zoom(1.25)

    def zoom_out(self) -> None:
        self.current_zoom_mode = "custom"
        self.graphics_view.zoom(0.8)

    def is_in_edit_mode(self) -> bool:
        return bool(
            getattr(self, "is_crop_mode", False)
            or getattr(self, "is_retouch_mode", False)
            or getattr(self, "is_red_eye_mode", False)
            or getattr(self, "is_straighten_active", False)
        )

    def _update_face_overlays_visibility(self) -> None:
        in_edit = self.is_in_edit_mode()
        for item in getattr(self, "_face_overlay_items", []):
            if isinstance(item, _FaceHoverRegionItem):
                item.setEnabled(not in_edit)
            else:
                if in_edit:
                    item.setVisible(False)

    def set_crop_mode(self, enabled: bool) -> None:
        self.is_crop_mode = bool(enabled)
        if hasattr(self, "scene") and hasattr(self.scene, "set_crop_mode"):
            self.scene.set_crop_mode(self.is_crop_mode)
        self.btn_crop_start.blockSignals(True)
        self.btn_crop_start.setChecked(self.is_crop_mode)
        self.btn_crop_start.blockSignals(False)
        self.btn_crop_apply.setEnabled(self.is_crop_mode)

        if self.is_crop_mode:
            self.stop_face_tagging_mode()
            self.set_retouch_mode(False)
            self.set_red_eye_mode(False)
            self._update_face_overlays_visibility()
            self.graphics_view.setDragMode(ImageGraphicsView.NoDrag)
            self.crop_toolbar.show()
            self._position_floating_toolbars()
            self.crop_toolbar.raise_()

            scale = getattr(self, "_proxy_scale", 1.0)
            master_scale = (1.0 / scale) if scale > 0 else 1.0
            self.scene.crop_rect_item.set_master_scale(master_scale)

            if self.scene.get_crop_rect() is None and self.current_pil_image is not None:
                w, h = self.current_pil_image.size
                cw, ch = w * 0.85, h * 0.85
                self.scene.crop_rect_item.setRect(QRectF((w - cw) / 2.0, (h - ch) / 2.0, cw, ch))
                self.scene.crop_rect_item.show()
        else:
            self.crop_toolbar.hide()
            self.scene.clear_crop_rect()
            if self.pixmap_item:
                self.pixmap_item.setRotation(0.0)
            self._update_face_overlays_visibility()
            if not self.is_retouch_mode and not self.is_red_eye_mode and not self.is_face_tag_mode:
                self.graphics_view.viewport().unsetCursor()
                self.graphics_view.setDragMode(ImageGraphicsView.ScrollHandDrag)

    def _on_crop_ratio_changed(self, text: str) -> None:
        if self.current_pil_image is None:
            return
        w, h = self.current_pil_image.size
        if w <= 0 or h <= 0:
            return

        if text == "Custom":
            self.scene.crop_rect_item.set_aspect_ratio(None)
            self.scene.crop_rect_updated.emit()
            return

        target_ratio: float
        if text == "Original":
            target_ratio = float(w) / float(h)
        elif text == "1:1 (Square)":
            target_ratio = 1.0
        elif text == "3:2":
            target_ratio = 3.0 / 2.0
        elif text == "4:3":
            target_ratio = 4.0 / 3.0
        elif text == "5:4":
            target_ratio = 5.0 / 4.0
        elif text == "16:9":
            target_ratio = 16.0 / 9.0
        else:
            return

        self.scene.crop_rect_item.set_aspect_ratio(target_ratio)
        self.scene.crop_rect_updated.emit()

    def apply_crop(self) -> bool:
        """Applies the active CropScene rectangle and rotation angle non-destructively to the recipe."""
        if self.master_np_rgb is None:
            return False
        rect = self.scene.get_crop_rect()
        if rect is None or rect.width() < 4 or rect.height() < 4:
            return False

        # Reset any live preview rotation on pixmap item
        rot_deg = getattr(self.scene, "crop_rotation_deg", 0.0)
        if self.pixmap_item:
            self.pixmap_item.setRotation(0.0)

        curr_w, curr_h = (
            self.current_pil_image.size
            if self.current_pil_image
            else (int(rect.right()), int(rect.bottom()))
        )
        left = max(0, min(curr_w - 1, int(round(rect.left()))))
        top = max(0, min(curr_h - 1, int(round(rect.top()))))
        right = max(left + 1, min(curr_w, int(round(rect.right()))))
        bottom = max(top + 1, min(curr_h, int(round(rect.bottom()))))

        before = self.current_recipe.copy()
        if abs(rot_deg) >= 0.05:
            master_crop = (left, top, right, bottom)
            self.current_recipe.straighten_deg = float(self.current_recipe.straighten_deg + rot_deg)
        elif self.current_recipe.crop_rect is not None:
            prev_l, prev_t, _, _ = self.current_recipe.crop_rect
            master_crop = (prev_l + left, prev_t + top, prev_l + right, prev_t + bottom)
        else:
            master_crop = (left, top, right, bottom)
        self.current_recipe.crop_rect = master_crop
        cmd = RecipeUndoCommand(self, before, self.current_recipe.copy(), "Crop Photo")
        self.undo_stack.push(cmd)
        self.set_crop_mode(False)
        self._render_current_recipe()
        self.set_zoom_mode("fit")
        self.recipe_changed.emit(self.current_recipe)
        return True

    # -------------------------------------------------------------------------
    # Retouch (Magic Eraser / Inpainting)
    # -------------------------------------------------------------------------
    def set_retouch_mode(self, enabled: bool) -> None:
        self.is_retouch_mode = bool(enabled)
        if self.is_retouch_mode:
            self.stop_face_tagging_mode()
            self.set_crop_mode(False)
            self.set_red_eye_mode(False)
            self._update_face_overlays_visibility()
            self.graphics_view.setDragMode(ImageGraphicsView.NoDrag)
            self.graphics_view.viewport().setCursor(Qt.CrossCursor)
            self.retouch_toolbar.show()
            self._position_floating_toolbars()
            self.retouch_toolbar.raise_()
        else:
            self.retouch_toolbar.hide()
            self._clear_retouch_overlay()
            self._update_face_overlays_visibility()
            if not self.is_crop_mode and not self.is_red_eye_mode:
                self.graphics_view.viewport().unsetCursor()
                self.graphics_view.setDragMode(ImageGraphicsView.ScrollHandDrag)

    def _on_retouch_size_changed(self, val: int) -> None:
        self.retouch_radius = max(6, int(val))

    def _clear_retouch_overlay(self) -> None:
        for it in self._retouch_preview_items:
            try:
                self.scene.removeItem(it)
            except Exception:
                pass
        self._retouch_preview_items.clear()
        self._retouch_mask = None
        self._retouch_active = False

    def _start_retouch(self, scene_pos: QPointF) -> None:
        if self.master_np_rgb is None:
            return
        mx, my = self._scene_to_master(scene_pos.x(), scene_pos.y())
        self._retouch_mask = np.zeros(self.master_np_rgb.shape[:2], dtype=np.uint8)
        rad = max(4, self.retouch_radius)
        cv2.circle(self._retouch_mask, (mx, my), rad, 255, -1)
        c_item = self.scene.addEllipse(
            scene_pos.x() - rad,
            scene_pos.y() - rad,
            rad * 2,
            rad * 2,
            QPen(QColor("#FF9900"), 1),
            QBrush(QColor(255, 153, 0, 110)),
        )
        self._retouch_preview_items.append(c_item)
        self._retouch_last_master_pt = (mx, my)
        self._retouch_active = True

    def _drag_retouch(self, scene_pos: QPointF) -> None:
        if not self._retouch_active or self._retouch_mask is None or self.master_np_rgb is None:
            return
        mx, my = self._scene_to_master(scene_pos.x(), scene_pos.y())
        rad = max(4, self.retouch_radius)
        if self._retouch_last_master_pt is not None:
            cv2.line(self._retouch_mask, self._retouch_last_master_pt, (mx, my), 255, thickness=rad * 2)
        cv2.circle(self._retouch_mask, (mx, my), rad, 255, -1)
        c_item = self.scene.addEllipse(
            scene_pos.x() - rad,
            scene_pos.y() - rad,
            rad * 2,
            rad * 2,
            QPen(Qt.NoPen),
            QBrush(QColor(255, 153, 0, 110)),
        )
        self._retouch_preview_items.append(c_item)
        self._retouch_last_master_pt = (mx, my)

    def _finish_retouch(self) -> None:
        if not self._retouch_active or self._retouch_mask is None or self.master_np_rgb is None:
            self._clear_retouch_overlay()
            return
        self._retouch_active = False

        # Inpaint with OpenCV Telea algorithm
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_dilated = cv2.dilate(self._retouch_mask, kernel, iterations=1)
        before = self.master_np_rgb.copy()
        inpainted = cv2.inpaint(
            self.master_np_rgb,
            mask_dilated,
            inpaintRadius=max(3, self.retouch_radius // 2),
            flags=cv2.INPAINT_TELEA,
        )
        self.master_np_rgb = inpainted
        self._update_display_proxy()

        self._clear_retouch_overlay()
        cmd = PixelArrayUndoCommand(self, before, self.master_np_rgb.copy(), "Retouch Blemish")
        self.undo_stack.push(cmd)
        self._render_current_recipe()
        if self.current_media_id is not None:
            self.pixel_edit_saved.emit(self.current_media_id, "")

    # -------------------------------------------------------------------------
    # Red Eye Removal
    # -------------------------------------------------------------------------
    def set_red_eye_mode(self, enabled: bool) -> None:
        self.is_red_eye_mode = bool(enabled)
        if self.is_red_eye_mode:
            self.stop_face_tagging_mode()
            self.set_crop_mode(False)
            self.set_retouch_mode(False)
            self._update_face_overlays_visibility()
            self.graphics_view.setDragMode(ImageGraphicsView.NoDrag)
            self.graphics_view.viewport().setCursor(Qt.CrossCursor)
            self.red_eye_toolbar.show()
            self._position_floating_toolbars()
            self.red_eye_toolbar.raise_()
        else:
            self.red_eye_toolbar.hide()
            if self._red_eye_item is not None:
                try:
                    self.scene.removeItem(self._red_eye_item)
                except Exception:
                    pass
                self._red_eye_item = None
            self._update_face_overlays_visibility()
            if not self.is_crop_mode and not self.is_retouch_mode:
                self.graphics_view.viewport().unsetCursor()
                self.graphics_view.setDragMode(ImageGraphicsView.ScrollHandDrag)

    def _start_red_eye(self, scene_pos: QPointF) -> None:
        self._red_eye_start_pos = scene_pos
        self._red_eye_active = True
        self._red_eye_item = self.scene.addRect(
            QRectF(scene_pos, scene_pos),
            QPen(QColor("#00FFCC"), 2, Qt.DashLine),
            QBrush(QColor(0, 255, 204, 30)),
        )

    def _drag_red_eye(self, scene_pos: QPointF) -> None:
        if not self._red_eye_active or self._red_eye_item is None or self._red_eye_start_pos is None:
            return
        rect = QRectF(self._red_eye_start_pos, scene_pos).normalized()
        self._red_eye_item.setRect(rect)

    def _finish_red_eye(self) -> None:
        if not self._red_eye_active or self._red_eye_item is None or self.master_np_rgb is None:
            self._red_eye_active = False
            return
        self._red_eye_active = False
        rect = self._red_eye_item.rect()
        try:
            self.scene.removeItem(self._red_eye_item)
        except Exception:
            pass
        self._red_eye_item = None

        if rect.width() < 3 or rect.height() < 3:
            return

        mx1, my1 = self._scene_to_master(rect.left(), rect.top())
        mx2, my2 = self._scene_to_master(rect.right(), rect.bottom())
        x1, x2 = min(mx1, mx2), max(mx1, mx2)
        y1, y2 = min(my1, my2), max(my1, my2)
        if x2 <= x1 or y2 <= y1:
            return

        patch = self.master_np_rgb[y1:y2, x1:x2].astype(np.float32)
        r = patch[:, :, 0]
        g = patch[:, :, 1]
        b = patch[:, :, 2]

        # Saliency and difference metrics to detect pupil reflection vs human skin
        gb_sum = g + b + 1.0
        max_gb = np.maximum(g, b)
        r_ratio = r / gb_sum
        r_diff = r - max_gb

        # High-confidence red pupil criteria
        is_red = (r > 70.0) & (r_ratio >= 1.25) & (r_diff >= 30.0)
        if not np.any(is_red):
            is_red = (r > 60.0) & (r_ratio >= 1.15) & (r_diff >= 20.0)

        if not np.any(is_red):
            return

        # Connected component analysis: isolate the pupil nearest to the box center
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            is_red.astype(np.uint8) * 255, connectivity=8
        )
        if num_labels <= 1:
            return

        box_center = np.array([patch.shape[0] / 2.0, patch.shape[1] / 2.0])
        best_label = 1
        best_dist = float("inf")
        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area < 2:
                continue
            cy, cx = centroids[lbl][1], centroids[lbl][0]
            dist = np.hypot(cy - box_center[0], cx - box_center[1])
            if dist < best_dist:
                best_dist = dist
                best_label = lbl

        pupil_mask = (labels == best_label).astype(np.uint8) * 255

        # Morphological close to bridge tiny specular highlights inside the pupil
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        pupil_mask = cv2.morphologyEx(pupil_mask, cv2.MORPH_CLOSE, k_close)

        # Smooth feathering along pupil perimeter so surrounding iris and skin blend naturally
        ksize = 5 if min(patch.shape[:2]) >= 10 else 3
        alpha = cv2.GaussianBlur(pupil_mask.astype(np.float32) / 255.0, (ksize, ksize), 1.0)
        alpha = np.clip(alpha, 0.0, 1.0)[:, :, np.newaxis]

        if np.max(alpha) < 0.05:
            return

        before = self.master_np_rgb.copy()
        # Natural neutral dark pupil (desaturated dark tone without any color cast)
        target_dark = np.clip(np.minimum(g, b) * 0.55, 12.0, 42.0)
        target_dark = np.repeat(target_dark[:, :, np.newaxis], 3, axis=2)

        # Seamless blend: non-pupil skin has alpha == 0 so remains completely untouched
        new_patch = patch * (1.0 - alpha) + target_dark * alpha
        self.master_np_rgb[y1:y2, x1:x2] = np.clip(new_patch, 0, 255).astype(np.uint8)

        self._update_display_proxy()
        cmd = PixelArrayUndoCommand(self, before, self.master_np_rgb.copy(), "Fix Red Eye")
        self.undo_stack.push(cmd)
        self._render_current_recipe()
        if self.current_media_id is not None:
            self.pixel_edit_saved.emit(self.current_media_id, "")

    # -------------------------------------------------------------------------
    # Quick Markup, Callouts & Privacy Redaction Tool
    # -------------------------------------------------------------------------
    def set_markup_mode(self, enabled: bool) -> None:
        self.is_markup_mode = bool(enabled)
        if self.is_markup_mode:
            self.stop_face_tagging_mode()
            self.set_crop_mode(False)
            self.set_retouch_mode(False)
            self.set_red_eye_mode(False)
            self._update_face_overlays_visibility()
            self.graphics_view.setDragMode(ImageGraphicsView.NoDrag)
            self.graphics_view.viewport().setCursor(Qt.CrossCursor)
            self.markup_toolbar.show()
            self._position_floating_toolbars()
            self.markup_toolbar.raise_()
        else:
            self.markup_toolbar.hide()
            self._clear_markup_overlays()
            self._update_face_overlays_visibility()
            if not self.is_crop_mode and not self.is_retouch_mode and not self.is_red_eye_mode:
                self.graphics_view.viewport().unsetCursor()
                self.graphics_view.setDragMode(ImageGraphicsView.ScrollHandDrag)

    def _start_markup(self, sp: QPointF) -> None:
        self._markup_active = True
        self._markup_start_pt = sp
        tool = self.markup_toolbar.active_tool
        col = self.markup_toolbar.current_color

        if tool in ("pixelate", "blur", "magnifier"):
            item = QGraphicsRectItem(QRectF(sp, sp))
            pen = QPen(col, 2.0, Qt.DashLine)
            item.setPen(pen)
            brush_col = QColor(col.red(), col.green(), col.blue(), 50)
            item.setBrush(QBrush(brush_col))
            self.scene.addItem(item)
            self._markup_current_item = item
        elif tool == "arrow":
            item = QGraphicsLineItem(sp.x(), sp.y(), sp.x(), sp.y())
            item.setPen(QPen(col, 4.0, Qt.SolidLine, Qt.RoundCap))
            self.scene.addItem(item)
            self._markup_current_item = item
        elif tool == "text":
            from PySide6.QtWidgets import QInputDialog
            txt, ok = QInputDialog.getText(self, "Text Callout", "Enter label text:")
            if ok and txt:
                rect = QRectF(sp.x(), sp.y(), 100, 30)
                el = MarkupElement(kind="text", rect=rect, text=txt, color=col)
                self._markup_elements.append(el)
                t_item = QGraphicsTextItem(txt)
                t_item.setDefaultTextColor(col)
                t_item.setPos(sp)
                font = QFont("Segoe UI", 12, QFont.Bold)
                t_item.setFont(font)
                self.scene.addItem(t_item)
                self._markup_preview_items.append(t_item)
            self._markup_active = False

    def _drag_markup(self, sp: QPointF) -> None:
        if not self._markup_active or self._markup_current_item is None or self._markup_start_pt is None:
            return
        p1 = self._markup_start_pt
        tool = self.markup_toolbar.active_tool
        if tool in ("pixelate", "blur", "magnifier"):
            r = QRectF(min(p1.x(), sp.x()), min(p1.y(), sp.y()), abs(sp.x() - p1.x()), abs(sp.y() - p1.y()))
            self._markup_current_item.setRect(r)
        elif tool == "arrow":
            self._markup_current_item.setLine(p1.x(), p1.y(), sp.x(), sp.y())

    def _finish_markup(self, sp: QPointF) -> None:
        if not self._markup_active:
            return
        self._markup_active = False
        p1 = self._markup_start_pt
        if p1 is None:
            return

        tool = self.markup_toolbar.active_tool
        col = self.markup_toolbar.current_color

        if tool in ("pixelate", "blur", "magnifier"):
            r = QRectF(min(p1.x(), sp.x()), min(p1.y(), sp.y()), abs(sp.x() - p1.x()), abs(sp.y() - p1.y()))
            if r.width() >= 8 and r.height() >= 8:
                el = MarkupElement(kind=tool, rect=r, color=col)
                self._markup_elements.append(el)
                if self._markup_current_item:
                    self._markup_preview_items.append(self._markup_current_item)
            else:
                if self._markup_current_item:
                    self.scene.removeItem(self._markup_current_item)
        elif tool == "arrow":
            import math
            if math.hypot(sp.x() - p1.x(), sp.y() - p1.y()) >= 8:
                el = MarkupElement(kind="arrow", start_pt=p1, end_pt=sp, color=col, stroke_width=4)
                self._markup_elements.append(el)
                if self._markup_current_item:
                    self._markup_preview_items.append(self._markup_current_item)
            else:
                if self._markup_current_item:
                    self.scene.removeItem(self._markup_current_item)

        self._markup_current_item = None

    def _undo_markup(self) -> None:
        if self._markup_elements:
            self._markup_elements.pop()
        if self._markup_preview_items:
            it = self._markup_preview_items.pop()
            try:
                self.scene.removeItem(it)
            except Exception:
                pass

    def _clear_markup_overlays(self) -> None:
        for it in self._markup_preview_items:
            try:
                self.scene.removeItem(it)
            except Exception:
                pass
        self._markup_preview_items.clear()
        self._markup_elements.clear()
        if self._markup_current_item:
            try:
                self.scene.removeItem(self._markup_current_item)
            except Exception:
                pass
            self._markup_current_item = None

    def _commit_markup(self) -> None:
        if not self._markup_elements or self.master_np_rgb is None:
            self.set_markup_mode(False)
            return

        before = self.master_np_rgb.copy()
        bgr = cv2.cvtColor(self.master_np_rgb, cv2.COLOR_RGB2BGR)
        flattened_bgr = flatten_markup_onto_image(bgr, self._markup_elements)
        self.master_np_rgb = cv2.cvtColor(flattened_bgr, cv2.COLOR_BGR2RGB)

        self._clear_markup_overlays()
        self.set_markup_mode(False)
        self._update_display_proxy()

        cmd = PixelArrayUndoCommand(self, before, self.master_np_rgb.copy(), "Markup & Privacy Redaction")
        self.undo_stack.push(cmd)
        self._render_current_recipe()
        if self.current_media_id is not None:
            self.pixel_edit_saved.emit(self.current_media_id, "")

    # -------------------------------------------------------------------------
    # OpenCV-Powered High-Fidelity Adjustments
    # -------------------------------------------------------------------------
    def begin_slider_adjustment(self) -> None:
        self._is_sliding = True
        self._recipe_before_slider = self.current_recipe.copy()

    def commit_slider_adjustment(self, action_name: str = "Adjust Parameters") -> None:
        self._is_sliding = False
        self._render_current_recipe()
        if getattr(self, "_recipe_before_slider", None) is not None:
            if self._recipe_before_slider != self.current_recipe:
                cmd = RecipeUndoCommand(
                    self, self._recipe_before_slider, self.current_recipe.copy(), action_name
                )
                self.undo_stack.push(cmd)
                self.recipe_changed.emit(self.current_recipe)
            self._recipe_before_slider = None

    def set_exposure_parameters(
        self, brightness: int, contrast: int, shadows: int, highlights: int, clarity: int = 0
    ) -> None:
        self.current_recipe.brightness = float(brightness)
        self.current_recipe.contrast = float(contrast)
        self.current_recipe.shadows = float(shadows)
        self.current_recipe.highlights = float(highlights)
        self.current_recipe.clarity = float(clarity)
        self._render_current_recipe()

    def set_color_parameters(
        self, temperature: int, tint: int, saturation: int, dehaze: int = 0
    ) -> None:
        self.current_recipe.temperature = float(temperature)
        self.current_recipe.tint = float(tint)
        self.current_recipe.saturation = float(saturation)
        self.current_recipe.dehaze = float(dehaze)
        self._render_current_recipe()

    def set_optics_parameters(self, vignette: int = 0) -> None:
        self.current_recipe.vignette = float(vignette)
        self._render_current_recipe()

    def set_levels(self, black_point: int = 0, white_point: int = 255) -> None:
        self.current_recipe.black_point = float(black_point)
        self.current_recipe.white_point = float(white_point)
        self._render_current_recipe()

    def reset_recipe(self) -> None:
        """Resets all non-crop visual adjustments and levels back to default."""
        before = self.current_recipe.copy()
        new_rec = PhotoAdjustmentRecipe(crop_rect=self.current_recipe.crop_rect)
        cmd = RecipeUndoCommand(self, before, new_rec, "Reset to Default")
        self.undo_stack.push(cmd)
        self.current_recipe = new_rec
        self._render_current_recipe()
        self.recipe_changed.emit(self.current_recipe)

    def set_compare_mode(self, active: bool) -> None:
        """Temporarily toggles between adjusted view and unedited original view."""
        self._is_comparing = active
        if active:
            if getattr(self, "_original_pixmap", None) is not None:
                self.pixmap_item.setPixmap(self._original_pixmap)
            else:
                target_np = self.master_np_rgb
                if target_np is not None:
                    h, w = target_np.shape[:2]
                    qimg = QImage(target_np.data, w, h, 3 * w, QImage.Format_RGB888).copy()
                    pix = QPixmap.fromImage(qimg)
                    self.pixmap_item.setPixmap(pix)
        else:
            if getattr(self, "_adjusted_pixmap", None) is not None:
                self.pixmap_item.setPixmap(self._adjusted_pixmap)
            else:
                self._render_current_recipe()

    def get_effect_preview_pixmaps(self, width: int = 96, height: int = 64) -> dict[str, QPixmap]:
        """Generates live mini-thumbnails of the current photo for each of the ribbon effect buttons."""
        if self.master_np_rgb is None or self.current_is_video:
            return {}
        target_np = self.display_proxy_rgb if self.display_proxy_rgb is not None else self.master_np_rgb
        th, tw = target_np.shape[:2]
        if th <= 0 or tw <= 0 or width <= 0 or height <= 0:
            return {}

        # Center-crop source to match the target thumbnail aspect ratio (prevents squishing portrait photos)
        target_aspect = width / float(height)
        src_aspect = tw / float(th)
        if src_aspect > target_aspect:
            # Source is wider than target: center crop horizontally
            crop_w = max(1, int(round(th * target_aspect)))
            x0 = max(0, (tw - crop_w) // 2)
            cropped = target_np[:, x0 : x0 + crop_w]
        else:
            # Source is taller than target (portrait): center crop vertically
            crop_h = max(1, int(round(tw / target_aspect)))
            y0 = max(0, (th - crop_h) // 2)
            cropped = target_np[y0 : y0 + crop_h, :]

        thumb = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_AREA)
        effect_keys = [
            "none",
            "classic_chrome",
            "warm_portrait",
            "vivid_landscape",
            "matte_vintage",
            "fine_art_bw",
            "cool_cinema",
            "sepia",
        ]
        result = {}
        for eff in effect_keys:
            rec = PhotoAdjustmentRecipe(effect=eff)
            rendered = render_adjusted_image(thumb, rec)
            rh, rw = rendered.shape[:2]
            qimg = QImage(rendered.data, rw, rh, 3 * rw, QImage.Format_RGB888).copy()
            result[eff] = QPixmap.fromImage(qimg)
        return result

    def apply_pixelate_redact(
        self, rect: tuple[int, int, int, int], block_size: int = 16
    ) -> None:
        """Pixelates a region of the image for privacy redaction, recorded onto the undo stack."""
        if self.master_np_rgb is None or self.current_is_video:
            return
        l, t, r, b = rect
        h, w = self.master_np_rgb.shape[:2]
        l = max(0, min(w - 1, int(l)))
        t = max(0, min(h - 1, int(t)))
        r = max(l + 1, min(w, int(r)))
        b = max(t + 1, min(h, int(b)))
        if r <= l or b <= t:
            return
        before_arr = self.master_np_rgb.copy()
        patch = self.master_np_rgb[t:b, l:r]
        ph, pw = patch.shape[:2]
        down_w = max(1, pw // block_size)
        down_h = max(1, ph // block_size)
        small_patch = cv2.resize(patch, (down_w, down_h), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small_patch, (pw, ph), interpolation=cv2.INTER_NEAREST)
        self.master_np_rgb[t:b, l:r] = pixelated
        cmd = _PixelEditUndoCommand(self, before_arr, self.master_np_rgb.copy(), "Redact / Pixelate")
        self.undo_stack.push(cmd)
        self._update_display_proxy()
        self._render_current_recipe()
        self.pixel_edit_saved.emit()

    def set_straighten(self, angle_deg: float) -> None:
        self.current_recipe.straighten_deg = float(angle_deg)
        self._render_current_recipe()

    def set_detail(self, sharpen: int, denoise: int) -> None:
        self.current_recipe.sharpen = float(sharpen)
        self.current_recipe.denoise = float(denoise)
        self._render_current_recipe()

    def apply_auto_adjust(self) -> None:
        """Analyzes scene dynamic range & color cast using OpenCV and applies balanced parameters."""
        if self.master_np_rgb is None:
            return
        auto_rec = analyze_auto_adjust(self.master_np_rgb)
        auto_rec.crop_rect = self.current_recipe.crop_rect
        auto_rec.straighten_deg = self.current_recipe.straighten_deg
        cmd = RecipeUndoCommand(self, self.current_recipe.copy(), auto_rec.copy(), "Auto Adjust")
        self.current_recipe = auto_rec
        self.undo_stack.push(cmd)
        self._render_current_recipe()
        self.recipe_changed.emit(self.current_recipe)

    def apply_auto_straighten(self) -> float:
        """Detects dominant horizon/vertical tilt using Canny & Hough line transforms."""
        if self.master_np_rgb is None:
            return 0.0
        angle = detect_auto_straighten_angle(self.master_np_rgb)
        before = self.current_recipe.copy()
        self.current_recipe.straighten_deg = angle
        cmd = RecipeUndoCommand(self, before, self.current_recipe.copy(), "Auto Straighten")
        self.undo_stack.push(cmd)
        self._render_current_recipe()
        self.recipe_changed.emit(self.current_recipe)
        return angle

    def apply_exposure_contrast(self, brightness: int, contrast: int) -> None:
        before = self.current_recipe.copy()
        self.current_recipe.brightness = float(brightness)
        self.current_recipe.contrast = float(contrast)
        cmd = RecipeUndoCommand(self, before, self.current_recipe.copy(), "Adjust Exposure")
        self.undo_stack.push(cmd)
        self._render_current_recipe()
        self.recipe_changed.emit(self.current_recipe)

    def apply_color_saturation(self, saturation: int) -> None:
        before = self.current_recipe.copy()
        self.current_recipe.saturation = float(saturation)
        cmd = RecipeUndoCommand(self, before, self.current_recipe.copy(), "Adjust Color")
        self.undo_stack.push(cmd)
        self._render_current_recipe()
        self.recipe_changed.emit(self.current_recipe)

    def apply_straighten(self, angle_deg: float) -> None:
        before = self.current_recipe.copy()
        self.current_recipe.straighten_deg = float(angle_deg)
        cmd = RecipeUndoCommand(self, before, self.current_recipe.copy(), "Straighten Photo")
        self.undo_stack.push(cmd)
        self._render_current_recipe()
        self.recipe_changed.emit(self.current_recipe)

    def apply_preset_effect(
        self, effect: str
    ) -> None:
        before = self.current_recipe.copy()
        self.current_recipe.effect = effect
        label = "Original" if effect == "none" else effect.replace("_", " ").capitalize()
        cmd = RecipeUndoCommand(self, before, self.current_recipe.copy(), f"Effect: {label}")
        self.undo_stack.push(cmd)
        self._render_current_recipe()
        self.recipe_changed.emit(self.current_recipe)

    def preview_recipe(self, recipe: PhotoAdjustmentRecipe | None) -> None:
        """Previews an adjustment recipe in real-time on the viewer without modifying undo stack or current_recipe."""
        if self.master_np_rgb is None or self.current_is_video:
            return
        if recipe is None:
            self._render_current_recipe()
            return
        target_np = self.display_proxy_rgb if self.display_proxy_rgb is not None else self.master_np_rgb
        scale = getattr(self, "_proxy_scale", 1.0)
        proxy_rec = recipe.copy()
        if proxy_rec.crop_rect is not None and scale < 0.999:
            l, t, r, b = proxy_rec.crop_rect
            proxy_rec.crop_rect = (
                int(round(l * scale)),
                int(round(t * scale)),
                int(round(r * scale)),
                int(round(b * scale)),
            )
        rendered_np = render_adjusted_image(target_np, proxy_rec)
        h, w = rendered_np.shape[:2]
        qimg = QImage(rendered_np.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        if scale < 0.999:
            target_w = int(round(w / scale))
            target_h = int(round(h / scale))
            if target_w != w or target_h != h:
                pix = pix.scaled(target_w, target_h, Qt.IgnoreAspectRatio, Qt.FastTransformation)
        self.pixmap_item.setPixmap(pix)

    def apply_recipe_adjustment(
        self, new_recipe: PhotoAdjustmentRecipe, action_name: str = "Adjust Photo"
    ) -> None:
        """Commits a new recipe to the undo stack and updates the display."""
        if self.master_np_rgb is None:
            return
        cmd = RecipeUndoCommand(self, self.current_recipe.copy(), new_recipe.copy(), action_name)
        self.current_recipe = new_recipe.copy()
        self.undo_stack.push(cmd)
        self._render_current_recipe()
        self.recipe_changed.emit(self.current_recipe)

    def apply_auto_exposure(self) -> None:
        """Analyzes and applies balanced exposure parameters without changing color or effects."""
        if self.master_np_rgb is None:
            return
        auto_rec = analyze_auto_adjust(self.master_np_rgb)
        new_rec = self.current_recipe.copy()
        new_rec.brightness = auto_rec.brightness
        new_rec.contrast = auto_rec.contrast
        new_rec.highlights = auto_rec.highlights
        new_rec.shadows = auto_rec.shadows
        self.apply_recipe_adjustment(new_rec, "Auto Exposure")

    def apply_auto_color(self) -> None:
        """Analyzes and applies balanced color parameters without changing exposure or effects."""
        if self.master_np_rgb is None:
            return
        auto_rec = analyze_auto_adjust(self.master_np_rgb)
        new_rec = self.current_recipe.copy()
        new_rec.temperature = auto_rec.temperature
        new_rec.tint = auto_rec.tint
        new_rec.saturation = auto_rec.saturation
        self.apply_recipe_adjustment(new_rec, "Auto Color")

    def get_preview_thumbnail(self, width: int = 54, height: int = 38) -> np.ndarray | None:
        """Returns a small RGB numpy array of the current image for popup variation menus."""
        if self.master_np_rgb is None or self.current_is_video:
            return None
        target = self.display_proxy_rgb if self.display_proxy_rgb is not None else self.master_np_rgb
        h, w = target.shape[:2]
        if h <= 0 or w <= 0:
            return None
        target_aspect = width / float(height)
        img_aspect = w / float(h)
        if img_aspect > target_aspect:
            crop_w = int(h * target_aspect)
            x1 = (w - crop_w) // 2
            crop = target[:, x1 : x1 + crop_w]
        else:
            crop_h = int(w / target_aspect)
            y1 = (h - crop_h) // 2
            crop = target[y1 : y1 + crop_h, :]
        return cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)

    # -------------------------------------------------------------------------
    # Non-Destructive '.OriginalFiles' Backup Stash & Revert
    # -------------------------------------------------------------------------
    def has_unsaved_adjustments(self) -> bool:
        """Returns True if the current photo has uncommitted recipe adjustments."""
        return not self.current_recipe.is_default()

    def save_current_image_with_backup(self) -> str | None:
        """
        Stashes the pristine original file into '<folder>/.OriginalFiles/<filename>'
        (if not already backed up) before saving the rendered recipe image to disk.
        Returns the backup file path.
        """
        if (
            self.master_np_rgb is None
            or self.current_file_path is None
            or self.current_media_id is None
            or self.current_is_video
        ):
            return None

        backup_dir = self.current_file_path.parent / ".OriginalFiles"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / self.current_file_path.name

        if not backup_path.exists() and self.current_file_path.exists():
            shutil.copy2(self.current_file_path, backup_path)

        crop_box = self.current_recipe.crop_rect
        norm_crop = None
        if crop_box is not None and self.master_np_rgb is not None:
            l, t, r, b = crop_box
            master_h, master_w = self.master_np_rgb.shape[:2]
            if master_w > 0 and master_h > 0:
                norm_crop = (
                    float(l) / master_w,
                    float(t) / master_h,
                    float(r - l) / master_w,
                    float(b - t) / master_h,
                )

        final_rendered = render_adjusted_image(self.master_np_rgb, self.current_recipe)
        save_img = Image.fromarray(final_rendered)

        save_kwargs = {"quality": 95} if self.current_file_path.suffix.lower() in (".jpg", ".jpeg") else {}
        save_img.save(self.current_file_path, **save_kwargs)

        if norm_crop is not None and self.current_media_id is not None:
            if hasattr(self, "catalog_repo") and self.catalog_repo:
                try:
                    self.catalog_repo.transform_face_regions_for_crop(
                        int(self.current_media_id), *norm_crop
                    )
                except Exception:
                    pass

        self.master_pil_image = save_img
        self.master_np_rgb = final_rendered
        self.current_recipe = PhotoAdjustmentRecipe()
        self.undo_stack.clear()
        self._render_current_recipe()

        self.pixel_edit_saved.emit(int(self.current_media_id), str(backup_path))
        self.update_revert_state()
        return str(backup_path)

    def update_revert_state(self) -> None:
        """Updates enabled state of the Revert button based on disk backup or in-memory changes."""
        has_backup = False
        if self.current_file_path is not None:
            bpath = self.current_file_path.parent / ".OriginalFiles" / self.current_file_path.name
            if bpath.exists():
                has_backup = True
        try:
            can_undo = hasattr(self, "undo_stack") and self.undo_stack is not None and self.undo_stack.canUndo()
        except RuntimeError:
            can_undo = False
        has_recipe_edits = (
            self.current_recipe is not None and not self.current_recipe.is_default()
        ) or can_undo
        can_revert = has_backup or has_recipe_edits
        if hasattr(self, "btn_revert"):
            self.btn_revert.setEnabled(can_revert)
        mw = getattr(self, "main_window", None) or self.window()
        if hasattr(mw, "_update_revert_button_state"):
            mw._update_revert_button_state()

    def revert_to_original(self, backup_path: str | Path | None = None) -> bool:
        """
        Restores the unedited photo:
        1. If a disk backup in '.OriginalFiles' exists, restores the original file.
        2. If uncommitted in-memory recipe adjustments exist, resets them back to 0.
        """
        if self.current_file_path is None or self.current_media_id is None:
            return False

        if backup_path is None:
            bpath = self.current_file_path.parent / ".OriginalFiles" / self.current_file_path.name
        else:
            bpath = Path(backup_path)

        if bpath.exists():
            shutil.copy2(bpath, self.current_file_path)
            try:
                bpath.unlink()
            except OSError:
                pass
            self.load_media(self.current_media_id, self.current_file_path, is_video=False)
        else:
            self.current_recipe = PhotoAdjustmentRecipe()
            self.undo_stack.clear()
            self._render_current_recipe()
            self.recipe_changed.emit(self.current_recipe)

        self.update_revert_state()
        self.pixel_edit_reverted.emit(int(self.current_media_id))
        return True

    # -------------------------------------------------------------------------
    # Static-Image Face Region Overlay & Interactive WLPG Face Tagging
    # -------------------------------------------------------------------------
    def _clear_face_overlays(self) -> None:
        for item in self._face_overlay_items:
            if item.scene() is self.scene:
                self.scene.removeItem(item)
        self._face_overlay_items.clear()

    def render_face_regions(self, people_regions: list[dict]) -> None:
        """Draws normalized XMP MP:RegionInfo face rectangles and screen-space name badges on static images."""
        self._clear_face_overlays()
        self._current_people_regions = list(people_regions or [])
        if self.current_is_video or self.current_pil_image is None:
            return
        w, h = self.current_pil_image.size
        for reg in self._current_people_regions:
            rx = float(reg.get("rect_x") or 0.0)
            ry = float(reg.get("rect_y") or 0.0)
            rw = float(reg.get("rect_w") or 0.0)
            rh = float(reg.get("rect_h") or 0.0)
            if rw <= 0.0 or rh <= 0.0:
                continue

            pid = reg.get("person_id")
            rect_scene = QRectF(rx * w, ry * h, rw * w, rh * h)

            shadow_box = QGraphicsRectItem(rect_scene)
            shadow_pen = QPen(QColor(0, 0, 0, 165), 3.0)
            shadow_pen.setCosmetic(True)
            shadow_box.setPen(shadow_pen)
            shadow_box.setBrush(QBrush(Qt.transparent))
            shadow_box.setZValue(2)
            shadow_box.setData(0, pid)
            shadow_box.setVisible(False)
            self.scene.addItem(shadow_box)
            self._face_overlay_items.append(shadow_box)

            box = QGraphicsRectItem(rect_scene)
            fg_pen = QPen(QColor("#FFFFFF"), 1.5)
            fg_pen.setCosmetic(True)
            box.setPen(fg_pen)
            box.setBrush(QBrush(Qt.transparent))
            box.setZValue(2.5)
            box.setData(0, pid)
            box.setVisible(False)
            self.scene.addItem(box)
            self._face_overlay_items.append(box)

            name_str = str(reg.get("name") or "").strip()
            if reg.get("is_unconfirmed") and name_str:
                name_str = f"{name_str}?"
            badge = None
            if name_str:
                badge = _FaceNameBadgeItem(name_str, pid)
                badge.setPos((rx + rw * 0.5) * w, (ry + rh) * h)
                badge.setData(0, pid)
                badge.setVisible(False)
                self.scene.addItem(badge)
                self._face_overlay_items.append(badge)

            # Interactive hover sensor over face region
            related = [shadow_box, box]
            if badge is not None:
                related.append(badge)

            sensor_rect = QRectF(rect_scene)
            sensor_rect.adjust(-4, -4, 4, 30 if name_str else 4)
            sensor = _FaceHoverRegionItem(sensor_rect, self, related, pid, region_data=reg)
            sensor.setData(0, pid)
            self.scene.addItem(sensor)
            self._face_overlay_items.append(sensor)

        if self.is_in_edit_mode():
            self._update_face_overlays_visibility()

    @property
    def is_face_tagging_mode(self) -> bool:
        return self.is_face_tag_mode

    @property
    def _face_tag_target_item(self) -> _FaceTagTargetBoxItem | None:
        return self._face_target_item

    @property
    def image_scene(self):
        return self.scene

    def highlight_person_face(self, person_id: int | None) -> None:
        """Highlights a specific person's face box and name badge on the image when hovered in the right pane."""
        if self.is_in_edit_mode():
            return
        self._highlighted_person_id = person_id
        for item in self._face_overlay_items:
            if isinstance(item, _FaceHoverRegionItem):
                continue
            item_pid = item.data(0)
            is_match = person_id is not None and item_pid == person_id
            if is_match:
                item.setVisible(True)
                if isinstance(item, QGraphicsRectItem) and item.zValue() > 2.2:
                    p = QPen(QColor("#3399FF"), 2.5)
                    p.setCosmetic(True)
                    item.setPen(p)
                elif isinstance(item, _FaceNameBadgeItem):
                    item.set_highlighted(True)
            else:
                if isinstance(item, QGraphicsRectItem) and item.zValue() > 2.2:
                    p = QPen(QColor("#FFFFFF"), 1.5)
                    p.setCosmetic(True)
                    item.setPen(p)
                elif isinstance(item, _FaceNameBadgeItem):
                    item.set_highlighted(False)
                item.setVisible(False)

    def start_face_tagging_mode(
        self,
        initial_rect: tuple[float, float, float, float] | None = None,
        replace_person_id: int | None = None,
    ) -> bool:
        """
        Enters WLPG interactive face tagging mode:
        - Displays a movable/resizable white L-bracket face target box on the photo
        - Displays the floating 'Tag someone' popup card with existing people list + search
        """
        if self.current_is_video or self.current_pil_image is None or self.current_media_id is None:
            return False

        if self.is_crop_mode:
            self.set_crop_mode(False)
        if self.is_retouch_mode:
            self.set_retouch_mode(False)
        if self.is_red_eye_mode:
            self.set_red_eye_mode(False)

        self.stop_face_tagging_mode()
        self.is_face_tag_mode = True
        self._face_tag_replace_person_id = replace_person_id

        w, h = self.current_pil_image.size
        img_rect = QRectF(0, 0, w, h)

        mw = getattr(self, "main_window", None) or self.window()
        repo = getattr(mw, "repo", None)

        norm_rect: tuple[float, float, float, float] | None = initial_rect
        if norm_rect is None and replace_person_id is not None:
            for reg in self._current_people_regions:
                if reg.get("person_id") == replace_person_id:
                    rx = float(reg.get("rect_x") or 0.0)
                    ry = float(reg.get("rect_y") or 0.0)
                    rw = float(reg.get("rect_w") or 0.0)
                    rh = float(reg.get("rect_h") or 0.0)
                    if rw > 0.0 and rh > 0.0:
                        norm_rect = (rx, ry, rw, rh)
                        break

        if norm_rect is None and repo is not None and hasattr(repo, "get_untagged_faces_for_media"):
            untagged = repo.get_untagged_faces_for_media(int(self.current_media_id))
            if untagged:
                u0 = untagged[0]
                norm_rect = (
                    float(u0.get("rect_x") or 0.38),
                    float(u0.get("rect_y") or 0.30),
                    float(u0.get("rect_w") or 0.22),
                    float(u0.get("rect_h") or 0.22),
                )

        if norm_rect is not None:
            rx, ry, rw, rh = norm_rect
            target_scene_rect = QRectF(rx * w, ry * h, max(32.0, rw * w), max(32.0, rh * h))
        else:
            side = max(48.0, min(w, h) * 0.22)
            target_scene_rect = QRectF((w - side) * 0.5, (h - side) * 0.42, side, side)

        self._face_target_item = _FaceTagTargetBoxItem(
            target_scene_rect,
            img_rect,
            on_changed=self._position_tag_someone_popup,
        )
        self.scene.addItem(self._face_target_item)

        people_rows = []
        if repo is not None and hasattr(repo, "get_people_list"):
            try:
                people_rows = repo.get_people_list()
            except Exception:
                people_rows = []
        self._tag_someone_popup.populate(people_rows)
        self._tag_someone_popup.show()
        self._tag_someone_popup.raise_()
        self._position_tag_someone_popup()
        self._tag_someone_popup.focus_input()
        return True

    def stop_face_tagging_mode(self) -> None:
        """Exits interactive WLPG face tagging mode and hides the 'Tag someone' popup."""
        self.is_face_tag_mode = False
        self._face_tag_replace_person_id = None
        if self._face_target_item is not None:
            if self._face_target_item.scene() is self.scene:
                self.scene.removeItem(self._face_target_item)
            self._face_target_item = None
        if hasattr(self, "_tag_someone_popup") and self._tag_someone_popup is not None:
            self._tag_someone_popup.hide()
        if getattr(self, "is_face_adjust_mode", False):
            self.stop_face_adjust_mode()

    def _position_tag_someone_popup(self) -> None:
        if not getattr(self, "is_face_tag_mode", False):
            return
        if self._face_target_item is None or not self._tag_someone_popup.isVisible():
            return
        r = self._face_target_item.rect()
        top_right_vp = self.graphics_view.mapFromScene(QPointF(r.right(), r.top()))
        top_left_vp = self.graphics_view.mapFromScene(QPointF(r.left(), r.top()))
        pw = self._tag_someone_popup.width()
        ph = self._tag_someone_popup.height()
        gw = self.graphics_view.width()
        gh = self.graphics_view.height()

        px = top_right_vp.x() + 10
        if px + pw > gw - 8:
            px = max(8, top_left_vp.x() - pw - 10)
        py = max(8, min(gh - ph - 8, top_right_vp.y()))
        self._tag_someone_popup.move(px, py)

    def _on_tag_someone_selected(self, person_name: str) -> None:
        if not person_name or not person_name.strip():
            return
        self.tag_selected_crop_as_face(person_name.strip())

    def tag_selected_crop_as_face(self, person_name: str | None = None) -> bool:
        """
        Converts the current face target box (or drawn crop rectangle) on a static image
        into normalized [0..1] MP:RegionInfo coordinates and emits face_region_tagged.
        """
        if self.current_is_video or self.current_pil_image is None or self.current_media_id is None:
            return False

        rect: QRectF | None = None
        if self._face_target_item is not None:
            rect = self._face_target_item.rect()
        else:
            rect = self.scene.get_crop_rect()

        if rect is None or rect.width() < 4 or rect.height() < 4:
            return False

        w, h = self.current_pil_image.size
        if w <= 0 or h <= 0:
            return False

        if person_name is None:
            name, ok = QInputDialog.getText(self, "Tag a Person", "Enter person's name:")
            if not ok or not name.strip():
                return False
            person_name = name.strip()

        rx = max(0.0, min(1.0, rect.left() / w))
        ry = max(0.0, min(1.0, rect.top() / h))
        rw = max(0.0, min(1.0 - rx, rect.width() / w))
        rh = max(0.0, min(1.0 - ry, rect.height() / h))

        replace_pid = self._face_tag_replace_person_id
        self.stop_face_tagging_mode()
        self.scene.clear_crop_rect()

        if replace_pid is not None:
            mw = getattr(self, "main_window", None) or self.window()
            repo = getattr(mw, "repo", None)
            if repo is not None and hasattr(repo, "remove_person_from_media"):
                try:
                    repo.remove_person_from_media(int(self.current_media_id), int(replace_pid))
                except Exception:
                    pass

        self.face_region_tagged.emit(int(self.current_media_id), person_name, rx, ry, rw, rh)
        return True

    def show_face_region_context_menu(
        self, screen_pos: QPoint, region_data: dict, person_id: int | None
    ) -> QMenu:
        """Context menu when right-clicking a face region in Single Image View."""
        menu = QMenu(self)
        menu.setFont(QFont("Segoe UI", 9))

        mw = getattr(self, "main_window", None) or self.window()
        face_id = region_data.get("face_id") or region_data.get("id") or 0
        if not face_id and self.current_media_id is not None and person_id is not None:
            if hasattr(mw, "catalog_repo"):
                row = mw.catalog_repo.engine.catalog_conn().execute(
                    "SELECT id FROM media_people WHERE media_id = ? AND (person_id = ? OR suggested_person_id = ?) LIMIT 1;",
                    (int(self.current_media_id), int(person_id), int(person_id)),
                ).fetchone()
                if row:
                    face_id = int(row["id"])

        is_unconfirmed = bool(region_data.get("is_unconfirmed"))
        person_name = str(region_data.get("name") or "").strip()

        # 1. Confirm (if unconfirmed suggestion)
        if is_unconfirmed and face_id and person_id:
            act_confirm = menu.addAction(f"Confirm '{person_name}'")
            bold = QFont(act_confirm.font())
            bold.setBold(True)
            act_confirm.setFont(bold)
            act_confirm.triggered.connect(
                lambda: mw._on_dock_person_confirm_requested(int(person_id), int(face_id))
            )

        # 2. Adjust bounding box
        act_adjust = menu.addAction("Adjust bounding box")
        act_adjust.triggered.connect(
            lambda: self.start_face_adjust_mode(region_data, face_id, person_id)
        )

        menu.addSeparator()

        # 3. Tag as someone else...
        act_retag = menu.addAction("Tag as someone else...")
        act_retag.triggered.connect(
            lambda: self._on_face_region_tag_as_someone_else(region_data, person_id)
        )

        # 4. Not this person (Remove tag)
        if person_id:
            act_not_person = menu.addAction("Not this person")
            act_not_person.triggered.connect(
                lambda: self._on_face_region_not_this_person(region_data, person_id, face_id)
            )

        # 5. Not a face (Delete region)
        act_not_face = menu.addAction("Not a face")
        act_not_face.triggered.connect(
            lambda: self._on_face_region_not_a_face(region_data, person_id, face_id)
        )

        # 6. Set as display picture
        if person_id:
            act_display_pic = menu.addAction("Set as display picture")
            act_display_pic.triggered.connect(
                lambda: self._on_face_region_set_display_pic(person_id, face_id)
            )

        if not getattr(self, "_suppress_menu_exec", False):
            menu.exec(screen_pos)
        return menu

    def start_face_adjust_mode(
        self,
        region_data: dict,
        face_id: int | None = None,
        person_id: int | None = None,
    ) -> bool:
        """Enters interactive bounding box adjustment mode for a face region."""
        if self.current_is_video or self.current_pil_image is None or self.current_media_id is None:
            return False

        if self.is_crop_mode:
            self.set_crop_mode(False)
        if self.is_retouch_mode:
            self.set_retouch_mode(False)
        if self.is_red_eye_mode:
            self.set_red_eye_mode(False)
        if self.is_face_tag_mode:
            self.stop_face_tagging_mode()

        self.stop_face_adjust_mode()
        self.is_face_adjust_mode = True
        self._adjusting_face_info = {
            "region": dict(region_data),
            "face_id": face_id,
            "person_id": person_id,
        }

        # Hide static face overlays while adjusting
        for it in self._face_overlay_items:
            it.setVisible(False)

        w, h = self.current_pil_image.size
        img_rect = QRectF(0, 0, w, h)
        rx = float(region_data.get("rect_x") or 0.0)
        ry = float(region_data.get("rect_y") or 0.0)
        rw = float(region_data.get("rect_w") or 0.2)
        rh = float(region_data.get("rect_h") or 0.2)
        target_scene_rect = QRectF(rx * w, ry * h, max(24.0, rw * w), max(24.0, rh * h))

        self._face_adjust_box_item = _FaceTagTargetBoxItem(
            target_scene_rect,
            img_rect,
            on_changed=self._position_adjust_toolbar,
        )
        self.scene.addItem(self._face_adjust_box_item)

        self._adjust_toolbar.show()
        self._adjust_toolbar.raise_()
        self._position_adjust_toolbar()
        return True

    def stop_face_adjust_mode(self) -> None:
        """Exits bounding box adjustment mode and restores face overlays."""
        self.is_face_adjust_mode = False
        self._adjusting_face_info = {}
        if hasattr(self, "_adjust_toolbar") and self._adjust_toolbar is not None:
            self._adjust_toolbar.hide()
        if hasattr(self, "_face_adjust_box_item") and self._face_adjust_box_item is not None:
            if self._face_adjust_box_item.scene() is self.scene:
                self.scene.removeItem(self._face_adjust_box_item)
            self._face_adjust_box_item = None
        for it in self._face_overlay_items:
            it.setVisible(False)

    def save_face_adjust(self) -> None:
        """Saves updated bounding box to catalog.db, updates face thumbnail, and re-renders."""
        if not getattr(self, "is_face_adjust_mode", False) or self._face_adjust_box_item is None:
            return
        if self.current_pil_image is None or self.current_media_id is None:
            self.stop_face_adjust_mode()
            return
        w, h = self.current_pil_image.size
        r = self._face_adjust_box_item.rect()
        norm_rect = (
            max(0.0, min(1.0, r.x() / w)),
            max(0.0, min(1.0, r.y() / h)),
            max(0.01, min(1.0, r.width() / w)),
            max(0.01, min(1.0, r.height() / h)),
        )
        face_info = getattr(self, "_adjusting_face_info", {})
        face_id = face_info.get("face_id") or 0
        person_id = face_info.get("person_id")
        mw = getattr(self, "main_window", None) or self.window()
        if hasattr(mw, "catalog_repo"):
            if not face_id and person_id:
                row = mw.catalog_repo.engine.catalog_conn().execute(
                    "SELECT id FROM media_people WHERE media_id = ? AND (person_id = ? OR suggested_person_id = ?) LIMIT 1;",
                    (int(self.current_media_id), int(person_id), int(person_id)),
                ).fetchone()
                if row:
                    face_id = int(row["id"])
            if face_id > 0:
                thumb_repo = getattr(mw, "thumb_repo", None)
                mw.catalog_repo.update_face_rect(face_id, norm_rect, thumb_repo=thumb_repo)
                mw.lbl_sync_status.setText("✓ Face bounding box updated")
            mw.refresh_navigation_and_autocomplete()
        self.stop_face_adjust_mode()
        if hasattr(mw, "catalog_repo"):
            detail = mw.catalog_repo.get_media_detail(int(self.current_media_id))
            self.render_face_regions(detail.get("people", []))

    def _position_adjust_toolbar(self) -> None:
        if not getattr(self, "is_face_adjust_mode", False):
            return
        if (
            self._face_adjust_box_item is None
            or not hasattr(self, "_adjust_toolbar")
            or not self._adjust_toolbar.isVisible()
        ):
            return
        r = self._face_adjust_box_item.rect()
        bottom_center_vp = self.graphics_view.mapFromScene(QPointF(r.center().x(), r.bottom()))
        tw = self._adjust_toolbar.width()
        th = self._adjust_toolbar.height()
        gw = self.graphics_view.width()
        gh = self.graphics_view.height()

        px = int(bottom_center_vp.x() - tw / 2)
        px = max(8, min(gw - tw - 8, px))
        py = int(bottom_center_vp.y() + 8)
        if py + th > gh - 8:
            top_center_vp = self.graphics_view.mapFromScene(QPointF(r.center().x(), r.top()))
            py = max(8, int(top_center_vp.y() - th - 8))
        self._adjust_toolbar.move(px, py)

    def _on_face_region_tag_as_someone_else(self, region_data: dict, person_id: int | None) -> None:
        mw = getattr(self, "main_window", None) or self.window()
        if not hasattr(mw, "catalog_repo") or self.current_media_id is None:
            return
        people_rows = mw.catalog_repo.get_people_list()
        new_name, new_pid = TagSomeoneDialog.prompt(people_rows, self, title="Tag someone")
        if not new_name or not new_name.strip():
            return
        new_name_clean = new_name.strip()
        if new_pid is None or new_pid < 0:
            new_pid = mw.catalog_repo.upsert_person(new_name_clean)
        rx = float(region_data.get("rect_x") or 0.0)
        ry = float(region_data.get("rect_y") or 0.0)
        rw = float(region_data.get("rect_w") or 0.0)
        rh = float(region_data.get("rect_h") or 0.0)
        rect = (rx, ry, rw, rh) if rw > 0 and rh > 0 else None
        if person_id:
            mw.catalog_repo.remove_person_tag(int(self.current_media_id), int(person_id))
        mw.catalog_repo.assign_person(int(self.current_media_id), new_name_clean, rect)
        mw.refresh_navigation_and_autocomplete()
        detail = mw.catalog_repo.get_media_detail(int(self.current_media_id))
        self.render_face_regions(detail.get("people", []))
        summary = mw.catalog_repo.get_selection_summary([int(self.current_media_id)])
        mw.right_tag_dock.bind_selection([int(self.current_media_id)], summary)
        mw.lbl_sync_status.setText(f"✓ Retagged face to '{new_name_clean}'")

    def _on_face_region_not_this_person(self, region_data: dict, person_id: int, face_id: int) -> None:
        mw = getattr(self, "main_window", None) or self.window()
        if not hasattr(mw, "catalog_repo") or self.current_media_id is None:
            return
        if face_id > 0:
            mw.catalog_repo.add_face_rejection(face_id, person_id)
        mw.catalog_repo.remove_person_tag(int(self.current_media_id), person_id)
        mw.refresh_navigation_and_autocomplete()
        detail = mw.catalog_repo.get_media_detail(int(self.current_media_id))
        self.render_face_regions(detail.get("people", []))
        summary = mw.catalog_repo.get_selection_summary([int(self.current_media_id)])
        mw.right_tag_dock.bind_selection([int(self.current_media_id)], summary)
        mw.lbl_sync_status.setText("✓ Removed person tag")

    def _on_face_region_not_a_face(self, region_data: dict, person_id: int | None, face_id: int) -> None:
        mw = getattr(self, "main_window", None) or self.window()
        if not hasattr(mw, "catalog_repo") or self.current_media_id is None:
            return
        if face_id > 0:
            mw.catalog_repo.mark_not_a_face(face_id)
            if person_id:
                with mw.catalog_repo.engine.write_lock:
                    conn = mw.catalog_repo.engine.catalog_conn()
                    conn.execute(
                        """
                        UPDATE people
                        SET confirmed_count = (
                            SELECT COUNT(*) FROM media_people WHERE person_id = people.id AND state = 1
                        )
                        WHERE id = ?;
                        """,
                        (person_id,),
                    )
                    conn.commit()
        mw.refresh_navigation_and_autocomplete()
        detail = mw.catalog_repo.get_media_detail(int(self.current_media_id))
        self.render_face_regions(detail.get("people", []))
        summary = mw.catalog_repo.get_selection_summary([int(self.current_media_id)])
        mw.right_tag_dock.bind_selection([int(self.current_media_id)], summary)
        mw.lbl_sync_status.setText("✓ Marked region as 'Not a face'")

    def _on_face_region_set_display_pic(self, person_id: int, face_id: int) -> None:
        mw = getattr(self, "main_window", None) or self.window()
        if not hasattr(mw, "catalog_repo") or self.current_media_id is None:
            return
        mw._on_face_crop_set_display_pic(int(self.current_media_id), int(person_id), face_id=face_id)

    # -------------------------------------------------------------------------
    # Video Playback Controls & Keyboard Shortcuts
    # -------------------------------------------------------------------------
    def _on_playback_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        if state == QMediaPlayer.PlayingState:
            self.btn_play_pause.setText("⏸ Pause")
        else:
            self.btn_play_pause.setText("▶ Play")

    def toggle_video_playback(self) -> None:
        if self.media_player.playbackState() == QMediaPlayer.PlayingState:
            self.media_player.pause()
            self.btn_play_pause.setText("▶ Play")
        else:
            self.media_player.play()
            self.btn_play_pause.setText("⏸ Pause")

    def stop_video_playback(self) -> None:
        self.media_player.stop()
        self.btn_play_pause.setText("▶ Play")

    def _show_video_context_menu(self, pos: QPoint) -> QMenu:
        menu = QMenu(self)
        menu.setFont(QFont("Segoe UI", 9))

        is_playing = self.media_player.playbackState() == QMediaPlayer.PlayingState
        act_toggle = menu.addAction("Pause" if is_playing else "Play")
        bold_font = QFont(act_toggle.font())
        bold_font.setBold(True)
        act_toggle.setFont(bold_font)
        act_toggle.triggered.connect(self.toggle_video_playback)

        act_stop = menu.addAction("Stop")
        act_stop.triggered.connect(self.stop_video_playback)

        is_muted = self.audio_output.isMuted()
        act_mute = menu.addAction("Unmute" if is_muted else "Mute")
        act_mute.triggered.connect(lambda: self.audio_output.setMuted(not is_muted))

        menu.addSeparator()

        main_win = self.window()
        mid = self.current_media_id

        open_with_menu = menu.addMenu("Open with")
        open_with_menu.setFont(QFont("Segoe UI", 9))
        act_default = open_with_menu.addAction("System default player")
        act_choose = open_with_menu.addAction("Choose another app...")
        
        def _on_open_default() -> None:
            if hasattr(main_win, "open_with_app_for_media") and mid is not None:
                main_win.open_with_app_for_media(mid, "default")
            elif self.current_file_path:
                from PySide6.QtCore import QUrl
                from PySide6.QtGui import QDesktopServices
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.current_file_path)))

        def _on_open_choose() -> None:
            if hasattr(main_win, "open_with_app_for_media") and mid is not None:
                main_win.open_with_app_for_media(mid, "choose")
            elif self.current_file_path:
                import sys
                if sys.platform == "win32":
                    import ctypes
                    ctypes.windll.shell32.OpenAs_RunDLLW(0, 0, str(Path(self.current_file_path).resolve()), 1)

        act_default.triggered.connect(_on_open_default)
        act_choose.triggered.connect(_on_open_choose)

        act_loc = menu.addAction("Open file location")
        def _on_open_loc() -> None:
            if hasattr(main_win, "open_file_location_for_media") and mid is not None:
                main_win.open_file_location_for_media(mid)
            elif self.current_file_path:
                import os, subprocess, sys
                p = Path(self.current_file_path)
                if sys.platform == "win32":
                    subprocess.Popen(f'explorer.exe /select,"{os.path.normpath(str(p.resolve()))}"')
                else:
                    from PySide6.QtCore import QUrl
                    from PySide6.QtGui import QDesktopServices
                    QDesktopServices.openUrl(QUrl.fromLocalFile(str(p.parent)))
        act_loc.triggered.connect(_on_open_loc)

        menu.addSeparator()

        act_prop = menu.addAction("Properties")
        def _on_prop() -> None:
            if hasattr(main_win, "toggle_right_pane"):
                main_win.toggle_right_pane("info")
        act_prop.triggered.connect(_on_prop)

        if not getattr(self, "_suppress_menu_exec", False):
            menu.exec(self.video_widget.mapToGlobal(pos))
        return menu

    def _on_video_position_changed(self, pos_ms: int) -> None:
        self.slider_position.blockSignals(True)
        self.slider_position.setValue(pos_ms)
        self.slider_position.blockSignals(False)
        dur_ms = max(0, self.media_player.duration())
        self.lbl_video_time.setText(
            f"{pos_ms // 60000:02d}:{(pos_ms // 1000) % 60:02d} / "
            f"{dur_ms // 60000:02d}:{(dur_ms // 1000) % 60:02d}"
        )

    def _on_video_duration_changed(self, dur_ms: int) -> None:
        self.slider_position.setRange(0, max(1, dur_ms))

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        # Intercept graphics_view viewport mouse events for Retouch and Red Eye
        if watched == self.graphics_view.viewport():
            if self.is_face_tag_mode and self._face_target_item is not None:
                if event.type() == QEvent.MouseButtonPress and getattr(event, "button", lambda: None)() == Qt.LeftButton:
                    sp = self.graphics_view.mapToScene(event.position().toPoint())
                    item_rect = self._face_target_item.rect().adjusted(-14, -14, 14, 14)
                    if not item_rect.contains(sp) and self.current_pil_image is not None:
                        w, h = self.current_pil_image.size
                        cur_r = self._face_target_item.rect()
                        nw, nh = cur_r.width(), cur_r.height()
                        nx = max(0.0, min(w - nw, sp.x() - nw * 0.5))
                        ny = max(0.0, min(h - nh, sp.y() - nh * 0.5))
                        self._face_target_item.set_rect(QRectF(nx, ny, nw, nh))
                        self._tag_someone_popup.focus_input()
                        return True
            if self.is_retouch_mode:
                if event.type() == QEvent.MouseButtonPress and getattr(event, "button", lambda: None)() == Qt.LeftButton:
                    sp = self.graphics_view.mapToScene(event.position().toPoint())
                    self._start_retouch(sp)
                    return True
                elif event.type() == QEvent.MouseMove and getattr(self, "_retouch_active", False):
                    sp = self.graphics_view.mapToScene(event.position().toPoint())
                    self._drag_retouch(sp)
                    return True
                elif event.type() == QEvent.MouseButtonRelease and getattr(event, "button", lambda: None)() == Qt.LeftButton and getattr(self, "_retouch_active", False):
                    self._finish_retouch()
                    return True

            if self.is_red_eye_mode:
                if event.type() == QEvent.MouseButtonPress and getattr(event, "button", lambda: None)() == Qt.LeftButton:
                    sp = self.graphics_view.mapToScene(event.position().toPoint())
                    self._start_red_eye(sp)
                    return True
                elif event.type() == QEvent.MouseMove and getattr(self, "_red_eye_active", False):
                    sp = self.graphics_view.mapToScene(event.position().toPoint())
                    self._drag_red_eye(sp)
                    return True
                elif event.type() == QEvent.MouseButtonRelease and getattr(event, "button", lambda: None)() == Qt.LeftButton and getattr(self, "_red_eye_active", False):
                    self._finish_red_eye()
                    return True

            if getattr(self, "is_markup_mode", False):
                if event.type() == QEvent.MouseButtonPress and getattr(event, "button", lambda: None)() == Qt.LeftButton:
                    sp = self.graphics_view.mapToScene(event.position().toPoint())
                    self._start_markup(sp)
                    return True
                elif event.type() == QEvent.MouseMove and getattr(self, "_markup_active", False):
                    sp = self.graphics_view.mapToScene(event.position().toPoint())
                    self._drag_markup(sp)
                    return True
                elif event.type() == QEvent.MouseButtonRelease and getattr(event, "button", lambda: None)() == Qt.LeftButton and getattr(self, "_markup_active", False):
                    sp = self.graphics_view.mapToScene(event.position().toPoint())
                    self._finish_markup(sp)
                    return True

        if event.type() == QEvent.KeyRelease and isinstance(event, QKeyEvent):
            if event.key() == Qt.Key_Backslash:
                self.set_compare_mode(False)
                event.accept()
                return True

        if event.type() == QEvent.KeyPress and isinstance(event, QKeyEvent):
            if event.key() == Qt.Key_Backslash:
                self.set_compare_mode(True)
                event.accept()
                return True

            if event.key() == Qt.Key_BracketLeft and self.is_retouch_mode:
                cur = getattr(self, "_retouch_radius", 25)
                self._on_retouch_size_changed(max(8, cur - 4))
                event.accept()
                return True

            if event.key() == Qt.Key_BracketRight and self.is_retouch_mode:
                cur = getattr(self, "_retouch_radius", 25)
                self._on_retouch_size_changed(min(120, cur + 4))
                event.accept()
                return True

            if event.key() == Qt.Key_Escape and self.is_face_tag_mode:
                self.stop_face_tagging_mode()
                event.accept()
                return True

            focus_w = QApplication.focusWidget()
            is_text_entry = isinstance(focus_w, (QLineEdit, QTextEdit, QPlainTextEdit))
            if is_text_entry:
                return super().eventFilter(watched, event)

            if event.key() == Qt.Key_Escape:
                if self.is_crop_mode:
                    self.set_crop_mode(False)
                    event.accept()
                    return True
                elif self.is_retouch_mode:
                    self.set_retouch_mode(False)
                    event.accept()
                    return True
                elif self.is_red_eye_mode:
                    self.set_red_eye_mode(False)
                    event.accept()
                    return True
                elif getattr(self, "is_markup_mode", False):
                    self.set_markup_mode(False)
                    event.accept()
                    return True
                else:
                    self.closed.emit()
                    event.accept()
                    return True

            if getattr(self, "is_face_adjust_mode", False):
                if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                    self.save_face_adjust()
                    event.accept()
                    return True
                elif event.key() == Qt.Key_Escape:
                    self.stop_face_adjust_mode()
                    event.accept()
                    return True

            if event.key() == Qt.Key_Space and self.current_is_video:
                self.toggle_video_playback()
                event.accept()
                return True

            if event.key() in (Qt.Key_Return, Qt.Key_Enter) and self.is_crop_mode:
                self.apply_crop()
                event.accept()
                return True

            # Previous / Next image navigation (Left / Right arrow)
            if not self.is_crop_mode and event.modifiers() in (Qt.NoModifier, Qt.KeypadModifier):
                if event.key() == Qt.Key_Left:
                    self.navigate_requested.emit(-1)
                    event.accept()
                    return True
                elif event.key() == Qt.Key_Right:
                    self.navigate_requested.emit(1)
                    event.accept()
                    return True

            # Zoom in (+) / Zoom out (-)
            if not self.is_crop_mode and not (event.modifiers() & (Qt.ControlModifier | Qt.AltModifier)):
                if event.key() in (Qt.Key_Plus, Qt.Key_Equal) or event.text() in ("+", "="):
                    self.zoom_in()
                    event.accept()
                    return True
                elif event.key() in (Qt.Key_Minus, Qt.Key_Underscore) or event.text() in ("-", "_"):
                    self.zoom_out()
                    event.accept()
                    return True

            # Star ratings: 0, 1, 2, 3, 4, 5
            if not self.is_crop_mode and event.modifiers() in (Qt.NoModifier, Qt.KeypadModifier):
                if Qt.Key_0 <= event.key() <= Qt.Key_5:
                    r_val = event.key() - Qt.Key_0
                    mw = getattr(self, "main_window", None) or self.window()
                    if hasattr(mw, "apply_rating_to_selection"):
                        mw.apply_rating_to_selection(r_val)
                    event.accept()
                    return True

            # Fullscreen / Slideshow: F11 / F12
            if event.key() in (Qt.Key_F11, Qt.Key_F12):
                mw = getattr(self, "main_window", None) or self.window()
                if hasattr(mw, "dispatch_ribbon_action"):
                    mw.dispatch_ribbon_action("slide_show")
                event.accept()
                return True

            # Ctrl shortcuts: Rotate left (Ctrl+,), Rotate right (Ctrl+.), Actual size (Ctrl+0)
            if event.modifiers() == Qt.ControlModifier:
                if event.key() == Qt.Key_Comma:
                    self.rotate_media(clockwise=False)
                    event.accept()
                    return True
                elif event.key() == Qt.Key_Period:
                    self.rotate_media(clockwise=True)
                    event.accept()
                    return True
                elif event.key() == Qt.Key_0:
                    self.set_zoom_mode("100")
                    event.accept()
                    return True

        return super().eventFilter(watched, event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if self.eventFilter(self, event):
            return
        super().keyPressEvent(event)

    def _show_viewer_context_menu(self, pos: QPoint) -> None:
        """Authentic context menu for SingleMediaViewer (Previous, Next, Rotate, Rescan for faces, etc.)."""
        if self.current_media_id is None:
            return
        menu = QMenu(self)
        menu.setFont(QFont("Segoe UI", 9))
        mw = getattr(self, "main_window", None) or self.window()
        is_mw = hasattr(mw, "rescan_faces_for_selection")

        act_next = menu.addAction("Next")
        act_next.triggered.connect(lambda: self.navigate_requested.emit(1))
        act_prev = menu.addAction("Previous")
        act_prev.triggered.connect(lambda: self.navigate_requested.emit(-1))

        menu.addSeparator()

        act_rot_cw = menu.addAction("Rotate clockwise")
        act_rot_cw.triggered.connect(lambda: self.rotate_media(clockwise=True))
        act_rot_ccw = menu.addAction("Rotate counter-clockwise")
        act_rot_ccw.triggered.connect(lambda: self.rotate_media(clockwise=False))

        menu.addSeparator()

        act_rescan = menu.addAction("Rescan for faces")
        if is_mw:
            act_rescan.triggered.connect(lambda: mw.rescan_faces_for_selection([self.current_media_id]))

        if is_mw and hasattr(mw, "open_file_location_for_media"):
            act_loc = menu.addAction("Open file location")
            act_loc.triggered.connect(lambda: mw.open_file_location_for_media(self.current_media_id))

        menu.addSeparator()

        if is_mw and hasattr(mw, "set_desktop_background_for_media"):
            act_bg = menu.addAction("Set as desktop background")
            act_bg.triggered.connect(lambda: mw.set_desktop_background_for_media(self.current_media_id))

        if is_mw and hasattr(mw, "open_print_dialog") and not self.current_is_video:
            act_print = menu.addAction("Print...")
            act_print.setShortcut("Ctrl+P")
            act_print.triggered.connect(mw.open_print_dialog)

        if is_mw and hasattr(mw, "toggle_right_pane"):
            act_prop = menu.addAction("Properties")
            act_prop.triggered.connect(lambda: mw.toggle_right_pane("info"))

        act_close = menu.addAction("Back to gallery")
        act_close.triggered.connect(self.closed.emit)

        menu.exec(self.graphics_view.mapToGlobal(pos))
