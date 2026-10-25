"""
Interactive Quick Markup, Callouts & Privacy Redaction Tool:
1. Privacy Redact Box / Brush (Mosaic pixelation & Gaussian blur).
2. Vector Arrows & Circular Magnifier Loupe (2.5x zoom callout).
3. Draggable Text Labels with glassmorphic pill background.
4. Flattens cleanly onto image with .OriginalFiles backup support.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any
import cv2
import numpy as np

from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import (
    QColor,
    QPen,
    QBrush,
    QPainter,
    QPainterPath,
    QFont,
    QPolygonF,
    QCursor,
    QImage,
    QPixmap,
)
from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QToolButton,
    QComboBox,
    QColorDialog,
    QInputDialog,
    QLabel,
    QFrame,
    QButtonGroup,
)

from src.imaging.creative_effects import apply_pixelate, apply_blur_redact


@dataclass
class MarkupElement:
    kind: str  # 'pixelate', 'blur', 'arrow', 'magnifier', 'text'
    rect: QRectF | None = None
    start_pt: QPointF | None = None
    end_pt: QPointF | None = None
    color: QColor = field(default_factory=lambda: QColor(255, 50, 50))
    stroke_width: int = 4
    text: str = ""
    zoom_factor: float = 2.5


def flatten_markup_onto_image(
    base_bgr: np.ndarray,
    elements: list[MarkupElement],
) -> np.ndarray:
    """
    Renders all markup elements (pixelation, blur, arrows, magnifiers, text)
    onto the base BGR image, returning the flattened BGR image.
    """
    if base_bgr is None:
        return base_bgr

    result = base_bgr.copy()
    h, w = result.shape[:2]

    # 1. Apply pixelate and blur redactions first
    for el in elements:
        if el.kind == "pixelate" and el.rect is not None:
            rx = int(el.rect.x())
            ry = int(el.rect.y())
            rw = int(el.rect.width())
            rh = int(el.rect.height())
            result = apply_pixelate(result, (rx, ry, rw, rh), block_size=18)
        elif el.kind == "blur" and el.rect is not None:
            rx = int(el.rect.x())
            ry = int(el.rect.y())
            rw = int(el.rect.width())
            rh = int(el.rect.height())
            result = apply_blur_redact(result, (rx, ry, rw, rh), radius=28)

    # 2. Render vector annotations (arrows, magnifiers, text labels) using QPainter
    rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()

    painter = QPainter(qimg)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.TextAntialiasing, True)
    painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

    for el in elements:
        if el.kind == "arrow" and el.start_pt and el.end_pt:
            p1 = el.start_pt
            p2 = el.end_pt
            dx = p2.x() - p1.x()
            dy = p2.y() - p1.y()
            length = math.hypot(dx, dy)
            if length < 5:
                continue

            # Draw shaft
            pen = QPen(el.color, el.stroke_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawLine(p1, p2)

            # Draw arrowhead
            angle = math.atan2(dy, dx)
            arrow_size = max(14.0, el.stroke_width * 3.5)
            left_angle = angle + math.pi * 0.82
            right_angle = angle - math.pi * 0.82

            p_left = QPointF(p2.x() + math.cos(left_angle) * arrow_size, p2.y() + math.sin(left_angle) * arrow_size)
            p_right = QPointF(p2.x() + math.cos(right_angle) * arrow_size, p2.y() + math.sin(right_angle) * arrow_size)

            arrow_head = QPolygonF([p2, p_left, p_right])
            painter.setBrush(QBrush(el.color))
            painter.drawPolygon(arrow_head)

        elif el.kind == "magnifier" and el.rect is not None:
            # Magnifier loupe
            cx = el.rect.center().x()
            cy = el.rect.center().y()
            radius = max(25.0, el.rect.width() / 2.0)

            # Extract region under center from original base image
            zoom = el.zoom_factor
            crop_rad = radius / zoom
            src_x1 = max(0, int(cx - crop_rad))
            src_y1 = max(0, int(cy - crop_rad))
            src_x2 = min(w, int(cx + crop_rad))
            src_y2 = min(h, int(cy + crop_rad))

            if src_x2 > src_x1 and src_y2 > src_y1:
                patch_bgr = base_bgr[src_y1:src_y2, src_x1:src_x2]
                patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
                patch_qimg = QImage(patch_rgb.data, src_x2 - src_x1, src_y2 - src_y1, 3 * (src_x2 - src_x1), QImage.Format.Format_RGB888)
                scaled_patch = patch_qimg.scaled(int(radius * 2), int(radius * 2), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)

                # Clip circular loupe
                painter.save()
                clip_path = QPainterPath()
                clip_path.addEllipse(QPointF(cx, cy), radius, radius)
                painter.setClipPath(clip_path)
                painter.drawImage(QRectF(cx - radius, cy - radius, radius * 2, radius * 2), scaled_patch)
                painter.restore()

                # Chrome/glass ring around magnifier
                painter.setPen(QPen(QColor(255, 255, 255, 220), 4.0))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(QPointF(cx, cy), radius, radius)
                painter.setPen(QPen(QColor(40, 120, 220, 200), 2.0))
                painter.drawEllipse(QPointF(cx, cy), radius - 2, radius - 2)

                # Badge showing "2.5x"
                badge_rect = QRectF(cx + radius * 0.4, cy + radius * 0.4, 46, 20)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(20, 20, 25, 220))
                painter.drawRoundedRect(badge_rect, 10, 10)
                painter.setPen(QPen(QColor(255, 255, 255)))
                font = QFont("Segoe UI", 9, QFont.Bold)
                painter.setFont(font)
                painter.drawText(badge_rect, Qt.AlignCenter, f"{zoom:.1f}x")

        elif el.kind == "text" and el.rect is not None and el.text:
            # Draggable Text Label with glassmorphic pill background
            font = QFont("Segoe UI", 12, QFont.Bold)
            painter.setFont(font)
            metrics = painter.fontMetrics()
            text_rect = metrics.boundingRect(el.text)
            pad_x, pad_y = 12, 6

            rx = el.rect.x()
            ry = el.rect.y()
            pill_w = text_rect.width() + 2 * pad_x
            pill_h = text_rect.height() + 2 * pad_y
            pill_rect = QRectF(rx, ry, pill_w, pill_h)

            # Dark pill backdrop
            painter.setPen(QPen(QColor(255, 255, 255, 160), 1.5))
            painter.setBrush(QColor(15, 18, 24, 215))
            painter.drawRoundedRect(pill_rect, pill_h / 2.0, pill_h / 2.0)

            # Crisp white text
            painter.setPen(QPen(el.color if el.color != QColor(0, 0, 0) else QColor(255, 255, 255)))
            painter.drawText(pill_rect, Qt.AlignCenter, el.text)

    painter.end()

    # Convert back to BGR numpy array
    ptr = qimg.bits()
    arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 3))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


class MarkupFloatingToolbar(QFrame):
    """
    Floating toolbar for Markup, Privacy Redaction, and Callouts inside SingleMediaViewer.
    """
    tool_changed = Signal(str)
    apply_requested = Signal()
    cancel_requested = Signal()
    undo_requested = Signal()
    color_changed = Signal(QColor)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.active_tool: str = "pixelate"
        self.current_color = QColor(255, 50, 50)
        self._init_ui()

    def _init_ui(self) -> None:
        self.setStyleSheet("""
            MarkupFloatingToolbar {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #eef2f7);
                border: 1px solid #7a96df;
                border-radius: 6px;
                padding: 4px;
            }
            QToolButton {
                border: 1px solid transparent;
                border-radius: 3px;
                padding: 4px 8px;
                font-size: 11px;
                font-weight: 500;
                color: #222222;
            }
            QToolButton:hover {
                background: #e0ecfa;
                border: 1px solid #99c0ec;
            }
            QToolButton:checked {
                background: #cce2f7;
                border: 1px solid #3399ff;
                font-weight: bold;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(6)

        lbl = QLabel("Markup & Redact:")
        lbl.setStyleSheet("font-weight: bold; font-size: 11px; color: #1e395b;")
        layout.addWidget(lbl)

        # Tool buttons
        self.btn_group = QButtonGroup(self)
        self.btn_group.setExclusive(True)

        tools = [
            ("pixelate", "Mosaic Pixelate"),
            ("blur", "Privacy Blur"),
            ("arrow", "Arrow Callout"),
            ("magnifier", "Magnifier (2.5x)"),
            ("text", "Text Label"),
        ]

        for tid, tname in tools:
            btn = QToolButton()
            btn.setText(tname)
            btn.setCheckable(True)
            if tid == "pixelate":
                btn.setChecked(True)
            btn.clicked.connect(lambda _, t=tid: self._set_tool(t))
            self.btn_group.addButton(btn)
            layout.addWidget(btn)

        layout.addSpacing(6)

        # Color picker button
        self.btn_color = QToolButton()
        self.btn_color.setText("Color")
        self.btn_color.setToolTip("Choose annotation color")
        self.btn_color.clicked.connect(self._pick_color)
        layout.addWidget(self.btn_color)

        # Undo button
        btn_undo = QToolButton()
        btn_undo.setText("Undo")
        btn_undo.clicked.connect(self.undo_requested.emit)
        layout.addWidget(btn_undo)

        layout.addSpacing(8)

        # Apply & Commit
        btn_apply = QPushButton("Apply")
        btn_apply.setStyleSheet("""
            QPushButton {
                background: #2D72BC; color: white; font-weight: bold;
                border: 1px solid #144e8c; border-radius: 3px; padding: 4px 12px;
            }
            QPushButton:hover { background: #3d86d6; }
        """)
        btn_apply.clicked.connect(self.apply_requested.emit)
        layout.addWidget(btn_apply)

        # Cancel
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setStyleSheet("background: #f0f0f0; border: 1px solid #bbb; border-radius: 3px; padding: 4px 10px;")
        btn_cancel.clicked.connect(self.cancel_requested.emit)
        layout.addWidget(btn_cancel)

    def _set_tool(self, tool_id: str) -> None:
        self.active_tool = tool_id
        self.tool_changed.emit(tool_id)

    def _pick_color(self) -> None:
        col = QColorDialog.getColor(self.current_color, self, "Annotation Color")
        if col.isValid():
            self.current_color = col
            self.color_changed.emit(col)
