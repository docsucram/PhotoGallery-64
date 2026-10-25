"""
RibbonToolbar (QWidget) implementing Windows Live Photo Gallery's 5-Tab Ribbon & Search Scope Header:
- Tabs: Home, Edit, Find, Create, View (with painted WLPG Aero vector icons)
- Home Tab: New (Add folder), Manage (Rotate left/right, Delete), Organize (People tag,
  Descriptive tag, Adjust time, Flag, World Map), Quick find (Horizontal Face Avatar Carousel),
  Share (Slide show)
- View Tab: Panes (Navigation pane, Tag and caption pane), Arrange list (Date, Geotag,
  Person, Name, Tag, Published, Folder, Media type, Rating + Reverse sort),
  Show details (Rating, Caption, Flags, File name, File size, Date taken, Image size),
  Layout Mode (WLPG Classic Grid vs Justified Tessellated Mosaic), Zoom slider
- Search by Scope Bar: Sits above the grid ('Search by: Folder: Pictures | Search all' + instant search input)
"""

from __future__ import annotations

import calendar
import datetime
import math
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QAction,
    QColor,
    QFont,
    QFontMetrics,
    QIcon,
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from src.db.repository import ArrangeMode
from src.imaging.opencv_engine import PhotoAdjustmentRecipe, render_adjusted_image

import src.ui.icon_provider as icon_provider

ARROW_DROPDOWN_PATH = (Path(__file__).resolve().parents[2] / "assets" / "icons" / "arrow_dropdown_mini.svg").as_posix()

_ICON_CACHE: dict[tuple[str, int], QIcon] = {}


def _render_aero_raw(kind: str, size: int = 20) -> QIcon:
    """Renders crisp Windows Live Photo Gallery Aero-style vector icons onto a cached QIcon."""
    key = (kind, size)
    if key in _ICON_CACHE:
        return _ICON_CACHE[key]

    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing, True)
    s = float(size)

    if kind in ("folder", "folder_add", "import"):
        grad = QLinearGradient(0, s * 0.2, 0, s * 0.88)
        grad.setColorAt(0.0, QColor("#FFE88C"))
        grad.setColorAt(1.0, QColor("#E2B034"))
        p.setPen(QPen(QColor("#B6851A"), 1.0))
        p.setBrush(grad)
        p.drawRoundedRect(QRectF(s * 0.10, s * 0.20, s * 0.36, s * 0.22), 2.0, 2.0)
        p.drawRoundedRect(QRectF(s * 0.08, s * 0.30, s * 0.84, s * 0.56), 2.2, 2.2)

        if kind in ("folder_add", "import"):
            cx, cy, r = s * 0.74, s * 0.72, s * 0.22
            p.setPen(QPen(QColor("#1F7A1F"), 1.0))
            p.setBrush(QColor("#4CC94C"))
            p.drawEllipse(QPointF(cx, cy), r, r)
            p.setPen(QPen(QColor("#FFFFFF"), 1.8, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(cx - r * 0.5, cy), QPointF(cx + r * 0.5, cy))
            p.drawLine(QPointF(cx, cy - r * 0.5), QPointF(cx, cy + r * 0.5))

    elif kind in ("calendar", "dates", "dates_grid"):
        p.setPen(QPen(QColor("#6E87A2"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.12, s * 0.16, s * 0.76, s * 0.72), 2.0, 2.0)
        p.setBrush(QColor("#3A78B8"))
        p.drawRoundedRect(QRectF(s * 0.12, s * 0.16, s * 0.76, s * 0.22), 2.0, 2.0)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#5C7A99"))
        for r_idx in range(3 if kind != "calendar" else 2):
            for c_idx in range(4 if kind != "calendar" else 3):
                p.drawRect(
                    QRectF(
                        s * (0.20 + c_idx * (0.16 if kind != "calendar" else 0.20)),
                        s * (0.44 + r_idx * (0.13 if kind != "calendar" else 0.17)),
                        s * (0.09 if kind != "calendar" else 0.11),
                        s * (0.08 if kind != "calendar" else 0.10),
                    )
                )

    elif kind in ("months", "months_grid"):
        p.setPen(QPen(QColor("#45729F"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.12, s * 0.16, s * 0.76, s * 0.72), 2.0, 2.0)
        p.setBrush(QColor("#2468A8"))
        p.drawRoundedRect(QRectF(s * 0.12, s * 0.16, s * 0.76, s * 0.22), 2.0, 2.0)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#5C88B0"))
        for r_i in range(2):
            for c_i in range(2):
                p.drawRect(QRectF(s * (0.22 + c_i * 0.32), s * (0.46 + r_i * 0.18), s * 0.24, s * 0.12))

    elif kind in ("years", "years_stack"):
        p.setPen(QPen(QColor("#45729F"), 1.0))
        p.setBrush(QColor("#E8F1FA"))
        p.drawRoundedRect(QRectF(s * 0.26, s * 0.12, s * 0.62, s * 0.58), 2.0, 2.0)
        p.drawRoundedRect(QRectF(s * 0.18, s * 0.22, s * 0.62, s * 0.58), 2.0, 2.0)
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.10, s * 0.32, s * 0.62, s * 0.58), 2.0, 2.0)
        p.setBrush(QColor("#2468A8"))
        p.drawRoundedRect(QRectF(s * 0.10, s * 0.32, s * 0.62, s * 0.18), 2.0, 2.0)

    elif kind in ("text_search", "binoculars"):
        p.setPen(QPen(QColor("#1E5288"), 1.3))
        p.setBrush(QColor("#4A88C7"))
        p.drawRoundedRect(QRectF(s * 0.18, s * 0.30, s * 0.26, s * 0.52), 3.0, 3.0)
        p.drawRoundedRect(QRectF(s * 0.56, s * 0.30, s * 0.26, s * 0.52), 3.0, 3.0)
        p.setBrush(QColor("#1E3E66"))
        p.drawRoundedRect(QRectF(s * 0.22, s * 0.22, s * 0.18, s * 0.10), 2.0, 2.0)
        p.drawRoundedRect(QRectF(s * 0.60, s * 0.22, s * 0.18, s * 0.10), 2.0, 2.0)
        p.setPen(QPen(QColor("#1E3E66"), 2.0))
        p.drawLine(QPointF(s * 0.44, s * 0.50), QPointF(s * 0.56, s * 0.50))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(255, 255, 255, 160))
        p.drawEllipse(QPointF(s * 0.28, s * 0.70), s * 0.06, s * 0.06)
        p.drawEllipse(QPointF(s * 0.66, s * 0.70), s * 0.06, s * 0.06)

    elif kind == "edited":
        p.setPen(QPen(QColor("#6B5218"), 1.0))
        p.setBrush(QColor("#F0B838"))
        pencil = QPolygonF(
            [
                QPointF(s * 0.76, s * 0.18),
                QPointF(s * 0.84, s * 0.26),
                QPointF(s * 0.34, s * 0.76),
                QPointF(s * 0.20, s * 0.82),
                QPointF(s * 0.26, s * 0.68),
            ]
        )
        p.drawPolygon(pencil)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#333333"))
        tip = QPolygonF(
            [
                QPointF(s * 0.20, s * 0.82),
                QPointF(s * 0.24, s * 0.73),
                QPointF(s * 0.29, s * 0.78),
            ]
        )
        p.drawPolygon(tip)

    elif kind == "media_type":
        p.setPen(QPen(QColor("#2468A8"), 1.0))
        p.setBrush(QColor("#E8F1FA"))
        p.drawRoundedRect(QRectF(s * 0.10, s * 0.24, s * 0.52, s * 0.54), 2.0, 2.0)
        p.setBrush(QColor("#5488BD"))
        p.drawPolygon(
            QPolygonF(
                [
                    QPointF(s * 0.14, s * 0.70),
                    QPointF(s * 0.32, s * 0.46),
                    QPointF(s * 0.44, s * 0.60),
                    QPointF(s * 0.58, s * 0.70),
                ]
            )
        )
        p.setPen(QPen(QColor("#1A4E80"), 1.0))
        p.setBrush(QColor("#4A78A8"))
        p.drawRoundedRect(QRectF(s * 0.54, s * 0.38, s * 0.36, s * 0.42), 2.0, 2.0)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#FFFFFF"))
        tri = QPolygonF(
            [
                QPointF(s * 0.64, s * 0.46),
                QPointF(s * 0.80, s * 0.59),
                QPointF(s * 0.64, s * 0.72),
            ]
        )
        p.drawPolygon(tri)

    elif kind in ("tag", "tag_add"):
        path = QPainterPath()
        path.moveTo(s * 0.14, s * 0.46)
        path.lineTo(s * 0.46, s * 0.14)
        path.lineTo(s * 0.84, s * 0.14)
        path.lineTo(s * 0.84, s * 0.52)
        path.lineTo(s * 0.52, s * 0.84)
        path.closeSubpath()
        grad = QLinearGradient(0, 0, s, s)
        grad.setColorAt(0.0, QColor("#74B2F0"))
        grad.setColorAt(1.0, QColor("#2D6FB7"))
        p.setPen(QPen(QColor("#1E5288"), 1.1))
        p.setBrush(grad)
        p.drawPath(path)
        p.setBrush(QColor("#FFFFFF"))
        p.drawEllipse(QPointF(s * 0.67, s * 0.31), s * 0.07, s * 0.07)

    elif kind == "not_tagged":
        path = QPainterPath()
        path.moveTo(s * 0.14, s * 0.46)
        path.lineTo(s * 0.46, s * 0.14)
        path.lineTo(s * 0.84, s * 0.14)
        path.lineTo(s * 0.84, s * 0.52)
        path.lineTo(s * 0.52, s * 0.84)
        path.closeSubpath()
        p.setPen(QPen(QColor("#7A8A9E"), 1.0))
        p.setBrush(QColor("#D9E2EC"))
        p.drawPath(path)
        p.setPen(QPen(QColor("#C5221F"), 1.6))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QRectF(s * 0.20, s * 0.20, s * 0.60, s * 0.60))
        p.drawLine(QPointF(s * 0.32, s * 0.32), QPointF(s * 0.68, s * 0.68))

    elif kind == "tag_hierarchy":
        p.setPen(QPen(QColor("#356AA0"), 1.0))
        p.setBrush(QColor("#F5FAFF"))
        p.drawLine(QPointF(s * 0.22, s * 0.24), QPointF(s * 0.22, s * 0.76))
        p.drawLine(QPointF(s * 0.22, s * 0.36), QPointF(s * 0.42, s * 0.36))
        p.drawLine(QPointF(s * 0.22, s * 0.68), QPointF(s * 0.42, s * 0.68))
        p.setBrush(QColor("#4B88C4"))
        p.drawRoundedRect(QRectF(s * 0.42, s * 0.24, s * 0.46, s * 0.22), 1.5, 1.5)
        p.drawRoundedRect(QRectF(s * 0.42, s * 0.56, s * 0.46, s * 0.22), 1.5, 1.5)

    elif kind in ("person", "people_tag", "batch_people", "person_question", "person_none"):
        if kind == "batch_people":
            p.setPen(QPen(QColor("#2A5E8A"), 1.0))
            p.setBrush(QColor("#82B8E6"))
            p.drawEllipse(QPointF(s * 0.66, s * 0.30), s * 0.15, s * 0.15)
            p.drawRoundedRect(
                QRectF(s * 0.44, s * 0.50, s * 0.44, s * 0.32), s * 0.14, s * 0.14
            )
        if kind == "person_none":
            grad = QLinearGradient(0, s * 0.15, 0, s * 0.88)
            grad.setColorAt(0.0, QColor("#CFD8DC"))
            grad.setColorAt(1.0, QColor("#90A4AE"))
            p.setPen(QPen(QColor("#78909C"), 1.1))
            p.setBrush(grad)
        else:
            grad = QLinearGradient(0, s * 0.15, 0, s * 0.88)
            grad.setColorAt(0.0, QColor("#5CA4E2"))
            grad.setColorAt(1.0, QColor("#2468A8"))
            p.setPen(QPen(QColor("#1B4E80"), 1.1))
            p.setBrush(grad)
        cx = s * 0.42 if kind == "batch_people" else s * 0.50
        p.drawEllipse(QPointF(cx, s * 0.32), s * 0.18, s * 0.18)
        body = QPainterPath()
        body.moveTo(cx - s * 0.30, s * 0.86)
        body.quadTo(cx - s * 0.28, s * 0.54, cx, s * 0.54)
        body.quadTo(cx + s * 0.28, s * 0.54, cx + s * 0.30, s * 0.86)
        body.closeSubpath()
        p.drawPath(body)

        if kind == "person_question":
            p.setPen(QPen(QColor("#C68400"), 1.0))
            p.setBrush(QColor("#FFD54F"))
            p.drawEllipse(QPointF(s * 0.78, s * 0.72), s * 0.18, s * 0.18)
            p.setPen(QPen(QColor("#5D4037"), 1.2))
            p.setFont(QFont("Segoe UI", int(s * 0.24), QFont.Bold))
            p.drawText(QRectF(s * 0.60, s * 0.54, s * 0.36, s * 0.36), Qt.AlignCenter, "?")
        elif kind == "person_none":
            p.setPen(QPen(QColor("#C9302C"), 1.0))
            p.setBrush(QColor("#FFEBEE"))
            p.drawEllipse(QPointF(s * 0.78, s * 0.72), s * 0.18, s * 0.18)
            p.setPen(QPen(QColor("#C9302C"), 1.8, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(s * 0.68, s * 0.72), QPointF(s * 0.88, s * 0.72))

    elif kind == "caption":
        # Landscape photo card with note/callout matching WLPG
        p.setPen(QPen(QColor("#6A8AA8"), 1.1))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.10, s * 0.12, s * 0.80, s * 0.68), 2.0, 2.0)
        # Inner photo area (sky + hills)
        photo_rect = QRectF(s * 0.16, s * 0.18, s * 0.68, s * 0.38)
        p.setClipRect(photo_rect)
        sky = QLinearGradient(0, s * 0.18, 0, s * 0.45)
        sky.setColorAt(0.0, QColor("#5FA9EE"))
        sky.setColorAt(1.0, QColor("#BBE1FD"))
        p.setPen(Qt.NoPen)
        p.setBrush(sky)
        p.drawRect(photo_rect)
        # Green hills
        p.setBrush(QColor("#48A238"))
        p.drawChord(QRectF(s * 0.12, s * 0.32, s * 0.50, s * 0.36), 0, 180 * 16)
        p.setBrush(QColor("#66BB44"))
        p.drawChord(QRectF(s * 0.38, s * 0.35, s * 0.50, s * 0.32), 0, 180 * 16)
        p.setClipping(False)
        # Caption box at bottom
        p.setPen(QPen(QColor("#2B74B9"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.20, s * 0.52, s * 0.60, s * 0.22), 1.5, 1.5)
        p.setPen(QPen(QColor("#2B74B9"), 1.2, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(s * 0.26, s * 0.59), QPointF(s * 0.70, s * 0.59))
        p.drawLine(QPointF(s * 0.26, s * 0.66), QPointF(s * 0.58, s * 0.66))

    elif kind in ("geotag", "globe"):
        grad = QLinearGradient(0, s * 0.1, s, s * 0.9)
        grad.setColorAt(0.0, QColor("#63B8F2"))
        grad.setColorAt(1.0, QColor("#1D6AB5"))
        p.setPen(QPen(QColor("#18528C"), 1.1))
        p.setBrush(grad)
        p.drawEllipse(QPointF(s * 0.5, s * 0.5), s * 0.38, s * 0.38)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#5CB85C"))
        p.drawEllipse(QPointF(s * 0.40, s * 0.40), s * 0.14, s * 0.10)
        p.drawEllipse(QPointF(s * 0.60, s * 0.56), s * 0.12, s * 0.15)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(255, 255, 255, 130), 0.9))
        p.drawEllipse(QPointF(s * 0.5, s * 0.5), s * 0.20, s * 0.37)
        p.drawLine(QPointF(s * 0.13, s * 0.5), QPointF(s * 0.87, s * 0.5))

    elif kind in ("rotate_ccw", "rotate_cw"):
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#70A4D4"))
        p.drawPolygon(
            QPolygonF(
                [
                    QPointF(s * 0.16, s * 0.82),
                    QPointF(s * 0.44, s * 0.50),
                    QPointF(s * 0.60, s * 0.68),
                    QPointF(s * 0.84, s * 0.82),
                ]
            )
        )
        p.setPen(QPen(QColor("#2468A8"), 1.8, Qt.SolidLine, Qt.RoundCap))
        p.setBrush(Qt.NoBrush)
        rect = QRectF(s * 0.20, s * 0.16, s * 0.60, s * 0.60)
        if kind == "rotate_ccw":
            p.drawArc(rect, 30 * 16, 210 * 16)
            head = QPolygonF(
                [
                    QPointF(s * 0.16, s * 0.40),
                    QPointF(s * 0.34, s * 0.20),
                    QPointF(s * 0.36, s * 0.44),
                ]
            )
        else:
            p.drawArc(rect, -60 * 16, 210 * 16)
            head = QPolygonF(
                [
                    QPointF(s * 0.84, s * 0.40),
                    QPointF(s * 0.66, s * 0.20),
                    QPointF(s * 0.64, s * 0.44),
                ]
            )
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#2468A8"))
        p.drawPolygon(head)

    elif kind in ("zoom_out", "zoom_in"):
        # Magnifying glass with metallic frame and glass reflection
        p.setPen(QPen(QColor("#7A96B2"), 1.4))
        glass_grad = QLinearGradient(s * 0.2, s * 0.2, s * 0.7, s * 0.7)
        glass_grad.setColorAt(0.0, QColor("#EAF4FD"))
        glass_grad.setColorAt(1.0, QColor("#CCE4F7"))
        p.setBrush(glass_grad)
        p.drawEllipse(QPointF(s * 0.44, s * 0.44), s * 0.32, s * 0.32)
        # Handle
        p.setPen(QPen(QColor("#5A6E82"), 3.2, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(s * 0.68, s * 0.68), QPointF(s * 0.88, s * 0.88))
        # Plus or minus sign inside
        p.setPen(QPen(QColor("#1976D2"), 2.2, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(s * 0.28, s * 0.44), QPointF(s * 0.60, s * 0.44))
        if kind == "zoom_in":
            p.drawLine(QPointF(s * 0.44, s * 0.28), QPointF(s * 0.44, s * 0.60))

    elif kind == "fit_window":
        p.setPen(QPen(QColor("#2B78C5"), 1.6, Qt.SolidLine, Qt.RoundCap))
        p.setBrush(Qt.NoBrush)
        p.drawLine(QPointF(s * 0.18, s * 0.36), QPointF(s * 0.18, s * 0.20))
        p.drawLine(QPointF(s * 0.18, s * 0.20), QPointF(s * 0.34, s * 0.20))
        p.drawLine(QPointF(s * 0.66, s * 0.20), QPointF(s * 0.82, s * 0.20))
        p.drawLine(QPointF(s * 0.82, s * 0.20), QPointF(s * 0.82, s * 0.36))
        p.drawLine(QPointF(s * 0.18, s * 0.64), QPointF(s * 0.18, s * 0.80))
        p.drawLine(QPointF(s * 0.18, s * 0.80), QPointF(s * 0.34, s * 0.80))
        p.drawLine(QPointF(s * 0.66, s * 0.80), QPointF(s * 0.82, s * 0.80))
        p.drawLine(QPointF(s * 0.82, s * 0.80), QPointF(s * 0.82, s * 0.64))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#2B78C5"))
        p.drawRect(QRectF(s * 0.42, s * 0.42, s * 0.16, s * 0.16))

    elif kind == "reverse_sort":
        p.setPen(QPen(QColor("#1A5C96"), 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.setBrush(QColor("#2468A8"))
        p.drawLine(QPointF(s * 0.35, s * 0.80), QPointF(s * 0.35, s * 0.24))
        tri_up = QPolygonF([QPointF(s * 0.20, s * 0.38), QPointF(s * 0.35, s * 0.18), QPointF(s * 0.50, s * 0.38)])
        p.drawPolygon(tri_up)
        p.drawLine(QPointF(s * 0.65, s * 0.20), QPointF(s * 0.65, s * 0.76))
        tri_dn = QPolygonF([QPointF(s * 0.50, s * 0.62), QPointF(s * 0.65, s * 0.82), QPointF(s * 0.80, s * 0.62)])
        p.drawPolygon(tri_dn)

    elif kind in ("actual_size", "actual_pixels"):
        p.setPen(QPen(QColor("#0078D7"), 1.6, Qt.SolidLine, Qt.RoundCap))
        p.setBrush(Qt.NoBrush)
        # 4 Corner brackets
        p.drawLine(QPointF(s * 0.16, s * 0.36), QPointF(s * 0.16, s * 0.20))
        p.drawLine(QPointF(s * 0.16, s * 0.20), QPointF(s * 0.32, s * 0.20))
        p.drawLine(QPointF(s * 0.68, s * 0.20), QPointF(s * 0.84, s * 0.20))
        p.drawLine(QPointF(s * 0.84, s * 0.20), QPointF(s * 0.84, s * 0.36))
        p.drawLine(QPointF(s * 0.16, s * 0.64), QPointF(s * 0.16, s * 0.80))
        p.drawLine(QPointF(s * 0.16, s * 0.80), QPointF(s * 0.32, s * 0.80))
        p.drawLine(QPointF(s * 0.68, s * 0.80), QPointF(s * 0.84, s * 0.80))
        p.drawLine(QPointF(s * 0.84, s * 0.80), QPointF(s * 0.84, s * 0.64))
        # Central filled blue rectangle
        p.setPen(QPen(QColor("#005A9E"), 1.0))
        p.setBrush(QColor("#0078D7"))
        p.drawRoundedRect(QRectF(s * 0.36, s * 0.36, s * 0.28, s * 0.28), 1.0, 1.0)
        # 4 inward arrows
        p.setBrush(QColor("#005A9E"))
        p.drawPolygon(QPolygonF([QPointF(s * 0.22, s * 0.22), QPointF(s * 0.32, s * 0.26), QPointF(s * 0.26, s * 0.32)]))
        p.drawPolygon(QPolygonF([QPointF(s * 0.78, s * 0.22), QPointF(s * 0.68, s * 0.26), QPointF(s * 0.74, s * 0.32)]))
        p.drawPolygon(QPolygonF([QPointF(s * 0.22, s * 0.78), QPointF(s * 0.32, s * 0.74), QPointF(s * 0.26, s * 0.68)]))
        p.drawPolygon(QPolygonF([QPointF(s * 0.78, s * 0.78), QPointF(s * 0.68, s * 0.74), QPointF(s * 0.74, s * 0.68)]))

    elif kind in ("name", "sort_name"):
        p.setPen(QPen(QColor("#2468A8"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.12, s * 0.14, s * 0.76, s * 0.72), 2.0, 2.0)
        p.setPen(QColor("#1A5C96"))
        p.setFont(QFont("Segoe UI", int(s * 0.34), QFont.Bold))
        p.drawText(QRectF(0, 0, s, s), Qt.AlignCenter, "AZ")

    elif kind == "published":
        p.setPen(QPen(QColor("#2468A8"), 1.2))
        p.setBrush(QColor("#D6EAF8"))
        p.drawEllipse(QPointF(s * 0.5, s * 0.5), s * 0.36, s * 0.36)
        p.drawLine(QPointF(s * 0.14, s * 0.5), QPointF(s * 0.86, s * 0.5))
        p.drawArc(QRectF(s * 0.30, s * 0.14, s * 0.40, s * 0.72), 0, 360 * 16)

    elif kind == "view_list":
        p.setPen(QPen(QColor("#2B78C5"), 1.8, Qt.SolidLine, Qt.RoundCap))
        for y_f in (0.30, 0.50, 0.70):
            p.drawPoint(QPointF(s * 0.24, s * y_f))
            p.drawLine(QPointF(s * 0.40, s * y_f), QPointF(s * 0.82, s * y_f))

    elif kind in ("prev_media", "next_media"):
        p.setPen(QPen(QColor("#2468A8"), 1.8, Qt.SolidLine, Qt.RoundCap))
        p.setBrush(QColor("#2468A8"))
        if kind == "prev_media":
            p.drawLine(QPointF(s * 0.25, s * 0.25), QPointF(s * 0.25, s * 0.75))
            tri = QPolygonF(
                [
                    QPointF(s * 0.75, s * 0.25),
                    QPointF(s * 0.35, s * 0.50),
                    QPointF(s * 0.75, s * 0.75),
                ]
            )
        else:
            p.drawLine(QPointF(s * 0.75, s * 0.25), QPointF(s * 0.75, s * 0.75))
            tri = QPolygonF(
                [
                    QPointF(s * 0.25, s * 0.25),
                    QPointF(s * 0.65, s * 0.50),
                    QPointF(s * 0.25, s * 0.75),
                ]
            )
        p.drawPolygon(tri)

    elif kind == "delete":
        p.setPen(QPen(QColor("#C9302C"), 2.4, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(s * 0.24, s * 0.24), QPointF(s * 0.76, s * 0.76))
        p.drawLine(QPointF(s * 0.76, s * 0.24), QPointF(s * 0.24, s * 0.76))

    elif kind in ("pin", "pin_active"):
        p.setPen(QPen(QColor("#24548A") if kind == "pin_active" else QColor("#5A738E"), 1.2))
        p.setBrush(QColor("#2B78C5") if kind == "pin_active" else QColor("#8AA4BE"))
        if kind == "pin_active":
            p.drawRoundedRect(QRectF(s * 0.35, s * 0.15, s * 0.30, s * 0.22), 1.5, 1.5)
            p.drawRect(QRectF(s * 0.42, s * 0.37, s * 0.16, s * 0.28))
            p.setPen(QPen(QColor("#1A3A60"), 1.8, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(s * 0.50, s * 0.65), QPointF(s * 0.50, s * 0.90))
        else:
            p.save()
            p.translate(s * 0.5, s * 0.5)
            p.rotate(-45)
            p.drawRoundedRect(QRectF(-s * 0.18, -s * 0.34, s * 0.36, s * 0.20), 1.5, 1.5)
            p.drawRect(QRectF(-s * 0.10, -s * 0.14, s * 0.20, s * 0.24))
            p.setPen(QPen(QColor("#455A6E"), 1.6, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(0, s * 0.10), QPointF(0, s * 0.38))
            p.restore()

    elif kind in ("flag", "flag_on", "flag_off"):
        p.setPen(QPen(QColor("#5A6978"), 1.5, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(s * 0.26, s * 0.16), QPointF(s * 0.26, s * 0.86))
        banner = QPolygonF(
            [
                QPointF(s * 0.26, s * 0.18),
                QPointF(s * 0.80, s * 0.30),
                QPointF(s * 0.26, s * 0.56),
            ]
        )
        if kind == "flag_off":
            p.setPen(QPen(QColor("#7A8B9C"), 1.2))
            p.setBrush(QColor(255, 255, 255, 180))
        else:
            p.setPen(QPen(QColor("#B52B27"), 1.1))
            p.setBrush(QColor("#E64833"))
        p.drawPolygon(banner)

    elif kind in ("star", "rating", "star_filled", "star_empty", "rate"):
        pts: list[QPointF] = []
        cx, cy = s * 0.5, s * 0.5
        outer_r, inner_r = s * 0.44, s * 0.19
        for i in range(10):
            ang = math.radians(-90 + i * 36)
            r = outer_r if i % 2 == 0 else inner_r
            pts.append(QPointF(cx + r * math.cos(ang), cy + r * math.sin(ang)))
        if kind == "star_empty":
            p.setPen(QPen(QColor("#8C9BAE"), 1.2))
            p.setBrush(Qt.NoBrush)
            p.drawPolygon(QPolygonF(pts))
        else:
            grad = QLinearGradient(0, s * 0.1, 0, s * 0.9)
            grad.setColorAt(0.0, QColor("#FFF4A3"))
            grad.setColorAt(0.4, QColor("#F8D038"))
            grad.setColorAt(1.0, QColor("#D99E10"))
            p.setPen(QPen(QColor("#B27B08"), 1.0))
            p.setBrush(grad)
            p.drawPolygon(QPolygonF(pts))
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 255, 255, 110))
            p.drawPolygon(QPolygonF([pts[0], pts[1], QPointF(cx, cy)]))
            p.drawPolygon(QPolygonF([pts[8], pts[9], QPointF(cx, cy)]))

    elif kind == "slideshow":
        # Projector screen on stand matching WLPG
        p.setPen(QPen(QColor("#455A64"), 1.2))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.12, s * 0.12, s * 0.76, s * 0.52), 1.5, 1.5)
        p.setClipRect(QRectF(s * 0.15, s * 0.15, s * 0.70, s * 0.46))
        sky = QLinearGradient(0, s * 0.15, 0, s * 0.45)
        sky.setColorAt(0.0, QColor("#64B5F6"))
        sky.setColorAt(1.0, QColor("#BBDEFB"))
        p.setPen(Qt.NoPen)
        p.setBrush(sky)
        p.drawRect(QRectF(s * 0.15, s * 0.15, s * 0.70, s * 0.46))
        p.setBrush(QColor("#43A047"))
        p.drawChord(QRectF(s * 0.10, s * 0.32, s * 0.55, s * 0.36), 0, 180 * 16)
        p.setBrush(QColor("#66BB6A"))
        p.drawChord(QRectF(s * 0.40, s * 0.35, s * 0.50, s * 0.32), 0, 180 * 16)
        p.setClipping(False)
        p.setPen(QPen(QColor("#455A64"), 1.6, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(s * 0.50, s * 0.64), QPointF(s * 0.50, s * 0.86))
        p.drawLine(QPointF(s * 0.50, s * 0.74), QPointF(s * 0.30, s * 0.88))
        p.drawLine(QPointF(s * 0.50, s * 0.74), QPointF(s * 0.70, s * 0.88))

    elif kind in ("detail_all", "all_details"):
        p.setPen(QPen(QColor("#6A85A0"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRect(QRectF(s * 0.16, s * 0.12, s * 0.68, s * 0.76))
        p.setPen(QPen(QColor("#2B74B9"), 1.2, Qt.SolidLine, Qt.RoundCap))
        for y_f in (0.30, 0.46, 0.62, 0.76):
            p.drawPoint(QPointF(s * 0.30, s * y_f))
            p.drawLine(QPointF(s * 0.42, s * y_f), QPointF(s * 0.72, s * y_f))

    elif kind in ("detail_filename", "filename"):
        p.setPen(QPen(QColor("#6A85A0"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRect(QRectF(s * 0.16, s * 0.12, s * 0.68, s * 0.76))
        p.setPen(QColor("#2468A8"))
        p.setFont(QFont("Segoe UI", int(s * 0.40), QFont.Bold))
        p.drawText(QRectF(s * 0.16, s * 0.16, s * 0.68, s * 0.68), Qt.AlignCenter, "ab")

    elif kind in ("detail_filesize", "filesize"):
        p.setPen(QPen(QColor("#6A85A0"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRect(QRectF(s * 0.16, s * 0.12, s * 0.68, s * 0.76))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#D3E3F5"))
        p.drawRect(QRectF(s * 0.26, s * 0.45, s * 0.48, s * 0.24))
        p.setBrush(QColor("#2B74B9"))
        p.drawRect(QRectF(s * 0.26, s * 0.45, s * 0.28, s * 0.24))

    elif kind in ("detail_date_taken", "date_taken"):
        p.setPen(QPen(QColor("#6A85A0"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.14, s * 0.16, s * 0.72, s * 0.70), 1.5, 1.5)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#D32F2F"))
        p.drawRoundedRect(QRectF(s * 0.14, s * 0.16, s * 0.72, s * 0.22), 1.5, 1.5)
        p.drawRect(QRectF(s * 0.14, s * 0.28, s * 0.72, s * 0.10))
        p.setPen(QPen(QColor("#78909C"), 1.0))
        p.drawLine(QPointF(s * 0.28, s * 0.52), QPointF(s * 0.72, s * 0.52))
        p.drawLine(QPointF(s * 0.28, s * 0.68), QPointF(s * 0.72, s * 0.68))

    elif kind in ("detail_dimensions", "dimensions"):
        p.setPen(QPen(QColor("#6A85A0"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRect(QRectF(s * 0.14, s * 0.18, s * 0.72, s * 0.64))
        p.setPen(QPen(QColor("#2B74B9"), 1.2, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(s * 0.22, s * 0.30), QPointF(s * 0.78, s * 0.30))
        p.drawLine(QPointF(s * 0.78, s * 0.30), QPointF(s * 0.78, s * 0.72))

    elif kind in ("detail_date_modified", "date_modified"):
        p.setPen(QPen(QColor("#6A85A0"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.14, s * 0.16, s * 0.72, s * 0.70), 1.5, 1.5)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#1976D2"))
        p.drawRoundedRect(QRectF(s * 0.14, s * 0.16, s * 0.72, s * 0.22), 1.5, 1.5)
        p.drawRect(QRectF(s * 0.14, s * 0.28, s * 0.72, s * 0.10))
        p.setPen(QPen(QColor("#546E7A"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawEllipse(QPointF(s * 0.62, s * 0.64), s * 0.20, s * 0.20)
        p.drawLine(QPointF(s * 0.62, s * 0.64), QPointF(s * 0.62, s * 0.52))
        p.drawLine(QPointF(s * 0.62, s * 0.64), QPointF(s * 0.72, s * 0.64))

    elif kind == "layout_grid":
        p.setPen(QPen(QColor("#2B6299"), 1.1))
        p.setBrush(QColor("#8ABEF2"))
        for r_i in range(2):
            for c_i in range(2):
                p.drawRect(
                    QRectF(
                        s * (0.14 + c_i * 0.40),
                        s * (0.14 + r_i * 0.40),
                        s * 0.32,
                        s * 0.32,
                    )
                )

    elif kind == "layout_mosaic":
        p.setPen(QPen(QColor("#2B6299"), 1.1))
        p.setBrush(QColor("#69ACEA"))
        p.drawRect(QRectF(s * 0.12, s * 0.14, s * 0.44, s * 0.32))
        p.drawRect(QRectF(s * 0.62, s * 0.14, s * 0.26, s * 0.32))
        p.drawRect(QRectF(s * 0.12, s * 0.54, s * 0.26, s * 0.32))
        p.drawRect(QRectF(s * 0.44, s * 0.54, s * 0.44, s * 0.32))

    elif kind in ("all_media", "all_photos_videos"):
        p.setPen(QPen(QColor("#356B9E"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.10, s * 0.16, s * 0.80, s * 0.68), 2.0, 2.0)
        grad_sky = QLinearGradient(0, s * 0.22, 0, s * 0.55)
        grad_sky.setColorAt(0.0, QColor("#68A6E2"))
        grad_sky.setColorAt(1.0, QColor("#CAE4FB"))
        p.setPen(Qt.NoPen)
        p.setBrush(grad_sky)
        p.drawRect(QRectF(s * 0.16, s * 0.22, s * 0.68, s * 0.36))
        p.setBrush(QColor("#54A045"))
        p.drawChord(QRectF(s * 0.10, s * 0.44, s * 0.80, s * 0.44), 0, 180 * 16)
        p.setBrush(QColor("#245380"))
        p.drawRoundedRect(QRectF(s * 0.08, s * 0.12, s * 0.84, s * 0.18), 1.5, 1.5)
        p.setBrush(QColor("#FFFFFF"))
        for sp_i in range(4):
            p.drawRect(QRectF(s * (0.14 + sp_i * 0.20), s * 0.15, s * 0.11, s * 0.10))

    elif kind in ("pictures_library", "picture_lib"):
        p.setPen(QPen(QColor("#2B689C"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.10, s * 0.16, s * 0.80, s * 0.68), 2.0, 2.0)
        grad_sky = QLinearGradient(0, s * 0.22, 0, s * 0.55)
        grad_sky.setColorAt(0.0, QColor("#5DA2E8"))
        grad_sky.setColorAt(1.0, QColor("#D4E8FC"))
        p.setPen(Qt.NoPen)
        p.setBrush(grad_sky)
        p.drawRect(QRectF(s * 0.16, s * 0.22, s * 0.68, s * 0.40))
        p.setBrush(QColor("#F5C342"))
        p.drawEllipse(QPointF(s * 0.32, s * 0.34), s * 0.10, s * 0.10)
        p.setBrush(QColor("#4EA03C"))
        p.drawChord(QRectF(s * 0.16, s * 0.42, s * 0.68, s * 0.44), 0, 180 * 16)

    elif kind in ("videos_library", "video_lib"):
        p.setPen(QPen(QColor("#1F4B78"), 1.0))
        p.setBrush(QColor("#356B9E"))
        p.drawRoundedRect(QRectF(s * 0.12, s * 0.16, s * 0.76, s * 0.68), 2.0, 2.0)
        p.setBrush(QColor("#FFFFFF"))
        for sp_i in range(3):
            p.drawRect(QRectF(s * 0.16, s * (0.22 + sp_i * 0.20), s * 0.12, s * 0.12))
            p.drawRect(QRectF(s * 0.72, s * (0.22 + sp_i * 0.20), s * 0.12, s * 0.12))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#F6FAFE"))
        tri = QPolygonF([
            QPointF(s * 0.40, s * 0.36),
            QPointF(s * 0.62, s * 0.50),
            QPointF(s * 0.40, s * 0.64),
        ])
        p.drawPolygon(tri)

    elif kind in ("pane_left", "pane_right", "tag_caption_pane"):
        # Window frame with blue titlebar and right panel matching WLPG
        p.setPen(QPen(QColor("#3A6A9C"), 1.1))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.10, s * 0.14, s * 0.80, s * 0.72), 2.0, 2.0)
        # Blue title bar
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#2468A8"))
        p.drawRoundedRect(QRectF(s * 0.10, s * 0.14, s * 0.80, s * 0.18), 2.0, 2.0)
        p.drawRect(QRectF(s * 0.10, s * 0.24, s * 0.80, s * 0.08))
        if kind == "pane_left":
            p.setBrush(QColor("#E1EFFC"))
            p.drawRect(QRectF(s * 0.10, s * 0.32, s * 0.32, s * 0.54))
            p.setPen(QPen(QColor("#8FBEE8"), 1.0))
            p.drawLine(QPointF(s * 0.42, s * 0.32), QPointF(s * 0.42, s * 0.86))
        else:
            # Right tag panel
            p.setBrush(QColor("#E1EFFC"))
            p.drawRect(QRectF(s * 0.58, s * 0.32, s * 0.32, s * 0.54))
            p.setPen(QPen(QColor("#8FBEE8"), 1.0))
            p.drawLine(QPointF(s * 0.58, s * 0.32), QPointF(s * 0.58, s * 0.86))
            # Blue Person silhouette badge overlay
            p.setPen(QPen(QColor("#185084"), 1.0))
            grad = QLinearGradient(0, s * 0.38, 0, s * 0.82)
            grad.setColorAt(0.0, QColor("#5CA4E2"))
            grad.setColorAt(1.0, QColor("#1F65A8"))
            p.setBrush(grad)
            p.drawEllipse(QPointF(s * 0.38, s * 0.44), s * 0.13, s * 0.13)
            body = QPainterPath()
            body.moveTo(s * 0.18, s * 0.84)
            body.quadTo(s * 0.20, s * 0.60, s * 0.38, s * 0.60)
            body.quadTo(s * 0.56, s * 0.60, s * 0.58, s * 0.84)
            body.closeSubpath()
            p.drawPath(body)

    elif kind == "auto_adjust":
        p.setPen(QPen(QColor("#7088A0"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.12, s * 0.16, s * 0.76, s * 0.68), 2.0, 2.0)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#6CB2E8"))
        p.drawRect(QRectF(s * 0.18, s * 0.22, s * 0.64, s * 0.56))
        p.setBrush(QColor("#48A038"))
        p.drawChord(QRectF(s * 0.16, s * 0.44, s * 0.68, s * 0.40), 0, 180 * 16)
        p.setBrush(QColor("#F9C846"))
        p.setPen(QPen(QColor("#D48817"), 0.8))
        cx, cy = s * 0.66, s * 0.34
        pts = [
            QPointF(cx, cy - s * 0.14), QPointF(cx + s * 0.04, cy - s * 0.04),
            QPointF(cx + s * 0.14, cy), QPointF(cx + s * 0.04, cy + s * 0.04),
            QPointF(cx, cy + s * 0.14), QPointF(cx - s * 0.04, cy + s * 0.04),
            QPointF(cx - s * 0.14, cy), QPointF(cx - s * 0.04, cy - s * 0.04),
        ]
        p.drawPolygon(QPolygonF(pts))

    elif kind == "straighten":
        p.save()
        p.translate(s * 0.5, s * 0.5)
        p.rotate(-8)
        p.translate(-s * 0.5, -s * 0.5)
        p.setPen(QPen(QColor("#7088A0"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.16, s * 0.18, s * 0.68, s * 0.60), 2.0, 2.0)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#6CB2E8"))
        p.drawRect(QRectF(s * 0.22, s * 0.24, s * 0.56, s * 0.48))
        p.setBrush(QColor("#48A038"))
        p.drawChord(QRectF(s * 0.20, s * 0.44, s * 0.60, s * 0.36), 0, 180 * 16)
        p.restore()
        p.setPen(QPen(QColor("#0078D7"), 1.6, Qt.SolidLine, Qt.RoundCap))
        path = QPainterPath()
        path.arcMoveTo(QRectF(s * 0.06, s * 0.06, s * 0.88, s * 0.88), 120)
        path.arcTo(QRectF(s * 0.06, s * 0.06, s * 0.88, s * 0.88), 120, -50)
        p.drawPath(path)

    elif kind == "noise_reduction":
        p.setPen(QPen(QColor("#7088A0"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.12, s * 0.16, s * 0.76, s * 0.68), 2.0, 2.0)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#6EB4F0"))
        p.drawRect(QRectF(s * 0.18, s * 0.22, s * 0.64, s * 0.56))
        p.setBrush(QColor("#2C6BA8"))
        p.drawPolygon(QPolygonF([QPointF(s * 0.18, s * 0.78), QPointF(s * 0.42, s * 0.42), QPointF(s * 0.66, s * 0.78)]))
        p.setBrush(QColor("#4A88C4"))
        p.drawPolygon(QPolygonF([QPointF(s * 0.46, s * 0.78), QPointF(s * 0.68, s * 0.48), QPointF(s * 0.82, s * 0.78)]))

    elif kind in ("adjust_color", "color_adjust", "color"):
        cx, cy = s * 0.5, s * 0.5
        r = s * 0.34
        colors = [QColor("#E81123"), QColor("#FFB900"), QColor("#107C41"), QColor("#0078D7")]
        for i, c in enumerate(colors):
            p.setPen(Qt.NoPen)
            p.setBrush(c)
            p.drawPie(QRectF(cx - r, cy - r, r * 2, r * 2), i * 90 * 16, 90 * 16)
        p.setPen(QPen(QColor("#404040"), 1.0))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(cx, cy), r, r)

    elif kind in ("color_spectrum", "rainbow_spectrum"):
        grad = QLinearGradient(s * 0.12, 0, s * 0.88, 0)
        grad.setColorAt(0.0, QColor("#E81123"))
        grad.setColorAt(0.2, QColor("#FF8C00"))
        grad.setColorAt(0.4, QColor("#FFD700"))
        grad.setColorAt(0.6, QColor("#107C41"))
        grad.setColorAt(0.8, QColor("#0078D7"))
        grad.setColorAt(1.0, QColor("#881798"))
        p.setPen(QPen(QColor("#708090"), 1.0))
        p.setBrush(grad)
        p.drawRoundedRect(QRectF(s * 0.12, s * 0.12, s * 0.76, s * 0.76), 2.0, 2.0)

    elif kind in ("adjust_exposure", "exposure_adjust", "exposure"):
        cx, cy = s * 0.5, s * 0.5
        r = s * 0.34
        p.setPen(QPen(QColor("#242424"), 1.4))
        p.setBrush(QColor("#FFFFFF"))
        p.drawPie(QRectF(cx - r, cy - r, r * 2, r * 2), 90 * 16, 180 * 16)
        p.setBrush(QColor("#242424"))
        p.drawPie(QRectF(cx - r, cy - r, r * 2, r * 2), 270 * 16, 180 * 16)

    elif kind == "red_eye":
        p.setPen(QPen(QColor("#2B5E8C"), 1.5))
        p.setBrush(QColor("#FFFFFF"))
        p.drawChord(QRectF(s * 0.12, s * 0.28, s * 0.76, s * 0.44), 0, 180 * 16)
        p.drawChord(QRectF(s * 0.12, s * 0.28, s * 0.76, s * 0.44), 180 * 16, 180 * 16)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#D82626"))
        p.drawEllipse(QPointF(s * 0.5, s * 0.5), s * 0.14, s * 0.14)
        p.setBrush(QColor("#222222"))
        p.drawEllipse(QPointF(s * 0.5, s * 0.5), s * 0.07, s * 0.07)

    elif kind == "retouch":
        p.setPen(QPen(QColor("#B8860B"), 1.2))
        p.setBrush(QColor("#F5DEB3"))
        p.save()
        p.translate(s * 0.5, s * 0.5)
        p.rotate(45)
        p.drawRoundedRect(QRectF(-s * 0.34, -s * 0.14, s * 0.68, s * 0.28), 4.0, 4.0)
        p.setBrush(QColor("#DEB887"))
        p.drawRect(QRectF(-s * 0.10, -s * 0.14, s * 0.20, s * 0.28))
        p.restore()
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#0078D7"))
        p.drawEllipse(QPointF(s * 0.76, s * 0.24), s * 0.06, s * 0.06)
        p.drawEllipse(QPointF(s * 0.24, s * 0.76), s * 0.05, s * 0.05)

    elif kind == "fine_tune":
        p.setPen(QPen(QColor("#5A7A9A"), 1.4, Qt.SolidLine, Qt.RoundCap))
        for x_f in (0.28, 0.50, 0.72):
            p.drawLine(QPointF(s * x_f, s * 0.20), QPointF(s * x_f, s * 0.80))
        p.setPen(QPen(QColor("#005A9E"), 1.0))
        p.setBrush(QColor("#0078D7"))
        p.drawRoundedRect(QRectF(s * 0.21, s * 0.32, s * 0.14, s * 0.10), 1.5, 1.5)
        p.drawRoundedRect(QRectF(s * 0.43, s * 0.56, s * 0.14, s * 0.10), 1.5, 1.5)
        p.drawRoundedRect(QRectF(s * 0.65, s * 0.38, s * 0.14, s * 0.10), 1.5, 1.5)

    elif kind in ("revert_original", "revert"):
        p.setPen(QPen(QColor("#7088A0"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.14, s * 0.18, s * 0.72, s * 0.64), 2.0, 2.0)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#EAF2FA"))
        p.drawRect(QRectF(s * 0.20, s * 0.24, s * 0.60, s * 0.52))
        p.setPen(QPen(QColor("#0078D7"), 2.0, Qt.SolidLine, Qt.RoundCap))
        p.setBrush(Qt.NoBrush)
        path = QPainterPath()
        path.arcMoveTo(QRectF(s * 0.24, s * 0.28, s * 0.52, s * 0.52), 40)
        path.arcTo(QRectF(s * 0.24, s * 0.28, s * 0.52, s * 0.52), 40, 160)
        p.drawPath(path)
        p.setBrush(QColor("#0078D7"))
        p.drawPolygon(QPolygonF([QPointF(s * 0.20, s * 0.48), QPointF(s * 0.38, s * 0.36), QPointF(s * 0.38, s * 0.60)]))

    elif kind == "view_file":
        p.setPen(QPen(QColor("#7088A0"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.12, s * 0.16, s * 0.76, s * 0.68), 2.0, 2.0)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#5EA8E0"))
        p.drawRect(QRectF(s * 0.18, s * 0.22, s * 0.64, s * 0.56))
        p.setBrush(QColor("#3BA030"))
        p.drawChord(QRectF(s * 0.16, s * 0.44, s * 0.68, s * 0.42), 0, 180 * 16)
        p.setBrush(QColor("#F9C846"))
        p.drawEllipse(QPointF(s * 0.66, s * 0.36), s * 0.08, s * 0.08)

    elif kind == "make_copy":
        p.setPen(QPen(QColor("#7088A0"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.26, s * 0.12, s * 0.62, s * 0.56), 2.0, 2.0)
        p.drawRoundedRect(QRectF(s * 0.12, s * 0.26, s * 0.62, s * 0.56), 2.0, 2.0)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#107C41"))
        p.drawEllipse(QPointF(s * 0.72, s * 0.72), s * 0.18, s * 0.18)
        p.setPen(QPen(QColor("#FFFFFF"), 2.0, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(s * 0.62, s * 0.72), QPointF(s * 0.82, s * 0.72))
        p.drawLine(QPointF(s * 0.72, s * 0.62), QPointF(s * 0.72, s * 0.82))

    elif kind in ("close_file", "close_viewer"):
        p.setPen(QPen(QColor("#C42B1C"), 2.6, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(s * 0.24, s * 0.24), QPointF(s * 0.76, s * 0.76))
        p.drawLine(QPointF(s * 0.76, s * 0.24), QPointF(s * 0.24, s * 0.76))

    elif kind == "crop":
        p.setPen(QPen(QColor("#2A5E8A"), 1.8))
        p.drawLine(QPointF(s * 0.26, s * 0.12), QPointF(s * 0.26, s * 0.74))
        p.drawLine(QPointF(s * 0.26, s * 0.74), QPointF(s * 0.88, s * 0.74))
        p.drawLine(QPointF(s * 0.12, s * 0.26), QPointF(s * 0.74, s * 0.26))
        p.drawLine(QPointF(s * 0.74, s * 0.26), QPointF(s * 0.74, s * 0.88))

    elif kind == "options":
        p.setPen(QPen(QColor("#486581"), 1.6))
        p.setBrush(QColor("#AFC6DD"))
        p.drawEllipse(QPointF(s * 0.5, s * 0.5), s * 0.30, s * 0.30)
        p.setBrush(QColor("#F5F8FC"))
        p.drawEllipse(QPointF(s * 0.5, s * 0.5), s * 0.13, s * 0.13)

    elif kind in ("nav_back", "nav_forward"):
        grad = QLinearGradient(0, 0, 0, s)
        grad.setColorAt(0.0, QColor("#44A5F5"))
        grad.setColorAt(1.0, QColor("#0A68C2"))
        p.setPen(QPen(QColor("#0B4A8C"), 1.0))
        p.setBrush(grad)
        p.drawEllipse(QRectF(s * 0.08, s * 0.08, s * 0.84, s * 0.84))
        p.setPen(QPen(QColor("#FFFFFF"), 2.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        if kind == "nav_back":
            p.drawLine(QPointF(s * 0.58, s * 0.28), QPointF(s * 0.38, s * 0.50))
            p.drawLine(QPointF(s * 0.38, s * 0.50), QPointF(s * 0.58, s * 0.72))
        else:
            p.drawLine(QPointF(s * 0.42, s * 0.28), QPointF(s * 0.62, s * 0.50))
            p.drawLine(QPointF(s * 0.62, s * 0.50), QPointF(s * 0.42, s * 0.72))
        p.end()

        # Disabled pixmap
        dis_pix = QPixmap(size, size)
        dis_pix.fill(Qt.transparent)
        dp = QPainter(dis_pix)
        dp.setRenderHint(QPainter.Antialiasing, True)
        d_grad = QLinearGradient(0, 0, 0, s)
        d_grad.setColorAt(0.0, QColor("#E8EFF7"))
        d_grad.setColorAt(1.0, QColor("#D0DDEB"))
        dp.setPen(QPen(QColor("#B8CADB"), 1.0))
        dp.setBrush(d_grad)
        dp.drawEllipse(QRectF(s * 0.08, s * 0.08, s * 0.84, s * 0.84))
        dp.setPen(QPen(QColor("#8FA5B8"), 2.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        if kind == "nav_back":
            dp.drawLine(QPointF(s * 0.58, s * 0.28), QPointF(s * 0.38, s * 0.50))
            dp.drawLine(QPointF(s * 0.38, s * 0.50), QPointF(s * 0.58, s * 0.72))
        else:
            dp.drawLine(QPointF(s * 0.42, s * 0.28), QPointF(s * 0.62, s * 0.50))
            dp.drawLine(QPointF(s * 0.62, s * 0.50), QPointF(s * 0.42, s * 0.72))
        dp.end()

        icon = QIcon(pix)
        icon.addPixmap(dis_pix, QIcon.Disabled, QIcon.Off)
        icon.addPixmap(dis_pix, QIcon.Disabled, QIcon.On)
        _ICON_CACHE[key] = icon
        return icon

    elif kind in ("undo", "redo"):
        p.setPen(QPen(QColor("#1D5A92"), 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.setBrush(Qt.NoBrush)
        path = QPainterPath()
        if kind == "undo":
            path.arcMoveTo(QRectF(s * 0.22, s * 0.24, s * 0.60, s * 0.60), 40)
            path.arcTo(QRectF(s * 0.22, s * 0.24, s * 0.60, s * 0.60), 40, 160)
            p.drawPath(path)
            p.setBrush(QColor("#1D5A92"))
            tri = QPolygonF([QPointF(s * 0.16, s * 0.44), QPointF(s * 0.34, s * 0.30), QPointF(s * 0.34, s * 0.56)])
            p.drawPolygon(tri)
        else:
            path.arcMoveTo(QRectF(s * 0.18, s * 0.24, s * 0.60, s * 0.60), 140)
            path.arcTo(QRectF(s * 0.18, s * 0.24, s * 0.60, s * 0.60), 140, -160)
            p.drawPath(path)
            p.setBrush(QColor("#1D5A92"))
            tri = QPolygonF([QPointF(s * 0.84, s * 0.44), QPointF(s * 0.66, s * 0.30), QPointF(s * 0.66, s * 0.56)])
            p.drawPolygon(tri)
        p.end()

        dis_pix = QPixmap(size, size)
        dis_pix.fill(Qt.transparent)
        dp = QPainter(dis_pix)
        dp.setRenderHint(QPainter.Antialiasing, True)
        dp.setPen(QPen(QColor("#9EB4CA"), 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        dp.setBrush(Qt.NoBrush)
        dpath = QPainterPath()
        if kind == "undo":
            dpath.arcMoveTo(QRectF(s * 0.22, s * 0.24, s * 0.60, s * 0.60), 40)
            dpath.arcTo(QRectF(s * 0.22, s * 0.24, s * 0.60, s * 0.60), 40, 160)
            dp.drawPath(dpath)
            dp.setBrush(QColor("#9EB4CA"))
            tri = QPolygonF([QPointF(s * 0.16, s * 0.44), QPointF(s * 0.34, s * 0.30), QPointF(s * 0.34, s * 0.56)])
            dp.drawPolygon(tri)
        else:
            dpath.arcMoveTo(QRectF(s * 0.18, s * 0.24, s * 0.60, s * 0.60), 140)
            dpath.arcTo(QRectF(s * 0.18, s * 0.24, s * 0.60, s * 0.60), 140, -160)
            dp.drawPath(dpath)
            dp.setBrush(QColor("#9EB4CA"))
            tri = QPolygonF([QPointF(s * 0.84, s * 0.44), QPointF(s * 0.66, s * 0.30), QPointF(s * 0.66, s * 0.56)])
            dp.drawPolygon(tri)
        dp.end()

        icon = QIcon(pix)
        icon.addPixmap(dis_pix, QIcon.Disabled, QIcon.Off)
        icon.addPixmap(dis_pix, QIcon.Disabled, QIcon.On)
        _ICON_CACHE[key] = icon
        return icon

    elif kind == "qat_menu":
        p.setPen(QPen(QColor("#2C4661"), 1.2, Qt.SolidLine, Qt.FlatCap))
        p.drawLine(QPointF(s * 0.15, s * 0.28), QPointF(s * 0.85, s * 0.28))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#2C4661"))
        tri = QPolygonF([
            QPointF(s * 0.22, s * 0.44),
            QPointF(s * 0.78, s * 0.44),
            QPointF(s * 0.50, s * 0.74),
        ])
        p.drawPolygon(tri)

    elif kind == "select_all":
        p.setPen(QPen(QColor("#245888"), 1.2, Qt.DashLine))
        p.setBrush(QColor(210, 230, 252, 100))
        p.drawRoundedRect(QRectF(s * 0.12, s * 0.12, s * 0.76, s * 0.76), 2.0, 2.0)
        p.setPen(QPen(QColor("#167A16"), 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.drawLine(QPointF(s * 0.30, s * 0.52), QPointF(s * 0.44, s * 0.68))
        p.drawLine(QPointF(s * 0.44, s * 0.68), QPointF(s * 0.72, s * 0.32))

    elif kind == "cut":
        p.setPen(QPen(QColor("#45729F"), 1.2, Qt.SolidLine, Qt.RoundCap))
        p.drawEllipse(QPointF(s * 0.32, s * 0.72), s * 0.13, s * 0.13)
        p.drawEllipse(QPointF(s * 0.68, s * 0.72), s * 0.13, s * 0.13)
        p.drawLine(QPointF(s * 0.38, s * 0.62), QPointF(s * 0.70, s * 0.20))
        p.drawLine(QPointF(s * 0.62, s * 0.62), QPointF(s * 0.30, s * 0.20))
        p.setBrush(QColor("#2B7CD6"))
        p.drawEllipse(QPointF(s * 0.50, s * 0.43), s * 0.04, s * 0.04)

    elif kind == "copy":
        p.setPen(QPen(QColor("#45729F"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.28, s * 0.12, s * 0.56, s * 0.60), 1.5, 1.5)
        p.drawRoundedRect(QRectF(s * 0.16, s * 0.28, s * 0.56, s * 0.60), 1.5, 1.5)

    elif kind == "paste":
        p.setPen(QPen(QColor("#8A6828"), 1.0))
        p.setBrush(QColor("#C99A46"))
        p.drawRoundedRect(QRectF(s * 0.20, s * 0.20, s * 0.60, s * 0.70), 2.0, 2.0)
        p.setPen(QPen(QColor("#45729F"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.28, s * 0.32, s * 0.48, s * 0.52), 1.5, 1.5)
        p.setBrush(QColor("#C0C0C0"))
        p.drawRoundedRect(QRectF(s * 0.36, s * 0.14, s * 0.28, s * 0.14), 1.5, 1.5)

    elif kind == "help":
        p.setPen(QPen(QColor("#246BB2"), 1.0))
        p.setBrush(QColor("#2B7CD6"))
        p.drawEllipse(QRectF(s * 0.12, s * 0.12, s * 0.76, s * 0.76))
        p.setPen(QPen(QColor("#FFFFFF"), 2.0, Qt.SolidLine, Qt.RoundCap))
        p.setFont(QFont("Segoe UI", int(s * 0.52), QFont.Bold))
        p.drawText(QRectF(0, 0, s, s), Qt.AlignCenter, "?")

    elif kind in ("gallery", "wlpg_app", "photo_gallery"):
        # Authentic Windows Live Photo Gallery overlapping orange & cyan photo cards
        # Back photo (orange)
        p.setPen(QPen(QColor("#D96B14"), 1.0))
        p.setBrush(QColor("#F58220"))
        p.drawRoundedRect(QRectF(s * 0.10, s * 0.12, s * 0.58, s * 0.58), 2.0, 2.0)
        p.setBrush(QColor("#FFFFFF"))
        p.drawRect(QRectF(s * 0.14, s * 0.16, s * 0.50, s * 0.40))
        p.setBrush(QColor("#E87010"))
        p.drawEllipse(QPointF(s * 0.46, s * 0.28), s * 0.08, s * 0.08)

        # Front photo (cyan/blue)
        p.setPen(QPen(QColor("#1A6FA8"), 1.0))
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.32, s * 0.30, s * 0.58, s * 0.58), 2.0, 2.0)
        p.setBrush(QColor("#2D9BE8"))
        p.drawRect(QRectF(s * 0.36, s * 0.34, s * 0.50, s * 0.38))
        p.setBrush(QColor("#F5C538"))
        p.drawEllipse(QPointF(s * 0.48, s * 0.44), s * 0.07, s * 0.07)
        p.setBrush(QColor("#4EA632"))
        p.drawChord(QRectF(s * 0.36, s * 0.50, s * 0.50, s * 0.38), 0, 180 * 16)


    elif kind in ("rename", "rename_file", "rename_photo"):
        # Document sheet
        doc_pen = QPen(QColor("#7A96B2"), 1.0)
        p.setPen(doc_pen)
        p.setBrush(QColor("#FFFFFF"))
        path = QPainterPath()
        path.moveTo(s * 0.16, s * 0.12)
        path.lineTo(s * 0.54, s * 0.12)
        path.lineTo(s * 0.70, s * 0.28)
        path.lineTo(s * 0.70, s * 0.86)
        path.lineTo(s * 0.16, s * 0.86)
        path.closeSubpath()
        p.drawPath(path)
        flap = QPainterPath()
        flap.moveTo(s * 0.54, s * 0.12)
        flap.lineTo(s * 0.54, s * 0.28)
        flap.lineTo(s * 0.70, s * 0.28)
        p.setBrush(QColor("#D6E6F5"))
        p.drawPath(flap)
        p.setPen(QPen(QColor("#A0B8D0"), 1.0))
        p.drawLine(QPointF(s * 0.24, s * 0.40), QPointF(s * 0.62, s * 0.40))
        p.drawLine(QPointF(s * 0.24, s * 0.54), QPointF(s * 0.52, s * 0.54))
        p.save()
        p.translate(s * 0.60, s * 0.65)
        p.rotate(-45)
        p.setPen(QPen(QColor("#B87700"), 1.0))
        p.setBrush(QColor("#F5B82E"))
        p.drawRect(QRectF(-s * 0.08, -s * 0.32, s * 0.16, s * 0.36))
        wood = QPainterPath()
        wood.moveTo(-s * 0.08, s * 0.04)
        wood.lineTo(s * 0.08, s * 0.04)
        wood.lineTo(0, s * 0.18)
        wood.closeSubpath()
        p.setPen(QPen(QColor("#C89B65"), 1.0))
        p.setBrush(QColor("#F5D7B5"))
        p.drawPath(wood)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#2C2C2C"))
        ptip = QPainterPath()
        ptip.moveTo(-s * 0.03, s * 0.12)
        ptip.lineTo(s * 0.03, s * 0.12)
        ptip.lineTo(0, s * 0.18)
        ptip.closeSubpath()
        p.drawPath(ptip)
        p.restore()

    elif kind in ("resize", "resize_photo"):
        p.setPen(QPen(QColor("#246BB2"), 1.0))
        p.setBrush(QColor("#E8F3FD"))
        p.drawRoundedRect(QRectF(s * 0.10, s * 0.12, s * 0.56, s * 0.56), 2.0, 2.0)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#F5C538"))
        p.drawEllipse(QPointF(s * 0.24, s * 0.26), s * 0.06, s * 0.06)
        p.setBrush(QColor("#4EA632"))
        mount = QPainterPath()
        mount.moveTo(s * 0.12, s * 0.66)
        mount.lineTo(s * 0.32, s * 0.40)
        mount.lineTo(s * 0.48, s * 0.56)
        mount.lineTo(s * 0.64, s * 0.38)
        mount.lineTo(s * 0.64, s * 0.66)
        mount.closeSubpath()
        p.drawPath(mount)
        arr_pen = QPen(QColor("#D9530F"), 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        p.setPen(arr_pen)
        p.drawLine(QPointF(s * 0.52, s * 0.52), QPointF(s * 0.88, s * 0.88))
        p.drawLine(QPointF(s * 0.88, s * 0.68), QPointF(s * 0.88, s * 0.88))
        p.drawLine(QPointF(s * 0.68, s * 0.88), QPointF(s * 0.88, s * 0.88))
        p.drawLine(QPointF(s * 0.52, s * 0.72), QPointF(s * 0.52, s * 0.52))
        p.drawLine(QPointF(s * 0.72, s * 0.52), QPointF(s * 0.52, s * 0.52))

    elif kind in ("rated_higher", "arrow_up"):
        p.setPen(QPen(QColor("#246BB2"), 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.drawLine(QPointF(s * 0.50, s * 0.22), QPointF(s * 0.50, s * 0.78))
        p.drawLine(QPointF(s * 0.24, s * 0.46), QPointF(s * 0.50, s * 0.20))
        p.drawLine(QPointF(s * 0.76, s * 0.46), QPointF(s * 0.50, s * 0.20))

    elif kind in ("rated_lower", "arrow_down"):
        p.setPen(QPen(QColor("#246BB2"), 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.drawLine(QPointF(s * 0.50, s * 0.22), QPointF(s * 0.50, s * 0.78))
        p.drawLine(QPointF(s * 0.24, s * 0.54), QPointF(s * 0.50, s * 0.80))
        p.drawLine(QPointF(s * 0.76, s * 0.54), QPointF(s * 0.50, s * 0.80))

    elif kind in ("rated_only", "equals"):
        p.setPen(QPen(QColor("#246BB2"), 2.0, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(s * 0.20, s * 0.38), QPointF(s * 0.80, s * 0.38))
        p.drawLine(QPointF(s * 0.20, s * 0.62), QPointF(s * 0.80, s * 0.62))

    elif kind in ("boolean_and", "plus"):
        p.setPen(QPen(QColor("#246BB2"), 2.0, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(s * 0.50, s * 0.20), QPointF(s * 0.50, s * 0.80))
        p.drawLine(QPointF(s * 0.20, s * 0.50), QPointF(s * 0.80, s * 0.50))

    elif kind in ("boolean_or", "slash"):
        p.setPen(QPen(QColor("#246BB2"), 2.0, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(s * 0.25, s * 0.80), QPointF(s * 0.75, s * 0.20))

    elif kind == "print":
        # Classic Aero Printer icon
        p.setPen(QPen(QColor("#406892"), 1.0))
        # Top paper feeder sheet
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(QRectF(s * 0.28, s * 0.12, s * 0.44, s * 0.35), 1.0, 1.0)
        # Printer body
        p.setBrush(QColor("#E8F1FC"))
        p.drawRoundedRect(QRectF(s * 0.14, s * 0.36, s * 0.72, s * 0.42), 2.5, 2.5)
        # Output slot
        p.setBrush(QColor("#244268"))
        p.drawRect(QRectF(s * 0.24, s * 0.60, s * 0.52, s * 0.12))
        # Paper coming out
        p.setBrush(QColor("#FFFFFF"))
        p.drawRect(QRectF(s * 0.28, s * 0.64, s * 0.44, s * 0.24))
        # Blue status LED
        p.setBrush(QColor("#3399FF"))
        p.drawEllipse(QRectF(s * 0.72, s * 0.44, s * 0.08, s * 0.08))

    else:
        p.setPen(QPen(QColor("#3E6B99"), 1.1))
        p.setBrush(QColor("#D8EAF9"))
        p.drawRoundedRect(QRectF(s * 0.12, s * 0.18, s * 0.76, s * 0.64), 2.0, 2.0)

    p.end()
    icon = QIcon(pix)
    _ICON_CACHE[key] = icon
    return icon


icon_provider.set_aero_renderer(_render_aero_raw)


def make_wlpg_icon(kind: str, size: int = 20) -> QIcon:
    """Returns a High-DPI icon respecting the active icon theme (WLPG Aero, Fluent, Minimalist)."""
    return icon_provider.get_icon(kind, size)


def make_person_avatar_pixmap(
    name: str, blob: bytes | None, size: int = 38, circular: bool = False
) -> QPixmap:
    """Renders a framed WLPG face avatar pixmap from JPEG/PNG bytes (or a silhouette fallback)."""
    canvas = QPixmap(size, size)
    canvas.fill(Qt.transparent)
    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

    frame_rect = QRectF(1.0, 1.0, size - 2.0, size - 2.0)
    painter.setPen(QPen(QColor("#9EB6CE"), 1.0))
    painter.setBrush(QColor("#FFFFFF"))
    painter.drawRoundedRect(frame_rect, 3.0, 3.0)

    inner_rect = QRectF(2.5, 2.5, size - 5.0, size - 5.0)
    clip = QPainterPath()
    clip.addRoundedRect(inner_rect, 2.0, 2.0)
    painter.setClipPath(clip)

    drew_img = False
    if blob:
        qimg = QImage.fromData(blob)
        if not qimg.isNull():
            scaled = qimg.scaled(
                int(inner_rect.width()),
                int(inner_rect.height()),
                Qt.KeepAspectRatioByExpanding,
                Qt.SmoothTransformation,
            )
            sx = max(0, (scaled.width() - int(inner_rect.width())) // 2)
            sy = max(0, (scaled.height() - int(inner_rect.height())) // 2)
            painter.drawImage(
                inner_rect.topLeft(),
                scaled,
                QRectF(sx, sy, inner_rect.width(), inner_rect.height()),
            )
            drew_img = True

    if not drew_img:
        painter.fillRect(inner_rect, QColor("#DCEBFA"))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#5B9BD5"))
        s = float(size)
        painter.drawEllipse(QPointF(s * 0.5, s * 0.38), s * 0.17, s * 0.17)
        painter.drawEllipse(QPointF(s * 0.5, s * 0.82), s * 0.28, s * 0.24)

    painter.end()
    return canvas


def make_default_variation_base_image(w: int = 54, h: int = 38) -> np.ndarray:
    """Generates the authentic WLPG poppy/flower reference image for adjustment menus and effect icons."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        factor = y / float(h)
        img[y, :] = [int(175 - factor * 40), int(205 - factor * 30), int(235 - factor * 20)]
    for y in range(int(h * 0.65), h):
        img[y, :] = [int(35 + (y - h * 0.65) * 5), int(135 - (y - h * 0.65) * 3), int(45)]
    cx, cy = int(w * 0.48), int(h * 0.52)
    cv2.line(img, (cx, h), (cx, cy + 6), (25, 110, 30), 2)
    for ang in range(0, 360, 45):
        rad = math.radians(ang)
        px = int(cx + math.cos(rad) * 9)
        py = int(cy + math.sin(rad) * 9)
        cv2.circle(img, (px, py), 7, (215, 38, 45), -1)
    cv2.circle(img, (cx, cy), 5, (255, 215, 0), -1)
    cv2.circle(img, (cx, cy), 2, (70, 40, 20), -1)
    return img


@lru_cache(maxsize=16)
def make_effect_preview_icon(effect_name: str) -> QIcon:
    """Generates an authentic photographic thumbnail preview matching WLPG Effects gallery."""
    img = make_default_variation_base_image(84, 54)
    h, w = img.shape[:2]
    rec = PhotoAdjustmentRecipe(effect=effect_name)
    rendered = render_adjusted_image(img, rec)
    qimg = QImage(rendered.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    return QIcon(QPixmap.fromImage(qimg))


class _HoverableToolButton(QToolButton):
    hovered = Signal()
    unhovered = Signal()

    def enterEvent(self, event: Any) -> None:
        super().enterEvent(event)
        self.hovered.emit()

    def leaveEvent(self, event: Any) -> None:
        super().leaveEvent(event)
        self.unhovered.emit()


class _VariationTile(QPushButton):
    hovered = Signal()
    unhovered = Signal()

    def __init__(self, is_center: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.is_center = is_center
        self.setFixedSize(58, 42)
        self.setCursor(Qt.PointingHandCursor)
        self._update_style(False)

    def _update_style(self, is_hover: bool) -> None:
        if is_hover:
            self.setStyleSheet(
                "QPushButton { border: 2px solid #3399FF; background-color: #CCE4F7; padding: 0px; border-radius: 2px; }"
            )
        elif self.is_center:
            self.setStyleSheet(
                "QPushButton { border: 2px solid #70A6E8; background-color: #F0F4FC; padding: 0px; border-radius: 2px; }"
            )
        else:
            self.setStyleSheet(
                "QPushButton { border: 1px solid #D5DFEB; background-color: #FFFFFF; padding: 1px; border-radius: 2px; }"
            )

    def enterEvent(self, event: Any) -> None:
        super().enterEvent(event)
        self._update_style(True)
        self.hovered.emit()

    def leaveEvent(self, event: Any) -> None:
        super().leaveEvent(event)
        self._update_style(False)
        self.unhovered.emit()


class _AutoAdjustTile(QToolButton):
    hovered = Signal()
    unhovered = Signal()

    def __init__(self, icon: QIcon, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(54, 54)
        self.setIcon(icon)
        self.setIconSize(QSize(38, 38))
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(
            """
            QToolButton {
                border: 1px solid #C4D5E8;
                border-radius: 2px;
                background-color: #F8FAFD;
            }
            QToolButton:hover {
                border: 1px solid #3399FF;
                background-color: #CCE4F7;
            }
            """
        )

    def enterEvent(self, event: Any) -> None:
        super().enterEvent(event)
        self.hovered.emit()

    def leaveEvent(self, event: Any) -> None:
        super().leaveEvent(event)
        self.unhovered.emit()


class _ExposureMenuWidget(QWidget):
    preview_requested = Signal(float, float)      # (brightness, contrast)
    auto_preview_requested = Signal()
    preview_cancelled = Signal()
    adjustment_selected = Signal(float, float)    # (brightness, contrast)
    auto_selected = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ExposureMenuWidget")
        self.setStyleSheet(
            """
            QWidget#ExposureMenuWidget {
                background-color: #FFFFFF;
                border: 1px solid #85B5F3;
            }
            QLabel {
                font-family: 'Segoe UI';
                font-size: 9pt;
                font-weight: 600;
                color: #1E395B;
            }
            """
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        lbl_auto = QLabel("Auto adjust exposure", self)
        layout.addWidget(lbl_auto)

        self.btn_auto = _AutoAdjustTile(make_wlpg_icon("adjust_exposure", 36), self)
        self.btn_auto.hovered.connect(self.auto_preview_requested.emit)
        self.btn_auto.unhovered.connect(self.preview_cancelled.emit)
        self.btn_auto.clicked.connect(self.auto_selected.emit)
        layout.addWidget(self.btn_auto)

        sep = QFrame(self)
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Plain)
        sep.setStyleSheet("color: #DCE6F2; background-color: #DCE6F2; max-height: 1px;")
        layout.addWidget(sep)

        lbl_grid = QLabel("Choose exposure adjustment", self)
        layout.addWidget(lbl_grid)

        grid = QGridLayout()
        grid.setSpacing(4)
        grid.setContentsMargins(0, 0, 0, 0)

        self.tiles: list[tuple[float, float, _VariationTile]] = []
        brightnesses = [-25.0, 0.0, 25.0]
        contrasts = [-20.0, 0.0, 20.0]

        for r_idx, b_val in enumerate(brightnesses):
            for c_idx, c_val in enumerate(contrasts):
                is_center = (r_idx == 1 and c_idx == 1)
                tile = _VariationTile(is_center=is_center, parent=self)
                tile.setIconSize(QSize(54, 38))
                tile.hovered.connect(lambda b=b_val, c=c_val: self.preview_requested.emit(b, c))
                tile.unhovered.connect(self.preview_cancelled.emit)
                tile.clicked.connect(lambda _, b=b_val, c=c_val: self.adjustment_selected.emit(b, c))
                grid.addWidget(tile, r_idx, c_idx)
                self.tiles.append((b_val, c_val, tile))

        layout.addLayout(grid)
        self.refresh_thumbnails()

    def refresh_thumbnails(self, base_img: np.ndarray | None = None) -> None:
        if base_img is None:
            base_img = make_default_variation_base_image(54, 38)
        else:
            base_img = cv2.resize(base_img, (54, 38), interpolation=cv2.INTER_AREA)

        for b_val, c_val, tile in self.tiles:
            rec = PhotoAdjustmentRecipe(brightness=b_val, contrast=c_val)
            rendered = render_adjusted_image(base_img, rec)
            h, w = rendered.shape[:2]
            qimg = QImage(rendered.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            tile.setIcon(QIcon(QPixmap.fromImage(qimg)))


class _ColorMenuWidget(QWidget):
    preview_requested = Signal(float, float)      # (temperature, tint)
    auto_preview_requested = Signal()
    preview_cancelled = Signal()
    adjustment_selected = Signal(float, float)    # (temperature, tint)
    auto_selected = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ColorMenuWidget")
        self.setStyleSheet(
            """
            QWidget#ColorMenuWidget {
                background-color: #FFFFFF;
                border: 1px solid #85B5F3;
            }
            QLabel {
                font-family: 'Segoe UI';
                font-size: 9pt;
                font-weight: 600;
                color: #1E395B;
            }
            """
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        lbl_auto = QLabel("Auto adjust color", self)
        layout.addWidget(lbl_auto)

        self.btn_auto = _AutoAdjustTile(make_wlpg_icon("color_spectrum", 36), self)
        self.btn_auto.hovered.connect(self.auto_preview_requested.emit)
        self.btn_auto.unhovered.connect(self.preview_cancelled.emit)
        self.btn_auto.clicked.connect(self.auto_selected.emit)
        layout.addWidget(self.btn_auto)

        sep = QFrame(self)
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Plain)
        sep.setStyleSheet("color: #DCE6F2; background-color: #DCE6F2; max-height: 1px;")
        layout.addWidget(sep)

        lbl_grid = QLabel("Choose color adjustment", self)
        layout.addWidget(lbl_grid)

        grid = QGridLayout()
        grid.setSpacing(4)
        grid.setContentsMargins(0, 0, 0, 0)

        self.tiles: list[tuple[float, float, _VariationTile]] = []
        tints = [-25.0, 0.0, 25.0]
        temps = [-25.0, 0.0, 25.0]

        for r_idx, tint_val in enumerate(tints):
            for c_idx, temp_val in enumerate(temps):
                is_center = (r_idx == 1 and c_idx == 1)
                tile = _VariationTile(is_center=is_center, parent=self)
                tile.setIconSize(QSize(54, 38))
                tile.hovered.connect(lambda t=temp_val, ti=tint_val: self.preview_requested.emit(t, ti))
                tile.unhovered.connect(self.preview_cancelled.emit)
                tile.clicked.connect(lambda _, t=temp_val, ti=tint_val: self.adjustment_selected.emit(t, ti))
                grid.addWidget(tile, r_idx, c_idx)
                self.tiles.append((temp_val, tint_val, tile))

        layout.addLayout(grid)
        self.refresh_thumbnails()

    def refresh_thumbnails(self, base_img: np.ndarray | None = None) -> None:
        if base_img is None:
            base_img = make_default_variation_base_image(54, 38)
        else:
            base_img = cv2.resize(base_img, (54, 38), interpolation=cv2.INTER_AREA)

        for temp_val, tint_val, tile in self.tiles:
            rec = PhotoAdjustmentRecipe(temperature=temp_val, tint=tint_val)
            rendered = render_adjusted_image(base_img, rec)
            h, w = rendered.shape[:2]
            qimg = QImage(rendered.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            tile.setIcon(QIcon(QPixmap.fromImage(qimg)))


class _RibbonEffectsGallery(QFrame):
    """
    Authentic WLPG Ribbon Effects Gallery with responsive clipping,
    scroll up/down arrows, and a dropdown menu showing a clean vertical list of all effects.
    """

    effect_selected = Signal(str)
    effect_hovered = Signal(str)
    effect_unhovered = Signal()

    CARD_WIDTH = 48
    CARD_HEIGHT = 56
    CARD_STRIDE = 49

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("RibbonEffectsGallery")
        self.setStyleSheet(
            """
            QFrame#RibbonEffectsGallery {
                background: #FFFFFF;
                border: 1px solid #C4D5E8;
                border-radius: 3px;
            }
            QToolButton.EffectCard {
                border: 1px solid transparent;
                border-radius: 1px;
                background: transparent;
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 7.5pt;
                padding: 0px;
                margin: 0px;
            }
            QToolButton.EffectCard:hover {
                background: #EBF3FD;
                border: 1px solid #70A6E8;
            }
            QToolButton.EffectCard:checked {
                background: #CCE4F7;
                border: 1px solid #3399FF;
            }
            """
        )
        hbox = QHBoxLayout(self)
        hbox.setContentsMargins(1, 1, 1, 1)
        hbox.setSpacing(1)

        # Viewport widget that clips the cards horizontally to exact card boundaries
        self.viewport_widget = QWidget(self)
        self.viewport_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.viewport_widget.setFixedHeight(58)
        self.viewport_widget.setStyleSheet("background: transparent;")

        self.cards_holder = QWidget(self.viewport_widget)
        self.cards_holder.setStyleSheet("background: transparent;")
        self.cards_layout = QHBoxLayout(self.cards_holder)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setSpacing(1)

        self.btn_group = QButtonGroup(self)
        self.btn_group.setExclusive(True)

        self._effects_data = [
            ("none", "None", "No effect (Original)"),
            ("classic_chrome", "Chrome", "Classic Chrome (Fujifilm style)"),
            ("warm_portrait", "Portrait", "Warm Portrait (Portra golden glow)"),
            ("vivid_landscape", "Vivid", "Vivid Landscape (Punchy Velvia look)"),
            ("matte_vintage", "Matte", "Matte Vintage (Lifted film blacks)"),
            ("fine_art_bw", "B&W Art", "Fine Art B&W (Deep Acros tonal scale)"),
            ("cool_cinema", "Cinema", "Cool Cinema (Teal and Orange)"),
            ("sepia", "Sepia", "Warm sepia tone"),
        ]

        self.buttons: dict[str, _HoverableToolButton] = {}

        for eff_id, label, tip in self._effects_data:
            btn = _HoverableToolButton(self.cards_holder)
            btn.setProperty("class", "EffectCard")
            btn.setCheckable(True)
            if eff_id == "none":
                btn.setChecked(True)
            btn.setText(label)
            btn.setIcon(make_effect_preview_icon(eff_id))
            btn.setIconSize(QSize(46, 32))
            btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            btn.setFixedSize(self.CARD_WIDTH, self.CARD_HEIGHT)
            btn.setToolTip(tip)
            btn.hovered.connect(lambda eff=eff_id: self.effect_hovered.emit(eff))
            btn.unhovered.connect(self.effect_unhovered.emit)
            btn.clicked.connect(lambda _, eff=eff_id: self.effect_selected.emit(eff))
            self.btn_group.addButton(btn)
            self.cards_layout.addWidget(btn)
            self.buttons[eff_id] = btn

        # Legacy effect aliases for backwards compatibility with existing tests
        if "cool_cinema" in self.buttons:
            self.buttons["cool"] = self.buttons["cool_cinema"]
        if "fine_art_bw" in self.buttons:
            self.buttons["bw"] = self.buttons["fine_art_bw"]
            self.buttons["bw_contrast"] = self.buttons["fine_art_bw"]
        if "warm_portrait" in self.buttons:
            self.buttons["warm"] = self.buttons["warm_portrait"]

        total_w = len(self._effects_data) * self.CARD_STRIDE
        self.cards_holder.setGeometry(0, 0, total_w, 58)

        hbox.addWidget(self.viewport_widget, 1)

        v_ctrl = QVBoxLayout()
        v_ctrl.setContentsMargins(0, 0, 0, 0)
        v_ctrl.setSpacing(1)

        def make_mini_btn(symbol: str) -> QPushButton:
            b = QPushButton(symbol, self)
            b.setFixedSize(14, 18)
            b.setFlat(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(
                """
                QPushButton {
                    border: 1px solid #C4D5E8;
                    border-radius: 1px;
                    background: #F7FAFD;
                    font-size: 6pt;
                    color: #5A7A9E;
                    padding: 0px;
                }
                QPushButton:hover {
                    background: #E4F0FC;
                    border-color: #70A6E8;
                }
                QPushButton:disabled {
                    color: #B0C4DE;
                    background: #F4F6F9;
                    border-color: #DCE6F2;
                }
                """
            )
            return b

        self.btn_up = make_mini_btn("▲")
        self.btn_up.setToolTip("Previous effects")
        self.btn_up.clicked.connect(self._scroll_prev)

        self.btn_dn = make_mini_btn("▼")
        self.btn_dn.setToolTip("Next effects")
        self.btn_dn.clicked.connect(self._scroll_next)

        self.btn_more = make_mini_btn("▾")
        self.btn_more.setToolTip("Show all effects")
        self.btn_more.clicked.connect(self._show_all_effects_menu)

        v_ctrl.addWidget(self.btn_up)
        v_ctrl.addWidget(self.btn_dn)
        v_ctrl.addWidget(self.btn_more)
        hbox.addLayout(v_ctrl)

        self._scroll_index: int = 0
        self._visible_tiles: int = len(self._effects_data)
        self.set_visible_tiles(len(self._effects_data))

    def set_visible_tiles(self, count: int) -> None:
        self._visible_tiles = max(1, min(len(self._effects_data), count))
        vw = self._visible_tiles * self.CARD_STRIDE
        total_w = vw + 18
        self.viewport_widget.setFixedSize(vw, 58)
        self.setFixedSize(total_w, 62)
        total_holder_w = len(self._effects_data) * self.CARD_STRIDE
        self.cards_holder.setGeometry(0, 0, total_holder_w, 58)
        self._update_scroll_buttons()

    def sizeHint(self) -> QSize:
        count = getattr(self, "_visible_tiles", len(self._effects_data))
        total_w = count * self.CARD_STRIDE + 18
        return QSize(total_w, 62)

    def minimumSizeHint(self) -> QSize:
        return QSize(self.CARD_STRIDE + 18, 62)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_scroll_buttons()

    def _update_scroll_buttons(self) -> None:
        visible_count = getattr(self, "_visible_tiles", 6)
        max_idx = max(0, len(self._effects_data) - visible_count)
        self._scroll_index = max(0, min(self._scroll_index, max_idx))
        for i, (eff_id, _, _) in enumerate(self._effects_data):
            btn = self.buttons.get(eff_id)
            if btn:
                btn.setVisible(self._scroll_index <= i < self._scroll_index + visible_count)
        self.btn_up.setEnabled(self._scroll_index > 0)
        self.btn_dn.setEnabled(self._scroll_index < max_idx)

    def _scroll_prev(self) -> None:
        if self._scroll_index > 0:
            self._scroll_index -= 1
            self._update_scroll_buttons()

    def _scroll_next(self) -> None:
        visible_count = getattr(self, "_visible_tiles", 6)
        max_idx = max(0, len(self._effects_data) - visible_count)
        if self._scroll_index < max_idx:
            self._scroll_index += 1
            self._update_scroll_buttons()

    def _show_all_effects_menu(self) -> None:
        menu = QMenu(self)
        menu.setMinimumWidth(320)
        menu.setStyleSheet(
            """
            QMenu {
                background-color: #FFFFFF;
                border: 1px solid #70A6E8;
                border-radius: 3px;
                padding: 4px;
            }
            """
        )
        active_eff = "none"
        for eff_id, btn in self.buttons.items():
            if btn.isChecked() and eff_id in [e[0] for e in self._effects_data]:
                active_eff = eff_id
                break

        from PySide6.QtWidgets import QWidgetAction

        for eff_id, label, tip in self._effects_data:
            btn = self.buttons[eff_id]
            action = QWidgetAction(menu)
            w = QWidget(menu)
            w.setCursor(Qt.PointingHandCursor)
            w.setFixedHeight(62)
            w_layout = QHBoxLayout(w)
            w_layout.setContentsMargins(8, 3, 10, 3)
            w_layout.setSpacing(12)

            # Left: preview image (2x size: 84x54)
            lbl_icon = QLabel(w)
            lbl_icon.setFixedSize(84, 54)
            lbl_icon.setStyleSheet("border: 1px solid #CBD5E1; border-radius: 2px;")
            live_pix = getattr(self, "_live_thumbnails", {}).get(eff_id)
            if live_pix and not live_pix.isNull():
                scaled_pix = live_pix.scaled(
                    84, 54, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
                )
            else:
                scaled_pix = btn.icon().pixmap(84, 54)
            lbl_icon.setPixmap(scaled_pix)
            lbl_icon.setScaledContents(True)
            w_layout.addWidget(lbl_icon)

            # Right: Effect Name and Description
            v_text = QVBoxLayout()
            v_text.setContentsMargins(0, 0, 0, 0)
            v_text.setSpacing(2)

            lbl_name = QLabel(label, w)
            lbl_name.setFont(QFont("Segoe UI", 9.5, QFont.Weight.Bold))
            lbl_name.setStyleSheet("color: #0F172A;")
            v_text.addWidget(lbl_name)

            lbl_desc = QLabel(tip, w)
            lbl_desc.setFont(QFont("Segoe UI", 8.5))
            lbl_desc.setStyleSheet("color: #556B82;")
            v_text.addWidget(lbl_desc)

            w_layout.addLayout(v_text, 1)

            lbl_icon.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            lbl_name.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            lbl_desc.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

            if eff_id == active_eff:
                lbl_check = QLabel("✓", w)
                lbl_check.setStyleSheet("color: #2563EB; font-weight: bold; font-size: 13pt;")
                lbl_check.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
                w_layout.addWidget(lbl_check)

            # Hover & Click handling
            def make_handlers(target_eff: str, item_widget: QWidget):
                def enter(ev):
                    item_widget.setStyleSheet("background-color: #EBF3FD; border-radius: 3px;")
                    self.effect_hovered.emit(target_eff)
                def leave(ev):
                    item_widget.setStyleSheet("background-color: transparent;")
                    self.effect_unhovered.emit()
                def click(ev):
                    menu.close()
                    self._select_from_menu(target_eff)
                item_widget.enterEvent = enter
                item_widget.leaveEvent = leave
                item_widget.mousePressEvent = click

            make_handlers(eff_id, w)
            action.setDefaultWidget(w)
            action.triggered.connect(lambda checked=False, e=eff_id: self._select_from_menu(e))
            menu.addAction(action)

        menu.aboutToHide.connect(self.effect_unhovered.emit)
        if not getattr(self, "_suppress_menu_exec", False):
            menu.exec(self.btn_more.mapToGlobal(QPoint(0, self.btn_more.height())))

    def _select_from_menu(self, eff_id: str) -> None:
        self.set_active_effect(eff_id)
        self.effect_selected.emit(eff_id)

    def set_active_effect(self, eff_id: str) -> None:
        target = eff_id if eff_id in self.buttons else "none"
        btn = self.buttons.get(target)
        if btn:
            btn.setChecked(True)

    def update_live_thumbnails(self, pixmap_dict: dict[str, QPixmap]) -> None:
        """Updates the effect buttons with live thumbnail previews of the active image."""
        self._live_thumbnails = dict(pixmap_dict)
        for eff_id, pix in pixmap_dict.items():
            btn = self.buttons.get(eff_id)
            if btn and not pix.isNull():
                btn.setIcon(QIcon(pix))


class _RibbonGroupBox(QFrame):
    """Visual WLPG Ribbon group with content row on top and subtle grey group caption at bottom."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("RibbonGroupBox")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(
            """
            QFrame#RibbonGroupBox {
                border: none;
                border-right: 1px solid #C5D5E6;
                background: transparent;
            }
            """
        )
        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(1, 0, 1, 1)
        vbox.setSpacing(1)

        self.content_widget = QWidget(self)
        self.content_widget.setFixedHeight(70)
        self.content_layout = QHBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(1)
        self.content_layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        vbox.addWidget(self.content_widget)

        self.lbl_title = QLabel(title, self)
        self.lbl_title.setFixedHeight(14)
        self.lbl_title.setAlignment(Qt.AlignCenter)
        self.lbl_title.setStyleSheet(
            "color: #55687C; font-size: 8pt; font-family: 'Segoe UI'; border: none; background: transparent;"
        )
        vbox.addWidget(self.lbl_title)

    def setTitle(self, title: str) -> None:
        self.lbl_title.setText(title)


class _RibbonPage(QWidget):
    """Ribbon tab page with relaxed minimumSizeHint to allow responsive window resizing down to 400px."""

    def minimumSizeHint(self) -> QSize:
        return QSize(100, 88)


class _FaceCarouselWidget(QWidget):
    """Horizontal scrollable strip of face avatar cards matching WLPG Home Quick find."""

    person_clicked = Signal(int)
    boolean_mode_selected = Signal(str)
    special_filter_selected = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._buttons: dict[int, QToolButton] = {}
        self._button_list: list[QToolButton] = []
        self._all_people_list: list[dict[str, Any]] = []
        self._top_people: list[dict[str, Any]] = []
        self._visible_faces: int = 5
        self._scroll_index: int = 0

        hbox = QHBoxLayout(self)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(1)

        # Bordered container frame matching WLPG (62px tall)
        self.container = QFrame(self)
        self.container.setObjectName("FaceCarouselContainer")
        self.container.setFixedHeight(62)
        self.container.setStyleSheet(
            """
            QFrame#FaceCarouselContainer {
                background: #FFFFFF;
                border: 1px solid #ABBFD3;
                border-radius: 2px;
            }
            """
        )
        c_layout = QHBoxLayout(self.container)
        c_layout.setContentsMargins(1, 1, 1, 1)
        c_layout.setSpacing(1)

        self.scroll = QScrollArea(self.container)
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setFixedHeight(58)
        self.scroll.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.strip = QWidget()
        self.strip.setStyleSheet("background: transparent;")
        self.strip_layout = QHBoxLayout(self.strip)
        self.strip_layout.setContentsMargins(1, 0, 1, 0)
        self.strip_layout.setSpacing(2)
        self.strip_layout.addStretch(1)
        self.scroll.setWidget(self.strip)
        c_layout.addWidget(self.scroll)

        # Vertical 3-button stack on the right edge matching WLPG: ▲, ▼, and ▾ (More)
        btn_vbox = QVBoxLayout()
        btn_vbox.setContentsMargins(0, 0, 0, 0)
        btn_vbox.setSpacing(1)

        self.btn_scroll_left = QPushButton("▲", self.container)
        self.btn_scroll_left.setToolTip("Scroll left")
        self.btn_scroll_right = QPushButton("▼", self.container)
        self.btn_scroll_right.setToolTip("Scroll right")
        self.btn_expand = QPushButton(self.container)
        self.btn_expand.setToolTip("Show all people and search options")

        # Draw classic ribbon gallery "More" icon (top line + down arrow)
        more_pix = QPixmap(10, 10)
        more_pix.fill(Qt.transparent)
        mp = QPainter(more_pix)
        mp.setRenderHint(QPainter.Antialiasing)
        mp.setPen(QPen(QColor("#55687C"), 1.2))
        mp.drawLine(1, 2, 9, 2)
        mp.setBrush(QColor("#55687C"))
        tri = QPolygonF([QPointF(2, 4.5), QPointF(8, 4.5), QPointF(5, 7.5)])
        mp.drawPolygon(tri)
        mp.end()
        self.btn_expand.setIcon(QIcon(more_pix))
        self.btn_expand.setIconSize(QSize(10, 10))

        btn_style = """
            QPushButton {
                border: 1px solid #D2DFEE;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F6F9FD, stop:1 #E2ECF7);
                color: #55687C;
                font-size: 7pt;
                padding: 0;
            }
            QPushButton:hover {
                background: #CCE4F7;
                border-color: #7BAEDB;
            }
            QPushButton:disabled {
                color: #B0C4DE;
                background: #F4F6F9;
                border-color: #DCE6F2;
            }
        """
        for b in (self.btn_scroll_left, self.btn_scroll_right, self.btn_expand):
            b.setFixedSize(14, 19)
            b.setFlat(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(btn_style)

        self.btn_scroll_left.clicked.connect(self._scroll_left)
        self.btn_scroll_right.clicked.connect(self._scroll_right)
        self.btn_expand.clicked.connect(self._show_people_dropdown)
        btn_vbox.addWidget(self.btn_scroll_left)
        btn_vbox.addWidget(self.btn_scroll_right)
        btn_vbox.addWidget(self.btn_expand)
        c_layout.addLayout(btn_vbox)

        hbox.addWidget(self.container)
        self.setFixedHeight(62)
        self.set_visible_faces(5)

    def set_visible_faces(self, count: int) -> None:
        self._visible_faces = max(1, min(6, count))
        scroll_w = self._visible_faces * 56
        total_w = scroll_w + 19
        self.scroll.setFixedWidth(scroll_w)
        self.container.setFixedWidth(total_w)
        self.setFixedWidth(total_w)
        self.setMinimumWidth(total_w)
        if count >= 5:
            self.setMaximumWidth(360)
        else:
            self.setMaximumWidth(total_w)
        self.setFixedHeight(62)
        self.container.setFixedHeight(62)
        self.scroll.setFixedHeight(58)
        self._update_visible_faces()

    def sizeHint(self) -> QSize:
        w = getattr(self, "_visible_faces", 5) * 56 + 19
        return QSize(w, 62)

    def minimumSizeHint(self) -> QSize:
        w = getattr(self, "_visible_faces", 1) * 56 + 19
        return QSize(w, 62)

    def _update_visible_faces(self) -> None:
        n = len(self._button_list)
        max_idx = max(0, n - self._visible_faces)
        self._scroll_index = max(0, min(self._scroll_index, max_idx))
        for i, btn in enumerate(self._button_list):
            btn.setVisible(self._scroll_index <= i < self._scroll_index + self._visible_faces)
        self.btn_scroll_left.setEnabled(self._scroll_index > 0)
        self.btn_scroll_right.setEnabled(self._scroll_index < max_idx)

    def _scroll_left(self) -> None:
        if self._scroll_index > 0:
            self._scroll_index -= 1
            self._update_visible_faces()

    def _scroll_right(self) -> None:
        n = len(self._button_list)
        max_idx = max(0, n - self._visible_faces)
        if self._scroll_index < max_idx:
            self._scroll_index += 1
            self._update_visible_faces()

    def populate(self, people: Sequence[dict[str, Any]]) -> None:
        while self.strip_layout.count() > 1:
            item = self.strip_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._buttons.clear()
        self._button_list.clear()
        self._scroll_index = 0
        self._all_people_list = list(people)

        # Sort by Top people (highest media count & confirmed count descending)
        self._top_people = sorted(
            self._all_people_list,
            key=lambda p: (int(p.get("media_count", 0)), int(p.get("confirmed_count", 0))),
            reverse=True,
        )

        for p in self._top_people:
            pid = int(p["id"])
            name = str(p["name"])
            cnt = int(p.get("media_count", 0))
            blob = p.get("thumbnail_blob")

            btn = QToolButton(self.strip)
            btn.setFixedSize(54, 54)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setToolTip(f"{name} ({cnt} items)")

            pix = make_person_avatar_pixmap(name, blob, size=50)
            btn.setIcon(QIcon(pix))
            btn.setIconSize(QSize(50, 50))
            btn.setStyleSheet(
                """
                QToolButton {
                    border: 1px solid transparent;
                    border-radius: 2px;
                    background: transparent;
                    padding: 0px;
                }
                QToolButton:hover {
                    border: 1px solid #E2B034;
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #FFFDF5, stop:1 #FCE7A2);
                }
                QToolTip {
                    background-color: #FFFFFF;
                    color: #1E395B;
                    border: 1px solid #7DA2CE;
                    border-radius: 2px;
                    padding: 3px 6px;
                    font-family: 'Segoe UI';
                    font-size: 8.5pt;
                }
                """
            )
            btn.clicked.connect(lambda _checked=False, person_id=pid: self.person_clicked.emit(person_id))
            self.strip_layout.insertWidget(self.strip_layout.count() - 1, btn)
            self._buttons[pid] = btn
            self._button_list.append(btn)

        self._update_visible_faces()

    def click_person(self, person_id: int) -> None:
        if person_id in self._buttons:
            self._buttons[person_id].click()
        else:
            self.person_clicked.emit(person_id)

    def _show_people_dropdown(self) -> None:
        menu = QMenu(self)
        menu.setObjectName("WLPGPeopleDropdownMenu")
        menu.setStyleSheet(
            """
            QMenu#WLPGPeopleDropdownMenu {
                background: #FFFFFF;
                border: 1px solid #7DA2CE;
                padding: 0px;
            }
            """
        )
        widget = _WLPGPeopleMenuWidget(menu, self._all_people_list)
        widget.person_selected.connect(self.person_clicked.emit)
        widget.boolean_mode_selected.connect(self.boolean_mode_selected.emit)
        widget.special_filter_selected.connect(self.special_filter_selected.emit)

        wa = QWidgetAction(menu)
        wa.setDefaultWidget(widget)
        menu.addAction(wa)

        pos = self.container.mapToGlobal(QPoint(0, self.container.height()))
        screen = self.screen() or (QApplication.primaryScreen() if QApplication.instance() else None)
        if screen:
            avail_w = screen.availableGeometry().right()
            if pos.x() + 480 > avail_w:
                pos.setX(max(10, avail_w - 490))
        menu.exec(pos)


class _WLPGPeopleMenuWidget(QWidget):
    """
    Authentic Windows Live Photo Gallery People dropdown widget matching media_1792919727459_8c5fe8cf.png:
    - Top people (1 row of top 10 people by photo count)
    - Other people A - Z (10-column scrollable grid of all other people sorted alphabetically)
    - Search options: and, or, All tagged or detected, All detected but not tagged, Not tagged or detected
    """

    person_selected = Signal(int)
    boolean_mode_selected = Signal(str)
    special_filter_selected = Signal(str)

    def __init__(
        self,
        menu: QMenu,
        people: Sequence[dict[str, Any]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent or menu)
        self.menu = menu
        self.setFixedWidth(476)
        self.setObjectName("WLPGPeopleMenuWidget")
        self.setStyleSheet(
            """
            QWidget#WLPGPeopleMenuWidget {
                background: #FFFFFF;
                border: 1px solid #7DA2CE;
            }
            """
        )

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        # Sort people into Top 10 and Other People A-Z
        all_people = list(people)
        top_people = sorted(
            all_people,
            key=lambda p: (int(p.get("media_count", 0)), int(p.get("confirmed_count", 0))),
            reverse=True,
        )
        top_10 = top_people[:10]
        top_ids = {int(p["id"]) for p in top_10}
        other_people = sorted(
            [p for p in all_people if int(p["id"]) not in top_ids],
            key=lambda p: str(p.get("name", "")).lower(),
        )

        # Scroll Area hosting Top people and Other people A-Z
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.NoFrame)
        est_height = 80 + (len(other_people) + 9) // 10 * 46
        scroll.setFixedHeight(min(380, max(140, est_height)))
        scroll.setStyleSheet(
            """
            QScrollArea {
                background: #FFFFFF;
                border: none;
            }
            QScrollBar:vertical {
                border: none;
                background: transparent;
                width: 8px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #CBD5E1;
                min-height: 24px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: #94A3B8;
            }
            QScrollBar::handle:vertical:pressed {
                background: #64748B;
            }
            QScrollBar::sub-line:vertical, QScrollBar::add-line:vertical {
                border: none;
                background: transparent;
                height: 0px;
                width: 0px;
            }
            QScrollBar::sub-page:vertical, QScrollBar::add-page:vertical {
                border: none;
                background: transparent;
            }
            QScrollBar:horizontal {
                border: none;
                background: transparent;
                height: 0px;
                width: 0px;
            }
            QAbstractScrollArea::corner {
                background: transparent;
                border: none;
            }
            """
        )

        scroll_content = QWidget()
        scroll_content.setStyleSheet("background: #FFFFFF;")
        content_layout = QVBoxLayout(scroll_content)
        content_layout.setContentsMargins(4, 4, 4, 4)
        content_layout.setSpacing(4)

        # 1. Section Header: Top people
        lbl_top = QLabel("Top people", scroll_content)
        lbl_top.setStyleSheet(
            "background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F9FBFE, stop:1 #EBF2F8);"
            "border-bottom: 1px solid #D5E0EA; color: #35506E; font-weight: bold; font-size: 8.5pt;"
            "font-family: 'Segoe UI'; padding: 3px 6px;"
        )
        content_layout.addWidget(lbl_top)

        top_grid_widget = QWidget(scroll_content)
        top_grid_widget.setStyleSheet("background: transparent;")
        top_grid = QGridLayout(top_grid_widget)
        top_grid.setContentsMargins(0, 2, 0, 4)
        top_grid.setSpacing(2)
        top_grid.setAlignment(Qt.AlignLeft)

        btn_style = """
            QToolButton {
                border: 1px solid #C8D7E6;
                border-radius: 2px;
                background: #FFFFFF;
                padding: 0px;
            }
            QToolButton:hover {
                border: 1px solid #E2B034;
                background: #FFF8E6;
            }
        """

        for col, p in enumerate(top_10):
            pid = int(p["id"])
            name = str(p.get("name", ""))
            cnt = int(p.get("media_count", 0))
            blob = p.get("thumbnail_blob")
            btn = QToolButton(top_grid_widget)
            btn.setFixedSize(42, 42)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setToolTip(f"{name} ({cnt} items)")
            pix = make_person_avatar_pixmap(name, blob, size=40)
            btn.setIcon(QIcon(pix))
            btn.setIconSize(QSize(40, 40))
            btn.setStyleSheet(btn_style)
            btn.clicked.connect(lambda _c=False, person_id=pid: self._on_person_clicked(person_id))
            top_grid.addWidget(btn, 0, col)

        content_layout.addWidget(top_grid_widget)

        # 2. Section Header: Other people A - Z
        if other_people:
            lbl_other = QLabel("Other people A - Z", scroll_content)
            lbl_other.setStyleSheet(
                "background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F9FBFE, stop:1 #EBF2F8);"
                "border-top: 1px solid #D5E0EA; border-bottom: 1px solid #D5E0EA; color: #35506E; font-weight: bold; font-size: 8.5pt;"
                "font-family: 'Segoe UI'; padding: 3px 6px;"
            )
            content_layout.addWidget(lbl_other)

            other_grid_widget = QWidget(scroll_content)
            other_grid_widget.setStyleSheet("background: transparent;")
            other_grid = QGridLayout(other_grid_widget)
            other_grid.setContentsMargins(0, 2, 0, 4)
            other_grid.setSpacing(2)
            other_grid.setAlignment(Qt.AlignLeft)

            for i, p in enumerate(other_people):
                r = i // 10
                c = i % 10
                pid = int(p["id"])
                name = str(p.get("name", ""))
                cnt = int(p.get("media_count", 0))
                blob = p.get("thumbnail_blob")
                btn = QToolButton(other_grid_widget)
                btn.setFixedSize(42, 42)
                btn.setCursor(Qt.PointingHandCursor)
                btn.setToolTip(f"{name} ({cnt} items)")
                pix = make_person_avatar_pixmap(name, blob, size=40)
                btn.setIcon(QIcon(pix))
                btn.setIconSize(QSize(40, 40))
                btn.setStyleSheet(btn_style)
                btn.clicked.connect(lambda _c=False, person_id=pid: self._on_person_clicked(person_id))
                other_grid.addWidget(btn, r, c)

            content_layout.addWidget(other_grid_widget)

        content_layout.addStretch(1)
        scroll.setWidget(scroll_content)
        vbox.addWidget(scroll)

        # 3. Pinned Bottom Actions
        bottom_frame = QFrame(self)
        bottom_frame.setStyleSheet(
            "background: #F6F9FD; border-top: 1px solid #D5E0EA;"
        )
        b_vbox = QVBoxLayout(bottom_frame)
        b_vbox.setContentsMargins(6, 4, 6, 4)
        b_vbox.setSpacing(2)

        opt_style = """
            QToolButton {
                border: 1px solid transparent;
                border-radius: 2px;
                background: transparent;
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                text-align: left;
                padding: 2px 6px;
            }
            QToolButton:hover {
                background: #E5F1FB;
                border: 1px solid #A0C5E8;
            }
        """

        self.btn_and = QToolButton(bottom_frame)
        self.btn_and.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_and.setText("&and")
        self.btn_and.setIcon(make_wlpg_icon("boolean_and", 14))
        self.btn_and.setIconSize(QSize(14, 14))
        self.btn_and.setStyleSheet(opt_style)
        self.btn_and.setFixedHeight(22)
        self.btn_and.clicked.connect(lambda: self._on_boolean_mode("AND"))
        b_vbox.addWidget(self.btn_and)

        self.btn_or = QToolButton(bottom_frame)
        self.btn_or.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_or.setText("&or")
        self.btn_or.setIcon(make_wlpg_icon("boolean_or", 14))
        self.btn_or.setIconSize(QSize(14, 14))
        self.btn_or.setStyleSheet(opt_style)
        self.btn_or.setFixedHeight(22)
        self.btn_or.clicked.connect(lambda: self._on_boolean_mode("OR"))
        b_vbox.addWidget(self.btn_or)

        self.btn_all_tagged = QToolButton(bottom_frame)
        self.btn_all_tagged.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_all_tagged.setText("All &tagged or detected")
        self.btn_all_tagged.setIcon(make_wlpg_icon("person", 14))
        self.btn_all_tagged.setIconSize(QSize(14, 14))
        self.btn_all_tagged.setStyleSheet(opt_style)
        self.btn_all_tagged.setFixedHeight(22)
        self.btn_all_tagged.clicked.connect(lambda: self._on_special_filter("all_people"))
        b_vbox.addWidget(self.btn_all_tagged)

        self.btn_detected = QToolButton(bottom_frame)
        self.btn_detected.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_detected.setText("All &detected but not tagged")
        self.btn_detected.setIcon(make_wlpg_icon("person_question", 14))
        self.btn_detected.setIconSize(QSize(14, 14))
        self.btn_detected.setStyleSheet(opt_style)
        self.btn_detected.setFixedHeight(22)
        self.btn_detected.clicked.connect(lambda: self._on_special_filter("detected_untagged"))
        b_vbox.addWidget(self.btn_detected)

        self.btn_untagged = QToolButton(bottom_frame)
        self.btn_untagged.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_untagged.setText("&Not tagged or detected")
        self.btn_untagged.setIcon(make_wlpg_icon("person_none", 14))
        self.btn_untagged.setIconSize(QSize(14, 14))
        self.btn_untagged.setStyleSheet(opt_style)
        self.btn_untagged.setFixedHeight(22)
        self.btn_untagged.clicked.connect(lambda: self._on_special_filter("untagged"))
        b_vbox.addWidget(self.btn_untagged)

        vbox.addWidget(bottom_frame)

    def _on_person_clicked(self, pid: int) -> None:
        self.menu.close()
        self.person_selected.emit(pid)

    def _on_boolean_mode(self, mode: str) -> None:
        self.menu.close()
        self.boolean_mode_selected.emit(mode)

    def _on_special_filter(self, kind: str) -> None:
        self.menu.close()
        self.special_filter_selected.emit(kind)


class WLPGTagsMenuWidget(QWidget):
    """
    Authentic Windows Live Photo Gallery Tags dropdown widget matching media_1792184066235_7e468e48.png.
    Renders inside a QWidgetAction within the Tags QMenu.
    """

    tag_selected = Signal(int)
    boolean_mode_changed = Signal(str)
    untagged_selected = Signal()
    show_hierarchy_selected = Signal()

    def __init__(self, menu: QMenu, parent: QWidget | None = None) -> None:
        super().__init__(parent or menu)
        self.menu = menu
        self.setFixedWidth(380)
        self.setObjectName("WLPGTagsMenuWidget")
        self.setStyleSheet(
            """
            QWidget#WLPGTagsMenuWidget {
                background: #FFFFFF;
                border: 1px solid #7DA2CE;
            }
            """
        )

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        # 1. Section Header: Most used tags
        self.lbl_hdr_most_used = QLabel("Most used tags", self)
        self.lbl_hdr_most_used.setStyleSheet(
            "background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F9FBFE, stop:1 #EBF2F8);"
            "border-bottom: 1px solid #D5E0EA; color: #35506E; font-weight: bold; font-size: 8.5pt;"
            "font-family: 'Segoe UI'; padding: 3px 8px;"
        )
        vbox.addWidget(self.lbl_hdr_most_used)

        # Container for top 10 most used tags
        self.top_tags_container = QWidget(self)
        self.top_tags_container.setStyleSheet("background: #FFFFFF;")
        self.top_tags_layout = QVBoxLayout(self.top_tags_container)
        self.top_tags_layout.setContentsMargins(0, 2, 0, 2)
        self.top_tags_layout.setSpacing(0)
        vbox.addWidget(self.top_tags_container)

        # 2. Section Header: All tags
        self.lbl_hdr_all_tags = QLabel("All tags", self)
        self.lbl_hdr_all_tags.setStyleSheet(
            "background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F9FBFE, stop:1 #EBF2F8);"
            "border-top: 1px solid #D5E0EA; border-bottom: 1px solid #D5E0EA; color: #35506E; font-weight: bold; font-size: 8.5pt;"
            "font-family: 'Segoe UI'; padding: 3px 8px;"
        )
        vbox.addWidget(self.lbl_hdr_all_tags)

        # Scrollable list for All tags
        self.list_all_tags = QListWidget(self)
        self.list_all_tags.setFixedHeight(220)
        self.list_all_tags.setFrameShape(QFrame.NoFrame)
        self.list_all_tags.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list_all_tags.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.list_all_tags.setStyleSheet(
            """
            QListWidget {
                background: #FFFFFF;
                border: none;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                color: #1E395B;
                padding: 1px 0;
            }
            QListWidget::item {
                height: 20px;
                padding-left: 8px;
            }
            QListWidget::item:hover {
                background: #E5F1FB;
                border: 1px solid #A0C5E8;
                border-radius: 2px;
            }
            QListWidget::item:selected {
                background: #CCE2F7;
                color: #000000;
            }
            """
        )
        self.list_all_tags.itemClicked.connect(self._on_all_tag_clicked)
        vbox.addWidget(self.list_all_tags)

        # 3. Pinned Bottom Bar
        bottom_frame = QFrame(self)
        bottom_frame.setStyleSheet(
            "background: #F6F9FD; border-top: 1px solid #D5E0EA;"
        )
        b_vbox = QVBoxLayout(bottom_frame)
        b_vbox.setContentsMargins(4, 3, 4, 3)
        b_vbox.setSpacing(2)

        # Mode toggles: [+ and] and [/ or]
        mode_hbox = QHBoxLayout()
        mode_hbox.setContentsMargins(2, 0, 2, 0)
        mode_hbox.setSpacing(4)

        btn_style = """
            QPushButton {
                border: 1px solid #BDD3E8;
                border-radius: 2px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #FDFEFE, stop:1 #EBF2F9);
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 8pt;
                padding: 0 8px;
            }
            QPushButton:hover {
                background: #E5F1FB;
                border-color: #7DA2CE;
            }
            QPushButton:checked {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #EAF3FD, stop:1 #BBDDF7);
                border: 1px solid #7DA2CE;
                font-weight: 600;
            }
        """
        self.btn_mode_and = QPushButton("and", bottom_frame)
        self.btn_mode_and.setIcon(make_wlpg_icon("boolean_and", 12))
        self.btn_mode_and.setIconSize(QSize(12, 12))
        self.btn_mode_and.setCheckable(True)
        self.btn_mode_and.setFixedHeight(20)
        self.btn_mode_and.setStyleSheet(btn_style)

        self.btn_mode_or = QPushButton("or", bottom_frame)
        self.btn_mode_or.setIcon(make_wlpg_icon("boolean_or", 12))
        self.btn_mode_or.setIconSize(QSize(12, 12))
        self.btn_mode_or.setCheckable(True)
        self.btn_mode_or.setFixedHeight(20)
        self.btn_mode_or.setChecked(True)
        self.btn_mode_or.setStyleSheet(btn_style)

        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_group.addButton(self.btn_mode_and)
        self.mode_group.addButton(self.btn_mode_or)

        self.btn_mode_and.toggled.connect(
            lambda checked: self.boolean_mode_changed.emit("AND" if checked else "OR")
        )

        mode_hbox.addWidget(self.btn_mode_and)
        mode_hbox.addWidget(self.btn_mode_or)
        mode_hbox.addStretch(1)
        b_vbox.addLayout(mode_hbox)

        sep = QFrame(bottom_frame)
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #DDE6EF; margin: 1px 0;")
        b_vbox.addWidget(sep)

        action_btn_style = """
            QPushButton {
                text-align: left;
                border: 1px solid transparent;
                background: transparent;
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                padding: 2px 6px;
                border-radius: 2px;
            }
            QPushButton:hover {
                background: #E5F1FB;
                border: 1px solid #A0C5E8;
            }
        """
        self.btn_not_tagged = QPushButton("  Not tagged", bottom_frame)
        self.btn_not_tagged.setIcon(make_wlpg_icon("not_tagged", 16))
        self.btn_not_tagged.setStyleSheet(action_btn_style)
        self.btn_not_tagged.clicked.connect(self._on_not_tagged_clicked)
        b_vbox.addWidget(self.btn_not_tagged)

        self.btn_show_hierarchy = QPushButton("Show tag hierarchy...", bottom_frame)
        self.btn_show_hierarchy.setIcon(make_wlpg_icon("tag_hierarchy", 16))
        self.btn_show_hierarchy.setStyleSheet(action_btn_style)
        self.btn_show_hierarchy.clicked.connect(self._on_show_hierarchy_clicked)
        b_vbox.addWidget(self.btn_show_hierarchy)

        vbox.addWidget(bottom_frame)

    def populate(self, tag_tree: Sequence[dict[str, Any]]) -> None:
        while self.top_tags_layout.count():
            item = self.top_tags_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()

        sorted_by_usage = sorted(
            tag_tree,
            key=lambda t: int(t.get("media_count", t.get("direct_count", 0))),
            reverse=True,
        )
        top_10 = sorted_by_usage[:10]
        for t in top_10:
            tid = int(t["id"])
            full_path = str(t.get("full_path") or t.get("name") or "Tag")
            cnt = int(t.get("media_count", t.get("direct_count", 0)))
            text = f"{full_path} ({cnt})" if cnt > 0 else full_path

            btn = QPushButton(text, self.top_tags_container)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(
                """
                QPushButton {
                    text-align: left;
                    border: 1px solid transparent;
                    background: transparent;
                    color: #1E395B;
                    font-family: 'Segoe UI';
                    font-size: 8.5pt;
                    padding: 2px 8px;
                    border-radius: 2px;
                    height: 18px;
                }
                QPushButton:hover {
                    background: #E5F1FB;
                    border: 1px solid #A0C5E8;
                }
                """
            )
            btn.clicked.connect(lambda _checked=False, tag_id=tid: self._select_tag(tag_id))
            self.top_tags_layout.addWidget(btn)

        self.list_all_tags.clear()
        alpha_tags = sorted(tag_tree, key=lambda t: str(t.get("full_path") or t.get("name") or "").lower())
        for t in alpha_tags:
            tid = int(t["id"])
            full_path = str(t.get("full_path") or t.get("name") or "Tag")
            item = QListWidgetItem(full_path, self.list_all_tags)
            item.setData(Qt.UserRole, tid)

    def _select_tag(self, tag_id: int) -> None:
        self.menu.close()
        self.tag_selected.emit(tag_id)

    def _on_all_tag_clicked(self, item: QListWidgetItem) -> None:
        tid = item.data(Qt.UserRole)
        if tid is not None:
            self._select_tag(int(tid))

    def _on_not_tagged_clicked(self) -> None:
        self.menu.close()
        self.untagged_selected.emit()

    def _on_show_hierarchy_clicked(self) -> None:
        self.menu.close()
        self.show_hierarchy_selected.emit()


class WLPGTagsMenu(QMenu):
    """QMenu container hosting the authentic WLPGTagsMenuWidget."""

    def __init__(self, ribbon: RibbonToolbar) -> None:
        super().__init__(ribbon)
        self.ribbon = ribbon
        self.setObjectName("WLPGTagsMenu")
        self.setStyleSheet(
            """
            QMenu#WLPGTagsMenu {
                background: #FFFFFF;
                border: 1px solid #7DA2CE;
                padding: 0px;
            }
            """
        )
        self.tags_widget = WLPGTagsMenuWidget(self)
        self._wa = QWidgetAction(self)
        self._wa.setDefaultWidget(self.tags_widget)
        super().addAction(self._wa)
        self._extra_actions: list[QAction] = []

        self.tags_widget.tag_selected.connect(
            lambda tid: self.ribbon.tag_filter_selected.emit(tid, "include")
        )
        self.tags_widget.boolean_mode_changed.connect(
            lambda mode: self.ribbon.tag_filter_selected.emit(None, mode)
        )
        self.tags_widget.untagged_selected.connect(
            lambda: self.ribbon.untagged_filter_selected.emit(True)
        )
        self.tags_widget.show_hierarchy_selected.connect(
            lambda: self.ribbon.action_triggered.emit("show_tag_hierarchy")
        )

    def populate(self, tag_tree: Sequence[dict[str, Any]]) -> None:
        self.tags_widget.populate(tag_tree)
        self._extra_actions.clear()
        for t in tag_tree:
            tid = int(t["id"])
            name = str(t.get("name") or t.get("full_path") or "Tag")
            act = QAction(name, self)
            act.setData(tid)
            act.triggered.connect(lambda _c=False, tag_id=tid: self.ribbon.tag_filter_selected.emit(tag_id, "include"))
            self._extra_actions.append(act)

        act_not_tagged = QAction("Not tagged", self)
        act_not_tagged.triggered.connect(lambda: self.ribbon.untagged_filter_selected.emit(True))
        self._extra_actions.append(act_not_tagged)

    def actions(self) -> list[QAction]:
        return [self._wa] + self._extra_actions


class RibbonToolbar(QWidget):
    """
    Top WLPG Ribbon (`Home`, `Edit`, `Find`, `Create`, `View`) + `Search by:` Scope Header Bar.
    """

    arrange_changed = Signal(str, bool)             # (ArrangeMode, reverse_sort)
    details_overlay_changed = Signal(set)           # set of active detail keys
    person_quick_find_selected = Signal(int)        # person_id
    search_text_changed = Signal(str)               # debounced search query
    search_all_requested = Signal()                 # clear folder/tag scope
    action_triggered = Signal(str)                  # ribbon command identifier
    layout_mode_changed = Signal(str)               # "wlpg_grid" | "mosaic"
    zoom_size_changed = Signal(int)                 # 72..320 px

    # Authentic WLPG Find Tab Filter Signals
    date_filter_selected = Signal(object, object, object)   # (year, month, day)
    rating_filter_selected = Signal(object)                 # tuple of ratings
    tag_filter_selected = Signal(object, str)              # (tag_id, boolean_mode)
    untagged_filter_selected = Signal(bool)
    flagged_filter_selected = Signal(bool)
    flag_filter_selected = Signal(object)                   # int | None: 1 = flagged, 0 = not flagged, None = all
    edited_filter_selected = Signal(bool)
    media_type_filter_selected = Signal(str)                # "all" | "image" | "video"
    person_boolean_mode_changed = Signal(str)               # "AND" | "OR"
    people_special_filter_selected = Signal(str)            # "all_people" | "detected_untagged" | "untagged"
    text_search_toggle_requested = Signal()

    # Quick Access Toolbar & Navigation Signals
    back_requested = Signal()
    forward_requested = Signal()
    undo_requested = Signal()
    redo_requested = Signal()
    filter_month_selected = Signal(int)                     # 0 for all, 1..12

    # Real-time preview & adjustment signals for Exposure and Color menus
    exposure_preview_requested = Signal(float, float)       # (brightness, contrast)
    auto_exposure_preview_requested = Signal()
    exposure_preview_cancelled = Signal()
    exposure_adjustment_selected = Signal(float, float)     # (brightness, contrast)
    auto_exposure_selected = Signal()

    color_preview_requested = Signal(float, float)          # (temperature, tint)
    auto_color_preview_requested = Signal()
    color_preview_cancelled = Signal()
    color_adjustment_selected = Signal(float, float)        # (temperature, tint)
    auto_color_selected = Signal()

    exposure_menu_about_to_show = Signal()
    color_menu_about_to_show = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("RibbonToolbar")

        self._arrange_by: ArrangeMode = "date"
        self._reverse_sort: bool = False
        self._visible_details: set[str] = {"rating"}
        self._detail_checkboxes: dict[str, QAbstractButton] = {}
        self._responsive_large_buttons: list[QToolButton] = []
        self._responsive_secondary_buttons: list[QToolButton] = []
        self._all_themed_buttons: list[QToolButton] = []
        icon_provider.signals.theme_changed.connect(self.refresh_icons)

        # Search input debounce timer (180ms)
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(180)
        self._search_timer.timeout.connect(self._emit_search_text)

        main_vbox = QVBoxLayout(self)
        main_vbox.setContentsMargins(0, 0, 0, 0)
        main_vbox.setSpacing(0)
        self.main_vbox = main_vbox

        # 0. Quick Access Toolbar & App Title Header
        self._qat_below_ribbon: bool = False

        self.qat_container = QWidget(self)
        self.qat_container.setObjectName("QATContainer")
        self.qat_layout = QHBoxLayout(self.qat_container)
        self.qat_layout.setContentsMargins(0, 0, 0, 0)
        self.qat_layout.setSpacing(2)

        self.btn_back = QToolButton(self.qat_container)
        self.btn_back.setObjectName("QATCircleBtn")
        self.btn_back.setIcon(make_wlpg_icon("nav_back", 18))
        self.btn_back.setIconSize(QSize(18, 18))
        self.btn_back.setFixedSize(22, 22)
        self.btn_back.setToolTip("Back (Alt+Left Arrow)")
        self.btn_back.setCursor(Qt.PointingHandCursor)
        self.btn_back.setEnabled(False)
        self.btn_back.clicked.connect(self.back_requested.emit)
        self.qat_layout.addWidget(self.btn_back)

        self.btn_forward = QToolButton(self.qat_container)
        self.btn_forward.setObjectName("QATCircleBtn")
        self.btn_forward.setIcon(make_wlpg_icon("nav_forward", 18))
        self.btn_forward.setIconSize(QSize(18, 18))
        self.btn_forward.setFixedSize(22, 22)
        self.btn_forward.setToolTip("Forward (Alt+Right Arrow)")
        self.btn_forward.setCursor(Qt.PointingHandCursor)
        self.btn_forward.setEnabled(False)
        self.btn_forward.clicked.connect(self.forward_requested.emit)
        self.qat_layout.addWidget(self.btn_forward)

        self.qat_layout.addSpacing(2)

        self.btn_print = QToolButton(self.qat_container)
        self.btn_print.setObjectName("QATBtn")
        self.btn_print.setIcon(make_wlpg_icon("print", 16))
        self.btn_print.setIconSize(QSize(16, 16))
        self.btn_print.setFixedSize(22, 22)
        self.btn_print.setToolTip("Print (Ctrl+P)")
        self.btn_print.setCursor(Qt.PointingHandCursor)
        self.btn_print.setVisible(False)
        self.btn_print.clicked.connect(lambda: self.action_triggered.emit("print"))
        self.qat_layout.addWidget(self.btn_print)

        self.btn_undo = QToolButton(self.qat_container)
        self.btn_undo.setObjectName("QATBtn")
        self.btn_undo.setIcon(make_wlpg_icon("undo", 16))
        self.btn_undo.setIconSize(QSize(16, 16))
        self.btn_undo.setFixedSize(22, 22)
        self.btn_undo.setToolTip("Undo (Ctrl+Z)")
        self.btn_undo.setCursor(Qt.PointingHandCursor)
        self.btn_undo.setEnabled(False)
        self.btn_undo.clicked.connect(self.undo_requested.emit)
        self.qat_layout.addWidget(self.btn_undo)

        self.btn_redo = QToolButton(self.qat_container)
        self.btn_redo.setObjectName("QATBtn")
        self.btn_redo.setIcon(make_wlpg_icon("redo", 16))
        self.btn_redo.setIconSize(QSize(16, 16))
        self.btn_redo.setFixedSize(22, 22)
        self.btn_redo.setToolTip("Redo (Ctrl+Y)")
        self.btn_redo.setCursor(Qt.PointingHandCursor)
        self.btn_redo.setEnabled(False)
        self.btn_redo.clicked.connect(self.redo_requested.emit)
        self.qat_layout.addWidget(self.btn_redo)

        self.btn_qat_menu = QToolButton(self.qat_container)
        self.btn_qat_menu.setObjectName("QATBtn")
        self.btn_qat_menu.setIcon(make_wlpg_icon("qat_menu", 14))
        self.btn_qat_menu.setIconSize(QSize(14, 14))
        self.btn_qat_menu.setFixedSize(14, 22)
        self.btn_qat_menu.setToolTip("Customize Quick Access Toolbar")
        self.btn_qat_menu.setCursor(Qt.PointingHandCursor)
        self.btn_qat_menu.setPopupMode(QToolButton.InstantPopup)

        # Build Customize QAT Menu
        qat_menu = QMenu(self.btn_qat_menu)
        qat_menu.setFont(QFont("Segoe UI", 9))

        act_title = qat_menu.addAction("Customize Quick Access Toolbar")
        act_title.setEnabled(False)
        bold_f = QFont(act_title.font())
        bold_f.setBold(True)
        act_title.setFont(bold_f)
        qat_menu.addSeparator()

        self.act_qat_print = qat_menu.addAction("Print")
        self.act_qat_print.setCheckable(True)
        self.act_qat_print.setChecked(False)
        self.act_qat_print.toggled.connect(self.btn_print.setVisible)

        self.act_qat_back = qat_menu.addAction("Back")
        self.act_qat_back.setCheckable(True)
        self.act_qat_back.setChecked(True)
        self.act_qat_back.toggled.connect(self.btn_back.setVisible)

        self.act_qat_forward = qat_menu.addAction("Forward")
        self.act_qat_forward.setCheckable(True)
        self.act_qat_forward.setChecked(True)
        self.act_qat_forward.toggled.connect(self.btn_forward.setVisible)

        self.act_qat_undo = qat_menu.addAction("Undo")
        self.act_qat_undo.setCheckable(True)
        self.act_qat_undo.setChecked(True)
        self.act_qat_undo.toggled.connect(self.btn_undo.setVisible)

        self.act_qat_redo = qat_menu.addAction("Redo")
        self.act_qat_redo.setCheckable(True)
        self.act_qat_redo.setChecked(True)
        self.act_qat_redo.toggled.connect(self.btn_redo.setVisible)

        qat_menu.addSeparator()

        self.act_qat_position = qat_menu.addAction("Show below the Ribbon")
        self.act_qat_position.triggered.connect(self._toggle_qat_position)

        self.act_qat_minimize = qat_menu.addAction("Minimize the Ribbon")
        self.act_qat_minimize.setCheckable(True)
        self.act_qat_minimize.setChecked(False)
        self.act_qat_minimize.triggered.connect(self._toggle_ribbon_collapse)

        self.btn_qat_menu.setMenu(qat_menu)
        self.qat_layout.addWidget(self.btn_qat_menu)

        # Above Ribbon Header Bar
        self.top_header = QWidget(self)
        self.top_header.setObjectName("RibbonTopHeader")
        self.top_header.setFixedHeight(26)
        self.top_hbox = QHBoxLayout(self.top_header)
        self.top_hbox.setContentsMargins(6, 2, 8, 2)
        self.top_hbox.setSpacing(3)

        self.top_hbox.addWidget(self.qat_container)

        self.header_div = QFrame(self.top_header)
        self.header_div.setVisible(False)

        self.lbl_app_title = QLabel("Photo Gallery", self.top_header)
        self.lbl_app_title.setVisible(False)

        self.top_hbox.addStretch(1)

        self.btn_help = QToolButton(self.top_header)
        self.btn_help.setObjectName("QATBtn")
        self.btn_help.setIcon(make_wlpg_icon("help", 16))
        self.btn_help.setIconSize(QSize(16, 16))
        self.btn_help.setFixedSize(22, 22)
        self.btn_help.setToolTip("Help (F1)")
        self.btn_help.setCursor(Qt.PointingHandCursor)
        self.btn_help.clicked.connect(lambda: self.action_triggered.emit("help"))
        self.top_hbox.addWidget(self.btn_help)

        main_vbox.addWidget(self.top_header)

        # 1. 5-Tab Ribbon Widget
        self.tab_widget = QTabWidget(self)
        self.tab_widget.setObjectName("RibbonTabs")
        self.tab_widget.setFixedHeight(118)

        # Blue File Button in TopLeftCorner
        self.btn_file = QPushButton("File", self.tab_widget)
        self.btn_file.setObjectName("RibbonFileBtn")
        self.btn_file.setCursor(Qt.PointingHandCursor)
        file_menu = QMenu(self.btn_file)
        file_menu.addAction(make_wlpg_icon("import", 16), "Import from camera or scanner...", lambda: self.action_triggered.emit("import_media"))
        file_menu.addAction(make_wlpg_icon("folder_add", 16), "Add folder to gallery...", lambda: self.action_triggered.emit("add_folder"))
        file_menu.addAction(make_wlpg_icon("calendar", 16), "Change date and time...", lambda: self.action_triggered.emit("change_time_taken"))
        file_menu.addSeparator()
        file_menu.addAction(make_wlpg_icon("print", 16), "Print...\tCtrl+P", lambda: self.action_triggered.emit("print"))
        file_menu.addSeparator()
        file_menu.addAction("Backup Catalog...", lambda: self.action_triggered.emit("backup_catalog"))
        file_menu.addSeparator()
        self.act_pause_write_back = QAction("Pause Background File Write-Back", self)
        self.act_pause_write_back.setCheckable(True)
        self.act_pause_write_back.setChecked(False)
        self.act_pause_write_back.triggered.connect(lambda: self.action_triggered.emit("toggle_pause_write_back"))
        file_menu.addAction(make_wlpg_icon("options", 16), "Options...", lambda: self.action_triggered.emit("open_settings"))
        file_menu.addSeparator()
        file_menu.addAction("Exit", lambda: self.action_triggered.emit("exit_app"))
        self.btn_file.setMenu(file_menu)
        self.tab_widget.setCornerWidget(self.btn_file, Qt.TopLeftCorner)

        self.tab_home = self._build_home_tab()
        self.tab_edit = self._build_edit_tab()
        self.tab_find = self._build_find_tab()
        self.tab_create = self._build_create_tab()
        self.tab_view = self._build_view_tab()

        self.tab_widget.addTab(self.tab_home, "Home")
        self.tab_widget.addTab(self.tab_edit, "Edit")
        self.tab_widget.addTab(self.tab_find, "Find")
        self.tab_widget.addTab(self.tab_create, "Create")
        self.tab_widget.addTab(self.tab_view, "View")

        self.tab_widget.tabBarDoubleClicked.connect(self._on_tab_bar_double_clicked)
        self.tab_widget.tabBarClicked.connect(self._on_tab_bar_clicked)

        # Right Corner: Minimize the Ribbon button
        right_corner = QWidget(self.tab_widget)
        rc_layout = QHBoxLayout(right_corner)
        rc_layout.setContentsMargins(2, 0, 8, 0)
        rc_layout.setSpacing(0)

        self.btn_collapse_ribbon = QPushButton("▲", right_corner)
        self.btn_collapse_ribbon.setFixedSize(22, 22)
        self.btn_collapse_ribbon.setFlat(True)
        self.btn_collapse_ribbon.setToolTip("Minimize the Ribbon (Ctrl+F1)")
        self.btn_collapse_ribbon.setCursor(Qt.PointingHandCursor)
        self.btn_collapse_ribbon.setStyleSheet(
            """
            QPushButton {
                border: 1px solid transparent;
                color: #4B6B8B;
                font-size: 8pt;
                border-radius: 2px;
                padding: 0px;
            }
            QPushButton:hover {
                background: #D9E8F7;
                border: 1px solid #AFC8E2;
            }
            """
        )
        self.btn_collapse_ribbon.clicked.connect(self._toggle_ribbon_collapse)
        rc_layout.addWidget(self.btn_collapse_ribbon)

        self.tab_widget.setCornerWidget(right_corner, Qt.TopRightCorner)

        main_vbox.addWidget(self.tab_widget)

        # 1.5. Dedicated Quick Access Toolbar Below Ribbon (Authentic WLPG Option)
        self.below_ribbon_bar = QWidget(self)
        self.below_ribbon_bar.setObjectName("RibbonBelowQAT")
        self.below_ribbon_bar.setFixedHeight(25)
        self.below_ribbon_layout = QHBoxLayout(self.below_ribbon_bar)
        self.below_ribbon_layout.setContentsMargins(6, 1, 6, 1)
        self.below_ribbon_layout.setSpacing(2)
        self.below_ribbon_layout.addStretch(1)
        self.below_ribbon_bar.hide()
        main_vbox.addWidget(self.below_ribbon_bar)
        self.scope_bar = QFrame(self)
        self.scope_bar.setObjectName("SearchScopeBar")
        self.scope_bar.setFixedHeight(30)
        scope_hbox = QHBoxLayout(self.scope_bar)
        scope_hbox.setContentsMargins(10, 2, 10, 2)
        scope_hbox.setSpacing(8)

        lbl_prefix = QLabel("Search by:", self.scope_bar)
        lbl_prefix.setStyleSheet(
            "color: #55687C; font-weight: 600; font-family: 'Segoe UI'; font-size: 9pt;"
        )
        scope_hbox.addWidget(lbl_prefix)

        self.lbl_scope = QLabel("All photos and videos", self.scope_bar)
        self.lbl_scope.setStyleSheet(
            "color: #1E395B; font-weight: 600; font-family: 'Segoe UI'; font-size: 9pt;"
        )
        scope_hbox.addWidget(self.lbl_scope)

        self.btn_search_all = QPushButton("Search all", self.scope_bar)
        self.btn_search_all.setFlat(True)
        self.btn_search_all.setCursor(Qt.PointingHandCursor)
        self.btn_search_all.setStyleSheet(
            "color: #0066CC; text-decoration: underline; border: none; padding: 0 4px; font-family: 'Segoe UI'; font-size: 8.5pt;"
        )
        self.btn_search_all.clicked.connect(self.search_all_requested.emit)
        scope_hbox.addWidget(self.btn_search_all)

        scope_hbox.addStretch(1)

        self.lbl_search_tag = QLabel("Search:", self.scope_bar)
        self.lbl_search_tag.setStyleSheet(
            "color: #55687C; font-family: 'Segoe UI'; font-size: 9pt;"
        )
        scope_hbox.addWidget(self.lbl_search_tag)

        self.txt_search = QLineEdit(self.scope_bar)
        self.txt_search.setObjectName("ScopeSearchInput")
        self.txt_search.setPlaceholderText("Search photos and videos...")
        self.txt_search.setClearButtonEnabled(True)
        self.txt_search.setFixedWidth(220)
        self.txt_search.setFixedHeight(22)
        self.txt_search.textChanged.connect(self._on_search_input_changed)
        scope_hbox.addWidget(self.txt_search)

        self.btn_close_scope = QPushButton("✕", self.scope_bar)
        self.btn_close_scope.setObjectName("ScopeCloseBtn")
        self.btn_close_scope.setToolTip("Close search")
        self.btn_close_scope.setFixedSize(22, 22)
        self.btn_close_scope.setCursor(Qt.PointingHandCursor)
        self.btn_close_scope.clicked.connect(self.hide_search_scope)
        scope_hbox.addWidget(self.btn_close_scope)

        main_vbox.addWidget(self.scope_bar)
        self._apply_aero_stylesheet()

    def remove_embedded_scope_bar(self) -> None:
        """Removes the embedded ribbon scope bar so only the central one above the grid is used."""
        if hasattr(self, "scope_bar") and self.scope_bar is not None:
            self.scope_bar.hide()
            self.scope_bar.setVisible(False)
            if hasattr(self, "main_vbox") and self.main_vbox is not None:
                self.main_vbox.removeWidget(self.scope_bar)

    def hide_search_scope(self) -> None:
        if hasattr(self, "scope_bar") and self.scope_bar is not None:
            self.scope_bar.setVisible(False)
        if hasattr(self, "txt_search") and self.txt_search is not None:
            try:
                self.txt_search.clear()
            except RuntimeError:
                pass

    def show_search_scope(self) -> None:
        if hasattr(self, "scope_bar") and self.scope_bar is not None:
            self.scope_bar.setVisible(True)
        if hasattr(self, "txt_search") and self.txt_search is not None:
            try:
                self.txt_search.setFocus()
                self.txt_search.selectAll()
            except RuntimeError:
                pass

    def toggle_search_scope(self) -> None:
        if hasattr(self, "scope_bar") and self.scope_bar is not None and self.scope_bar.isVisible():
            self.hide_search_scope()
        else:
            self.show_search_scope()

    def _set_button_compact(self, btn: QToolButton | QPushButton, compact: bool) -> None:
        orig = getattr(btn, "original_text", getattr(btn, "_orig_text", btn.text()))
        btn.original_text = orig
        btn._orig_text = orig
        if compact:
            if isinstance(btn, QToolButton):
                btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
            btn.setText("")
            has_menu = False
            if isinstance(btn, QToolButton):
                has_menu = btn.menu() is not None or btn.popupMode() in (
                    QToolButton.MenuButtonPopup,
                    QToolButton.InstantPopup,
                )
            w = 34 if has_menu else 26
            btn.setFixedWidth(w)
            btn.setToolTip(orig.replace("\n", " "))
        else:
            if isinstance(btn, QToolButton):
                btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            btn.setText(orig)
            btn.setMinimumWidth(btn.sizeHint().width())
            btn.setMaximumWidth(16777215)
            btn.setToolTip(orig.replace("\n", " "))

    def _make_tool_btn(
        self,
        text: str,
        icon_kind: str,
        parent: QWidget,
        large: bool = True,
        popup_mode: QToolButton.ToolButtonPopupMode = QToolButton.InstantPopup,
    ) -> QToolButton:
        btn = QToolButton(parent)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setPopupMode(popup_mode)
        btn.original_text = text
        btn._orig_text = text
        btn.setToolTip(text.replace("\n", " "))
        setattr(btn, "set_compact", lambda compact, b=btn: self._set_button_compact(b, compact))
        if large:
            btn.setObjectName("LargeRibbonToolBtn")
            btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            btn.setIcon(make_wlpg_icon(icon_kind, 32))
            btn.setIconSize(QSize(32, 32))
            btn.setText(text)
            fm = btn.fontMetrics()
            lines = text.split("\n")
            max_w = max((fm.horizontalAdvance(line) for line in lines), default=36)
            btn._orig_min_w = max(42, max_w + 6)
            btn.setMinimumWidth(btn._orig_min_w)
            btn.setFixedHeight(70)
            btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            self._responsive_large_buttons.append(btn)
        else:
            btn.setObjectName("CompactRibbonToolBtn")
            btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            btn.setIcon(make_wlpg_icon(icon_kind, 16))
            btn.setIconSize(QSize(16, 16))
            btn.setText(text)
            btn.setFixedHeight(21)
            btn.setMinimumWidth(btn.sizeHint().width())
            self._responsive_secondary_buttons.append(btn)
        btn._icon_kind = icon_kind
        btn._is_large = large
        if hasattr(self, "_all_themed_buttons"):
            self._all_themed_buttons.append(btn)
        return btn

    def _make_mode_toggle_btn(
        self,
        text: str,
        parent: QWidget,
        height: int = 18,
        icon_kind: str | None = None,
    ) -> QPushButton:
        btn = QPushButton(text, parent)
        btn.setObjectName("RibbonModeToggleBtn")
        btn.setCheckable(True)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFixedHeight(height)
        if icon_kind:
            btn.setIcon(make_wlpg_icon(icon_kind, 12))
            btn.setIconSize(QSize(12, 12))
            btn._icon_kind = icon_kind
            btn._is_large = False
            if hasattr(self, "_all_themed_buttons"):
                self._all_themed_buttons.append(btn)
        return btn

    def _make_btn(
        self,
        text: str,
        icon_kind: str,
        parent: QWidget,
        large: bool = False,
        checkable: bool = False,
    ) -> QToolButton:
        btn = QToolButton(parent)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setCheckable(checkable)
        btn.original_text = text
        btn._orig_text = text
        btn.setToolTip(text.replace("\n", " "))
        setattr(btn, "set_compact", lambda compact, b=btn: self._set_button_compact(b, compact))
        if large:
            btn.setObjectName("LargeRibbonBtn")
            btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            btn.setIcon(make_wlpg_icon(icon_kind, 32))
            btn.setIconSize(QSize(30, 30))
            btn.setText(text)
            fm = btn.fontMetrics()
            lines = text.split("\n")
            max_w = max((fm.horizontalAdvance(line) for line in lines), default=36)
            btn._orig_min_w = max(42, max_w + 6)
            btn.setMinimumWidth(btn._orig_min_w)
            btn.setFixedHeight(70)
            btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            self._responsive_large_buttons.append(btn)
        else:
            btn.setObjectName("CompactRibbonBtn")
            btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            btn.setIcon(make_wlpg_icon(icon_kind, 16))
            btn.setIconSize(QSize(16, 16))
            btn.setText(text)
            btn.setFixedHeight(21)
            btn.setMinimumWidth(btn.sizeHint().width())
            self._responsive_secondary_buttons.append(btn)
        btn._icon_kind = icon_kind
        btn._is_large = large
        if hasattr(self, "_all_themed_buttons"):
            self._all_themed_buttons.append(btn)
        return btn

    def refresh_icons(self, _theme_name: str = "") -> None:
        """Dynamically refreshes all ribbon button icons when the icon theme changes."""
        self.setUpdatesEnabled(False)
        try:
            for btn in getattr(self, "_all_themed_buttons", []):
                kind = getattr(btn, "_icon_kind", None)
                if kind:
                    size = 32 if getattr(btn, "_is_large", False) else 16
                    btn.setIcon(make_wlpg_icon(kind, size))
                    btn.setIconSize(QSize(size, size))
            if hasattr(self, "btn_undo"):
                self.btn_undo.setIcon(make_wlpg_icon("undo", 16))
            if hasattr(self, "btn_redo"):
                self.btn_redo.setIcon(make_wlpg_icon("redo", 16))
        finally:
            self.setUpdatesEnabled(True)

    # -------------------------------------------------------------------------
    # Responsive Resizing & Layout Breakpoints (WP-07)
    # -------------------------------------------------------------------------
    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._apply_responsive_layout(event.size().width())

    def _apply_responsive_layout(self, width: int) -> None:
        """
        Authentic WLPG priority collapse order across all tabs:

        Home Tab:
        - width >= 1220: 5-6 faces in Quick find; all buttons Expanded.
        - 1050 <= width < 1220: Shrink Quick find to 2-3 faces; collapse Months / Rated / Flagged to Icon-Only.
        - 880 <= width < 1050: Shrink Quick find to 1 face; collapse Rotate left / Rotate right / Delete
          to Icon-Only while keeping Organize and Select all / Copy / Paste Expanded with full text!
        - width < 880: Collapse Geotag / Rate / Flag and Select all / Copy / Paste to Icon-Only.
        - width < 740: Compact Batch people tag / Descriptive tag / Caption to Icon-Only.

        Edit Tab:
        - Shrink visible Effects tiles: 6 (>=1220) -> 4 (1050-1219) -> 2 (<1050).
        - Noise reduction / Color / Exposure & Rotate left/right/Delete collapse to Icon-Only at < 1050.
        - Rate / Flag / Properties & Select all/copy/paste collapse to Icon-Only at < 880.

        Find Tab:
        - People face strip: 5 (>=1220) -> 3 (1050-1219) -> 1 (<1050).
        - Edited / Media type collapse to Icon-Only at < 1050.

        View Tab:
        - Show details checkboxes collapse to Icon-Only at < 1150.
        - Arrange list buttons collapse to Icon-Only at < 980.
        """
        # --- 1. Large Button Scaling (Always vertical, authentic WLPG style) ---
        for btn in self._responsive_large_buttons:
            btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            btn.setFixedHeight(70)
            if hasattr(btn, "_orig_min_w"):
                btn.setMinimumWidth(btn._orig_min_w if width >= 800 else 44)
            btn.setText(getattr(btn, "_orig_text", btn.text()))

        # --- 2. Home Tab Priority Collapse ---
        if hasattr(self, "home_face_carousel"):
            if width >= 1220:
                self.home_face_carousel.set_visible_faces(5)
            elif width >= 1050:
                self.home_face_carousel.set_visible_faces(2)
            else:
                self.home_face_carousel.set_visible_faces(1)

        # Months / Rated / Flagged
        for btn in getattr(self, "home_qf_buttons", []):
            self._set_button_compact(btn, width < 1220)

        # Rotate left / Rotate right / Delete
        for btn in getattr(self, "home_manage_rot_delete", []):
            self._set_button_compact(btn, width < 1050)

        # Select all / Copy / Paste
        for btn in getattr(self, "home_manage_select_copy_paste", []):
            self._set_button_compact(btn, width < 880)

        # Geotag / Rate / Flag
        for btn in getattr(self, "home_org_geotag_rate_flag", []):
            self._set_button_compact(btn, width < 880)

        # Batch people tag / Descriptive tag / Caption
        for btn in getattr(self, "home_org_batch_desc_caption", []):
            self._set_button_compact(btn, width < 740)

        # --- 3. Edit Tab Priority Collapse ---
        if hasattr(self, "effects_gallery"):
            if width >= 1380:
                self.effects_gallery.set_visible_tiles(6)
            elif width >= 1240:
                self.effects_gallery.set_visible_tiles(5)
            elif width >= 1060:
                self.effects_gallery.set_visible_tiles(4)
            elif width >= 900:
                self.effects_gallery.set_visible_tiles(3)
            elif width >= 750:
                self.effects_gallery.set_visible_tiles(2)
            else:
                self.effects_gallery.set_visible_tiles(1)

        for btn in getattr(self, "edit_redeye_retouch_straighten", []):
            self._set_button_compact(btn, width < 1180)

        for btn in getattr(self, "edit_noise_color_exposure", []):
            self._set_button_compact(btn, width < 1180)

        for btn in getattr(self, "edit_manage_rot_delete", []):
            self._set_button_compact(btn, width < 1180)

        for btn in getattr(self, "edit_manage_select_copy_paste", []):
            self._set_button_compact(btn, width < 880)

        for btn in getattr(self, "edit_org_rate_flag_props", []):
            self._set_button_compact(btn, width < 880)

        for btn in getattr(self, "edit_grid_properties", []):
            self._set_button_compact(btn, width < 880)

        # --- 4. Find Tab Priority Collapse ---
        if hasattr(self, "find_face_carousel"):
            if width >= 1220:
                self.find_face_carousel.set_visible_faces(5)
            elif width >= 1050:
                self.find_face_carousel.set_visible_faces(3)
            else:
                self.find_face_carousel.set_visible_faces(1)

        for btn in getattr(self, "find_file_edited_media", []):
            self._set_button_compact(btn, width < 1050)

        # --- 5. View Tab Priority Collapse ---
        for btn in getattr(self, "view_details_checkboxes", []):
            self._set_button_compact(btn, width < 1150)

        for btn in getattr(self, "view_arrange_buttons", []):
            self._set_button_compact(btn, width < 980)

        # --- 6. Group Box Minimum Width Synchronization ---
        for page in (
            getattr(self, "tab_home", None),
            getattr(self, "tab_edit", None),
            getattr(self, "tab_find", None),
            getattr(self, "tab_create", None),
            getattr(self, "tab_view", None),
        ):
            if page and page.layout():
                for i in range(page.layout().count()):
                    w = page.layout().itemAt(i).widget()
                    if w and hasattr(w, "content_layout") and hasattr(w, "content_widget"):
                        # Invalidate sub-layouts to ensure compacted button sizes are reflected
                        for j in range(w.content_layout.count()):
                            sub = w.content_layout.itemAt(j)
                            if sub and sub.layout():
                                sub.layout().invalidate()
                        w.content_layout.invalidate()
                        cw_min = w.content_layout.sizeHint().width()
                        w.content_widget.setMinimumWidth(cw_min)
                        w.setMinimumWidth(cw_min + 4)

    # -------------------------------------------------------------------------
    # Tab Builders & Minimization
    # -------------------------------------------------------------------------
    def _on_tab_bar_double_clicked(self, index: int) -> None:
        """Double clicking any tab header minimizes or restores the ribbon."""
        import time
        self._last_double_click_time = time.monotonic()
        self._last_double_click_index = index
        self._toggle_ribbon_collapse()

    def _on_tab_bar_clicked(self, index: int) -> None:
        """When ribbon is collapsed, clicking another tab header restores it."""
        import time
        if (
            index == getattr(self, "_last_double_click_index", -1)
            and time.monotonic() - getattr(self, "_last_double_click_time", 0.0) < 0.35
        ):
            return
        if self.tab_widget.height() <= 36:
            self.tab_widget.setFixedHeight(118)
            self.btn_collapse_ribbon.setText("▲")
            if hasattr(self, "act_qat_minimize"):
                self.act_qat_minimize.setChecked(False)

    @property
    def _ribbon_collapsed(self) -> bool:
        return self.tab_widget.height() <= 36

    @property
    def collapse_btn(self) -> QPushButton:
        return self.btn_collapse_ribbon

    def _toggle_ribbon_collapse(self) -> None:
        if self.tab_widget.height() > 36:
            self.tab_widget.setFixedHeight(28)
            self.btn_collapse_ribbon.setText("▼")
            if hasattr(self, "act_qat_minimize"):
                self.act_qat_minimize.setChecked(True)
        else:
            self.tab_widget.setFixedHeight(118)
            self.btn_collapse_ribbon.setText("▲")
            if hasattr(self, "act_qat_minimize"):
                self.act_qat_minimize.setChecked(False)

    def _toggle_qat_position(self) -> None:
        self.set_qat_below_ribbon(not getattr(self, "_qat_below_ribbon", False))

    def set_qat_below_ribbon(self, below: bool) -> None:
        self._qat_below_ribbon = bool(below)
        if self._qat_below_ribbon:
            self.top_hbox.removeWidget(self.qat_container)
            self.below_ribbon_layout.insertWidget(0, self.qat_container)
            self.top_header.hide()
            self.below_ribbon_bar.show()
            self.act_qat_position.setText("Show above the Ribbon")
        else:
            self.below_ribbon_layout.removeWidget(self.qat_container)
            self.top_hbox.insertWidget(0, self.qat_container)
            self.below_ribbon_bar.hide()
            self.top_header.show()
            self.act_qat_position.setText("Show below the Ribbon")

    def set_back_enabled(self, enabled: bool) -> None:
        self.btn_back.setEnabled(enabled)

    def set_forward_enabled(self, enabled: bool) -> None:
        self.btn_forward.setEnabled(enabled)

    def set_undo_enabled(self, enabled: bool) -> None:
        self.btn_undo.setEnabled(enabled)

    def set_redo_enabled(self, enabled: bool) -> None:
        self.btn_redo.setEnabled(enabled)

    def set_revert_enabled(self, enabled: bool) -> None:
        """Enables or greys out the 'Revert to original' ribbon button."""
        if hasattr(self, "btn_edit_revert"):
            self.btn_edit_revert.setEnabled(enabled)

    def set_write_back_paused(self, paused: bool) -> None:
        if hasattr(self, "act_pause_write_back"):
            self.act_pause_write_back.setChecked(bool(paused))

    def minimumSizeHint(self) -> QSize:
        return QSize(350, 88)

    @staticmethod
    def _wrap_tab_page(page: QWidget) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        scroll.setWidget(page)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        return scroll

    def _build_home_tab(self) -> QWidget:
        page = _RibbonPage()
        hbox = QHBoxLayout(page)
        hbox.setContentsMargins(4, 2, 4, 2)
        hbox.setSpacing(4)

        # ---------------------------------------------------------------------
        # Group 1: New (Import, New folder)
        # ---------------------------------------------------------------------
        grp_new = _RibbonGroupBox("New", page)
        btn_import = self._make_btn("Import", "import", grp_new, large=True)
        btn_import.clicked.connect(lambda: self.action_triggered.emit("import_media"))
        btn_add_folder = self._make_btn("New\nfolder", "folder_add", grp_new, large=True)
        btn_add_folder.clicked.connect(lambda: self.action_triggered.emit("add_folder"))
        grp_new.content_layout.addWidget(btn_import)
        grp_new.content_layout.addWidget(btn_add_folder)
        hbox.addWidget(grp_new)

        # ---------------------------------------------------------------------
        # Group 2: Manage (2 columns of 3 rows: Select all/Copy/Paste & Rotate left/right/Delete)
        # ---------------------------------------------------------------------
        grp_manage = _RibbonGroupBox("Manage", page)

        m_grid = QGridLayout()
        m_grid.setContentsMargins(0, 0, 0, 0)
        m_grid.setHorizontalSpacing(2)
        m_grid.setVerticalSpacing(1)

        self.btn_select_all = self._make_btn("Select all", "select_all", grp_manage)
        self.btn_select_all.clicked.connect(lambda: self.action_triggered.emit("select_all"))
        self.btn_copy = self._make_btn("Copy", "copy", grp_manage)
        self.btn_copy.clicked.connect(lambda: self.action_triggered.emit("copy"))
        self.btn_paste = self._make_btn("Paste", "paste", grp_manage)
        self.btn_paste.clicked.connect(lambda: self.action_triggered.emit("paste"))

        self.btn_rot_ccw = self._make_btn("Rotate left", "rotate_ccw", grp_manage)
        self.btn_rot_cw = self._make_btn("Rotate right", "rotate_cw", grp_manage)
        self.btn_delete = self._make_btn("Delete", "delete", grp_manage)
        self.btn_rot_ccw.clicked.connect(lambda: self.action_triggered.emit("rotate_left"))
        self.btn_rot_cw.clicked.connect(lambda: self.action_triggered.emit("rotate_right"))
        self.btn_delete.clicked.connect(lambda: self.action_triggered.emit("delete"))

        # Col 0 (left): Select all, Copy, Paste
        m_grid.addWidget(self.btn_select_all, 0, 0)
        m_grid.addWidget(self.btn_copy, 1, 0)
        m_grid.addWidget(self.btn_paste, 2, 0)

        # Col 1 (right): Rotate left, Rotate right, Delete
        m_grid.addWidget(self.btn_rot_ccw, 0, 1)
        m_grid.addWidget(self.btn_rot_cw, 1, 1)
        m_grid.addWidget(self.btn_delete, 2, 1)

        grp_manage.content_layout.addLayout(m_grid)
        hbox.addWidget(grp_manage)

        # ---------------------------------------------------------------------
        # Group 3: Organize (People tag, Batch people tag, Descriptive tag, Caption | Geotag, Rate, Flag)
        # ---------------------------------------------------------------------
        grp_org = _RibbonGroupBox("Organize", page)

        self.btn_people_tag = self._make_btn("People\ntag", "people_tag", grp_org, large=True)
        self.btn_people_tag.setToolTip("Identify a person in a photo")
        self.btn_people_tag.clicked.connect(lambda: self.action_triggered.emit("people_tag"))
        grp_org.content_layout.addWidget(self.btn_people_tag)

        # Middle sub-column (3 stacked rows): Batch people tag, Descriptive tag, Caption
        o_mid_col = QVBoxLayout()
        o_mid_col.setContentsMargins(0, 0, 0, 0)
        o_mid_col.setSpacing(1)

        self.btn_batch_people = self._make_btn("Batch people tag", "batch_people", grp_org, large=False)
        self.btn_batch_people.clicked.connect(lambda: self.action_triggered.emit("batch_people_tag"))
        o_mid_col.addWidget(self.btn_batch_people)

        self.btn_tag_desc = self._make_btn("Descriptive tag", "tag", grp_org, large=False)
        self.btn_tag_desc.clicked.connect(lambda: self.action_triggered.emit("toggle_tag_pane"))
        o_mid_col.addWidget(self.btn_tag_desc)

        self.btn_caption = self._make_btn("Caption", "caption", grp_org, large=False)
        self.btn_caption.clicked.connect(lambda: self.action_triggered.emit("edit_caption"))
        o_mid_col.addWidget(self.btn_caption)
        grp_org.content_layout.addLayout(o_mid_col)

        # Right sub-column: Geotag, Rate, Flag (3 stacked rows)
        o_right_col = QVBoxLayout()
        o_right_col.setContentsMargins(0, 0, 0, 0)
        o_right_col.setSpacing(1)

        self.btn_geotag = self._make_btn("Geotag", "geotag", grp_org)
        self.btn_geotag.clicked.connect(lambda: self.action_triggered.emit("geotag"))

        self.btn_home_rate = self._make_btn("Rate", "rate", grp_org)
        rate_menu = QMenu(self.btn_home_rate)
        for s in (5, 4, 3, 2, 1):
            stars_str = "★" * s + "☆" * (5 - s)
            rate_menu.addAction(f"{stars_str} ({s} stars)", lambda r=s: self.action_triggered.emit(f"rate_{r}"))
        rate_menu.addSeparator()
        rate_menu.addAction("Clear rating (unrated)", lambda: self.action_triggered.emit("rate_0"))
        self.btn_home_rate.setMenu(rate_menu)

        self.btn_home_flag = self._make_btn("Flag", "flag", grp_org)
        self.btn_home_flag.clicked.connect(lambda: self.action_triggered.emit("toggle_flag"))

        o_right_col.addWidget(self.btn_geotag)
        o_right_col.addWidget(self.btn_home_rate)
        o_right_col.addWidget(self.btn_home_flag)
        grp_org.content_layout.addLayout(o_right_col)

        hbox.addWidget(grp_org)

        # Home group button collections for priority collapse
        self.home_manage_rot_delete = [self.btn_rot_ccw, self.btn_rot_cw, self.btn_delete]
        self.home_manage_select_copy_paste = [self.btn_select_all, self.btn_copy, self.btn_paste]
        self.home_org_geotag_rate_flag = [self.btn_geotag, self.btn_home_rate, self.btn_home_flag]
        self.home_org_batch_desc_caption = [self.btn_batch_people, self.btn_tag_desc, self.btn_caption]

        # ---------------------------------------------------------------------
        # Group 4: Quick find (Face carousel + Months/Rated/Flagged + Text search)
        # ---------------------------------------------------------------------
        grp_qfind = _RibbonGroupBox("Quick find", page)

        # 1. Flexible Face Carousel (showing 1-6 faces)
        self.home_face_carousel = _FaceCarouselWidget(grp_qfind)
        self.home_face_carousel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.home_face_carousel.person_clicked.connect(self.person_quick_find_selected.emit)
        self.home_face_carousel.boolean_mode_selected.connect(self._on_carousel_boolean_mode)
        self.home_face_carousel.special_filter_selected.connect(self.people_special_filter_selected.emit)

        # 2. Dropdown Filter Buttons: Months, Rated, Flagged (3 stacked rows)
        qf_filters_col = QVBoxLayout()
        qf_filters_col.setContentsMargins(0, 0, 0, 0)
        qf_filters_col.setSpacing(1)
        qf_filters_col.setAlignment(Qt.AlignLeft)

        self.btn_qf_months = self._make_btn("Months", "months", grp_qfind)
        self.btn_qf_months.setFixedHeight(19)
        self.btn_qf_months.setStyleSheet("font-size: 8pt; padding: 0 10px 0 3px;")
        months_menu = QMenu(self.btn_qf_months)
        months_menu.addAction("All dates", lambda: self.filter_month_selected.emit(0))
        months_menu.addSeparator()
        for m_num in range(1, 13):
            m_name = calendar.month_name[m_num]
            months_menu.addAction(m_name, lambda mn=m_num: self.filter_month_selected.emit(mn))
        self.btn_qf_months.setMenu(months_menu)

        self.btn_qf_rated = self._make_btn("Rated", "rate", grp_qfind)
        self.btn_qf_rated.setFixedHeight(19)
        self.btn_qf_rated.setStyleSheet("font-size: 8pt; padding: 0 10px 0 3px;")
        rated_menu = QMenu(self.btn_qf_rated)
        rated_menu.addAction("All ratings", lambda: self.rating_filter_selected.emit(()))
        rated_menu.addSeparator()
        rated_menu.addAction("★★★★★ (5 stars)", lambda: self.rating_filter_selected.emit((5,)))
        rated_menu.addAction("★★★★☆ or higher", lambda: self.rating_filter_selected.emit((4, 5)))
        rated_menu.addAction("★★★☆☆ or higher", lambda: self.rating_filter_selected.emit((3, 4, 5)))
        rated_menu.addAction("★★☆☆☆ or higher", lambda: self.rating_filter_selected.emit((2, 3, 4, 5)))
        rated_menu.addAction("★☆☆☆☆ or higher", lambda: self.rating_filter_selected.emit((1, 2, 3, 4, 5)))
        rated_menu.addAction("Unrated", lambda: self.rating_filter_selected.emit((0,)))
        self.btn_qf_rated.setMenu(rated_menu)

        # Split Flagged Button: Left part toggles red flag filter, right arrow opens choice menu
        self.btn_qf_flagged = QToolButton(grp_qfind)
        self.btn_qf_flagged.setPopupMode(QToolButton.MenuButtonPopup)
        self.btn_qf_flagged.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_qf_flagged.setIcon(make_wlpg_icon("flag", 13))
        self.btn_qf_flagged.setText("Flagged")
        self.btn_qf_flagged.setFixedHeight(19)
        self.btn_qf_flagged.setCheckable(True)
        self.btn_qf_flagged.original_text = "Flagged"
        self.btn_qf_flagged._orig_text = "Flagged"
        setattr(self.btn_qf_flagged, "set_compact", lambda compact, b=self.btn_qf_flagged: self._set_button_compact(b, compact))
        self.btn_qf_flagged.setCheckable(True)
        self.btn_qf_flagged.setCursor(Qt.PointingHandCursor)
        self.btn_qf_flagged.setStyleSheet(
            f"""
            QToolButton {{
                font-family: 'Segoe UI';
                font-size: 8pt;
                padding: 0 10px 0 3px;
                border: 1px solid transparent;
                border-radius: 2px;
                color: #1E395B;
                background: transparent;
            }}
            QToolButton:hover {{
                background: #E8F1FC;
                border-color: #B8D6FB;
            }}
            QToolButton:checked {{
                background: #CCE4F7;
                border-color: #7BAEDB;
                color: #C00000;
                font-weight: bold;
            }}
            QToolButton::menu-button {{
                border: none;
                background: transparent;
                width: 10px;
            }}
            QToolButton::menu-arrow {{
                image: url("{ARROW_DROPDOWN_PATH}");
                width: 5px;
                height: 3px;
            }}
            """
        )
        flagged_menu = QMenu(self.btn_qf_flagged)
        act_f_all = flagged_menu.addAction("All items")
        act_f_flagged = flagged_menu.addAction("Flagged")
        act_f_unflagged = flagged_menu.addAction("Not flagged")

        def _on_flag_btn_clicked(checked: bool):
            self.flag_filter_selected.emit(1 if checked else None)
            self.flagged_filter_selected.emit(checked)

        self.btn_qf_flagged.clicked.connect(_on_flag_btn_clicked)

        def _on_menu_all():
            self.btn_qf_flagged.blockSignals(True)
            self.btn_qf_flagged.setChecked(False)
            self.btn_qf_flagged.blockSignals(False)
            self.flag_filter_selected.emit(None)
            self.flagged_filter_selected.emit(False)

        def _on_menu_flagged():
            self.btn_qf_flagged.blockSignals(True)
            self.btn_qf_flagged.setChecked(True)
            self.btn_qf_flagged.blockSignals(False)
            self.flag_filter_selected.emit(1)
            self.flagged_filter_selected.emit(True)

        def _on_menu_unflagged():
            self.btn_qf_flagged.blockSignals(True)
            self.btn_qf_flagged.setChecked(False)
            self.btn_qf_flagged.blockSignals(False)
            self.flag_filter_selected.emit(0)
            self.flagged_filter_selected.emit(False)

        act_f_all.triggered.connect(_on_menu_all)
        act_f_flagged.triggered.connect(_on_menu_flagged)
        act_f_unflagged.triggered.connect(_on_menu_unflagged)
        self.btn_qf_flagged.setMenu(flagged_menu)

        qf_filters_col.addWidget(self.btn_qf_months)
        qf_filters_col.addWidget(self.btn_qf_rated)
        qf_filters_col.addWidget(self.btn_qf_flagged)
        self.home_qf_buttons = [self.btn_qf_months, self.btn_qf_rated, self.btn_qf_flagged]

        # 3. Text search large button (toggles Search Scope Bar)
        btn_text_search = self._make_btn("Text\nsearch", "text_search", grp_qfind, large=True)
        btn_text_search.clicked.connect(lambda: self.action_triggered.emit("toggle_scope_bar"))

        grp_qfind.content_layout.setContentsMargins(0, 0, 0, 0)
        grp_qfind.content_layout.setSpacing(0)
        grp_qfind.content_layout.addWidget(self.home_face_carousel)
        grp_qfind.content_layout.addSpacing(2)
        grp_qfind.content_layout.addLayout(qf_filters_col)
        grp_qfind.content_layout.addSpacing(2)
        grp_qfind.content_layout.addWidget(btn_text_search)

        hbox.addWidget(grp_qfind)

        # ---------------------------------------------------------------------
        # Group 5: Share / Presentation (Slide show)
        # ---------------------------------------------------------------------
        grp_share = _RibbonGroupBox("Share", page)
        btn_ss = self._make_btn("Slide\nshow", "slideshow", grp_share, large=True)
        btn_ss.clicked.connect(lambda: self.action_triggered.emit("slide_show"))
        grp_share.content_layout.addWidget(btn_ss)
        hbox.addWidget(grp_share)

        hbox.addStretch(1)

        return page

    def _build_edit_tab(self) -> QWidget:
        """
        Contextual 'Edit' tab matching Windows Live Photo Gallery:
        - In Grid View: Manage, Properties, Quick adjustments, Effects, Revert to original, View file.
        - In Single View: Iterate, Manage, Organize, Adjustments, Effects, Revert to original, Close file.
        """
        page = _RibbonPage()
        hbox = QHBoxLayout(page)
        hbox.setContentsMargins(2, 2, 2, 2)
        hbox.setSpacing(2)

        # 1. Iterate Group (Single View only: Previous |<<, Next >>|)
        self.grp_edit_iterate = _RibbonGroupBox("Iterate", page)
        self.btn_edit_prev = self._make_btn("Previous", "prev_media", self.grp_edit_iterate, large=True)
        self.btn_edit_prev.setToolTip("Previous image (Left Arrow)")
        self.btn_edit_prev.clicked.connect(lambda: self.action_triggered.emit("prev_media"))

        self.btn_edit_next = self._make_btn("Next", "next_media", self.grp_edit_iterate, large=True)
        self.btn_edit_next.setToolTip("Next image (Right Arrow)")
        self.btn_edit_next.clicked.connect(lambda: self.action_triggered.emit("next_media"))

        self.grp_edit_iterate.content_layout.addWidget(self.btn_edit_prev)
        self.grp_edit_iterate.content_layout.addWidget(self.btn_edit_next)
        hbox.addWidget(self.grp_edit_iterate)

        # 2. Manage Group
        self.grp_edit_manage = _RibbonGroupBox("Manage", page)

        # Col 1: Grid only (Select all, Copy, Paste)
        self.col_grid_manage = QWidget(self.grp_edit_manage)
        cgm_vbox = QVBoxLayout(self.col_grid_manage)
        cgm_vbox.setContentsMargins(0, 0, 0, 0)
        cgm_vbox.setSpacing(1)

        self.btn_edit_select_all = self._make_btn("Select all", "select_all", self.col_grid_manage)
        self.btn_edit_select_all.setToolTip("Select all (Ctrl+A)")
        self.btn_edit_select_all.clicked.connect(lambda: self.action_triggered.emit("select_all"))

        self.btn_edit_copy = self._make_btn("Copy", "copy", self.col_grid_manage)
        self.btn_edit_copy.setToolTip("Copy (Ctrl+C)")
        self.btn_edit_copy.clicked.connect(lambda: self.action_triggered.emit("copy"))

        self.btn_edit_paste = self._make_btn("Paste", "paste", self.col_grid_manage)
        self.btn_edit_paste.setToolTip("Paste (Ctrl+V)")
        self.btn_edit_paste.clicked.connect(lambda: self.action_triggered.emit("paste"))

        cgm_vbox.addWidget(self.btn_edit_select_all)
        cgm_vbox.addWidget(self.btn_edit_copy)
        cgm_vbox.addWidget(self.btn_edit_paste)
        self.grp_edit_manage.content_layout.addWidget(self.col_grid_manage)

        # Col 2: Common (Rotate left, Rotate right, Delete)
        self.col_common_manage = QWidget(self.grp_edit_manage)
        ccm_vbox = QVBoxLayout(self.col_common_manage)
        ccm_vbox.setContentsMargins(0, 0, 0, 0)
        ccm_vbox.setSpacing(1)

        self.btn_edit_rot_l = self._make_btn("Rotate left", "rotate_ccw", self.col_common_manage)
        self.btn_edit_rot_l.setToolTip("Rotate left (Ctrl+,)")
        self.btn_edit_rot_l.clicked.connect(lambda: self.action_triggered.emit("rotate_left"))

        self.btn_edit_rot_r = self._make_btn("Rotate right", "rotate_cw", self.col_common_manage)
        self.btn_edit_rot_r.setToolTip("Rotate right (Ctrl+.)")
        self.btn_edit_rot_r.clicked.connect(lambda: self.action_triggered.emit("rotate_right"))

        self.btn_edit_del = self._make_btn("Delete", "delete", self.col_common_manage)
        self.btn_edit_del.setToolTip("Delete (Del)")
        self.btn_edit_del.clicked.connect(lambda: self.action_triggered.emit("delete"))

        ccm_vbox.addWidget(self.btn_edit_rot_l)
        ccm_vbox.addWidget(self.btn_edit_rot_r)
        ccm_vbox.addWidget(self.btn_edit_del)
        self.grp_edit_manage.content_layout.addWidget(self.col_common_manage)

        # Single View button: Make a copy
        self.btn_edit_make_copy = self._make_btn("Make\na copy", "make_copy", self.grp_edit_manage, large=True)
        self.btn_edit_make_copy.setToolTip("Save a duplicate copy of this photo")
        self.btn_edit_make_copy.clicked.connect(lambda: self.action_triggered.emit("make_copy"))
        self.grp_edit_manage.content_layout.addWidget(self.btn_edit_make_copy)
        hbox.addWidget(self.grp_edit_manage)

        # 3. Properties Group (Grid view only)
        self.grp_edit_properties = _RibbonGroupBox("Properties", page)
        props_vbox = QVBoxLayout()
        props_vbox.setContentsMargins(0, 0, 0, 0)
        props_vbox.setSpacing(2)

        self.btn_edit_rename = self._make_btn("Rename", "rename", self.grp_edit_properties)
        self.btn_edit_rename.setToolTip("Rename file (F2)")
        self.btn_edit_rename.clicked.connect(lambda: self.action_triggered.emit("rename_file"))

        self.btn_edit_resize = self._make_btn("Resize", "resize", self.grp_edit_properties)
        self.btn_edit_resize.setToolTip("Resize photo...")
        self.btn_edit_resize.clicked.connect(lambda: self.action_triggered.emit("resize_photo"))

        self.btn_edit_time = self._make_btn("Adjust time", "calendar", self.grp_edit_properties)
        self.btn_edit_time.setToolTip("Adjust date and time taken...")
        self.btn_edit_time.clicked.connect(lambda: self.action_triggered.emit("change_time_taken"))

        props_vbox.addWidget(self.btn_edit_rename)
        props_vbox.addWidget(self.btn_edit_resize)
        props_vbox.addWidget(self.btn_edit_time)
        self.grp_edit_properties.content_layout.addLayout(props_vbox)
        hbox.addWidget(self.grp_edit_properties)

        # 4. Organize Group (Single view only: Tag and caption, Rate ▾, Flag, Properties ▾)
        self.grp_edit_organize = _RibbonGroupBox("Organize", page)

        self.btn_edit_tag_caption = self._make_btn("Tag and\ncaption", "tag", self.grp_edit_organize, large=True)
        self.btn_edit_tag_caption.setToolTip("Show information pane (Ctrl+I)")
        self.btn_edit_tag_caption.clicked.connect(lambda: self.action_triggered.emit("toggle_tag_pane"))
        self.grp_edit_organize.content_layout.addWidget(self.btn_edit_tag_caption)

        # Retain btn_edit_people_tag attribute for backwards compatibility
        self.btn_edit_people_tag = self._make_btn("People\ntag", "people_tag", self.grp_edit_organize, large=True)
        self.btn_edit_people_tag.setToolTip("Identify a person in this photo")
        self.btn_edit_people_tag.clicked.connect(lambda: self.action_triggered.emit("people_tag"))
        self.btn_edit_people_tag.setVisible(False)

        org_col = QVBoxLayout()
        org_col.setContentsMargins(0, 0, 0, 0)
        org_col.setSpacing(2)

        self.btn_edit_rate = self._make_btn("Rate", "rate", self.grp_edit_organize)
        rate_menu = QMenu(self.btn_edit_rate)
        rate_menu.addAction("Unrated (0)", lambda: self.action_triggered.emit("rate_0"))
        rate_menu.addAction("★ (1)", lambda: self.action_triggered.emit("rate_1"))
        rate_menu.addAction("★★ (2)", lambda: self.action_triggered.emit("rate_2"))
        rate_menu.addAction("★★★ (3)", lambda: self.action_triggered.emit("rate_3"))
        rate_menu.addAction("★★★★ (4)", lambda: self.action_triggered.emit("rate_4"))
        rate_menu.addAction("★★★★★ (5)", lambda: self.action_triggered.emit("rate_5"))
        self.btn_edit_rate.setMenu(rate_menu)

        self.btn_edit_flag = self._make_btn("Flag", "flag", self.grp_edit_organize)
        self.btn_edit_flag.setToolTip("Toggle flag")
        self.btn_edit_flag.clicked.connect(lambda: self.action_triggered.emit("toggle_flag"))

        self.btn_edit_prop_drop = self._make_btn("Properties", "options", self.grp_edit_organize)
        prop_menu = QMenu(self.btn_edit_prop_drop)
        prop_menu.addAction("Adjust date and time...", lambda: self.action_triggered.emit("change_time_taken"))
        prop_menu.addAction("Resize photo...", lambda: self.action_triggered.emit("resize_photo"))
        prop_menu.addAction("Rename photo...", lambda: self.action_triggered.emit("rename_file"))
        self.btn_edit_prop_drop.setMenu(prop_menu)

        org_col.addWidget(self.btn_edit_rate)
        org_col.addWidget(self.btn_edit_flag)
        org_col.addWidget(self.btn_edit_prop_drop)
        self.grp_edit_organize.content_layout.addLayout(org_col)
        hbox.addWidget(self.grp_edit_organize)

        # 5. Quick Adjustments / Adjustments Group
        self.grp_edit_adjust = _RibbonGroupBox("Quick adjustments", page)

        self.btn_edit_auto = self._make_btn("Auto\nadjust", "auto_adjust", self.grp_edit_adjust, large=True)
        self.btn_edit_auto.setToolTip("Automatically adjust brightness, contrast, and color")
        self.btn_edit_auto.clicked.connect(lambda: self.action_triggered.emit("auto_adjust"))
        self.grp_edit_adjust.content_layout.addWidget(self.btn_edit_auto)

        self.btn_edit_crop = self._make_btn("Crop", "crop", self.grp_edit_adjust, large=True)
        self.btn_edit_crop.setToolTip("Crop photo (draw crop rectangle)")
        self.btn_edit_crop.clicked.connect(lambda: self.action_triggered.emit("crop_photo"))
        self.grp_edit_adjust.content_layout.addWidget(self.btn_edit_crop)

        # 3-row stack of canvas repair tools: Red eye, Retouch, Straighten (WLPG style)
        self.stack_canvas_tools = QVBoxLayout()
        self.stack_canvas_tools.setContentsMargins(0, 0, 0, 0)
        self.stack_canvas_tools.setSpacing(1)

        self.btn_edit_red_eye = self._make_btn("Red eye", "red_eye", self.grp_edit_adjust, large=False)
        self.btn_edit_red_eye.setToolTip("Fix red eye")
        self.btn_edit_red_eye.clicked.connect(lambda: self.action_triggered.emit("red_eye"))

        self.btn_edit_retouch = self._make_btn("Retouch", "retouch", self.grp_edit_adjust, large=False)
        self.btn_edit_retouch.setToolTip("Retouch blemishes")
        self.btn_edit_retouch.clicked.connect(lambda: self.action_triggered.emit("retouch"))

        self.btn_edit_straighten = self._make_btn("Straighten", "straighten", self.grp_edit_adjust, large=False)
        self.btn_edit_straighten.setToolTip("Straighten photo horizon / vertical alignment")
        self.btn_edit_straighten.clicked.connect(lambda: self.action_triggered.emit("straighten"))

        self.stack_canvas_tools.addWidget(self.btn_edit_red_eye)
        self.stack_canvas_tools.addWidget(self.btn_edit_retouch)
        self.stack_canvas_tools.addWidget(self.btn_edit_straighten)
        self.grp_edit_adjust.content_layout.addLayout(self.stack_canvas_tools)

        # Stack of 3 small buttons: Noise reduction, Color, Exposure
        stack_adj = QVBoxLayout()
        stack_adj.setContentsMargins(0, 0, 0, 0)
        stack_adj.setSpacing(1)

        self.btn_edit_noise = self._make_btn("Noise reduction", "noise_reduction", self.grp_edit_adjust)
        self.btn_edit_noise.setToolTip("Reduce digital sensor noise")
        self.btn_edit_noise.clicked.connect(lambda: self.action_triggered.emit("noise_reduction"))

        self.btn_edit_color = self._make_btn("Color", "adjust_color", self.grp_edit_adjust)
        self.btn_edit_color.setPopupMode(QToolButton.InstantPopup)
        self.menu_color = QMenu(self.btn_edit_color)
        self.menu_color.setStyleSheet("QMenu { background-color: #FFFFFF; border: 1px solid #70A6E8; }")
        self.color_menu_widget = _ColorMenuWidget(self.menu_color)
        self.color_menu_widget.preview_requested.connect(self.color_preview_requested.emit)
        self.color_menu_widget.auto_preview_requested.connect(self.auto_color_preview_requested.emit)
        self.color_menu_widget.preview_cancelled.connect(self.color_preview_cancelled.emit)
        self.color_menu_widget.adjustment_selected.connect(self.color_adjustment_selected.emit)
        self.color_menu_widget.auto_selected.connect(self.auto_color_selected.emit)
        act_col = QWidgetAction(self.menu_color)
        act_col.setDefaultWidget(self.color_menu_widget)
        self.menu_color.addAction(act_col)
        self.btn_edit_color.setMenu(self.menu_color)
        self.menu_color.aboutToShow.connect(self.color_menu_about_to_show.emit)
        self.menu_color.aboutToHide.connect(self.color_preview_cancelled.emit)

        self.btn_edit_exposure = self._make_btn("Exposure", "adjust_exposure", self.grp_edit_adjust)
        self.btn_edit_exposure.setPopupMode(QToolButton.InstantPopup)
        self.menu_exposure = QMenu(self.btn_edit_exposure)
        self.menu_exposure.setStyleSheet("QMenu { background-color: #FFFFFF; border: 1px solid #70A6E8; }")
        self.exposure_menu_widget = _ExposureMenuWidget(self.menu_exposure)
        self.exposure_menu_widget.preview_requested.connect(self.exposure_preview_requested.emit)
        self.exposure_menu_widget.auto_preview_requested.connect(self.auto_exposure_preview_requested.emit)
        self.exposure_menu_widget.preview_cancelled.connect(self.exposure_preview_cancelled.emit)
        self.exposure_menu_widget.adjustment_selected.connect(self.exposure_adjustment_selected.emit)
        self.exposure_menu_widget.auto_selected.connect(self.auto_exposure_selected.emit)
        act_exp = QWidgetAction(self.menu_exposure)
        act_exp.setDefaultWidget(self.exposure_menu_widget)
        self.menu_exposure.addAction(act_exp)
        self.btn_edit_exposure.setMenu(self.menu_exposure)
        self.menu_exposure.aboutToShow.connect(self.exposure_menu_about_to_show.emit)
        self.menu_exposure.aboutToHide.connect(self.exposure_preview_cancelled.emit)

        stack_adj.addWidget(self.btn_edit_noise)
        stack_adj.addWidget(self.btn_edit_color)
        stack_adj.addWidget(self.btn_edit_exposure)
        self.grp_edit_adjust.content_layout.addLayout(stack_adj)

        self.btn_edit_fine_tune = self._make_btn("Fine\ntune", "fine_tune", self.grp_edit_adjust, large=True)
        self.btn_edit_fine_tune.setToolTip("Fine-tune exposure, color, straighten, and detail (Ctrl+F)")
        self.btn_edit_fine_tune.clicked.connect(lambda: self.action_triggered.emit("fine_tune"))
        self.grp_edit_adjust.content_layout.addWidget(self.btn_edit_fine_tune)
        hbox.addWidget(self.grp_edit_adjust)

        # 6. Effects Group (Thumbnail preview gallery matching WLPG)
        self.grp_edit_effects = _RibbonGroupBox("Effects", page)
        self.effects_gallery = _RibbonEffectsGallery(self.grp_edit_effects)
        self.effects_gallery.effect_selected.connect(
            lambda eff: self.action_triggered.emit(f"effect_{eff}")
        )
        self.grp_edit_effects.content_layout.addWidget(self.effects_gallery)
        hbox.addWidget(self.grp_edit_effects)

        # 7. Revert to original
        self.grp_edit_revert = _RibbonGroupBox("Revert", page)
        self.btn_edit_revert = self._make_btn(
            "Revert to\noriginal", "revert_original", self.grp_edit_revert, large=True
        )
        self.btn_edit_revert.setToolTip("Revert all edits back to the pristine original photo")
        self.btn_edit_revert.setEnabled(False)
        self.btn_edit_revert.clicked.connect(lambda: self.action_triggered.emit("revert_original"))
        self.grp_edit_revert.content_layout.addWidget(self.btn_edit_revert)
        hbox.addWidget(self.grp_edit_revert)

        # 8. View file / Close file Group
        self.grp_edit_close = _RibbonGroupBox("View", page)

        self.btn_edit_view_file = self._make_btn("View\nfile", "view_file", self.grp_edit_close, large=True)
        self.btn_edit_view_file.setToolTip("Open selected photo in viewer (Enter)")
        self.btn_edit_view_file.clicked.connect(lambda: self.action_triggered.emit("view_file"))
        self.grp_edit_close.content_layout.addWidget(self.btn_edit_view_file)

        self.btn_edit_close_file = self._make_btn("Close\nfile", "close_file", self.grp_edit_close, large=True)
        self.btn_edit_close_file.setToolTip("Close viewer and return to grid (Esc)")
        self.btn_edit_close_file.clicked.connect(lambda: self.action_triggered.emit("close_viewer"))
        self.grp_edit_close.content_layout.addWidget(self.btn_edit_close_file)
        hbox.addWidget(self.grp_edit_close)

        # Edit group button collections for priority collapse
        self.edit_redeye_retouch_straighten = [self.btn_edit_red_eye, self.btn_edit_retouch, self.btn_edit_straighten]
        self.edit_noise_color_exposure = [self.btn_edit_noise, self.btn_edit_color, self.btn_edit_exposure]
        self.edit_manage_rot_delete = [self.btn_edit_rot_l, self.btn_edit_rot_r, self.btn_edit_del]
        self.edit_manage_select_copy_paste = [self.btn_edit_select_all, self.btn_edit_copy, self.btn_edit_paste]
        self.edit_org_rate_flag_props = [self.btn_edit_rate, self.btn_edit_flag, self.btn_edit_prop_drop]
        self.edit_grid_properties = [self.btn_edit_rename, self.btn_edit_resize, self.btn_edit_time]

        hbox.addStretch(1)

        self._set_edit_tab_mode("grid")
        return page

    def _set_edit_tab_mode(self, mode: Literal["grid", "viewer"]) -> None:
        is_viewer = (mode == "viewer")
        if hasattr(self, "grp_edit_iterate"):
            self.grp_edit_iterate.setVisible(is_viewer)
        if hasattr(self, "col_grid_manage"):
            self.col_grid_manage.setVisible(not is_viewer)
        if hasattr(self, "btn_edit_make_copy"):
            self.btn_edit_make_copy.setVisible(is_viewer)
        if hasattr(self, "grp_edit_properties"):
            self.grp_edit_properties.setVisible(not is_viewer)
        if hasattr(self, "grp_edit_organize"):
            self.grp_edit_organize.setVisible(is_viewer)
        # Canvas tools remain accessible in Grid View, routing to Single Viewer with the tool active
        if hasattr(self, "btn_edit_crop"):
            self.btn_edit_crop.setVisible(True)
        if hasattr(self, "btn_edit_red_eye"):
            self.btn_edit_red_eye.setVisible(True)
        if hasattr(self, "btn_edit_retouch"):
            self.btn_edit_retouch.setVisible(True)
        if hasattr(self, "btn_edit_straighten"):
            self.btn_edit_straighten.setVisible(True)
        if hasattr(self, "btn_edit_fine_tune"):
            self.btn_edit_fine_tune.setVisible(True)
        if hasattr(self, "btn_edit_view_file"):
            self.btn_edit_view_file.setVisible(not is_viewer)
        if hasattr(self, "btn_edit_close_file"):
            self.btn_edit_close_file.setVisible(is_viewer)
        if hasattr(self, "grp_edit_adjust"):
            self.grp_edit_adjust.setTitle("Adjustments" if is_viewer else "Quick adjustments")
        if hasattr(self, "grp_edit_close"):
            self.grp_edit_close.setTitle("Close" if is_viewer else "View")

    def _build_find_tab(self) -> QWidget:
        """
        'Find' tab matching Windows Live Photo Gallery reference screenshots:
        1. Date taken: Dates, Months, Years dropdown buttons
        2. People: Face carousel + stacked (+ and / / or) toggle buttons
        3. Rated: Rated dropdown button + stacked (↑ and higher / ↓ and lower / = only) mode buttons
        4. Tags: Tags dropdown button + stacked (+ and / / or) toggle buttons
        5. File details: Flagged, Edited, Media type dropdown button
        6. Text search: Large Text search binoculars button (Ctrl+F)
        """
        page = _RibbonPage()
        hbox = QHBoxLayout(page)
        hbox.setContentsMargins(4, 2, 4, 2)
        hbox.setSpacing(4)

        # ---------------------------------------------------------------------
        # 1. Date taken Group (Dates, Months, Years)
        # ---------------------------------------------------------------------
        grp_dates = _RibbonGroupBox("Date taken", page)
        d_hbox = QHBoxLayout()
        d_hbox.setContentsMargins(0, 0, 0, 0)
        d_hbox.setSpacing(2)

        self.btn_find_dates = self._make_tool_btn("Dates", "dates_grid", grp_dates, large=True)
        self.btn_find_dates.setToolTip("Filter by relative date range")
        self._build_dates_menu(self.btn_find_dates)
        d_hbox.addWidget(self.btn_find_dates)

        self.btn_find_months = self._make_tool_btn("Months", "months_grid", grp_dates, large=True)
        self.btn_find_months.setToolTip("Filter by month")
        self._build_months_menu(self.btn_find_months)
        d_hbox.addWidget(self.btn_find_months)

        self.btn_find_years = self._make_tool_btn("Years", "years_stack", grp_dates, large=True)
        self.btn_find_years.setToolTip("Filter by year")
        self._build_years_menu(self.btn_find_years)
        d_hbox.addWidget(self.btn_find_years)

        grp_dates.content_layout.addLayout(d_hbox)
        hbox.addWidget(grp_dates)

        # ---------------------------------------------------------------------
        # 2. People Group (Face Carousel + [+ and / / or])
        # ---------------------------------------------------------------------
        grp_people = _RibbonGroupBox("People", page)
        p_hbox = QHBoxLayout()
        p_hbox.setContentsMargins(0, 0, 0, 0)
        p_hbox.setSpacing(3)

        self.find_face_carousel = _FaceCarouselWidget(grp_people)
        self.find_face_carousel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.find_face_carousel.person_clicked.connect(
            self.person_quick_find_selected.emit
        )
        self.find_face_carousel.boolean_mode_selected.connect(
            self._on_carousel_boolean_mode
        )
        self.find_face_carousel.special_filter_selected.connect(
            self.people_special_filter_selected.emit
        )
        p_hbox.addWidget(self.find_face_carousel)

        p_mode_col = QVBoxLayout()
        p_mode_col.setContentsMargins(0, 0, 0, 0)
        p_mode_col.setSpacing(1)
        self.btn_people_and = self._make_mode_toggle_btn("and", grp_people, height=22, icon_kind="boolean_and")
        self.btn_people_and.setToolTip("Match all selected people (AND)")
        self.btn_people_or = self._make_mode_toggle_btn("or", grp_people, height=22, icon_kind="boolean_or")
        self.btn_people_or.setToolTip("Match any selected person (OR)")
        self.btn_people_or.setChecked(True)

        self._people_btn_group = QButtonGroup(self)
        self._people_btn_group.setExclusive(True)
        self._people_btn_group.addButton(self.btn_people_and)
        self._people_btn_group.addButton(self.btn_people_or)
        self.btn_people_and.toggled.connect(
            lambda checked: self.person_boolean_mode_changed.emit("AND" if checked else "OR")
        )
        p_mode_col.addWidget(self.btn_people_and)
        p_mode_col.addWidget(self.btn_people_or)
        p_hbox.addLayout(p_mode_col)

        grp_people.content_layout.addLayout(p_hbox)
        hbox.addWidget(grp_people)

        # ---------------------------------------------------------------------
        # 3. Rated Group (Rated tool button + [↑ and higher / ↓ and lower / = only])
        # ---------------------------------------------------------------------
        grp_rated = _RibbonGroupBox("Rated", page)
        r_hbox = QHBoxLayout()
        r_hbox.setContentsMargins(0, 0, 0, 0)
        r_hbox.setSpacing(3)

        self.btn_find_rated = self._make_tool_btn("Rated", "rate", grp_rated, large=True)
        self.btn_find_rated.setToolTip("Filter by star rating")
        self._build_rated_menu(self.btn_find_rated)
        r_hbox.addWidget(self.btn_find_rated)

        r_mode_col = QVBoxLayout()
        r_mode_col.setContentsMargins(0, 0, 0, 0)
        r_mode_col.setSpacing(1)

        self.btn_rated_higher = self._make_mode_toggle_btn("and higher", grp_rated, height=19, icon_kind="rated_higher")
        self.btn_rated_higher.setToolTip("Rated this or higher")
        self.btn_rated_lower = self._make_mode_toggle_btn("and lower", grp_rated, height=19, icon_kind="rated_lower")
        self.btn_rated_lower.setToolTip("Rated this or lower")
        self.btn_rated_only = self._make_mode_toggle_btn("only", grp_rated, height=19, icon_kind="rated_only")
        self.btn_rated_only.setToolTip("Rated this only")
        self.btn_rated_higher.setChecked(True)

        self._rated_btn_group = QButtonGroup(self)
        self._rated_btn_group.setExclusive(True)
        self._rated_btn_group.addButton(self.btn_rated_higher)
        self._rated_btn_group.addButton(self.btn_rated_lower)
        self._rated_btn_group.addButton(self.btn_rated_only)

        self.btn_rated_higher.toggled.connect(self._on_rating_mode_changed)
        self.btn_rated_lower.toggled.connect(self._on_rating_mode_changed)
        self.btn_rated_only.toggled.connect(self._on_rating_mode_changed)

        r_mode_col.addWidget(self.btn_rated_higher)
        r_mode_col.addWidget(self.btn_rated_lower)
        r_mode_col.addWidget(self.btn_rated_only)
        r_hbox.addLayout(r_mode_col)

        grp_rated.content_layout.addLayout(r_hbox)
        hbox.addWidget(grp_rated)

        # ---------------------------------------------------------------------
        # 4. File details Group (Tags, Flagged, Edited, Media type)
        # ---------------------------------------------------------------------
        grp_file = _RibbonGroupBox("File details", page)
        f_hbox = QHBoxLayout()
        f_hbox.setContentsMargins(0, 0, 0, 0)
        f_hbox.setSpacing(3)

        self.btn_find_tags = self._make_tool_btn("Tags", "tag", grp_file, large=True)
        self.btn_find_tags.setToolTip("Filter by descriptive tags")
        self.menu_find_tags = WLPGTagsMenu(self)
        self.btn_find_tags.setMenu(self.menu_find_tags)
        f_hbox.addWidget(self.btn_find_tags)

        self.btn_find_flagged = self._make_btn("Flagged", "flag", grp_file, checkable=True, large=True)
        self.btn_find_flagged.setToolTip("Filter flagged photos only")
        self.btn_find_flagged.toggled.connect(self.flagged_filter_selected.emit)
        f_hbox.addWidget(self.btn_find_flagged)

        f_col = QVBoxLayout()
        f_col.setContentsMargins(0, 0, 0, 0)
        f_col.setSpacing(2)

        self.btn_find_edited = self._make_btn("Edited", "edited", grp_file, checkable=True)
        self.btn_find_edited.setFixedHeight(22)
        self.btn_find_edited.toggled.connect(self.edited_filter_selected.emit)
        f_col.addWidget(self.btn_find_edited)

        self.btn_find_media_type = self._make_tool_btn("Media type", "media_type", grp_file, large=False)
        self.btn_find_media_type.setToolTip("Filter by photos or videos")
        self.btn_find_media_type.setFixedHeight(22)
        self._build_media_type_menu(self.btn_find_media_type)
        f_col.addWidget(self.btn_find_media_type)

        f_hbox.addLayout(f_col)
        grp_file.content_layout.addLayout(f_hbox)
        hbox.addWidget(grp_file)

        self.find_file_edited_media = [self.btn_find_edited, self.btn_find_media_type]

        # ---------------------------------------------------------------------
        # 5. Text search Group (Binoculars button, Ctrl+F)
        # ---------------------------------------------------------------------
        grp_search = _RibbonGroupBox("Text search", page)
        self.btn_find_text_search = self._make_btn("Text search", "text_search", grp_search, large=True)
        self.btn_find_text_search.setToolTip("Find text (Ctrl+F)")
        self.btn_find_text_search.clicked.connect(self.text_search_toggle_requested.emit)
        grp_search.content_layout.addWidget(self.btn_find_text_search)
        hbox.addWidget(grp_search)

        # Authentic stretch pushes all groups compactly to the left
        hbox.addStretch(1)

        return page

    # -------------------------------------------------------------------------
    # Find Tab Menu Builders & Signal Dispatchers
    # -------------------------------------------------------------------------
    def _build_dates_menu(self, btn: QToolButton) -> None:
        menu = QMenu(btn)
        dates_items = [
            ("Today", "today"),
            ("Yesterday", "yesterday"),
            ("This week", "this_week"),
            ("Last week", "last_week"),
            ("This month", "this_month"),
            ("Last month", "last_month"),
            ("This year", "this_year"),
            ("Last year", "last_year"),
        ]
        for label, code in dates_items:
            act = menu.addAction(label)
            act.triggered.connect(lambda _c=False, cd=code: self._on_date_code_selected(cd))
        menu.addSeparator()
        act_clear = menu.addAction("Clear date filter")
        act_clear.triggered.connect(lambda: self.date_filter_selected.emit(None, None, None))
        btn.setMenu(menu)

    def _on_date_code_selected(self, code: str) -> None:
        now = datetime.datetime.now()
        if code == "today":
            self.date_filter_selected.emit(now.year, now.month, now.day)
        elif code == "yesterday":
            y = now - datetime.timedelta(days=1)
            self.date_filter_selected.emit(y.year, y.month, y.day)
        elif code == "this_week":
            start = now - datetime.timedelta(days=now.weekday())
            self.date_filter_selected.emit(start.year, start.month, start.day)
        elif code == "last_week":
            start = now - datetime.timedelta(days=now.weekday() + 7)
            self.date_filter_selected.emit(start.year, start.month, start.day)
        elif code == "this_month":
            self.date_filter_selected.emit(now.year, now.month, None)
        elif code == "last_month":
            month = now.month - 1 if now.month > 1 else 12
            year = now.year if now.month > 1 else now.year - 1
            self.date_filter_selected.emit(year, month, None)
        elif code == "this_year":
            self.date_filter_selected.emit(now.year, None, None)
        elif code == "last_year":
            self.date_filter_selected.emit(now.year - 1, None, None)

    def _build_months_menu(self, btn: QToolButton) -> None:
        menu = QMenu(btn)
        for m in range(1, 13):
            month_name = calendar.month_name[m]
            act = menu.addAction(month_name)
            act.triggered.connect(lambda _c=False, mn=m: self.date_filter_selected.emit(None, mn, None))
        menu.addSeparator()
        act_clear = menu.addAction("Clear month filter")
        act_clear.triggered.connect(lambda: self.date_filter_selected.emit(None, None, None))
        btn.setMenu(menu)

    def _build_years_menu(self, btn: QToolButton) -> None:
        menu = QMenu(btn)
        current_year = datetime.datetime.now().year
        for yr in range(current_year, current_year - 12, -1):
            act = menu.addAction(str(yr))
            act.triggered.connect(lambda _c=False, y=yr: self.date_filter_selected.emit(y, None, None))
        menu.addSeparator()
        act_clear = menu.addAction("Clear year filter")
        act_clear.triggered.connect(lambda: self.date_filter_selected.emit(None, None, None))
        btn.setMenu(menu)

    def _build_rated_menu(self, btn: QToolButton) -> None:
        menu = QMenu(btn)
        star_items = [
            ("★★★★★ 5 stars", 5),
            ("★★★★☆ 4 stars", 4),
            ("★★★☆☆ 3 stars", 3),
            ("★★☆☆☆ 2 stars", 2),
            ("★☆☆☆☆ 1 star", 1),
            ("⊘ Unrated", 0),
        ]
        for label, val in star_items:
            act = menu.addAction(label)
            act.triggered.connect(lambda _c=False, v=val: self._on_rating_item_clicked(v))
        menu.addSeparator()
        act_clear = menu.addAction("Clear rating filter")
        act_clear.triggered.connect(lambda: self._on_rating_item_clicked(-1))
        btn.setMenu(menu)

    def _on_rating_item_clicked(self, val: int) -> None:
        self._last_rated_val = val
        self._emit_current_rating_filter()

    def _on_rating_mode_changed(self, _checked: bool = True) -> None:
        if self._last_rated_val >= 0:
            self._emit_current_rating_filter()

    def _emit_current_rating_filter(self) -> None:
        val = self._last_rated_val
        if val == -1:
            self.rating_filter_selected.emit(())
            return
        if val == 0:
            self.rating_filter_selected.emit((0,))
            return
        if self.btn_rated_higher.isChecked():
            ratings = tuple(range(val, 6))
        elif self.btn_rated_lower.isChecked():
            ratings = tuple(range(1, val + 1))
        else:
            ratings = (val,)
        self.rating_filter_selected.emit(ratings)

    def _build_media_type_menu(self, btn: QToolButton) -> None:
        menu = QMenu(btn)
        act_all = menu.addAction("All media")
        act_photos = menu.addAction("Photos only")
        act_videos = menu.addAction("Videos only")
        act_raw = menu.addAction("RAW photos only")
        act_all.triggered.connect(lambda: self.media_type_filter_selected.emit("all"))
        act_photos.triggered.connect(lambda: self.media_type_filter_selected.emit("image"))
        act_videos.triggered.connect(lambda: self.media_type_filter_selected.emit("video"))
        act_raw.triggered.connect(lambda: self.media_type_filter_selected.emit("raw"))
        btn.setMenu(menu)

    def populate_tags_menu(self, tag_tree: Sequence[dict[str, Any]]) -> None:
        """Populates the Find tab's authentic WLPG 'Tags' popup menu."""
        if hasattr(self, "menu_find_tags") and isinstance(self.menu_find_tags, WLPGTagsMenu):
            self.menu_find_tags.populate(tag_tree)
        elif hasattr(self, "menu_find_tags"):
            self.menu_find_tags.clear()
            for t in tag_tree:
                tid = int(t["id"])
                name = str(t.get("name", t.get("full_path", "Tag")))
                act = self.menu_find_tags.addAction(name)
                act.triggered.connect(
                    lambda _c=False, tag_id=tid: self.tag_filter_selected.emit(tag_id, "include")
                )

    def is_people_and_mode(self) -> bool:
        return bool(hasattr(self, "btn_people_and") and self.btn_people_and.isChecked())

    def _on_carousel_boolean_mode(self, mode: str) -> None:
        if hasattr(self, "btn_people_and") and hasattr(self, "btn_people_or"):
            if mode == "AND":
                self.btn_people_and.setChecked(True)
            else:
                self.btn_people_or.setChecked(True)
        self.person_boolean_mode_changed.emit(mode)

    def _build_create_tab(self) -> QWidget:
        page = _RibbonPage()
        hbox = QHBoxLayout(page)
        hbox.setContentsMargins(4, 2, 4, 2)
        hbox.setSpacing(4)

        grp_share = _RibbonGroupBox("Share", page)
        btn_slide = self._make_btn("Slide show (F11)", "slideshow", grp_share, large=True)
        btn_slide.clicked.connect(lambda: self.action_triggered.emit("slide_show"))
        grp_share.content_layout.addWidget(btn_slide)
        hbox.addWidget(grp_share)

        grp_opt = _RibbonGroupBox("Settings", page)
        btn_opt = self._make_btn("Options...", "options", grp_opt, large=True)
        btn_opt.clicked.connect(lambda: self.action_triggered.emit("open_settings"))
        grp_opt.content_layout.addWidget(btn_opt)
        hbox.addWidget(grp_opt)

        hbox.addStretch(1)
        return page

    def _build_view_tab(self) -> QWidget:
        """'View' tab matching WLPG reference `docs/reference_ui/view_tab.png` and single image view."""
        page = _RibbonPage()
        hbox = QHBoxLayout(page)
        hbox.setContentsMargins(4, 2, 4, 2)
        hbox.setSpacing(4)

        # ---------------------------------------------------------------------
        # Group 1: Arrange list (4x2 button matrix + Rating Gallery Box + Reverse sort)
        # ---------------------------------------------------------------------
        grp_arrange = _RibbonGroupBox("Arrange list", page)
        self.grp_arrange = grp_arrange
        arr_hbox = QHBoxLayout()
        arr_hbox.setContentsMargins(0, 0, 0, 0)
        arr_hbox.setSpacing(4)

        matrix_grid = QGridLayout()
        matrix_grid.setContentsMargins(0, 0, 0, 0)
        matrix_grid.setHorizontalSpacing(2)
        matrix_grid.setVerticalSpacing(2)

        self._arrange_buttons: dict[ArrangeMode, QPushButton | QToolButton] = {}
        self._arrange_btn_group = QButtonGroup(self)
        self._arrange_btn_group.setExclusive(True)

        # 5 columns x 2 rows matching WLPG
        # Col 0: Date, Geotag
        # Col 1: Person, Name
        # Col 2: Tag, Published
        # Col 3: Folder, Media type
        # Col 4: Rating
        matrix_items = [
            (0, 0, "Date", "calendar", "date"),
            (1, 0, "Geotag", "geotag", "geotag"),
            (0, 1, "Person", "person", "person"),
            (1, 1, "Name", "name", "name"),
            (0, 2, "Tag", "tag", "tag"),
            (1, 2, "Published", "published", "published"),
            (0, 3, "Folder", "folder", "folder"),
            (1, 3, "Media type", "media_type", "media_type"),
            (0, 4, "Rating", "star", "rating"),
        ]
        for row, col, label, icon_kind, mode in matrix_items:
            btn = self._make_btn(label, icon_kind, grp_arrange, checkable=True)
            btn.setFixedHeight(22)
            btn.setMinimumWidth(0)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.clicked.connect(lambda _c=False, m=mode: self._on_arrange_button_clicked(m))
            matrix_grid.addWidget(btn, row, col)
            self._arrange_buttons[mode] = btn
            self._arrange_btn_group.addButton(btn)

        self.btn_arr_rating = self._arrange_buttons["rating"]
        arr_hbox.addLayout(matrix_grid)

        # Mini vertical scroll buttons matching Aero Ribbon gallery
        v_arrows = QVBoxLayout()
        v_arrows.setContentsMargins(0, 0, 0, 0)
        v_arrows.setSpacing(1)
        btn_up = QPushButton("▲", grp_arrange)
        btn_up.setFixedSize(13, 14)
        btn_up.setStyleSheet("QPushButton { border: 1px solid #C4D5E8; border-radius: 1px; background: #F7FAFD; font-size: 5pt; color: #5A7A9E; padding: 0px; } QPushButton:hover { background: #E4F0FC; }")
        btn_dn = QPushButton("▼", grp_arrange)
        btn_dn.setFixedSize(13, 14)
        btn_dn.setStyleSheet("QPushButton { border: 1px solid #C4D5E8; border-radius: 1px; background: #F7FAFD; font-size: 5pt; color: #5A7A9E; padding: 0px; } QPushButton:hover { background: #E4F0FC; }")
        btn_more = QPushButton("▾", grp_arrange)
        btn_more.setFixedSize(13, 14)
        btn_more.setStyleSheet("QPushButton { border: 1px solid #C4D5E8; border-radius: 1px; background: #F7FAFD; font-size: 5pt; color: #5A7A9E; padding: 0px; } QPushButton:hover { background: #E4F0FC; }")
        v_arrows.addWidget(btn_up)
        v_arrows.addWidget(btn_dn)
        v_arrows.addWidget(btn_more)
        arr_hbox.addLayout(v_arrows)

        # Large Reverse sort button (with dual blue vertical arrows)
        self.btn_reverse_sort = self._make_btn("Reverse\nsort", "reverse_sort", grp_arrange, large=True, checkable=True)
        self.btn_reverse_sort.setFixedWidth(52)
        self.btn_reverse_sort.clicked.connect(self._on_reverse_sort_clicked)
        arr_hbox.addWidget(self.btn_reverse_sort)
        self.chk_reverse_sort = self.btn_reverse_sort  # Legacy alias

        # Hidden QComboBox for backward test compatibility
        self.cmb_arrange = QComboBox(page)
        self.cmb_arrange.hide()
        for _, _, label, _, mode in matrix_items:
            self.cmb_arrange.addItem(label, mode)
        self.cmb_arrange.addItem("Rating", "rating")

        # Set default active
        self._arrange_buttons["date"].setChecked(True)

        grp_arrange.content_layout.addLayout(arr_hbox)
        hbox.addWidget(grp_arrange)

        # ---------------------------------------------------------------------
        # Group 2: Show details (Large Rating, Large Caption, 2x3 checkboxes, Large Flags)
        # ---------------------------------------------------------------------
        grp_details = _RibbonGroupBox("Show details", page)
        self.grp_details = grp_details
        det_hbox = QHBoxLayout()
        det_hbox.setContentsMargins(0, 0, 0, 0)
        det_hbox.setSpacing(4)

        self.btn_detail_rating = self._make_btn("Rating", "star", grp_details, large=True, checkable=True)
        self.btn_detail_rating.setChecked("rating" in self._visible_details)
        det_hbox.addWidget(self.btn_detail_rating)

        self.btn_detail_caption = self._make_btn("Caption", "caption", grp_details, large=True, checkable=True)
        self.btn_detail_caption.setChecked("caption" in self._visible_details)
        det_hbox.addWidget(self.btn_detail_caption)

        chk_col1 = QVBoxLayout()
        chk_col1.setContentsMargins(0, 0, 0, 0)
        chk_col1.setSpacing(1)

        self.chk_all_details = self._make_btn("All details", "detail_all", grp_details, large=False, checkable=True)
        self.chk_filename = self._make_btn("File name", "detail_filename", grp_details, large=False, checkable=True)
        self.chk_filesize = self._make_btn("File size", "detail_filesize", grp_details, large=False, checkable=True)

        chk_col1.addWidget(self.chk_all_details)
        chk_col1.addWidget(self.chk_filename)
        chk_col1.addWidget(self.chk_filesize)

        chk_col2 = QVBoxLayout()
        chk_col2.setContentsMargins(0, 0, 0, 0)
        chk_col2.setSpacing(1)

        self.chk_date_taken = self._make_btn("Date taken", "detail_date_taken", grp_details, large=False, checkable=True)
        self.chk_dimensions = self._make_btn("Image size", "detail_dimensions", grp_details, large=False, checkable=True)
        self.chk_date_modified = self._make_btn("Date modified", "detail_date_modified", grp_details, large=False, checkable=True)

        chk_col2.addWidget(self.chk_date_taken)
        chk_col2.addWidget(self.chk_dimensions)
        chk_col2.addWidget(self.chk_date_modified)

        det_hbox.addLayout(chk_col1)
        det_hbox.addLayout(chk_col2)

        self.btn_detail_flags = self._make_btn("Flags", "flag", grp_details, large=True, checkable=True)
        self.btn_detail_flags.setChecked("flag" in self._visible_details)
        det_hbox.addWidget(self.btn_detail_flags)

        self._detail_checkboxes = {
            "rating": self.btn_detail_rating,
            "caption": self.btn_detail_caption,
            "flag": self.btn_detail_flags,
            "filename": self.chk_filename,
            "size": self.chk_filesize,
            "date_taken": self.chk_date_taken,
            "dimensions": self.chk_dimensions,
            "date_modified": self.chk_date_modified,
        }

        for chk in self._detail_checkboxes.values():
            chk.toggled.connect(self._on_detail_checkbox_toggled)

        self.chk_all_details.toggled.connect(self._on_all_details_toggled)

        self.view_details_checkboxes = [
            self.chk_all_details,
            self.chk_filename,
            self.chk_filesize,
            self.chk_date_taken,
            self.chk_dimensions,
            self.chk_date_modified,
        ]
        self.view_arrange_buttons = list(self._arrange_buttons.values())

        grp_details.content_layout.addLayout(det_hbox)
        hbox.addWidget(grp_details)

        # ---------------------------------------------------------------------
        # Group 3: Zoom Group (two large buttons matching WLPG)
        # ---------------------------------------------------------------------
        grp_zoom = _RibbonGroupBox("Zoom", page)
        self.grp_zoom = grp_zoom
        z_hbox = QHBoxLayout()
        z_hbox.setContentsMargins(0, 0, 0, 0)
        z_hbox.setSpacing(3)

        self.btn_zoom_in = self._make_btn("Zoom\nin", "zoom_in", grp_zoom, large=True)
        self.btn_zoom_in.clicked.connect(lambda: self.action_triggered.emit("zoom_in"))
        z_hbox.addWidget(self.btn_zoom_in)

        self.btn_zoom_out = self._make_btn("Zoom\nout", "zoom_out", grp_zoom, large=True)
        self.btn_zoom_out.clicked.connect(lambda: self.action_triggered.emit("zoom_out"))
        z_hbox.addWidget(self.btn_zoom_out)

        self.btn_zoom_actual = self._make_btn("Actual\nsize", "actual_size", grp_zoom, large=True)
        self.btn_zoom_actual.clicked.connect(lambda: self.action_triggered.emit("viewer_zoom_actual"))
        self.btn_zoom_actual.setVisible(False)
        z_hbox.addWidget(self.btn_zoom_actual)

        self.btn_zoom_fit = self._make_btn("Fit to\nwindow", "fit_window", grp_zoom, large=True)
        self.btn_zoom_fit.clicked.connect(lambda: self.action_triggered.emit("viewer_zoom_fit"))
        self.btn_zoom_fit.setVisible(False)
        z_hbox.addWidget(self.btn_zoom_fit)

        # Thumbnail zoom slider kept for API/status bar compatibility, hidden from ribbon to match WLPG
        self.slider_zoom = QSlider(Qt.Horizontal, grp_zoom)
        self.slider_zoom.setRange(72, 320)
        self.slider_zoom.setValue(150)
        self.slider_zoom.setFixedWidth(80)
        self.slider_zoom.valueChanged.connect(self.zoom_size_changed.emit)
        self.slider_zoom.setVisible(False)

        grp_zoom.content_layout.addLayout(z_hbox)
        hbox.addWidget(grp_zoom)

        # ---------------------------------------------------------------------
        # Group 4: Views Group (Slide show & World Map)
        # ---------------------------------------------------------------------
        grp_slideshow = _RibbonGroupBox("Views", page)
        self.grp_slideshow = grp_slideshow
        self.btn_view_slideshow = self._make_btn("Slide\nshow ▾", "slideshow", grp_slideshow, large=True)
        self.btn_view_slideshow.clicked.connect(lambda: self.action_triggered.emit("slide_show"))
        grp_slideshow.content_layout.addWidget(self.btn_view_slideshow)

        self.btn_view_map = self._make_btn("World\nMap", "globe", grp_slideshow, large=True, checkable=True)
        self.btn_view_map.setToolTip("Explore geotagged photos on the World Map (Leaflet / Offline)")
        self.btn_view_map.clicked.connect(lambda: self.action_triggered.emit("toggle_map_view"))
        grp_slideshow.content_layout.addWidget(self.btn_view_map)
        hbox.addWidget(grp_slideshow)

        # ---------------------------------------------------------------------
        # Group 5: Panes Group (Tag & Caption pane matching WLPG)
        # ---------------------------------------------------------------------
        grp_panes = _RibbonGroupBox("Panes", page)
        self.grp_panes = grp_panes
        p_hbox = QHBoxLayout()
        p_hbox.setContentsMargins(0, 0, 0, 0)
        p_hbox.setSpacing(3)

        self.btn_toggle_nav_pane = self._make_btn(
            "Navigation\npane", "pane_left", grp_panes, large=True, checkable=True
        )
        self.btn_toggle_nav_pane.setChecked(True)
        self.btn_toggle_nav_pane.clicked.connect(
            lambda: self.action_triggered.emit("toggle_nav_pane")
        )
        p_hbox.addWidget(self.btn_toggle_nav_pane)

        self.btn_toggle_tag_pane = self._make_btn(
            "Tag and\ncaption pane", "pane_right", grp_panes, large=True, checkable=True
        )
        self.btn_toggle_tag_pane.clicked.connect(
            lambda: self.action_triggered.emit("toggle_tag_pane")
        )
        p_hbox.addWidget(self.btn_toggle_tag_pane)

        grp_panes.content_layout.addLayout(p_hbox)
        hbox.addWidget(grp_panes)

        # ---------------------------------------------------------------------
        # Group 6: Photo layout (Grid mode only - keeping custom addition)
        # ---------------------------------------------------------------------
        grp_layout = _RibbonGroupBox("Photo layout", page)
        self.grp_layout = grp_layout
        self.btn_layout_grid = self._make_btn(
            "WLPG Grid", "layout_grid", grp_layout, large=True, checkable=True
        )
        self.btn_layout_grid.setChecked(True)
        self.btn_layout_mosaic = self._make_btn(
            "Mosaic", "layout_mosaic", grp_layout, large=True, checkable=True
        )

        self._layout_btn_group = QButtonGroup(self)
        self._layout_btn_group.setExclusive(True)
        self._layout_btn_group.addButton(self.btn_layout_grid)
        self._layout_btn_group.addButton(self.btn_layout_mosaic)

        self.btn_layout_grid.clicked.connect(
            lambda: self.layout_mode_changed.emit("wlpg_grid")
        )
        self.btn_layout_mosaic.clicked.connect(
            lambda: self.layout_mode_changed.emit("mosaic")
        )
        l_hbox = QHBoxLayout()
        l_hbox.setContentsMargins(0, 0, 0, 0)
        l_hbox.setSpacing(3)
        l_hbox.addWidget(self.btn_layout_grid)
        l_hbox.addWidget(self.btn_layout_mosaic)
        grp_layout.content_layout.addLayout(l_hbox)
        hbox.addWidget(grp_layout)

        hbox.addStretch(1)
        return page

    # -------------------------------------------------------------------------
    # Public Control & Inspection API
    # -------------------------------------------------------------------------
    def populate_face_carousel(self, people: Sequence[dict[str, Any]]) -> None:
        self.home_face_carousel.populate(people)
        self.find_face_carousel.populate(people)

    def set_arrange_by(
        self, mode: ArrangeMode, reverse_sort: bool | None = None
    ) -> None:
        self._arrange_by = mode
        if reverse_sort is not None:
            self._reverse_sort = bool(reverse_sort)

        if hasattr(self, "_arrange_buttons") and mode in self._arrange_buttons:
            btn = self._arrange_buttons[mode]
            btn.blockSignals(True)
            btn.setChecked(True)
            btn.blockSignals(False)

        if hasattr(self, "cmb_arrange"):
            self.cmb_arrange.blockSignals(True)
            for i in range(self.cmb_arrange.count()):
                if self.cmb_arrange.itemData(i) == mode:
                    self.cmb_arrange.setCurrentIndex(i)
                    break
            self.cmb_arrange.blockSignals(False)

        if hasattr(self, "btn_reverse_sort"):
            self.btn_reverse_sort.blockSignals(True)
            self.btn_reverse_sort.setChecked(self._reverse_sort)
            self.btn_reverse_sort.blockSignals(False)

        self.arrange_changed.emit(self._arrange_by, self._reverse_sort)

    def current_arrange_state(self) -> tuple[ArrangeMode, bool]:
        return (self._arrange_by, self._reverse_sort)

    def _on_arrange_button_clicked(self, mode: ArrangeMode) -> None:
        self._arrange_by = mode
        if hasattr(self, "cmb_arrange"):
            self.cmb_arrange.blockSignals(True)
            for i in range(self.cmb_arrange.count()):
                if self.cmb_arrange.itemData(i) == mode:
                    self.cmb_arrange.setCurrentIndex(i)
                    break
            self.cmb_arrange.blockSignals(False)
        self.arrange_changed.emit(self._arrange_by, self._reverse_sort)

    def _on_reverse_sort_clicked(self) -> None:
        self._reverse_sort = self.btn_reverse_sort.isChecked()
        self.arrange_changed.emit(self._arrange_by, self._reverse_sort)

    def _on_arrange_ui_changed(self) -> None:
        mode = self.cmb_arrange.currentData()
        self._arrange_by = mode if mode else "date"
        self._reverse_sort = self.chk_reverse_sort.isChecked()
        self.arrange_changed.emit(self._arrange_by, self._reverse_sort)

    def set_detail_flag(self, key: str, enabled: bool) -> None:
        if key in self._detail_checkboxes:
            chk = self._detail_checkboxes[key]
            chk.blockSignals(True)
            chk.setChecked(bool(enabled))
            chk.blockSignals(False)
            self._on_detail_checkbox_toggled()

    def current_visible_details(self) -> set[str]:
        return set(self._visible_details)

    def _on_all_details_toggled(self, checked: bool) -> None:
        sub_chks = (
            self.chk_filename,
            self.chk_filesize,
            self.chk_date_taken,
            self.chk_dimensions,
            self.chk_date_modified,
        )
        for chk in sub_chks:
            chk.blockSignals(True)
            chk.setChecked(checked)
            chk.blockSignals(False)
        self._on_detail_checkbox_toggled()

    def _on_detail_checkbox_toggled(self) -> None:
        active = {
            k for k, chk in self._detail_checkboxes.items() if chk.isChecked()
        }
        self._visible_details = active
        sub_chks = (
            self.chk_filename,
            self.chk_filesize,
            self.chk_date_taken,
            self.chk_dimensions,
            self.chk_date_modified,
        )
        all_on = all(c.isChecked() for c in sub_chks)
        if hasattr(self, "chk_all_details"):
            self.chk_all_details.blockSignals(True)
            self.chk_all_details.setChecked(all_on)
            self.chk_all_details.blockSignals(False)
        self.details_overlay_changed.emit(set(active))

    def set_layout_mode(self, mode: Literal["wlpg_grid", "mosaic"]) -> None:
        if mode == "mosaic":
            self.btn_layout_mosaic.setChecked(True)
        else:
            self.btn_layout_grid.setChecked(True)
        self.layout_mode_changed.emit(mode)

    def set_scope_label(self, text: str) -> None:
        if hasattr(self, "lbl_scope") and self.lbl_scope is not None:
            try:
                self.lbl_scope.setText(text)
            except RuntimeError:
                pass
        if hasattr(self, "scope_bar") and hasattr(self.scope_bar, "set_scope_label"):
            try:
                self.scope_bar.set_scope_label(text)
            except RuntimeError:
                pass

    def set_search_text(self, text: str, immediate: bool = False) -> None:
        if hasattr(self, "txt_search") and self.txt_search is not None:
            try:
                self.txt_search.blockSignals(True)
                self.txt_search.setText(text)
                self.txt_search.blockSignals(False)
            except RuntimeError:
                pass
        if immediate:
            self._search_timer.stop()
            self.search_text_changed.emit(text.strip())
        else:
            self._search_timer.start()

    def _on_search_input_changed(self, _text: str) -> None:
        self._search_timer.start()

    def _emit_search_text(self) -> None:
        if hasattr(self, "txt_search") and self.txt_search is not None:
            try:
                self.search_text_changed.emit(self.txt_search.text().strip())
            except RuntimeError:
                pass

    def set_active_mode(self, mode: Literal["grid", "viewer", "map"]) -> None:
        """Switches Ribbon context when entering/exiting Single Media Viewer or World Map."""
        if mode == "viewer":
            # In Single Image View, hide Home (0) and Find (2), showing File, Edit, Create, View
            self.tab_widget.tabBar().setTabVisible(0, False)
            self.tab_widget.tabBar().setTabVisible(2, False)
            self._set_edit_tab_mode("viewer")
            self.tab_widget.setCurrentWidget(self.tab_edit)
            if hasattr(self, "scope_bar") and self.scope_bar is not None:
                self.scope_bar.setVisible(False)
            if hasattr(self, "grp_arrange"):
                self.grp_arrange.setVisible(False)
            if hasattr(self, "grp_details"):
                self.grp_details.setVisible(False)
            if hasattr(self, "grp_layout"):
                self.grp_layout.setVisible(False)
            if hasattr(self, "btn_toggle_nav_pane"):
                self.btn_toggle_nav_pane.setVisible(False)
            if hasattr(self, "slider_zoom"):
                self.slider_zoom.setVisible(False)
            if hasattr(self, "btn_zoom_actual"):
                self.btn_zoom_actual.setVisible(True)
            if hasattr(self, "btn_zoom_fit"):
                self.btn_zoom_fit.setVisible(True)
        elif mode == "map":
            self.tab_widget.tabBar().setTabVisible(0, True)
            self.tab_widget.tabBar().setTabVisible(2, True)
            self.tab_widget.setCurrentWidget(self.tab_view)
            if hasattr(self, "btn_view_map"):
                self.btn_view_map.setChecked(True)
            if hasattr(self, "scope_bar") and self.scope_bar is not None:
                self.scope_bar.setVisible(False)
        else:
            # Grid mode: restore Home and Find tabs
            self.tab_widget.tabBar().setTabVisible(0, True)
            self.tab_widget.tabBar().setTabVisible(2, True)
            if hasattr(self, "btn_view_map"):
                self.btn_view_map.setChecked(False)
            self._set_edit_tab_mode("grid")
            if self.tab_widget.currentWidget() == self.tab_edit:
                self.tab_widget.setCurrentWidget(self.tab_home)
            if hasattr(self, "scope_bar") and self.scope_bar is not None:
                self.scope_bar.setVisible(True)
            if hasattr(self, "grp_arrange"):
                self.grp_arrange.setVisible(True)
            if hasattr(self, "grp_details"):
                self.grp_details.setVisible(True)
            if hasattr(self, "grp_layout"):
                self.grp_layout.setVisible(True)
            if hasattr(self, "btn_toggle_nav_pane"):
                self.btn_toggle_nav_pane.setVisible(True)
            if hasattr(self, "slider_zoom"):
                self.slider_zoom.setVisible(False)
            if hasattr(self, "btn_zoom_actual"):
                self.btn_zoom_actual.setVisible(False)
            if hasattr(self, "btn_zoom_fit"):
                self.btn_zoom_fit.setVisible(False)


    def _apply_aero_stylesheet(self) -> None:
        icon_check = (Path(__file__).resolve().parents[2] / "assets" / "icons" / "check_dark.svg").as_posix()
        arrow_drop_down = (Path(__file__).resolve().parents[2] / "assets" / "icons" / "arrow_dropdown_mini.svg").as_posix()
        self.setStyleSheet(
            """
            QTabWidget#RibbonTabs::pane {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #F4F8FD, stop:0.55 #EBF2FA, stop:1 #DDE8F5);
                border-top: 1px solid #9EB6CE;
                border-bottom: 1px solid #A8BDD4;
                top: -1px;
            }
            QTabWidget#RibbonTabs > QTabBar::tab {
                background: transparent;
                border: 1px solid transparent;
                border-bottom: none;
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
                padding: 3px 14px;
                margin-top: 2px;
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 9pt;
            }
            QTabWidget#RibbonTabs > QTabBar::tab:hover {
                background: rgba(255, 255, 255, 140);
                border-color: #B2C7DE;
            }
            QTabWidget#RibbonTabs > QTabBar::tab:selected {
                background: #F4F8FD;
                border: 1px solid #9EB6CE;
                border-bottom: 1px solid #F4F8FD;
                margin-bottom: -1px;
                font-weight: 600;
            }
            QPushButton#CornerOptionsBtn {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 3px;
                padding: 2px 8px;
                margin-right: 6px;
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
            }
            QPushButton#CornerOptionsBtn:hover {
                background: #E4F0FC;
                border-color: #9CB8D6;
            }
            QFrame#RibbonGroupBox {
                border-right: 1px solid #C5D5E6;
            }
            QPushButton#LargeRibbonBtn,
            QToolButton#LargeRibbonBtn,
            QToolButton#LargeRibbonToolBtn {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 3px;
                padding: 0px 1px 1px 1px;
                margin: 0px;
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 8pt;
                text-align: center;
            }
            QPushButton#CompactRibbonBtn,
            QToolButton#CompactRibbonBtn {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 2px;
                padding: 1px 4px;
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                text-align: left;
            }
            QPushButton#LargeRibbonBtn:hover,
            QToolButton#LargeRibbonBtn:hover,
            QPushButton#CompactRibbonBtn:hover,
            QToolButton#CompactRibbonBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FFFDF5, stop:1 #FCE7A2);
                border: 1px solid #D9B152;
            }
            QPushButton#LargeRibbonBtn:pressed,
            QToolButton#LargeRibbonBtn:pressed,
            QPushButton#CompactRibbonBtn:pressed,
            QToolButton#CompactRibbonBtn:pressed,
            QToolButton#LargeRibbonToolBtn:pressed,
            QToolButton#CompactRibbonToolBtn:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FCE293, stop:1 #F7CE65);
                border: 1px solid #C28B10;
            }
            QPushButton#LargeRibbonBtn:checked,
            QToolButton#LargeRibbonBtn:checked,
            QPushButton#CompactRibbonBtn:checked,
            QToolButton#CompactRibbonBtn:checked {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #EAF4FD, stop:1 #BBDDF7);
                border: 1px solid #7DA2CE;
            }
            QPushButton#LargeRibbonBtn:checked:hover,
            QToolButton#LargeRibbonBtn:checked:hover,
            QPushButton#CompactRibbonBtn:checked:hover,
            QToolButton#CompactRibbonBtn:checked:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #F0F7FE, stop:1 #A8D3F5);
                border: 1px solid #6BA0D6;
            }
            QCheckBox {
                spacing: 4px;
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 8pt;
            }
            QCheckBox::indicator {
                width: 12px;
                height: 12px;
                border: 1px solid #8E9CAE;
                border-radius: 1px;
                background: #FFFFFF;
            }
            QCheckBox::indicator:hover {
                border-color: #3399FF;
                background: #F0F7FE;
            }
            QCheckBox::indicator:checked {
                border-color: #2468A8;
                background: #E5F1FB;
                image: url({icon_check});
            }
            QToolButton#LargeRibbonToolBtn {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 3px;
                padding: 1px 2px;
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 8pt;
                text-align: center;
            }
            QToolButton#CompactRibbonToolBtn {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 2px;
                padding: 1px 4px;
                color: #1E395B;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                text-align: left;
            }
            QToolButton#LargeRibbonToolBtn:hover,
            QToolButton#CompactRibbonToolBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FFFDF5, stop:1 #FCE7A2);
                border: 1px solid #D9B152;
            }
            QToolButton::menu-indicator,
            QToolButton#CompactRibbonBtn::menu-indicator,
            QToolButton#CompactRibbonToolBtn::menu-indicator,
            QToolButton#LargeRibbonBtn::menu-indicator,
            QToolButton#LargeRibbonToolBtn::menu-indicator {
                image: url({arrow_drop_down});
                width: 5px;
                height: 3px;
                subcontrol-position: center right;
                subcontrol-origin: padding;
                right: 2px;
                top: 1px;
            }
            QToolButton::menu-button {
                border: none;
                background: transparent;
                width: 10px;
            }
            QToolButton::menu-arrow {
                image: url({arrow_drop_down});
                width: 5px;
                height: 3px;
            }
            QPushButton#RibbonModeToggleBtn {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 2px;
                padding: 1px 4px;
                color: #33557A;
                font-family: 'Segoe UI';
                font-size: 8pt;
                text-align: left;
            }
            QPushButton#RibbonModeToggleBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FFFDF5, stop:1 #FCE7A2);
                border: 1px solid #D9B152;
            }
            QPushButton#RibbonModeToggleBtn:checked {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #EAF3FD, stop:0.5 #D2E7FA, stop:1 #BBDDF7);
                border: 1px solid #7DA2CE;
                font-weight: 600;
                color: #1E395B;
            }
            QFrame#SearchScopeBar {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #F7FAFE, stop:1 #EAF1F9);
                border-bottom: 1px solid #C5D5E6;
            }
            QLineEdit#ScopeSearchInput {
                background: #FFFFFF;
                border: 1px solid #9CB4CC;
                border-radius: 2px;
                padding: 1px 6px;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
                color: #1E395B;
            }
            QLineEdit#ScopeSearchInput:focus {
                border: 1px solid #3380CC;
            }
            QWidget#RibbonTopHeader {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F4F8FD, stop:1 #E0EBF7);
                border-bottom: 1px solid #C4D6E8;
            }
            QWidget#RibbonBelowQAT {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F4F8FC, stop:1 #E4ECF5);
                border-bottom: 1px solid #C4D3E3;
            }
            QToolButton#QATCircleBtn {
                border: none;
                border-radius: 11px;
                background: transparent;
                padding: 0px;
            }
            QToolButton#QATCircleBtn:hover {
                background: rgba(255, 255, 255, 160);
            }
            QToolButton#QATCircleBtn:pressed {
                background: rgba(0, 100, 200, 50);
            }
            QToolButton#QATBtn {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 3px;
                padding: 1px;
            }
            QToolButton#QATBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #FDFEFE, stop:1 #E5F1FC);
                border: 1px solid #A8C8E8;
            }
            QToolButton#QATBtn:disabled {
                opacity: 0.35;
            }
            QPushButton#RibbonFileBtn {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2A79C6, stop:1 #175B9E);
                color: #FFFFFF;
                font-family: 'Segoe UI';
                font-size: 9pt;
                font-weight: bold;
                border: 1px solid #104C85;
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
                padding: 3px 14px;
                margin-left: 2px;
                margin-right: 4px;
            }
            QPushButton#RibbonFileBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3B8BD8, stop:1 #1C67B0);
            }
            QPushButton#RibbonFileBtn::menu-indicator {
                image: none;
                width: 0px;
            }
            QToolTip {
                background-color: #FFFFFF;
                color: #1E395B;
                border: 1px solid #7DA2CE;
                border-radius: 2px;
                padding: 3px 6px;
                font-family: 'Segoe UI';
                font-size: 8.5pt;
            }
            """.replace("{icon_check}", icon_check).replace("{arrow_drop_down}", arrow_drop_down)
        )
