"""
FullscreenSlideshowDialog (QDialog) — Authentic Fullscreen Slideshow for PhotoGallery64.
Matches Windows Live Photo Gallery's Slideshow mode:
- Pure black fullscreen canvas (#000000)
- High-quality smooth scaling (Qt.SmoothTransformation, preserving aspect ratio)
- Auto-advance slideshow timer (4 seconds per slide)
- Translucent floating HUD (Heads-Up Display) with Prev, Play/Pause, Next, and Close controls
- Auto-hiding HUD and mouse cursor on inactivity (2.5s)
- Comprehensive keyboard navigation (Space, Left, Right, Escape, F12, Backspace, etc.)
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps
from PySide6.QtCore import (
    QEvent,
    QObject,
    QPoint,
    QRect,
    QRunnable,
    QSize,
    Qt,
    QThreadPool,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QCursor,
    QFont,
    QIcon,
    QImage,
    QKeyEvent,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.db.repository import CatalogRepository, MediaItemRecord
from src.ui.ribbon_toolbar import make_wlpg_icon


class SlideshowHud(QFrame):
    """Semi-transparent floating control pill at bottom of screen."""

    prev_clicked = Signal()
    play_pause_clicked = Signal()
    next_clicked = Signal()
    close_clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("SlideshowHud")
        self.setFixedHeight(48)
        self.setStyleSheet(
            """
            QFrame#SlideshowHud {
                background-color: rgba(20, 24, 30, 210);
                border: 1px solid rgba(120, 150, 190, 160);
                border-radius: 24px;
            }
            QToolButton {
                border: 1px solid transparent;
                border-radius: 18px;
                background: transparent;
                color: #FFFFFF;
                font-family: 'Segoe UI';
                font-size: 11pt;
                padding: 0px;
            }
            QToolButton:hover {
                background: rgba(255, 255, 255, 60);
                border: 1px solid rgba(255, 255, 255, 120);
            }
            QToolButton:pressed {
                background: rgba(0, 120, 215, 180);
            }
            """
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 4, 16, 4)
        layout.setSpacing(12)

        # Previous button (◀)
        self.btn_prev = QToolButton(self)
        self.btn_prev.setIcon(make_wlpg_icon("prev_media", 20))
        self.btn_prev.setIconSize(QSize(20, 20))
        self.btn_prev.setFixedSize(36, 36)
        self.btn_prev.setToolTip("Previous photo (Left Arrow / Backspace)")
        self.btn_prev.setCursor(Qt.PointingHandCursor)
        self.btn_prev.clicked.connect(self.prev_clicked.emit)
        layout.addWidget(self.btn_prev)

        # Play / Pause button (⏸ / ▶)
        self.btn_play_pause = QToolButton(self)
        self.btn_play_pause.setIcon(make_wlpg_icon("pause", 22))
        self.btn_play_pause.setIconSize(QSize(22, 22))
        self.btn_play_pause.setFixedSize(36, 36)
        self.btn_play_pause.setToolTip("Pause / Resume (Space)")
        self.btn_play_pause.setCursor(Qt.PointingHandCursor)
        self.btn_play_pause.clicked.connect(self.play_pause_clicked.emit)
        layout.addWidget(self.btn_play_pause)

        # Next button (▶)
        self.btn_next = QToolButton(self)
        self.btn_next.setIcon(make_wlpg_icon("next_media", 20))
        self.btn_next.setIconSize(QSize(20, 20))
        self.btn_next.setFixedSize(36, 36)
        self.btn_next.setToolTip("Next photo (Right Arrow / Space)")
        self.btn_next.setCursor(Qt.PointingHandCursor)
        self.btn_next.clicked.connect(self.next_clicked.emit)
        layout.addWidget(self.btn_next)

        # Separator line
        sep = QFrame(self)
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color: rgba(255, 255, 255, 80); margin: 6px 2px;")
        layout.addWidget(sep)

        # Close / Exit button (✕)
        self.btn_close = QToolButton(self)
        self.btn_close.setIcon(make_wlpg_icon("close_file", 18))
        self.btn_close.setIconSize(QSize(18, 18))
        self.btn_close.setFixedSize(36, 36)
        self.btn_close.setToolTip("Exit slide show (Esc / F12)")
        self.btn_close.setCursor(Qt.PointingHandCursor)
        self.btn_close.clicked.connect(self.close_clicked.emit)
        layout.addWidget(self.btn_close)

        self.setFocusPolicy(Qt.NoFocus)
        for b in (self.btn_prev, self.btn_play_pause, self.btn_next, self.btn_close):
            b.setFocusPolicy(Qt.NoFocus)

    def set_playing(self, is_playing: bool) -> None:
        icon_name = "pause" if is_playing else "slideshow"
        self.btn_play_pause.setIcon(make_wlpg_icon(icon_name, 22))
        self.btn_play_pause.setToolTip("Pause (Space)" if is_playing else "Play (Space)")


class _SlidePreloadSignals(QObject):
    preloaded = Signal(int, QImage)


class _SlidePreloadWorker(QRunnable):
    def __init__(
        self,
        media_id: int,
        file_path: str,
        max_w: int,
        max_h: int,
        signals: _SlidePreloadSignals,
    ) -> None:
        super().__init__()
        self.media_id = media_id
        self.file_path = file_path
        self.max_w = max_w
        self.max_h = max_h
        self.signals = signals

    def run(self) -> None:
        try:
            p = Path(self.file_path)
            if not p.exists():
                return
            with Image.open(p) as img:
                img = ImageOps.exif_transpose(img)
                img.thumbnail((self.max_w, self.max_h), Image.Resampling.BILINEAR)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                data = img.tobytes("raw", "RGB")
                qimg = QImage(data, img.width, img.height, img.width * 3, QImage.Format_RGB888)
                self.signals.preloaded.emit(self.media_id, qimg)
        except Exception:
            pass


class FullscreenSlideshowDialog(QDialog):
    """
    True Fullscreen Slideshow Player.
    Loops through photos in the active filter with 4-second intervals.
    """

    slide_changed = Signal(int)  # emits media_id of currently shown photo

    def __init__(
        self,
        records: Sequence[MediaItemRecord],
        start_index: int = 0,
        catalog_repo: CatalogRepository | None = None,
        interval_ms: int = 4000,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent, Qt.Window | Qt.FramelessWindowHint)
        self.setObjectName("FullscreenSlideshowDialog")
        self.setStyleSheet("background-color: #000000;")
        self.hud: SlideshowHud | None = None
        self.setMouseTracking(True)

        self.records: list[MediaItemRecord] = list(records)
        self.catalog_repo = catalog_repo
        self.current_index: int = max(0, min(start_index, len(self.records) - 1)) if self.records else 0
        self.is_playing: bool = True
        self.interval_ms: int = interval_ms

        self._current_pixmap: QPixmap | None = None
        self._cache: dict[int, QPixmap] = {}
        self._pending_preload: set[int] = set()
        self._preload_pool = QThreadPool(self)
        self._preload_pool.setMaxThreadCount(2)
        self._preload_signals = _SlidePreloadSignals()
        self._preload_signals.preloaded.connect(self._on_slide_preloaded)

        # HUD control bar
        self.hud = SlideshowHud(self)
        self.hud.prev_clicked.connect(self.prev_slide)
        self.hud.play_pause_clicked.connect(self.toggle_play_pause)
        self.hud.next_clicked.connect(self.next_slide)
        self.hud.close_clicked.connect(self.accept)

        # Set fullscreen state
        self.setWindowState(Qt.WindowFullScreen)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

        # Keyboard shortcuts for guaranteed fullscreen navigation regardless of focus
        QShortcut(QKeySequence(Qt.Key_Left), self, self.prev_slide)
        QShortcut(QKeySequence(Qt.Key_Right), self, self.next_slide)
        QShortcut(QKeySequence(Qt.Key_Up), self, self.prev_slide)
        QShortcut(QKeySequence(Qt.Key_Down), self, self.next_slide)
        QShortcut(QKeySequence(Qt.Key_Space), self, self.toggle_play_pause)
        QShortcut(QKeySequence(Qt.Key_Backspace), self, self.prev_slide)

        # Auto-advance slideshow timer
        self.slide_timer = QTimer(self)
        self.slide_timer.setInterval(self.interval_ms)
        self.slide_timer.timeout.connect(self.next_slide)

        # HUD and cursor auto-hide timer (2.5 seconds)
        self.hud_timer = QTimer(self)
        self.hud_timer.setSingleShot(True)
        self.hud_timer.setInterval(2500)
        self.hud_timer.timeout.connect(self._hide_hud)

        # Load first photo
        if self.records:
            self._load_slide(self.current_index)
            self.slide_timer.start()

        self.hud_timer.start()

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        self._position_hud()

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._position_hud()

    def _position_hud(self) -> None:
        if getattr(self, "hud", None) is None:
            return
        hud_w = 210
        hud_h = 48
        x = (self.width() - hud_w) // 2
        y = self.height() - hud_h - 32
        self.hud.setGeometry(x, y, hud_w, hud_h)
        self.hud.raise_()

    def _show_hud(self) -> None:
        self.unsetCursor()
        self.hud.show()
        self.hud_timer.start()

    def _hide_hud(self) -> None:
        self.hud.hide()
        self.setCursor(Qt.BlankCursor)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        super().mouseMoveEvent(event)
        self._show_hud()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            # Advance slide on click outside the HUD
            if not self.hud.geometry().contains(event.pos()):
                self.next_slide()
                self._show_hud()
                return
        super().mousePressEvent(event)

    def event(self, event: QEvent) -> bool:
        if event.type() == QEvent.KeyPress:
            key = getattr(event, "key", lambda: -1)()
            if key in (Qt.Key_Right, Qt.Key_Down, Qt.Key_PageDown):
                self._show_hud()
                self.next_slide()
                return True
            if key in (Qt.Key_Left, Qt.Key_Up, Qt.Key_PageUp, Qt.Key_Backspace):
                self._show_hud()
                self.prev_slide()
                return True
            if key in (Qt.Key_Space, Qt.Key_P):
                self._show_hud()
                self.toggle_play_pause()
                return True
            if key in (Qt.Key_Escape, Qt.Key_F12, Qt.Key_Q):
                self.accept()
                return True
        return super().event(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        self._show_hud()
        key = event.key()
        if key in (Qt.Key_Escape, Qt.Key_F12, Qt.Key_Q):
            self.accept()
            event.accept()
            return
        if key in (Qt.Key_Space, Qt.Key_P):
            self.toggle_play_pause()
            event.accept()
            return
        if key in (Qt.Key_Right, Qt.Key_Down, Qt.Key_PageDown):
            self.next_slide()
            event.accept()
            return
        if key in (Qt.Key_Left, Qt.Key_Up, Qt.Key_PageUp, Qt.Key_Backspace):
            self.prev_slide()
            event.accept()
            return
        super().keyPressEvent(event)

    def toggle_play_pause(self) -> None:
        self.is_playing = not self.is_playing
        self.hud.set_playing(self.is_playing)
        if self.is_playing:
            self.slide_timer.start()
        else:
            self.slide_timer.stop()

    def next_slide(self) -> None:
        if not self.records:
            return
        self.current_index = (self.current_index + 1) % len(self.records)
        self._load_slide(self.current_index)
        if self.is_playing:
            self.slide_timer.start()

    def prev_slide(self) -> None:
        if not self.records:
            return
        self.current_index = (self.current_index - 1 + len(self.records)) % len(self.records)
        self._load_slide(self.current_index)
        if self.is_playing:
            self.slide_timer.start()

    def current_media_id(self) -> int | None:
        if self.records and 0 <= self.current_index < len(self.records):
            return self.records[self.current_index].id
        return None

    def _load_slide(self, index: int) -> None:
        if not self.records or not (0 <= index < len(self.records)):
            return
        rec = self.records[index]
        mid = rec.id

        if mid in self._cache:
            self._current_pixmap = self._cache[mid]
        else:
            file_path = getattr(rec, "file_path", None)
            if not file_path and self.catalog_repo is not None:
                try:
                    detail = self.catalog_repo.get_media_detail(mid)
                    file_path = str(Path(detail["folder_path"]) / str(detail["filename"]))
                except Exception:
                    file_path = None
            pix = self._load_high_res_pixmap(file_path) if file_path else None
            if pix is not None and not pix.isNull():
                self._current_pixmap = pix
                self._cache[mid] = pix
            else:
                self._current_pixmap = None

        # Clean cache if it exceeds 10 items
        if len(self._cache) > 10:
            keys_to_remove = [k for k in self._cache if k != mid][:5]
            for k in keys_to_remove:
                self._cache.pop(k, None)

        self.slide_changed.emit(mid)
        self.update()

        # Trigger async preload for next and previous slides
        if len(self.records) > 1:
            next_idx = (index + 1) % len(self.records)
            self._trigger_preload(next_idx)
            prev_idx = (index - 1 + len(self.records)) % len(self.records)
            self._trigger_preload(prev_idx)

    def _resolve_file_path(self, rec: MediaItemRecord) -> str | None:
        file_path = getattr(rec, "file_path", None)
        if not file_path and self.catalog_repo is not None:
            try:
                detail = self.catalog_repo.get_media_detail(rec.id)
                file_path = str(Path(detail["folder_path"]) / str(detail["filename"]))
            except Exception:
                file_path = None
        return file_path

    def _trigger_preload(self, index: int) -> None:
        if not self.records or not (0 <= index < len(self.records)):
            return
        rec = self.records[index]
        mid = rec.id
        if mid in self._cache or mid in self._pending_preload:
            return
        file_path = self._resolve_file_path(rec)
        if not file_path:
            return
        self._pending_preload.add(mid)
        max_w = max(1920, self.width())
        max_h = max(1080, self.height())
        worker = _SlidePreloadWorker(mid, file_path, max_w, max_h, self._preload_signals)
        self._preload_pool.start(worker)

    def _on_slide_preloaded(self, media_id: int, qimg: QImage) -> None:
        self._pending_preload.discard(media_id)
        if not qimg.isNull() and media_id not in self._cache:
            self._cache[media_id] = QPixmap.fromImage(qimg)

    def accept(self) -> None:
        self._preload_pool.waitForDone(300)
        super().accept()

    def reject(self) -> None:
        self._preload_pool.waitForDone(300)
        super().reject()

    def closeEvent(self, event: Any) -> None:
        self._preload_pool.waitForDone(300)
        super().closeEvent(event)

    def _load_high_res_pixmap(self, file_path: str) -> QPixmap | None:
        try:
            p = Path(file_path)
            if not p.exists():
                return None
            with Image.open(p) as img:
                img = ImageOps.exif_transpose(img)
                # Keep high resolution for crisp fullscreen display
                max_w = max(1920, self.width())
                max_h = max(1080, self.height())
                img.thumbnail((max_w, max_h), Image.Resampling.BILINEAR)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                data = img.tobytes("raw", "RGB")
                qimg = QImage(data, img.width, img.height, img.width * 3, QImage.Format_RGB888)
                return QPixmap.fromImage(qimg)
        except Exception:
            return None

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0))

        if self._current_pixmap is not None and not self._current_pixmap.isNull():
            scaled = self._current_pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        else:
            # Draw placeholder if image is missing
            painter.setPen(QColor(180, 180, 180))
            painter.setFont(QFont("Segoe UI", 14))
            painter.drawText(
                self.rect(),
                Qt.AlignCenter,
                "Unable to display image" if self.records else "No photos to display",
            )
