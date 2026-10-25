"""
FullscreenSlideshowDialog (QDialog) & SlideshowView — 60 FPS Authentic Slideshow Engine for PhotoGallery64.
Matches Windows Live Photo Gallery's Slideshow themes and presentation:
- 6 Themes:
  1. Pan and Zoom (Face-Aware Ken Burns with smooth pan toward detected face centroids)
  2. Contemporary (Polaroid Table Scatter on dark slate canvas with tilt & date/caption stamps)
  3. Cinematic (Ambient edge-to-edge blurred backdrop, subtle horizontal drift, film vignette)
  4. Fade (Smooth 1.2s cosine ease-in-out cross-fade over pure black)
  5. Black and White (Live high-contrast monochrome conversion + gentle pan/zoom)
  6. Sepia (Warm golden sepia duotone conversion + gentle pan/zoom)
- 60 FPS animation timer (16ms) driving normalized progress t and transition alpha
- Translucent floating HUD (Heads-Up Display) with Prev, Play/Pause, Next, Theme selector,
  Speed selector (Slow 7s / Medium 5s / Fast 3s), Shuffle toggle, Background Music player (.mp3/.wav/.flac),
  and Close button
- Auto-hiding HUD and mouse cursor on inactivity (2.5s)
- Full keyboard navigation (Space, Left, Right, Escape, F11, F12, Backspace, S, M, etc.)
"""

from __future__ import annotations

import math
import os
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps
from PySide6.QtCore import (
    QEvent,
    QObject,
    QPoint,
    QRect,
    QRectF,
    QRunnable,
    QSize,
    Qt,
    QThreadPool,
    QTimer,
    QUrl,
    Signal,
)
from PySide6.QtGui import (
    QAction,
    QBrush,
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
    QPen,
    QPixmap,
    QRadialGradient,
    QShortcut,
)
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
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


def make_slideshow_theme_icon(theme_id: str) -> QIcon:
    """Renders a rich 48x36 preview icon for the given slideshow theme."""
    pix = QPixmap(48, 36)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing, True)
    rect = QRectF(1, 1, 46, 34)

    if theme_id == "pan_and_zoom":
        p.fillRect(rect, QColor("#1B3B5F"))
        p.fillRect(QRectF(4, 4, 40, 28), QColor("#2D6296"))
        p.setPen(QPen(QColor("#FFE082"), 1.8))
        # Viewfinder corners
        p.drawLine(8, 8, 16, 8)
        p.drawLine(8, 8, 8, 14)
        p.drawLine(40, 8, 32, 8)
        p.drawLine(40, 8, 40, 14)
        p.drawLine(8, 28, 16, 28)
        p.drawLine(8, 28, 8, 22)
        p.drawLine(40, 28, 32, 28)
        p.drawLine(40, 28, 40, 22)
        # Small zoom center marker
        p.setPen(QPen(QColor("#FFFFFF"), 1.5))
        p.drawLine(20, 18, 28, 18)
        p.drawLine(24, 14, 24, 22)

    elif theme_id == "contemporary":
        p.fillRect(rect, QColor("#1E242B"))
        # Polaroid white border card tilted -6 deg
        p.save()
        p.translate(24, 18)
        p.rotate(-6)
        p.fillRect(QRectF(-14, -13, 28, 26), QColor("#FFFFFF"))
        p.fillRect(QRectF(-12, -11, 24, 17), QColor("#3B78BA"))
        p.restore()

    elif theme_id == "cinematic":
        p.fillRect(rect, QColor("#080B10"))
        # Ambient backdrop glow
        p.fillRect(QRectF(2, 5, 44, 26), QColor("#1D3B66"))
        # Cinema frame with film letterboxing
        p.fillRect(QRectF(1, 1, 46, 6), QColor("#000000"))
        p.fillRect(QRectF(1, 29, 46, 6), QColor("#000000"))
        p.fillRect(QRectF(7, 7, 34, 22), QColor("#4B88D6"))

    elif theme_id == "fade":
        p.fillRect(rect, QColor("#000000"))
        # Two dissolving overlapping frames
        p.fillRect(QRectF(5, 6, 24, 24), QColor(255, 255, 255, 110))
        p.fillRect(QRectF(17, 6, 26, 24), QColor(70, 130, 220, 200))

    elif theme_id == "black_and_white":
        p.fillRect(rect, QColor("#1F1F1F"))
        p.fillRect(QRectF(5, 5, 38, 26), QColor("#E5E5E5"))
        p.fillRect(QRectF(5, 18, 38, 13), QColor("#4A4A4A"))
        p.setPen(QPen(QColor("#FFFFFF"), 1.2))
        p.drawEllipse(12, 9, 6, 6)

    elif theme_id == "sepia":
        p.fillRect(rect, QColor("#3D2611"))
        p.fillRect(QRectF(5, 5, 38, 26), QColor("#DDB88C"))
        p.fillRect(QRectF(5, 18, 38, 13), QColor("#8C5B2F"))
        p.setPen(QPen(QColor("#FFF0DE"), 1.2))
        p.drawEllipse(12, 9, 6, 6)

    else:
        p.fillRect(rect, QColor("#2A5B8A"))

    p.setPen(QPen(QColor("#7DA2CE"), 1))
    p.drawRoundedRect(rect, 2, 2)
    p.end()
    return QIcon(pix)


class SlideshowHud(QFrame):
    """
    Semi-transparent floating control pill at bottom of screen.
    Includes Prev, Play/Pause, Next, Theme Selector, Speed Selector, Shuffle, Background Music, and Exit.
    """

    prev_clicked = Signal()
    play_pause_clicked = Signal()
    next_clicked = Signal()
    close_clicked = Signal()
    theme_changed = Signal(str)
    speed_changed = Signal(int)
    shuffle_toggled = Signal(bool)
    music_clicked = Signal()

    THEME_OPTIONS = [
        ("pan_and_zoom", "Pan and Zoom"),
        ("contemporary", "Contemporary"),
        ("cinematic", "Cinematic"),
        ("fade", "Fade"),
        ("black_and_white", "Black and White"),
        ("sepia", "Sepia"),
    ]

    SPEED_OPTIONS = [
        (7000, "Slow (7s)"),
        (5000, "Medium (5s)"),
        (3000, "Fast (3s)"),
    ]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("SlideshowHud")
        self.setFixedHeight(48)
        self.setStyleSheet(
            """
            QFrame#SlideshowHud {
                background-color: rgba(18, 22, 28, 225);
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
                background: rgba(255, 255, 255, 55);
                border: 1px solid rgba(255, 255, 255, 120);
            }
            QToolButton:pressed {
                background: rgba(0, 120, 215, 180);
            }
            QToolButton:checked {
                background: rgba(0, 120, 215, 200);
                border: 1px solid rgba(120, 190, 255, 220);
            }
            QComboBox {
                background-color: rgba(30, 36, 46, 220);
                color: #FFFFFF;
                border: 1px solid rgba(120, 150, 190, 140);
                border-radius: 13px;
                padding: 2px 10px 2px 10px;
                font-family: 'Segoe UI';
                font-size: 9pt;
            }
            QComboBox:hover {
                background-color: rgba(45, 54, 70, 230);
                border: 1px solid rgba(160, 195, 240, 200);
            }
            QComboBox::drop-down {
                border: none;
                width: 14px;
            }
            QComboBox QAbstractItemView {
                background-color: #1E242E;
                color: #FFFFFF;
                selection-background-color: #0078D7;
                selection-color: #FFFFFF;
                border: 1px solid #70A6E8;
            }
            """
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 4, 14, 4)
        layout.setSpacing(8)

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

        # Separator 1
        sep1 = QFrame(self)
        sep1.setFrameShape(QFrame.VLine)
        sep1.setStyleSheet("color: rgba(255, 255, 255, 70); margin: 6px 2px;")
        layout.addWidget(sep1)

        # Theme selector combo
        self.combo_theme = QComboBox(self)
        self.combo_theme.setFixedHeight(28)
        self.combo_theme.setCursor(Qt.PointingHandCursor)
        self.combo_theme.setToolTip("Select slide show theme")
        for tid, tlabel in self.THEME_OPTIONS:
            self.combo_theme.addItem(tlabel, tid)
        self.combo_theme.currentIndexChanged.connect(self._on_theme_index_changed)
        layout.addWidget(self.combo_theme)

        # Speed selector combo
        self.combo_speed = QComboBox(self)
        self.combo_speed.setFixedHeight(28)
        self.combo_speed.setCursor(Qt.PointingHandCursor)
        self.combo_speed.setToolTip("Slide transition speed")
        for spd_ms, spd_label in self.SPEED_OPTIONS:
            self.combo_speed.addItem(spd_label, spd_ms)
        self.combo_speed.setCurrentIndex(1)  # Medium (5s) default
        self.combo_speed.currentIndexChanged.connect(self._on_speed_index_changed)
        layout.addWidget(self.combo_speed)

        # Shuffle toggle button
        self.btn_shuffle = QToolButton(self)
        self.btn_shuffle.setText("🔀")
        self.btn_shuffle.setCheckable(True)
        self.btn_shuffle.setFixedSize(32, 32)
        self.btn_shuffle.setToolTip("Shuffle photos")
        self.btn_shuffle.setCursor(Qt.PointingHandCursor)
        self.btn_shuffle.toggled.connect(self.shuffle_toggled.emit)
        layout.addWidget(self.btn_shuffle)

        # Background music button
        self.btn_music = QToolButton(self)
        self.btn_music.setText("🎵")
        self.btn_music.setFixedSize(32, 32)
        self.btn_music.setToolTip("Select background music (.mp3, .wav, .flac)")
        self.btn_music.setCursor(Qt.PointingHandCursor)
        self.btn_music.clicked.connect(self.music_clicked.emit)
        layout.addWidget(self.btn_music)

        # Separator 2
        sep2 = QFrame(self)
        sep2.setFrameShape(QFrame.VLine)
        sep2.setStyleSheet("color: rgba(255, 255, 255, 70); margin: 6px 2px;")
        layout.addWidget(sep2)

        # Close / Exit button (✕)
        self.btn_close = QToolButton(self)
        self.btn_close.setIcon(make_wlpg_icon("close_file", 18))
        self.btn_close.setIconSize(QSize(18, 18))
        self.btn_close.setFixedSize(36, 36)
        self.btn_close.setToolTip("Exit slide show (Esc / F11 / F12)")
        self.btn_close.setCursor(Qt.PointingHandCursor)
        self.btn_close.clicked.connect(self.close_clicked.emit)
        layout.addWidget(self.btn_close)

        self.setFocusPolicy(Qt.NoFocus)
        for b in (
            self.btn_prev,
            self.btn_play_pause,
            self.btn_next,
            self.combo_theme,
            self.combo_speed,
            self.btn_shuffle,
            self.btn_music,
            self.btn_close,
        ):
            b.setFocusPolicy(Qt.NoFocus)

    def set_playing(self, is_playing: bool) -> None:
        icon_name = "pause" if is_playing else "slideshow"
        self.btn_play_pause.setIcon(make_wlpg_icon(icon_name, 22))
        self.btn_play_pause.setToolTip("Pause (Space)" if is_playing else "Play (Space)")

    def set_theme(self, theme_id: str) -> None:
        idx = self.combo_theme.findData(theme_id)
        if idx >= 0 and self.combo_theme.currentIndex() != idx:
            self.combo_theme.blockSignals(True)
            self.combo_theme.setCurrentIndex(idx)
            self.combo_theme.blockSignals(False)

    def set_speed_ms(self, interval_ms: int) -> None:
        idx = self.combo_speed.findData(interval_ms)
        if idx >= 0 and self.combo_speed.currentIndex() != idx:
            self.combo_speed.blockSignals(True)
            self.combo_speed.setCurrentIndex(idx)
            self.combo_speed.blockSignals(False)

    def set_shuffled(self, is_shuffled: bool) -> None:
        self.btn_shuffle.blockSignals(True)
        self.btn_shuffle.setChecked(bool(is_shuffled))
        self.btn_shuffle.blockSignals(False)

    def set_music_title(self, title: str) -> None:
        self.btn_music.setToolTip(f"Playing music: {title} (Click to change)")

    def _on_theme_index_changed(self, idx: int) -> None:
        tid = self.combo_theme.itemData(idx)
        if tid:
            self.theme_changed.emit(str(tid))

    def _on_speed_index_changed(self, idx: int) -> None:
        spd = self.combo_speed.itemData(idx)
        if spd:
            self.speed_changed.emit(int(spd))


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
    60 FPS Animated Fullscreen Slideshow Player.
    Supports all 6 WLPG themes with smooth cross-fades, pan/zoom motion, and face tracking.
    """

    slide_changed = Signal(int)  # emits media_id of currently shown photo
    theme_changed = Signal(str)  # emits theme_id when changed

    def __init__(
        self,
        records: Sequence[MediaItemRecord],
        start_index: int = 0,
        catalog_repo: CatalogRepository | None = None,
        interval_ms: int = 5000,
        theme: str = "pan_and_zoom",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent, Qt.Window | Qt.FramelessWindowHint)
        self.setObjectName("FullscreenSlideshowDialog")
        self.setStyleSheet("background-color: #000000;")
        self.setMouseTracking(True)

        self.records: list[MediaItemRecord] = list(records)
        self.catalog_repo = catalog_repo
        self.current_index: int = max(0, min(start_index, len(self.records) - 1)) if self.records else 0
        self.current_theme: str = str(theme)
        self.is_playing: bool = True
        self.interval_ms: int = int(interval_ms)
        self.is_shuffled: bool = False
        self._shuffled_order: list[int] = []

        self._current_pixmap: QPixmap | None = None
        self._prev_pixmap: QPixmap | None = None
        self._cache: dict[int, QPixmap] = {}
        self._cinematic_backdrops: dict[int, QPixmap] = {}
        self._bw_cache: dict[int, QPixmap] = {}
        self._sepia_cache: dict[int, QPixmap] = {}
        self._face_targets: dict[int, tuple[float, float]] = {}
        self._contemporary_stack: list[dict[str, Any]] = []
        self._slide_elapsed_ms: int = 0

        self._pending_preload: set[int] = set()
        self._preload_pool = QThreadPool(self)
        self._preload_pool.setMaxThreadCount(2)
        self._preload_signals = _SlidePreloadSignals()
        self._preload_signals.preloaded.connect(self._on_slide_preloaded)

        # Audio playback
        self.audio_output = QAudioOutput(self)
        self.music_player = QMediaPlayer(self)
        self.music_player.setAudioOutput(self.audio_output)
        self.music_player.mediaStatusChanged.connect(self._on_media_status_changed)

        # HUD control bar
        self.hud = SlideshowHud(self)
        self.hud.set_theme(self.current_theme)
        self.hud.set_speed_ms(self.interval_ms)
        self.hud.prev_clicked.connect(self.prev_slide)
        self.hud.play_pause_clicked.connect(self.toggle_play_pause)
        self.hud.next_clicked.connect(self.next_slide)
        self.hud.close_clicked.connect(self.accept)
        self.hud.theme_changed.connect(self.set_theme)
        self.hud.speed_changed.connect(self.set_speed)
        self.hud.shuffle_toggled.connect(self.toggle_shuffle)
        self.hud.music_clicked.connect(self.choose_background_music)

        # Set fullscreen state
        self.setWindowState(Qt.WindowFullScreen)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

        # Keyboard shortcuts
        QShortcut(QKeySequence(Qt.Key_Left), self, self.prev_slide)
        QShortcut(QKeySequence(Qt.Key_Right), self, self.next_slide)
        QShortcut(QKeySequence(Qt.Key_Up), self, self.prev_slide)
        QShortcut(QKeySequence(Qt.Key_Down), self, self.next_slide)
        QShortcut(QKeySequence(Qt.Key_Space), self, self.toggle_play_pause)
        QShortcut(QKeySequence(Qt.Key_Backspace), self, self.prev_slide)
        QShortcut(QKeySequence(Qt.Key_S), self, lambda: self.toggle_shuffle())
        QShortcut(QKeySequence(Qt.Key_M), self, self.choose_background_music)

        # Primary advance timer (kept for compatibility with existing tests)
        self.slide_timer = QTimer(self)
        self.slide_timer.setInterval(self.interval_ms)
        self.slide_timer.timeout.connect(self.next_slide)

        # 60 FPS smooth animation timer (16ms tick)
        self.anim_timer = QTimer(self)
        self.anim_timer.setInterval(16)
        self.anim_timer.timeout.connect(self._on_anim_tick)

        # HUD and cursor auto-hide timer (2.5 seconds)
        self.hud_timer = QTimer(self)
        self.hud_timer.setSingleShot(True)
        self.hud_timer.setInterval(2500)
        self.hud_timer.timeout.connect(self._hide_hud)

        # Load first photo
        if self.records:
            self._load_slide(self.current_index)
            self.slide_timer.start()
            self.anim_timer.start()

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
        hud_w = min(680, max(520, self.width() - 40))
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
            if key in (Qt.Key_Escape, Qt.Key_F11, Qt.Key_F12, Qt.Key_Q):
                self.accept()
                return True
        return super().event(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        self._show_hud()
        key = event.key()
        if key in (Qt.Key_Escape, Qt.Key_F11, Qt.Key_F12, Qt.Key_Q):
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

    def _on_anim_tick(self) -> None:
        if self.is_playing:
            self._slide_elapsed_ms += 16
            if self._slide_elapsed_ms >= self.interval_ms:
                self._slide_elapsed_ms = 0
                self.next_slide()
                return
        self.update()

    def _on_media_status_changed(self, status: QMediaPlayer.MediaStatus) -> None:
        # Loop background music seamlessly
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.music_player.setPosition(0)
            self.music_player.play()

    def set_theme(self, theme: str) -> None:
        """Switches the active slideshow theme."""
        self.current_theme = str(theme)
        if self.hud is not None:
            self.hud.set_theme(self.current_theme)
        self.theme_changed.emit(self.current_theme)
        self.update()

    def set_speed(self, speed_ms: int) -> None:
        """Sets the slide duration in milliseconds."""
        self.interval_ms = max(1000, int(speed_ms))
        self.slide_timer.setInterval(self.interval_ms)
        if self.hud is not None:
            self.hud.set_speed_ms(self.interval_ms)

    def toggle_shuffle(self, enabled: bool | None = None) -> None:
        """Toggles randomized shuffle mode."""
        self.is_shuffled = (not self.is_shuffled) if enabled is None else bool(enabled)
        self._shuffled_order.clear()
        if self.hud is not None:
            self.hud.set_shuffled(self.is_shuffled)

    def choose_background_music(self) -> None:
        """Opens a file dialog to pick an audio file for background music."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Background Music",
            "",
            "Audio Files (*.mp3 *.wav *.flac *.m4a *.aac *.ogg);;All Files (*)",
        )
        if file_path:
            self.load_background_music(file_path)

    def load_background_music(self, file_path: str) -> None:
        """Loads and plays background music."""
        p = Path(file_path)
        if not p.exists():
            return
        self.music_player.setSource(QUrl.fromLocalFile(str(p.resolve())))
        self.audio_output.setVolume(0.7)
        self.music_player.play()
        if self.hud is not None:
            self.hud.set_music_title(p.name)

    def toggle_play_pause(self) -> None:
        self.is_playing = not self.is_playing
        self.hud.set_playing(self.is_playing)
        if self.is_playing:
            self.slide_timer.start()
            if self.music_player.playbackState() == QMediaPlayer.PlaybackState.PausedState:
                self.music_player.play()
        else:
            self.slide_timer.stop()
            if self.music_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                self.music_player.pause()

    def next_slide(self) -> None:
        if not self.records:
            return
        self._prev_pixmap = self._current_pixmap
        self._slide_elapsed_ms = 0
        if self.is_shuffled and len(self.records) > 1:
            if not self._shuffled_order:
                candidates = [i for i in range(len(self.records)) if i != self.current_index]
                random.shuffle(candidates)
                self._shuffled_order = candidates
            self.current_index = self._shuffled_order.pop(0)
        else:
            self.current_index = (self.current_index + 1) % len(self.records)
        self._load_slide(self.current_index)
        if self.is_playing:
            self.slide_timer.start()

    def prev_slide(self) -> None:
        if not self.records:
            return
        self._prev_pixmap = self._current_pixmap
        self._slide_elapsed_ms = 0
        self.current_index = (self.current_index - 1 + len(self.records)) % len(self.records)
        self._load_slide(self.current_index)
        if self.is_playing:
            self.slide_timer.start()

    def current_media_id(self) -> int | None:
        if self.records and 0 <= self.current_index < len(self.records):
            return self.records[self.current_index].id
        return None

    def set_media_face_center(self, media_id: int, center: tuple[float, float]) -> None:
        """Sets or overrides the target normalized face center coordinates (cx, cy) for a media item."""
        self._face_targets[media_id] = (
            max(0.1, min(0.9, float(center[0]))),
            max(0.1, min(0.9, float(center[1]))),
        )

    def _get_face_center_for_media(self, media_id: int) -> tuple[float, float]:
        """Queries face regions from media_people or returns a scenic drift center."""
        if media_id in self._face_targets:
            return self._face_targets[media_id]

        avg_cx, avg_cy = 0.5, 0.45
        if self.catalog_repo:
            try:
                conn = self.catalog_repo.engine.catalog_conn()
                rows = conn.execute(
                    "SELECT rect_x, rect_y, rect_w, rect_h FROM media_people WHERE media_id = ? AND rect_w IS NOT NULL AND rect_h IS NOT NULL;",
                    (media_id,),
                ).fetchall()
                if rows:
                    cxs = [float(r["rect_x"]) + float(r["rect_w"]) / 2.0 for r in rows]
                    cys = [float(r["rect_y"]) + float(r["rect_h"]) / 2.0 for r in rows]
                    avg_cx = sum(cxs) / len(cxs)
                    avg_cy = sum(cys) / len(cys)
                    avg_cx = max(0.2, min(0.8, avg_cx))
                    avg_cy = max(0.2, min(0.8, avg_cy))
            except Exception:
                pass

        self._face_targets[media_id] = (avg_cx, avg_cy)
        return (avg_cx, avg_cy)

    def _get_caption_or_date(self, rec: MediaItemRecord) -> str:
        """Retrieves formatted caption or date for the Polaroid border in Contemporary mode."""
        mid = rec.id
        title = getattr(rec, "title", None) or getattr(rec, "filename", None) or ""
        date_taken = getattr(rec, "date_taken", None)
        if not date_taken and self.catalog_repo:
            try:
                detail = self.catalog_repo.get_media_detail(mid)
                date_taken = detail.get("date_taken")
                if not title:
                    title = detail.get("caption") or detail.get("filename") or ""
            except Exception:
                pass

        if date_taken:
            d_str = str(date_taken).split("T")[0].split(" ")[0]
            if title and title != str(getattr(rec, "filename", "")):
                return f"{title} — {d_str}"
            return d_str
        return title or "Photo"

    def _load_slide(self, index: int) -> None:
        if not self.records or not (0 <= index < len(self.records)):
            return
        rec = self.records[index]
        mid = rec.id

        if mid in self._cache:
            self._current_pixmap = self._cache[mid]
        else:
            file_path = self._resolve_file_path(rec)
            pix = self._load_high_res_pixmap(file_path) if file_path else None
            if pix is not None and not pix.isNull():
                self._current_pixmap = pix
                self._cache[mid] = pix
            else:
                self._current_pixmap = None

        # Clean cache if it exceeds 15 items
        if len(self._cache) > 15:
            keys_to_remove = [k for k in self._cache if k != mid][:6]
            for k in keys_to_remove:
                self._cache.pop(k, None)
                self._cinematic_backdrops.pop(k, None)
                self._bw_cache.pop(k, None)
                self._sepia_cache.pop(k, None)

        # Update Contemporary table scatter stack
        if self._current_pixmap is not None:
            tilt = (((mid * 29) % 70) - 35) / 10.0  # -3.5 deg to +3.5 deg
            caption = self._get_caption_or_date(rec)
            ox = (((mid * 13) % 40) - 20)  # -20px to +20px
            oy = (((mid * 19) % 30) - 15)  # -15px to +15px
            card = {
                "media_id": mid,
                "pixmap": self._current_pixmap,
                "tilt": tilt,
                "caption": caption,
                "offset_x": ox,
                "offset_y": oy,
            }
            self._contemporary_stack.append(card)
            if len(self._contemporary_stack) > 3:
                self._contemporary_stack = self._contemporary_stack[-3:]

        self.slide_changed.emit(mid)
        self.update()

        # Preload adjacent slides
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
        self.anim_timer.stop()
        self.slide_timer.stop()
        self.music_player.stop()
        self._preload_pool.waitForDone(300)
        super().accept()

    def reject(self) -> None:
        self.anim_timer.stop()
        self.slide_timer.stop()
        self.music_player.stop()
        self._preload_pool.waitForDone(300)
        super().reject()

    def closeEvent(self, event: Any) -> None:
        self.anim_timer.stop()
        self.slide_timer.stop()
        self.music_player.stop()
        self._preload_pool.waitForDone(300)
        super().closeEvent(event)

    def _load_high_res_pixmap(self, file_path: str) -> QPixmap | None:
        try:
            p = Path(file_path)
            if not p.exists():
                return None
            with Image.open(p) as img:
                img = ImageOps.exif_transpose(img)
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

    def _get_bw_pixmap(self, media_id: int, base_pix: QPixmap) -> QPixmap:
        """Returns cached high-contrast black & white pixmap."""
        if media_id in self._bw_cache:
            return self._bw_cache[media_id]
        qimg = base_pix.toImage().convertToFormat(QImage.Format_Grayscale8).convertToFormat(QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self._bw_cache[media_id] = pix
        return pix

    def _get_sepia_pixmap(self, media_id: int, base_pix: QPixmap) -> QPixmap:
        """Returns cached warm sepia duotone pixmap."""
        if media_id in self._sepia_cache:
            return self._sepia_cache[media_id]
        qimg = base_pix.toImage().convertToFormat(QImage.Format_RGB888)
        w, h = qimg.width(), qimg.height()
        b = qimg.constBits()
        arr = np.frombuffer(b, dtype=np.uint8).reshape((h, w, 3))
        kernel = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131],
        ])
        sepia_arr = cv2.transform(arr, kernel)
        sepia_arr = np.clip(sepia_arr, 0, 255).astype(np.uint8)
        out_img = QImage(sepia_arr.tobytes(), w, h, w * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(out_img)
        self._sepia_cache[media_id] = pix
        return pix

    def _get_cinematic_backdrop(self, media_id: int, base_pix: QPixmap) -> QPixmap:
        """Returns cached edge-to-edge Gaussian blurred ambient backdrop."""
        if media_id in self._cinematic_backdrops:
            return self._cinematic_backdrops[media_id]
        small = base_pix.scaled(160, 90, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        qimg = small.toImage().convertToFormat(QImage.Format_RGB888)
        w, h = qimg.width(), qimg.height()
        b = qimg.constBits()
        arr = np.frombuffer(b, dtype=np.uint8).reshape((h, w, 3))
        blurred = cv2.GaussianBlur(arr, (25, 25), 12)
        blurred = (blurred * 0.55).astype(np.uint8)
        out_img = QImage(blurred.tobytes(), w, h, w * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(out_img)
        self._cinematic_backdrops[media_id] = pix
        return pix

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        if not self.records or self._current_pixmap is None or self._current_pixmap.isNull():
            painter.fillRect(self.rect(), QColor(0, 0, 0))
            painter.setPen(QColor(180, 180, 180))
            painter.setFont(QFont("Segoe UI", 14))
            painter.drawText(
                self.rect(),
                Qt.AlignCenter,
                "Unable to display image" if self.records else "No photos to display",
            )
            return

        # Calculate normalized progress t in [0.0, 1.0]
        total_ms = float(max(1000, self.interval_ms))
        t = min(1.0, max(0.0, self._slide_elapsed_ms / total_ms))

        # Transition cross-fade alpha in [0.0, 1.0]
        fade_dur = 1200 if self.current_theme == "fade" else 1000
        if self._slide_elapsed_ms < fade_dur and self._prev_pixmap is not None:
            fade_t = min(1.0, self._slide_elapsed_ms / float(fade_dur))
            alpha = 0.5 * (1.0 - math.cos(fade_t * math.pi))
        else:
            alpha = 1.0

        theme = self.current_theme
        if theme == "contemporary":
            self._paint_contemporary(painter, t, alpha)
        elif theme == "cinematic":
            self._paint_cinematic(painter, t, alpha)
        elif theme == "fade":
            self._paint_fade(painter, t, alpha)
        elif theme == "black_and_white":
            self._paint_bw(painter, t, alpha)
        elif theme == "sepia":
            self._paint_sepia(painter, t, alpha)
        else:
            self._paint_pan_and_zoom(painter, t, alpha)

    def _paint_pan_and_zoom(
        self,
        painter: QPainter,
        t: float,
        alpha: float,
        override_pixmap: QPixmap | None = None,
    ) -> None:
        """Face-Aware Ken Burns: smooth zoom 1.0x-1.16x and pan toward face centroid with cross-fade."""
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        cur_pix = override_pixmap or self._current_pixmap
        if cur_pix is None or cur_pix.isNull():
            return

        mid = self.current_media_id() or 0
        face_center = self._get_face_center_for_media(mid)

        # Alternating zoom direction and smooth ease-in-out
        smooth_t = 0.5 * (1.0 - math.cos(t * math.pi))
        if self.current_index % 2 == 0:
            scale = 1.0 + 0.16 * smooth_t
            cx = 0.5 + (face_center[0] - 0.5) * smooth_t
            cy = 0.5 + (face_center[1] - 0.5) * smooth_t
        else:
            scale = 1.16 - 0.16 * smooth_t
            cx = face_center[0] + (0.5 - face_center[0]) * smooth_t
            cy = face_center[1] + (0.5 - face_center[1]) * smooth_t

        sw_w = self.width()
        sw_h = self.height()
        pw, ph = cur_pix.width(), cur_pix.height()
        base_scale = min(sw_w / float(pw), sw_h / float(ph))
        bw = pw * base_scale
        bh = ph * base_scale
        target_w = bw * scale
        target_h = bh * scale

        img_px = target_w * cx
        img_py = target_h * cy
        draw_x = (sw_w / 2.0) - img_px
        draw_y = (sw_h / 2.0) - img_py

        if target_w >= sw_w:
            draw_x = min(0.0, max(float(sw_w - target_w), draw_x))
        else:
            draw_x = (sw_w - target_w) / 2.0

        if target_h >= sw_h:
            draw_y = min(0.0, max(float(sw_h - target_h), draw_y))
        else:
            draw_y = (sw_h - target_h) / 2.0

        # Draw previous slide if cross-fading
        if alpha < 1.0 and self._prev_pixmap is not None and not self._prev_pixmap.isNull():
            prev_scaled = self._prev_pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            px = (sw_w - prev_scaled.width()) // 2
            py = (sw_h - prev_scaled.height()) // 2
            painter.setOpacity(1.0 - alpha)
            painter.drawPixmap(px, py, prev_scaled)

        # Draw current slide with alpha opacity
        painter.setOpacity(alpha)
        scaled_curr = cur_pix.scaled(
            int(target_w), int(target_h), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        painter.drawPixmap(int(draw_x), int(draw_y), scaled_curr)
        painter.setOpacity(1.0)

    def _paint_contemporary(self, painter: QPainter, t: float, alpha: float) -> None:
        """Polaroid Table Scatter on dark slate canvas with tilt, drop shadow, and caption stamps."""
        # 1. Warm dark slate table canvas background
        sw_w, sw_h = self.width(), self.height()
        grad = QRadialGradient(sw_w / 2.0, sw_h / 2.0, max(sw_w, sw_h) * 0.7)
        grad.setColorAt(0.0, QColor("#222832"))
        grad.setColorAt(1.0, QColor("#14171D"))
        painter.fillRect(self.rect(), grad)

        if not self._contemporary_stack:
            return

        # Draw previous 2 cards underneath if available
        cards_to_draw = self._contemporary_stack[-3:]
        for idx, card in enumerate(cards_to_draw):
            is_current = (idx == len(cards_to_draw) - 1)
            pix: QPixmap = card["pixmap"]
            tilt: float = card["tilt"]
            caption: str = card["caption"]
            ox: float = card["offset_x"]
            oy: float = card["offset_y"]

            # Polaroid dimensions
            pw, ph = pix.width(), pix.height()
            inner_w = max(260, min(int(sw_w * 0.52), 620))
            inner_h = int(inner_w * (ph / float(pw)))
            if inner_h > int(sw_h * 0.62):
                inner_h = int(sw_h * 0.62)
                inner_w = int(inner_h * (pw / float(ph)))

            card_w = inner_w + 28
            card_h = inner_h + 14 + 48

            card_cx = (sw_w / 2.0) + ox
            card_cy = (sw_h / 2.0) + oy

            # Entry animation for current card (first 0.7s)
            card_alpha = 1.0
            card_scale = 1.0
            card_y_offset = 0.0
            if is_current:
                entry_t = min(1.0, self._slide_elapsed_ms / 700.0)
                ease = 1.0 - math.pow(1.0 - entry_t, 3)
                card_y_offset = (1.0 - ease) * 50.0
                card_scale = 0.94 + 0.06 * ease
                card_alpha = alpha * ease

            painter.save()
            painter.setOpacity(card_alpha)
            painter.translate(card_cx, card_cy + card_y_offset)
            painter.rotate(tilt)
            painter.scale(card_scale, card_scale)

            # Drop shadow
            shadow_rect = QRectF(-card_w / 2.0 + 4, -card_h / 2.0 + 7, card_w, card_h)
            painter.fillRect(shadow_rect, QColor(0, 0, 0, 75))

            # White Polaroid Card
            card_rect = QRectF(-card_w / 2.0, -card_h / 2.0, card_w, card_h)
            painter.fillRect(card_rect, QColor("#FFFFFF"))
            painter.setPen(QPen(QColor("#E2E8F0"), 1.0))
            painter.drawRect(card_rect)

            # Photo
            photo_rect = QRectF(-card_w / 2.0 + 14, -card_h / 2.0 + 14, inner_w, inner_h)
            scaled_photo = pix.scaled(
                int(inner_w), int(inner_h), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            painter.drawPixmap(photo_rect.toRect(), scaled_photo)

            # Caption on bottom border
            caption_rect = QRectF(-card_w / 2.0 + 14, card_h / 2.0 - 44, inner_w, 36)
            painter.setPen(QColor("#2D3748"))
            painter.setFont(QFont("Segoe UI", 11))
            painter.drawText(caption_rect, Qt.AlignCenter | Qt.AlignVCenter, caption)

            painter.restore()

    def _paint_cinematic(self, painter: QPainter, t: float, alpha: float) -> None:
        """Ambient blurred edge-to-edge backdrop, subtle horizontal drift, film vignette."""
        mid = self.current_media_id() or 0
        sw_w, sw_h = self.width(), self.height()

        # 1. Edge-to-edge ambient blurred backdrop
        bg_pix = self._get_cinematic_backdrop(mid, self._current_pixmap)
        scaled_bg = bg_pix.scaled(self.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        painter.drawPixmap(0, 0, scaled_bg)

        # 2. Film vignette
        vignette = QRadialGradient(sw_w / 2.0, sw_h / 2.0, max(sw_w, sw_h) * 0.65)
        vignette.setColorAt(0.0, QColor(0, 0, 0, 40))
        vignette.setColorAt(0.8, QColor(0, 0, 0, 160))
        vignette.setColorAt(1.0, QColor(0, 0, 0, 220))
        painter.fillRect(self.rect(), vignette)

        # 3. Foreground sharp photo with subtle horizontal drift
        pw, ph = self._current_pixmap.width(), self._current_pixmap.height()
        fit_scale = min((sw_w * 0.84) / float(pw), (sw_h * 0.84) / float(ph))
        target_w = int(pw * fit_scale)
        target_h = int(ph * fit_scale)

        drift_dir = 1.0 if (self.current_index % 2 == 0) else -1.0
        drift_dx = 28.0 * (t - 0.5) * drift_dir
        draw_x = int((sw_w - target_w) / 2.0 + drift_dx)
        draw_y = int((sw_h - target_h) / 2.0)

        # Draw drop shadow behind foreground photo
        painter.fillRect(
            QRect(draw_x + 5, draw_y + 8, target_w, target_h), QColor(0, 0, 0, 90)
        )

        scaled_fg = self._current_pixmap.scaled(
            target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        painter.setOpacity(alpha)
        painter.drawPixmap(draw_x, draw_y, scaled_fg)
        painter.setOpacity(1.0)

    def _paint_fade(self, painter: QPainter, t: float, alpha: float) -> None:
        """Pure black background with smooth 1.2s cosine ease-in-out opacity cross-fade."""
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        sw_w, sw_h = self.width(), self.height()

        if alpha < 1.0 and self._prev_pixmap is not None and not self._prev_pixmap.isNull():
            prev_scaled = self._prev_pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            px = (sw_w - prev_scaled.width()) // 2
            py = (sw_h - prev_scaled.height()) // 2
            painter.setOpacity(1.0 - alpha)
            painter.drawPixmap(px, py, prev_scaled)

        curr_scaled = self._current_pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        cx = (sw_w - curr_scaled.width()) // 2
        cy = (sw_h - curr_scaled.height()) // 2
        painter.setOpacity(alpha)
        painter.drawPixmap(cx, cy, curr_scaled)
        painter.setOpacity(1.0)

    def _paint_bw(self, painter: QPainter, t: float, alpha: float) -> None:
        """Live high-contrast monochrome conversion + gentle pan/zoom and cross-fade."""
        mid = self.current_media_id() or 0
        bw_pix = self._get_bw_pixmap(mid, self._current_pixmap)
        self._paint_pan_and_zoom(painter, t, alpha, override_pixmap=bw_pix)

    def _paint_sepia(self, painter: QPainter, t: float, alpha: float) -> None:
        """Warm sepia duotone conversion + gentle pan/zoom and cross-fade."""
        mid = self.current_media_id() or 0
        sepia_pix = self._get_sepia_pixmap(mid, self._current_pixmap)
        self._paint_pan_and_zoom(painter, t, alpha, override_pixmap=sepia_pix)


# Alias SlideshowView to FullscreenSlideshowDialog
SlideshowView = FullscreenSlideshowDialog
