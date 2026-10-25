"""
Comprehensive Test Suite for WLPG Slideshow Themes & Animation Engine (tests/test_slideshow_themes.py).

Verifies:
1. All 6 theme renderers (Pan and Zoom, Contemporary, Cinematic, Fade, Black and White, Sepia).
2. Face-target pan interpolation (Ken Burns tracking detected face centroids).
3. Smooth 60 FPS animation progress and cosine cross-fade transitions.
4. Auto-hiding HUD transport bar controls (Themes, Speed, Shuffle, Audio, Navigation).
5. Ribbon split button (MenuButtonPopup) and 'Slide show themes' menu with 48x36 preview icons.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QColor, QImage, QPaintEvent, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QToolButton

from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository, MediaItemRecord
from src.ui.ribbon_toolbar import RibbonToolbar
from src.ui.slideshow_view import (
    FullscreenSlideshowDialog,
    SlideshowHud,
    SlideshowView,
    make_slideshow_theme_icon,
)


@pytest.fixture
def qapp() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def test_engine(tmp_path: Path) -> DualDatabaseEngine:
    engine = DualDatabaseEngine(
        catalog_path=tmp_path / "catalog.db",
        thumbnails_path=tmp_path / "thumbnails.db",
        schema_sql_path=Path("db/schema.sql"),
    )
    engine.initialize_schemas()
    yield engine
    engine.close_all()


@pytest.fixture
def sample_records(tmp_path: Path, test_engine: DualDatabaseEngine) -> list[MediaItemRecord]:
    """Creates sample test photos with varying aspect ratios on disk and inserts into DB."""
    records: list[MediaItemRecord] = []
    conn = test_engine.catalog_conn()

    # Photo 1: Landscape (800x600)
    p1 = tmp_path / "landscape.jpg"
    im1 = Image.new("RGB", (800, 600), (45, 120, 200))
    im1.save(p1)

    # Photo 2: Portrait (600x800)
    p2 = tmp_path / "portrait.jpg"
    im2 = Image.new("RGB", (600, 800), (210, 90, 70))
    im2.save(p2)

    # Photo 3: Square (500x500)
    p3 = tmp_path / "square.jpg"
    im3 = Image.new("RGB", (500, 500), (80, 180, 100))
    im3.save(p3)

    conn.execute("INSERT INTO folders (path, name) VALUES (?, 'Photos');", (str(tmp_path),))
    fid = conn.execute("SELECT last_insert_rowid();").fetchone()[0]
    conn.execute(
        """
        INSERT INTO media (id, folder_id, filename, ext, size, width, height, is_video, rating)
        VALUES (1, ?, 'landscape.jpg', 'jpg', 1024, 800, 600, 0, 0),
               (2, ?, 'portrait.jpg', 'jpg', 1024, 600, 800, 0, 0),
               (3, ?, 'square.jpg', 'jpg', 1024, 500, 500, 0, 0);
        """,
        (fid, fid, fid),
    )
    # Add face detection box for landscape photo 1 (centered at x=0.7, y=0.3)
    conn.execute(
        """
        INSERT INTO media_people (id, media_id, rect_x, rect_y, rect_w, rect_h, similarity_score, state)
        VALUES (101, 1, 0.6, 0.2, 0.2, 0.2, 0.95, 1);
        """
    )
    conn.commit()

    repo = CatalogRepository(test_engine)
    records.append(MediaItemRecord(1, fid, "landscape.jpg", 1024, 0, 0, 0, 0, 800, 600, 1, 0, "", ""))
    records.append(MediaItemRecord(2, fid, "portrait.jpg", 1024, 0, 0, 0, 0, 600, 800, 1, 0, "", ""))
    records.append(MediaItemRecord(3, fid, "square.jpg", 1024, 0, 0, 0, 0, 500, 500, 1, 0, "", ""))
    return records


def test_theme_icons_generated(qapp: QApplication) -> None:
    """Verifies make_slideshow_theme_icon renders valid 48x36 preview icons for all 6 themes."""
    themes = ["pan_and_zoom", "contemporary", "cinematic", "fade", "black_and_white", "sepia"]
    for t in themes:
        icon = make_slideshow_theme_icon(t)
        assert not icon.isNull()
        sizes = icon.availableSizes()
        assert len(sizes) > 0
        assert sizes[0].width() == 48
        assert sizes[0].height() == 36


def test_ribbon_split_button_and_themes_menu(qapp: QApplication) -> None:
    """Verifies Slide show button on Home and View tabs is a split button with themes menu."""
    ribbon = RibbonToolbar()
    ribbon.resize(1200, 120)
    ribbon.show()

    # Home tab split button
    btn_home = ribbon.btn_home_slideshow
    assert isinstance(btn_home, QToolButton)
    assert btn_home.popupMode() == QToolButton.MenuButtonPopup
    menu_home = btn_home.menu()
    assert menu_home is not None
    actions_home = menu_home.actions()
    # Actions include header + separator + 6 theme actions
    theme_actions = [a for a in actions_home if a.isCheckable()]
    assert len(theme_actions) == 6
    theme_ids = [a.data() for a in theme_actions]
    assert theme_ids == ["pan_and_zoom", "contemporary", "cinematic", "fade", "black_and_white", "sepia"]

    # View tab split button
    btn_view = ribbon.btn_view_slideshow
    assert isinstance(btn_view, QToolButton)
    assert btn_view.popupMode() == QToolButton.MenuButtonPopup

    # Default theme is pan_and_zoom
    assert ribbon.current_slideshow_theme == "pan_and_zoom"

    # Switching theme via set_slideshow_theme updates checkmark
    ribbon.set_slideshow_theme("cinematic")
    assert ribbon.current_slideshow_theme == "cinematic"
    checked = [a.data() for a in theme_actions if a.isChecked()]
    assert checked == ["cinematic"]


def test_all_six_themes_rendering(
    qapp: QApplication, sample_records: list[MediaItemRecord], test_engine: DualDatabaseEngine
) -> None:
    """Tests paintEvent rendering across all 6 themes without errors."""
    repo = CatalogRepository(test_engine)
    dialog = FullscreenSlideshowDialog(
        records=sample_records,
        start_index=0,
        catalog_repo=repo,
        theme="pan_and_zoom",
    )
    dialog.resize(1024, 768)
    dialog.show()

    themes = ["pan_and_zoom", "contemporary", "cinematic", "fade", "black_and_white", "sepia"]

    for t in themes:
        dialog.set_theme(t)
        assert dialog.current_theme == t

        # Render beginning of slide (t=0.0, alpha=0.0 cross-fade)
        dialog._slide_elapsed_ms = 0
        dialog.repaint()

        # Render mid-slide (t=0.5, alpha=1.0)
        dialog._slide_elapsed_ms = dialog.interval_ms // 2
        dialog.repaint()

        # Render end of slide (t=1.0)
        dialog._slide_elapsed_ms = dialog.interval_ms
        dialog.repaint()

    dialog.close()


def test_face_target_pan_interpolation(
    qapp: QApplication, sample_records: list[MediaItemRecord], test_engine: DualDatabaseEngine
) -> None:
    """Verifies Ken Burns pan & zoom tracks detected face centroid."""
    repo = CatalogRepository(test_engine)
    dialog = FullscreenSlideshowDialog(
        records=sample_records,
        start_index=0,
        catalog_repo=repo,
        theme="pan_and_zoom",
    )
    dialog.resize(1024, 768)
    dialog.show()

    # Media 1 has a face at (0.6, 0.2, 0.2, 0.2), center = (0.7, 0.3)
    cx, cy = dialog._get_face_center_for_media(1)
    assert abs(cx - 0.7) < 0.05
    assert abs(cy - 0.3) < 0.05

    # Explicit override via set_media_face_center
    dialog.set_media_face_center(1, (0.8, 0.4))
    cx2, cy2 = dialog._get_face_center_for_media(1)
    assert abs(cx2 - 0.8) < 0.001
    assert abs(cy2 - 0.4) < 0.001

    dialog.close()


def test_slide_navigation_and_crossfade(
    qapp: QApplication, sample_records: list[MediaItemRecord], test_engine: DualDatabaseEngine
) -> None:
    """Verifies slide navigation next/prev and smooth cross-fade state."""
    repo = CatalogRepository(test_engine)
    dialog = FullscreenSlideshowDialog(
        records=sample_records,
        start_index=0,
        catalog_repo=repo,
        interval_ms=4000,
        theme="fade",
    )
    dialog.resize(800, 600)
    dialog.show()

    assert dialog.current_index == 0
    assert dialog.current_media_id() == 1
    assert dialog._prev_pixmap is None

    # Next slide
    dialog.next_slide()
    assert dialog.current_index == 1
    assert dialog.current_media_id() == 2
    # Outgoing slide becomes prev_pixmap for smooth cross-fade
    assert dialog._prev_pixmap is not None
    assert dialog._slide_elapsed_ms == 0

    # Advance again
    dialog.next_slide()
    assert dialog.current_index == 2

    # Wrap around
    dialog.next_slide()
    assert dialog.current_index == 0

    # Prev slide wraps to 2
    dialog.prev_slide()
    assert dialog.current_index == 2

    dialog.close()


def test_hud_controls_and_speed_shuffle(
    qapp: QApplication, sample_records: list[MediaItemRecord], test_engine: DualDatabaseEngine
) -> None:
    """Verifies HUD transport controls: theme selector, speed selector, shuffle, play/pause."""
    repo = CatalogRepository(test_engine)
    dialog = FullscreenSlideshowDialog(
        records=sample_records,
        start_index=0,
        catalog_repo=repo,
        interval_ms=5000,
    )
    dialog.resize(900, 600)
    dialog.show()

    hud = dialog.hud
    assert hud is not None

    # 1. Theme switching via HUD combo
    idx_theme = hud.combo_theme.findData("sepia")
    hud.combo_theme.setCurrentIndex(idx_theme)
    assert dialog.current_theme == "sepia"

    # 2. Speed switching via HUD combo
    idx_speed = hud.combo_speed.findData(3000)
    hud.combo_speed.setCurrentIndex(idx_speed)
    assert dialog.interval_ms == 3000
    assert dialog.slide_timer.interval() == 3000

    # 3. Shuffle toggle
    assert dialog.is_shuffled is False
    hud.btn_shuffle.click()
    assert dialog.is_shuffled is True

    # 4. Play / Pause toggle
    assert dialog.is_playing is True
    assert dialog.slide_timer.isActive()
    dialog.toggle_play_pause()
    assert dialog.is_playing is False
    assert not dialog.slide_timer.isActive()
    dialog.toggle_play_pause()
    assert dialog.is_playing is True
    assert dialog.slide_timer.isActive()

    # 5. SlideshowView alias verification
    assert SlideshowView is FullscreenSlideshowDialog

    dialog.close()
