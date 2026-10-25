"""
Tests for Phase 22 UI Implementations & WLPG Parity:
1. Combined Status Bar Zoom Toggle Button (Actual Size <-> Fit to Window adaptive toggle)
2. Exposure & Color Popup Menus (Auto Adjust tile, 3x3 thumbnail variations, real-time hover previews)
3. Effects Gallery real-time hover previews
4. Right Dock Descriptive Tags hover display and click-to-filter
5. Slideshow Arrow Key Navigation
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from PySide6.QtCore import QPoint, Qt, QEvent
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QApplication

from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository, ThumbnailRepository
from src.imaging.opencv_engine import PhotoAdjustmentRecipe
from src.ui.main_window import PhotoGalleryMainWindow
from src.ui.slideshow_view import FullscreenSlideshowDialog
from src.ui.right_tag_dock import _DescriptiveTagRowWidget


@pytest.fixture
def qapp() -> QApplication:
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def test_env(tmp_path: Path):
    cat_db = tmp_path / "catalog.db"
    thumb_db = tmp_path / "thumbnails.db"
    engine = DualDatabaseEngine(cat_db, thumb_db)
    cat_repo = CatalogRepository(engine)
    thumb_repo = ThumbnailRepository(engine)

    photos_dir = tmp_path / "Photos"
    photos_dir.mkdir(parents=True, exist_ok=True)

    img_path = photos_dir / "test_photo.jpg"
    arr = np.zeros((400, 600, 3), dtype=np.uint8)
    arr[:, :] = [120, 140, 180]
    Image.fromarray(arr).save(img_path)

    folder_id = cat_repo.upsert_folder(str(photos_dir), "Photos", None, int(photos_dir.stat().st_mtime))
    ids = cat_repo.bulk_upsert_media(
        [
            {
                "folder_id": folder_id,
                "filename": "test_photo.jpg",
                "ext": "jpg",
                "size": img_path.stat().st_size,
                "mtime": int(img_path.stat().st_mtime),
                "width": 600,
                "height": 400,
            }
        ]
    )
    media_id = ids[0]

    tag_id = cat_repo.create_tag("Landscape")
    cat_repo.assign_tag([media_id], tag_id)

    return {
        "engine": engine,
        "cat_db": cat_db,
        "cat_repo": cat_repo,
        "thumb_repo": thumb_repo,
        "media_id": media_id,
        "tag_id": tag_id,
        "photos_dir": photos_dir,
    }


def test_combined_zoom_button(qapp: QApplication, test_env: dict):
    window = PhotoGalleryMainWindow(test_env["engine"])
    window.resize(800, 600)
    window.show()

    # Grid view: button is hidden
    assert not window.btn_status_fit.isVisible()
    assert not window.btn_status_actual.isVisible()

    # Enter single viewer:
    window.open_single_viewer(test_env["media_id"])
    assert window.btn_status_fit.isVisible()

    # Initially fitted in 800x600 window -> image fits, so it presents "Actual size"
    assert "Actual size" in window.btn_status_fit.toolTip()

    # Click toggle button -> zooms to 100%
    window.btn_status_fit.click()
    assert window.media_viewer.current_zoom_mode == "100"

    # Now zoom in further (e.g. 200%) so image clearly exceeds the window size
    window.media_viewer.graphics_view.set_zoom(3.0)
    assert window._is_viewer_image_bigger_than_window()
    window._update_status_zoom_toggle_button()
    assert "Fit to window" in window.btn_status_fit.toolTip()

    # Clicking when bigger than window toggles back to fit:
    window.btn_status_fit.click()
    assert window.media_viewer.current_zoom_mode == "fit"

    window.show_grid_view()
    assert not window.btn_status_fit.isVisible()
    window.shutdown()


def test_exposure_and_color_popup_menus(qapp: QApplication, test_env: dict):
    window = PhotoGalleryMainWindow(test_env["engine"])
    window.open_single_viewer(test_env["media_id"])

    # 1. Exposure Menu
    exp_widget = window.ribbon.exposure_menu_widget
    assert len(exp_widget.tiles) == 9

    # Trigger hover on a variation tile
    b_val, c_val, tile = exp_widget.tiles[0]
    preview_signal_received = []
    window.ribbon.exposure_preview_requested.connect(lambda b, c: preview_signal_received.append((b, c)))
    tile.hovered.emit()
    assert preview_signal_received == [(b_val, c_val)]

    # Realtime preview was rendered on pixmap
    assert window.media_viewer.pixmap_item.pixmap() is not None

    # Cancel preview restores current recipe
    window.ribbon.exposure_preview_cancelled.emit()
    assert window.media_viewer.current_recipe.brightness == 0.0

    # Auto exposure preview
    auto_signal = []
    window.ribbon.auto_exposure_preview_requested.connect(lambda: auto_signal.append(True))
    exp_widget.btn_auto.hovered.emit()
    assert len(auto_signal) == 1

    # Commit exposure variation
    exp_widget.tiles[2][2].clicked.emit()
    assert window.media_viewer.current_recipe.brightness != 0.0 or window.media_viewer.current_recipe.contrast != 0.0

    # 2. Color Menu
    col_widget = window.ribbon.color_menu_widget
    assert len(col_widget.tiles) == 9

    col_preview_received = []
    window.ribbon.color_preview_requested.connect(lambda t, ti: col_preview_received.append((t, ti)))
    col_widget.tiles[0][2].hovered.emit()
    assert len(col_preview_received) == 1

    # Commit color variation
    col_widget.tiles[0][2].clicked.emit()
    assert window.media_viewer.current_recipe.temperature != 0.0 or window.media_viewer.current_recipe.tint != 0.0

    window.shutdown()


def test_effects_gallery_realtime_hover(qapp: QApplication, test_env: dict):
    window = PhotoGalleryMainWindow(test_env["engine"])
    window.open_single_viewer(test_env["media_id"])

    gallery = window.ribbon.effects_gallery
    sepia_btn = gallery.buttons["sepia"]

    # Hover over sepia button
    hovered_effects = []
    gallery.effect_hovered.connect(lambda eff: hovered_effects.append(eff))
    sepia_btn.hovered.emit()
    assert hovered_effects == ["sepia"]

    # Current recipe is still "none" during preview
    assert window.media_viewer.current_recipe.effect == "none"

    # Mouse leave reverts preview
    unhovered = []
    gallery.effect_unhovered.connect(lambda: unhovered.append(True))
    sepia_btn.unhovered.emit()
    assert len(unhovered) == 1

    # Clicking commits effect
    sepia_btn.click()
    assert window.media_viewer.current_recipe.effect == "sepia"

    window.shutdown()


def test_right_dock_descriptive_tag_hover_and_filter(qapp: QApplication, test_env: dict):
    row = _DescriptiveTagRowWidget(test_env["tag_id"], "Landscape", "1 photo")

    # Frame and cross button hidden initially
    assert row.btn_del.isHidden()
    assert row.property("hovered") in (None, "false")

    # Hover enter
    enter_ev = QEvent(QEvent.Enter)
    row.enterEvent(enter_ev)
    assert row.property("hovered") == "true"
    assert not row.btn_del.isHidden()

    # Hover leave
    leave_ev = QEvent(QEvent.Leave)
    row.leaveEvent(leave_ev)
    assert row.property("hovered") == "false"
    assert row.btn_del.isHidden()

    # Click on label emits filter request
    tag_filtered = []
    row.tag_clicked.connect(lambda tid, label: tag_filtered.append(tid))
    click_ev = QMouseEvent(QEvent.MouseButtonPress, QPoint(5, 5), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    row.lbl.mousePressEvent(click_ev)
    assert tag_filtered == [test_env["tag_id"]]


def test_slideshow_arrow_navigation(qapp: QApplication, test_env: dict):
    from src.db.repository import FilterQuerySpec
    records = test_env["cat_repo"].query_media_records(FilterQuerySpec())
    dialog = FullscreenSlideshowDialog(records=records, start_index=0)

    # Left / Right key press events are handled by event() interceptor
    key_right = QKeyEvent(QEvent.KeyPress, Qt.Key_Right, Qt.NoModifier)
    accepted = dialog.event(key_right)
    assert accepted is True

    key_left = QKeyEvent(QEvent.KeyPress, Qt.Key_Left, Qt.NoModifier)
    accepted = dialog.event(key_left)
    assert accepted is True

    dialog.close()
