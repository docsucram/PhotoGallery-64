"""
Tests for Phase 20 Fixes:
1. WLPG & Photoshop-Style Crop Tool (Rule of Thirds, Dimmed Mask, 8 Handles, Rotation Zone, Resolution Badge).
2. Left Side Navigation Dock Visibility Preservation across Single Image Viewer Navigation & App Restarts.
3. Ribbon Quick Access Toolbar Header Title Deduplication.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path
import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QCursor, QImage, QPainter
from PySide6.QtWidgets import QApplication

from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository, MediaItemRecord
from src.ui.crop_tool import (
    CropOverlayItem,
    CropScene,
    get_rotation_cursor,
    make_curved_rotation_cursor,
)
from src.ui.main_window import PhotoGalleryMainWindow
from src.ui.media_viewer import SingleMediaViewer


@pytest.fixture(scope="session")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def temp_gallery_env():
    tmp_dir = tempfile.mkdtemp(prefix="pg64_phase20_")
    db_dir = os.path.join(tmp_dir, "db")
    media_dir = os.path.join(tmp_dir, "photos")
    os.makedirs(db_dir, exist_ok=True)
    os.makedirs(media_dir, exist_ok=True)

    catalog_path = os.path.join(db_dir, "catalog.db")
    thumbs_path = os.path.join(db_dir, "thumbnails.db")
    engine = DualDatabaseEngine(catalog_path, thumbs_path)

    yield {
        "root": tmp_dir,
        "media_dir": media_dir,
        "engine": engine,
        "catalog_path": catalog_path,
        "thumbs_path": thumbs_path,
    }

    try:
        engine.close_all()
    except Exception:
        pass
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_crop_overlay_visuals_and_hit_test(qapp: QApplication) -> None:
    """Verifies 8 handles, Rule of Thirds, outer dimming, and Photoshop rotation zone hit testing."""
    scene = CropScene()
    scene.setSceneRect(QRectF(0, 0, 1920, 1080))
    item = scene.crop_rect_item
    item.setRect(QRectF(200, 200, 600, 400))
    item.show()

    # Hit test 8 handles
    action_tl, cur_tl = item.hit_test(QPointF(200, 200))
    assert action_tl == "tl"
    assert cur_tl.shape() == Qt.SizeFDiagCursor

    action_br, cur_br = item.hit_test(QPointF(800, 600))
    assert action_br == "br"
    assert cur_br.shape() == Qt.SizeFDiagCursor

    action_tr, cur_tr = item.hit_test(QPointF(800, 200))
    assert action_tr == "tr"
    assert cur_tr.shape() == Qt.SizeBDiagCursor

    action_bl, cur_bl = item.hit_test(QPointF(200, 600))
    assert action_bl == "bl"
    assert cur_bl.shape() == Qt.SizeBDiagCursor

    action_tc, cur_tc = item.hit_test(QPointF(500, 200))
    assert action_tc == "tc"
    assert cur_tc.shape() == Qt.SizeVerCursor

    action_bc, cur_bc = item.hit_test(QPointF(500, 600))
    assert action_bc == "bc"
    assert cur_bc.shape() == Qt.SizeVerCursor

    action_lc, cur_lc = item.hit_test(QPointF(200, 400))
    assert action_lc == "lc"
    assert cur_lc.shape() == Qt.SizeHorCursor

    action_rc, cur_rc = item.hit_test(QPointF(800, 400))
    assert action_rc == "rc"
    assert cur_rc.shape() == Qt.SizeHorCursor

    # Hit test inside -> move
    action_in, cur_in = item.hit_test(QPointF(500, 400))
    assert action_in == "move"
    assert cur_in.shape() == Qt.SizeAllCursor

    # Hit test Photoshop rotation zone (outside corner within 10-52px)
    action_rot, cur_rot = item.hit_test(QPointF(175, 175))
    assert action_rot == "rotate"
    assert cur_rot is not None

    # Hit test far outside -> outside (cross cursor)
    action_out, cur_out = item.hit_test(QPointF(50, 50))
    assert action_out == "outside"
    assert cur_out.shape() == Qt.CrossCursor


def test_crop_overlay_paint_rendering(qapp: QApplication) -> None:
    """Verifies that CropOverlayItem paints dimmed mask, grid, handles, and resolution badge without error."""
    scene = CropScene()
    scene.setSceneRect(QRectF(0, 0, 800, 600))
    item = scene.crop_rect_item
    item.setRect(QRectF(100, 100, 400, 300))
    item.set_master_scale(2.0)
    item.set_rotation_deg(15.5)
    item.show()

    img = QImage(800, 600, QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.white)
    painter = QPainter(img)
    scene.render(painter)
    painter.end()

    # The region outside crop rect (e.g. at 50, 50) should have dark dimmed overlay
    p_out = img.pixelColor(50, 50)
    # The region inside crop rect (e.g. at 250, 250) should remain clear/white
    p_in = img.pixelColor(250, 250)
    assert p_out.red() < 255  # Dimmed by black mask
    assert p_in.red() == 255  # Un-dimmed inside crop area


def test_crop_rotation_and_resolution_in_media_viewer(qapp: QApplication, tmp_path: Path) -> None:
    """Verifies crop rotation applying to straighten_deg and resolution scaling in SingleMediaViewer."""
    viewer = SingleMediaViewer()
    test_img = tmp_path / "crop_rot_test.jpg"
    im = QImage(640, 480, QImage.Format_RGB888)
    im.fill(Qt.green)
    im.save(str(test_img))

    ok = viewer.load_media(media_id=1, file_path=str(test_img))
    assert ok is True

    viewer.set_crop_mode(True)
    assert viewer.is_crop_mode is True
    assert viewer.scene.crop_rect_item.isVisible() is True

    # Set crop rect and simulate rotation angle
    viewer.scene.crop_rect_item.setRect(QRectF(50, 50, 300, 200))
    viewer.scene.crop_rotation_deg = 5.0

    # Apply crop
    ok_crop = viewer.apply_crop()
    assert ok_crop is True
    assert viewer.is_crop_mode is False
    assert viewer.current_recipe.crop_rect is not None
    assert abs(viewer.current_recipe.straighten_deg - 5.0) < 0.01


def test_left_nav_dock_preservation_across_viewer_navigation(qapp: QApplication, temp_gallery_env) -> None:
    """Verifies that navigating between photos in viewer mode never hides left navigation dock in grid view."""
    engine = temp_gallery_env["engine"]
    media_dir = temp_gallery_env["media_dir"]
    repo = CatalogRepository(engine)

    p1 = os.path.join(media_dir, "img1.jpg")
    p2 = os.path.join(media_dir, "img2.jpg")
    for p in (p1, p2):
        im = QImage(100, 100, QImage.Format_RGB888)
        im.fill(Qt.blue)
        im.save(p)

    fid = repo.upsert_folder(media_dir, "Photos", parent_id=None, mtime=int(time.time()))
    repo.bulk_upsert_media([
        {
            "folder_id": fid,
            "filename": "img1.jpg",
            "ext": "jpg",
            "size": 100,
            "mtime": int(time.time()),
            "date_taken": int(time.time()),
            "is_video": 0,
            "width": 100,
            "height": 100,
        },
        {
            "folder_id": fid,
            "filename": "img2.jpg",
            "ext": "jpg",
            "size": 100,
            "mtime": int(time.time()),
            "date_taken": int(time.time()),
            "is_video": 0,
            "width": 100,
            "height": 100,
        },
    ])

    win = PhotoGalleryMainWindow(engine=engine)
    win.show()

    # Initial state: in grid view, left_nav_dock is visible
    assert not win.left_nav_dock.isHidden()

    # Open photo 1 in viewer
    win.open_single_viewer(1)
    assert win.central_stack.currentWidget() == win.media_viewer
    assert win.left_nav_dock.isHidden()
    assert win._nav_dock_was_visible is True

    # Navigate to photo 2 while remaining in viewer mode
    win.open_single_viewer(2)
    assert win.central_stack.currentWidget() == win.media_viewer
    assert win.left_nav_dock.isHidden()
    # Crucial assertion: _nav_dock_was_visible must NOT be corrupted to False!
    assert win._nav_dock_was_visible is True

    # Return to grid view
    win.show_grid_view()
    assert win.central_stack.currentWidget() == win.grid_view
    assert not win.left_nav_dock.isHidden()

    win.shutdown()


def test_ribbon_qat_header_title_deduplication(qapp: QApplication, temp_gallery_env) -> None:
    """Verifies that ribbon top header does not display a duplicated window title next to the QAT."""
    engine = temp_gallery_env["engine"]
    win = PhotoGalleryMainWindow(engine=engine)
    # Both lbl_app_title and header_div must be hidden
    assert win.ribbon.lbl_app_title.isHidden()
    assert win.ribbon.header_div.isHidden()
    win.shutdown()


def test_left_nav_dock_minimum_width_and_resize(qapp: QApplication, temp_gallery_env) -> None:
    """Verifies that LeftNavigationDock enforces minimum width 200px and expands via resizeDocks."""
    engine = temp_gallery_env["engine"]
    win = PhotoGalleryMainWindow(engine=engine)
    win.show()
    qapp.processEvents()

    assert win.left_nav_dock.minimumWidth() >= 200
    assert win.left_nav_dock.widget().minimumWidth() >= 200
    assert win.left_nav_dock.width() >= 200
    win.shutdown()


def test_search_scope_bar_card_styling_and_geometry(qapp: QApplication, temp_gallery_env) -> None:
    """Verifies that SearchScopeBar is styled as a white card with proper padding and not obscured."""
    engine = temp_gallery_env["engine"]
    win = PhotoGalleryMainWindow(engine=engine)
    win.show()
    win.resize(1024, 768)
    qapp.processEvents()

    win.toggle_search_scope_bar(force_visible=True)
    qapp.processEvents()

    scope_bar = win.scope_bar
    assert scope_bar.isVisible()
    # Check that it has positive margins on left and right
    geom = scope_bar.geometry()
    assert geom.left() >= 10
    assert geom.right() < win.grid_view.width()
    assert scope_bar.btn_close.isVisible()
    assert scope_bar.btn_pin.isVisible()
    win.shutdown()


def test_ribbon_corner_no_options_and_slideshow(qapp: QApplication, temp_gallery_env) -> None:
    """Verifies no Options button in ribbon tab corner, and slideshow dialog executes without crash."""
    from src.ui.slideshow_view import FullscreenSlideshowDialog

    engine = temp_gallery_env["engine"]
    win = PhotoGalleryMainWindow(engine=engine)
    # Options button should not exist on ribbon
    assert not hasattr(win.ribbon, "btn_corner_options")

    # FullscreenSlideshowDialog creation and immediate accept
    records = win.catalog_model._records
    dlg = FullscreenSlideshowDialog(records=records, parent=win)
    dlg.show()
    qapp.processEvents()
    dlg.accept()
    qapp.processEvents()
    # current_media_id should be callable without C++ deletion error
    assert dlg.current_media_id() is None or isinstance(dlg.current_media_id(), int)
    win.shutdown()
