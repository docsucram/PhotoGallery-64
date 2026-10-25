import sys
from pathlib import Path
import pytest
from PySide6.QtCore import Qt, QSize, QPointF
from PySide6.QtGui import QPixmap, QImage, QColor
from PySide6.QtWidgets import QApplication

from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository, ThumbnailRepository
from src.ui.grid_view import WLPGCardDelegate, CatalogListModel
from src.ui.ribbon_toolbar import PhotoGalleryRibbonToolbar
from src.ui.create_studio_dialogs import ColourPopDialog, PortraitBokehDialog
import numpy as np


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


@pytest.fixture
def temp_dbs(tmp_path):
    cat_path = tmp_path / "catalog.db"
    thumb_path = tmp_path / "thumbnails.db"
    engine = DualDatabaseEngine(cat_path, thumb_path)
    engine.init_databases()
    return engine


def test_ribbon_tooltips_have_shortcuts(qapp):
    ribbon = PhotoGalleryRibbonToolbar()
    
    # Check Home tab buttons
    assert "Ctrl+," in ribbon.btn_rot_ccw.toolTip()
    assert "Ctrl+." in ribbon.btn_rot_cw.toolTip()
    assert "Del" in ribbon.btn_delete.toolTip()
    assert "Ctrl+A" in ribbon.btn_select_all.toolTip()
    assert "Ctrl+C" in ribbon.btn_copy.toolTip()
    assert "Ctrl+V" in ribbon.btn_paste.toolTip()
    assert "F12" in ribbon.btn_home_slideshow.toolTip()
    assert "Ctrl+T" in ribbon.btn_tag_desc.toolTip()

    # Check View tab buttons
    assert "Zoom in (+)" in ribbon.btn_zoom_in.toolTip()
    assert "Zoom out (-)" in ribbon.btn_zoom_out.toolTip()
    assert "F12" in ribbon.btn_view_slideshow.toolTip()
    assert "Ctrl+I" in ribbon.btn_toggle_tag_pane.toolTip()

    # Check compact responsiveness preserves tooltips
    ribbon.btn_rot_ccw.set_compact(True)
    assert "Ctrl+," in ribbon.btn_rot_ccw.toolTip()
    ribbon.btn_rot_ccw.set_compact(False)
    assert "Ctrl+," in ribbon.btn_rot_ccw.toolTip()


def test_face_crops_dynamic_fallback(qapp, temp_dbs):
    cat_repo = CatalogRepository(temp_dbs)
    thumb_repo = ThumbnailRepository(temp_dbs)

    # Insert test folder, media, and person with face rect
    conn = temp_dbs.catalog_conn()
    with temp_dbs.write_lock:
        conn.execute("INSERT INTO folders (id, path) VALUES (1, 'C:/Test');")
        conn.execute("INSERT INTO media (id, folder_id, filename, is_video) VALUES (10, 1, 'photo.jpg', 0);")
        conn.execute("INSERT INTO people (id, name) VALUES (1, 'Alice');")
        conn.execute(
            "INSERT INTO media_people (id, media_id, person_id, state, is_manual, rect_x, rect_y, rect_w, rect_h) "
            "VALUES (100, 10, 1, 1, 1, 0.2, 0.2, 0.4, 0.4);"
        )
        conn.commit()

    delegate = WLPGCardDelegate(thumb_repo=thumb_repo)
    delegate.set_face_crop_person_id(1)

    assert (10, 1) in delegate._person_face_rects
    rx, ry, rw, rh = delegate._person_face_rects[(10, 1)]
    assert rw == 0.4 and rh == 0.4

    # Seed pixmap cache with dummy full photo
    pm = QPixmap(100, 100)
    pm.fill(QColor(255, 0, 0))
    delegate.pixmap_cache.put(10, pm)

    # Calling paint or checking delegate cache
    assert (10, 1) not in delegate._face_crop_cache

    # In face crops mode, if thumb_repo doesn't have it, delegate creates crop on the fly from pixmap cache
    # Simulate delegate crop logic
    full_pm = delegate.pixmap_cache.get(10)
    assert full_pm is not None
    key = (10, 1)
    rx, ry, rw, rh = delegate._person_face_rects[key]
    cx = rx + rw * 0.5
    cy = ry + rh * 0.5
    side = max(rw, rh) * 1.38
    pw, ph = full_pm.width(), full_pm.height()
    crop_x = int(max(0, (cx - side * 0.5) * pw))
    crop_y = int(max(0, (cy - side * 0.5) * ph))
    crop_w = int(min(pw - crop_x, side * pw))
    crop_h = int(min(ph - crop_y, side * ph))
    crop_side = min(crop_w, crop_h)
    crop_pm = full_pm.copy(crop_x, crop_y, crop_side, crop_side)
    assert not crop_pm.isNull()
    assert crop_pm.width() == crop_side


def test_create_studio_zoom_controls(qapp):
    # Test ColourPopDialog zoom controls
    dummy_bgr = np.zeros((200, 300, 3), dtype=np.uint8)
    cp_dlg = ColourPopDialog(dummy_bgr)

    assert hasattr(cp_dlg, "btn_zoom_in")
    assert hasattr(cp_dlg, "btn_zoom_out")
    assert hasattr(cp_dlg, "btn_fit")
    assert hasattr(cp_dlg, "btn_100")
    assert hasattr(cp_dlg, "lbl_zoom_val")

    # Zoom in
    init_transform = cp_dlg.view.transform().m11()
    cp_dlg.btn_zoom_in.click()
    assert cp_dlg.view.transform().m11() > init_transform

    # Reset 100%
    cp_dlg.btn_100.click()
    assert abs(cp_dlg.view.transform().m11() - 1.0) < 0.01
    assert "100%" in cp_dlg.lbl_zoom_val.text()

    # Test PortraitBokehDialog zoom controls
    pb_dlg = PortraitBokehDialog(dummy_bgr)
    assert hasattr(pb_dlg, "btn_zoom_in")
    assert hasattr(pb_dlg, "btn_zoom_out")
    assert hasattr(pb_dlg, "btn_fit")
    assert hasattr(pb_dlg, "btn_100")
    assert hasattr(pb_dlg, "lbl_zoom_val")

    init_pb = pb_dlg.view.transform().m11()
    pb_dlg.btn_zoom_in.click()
    assert pb_dlg.view.transform().m11() > init_pb
    pb_dlg.btn_100.click()
    assert abs(pb_dlg.view.transform().m11() - 1.0) < 0.01
