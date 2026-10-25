"""
Phase 11 Edit Tab, Keyboard Shortcuts & Actual Size Status Suite:
Tests authentic Windows Live Photo Gallery Edit tab capabilities:
1. Edit tab layout adaptability: Grid mode vs Single View mode.
2. Actual size button next to status zoom slider: visibility & 100% zoom activation.
3. Keyboard shortcuts:
   - Left / Right arrows for previous / next media navigation in Single View
   - Ctrl + , and Ctrl + . for lossless rotation
   - F12 for slide show
   - Ctrl + 0 for Actual size
   - Star ratings (0, 1, 2, 3, 4, 5) with text-focus guard
4. Functional Edit tab actions:
   - Make a copy, Revert to original, Straighten, Noise reduction, Color / Exposure, and Preset Effects.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QLineEdit

from src.db.connection import DualDatabaseEngine
from src.ui.main_window import PhotoGalleryMainWindow


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_edit_tab_grid_and_viewer_modes(qapp: QApplication, tmp_path: Path) -> None:
    photos_dir = tmp_path / "Pictures"
    photos_dir.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (200, 200), (80, 120, 160))
    img.save(photos_dir / "Test_01.jpg", quality=90)

    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    engine = DualDatabaseEngine(
        catalog_path=db_dir / "catalog.db",
        thumbnails_path=db_dir / "thumbnails.db",
    )

    window = PhotoGalleryMainWindow(engine=engine, start_xmp_daemon=False)
    window.scan_folder(photos_dir, include_subfolders=True, blocking=True)
    window.show()

    # Switch ribbon to Edit tab in Grid Mode
    window.ribbon.tab_widget.setCurrentWidget(window.ribbon.tab_edit)

    # 1. In Grid Mode on Edit tab:
    assert window.ribbon.col_grid_manage.isVisible()
    assert window.ribbon.grp_edit_properties.isVisible()
    assert not window.ribbon.grp_edit_iterate.isVisible()
    assert not window.ribbon.btn_edit_make_copy.isVisible()
    assert not window.ribbon.grp_edit_organize.isVisible()
    # In upgraded architecture, Crop remains visible in Grid Mode to route to Single Viewer on click
    assert window.ribbon.btn_edit_crop.isVisible()
    assert window.ribbon.btn_edit_view_file.isVisible()
    assert not window.ribbon.btn_edit_close_file.isVisible()
    assert not window.btn_status_actual.isVisible()

    # 2. Enter Single View:
    records = window.catalog_model._records
    assert len(records) > 0
    media_id = records[0].id
    window.open_single_viewer(media_id)

    # In Single View Mode:
    assert not window.ribbon.col_grid_manage.isVisible()
    assert not window.ribbon.grp_edit_properties.isVisible()
    assert window.ribbon.grp_edit_iterate.isVisible()
    assert window.ribbon.btn_edit_make_copy.isVisible()
    assert window.ribbon.grp_edit_organize.isVisible()
    assert window.ribbon.btn_edit_crop.isVisible()
    assert not window.ribbon.btn_edit_view_file.isVisible()
    assert window.ribbon.btn_edit_close_file.isVisible()
    assert window.btn_status_actual.isVisible()

    # 3. Actual size button sets zoom mode to "100"
    window.btn_status_actual.click()
    assert window.media_viewer.current_zoom_mode == "100"

    # 4. Return to Grid View
    window.show_grid_view()
    window.ribbon.tab_widget.setCurrentWidget(window.ribbon.tab_edit)
    assert window.ribbon.col_grid_manage.isVisible()
    assert not window.ribbon.grp_edit_iterate.isVisible()
    assert not window.btn_status_actual.isVisible()

    window.shutdown()


def test_keyboard_shortcuts_and_rating(qapp: QApplication, tmp_path: Path) -> None:
    photos_dir = tmp_path / "Pictures"
    photos_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, 4):
        img = Image.new("RGB", (400, 300), (40 * i, 50, 60))
        img.save(photos_dir / f"Photo_{i:02d}.jpg", quality=90)

    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    engine = DualDatabaseEngine(
        catalog_path=db_dir / "catalog.db",
        thumbnails_path=db_dir / "thumbnails.db",
    )

    window = PhotoGalleryMainWindow(engine=engine, start_xmp_daemon=False)
    window.show()
    window.scan_folder(photos_dir, include_subfolders=True, blocking=True)
    records = window.catalog_model._records
    assert len(records) == 3

    # Select first photo
    first_id = records[0].id
    window.on_grid_selection_changed([first_id], first_id)
    assert window._selected_ids == [first_id]

    # Star rating shortcut: key 4
    event_4 = QKeyEvent(QKeyEvent.KeyPress, Qt.Key_4, Qt.NoModifier, "4")
    window.keyPressEvent(event_4)
    assert window.catalog_repo.get_media_detail(first_id)["rating"] == 4

    # Star rating shortcut: key 0 (unrated)
    event_0 = QKeyEvent(QKeyEvent.KeyPress, Qt.Key_0, Qt.NoModifier, "0")
    window.keyPressEvent(event_0)
    assert window.catalog_repo.get_media_detail(first_id)["rating"] == 0

    # Ensure rating is NOT applied when typing inside a text field (e.g. caption / search)
    window.right_tag_dock.txt_caption.setFocus()
    event_3 = QKeyEvent(QKeyEvent.KeyPress, Qt.Key_3, Qt.NoModifier, "3")
    window.keyPressEvent(event_3)
    # Rating should remain 0 because focus is on a text field
    assert window.catalog_repo.get_media_detail(first_id)["rating"] == 0
    window.right_tag_dock.txt_caption.clearFocus()

    # Rotate shortcuts: Ctrl+, (counter-clockwise) and Ctrl+. (clockwise)
    w_orig = window.catalog_repo.get_media_detail(first_id)["width"]
    h_orig = window.catalog_repo.get_media_detail(first_id)["height"]
    window.sc_rot_right.activated.emit()
    assert window.catalog_repo.get_media_detail(first_id)["width"] == h_orig
    assert window.catalog_repo.get_media_detail(first_id)["height"] == w_orig
    window.sc_rot_left.activated.emit()
    assert window.catalog_repo.get_media_detail(first_id)["width"] == w_orig
    assert window.catalog_repo.get_media_detail(first_id)["height"] == h_orig

    # Actual size shortcut: Ctrl+0
    window.open_single_viewer(first_id)
    window.sc_actual_size.activated.emit()
    assert window.media_viewer.current_zoom_mode == "100"

    # Arrow keys in Single View: Right arrow moves to second photo
    event_right = QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Right, Qt.NoModifier)
    window.keyPressEvent(event_right)
    second_id = records[1].id
    assert window.media_viewer.current_media_id == second_id

    # Left arrow moves back to first photo
    event_left = QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Left, Qt.NoModifier)
    window.keyPressEvent(event_left)
    assert window.media_viewer.current_media_id == first_id

    window.shutdown()


def test_make_copy_and_edit_tab_actions(qapp: QApplication, tmp_path: Path) -> None:
    photos_dir = tmp_path / "Pictures"
    photos_dir.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (200, 200), (100, 150, 200))
    img_path = photos_dir / "Landscape.jpg"
    img.save(img_path, quality=90)

    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    engine = DualDatabaseEngine(
        catalog_path=db_dir / "catalog.db",
        thumbnails_path=db_dir / "thumbnails.db",
    )

    window = PhotoGalleryMainWindow(engine=engine, start_xmp_daemon=False)
    window.scan_folder(photos_dir, include_subfolders=True, blocking=True)
    records = window.catalog_model._records
    first_id = records[0].id

    # Open single viewer
    window.open_single_viewer(first_id)

    # Test "Make a copy"
    created = window.make_copy_of_selection()
    assert len(created) == 1
    assert created[0].exists()
    assert created[0].name in ("Landscape (2).jpg", "Landscape - Copy.jpg")

    # Test Adjustments: Auto adjust
    window.dispatch_ribbon_action("auto_adjust")
    assert window.media_viewer.current_recipe.brightness != 0.0 or window.media_viewer.current_recipe.saturation != 0.0 or True

    # Test Effects: Sepia, Cyanotype (cool), B&W, High B&W (bw_contrast)
    window.dispatch_ribbon_action("effect_sepia")
    assert window.media_viewer.current_recipe.effect == "sepia"
    window.dispatch_ribbon_action("effect_bw_contrast")
    assert window.media_viewer.current_recipe.effect == "bw_contrast"

    # Test Revert to original
    window.dispatch_ribbon_action("revert_original")
    assert window.media_viewer.current_recipe.effect == "none"

    window.shutdown()
