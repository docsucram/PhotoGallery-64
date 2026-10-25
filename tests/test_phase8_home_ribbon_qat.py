"""
Phase 8 Home Tab & Quick Access Toolbar (QAT) Verification Suite:
Tests authentic Windows Live Photo Gallery top-level capabilities:
1. Quick Access Toolbar (QAT) Navigation History (Back / Forward buttons & state transitions).
2. QUndoStack integration for Lossless Rotation, Ratings, Flags, and Captions (Undo / Redo buttons & state transitions).
3. Ribbon Home Tab Actions: Select All, Copy to Clipboard (files & URLs), Toggle Flag, Rate 0..5, Month filtering, and Compact Face Carousel.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication

from src.db.connection import DualDatabaseEngine
from src.ui.main_window import PhotoGalleryMainWindow


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    from PySide6.QtCore import QSettings
    QSettings("PhotoGallery64", "PhotoGallery64").remove("filter_state")
    return app


def test_qat_navigation_history(qapp: QApplication, tmp_path: Path) -> None:
    photos_dir = tmp_path / "Pictures"
    photos_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(1, 4):
        img = Image.new("RGB", (200, 200), (50 * idx, 60, 70))
        img.save(photos_dir / f"Photo_{idx:02d}.jpg", quality=90)

    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    engine = DualDatabaseEngine(
        catalog_path=db_dir / "catalog.db",
        thumbnails_path=db_dir / "thumbnails.db",
    )

    window = PhotoGalleryMainWindow(engine=engine, start_xmp_daemon=False)

    window.scan_folder(photos_dir, include_subfolders=True, blocking=True)

    # Initial state: no back or forward history
    assert not window.ribbon.btn_back.isEnabled()
    assert not window.ribbon.btn_forward.isEnabled()

    # Step 1: Change filter state (e.g. filter by flagged)
    window.left_nav_dock.set_flagged_only(True)
    assert window.ribbon.btn_back.isEnabled()
    assert not window.ribbon.btn_forward.isEnabled()

    # Step 2: Change filter state again (select month 5)
    window.left_nav_dock.select_month(5)
    assert window.ribbon.btn_back.isEnabled()
    assert not window.ribbon.btn_forward.isEnabled()

    # Step 3: Navigate Back
    window.navigate_history_back()
    assert window.left_nav_dock._date_month is None
    assert window.left_nav_dock._flagged_only is True
    assert window.ribbon.btn_back.isEnabled()
    assert window.ribbon.btn_forward.isEnabled()

    # Step 4: Navigate Back to initial state
    window.navigate_history_back()
    assert window.left_nav_dock._flagged_only is False
    assert not window.ribbon.btn_back.isEnabled()
    assert window.ribbon.btn_forward.isEnabled()

    # Step 5: Navigate Forward
    window.navigate_history_forward()
    assert window.left_nav_dock._flagged_only is True
    assert window.ribbon.btn_back.isEnabled()
    assert window.ribbon.btn_forward.isEnabled()

    window.navigate_history_forward()
    assert window.left_nav_dock._date_month == 5
    assert window.ribbon.btn_back.isEnabled()
    assert not window.ribbon.btn_forward.isEnabled()

    window.shutdown()


def test_qat_undo_redo_stack(qapp: QApplication, tmp_path: Path) -> None:
    photos_dir = tmp_path / "Pictures"
    photos_dir.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGB", (400, 300), (100, 150, 200))
    img.save(photos_dir / "Test_Undo.jpg", quality=90)

    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    engine = DualDatabaseEngine(
        catalog_path=db_dir / "catalog.db",
        thumbnails_path=db_dir / "thumbnails.db",
    )

    window = PhotoGalleryMainWindow(engine=engine, start_xmp_daemon=False)

    window.scan_folder(photos_dir, include_subfolders=True, blocking=True)

    all_ids = [r.id for r in window.catalog_model._records]
    assert len(all_ids) == 1
    media_id = all_ids[0]

    # Select the photo
    window.grid_view.on_card_checkbox_toggled(media_id, True)

    # Initial undo/redo state
    assert not window.ribbon.btn_undo.isEnabled()
    assert not window.ribbon.btn_redo.isEnabled()

    # 1. Rating Undo / Redo
    window.dispatch_ribbon_action("rate_4")
    assert window.catalog_repo.get_media_detail(media_id)["rating"] == 4
    assert window.ribbon.btn_undo.isEnabled()
    assert not window.ribbon.btn_redo.isEnabled()

    window.undo_stack.undo()
    assert window.catalog_repo.get_media_detail(media_id)["rating"] == 0
    assert not window.ribbon.btn_undo.isEnabled()
    assert window.ribbon.btn_redo.isEnabled()

    window.undo_stack.redo()
    assert window.catalog_repo.get_media_detail(media_id)["rating"] == 4
    assert window.ribbon.btn_undo.isEnabled()
    assert not window.ribbon.btn_redo.isEnabled()

    # 2. Flag Undo / Redo
    window.dispatch_ribbon_action("toggle_flag")
    assert window.catalog_repo.get_media_detail(media_id)["flag"] == 1
    window.undo_stack.undo()
    assert window.catalog_repo.get_media_detail(media_id)["flag"] == 0

    # 3. Caption Undo / Redo
    window.apply_caption_to_selection("WLPG Scenic View")
    assert window.catalog_repo.get_media_detail(media_id)["caption"] == "WLPG Scenic View"
    window.undo_stack.undo()
    assert window.catalog_repo.get_media_detail(media_id)["caption"] == ""

    # 4. Lossless Rotation Undo / Redo
    detail_before = window.catalog_repo.get_media_detail(media_id)
    w_before, h_before = detail_before["width"], detail_before["height"]
    window.dispatch_ribbon_action("rotate_right")

    detail_after = window.catalog_repo.get_media_detail(media_id)
    assert detail_after["width"] == h_before
    assert detail_after["height"] == w_before

    window.undo_stack.undo()
    detail_reverted = window.catalog_repo.get_media_detail(media_id)
    assert detail_reverted["width"] == w_before
    assert detail_reverted["height"] == h_before

    window.shutdown()


def test_ribbon_home_actions_and_quick_find(qapp: QApplication, tmp_path: Path) -> None:
    photos_dir = tmp_path / "Pictures"
    photos_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(1, 4):
        img = Image.new("RGB", (300, 300), (20 * idx, 40 * idx, 60))
        img.save(photos_dir / f"Photo_{idx:02d}.jpg", quality=90)

    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    engine = DualDatabaseEngine(
        catalog_path=db_dir / "catalog.db",
        thumbnails_path=db_dir / "thumbnails.db",
    )

    window = PhotoGalleryMainWindow(engine=engine, start_xmp_daemon=False)

    window.scan_folder(photos_dir, include_subfolders=True, blocking=True)

    # 1. Select All Action
    window.dispatch_ribbon_action("select_all")
    assert len(window.grid_view.selected_media_ids()) == 3

    # 2. Copy Action
    window.dispatch_ribbon_action("copy")
    clipboard = QApplication.clipboard()
    mime = clipboard.mimeData()
    urls = mime.urls() if mime.hasUrls() else getattr(window, "_copied_urls", [])
    assert len(urls) == 3

    # 3. Quick Find Month Filter
    window.ribbon.filter_month_selected.emit(11)
    assert window.left_nav_dock._date_month == 11

    window.ribbon.filter_month_selected.emit(0)
    assert window.left_nav_dock._date_month is None

    # 4. Quick Find Face Carousel
    all_ids = [r.id for r in window.catalog_model._records]
    window.catalog_repo.assign_person(all_ids[0], "Tu May Robinson")
    window.refresh_navigation_and_autocomplete()

    assert len(window.ribbon.home_face_carousel._buttons) == 1
    person_btn = list(window.ribbon.home_face_carousel._buttons.values())[0]
    assert person_btn.width() == 54
    assert person_btn.height() == 54
    assert "Tu May Robinson" in person_btn.toolTip()

    # 5. Quick Find Expand Button & WLPG People Menu Options
    assert hasattr(window.ribbon.home_face_carousel, "btn_expand")
    from src.ui.ribbon_toolbar import _WLPGPeopleMenuWidget
    from PySide6.QtWidgets import QMenu
    dummy_menu = QMenu()
    people_list = window.catalog_repo.get_people_list()
    menu_widget = _WLPGPeopleMenuWidget(dummy_menu, people_list)
    assert hasattr(menu_widget, "btn_and")
    assert hasattr(menu_widget, "btn_or")
    assert hasattr(menu_widget, "btn_all_tagged")
    assert hasattr(menu_widget, "btn_detected")
    assert hasattr(menu_widget, "btn_untagged")

    # 6. Special People Filter Execution
    window._on_people_special_filter("all_people")
    assert window.ribbon.lbl_scope.text() == "All tagged or detected"
    window._on_people_special_filter("untagged")
    assert window.ribbon.lbl_scope.text() == "Not tagged or detected"

    window.shutdown()
