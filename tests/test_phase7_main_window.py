"""
Phase 7 End-to-End Integration Verification Suite:
Tests PhotoGalleryMainWindow coordinating:
1. FolderSyncWorker scanning real JPEG images into catalog.db.
2. Virtualized CatalogListModel + ThumbnailGridView (WLPG Grid & Mosaic layout modes).
3. Inline card star rating & flag toggling + Right TagAndCaptionDock multi-selection tagging.
4. LeftNavigationDock Boolean tag filtering + RibbonToolbar Quick Find face avatar filtering.
5. Relative 'Change Time Taken' across multiple selected photos.
6. SingleMediaViewer open, 'Fix' adjustment, '.OriginalFiles' backup save, and Revert to original.
7. WorldMapView toggle, MetadataSyncDaemon lossless XMP write-back flush, and clean shutdown.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication

from src.db.connection import DualDatabaseEngine
from src.ui.main_window import PhotoGalleryMainWindow
from src.workers.xmp_sync_worker import read_xmp_from_file


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_end_to_end_main_window_workflow(
    qapp: QApplication, tmp_path: Path
) -> None:
    """Full end-to-end integration test of PhotoGalleryMainWindow."""
    photos_dir = tmp_path / "Pictures"
    photos_dir.mkdir(parents=True, exist_ok=True)

    # Create 3 real JPEG images with distinct aspect ratios (landscape, portrait, wide)
    for idx, (w, h, color) in enumerate(
        [
            (400, 300, (60, 120, 200)),
            (300, 400, (200, 100, 80)),
            (480, 270, (80, 170, 110)),
        ],
        start=1,
    ):
        img = Image.new("RGB", (w, h), color)
        img.save(photos_dir / f"Photo_{idx:02d}.jpg", quality=92)

    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    engine = DualDatabaseEngine(
        catalog_path=db_dir / "catalog.db",
        thumbnails_path=db_dir / "thumbnails.db",
    )

    window = PhotoGalleryMainWindow(engine=engine, start_xmp_daemon=False)
    window.resize(1280, 800)

    # 1. Scan sample folder synchronously
    window.scan_folder(photos_dir, include_subfolders=True, blocking=True)
    assert window.catalog_model.total_media_count == 3
    assert "3 items" in window.lbl_status_counts.text()

    all_ids = [r.id for r in window.catalog_model._records]
    assert len(all_ids) == 3

    # 2. Inline card rating & flag toggle
    window.card_delegate.rating_clicked.emit(all_ids[0], 5)
    window.card_delegate.flag_toggled.emit(all_ids[0], 1)
    d0 = window.catalog_repo.get_media_detail(all_ids[0])
    assert d0["rating"] == 5
    assert d0["flag"] == 1
    assert d0["metadata_dirty"] == 1

    # 3. Select first 2 items via checkbox and apply hierarchical tag + person via Right Dock
    window.grid_view.on_card_checkbox_toggled(all_ids[0], True)
    window.grid_view.on_card_checkbox_toggled(all_ids[1], True)
    assert set(window.grid_view.selected_media_ids()) == {all_ids[0], all_ids[1]}
    assert "3 items, 2 selected" in window.lbl_status_counts.text()

    window.right_tag_dock.add_descriptive_tag_text("PeopleTags/Family/Craig")
    window.right_tag_dock.add_person_tag_text("Craig")
    window.right_tag_dock.commit_caption("Family Reunion")

    # 4. Filter via Quick Find Face Carousel in RibbonToolbar
    people = window.catalog_repo.get_people_list()
    assert len(people) == 1
    craig_pid = int(people[0]["id"])
    window.ribbon.home_face_carousel.click_person(craig_pid)
    assert window.catalog_model.total_media_count == 2

    # Clear filter -> back to 3 items
    window.left_nav_dock.clear_all_filters()
    assert window.catalog_model.total_media_count == 3

    # Switch layout mode between WLPG Classic Grid and Tessellated Mosaic
    window.ribbon.set_layout_mode("mosaic")
    assert window.catalog_model.layout_mode == "mosaic"
    window.ribbon.set_layout_mode("wlpg_grid")
    assert window.catalog_model.layout_mode == "wlpg_grid"

    # 5. Relative 'Change Time Taken' across selected items
    window.on_grid_selection_changed([all_ids[0], all_ids[1]], all_ids[0])
    orig_dt0 = int(window.catalog_repo.get_media_detail(all_ids[0])["date_taken"])
    orig_dt1 = int(window.catalog_repo.get_media_detail(all_ids[1])["date_taken"])
    window.open_change_time_taken_dialog(auto_accept_timestamp=orig_dt0 + 3600)

    new_dt0 = int(window.catalog_repo.get_media_detail(all_ids[0])["date_taken"])
    new_dt1 = int(window.catalog_repo.get_media_detail(all_ids[1])["date_taken"])
    assert new_dt0 - orig_dt0 == 3600
    assert new_dt1 - orig_dt1 == 3600

    # 6. Open SingleMediaViewer, apply Auto Adjust, save with .OriginalFiles backup, and revert
    assert window.open_single_viewer(all_ids[0]) is True
    assert window.central_stack.currentWidget() == window.media_viewer
    window.media_viewer.apply_auto_adjust()
    bpath = window.media_viewer.save_current_image_with_backup()
    assert bpath is not None and Path(bpath).exists()

    detail_backed = window.catalog_repo.get_media_detail(all_ids[0])
    assert detail_backed["original_backup_path"] == bpath

    assert window.media_viewer.revert_to_original(bpath) is True
    detail_reverted = window.catalog_repo.get_media_detail(all_ids[0])
    assert detail_reverted["original_backup_path"] is None

    # Navigate next in viewer then return to grid
    assert window.navigate_single_viewer(1) is True
    window.show_grid_view()
    assert window.central_stack.currentWidget() == window.grid_view

    # 7. Toggle WorldMapView and back
    window.dispatch_ribbon_action("toggle_map_view")
    assert window.central_stack.currentWidget() == window.map_view
    window.dispatch_ribbon_action("toggle_map_view")
    assert window.central_stack.currentWidget() == window.grid_view

    # 8. Flush dirty XMP metadata to disk and verify lossless XMP packet on Photo_01.jpg
    synced_count = window.xmp_daemon.flush_now()
    assert synced_count >= 2
    parsed_xmp = read_xmp_from_file(photos_dir / "Photo_01.jpg")
    assert parsed_xmp["rating"] == 5
    assert "PeopleTags/Family/Craig" in parsed_xmp["hierarchical_tags"]
    assert parsed_xmp["caption"] == "Family Reunion"

    # 9. Clean shutdown
    window.shutdown()


def test_right_inspector_dual_modes(qapp: QApplication, tmp_path: Path) -> None:
    """Verify WLPG Right Inspector mutually exclusive toggle between Info (Ctrl+I) and Fine-Tune (Ctrl+F)."""
    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    engine = DualDatabaseEngine(
        catalog_path=db_dir / "catalog.db",
        thumbnails_path=db_dir / "thumbnails.db",
    )
    window = PhotoGalleryMainWindow(engine=engine, start_xmp_daemon=False)
    window.show()

    # Initially right dock is visible in 'info' mode
    assert window.right_tag_dock.current_mode() == "info"
    assert window.right_tag_dock.mode_stack.currentIndex() == 0

    # Toggle info off
    window.toggle_right_pane("info")
    assert window.right_tag_dock.isHidden()

    # Toggle info on
    window.toggle_right_pane("info")
    assert not window.right_tag_dock.isHidden()
    assert window.right_tag_dock.current_mode() == "info"
    assert window.right_tag_dock.mode_stack.currentIndex() == 0

    # Toggle fine-tune on (switches mode without creating a duplicate pane)
    window.toggle_right_pane("fine_tune")
    assert not window.right_tag_dock.isHidden()
    assert window.right_tag_dock.current_mode() == "fine_tune"
    assert window.right_tag_dock.mode_stack.currentIndex() == 1

    # Ribbon actions
    window.dispatch_ribbon_action("toggle_tag_pane")
    assert not window.right_tag_dock.isHidden()
    assert window.right_tag_dock.current_mode() == "info"

    window.dispatch_ribbon_action("fine_tune")
    assert not window.right_tag_dock.isHidden()
    assert window.right_tag_dock.current_mode() == "fine_tune"

    # Returning to grid view resets fine-tune to info
    window.show_grid_view()
    assert window.right_tag_dock.current_mode() == "info"

    window.shutdown()


def test_find_tab_integration_and_ctrl_f_dual_mode(
    tmp_path: Path, qapp: QApplication
) -> None:
    """
    Verify Find Tab integration in PhotoGalleryMainWindow:
    - In Grid mode: Ctrl+F toggles the Search Scope bar
    - In Single Viewer mode: Ctrl+F toggles the Fine-tune right inspector pane
    - Clicking Text search in Find tab opens the search scope bar
    - Ribbon Find tab filter signals update navigation and query records
    """
    test_dir = tmp_path / "GalleryFindTest"
    test_dir.mkdir(parents=True, exist_ok=True)
    img_p1 = test_dir / "sunset.jpg"
    img_p2 = test_dir / "portrait.jpg"

    from PIL import Image
    im = Image.new("RGB", (640, 480), color=(255, 120, 0))
    im.save(img_p1, "JPEG")
    im2 = Image.new("RGB", (480, 640), color=(0, 150, 255))
    im2.save(img_p2, "JPEG")

    db_dir = tmp_path / "dbs_find"
    db_dir.mkdir()
    engine = DualDatabaseEngine(
        catalog_path=db_dir / "catalog.db",
        thumbnails_path=db_dir / "thumbnails.db",
    )

    window = PhotoGalleryMainWindow(engine=engine)
    window.show()
    window.scan_folder(test_dir, include_subfolders=False, blocking=True)
    assert window.catalog_model.total_media_count == 2

    # 1. Ctrl+F in Grid Mode toggles Search Scope Bar
    assert window.central_stack.currentWidget() == window.grid_view
    window.ribbon.hide_search_scope()
    assert not window.ribbon.scope_bar.isVisible()

    window._on_ctrl_f()
    qapp.processEvents()
    assert window.ribbon.scope_bar.isVisible()

    # 2. Text Search button in Find tab toggles search scope bar
    window.ribbon.hide_search_scope()
    assert not window.ribbon.scope_bar.isVisible()
    window.ribbon.btn_find_text_search.click()
    assert window.ribbon.scope_bar.isVisible()

    # 3. Ctrl+F in Single Viewer Mode toggles Fine-Tune right inspector
    assert len(window.catalog_model._records) == 2
    mid1 = window.catalog_model._records[0].id
    assert mid1 is not None
    window.open_single_viewer(mid1)
    assert window.central_stack.currentWidget() == window.media_viewer

    # Ensure right dock is hidden or info mode
    window.right_tag_dock.hide()
    window._on_ctrl_f()
    assert not window.right_tag_dock.isHidden()
    assert window.right_tag_dock.current_mode() == "fine_tune"

    # Close viewer back to grid
    window.show_grid_view()
    assert window.central_stack.currentWidget() == window.grid_view

    # 4. Filter signals from Find tab:
    # 4a. Rating filter
    window.catalog_repo.set_rating([mid1], 5)
    window.rebuild_and_run_query()

    window.ribbon.rating_filter_selected.emit((5,))
    assert window.left_nav_dock._ratings == {5}
    assert window.catalog_model.total_media_count == 1

    # Clear ratings
    window.ribbon.rating_filter_selected.emit(())
    assert len(window.left_nav_dock._ratings) == 0
    assert window.catalog_model.total_media_count == 2

    # 4b. Flag filter
    window.catalog_repo.set_flag([mid1], 1)
    window.ribbon.flagged_filter_selected.emit(True)
    assert window.catalog_model.total_media_count == 1
    window.ribbon.flagged_filter_selected.emit(False)
    assert window.catalog_model.total_media_count == 2

    # 4c. Edited filter
    window.catalog_repo.record_pixel_edit_backup(mid1, str(test_dir / "backup.jpg"))
    window.ribbon.edited_filter_selected.emit(True)
    assert window.catalog_model.total_media_count == 1
    window.ribbon.edited_filter_selected.emit(False)
    assert window.catalog_model.total_media_count == 2

    # 4d. Media type filter
    window.ribbon.media_type_filter_selected.emit("image")
    assert window.catalog_model.total_media_count == 2
    window.ribbon.media_type_filter_selected.emit("video")
    assert window.catalog_model.total_media_count == 0
    window.ribbon.media_type_filter_selected.emit("all")
    assert window.catalog_model.total_media_count == 2

    # 4e. People boolean mode toggle (+ and / / or)
    window.ribbon.person_boolean_mode_changed.emit("AND")
    assert window.left_nav_dock._person_boolean_mode == "AND"
    window.ribbon.person_boolean_mode_changed.emit("OR")
    assert window.left_nav_dock._person_boolean_mode == "OR"

    window.shutdown()

