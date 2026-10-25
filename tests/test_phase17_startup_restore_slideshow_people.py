"""
Phase 17 Verification Tests:
1. State persistence on startup (QSettings: folder, filter state, view mode, selection, docks).
2. People section in Home & Find menus (comfortable width for 6-7 faces, tooltip readability).
3. Fullscreen Slideshow (frameless fullscreen dialog, timer advance, HUD controls, play/pause, close).
4. Information Panel resilience (switching between single image and grid view, restoring via toggle).
"""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QSettings, QSize, Qt
from PySide6.QtWidgets import QApplication
import pytest

from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository, MediaItemRecord
from src.ui.main_window import PhotoGalleryMainWindow
from src.ui.ribbon_toolbar import RibbonToolbar
from src.ui.slideshow_view import FullscreenSlideshowDialog, SlideshowHud


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_people_carousel_sizing_and_tooltips(qapp: QApplication) -> None:
    """Verify that people carousels in Home and Find tabs accommodate ~6-7 faces and have tooltips."""
    ribbon = RibbonToolbar()

    # Home face carousel sizing
    assert ribbon.home_face_carousel.minimumWidth() >= 240
    assert ribbon.home_face_carousel.maximumWidth() >= 340

    # Find face carousel sizing
    assert ribbon.find_face_carousel.minimumWidth() >= 240

    # Populate with sample people
    people = [
        {"id": 1, "name": "Alice", "media_count": 12, "thumbnail_blob": None},
        {"id": 2, "name": "Bob", "media_count": 8, "thumbnail_blob": None},
        {"id": 3, "name": "Charlie", "media_count": 25, "thumbnail_blob": None},
        {"id": 4, "name": "Diana", "media_count": 4, "thumbnail_blob": None},
        {"id": 5, "name": "Evan", "media_count": 19, "thumbnail_blob": None},
        {"id": 6, "name": "Fiona", "media_count": 3, "thumbnail_blob": None},
    ]
    ribbon.populate_face_carousel(people)

    # Check button tooltips
    assert 1 in ribbon.home_face_carousel._buttons
    btn_alice = ribbon.home_face_carousel._buttons[1]
    assert "Alice (12 items)" in btn_alice.toolTip()
    # Ensure styling has tooltip block
    assert "QToolTip" in btn_alice.styleSheet()


def test_fullscreen_slideshow_dialog(qapp: QApplication, tmp_path: Path) -> None:
    """Verify FullscreenSlideshowDialog initialization, HUD, slide navigation, and play/pause."""
    records = [
        MediaItemRecord(id=1, folder_id=1, filename="img1.jpg", size=100, mtime=1000, date_taken=1000, rating=0, flag=0, width=100, height=100, orientation=1, is_video=0, caption="", group_key=""),
        MediaItemRecord(id=2, folder_id=1, filename="img2.jpg", size=100, mtime=1000, date_taken=1000, rating=0, flag=0, width=100, height=100, orientation=1, is_video=0, caption="", group_key=""),
        MediaItemRecord(id=3, folder_id=1, filename="img3.jpg", size=100, mtime=1000, date_taken=1000, rating=0, flag=0, width=100, height=100, orientation=1, is_video=0, caption="", group_key=""),
    ]

    dialog = FullscreenSlideshowDialog(
        records=records,
        start_index=1,
        interval_ms=5000,
    )

    assert dialog.current_index == 1
    assert dialog.current_media_id() == 2
    assert dialog.is_playing is True
    assert dialog.slide_timer.isActive()

    # Next slide
    dialog.next_slide()
    assert dialog.current_index == 2
    assert dialog.current_media_id() == 3

    # Next slide wraps to 0
    dialog.next_slide()
    assert dialog.current_index == 0
    assert dialog.current_media_id() == 1

    # Prev slide wraps to 2
    dialog.prev_slide()
    assert dialog.current_index == 2
    assert dialog.current_media_id() == 3

    # Toggle pause
    dialog.toggle_play_pause()
    assert dialog.is_playing is False
    assert not dialog.slide_timer.isActive()

    # Toggle resume
    dialog.toggle_play_pause()
    assert dialog.is_playing is True
    assert dialog.slide_timer.isActive()

    dialog.close()


def test_state_persistence_save_and_restore(qapp: QApplication, tmp_path: Path, monkeypatch) -> None:
    """Verify that _save_app_state persists folder/view/docks and _restore_app_state restores them."""
    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    engine = DualDatabaseEngine(
        catalog_path=db_dir / "catalog.db",
        thumbnails_path=db_dir / "thumbnails.db",
    )
    repo = CatalogRepository(engine)
    f_path = str(tmp_path / "Vacation")
    fid = repo.upsert_folder(f_path, "Vacation", None, 1000)
    conn = engine.catalog_conn()
    conn.execute(
        "INSERT INTO media (folder_id, filename, ext, size, mtime, date_taken) VALUES (?, 'photo1.jpg', 'jpg', 1000, 1000, 1000);",
        (fid,),
    )
    conn.commit()

    window = PhotoGalleryMainWindow(engine=engine, start_xmp_daemon=False)
    window.refresh_navigation_and_autocomplete()
    window.rebuild_and_run_query()

    # Select the folder and zoom
    window.left_nav_dock.select_folder(fid)
    window.slider_status_zoom.setValue(220)

    # Allow save_app_state to run (bypass test guard)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    window._save_app_state()

    # Read QSettings directly to verify
    settings = QSettings("PhotoGallery64", "PhotoGallery64")
    f_json = settings.value("filter_state")
    assert f_json is not None
    f_dict = json.loads(f_json)
    assert f_dict.get("folder_id") == fid
    assert settings.value("zoom_val", type=int) == 220

    # Create a fresh second window and restore
    window2 = PhotoGalleryMainWindow(engine=engine, start_xmp_daemon=False)
    window2._restore_app_state()

    assert window2.left_nav_dock._folder_id == fid
    assert window2.slider_status_zoom.value() == 220

    window.close()
    window2.close()


def test_information_panel_resilience_grid_and_viewer(qapp: QApplication, tmp_path: Path) -> None:
    """Verify info panel does not disappear or get stuck when switching between single view and grid view."""
    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    engine = DualDatabaseEngine(
        catalog_path=db_dir / "catalog.db",
        thumbnails_path=db_dir / "thumbnails.db",
    )
    repo = CatalogRepository(engine)
    fid = repo.upsert_folder(str(tmp_path), "TestFolder", None, 1000)
    # Create sample image on disk
    from PIL import Image
    im = Image.new("RGB", (200, 200), color="blue")
    im_path = tmp_path / "sample.jpg"
    im.save(im_path)
    conn = engine.catalog_conn()
    c = conn.execute(
        "INSERT INTO media (folder_id, filename, ext, size, mtime, date_taken) VALUES (?, 'sample.jpg', 'jpg', 2000, 2000, 2000) RETURNING id;",
        (fid,),
    )
    mid = int(c.fetchone()[0])
    conn.commit()

    window = PhotoGalleryMainWindow(engine=engine, start_xmp_daemon=False)
    window.refresh_navigation_and_autocomplete()
    window.rebuild_and_run_query()

    # 1. Right dock is initially shown in info mode
    assert not window.right_tag_dock.isHidden()
    assert window.right_tag_dock.current_mode() == "info"

    # 2. Open single viewer
    ok = window.open_single_viewer(mid)
    assert ok is True
    assert window.central_stack.currentWidget() == window.media_viewer
    # Right dock should remain accessible
    assert not window.right_tag_dock.isHidden()

    # Switch to fine tune in single viewer
    window.toggle_right_pane("fine_tune")
    assert window.right_tag_dock.current_mode() == "fine_tune"

    # 3. Return to grid view
    window.show_grid_view()
    assert window.central_stack.currentWidget() == window.grid_view
    # In grid view, right dock should automatically revert to info mode and be visible
    assert window.right_tag_dock.current_mode() == "info"
    assert not window.right_tag_dock.isHidden()

    # 4. Toggle right pane off and on
    window.toggle_right_pane("info")
    assert window.right_tag_dock.isHidden()

    window.toggle_right_pane("info")
    assert not window.right_tag_dock.isHidden()
    assert window.right_tag_dock.current_mode() == "info"

    window.close()
