"""
Phase 16 Verification Suite: Context Menus, Split Flagged Button, Independent Grouping/Sorting,
Responsive Window Width, and Single Media Viewer Arrow Key Navigation.
"""

from __future__ import annotations

import os
from pathlib import Path
from PIL import Image
import pytest
from PySide6.QtCore import Qt, QEvent, QPoint
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QToolButton

from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository, ThumbnailRepository, FilterQuerySpec
from src.ui.ribbon_toolbar import RibbonToolbar
from src.ui.media_viewer import SingleMediaViewer
from src.ui.main_window import PhotoGalleryMainWindow


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_remove_folder_tree_from_library_preserves_disk_files(tmp_path: Path) -> None:
    """Verify removing a folder removes DB records and thumbnails but NEVER deletes disk files."""
    cat_path = tmp_path / "catalog.db"
    thumb_path = tmp_path / "thumbnails.db"
    engine = DualDatabaseEngine(cat_path, thumb_path)
    cat_repo = CatalogRepository(engine)
    thumb_repo = ThumbnailRepository(engine)

    # Create directory tree with files on disk
    parent_dir = tmp_path / "Vacation"
    parent_dir.mkdir(parents=True, exist_ok=True)
    child_dir = parent_dir / "Beach"
    child_dir.mkdir(parents=True, exist_ok=True)

    file_1 = parent_dir / "img1.jpg"
    file_2 = child_dir / "img2.jpg"
    file_1.write_bytes(b"dummy image 1")
    file_2.write_bytes(b"dummy image 2")

    # Insert folders
    parent_id = cat_repo.upsert_folder(str(parent_dir), "Vacation", None, 1000)
    child_id = cat_repo.upsert_folder(str(child_dir), "Beach", parent_id, 1000)

    # Insert media
    ids = cat_repo.bulk_upsert_media(
        [
            {
                "folder_id": parent_id,
                "filename": "img1.jpg",
                "ext": "jpg",
                "size": 100,
                "mtime": 1000,
                "rating": 5,
            },
            {
                "folder_id": child_id,
                "filename": "img2.jpg",
                "ext": "jpg",
                "size": 100,
                "mtime": 1000,
                "rating": 4,
            },
        ]
    )
    m1_id, m2_id = ids[0], ids[1]

    # Insert thumbnails
    thumb_repo.put_webp_blob(m1_id, b"webp_thumb_1", 1000)
    thumb_repo.put_webp_blob(m2_id, b"webp_thumb_2", 1000)
    assert thumb_repo.get_webp_blob(m1_id) is not None
    assert thumb_repo.get_webp_blob(m2_id) is not None

    # Call remove_folder_tree on parent
    cat_repo.remove_folder_tree(parent_id)

    # Verify DB records deleted
    all_media = cat_repo.query_media_records(FilterQuerySpec())
    assert len(all_media) == 0
    folders = cat_repo.get_folders()
    assert len(folders) == 0

    # Verify thumbnails deleted from thumbnails.db
    assert thumb_repo.get_webp_blob(m1_id, 256) is None
    assert thumb_repo.get_webp_blob(m2_id, 256) is None

    # CRITICAL: Verify disk files still exist!
    assert file_1.exists()
    assert file_2.exists()
    assert parent_dir.exists()
    assert child_dir.exists()

    engine.close_all()


def test_split_flagged_button_behavior(qapp: QApplication) -> None:
    """Verify split flagged button: click toggles filter, menu selects specific option."""
    ribbon = RibbonToolbar()

    assert hasattr(ribbon, "btn_qf_flagged")
    btn = ribbon.btn_qf_flagged
    assert btn.popupMode() == QToolButton.MenuButtonPopup

    signals = []
    ribbon.flag_filter_selected.connect(lambda val: signals.append(val))

    # Click body -> toggles to checked (Flagged)
    btn.click()
    assert signals[-1] == 1
    assert btn.isChecked() is True

    # Click body again -> toggles back to unchecked (All)
    btn.click()
    assert signals[-1] is None
    assert btn.isChecked() is False

    # Menu items
    menu = btn.menu()
    assert menu is not None
    actions = {act.text(): act for act in menu.actions()}
    assert "Flagged" in actions
    assert "Not flagged" in actions
    assert "All items" in actions

    actions["Not flagged"].trigger()
    assert signals[-1] == 0
    assert btn.isChecked() is False

    actions["Flagged"].trigger()
    assert signals[-1] == 1
    assert btn.isChecked() is True

    actions["All items"].trigger()
    assert signals[-1] is None
    assert btn.isChecked() is False

    ribbon.deleteLater()


def test_no_double_arrows_in_ribbon_button_labels(qapp: QApplication) -> None:
    """Verify dropdown buttons on ribbon don't contain hardcoded unicode arrows."""
    ribbon = RibbonToolbar()

    for btn in [
        getattr(ribbon, "btn_qf_months", None),
        getattr(ribbon, "btn_qf_rated", None),
        getattr(ribbon, "btn_qf_flagged", None),
        getattr(ribbon, "btn_find_dates", None),
        getattr(ribbon, "btn_find_months", None),
        getattr(ribbon, "btn_find_years", None),
        getattr(ribbon, "btn_find_rated", None),
        getattr(ribbon, "btn_find_flagged", None),
        getattr(ribbon, "btn_tag_desc", None),
        getattr(ribbon, "btn_tag_people", None),
        getattr(ribbon, "btn_slide_show", None),
        getattr(ribbon, "btn_view_props", None),
        getattr(ribbon, "btn_view_rating", None),
    ]:
        if btn is not None:
            text = btn.text()
            assert "▾" not in text, f"Button '{text}' contains redundant '▾' character"
            assert "▼" not in text, f"Button '{text}' contains redundant '▼' character"

    ribbon.deleteLater()


def test_independent_group_by_and_sort_by(tmp_path: Path) -> None:
    """Verify media items are grouped first, then sorted within groups."""
    cat_path = tmp_path / "catalog.db"
    thumb_path = tmp_path / "thumbnails.db"
    engine = DualDatabaseEngine(cat_path, thumb_path)
    repo = CatalogRepository(engine)

    folder_id = repo.upsert_folder(str(tmp_path), "Photos", None, 1000)

    # Insert 4 items with distinct rating, name, and dates
    # A: rating 1, name "Zebra.jpg", date 2024-01-10 (timestamp: 1704880800), flag 1
    # B: rating 1, name "Apple.jpg", date 2024-01-20 (timestamp: 1705744800), flag 0
    # C: rating 5, name "Mango.jpg", date 2024-02-05 (timestamp: 1707127200), flag 1
    # D: rating 5, name "Banana.jpg", date 2024-02-15 (timestamp: 1707991200), flag 0
    repo.bulk_upsert_media(
        [
            {
                "folder_id": folder_id,
                "filename": "Zebra.jpg",
                "ext": "jpg",
                "size": 100,
                "mtime": 1000,
                "rating": 1,
                "date_taken": 1704880800,
                "flag": 1,
            },
            {
                "folder_id": folder_id,
                "filename": "Apple.jpg",
                "ext": "jpg",
                "size": 200,
                "mtime": 1000,
                "rating": 1,
                "date_taken": 1705744800,
                "flag": 0,
            },
            {
                "folder_id": folder_id,
                "filename": "Mango.jpg",
                "ext": "jpg",
                "size": 300,
                "mtime": 1000,
                "rating": 5,
                "date_taken": 1707127200,
                "flag": 1,
            },
            {
                "folder_id": folder_id,
                "filename": "Banana.jpg",
                "ext": "jpg",
                "size": 400,
                "mtime": 1000,
                "rating": 5,
                "date_taken": 1707991200,
                "flag": 0,
            },
        ]
    )

    # 1. Group by Rating (descending), Sort by Name (ascending)
    spec1 = FilterQuerySpec(group_by="rating", sort_by="name", sort_reverse=False)
    results1 = repo.query_media_records(spec1)
    # Group rating 5 first, then within rating 5: Banana, Mango
    # Group rating 1 next, then within rating 1: Apple, Zebra
    names1 = [r.filename for r in results1]
    assert names1 == ["Banana.jpg", "Mango.jpg", "Apple.jpg", "Zebra.jpg"]

    # 2. Group by Month (ascending), Sort by Rating (descending)
    spec2 = FilterQuerySpec(group_by="month", group_reverse=False, sort_by="rating", sort_reverse=False)
    results2 = repo.query_media_records(spec2)
    # February 2024 group (newest first): Banana (rating 5) and Mango (rating 5)
    # January 2024 group: Apple (rating 1) and Zebra (rating 1)
    gkeys2 = [r.group_key for r in results2]
    assert gkeys2[:2] == ["February 2024", "February 2024"]
    assert gkeys2[2:] == ["January 2024", "January 2024"]

    # 3. Flag filter test
    spec_flagged = FilterQuerySpec(flag_filter=1)
    results_fl = repo.query_media_records(spec_flagged)
    assert {r.filename for r in results_fl} == {"Zebra.jpg", "Mango.jpg"}

    spec_unflagged = FilterQuerySpec(flag_filter=0)
    results_unfl = repo.query_media_records(spec_unflagged)
    assert {r.filename for r in results_unfl} == {"Apple.jpg", "Banana.jpg"}

    engine.close_all()


def test_responsive_window_minimum_width(qapp: QApplication, tmp_path: Path) -> None:
    """Verify application can shrink comfortably down to 650px width without layout blockage."""
    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    engine = DualDatabaseEngine(
        catalog_path=db_dir / "catalog.db",
        thumbnails_path=db_dir / "thumbnails.db",
    )
    main_win = PhotoGalleryMainWindow(engine=engine)
    main_win.show()
    qapp.processEvents()

    # Minimum width should be <= 650
    assert main_win.minimumWidth() <= 650

    # Resize window to 600px width
    main_win.resize(600, 480)
    qapp.processEvents()

    assert main_win.width() <= 650

    main_win.close()
    main_win.deleteLater()
    engine.close_all()


def test_viewer_arrow_key_navigation_intercepted(qapp: QApplication) -> None:
    """Verify Left and Right arrow keys in SingleMediaViewer emit navigate_requested even when child has focus."""
    viewer = SingleMediaViewer()
    viewer.show()
    qapp.processEvents()

    # Give focus to graphics_view viewport
    viewer.graphics_view.viewport().setFocus()
    assert viewer.graphics_view.viewport().hasFocus() or viewer.graphics_view.hasFocus()

    nav_events = []
    viewer.navigate_requested.connect(lambda delta: nav_events.append(delta))

    # Send Right arrow
    key_right = QKeyEvent(QEvent.KeyPress, Qt.Key_Right, Qt.NoModifier)
    QApplication.sendEvent(viewer.graphics_view.viewport(), key_right)
    assert nav_events == [1]

    # Send Left arrow
    key_left = QKeyEvent(QEvent.KeyPress, Qt.Key_Left, Qt.NoModifier)
    QApplication.sendEvent(viewer.graphics_view.viewport(), key_left)
    assert nav_events == [1, -1]

    viewer.close()
    viewer.deleteLater()
