"""
Phase 12 Tests: Information Panel In-Place Editing, Star Rating Hover Previews,
and Descriptive Tags Recent Suggestions (matching Windows Live Photo Gallery).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PySide6.QtCore import QDate, QTime, QTimeZone, Qt
from PySide6.QtWidgets import QApplication

from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository
from src.ui.main_window import PhotoGalleryMainWindow
from src.ui.right_tag_dock import TagAndCaptionDock, _StarRatingHoverWidget


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_tag_and_caption_dock_inplace_editing(qapp: QApplication, tmp_path: Path) -> None:
    """Test in-place editing controls: filename, date/time, author, size tooltip, star hover."""
    engine = DualDatabaseEngine(tmp_path / "catalog.db", tmp_path / "thumbs.db")
    repo = CatalogRepository(engine)

    photo_dir = tmp_path / "Photos"
    photo_dir.mkdir()
    f1 = photo_dir / "IMG_1001.jpg"
    f1.write_bytes(b"\xFF\xD8\xFF\xE0" + b"\x00" * 2048)

    fid = repo.upsert_folder(str(photo_dir), "Photos", None, 1700000000)
    ids = repo.bulk_upsert_media(
        [
            {
                "folder_id": fid,
                "filename": "IMG_1001.jpg",
                "ext": ".jpg",
                "size": 2052,
                "mtime": 1690000000,
                "date_taken": 1690464819,  # 27/07/2023 13:33:39 UTC
                "rating": 2,
                "flag": 0,
                "width": 1920,
                "height": 1080,
                "author": "Marcus",
            }
        ]
    )
    mid = ids[0]

    dock = TagAndCaptionDock()
    summary = repo.get_selection_summary([mid])
    dock.bind_selection([mid], summary)

    # 1. Filename in-place edit widget
    assert dock.txt_filename.text() == "IMG_1001.jpg"
    assert dock.txt_filename.isEnabled() is True
    assert dock.lbl_filename.text() == "IMG_1001.jpg"

    # 2. Date and Time pickers
    assert dock.date_edit.date() == QDate(2023, 7, 27)
    assert dock.time_edit.time() == QTime(13, 33, 39)
    assert dock.time_edit.toolTip() == "13:33"

    # 3. Size tooltip
    assert "KB" in dock.lbl_size.text() or "B" in dock.lbl_size.text()
    assert dock.lbl_size.toolTip() == dock.lbl_size.text()

    # 4. Author in-place edit widget
    assert dock.txt_author.text() == "Marcus"
    assert dock.lbl_author.text() == "Marcus"

    # 5. Star Rating widget & hover preview
    assert dock.star_rating_widget.rating == 2
    assert dock.star_rating_widget.toolTip() == "Click a star to assign a rating. (0 - 5)"
    # Hover star 4
    dock.star_rating_widget.set_hover_star(4)
    assert dock.star_rating_widget.hover_star == 4
    # Reset hover
    dock.star_rating_widget.set_hover_star(0)
    assert dock.star_rating_widget.hover_star == 0

    # Test clicking 5th star
    emitted_ratings: list[int] = []
    dock.rating_edited.connect(emitted_ratings.append)
    dock.star_buttons[4].click()
    assert dock.star_rating_widget.rating == 5
    assert emitted_ratings == [5]

    # Test toggling rating off by clicking 5th star again
    dock.star_buttons[4].click()
    assert dock.star_rating_widget.rating == 0
    assert emitted_ratings == [5, 0]


def test_main_window_inplace_filename_rename(qapp: QApplication, tmp_path: Path) -> None:
    """Test inline filename editing in right inspector renames file on disk and in database."""
    photo_dir = tmp_path / "GalleryFolder"
    photo_dir.mkdir()
    f1 = photo_dir / "DSC_0001.jpg"
    f1.write_bytes(b"\xFF\xD8\xFF\xE0" + b"\x00" * 1024)

    engine = DualDatabaseEngine(tmp_path / "catalog.db", tmp_path / "thumbs.db")
    win = PhotoGalleryMainWindow(engine, start_xmp_daemon=False)
    win.scan_folder(str(photo_dir), include_subfolders=False, blocking=True)

    mids = [r.id for r in win.catalog_model._records]
    assert len(mids) == 1
    mid = mids[0]

    # Select the media item
    win.grid_view.on_card_checkbox_toggled(mid, True)

    # Edit filename in right dock and commit
    win.right_tag_dock.txt_filename.setText("DSC_Renamed.jpg")
    win.right_tag_dock.filename_committed.emit("DSC_Renamed.jpg")

    # Verify disk rename
    assert not (photo_dir / "DSC_0001.jpg").exists()
    assert (photo_dir / "DSC_Renamed.jpg").exists()

    # Verify catalog.db update
    detail = win.catalog_repo.get_media_detail(mid)
    assert detail["filename"] == "DSC_Renamed.jpg"
    assert win.right_tag_dock.txt_filename.text() == "DSC_Renamed.jpg"

    win.close()


def test_main_window_inplace_datetime_and_author_edit(qapp: QApplication, tmp_path: Path) -> None:
    """Test date/time picker and author edit in right inspector."""
    photo_dir = tmp_path / "GalleryFolder2"
    photo_dir.mkdir()
    f1 = photo_dir / "PhotoA.jpg"
    f2 = photo_dir / "PhotoB.jpg"
    f1.write_bytes(b"\xFF\xD8\xFF\xE0" + b"\x00" * 1024)
    f2.write_bytes(b"\xFF\xD8\xFF\xE0" + b"\x00" * 1024)

    engine = DualDatabaseEngine(tmp_path / "catalog.db", tmp_path / "thumbs.db")
    win = PhotoGalleryMainWindow(engine, start_xmp_daemon=False)
    win.scan_folder(str(photo_dir), include_subfolders=False, blocking=True)

    mids = [r.id for r in win.catalog_model._records]
    assert len(mids) == 2

    # 1. Author editing on single selection
    win.grid_view.on_card_checkbox_toggled(mids[0], True)
    win.right_tag_dock.txt_author.setText("Jane Doe")
    win.right_tag_dock.author_committed.emit("Jane Doe")

    detail1 = win.catalog_repo.get_media_detail(mids[0])
    assert detail1["author"] == "Jane Doe"
    assert detail1["metadata_dirty"] == 1

    # 2. Date Taken editing on single selection
    target_ts = 1720000000
    win.right_tag_dock.date_taken_edited.emit(target_ts)
    detail1 = win.catalog_repo.get_media_detail(mids[0])
    assert detail1["date_taken"] == target_ts

    # 3. Relative Date Taken shift on multi-selection
    win.catalog_repo.set_date_taken([mids[0]], 1000)
    win.catalog_repo.set_date_taken([mids[1]], 2000)

    win.grid_view.on_card_checkbox_toggled(mids[0], True)
    win.grid_view.on_card_checkbox_toggled(mids[1], True)
    assert set(win.grid_view.selected_media_ids()) == {mids[0], mids[1]}
    # Lead is mids[0] (ts=1000). Shift lead to ts=1500 (delta = +500)
    win.right_tag_dock.date_taken_edited.emit(1500)

    det_a = win.catalog_repo.get_media_detail(mids[0])
    det_b = win.catalog_repo.get_media_detail(mids[1])
    assert det_a["date_taken"] == 1500
    assert det_b["date_taken"] == 2500

    win.close()


def test_recent_descriptive_tags_popup(qapp: QApplication, tmp_path: Path) -> None:
    """Test top-10 recent tags suggestions in descriptive tag input."""
    engine = DualDatabaseEngine(tmp_path / "catalog.db", tmp_path / "thumbs.db")
    repo = CatalogRepository(engine)

    fid = repo.upsert_folder("D:/Photos", "Photos", None, 1700000000)
    mids = repo.bulk_upsert_media([
        {"folder_id": fid, "filename": f"img_{i}.jpg", "ext": ".jpg", "size": 1000, "mtime": 1700000000}
        for i in range(12)
    ])

    for i in range(12):
        t_id = repo.create_tag(f"Tag_{i}")
        repo.assign_tag([mids[i]], t_id)

    recent = repo.get_recent_tags(10)
    assert len(recent) == 10

    dock = TagAndCaptionDock()
    dock.set_autocomplete_lists([f"Tag_{i}" for i in range(12)], [], recent_tags=recent)

    assert dock.txt_add_tag._recent_tags == recent
    assert len(dock.txt_add_tag._recent_tags) == 10

    # Emitting tag_selected adds tag
    received: list[str] = []
    dock.tag_added.connect(received.append)
    dock.txt_add_tag.tag_selected.emit("Tag_0")
    assert received == ["Tag_0"]
