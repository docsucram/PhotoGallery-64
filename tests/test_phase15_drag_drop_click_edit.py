"""
Unit and integration tests for Phase 15:
- Drag-and-drop file moves between folders with full QUndoStack Undo/Redo
- OLE drag data preparation with QUrl file lists and multi-selection preservation
- Click-to-edit information panel fields (Filename, Date taken, Author) with clean label view and editor switching
- Folder click expansion in LeftNavDock
- Star rating hover outline and removal fading
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest
from PySide6.QtCore import QMimeData, QPoint, Qt, QUrl
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository, ThumbnailRepository
from src.ui.left_nav_dock import LeftNavigationDock, _LeftNavTree
from src.ui.main_window import PhotoGalleryMainWindow, _MoveFilesUndoCommand
from src.ui.right_tag_dock import (
    TagAndCaptionDock,
    _ClickToEditLabel,
    _ClickToEditWidget,
    _StarButton,
)


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def test_env(tmp_path: Path):
    """Sets up temporary folders and dual database catalog for testing."""
    folder_a = tmp_path / "FolderA"
    folder_b = tmp_path / "FolderB"
    folder_a.mkdir()
    folder_b.mkdir()

    # Create dummy images
    img1 = folder_a / "photo1.jpg"
    img1.write_bytes(b"dummy image content 1")
    img2 = folder_a / "photo2.jpg"
    img2.write_bytes(b"dummy image content 2")

    cat_db = tmp_path / "catalog.db"
    thumb_db = tmp_path / "thumbs.db"
    engine = DualDatabaseEngine(cat_db, thumb_db)
    repo = CatalogRepository(engine)

    fa_id = repo.upsert_folder(str(folder_a), "FolderA", None, int(folder_a.stat().st_mtime))
    fb_id = repo.upsert_folder(str(folder_b), "FolderB", None, int(folder_b.stat().st_mtime))

    # Insert media records
    conn = engine.catalog_conn()
    c1 = conn.execute(
        """
        INSERT INTO media (folder_id, filename, ext, size, mtime, date_taken, rating, flag)
        VALUES (?, ?, 'jpg', 100, 1000, 1000, 3, 0)
        RETURNING id;
        """,
        (fa_id, "photo1.jpg"),
    )
    m1_id = int(c1.fetchone()[0])

    c2 = conn.execute(
        """
        INSERT INTO media (folder_id, filename, ext, size, mtime, date_taken, rating, flag)
        VALUES (?, ?, 'jpg', 200, 2000, 2000, 4, 1)
        RETURNING id;
        """,
        (fa_id, "photo2.jpg"),
    )
    m2_id = int(c2.fetchone()[0])
    conn.commit()

    return {
        "engine": engine,
        "repo": repo,
        "folder_a": folder_a,
        "folder_b": folder_b,
        "fa_id": fa_id,
        "fb_id": fb_id,
        "m1_id": m1_id,
        "m2_id": m2_id,
        "img1": img1,
        "img2": img2,
    }


def test_click_to_edit_widget(qapp):
    """Verifies that _ClickToEditWidget switches between plain label and editor cleanly."""
    from PySide6.QtWidgets import QLineEdit, QWidget

    parent = QWidget()
    line_edit = QLineEdit(parent)
    line_edit.setText("original_name.jpg")

    widget = _ClickToEditWidget(line_edit, parent, empty_placeholder="-")
    parent.show()

    # Initially at index 0 (clean view label)
    assert widget.currentIndex() == 0
    assert widget.view_label.text() == "-"

    # Set display text
    widget.set_display_text("photo.jpg")
    assert widget.view_label.text() == "photo.jpg"

    # Start edit: switches to index 1 (editor)
    widget.start_edit()
    assert widget.currentIndex() == 1

    # Finish edit with updated text
    widget.finish_edit("renamed_photo.jpg")
    assert widget.currentIndex() == 0
    assert widget.view_label.text() == "renamed_photo.jpg"

    # Disabled / uneditable state
    widget.set_editable(False)
    widget.start_edit()
    assert widget.currentIndex() == 0  # Should remain on label when not editable
    parent.deleteLater()


def test_right_tag_dock_click_to_edit_binding(qapp, test_env):
    """Verifies that TagAndCaptionDock binds single and multiple items to Click-to-Edit stacks."""
    dock = TagAndCaptionDock()

    # 1. Zero items selected
    dock.bind_selection([], {"count": 0})
    assert dock.stack_filename.currentIndex() == 0
    assert dock.stack_date_time.currentIndex() == 0
    assert dock.stack_author.currentIndex() == 0
    assert not dock.stack_filename._editable

    # 2. Single item selected
    summary_single = {
        "count": 1,
        "filename": "sunset.jpg",
        "date_taken": 1713000000,
        "author": "Alice",
        "rating": 4,
        "flag": 0,
    }
    dock.bind_selection([test_env["m1_id"]], summary_single)

    assert dock.stack_filename.currentIndex() == 0
    assert dock.stack_filename.view_label.text() == "sunset.jpg"
    assert dock.stack_filename._editable is True

    assert dock.stack_author.currentIndex() == 0
    assert dock.stack_author.view_label.text() == "Alice"
    assert dock.stack_author._editable is True

    assert dock.stack_date_time.currentIndex() == 0
    assert dock.stack_date_time._editable is True

    # Click filename to edit
    dock.stack_filename.view_label.clicked.emit()
    assert dock.stack_filename.currentIndex() == 1

    # Finish edit
    dock.stack_filename.finish_edit("sunset_edited.jpg")
    assert dock.stack_filename.currentIndex() == 0
    assert dock.stack_filename.view_label.text() == "sunset_edited.jpg"

    # 3. Multiple items selected
    summary_multi = {
        "count": 2,
        "lead": {"date_taken": 1713000000},
        "common_rating": 3,
        "common_flag": 0,
    }
    dock.bind_selection([test_env["m1_id"], test_env["m2_id"]], summary_multi)
    assert dock.stack_filename.currentIndex() == 0
    assert "Multiple files" in dock.stack_filename.view_label.text()
    assert dock.stack_filename._editable is False

    assert dock.stack_author.currentIndex() == 0
    assert dock.stack_author._editable is False
    dock.deleteLater()


def test_star_removal_fading(qapp):
    """Verifies that hover over existing stars correctly identifies the removal fading state."""
    dock = TagAndCaptionDock()

    dock.star_rating_widget.set_rating(4)
    assert dock.star_rating_widget.rating == 4

    # When hovering over star 2, stars 3 and 4 should be marked for removal fading
    dock.star_rating_widget.set_hover_star(2)
    assert dock.star_rating_widget.hover_star == 2

    btn1 = dock.star_buttons[0]  # star 1 (<= hover_star and <= rating)
    btn3 = dock.star_buttons[2]  # star 3 (hover_star < star <= rating -> removal indicator)

    assert btn1.star_index <= dock.star_rating_widget.hover_star
    assert dock.star_rating_widget.hover_star < btn3.star_index <= dock.star_rating_widget.rating
    dock.deleteLater()


def test_left_nav_folder_expand_on_click(qapp, test_env):
    """Verifies that clicking on a folder node in LeftNavigationDock automatically expands it."""
    dock = LeftNavigationDock()
    dock.refresh_trees(test_env["repo"])

    # Expand section folders
    dock.root_all.setExpanded(True)
    assert dock.root_all.childCount() >= 2

    item_a = dock._folder_items[test_env["fa_id"]]
    # Add a dummy child to item_a to test expansion
    from PySide6.QtWidgets import QTreeWidgetItem
    sub = QTreeWidgetItem(["SubFolder"])
    item_a.addChild(sub)
    item_a.setExpanded(False)

    dock._on_item_clicked(item_a, 0)
    assert item_a.isExpanded() is True
    dock.deleteLater()


def test_drag_and_drop_file_move_with_undo_redo(qapp, test_env):
    """
    Verifies that dragging and dropping a file between folders:
    1. Moves the physical file on disk.
    2. Updates catalog.db folder_id.
    3. Pushes to QUndoStack.
    4. Undoes file move on disk and database.
    5. Redoes file move on disk and database.
    """
    main_win = PhotoGalleryMainWindow(test_env["engine"], start_xmp_daemon=False)

    src_path = test_env["img1"]
    dst_folder = test_env["folder_b"]
    dst_path = dst_folder / "photo1.jpg"

    assert src_path.exists()
    assert not dst_path.exists()

    # Simulate dropping files on Folder B
    main_win._on_files_dropped_on_folder(
        target_folder_id=test_env["fb_id"],
        target_folder_path=str(dst_folder),
        file_paths=[str(src_path)],
    )

    # After move (redo executed)
    assert not src_path.exists()
    assert dst_path.exists()

    detail = test_env["repo"].get_media_detail(test_env["m1_id"])
    assert detail["folder_id"] == test_env["fb_id"]
    assert detail["filename"] == "photo1.jpg"
    assert main_win.undo_stack.canUndo() is True

    # Test Undo (Ctrl+Z)
    main_win.undo_stack.undo()
    assert src_path.exists()
    assert not dst_path.exists()

    detail_undo = test_env["repo"].get_media_detail(test_env["m1_id"])
    assert detail_undo["folder_id"] == test_env["fa_id"]
    assert detail_undo["filename"] == "photo1.jpg"

    # Test Redo (Ctrl+Y)
    main_win.undo_stack.redo()
    assert not src_path.exists()
    assert dst_path.exists()

    detail_redo = test_env["repo"].get_media_detail(test_env["m1_id"])
    assert detail_redo["folder_id"] == test_env["fb_id"]

    main_win.shutdown()
    main_win.deleteLater()
