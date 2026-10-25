"""
Tests for Phase 23:
1. Folder right-click context menu (Expand/Collapse, Cut, Copy, Delete, Rename, Create new folder).
2. In-place inline folder renaming without modal popups.
3. CatalogRepository.rename_folder path hierarchy updating.
4. Photo Gallery Options multi-tab dialog (General, Import, Publish, Originals, Edit, Trust Center).
5. Navigation pane section visibility toggles (Date taken, Descriptive tags, People, Ratings).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication, QTreeWidget, QTreeWidgetItem

from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository, FilterQuerySpec
from src.ui.dialogs import AppSettingsDialog
from src.ui.left_nav_dock import LeftNavigationDock, _NODE_TYPE_ROLE, _NODE_PAYLOAD_ROLE


@pytest.fixture
def qapp() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def test_db(tmp_path: Path):
    cat_path = tmp_path / "catalog.db"
    thumb_path = tmp_path / "thumbnails.db"
    engine = DualDatabaseEngine(str(cat_path), str(thumb_path))
    repo = CatalogRepository(engine)
    yield repo, tmp_path
    engine.close_all()


def test_catalog_repository_rename_folder(test_db):
    """Verifies CatalogRepository.rename_folder updates folder row, subfolders, and media items."""
    repo, tmp_path = test_db

    root_dir = tmp_path / "Photos"
    sub_dir = root_dir / "2026_Vacation"
    root_dir.mkdir()
    sub_dir.mkdir()

    fid_root = repo.upsert_folder(str(root_dir), "Photos", None, 1000)
    fid_sub = repo.upsert_folder(str(sub_dir), "2026_Vacation", fid_root, 1000)

    # Insert media in both folders
    m_root = root_dir / "root.jpg"
    m_sub = sub_dir / "beach.jpg"
    m_root.write_bytes(b"root")
    m_sub.write_bytes(b"beach")

    ids = repo.bulk_upsert_media([
        {
            "folder_id": fid_root,
            "filename": "root.jpg",
            "ext": ".jpg",
            "size": 4,
            "mtime": 1000,
            "date_taken": 1000,
            "width": 100,
            "height": 100,
            "orientation": 1,
        },
        {
            "folder_id": fid_sub,
            "filename": "beach.jpg",
            "ext": ".jpg",
            "size": 5,
            "mtime": 1000,
            "date_taken": 1000,
            "width": 100,
            "height": 100,
            "orientation": 1,
        },
    ])

    # Rename subfolder
    new_sub_dir = root_dir / "2026_Summer_Holiday"
    repo.rename_folder(str(sub_dir), str(new_sub_dir), "2026_Summer_Holiday")

    # Verify folder row
    folders = repo.get_folders()
    folder_dict = {f["id"]: f for f in folders}
    assert folder_dict[fid_sub]["name"] == "2026_Summer_Holiday"
    assert "2026_Summer_Holiday" in folder_dict[fid_sub]["path"]

    # Verify get_media_detail resolves updated folder path
    detail = repo.get_media_detail(ids[1])
    assert "2026_Summer_Holiday" in detail["folder_path"]


def test_left_nav_inline_rename_and_validation(qapp: QApplication, tmp_path: Path):
    """Verifies in-place inline renaming on LeftNavigationDock tree items."""
    dock = LeftNavigationDock()
    dock.show()

    test_dir = tmp_path / "Albums" / "Trip_2025"
    test_dir.mkdir(parents=True)

    # Add item
    item = QTreeWidgetItem(dock.root_folders, ["Trip_2025"])
    item.setData(0, _NODE_TYPE_ROLE, "folder")
    item.setData(0, _NODE_PAYLOAD_ROLE, 42)
    item.setToolTip(0, str(test_dir))

    # 1. Start inline rename
    dock._start_inline_rename(item)
    assert bool(item.flags() & Qt.ItemIsEditable) is True

    # 2. Rename with valid new name
    renamed_events = []
    dock.folder_renamed.connect(lambda old_p, new_p, name: renamed_events.append((old_p, new_p, name)))

    item.setText(0, "Trip_2025_Rome")

    new_expected_dir = tmp_path / "Albums" / "Trip_2025_Rome"
    assert new_expected_dir.exists()
    assert not test_dir.exists()
    assert item.text(0) == "Trip_2025_Rome"
    assert item.toolTip(0) == str(new_expected_dir)
    assert len(renamed_events) == 1
    assert renamed_events[0][2] == "Trip_2025_Rome"

    # 3. Rename with invalid characters (e.g. contain '*') -> reverts name
    dock._start_inline_rename(item)
    item.setText(0, "Invalid*Name")
    assert item.text(0) == "Trip_2025_Rome"

    # 4. Rename to existing folder name -> reverts name
    collision_dir = tmp_path / "Albums" / "Existing_Dir"
    collision_dir.mkdir()
    dock._start_inline_rename(item)
    item.setText(0, "Existing_Dir")
    assert item.text(0) == "Trip_2025_Rome"


def test_left_nav_create_folder_and_clipboard(qapp: QApplication, tmp_path: Path):
    """Verifies Create new folder action and clipboard Cut/Copy on folder items."""
    dock = LeftNavigationDock()
    parent_dir = tmp_path / "ParentAlbum"
    parent_dir.mkdir()

    parent_item = QTreeWidgetItem(dock.root_folders, ["ParentAlbum"])
    parent_item.setData(0, _NODE_TYPE_ROLE, "folder")
    parent_item.setToolTip(0, str(parent_dir))

    # 1. Create new folder
    created_events = []
    dock.folder_created.connect(created_events.append)

    dock._create_new_folder_action(parent_item, str(parent_dir))
    created_dir = parent_dir / "New folder"
    assert created_dir.exists()
    assert parent_item.childCount() == 1
    assert parent_item.child(0).text(0) == "New folder"
    assert len(created_events) == 1

    # 2. Clipboard copy and cut
    dock._clipboard_copy_folder(str(created_dir), is_cut=False)
    mime = QGuiApplication.clipboard().mimeData()
    assert mime.hasUrls()
    assert Path(mime.urls()[0].toLocalFile()) == created_dir
    assert mime.data("Preferred DropEffect") == b"\x05\x00\x00\x00"

    dock._clipboard_copy_folder(str(created_dir), is_cut=True)
    mime2 = QGuiApplication.clipboard().mimeData()
    assert Path(mime2.urls()[0].toLocalFile()) == created_dir
    assert mime2.data("Preferred DropEffect") == b"\x02\x00\x00\x00"


def test_left_nav_section_visibility_controls(qapp: QApplication):
    """Verifies LeftNavigationDock show/hide sections per Options dialog."""
    dock = LeftNavigationDock()

    assert not dock.root_dates.isHidden()
    assert not dock.root_tags.isHidden()
    assert not dock.root_people.isHidden()
    assert not dock.root_ratings.isHidden()

    # Hide dates and ratings
    dock.set_navigation_sections_visibility(
        show_dates=False,
        show_tags=True,
        show_people=True,
        show_ratings=False,
    )
    assert dock.root_dates.isHidden() is True
    assert dock.root_ratings.isHidden() is True
    assert dock.root_tags.isHidden() is False
    assert dock.root_people.isHidden() is False

    # Toggle individual section
    dock.set_section_visible("dates", True)
    assert dock.root_dates.isHidden() is False


def test_app_settings_dialog_tabs_and_controls(qapp: QApplication, tmp_path: Path):
    """Verifies Photo Gallery Options multi-tab dialog structure and settings bindings."""
    dlg = AppSettingsDialog(
        retention_days=180,
        current_theme="wlpg_aero",
        show_dates=True,
        show_tags=False,
        show_people=True,
        show_ratings=True,
        find_people_enabled=True,
        tooltip_previews=True,
        jpeg_quality=85,
        auto_straighten=True,
        auto_noise=False,
        auto_color=True,
        auto_exposure=True,
        originals_dir=str(tmp_path),
    )

    # 1. Tab count and titles
    assert dlg.tab_widget.count() == 6
    titles = [dlg.tab_widget.tabText(i) for i in range(dlg.tab_widget.count())]
    assert titles == ["General", "Face Recognition", "Import", "Originals", "Edit", "Database"]

    # 2. General tab controls
    assert dlg.show_dates() is True
    assert dlg.show_tags() is False
    assert dlg.show_people() is True
    assert dlg.show_ratings() is True
    assert dlg.find_people_enabled() is True
    assert dlg.tooltip_previews_enabled() is True

    # 3. Originals tab controls
    assert dlg.retention_days() == 180
    assert dlg.cmb_retention.currentText() == "Six months"

    # Selecting "One month"
    dlg.cmb_retention.setCurrentIndex(3)
    assert dlg.retention_days() == 30

    # 4. Edit tab controls
    assert dlg.jpeg_quality() == 85
    assert dlg.auto_straighten() is True
    assert dlg.auto_noise() is False
    assert dlg.auto_color() is True
    assert dlg.auto_exposure() is True

    dlg.slider_jpeg_quality.setValue(92)
    assert dlg.jpeg_quality() == 92
    assert "92%" in dlg.grp_jpeg.title()

    # 5. Import tab dynamic example
    dlg.cmb_folder_naming.setCurrentIndex(1)  # Date + Name
    dlg.cmb_file_naming.setCurrentIndex(0)    # Original File Name
    dlg._update_import_example()
    assert "2026-10-17 Himalayas" in dlg.lbl_import_example.text()
