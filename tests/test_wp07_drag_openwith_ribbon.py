"""
Tests for Work Package 07:
1. Dynamic Responsive Ribbon button compaction across Wide (>=1180px), Medium (880px-1180px), and Narrow (<880px).
2. Ribbon tab header double-click to minimize and restore ribbon.
3. Drag-and-drop to LeftNavTree emitting clean files_dropped signal without IndexError.
4. Main window handling dropped internal catalog files and external filesystem files into folders.
5. Windows Open With registry app querying.
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest
from PySide6.QtCore import QPoint, Qt, QUrl, QMimeData, QSize
from PySide6.QtGui import QDropEvent
from PySide6.QtWidgets import QApplication, QTreeWidgetItem

from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository
from src.ui.grid_view import get_windows_open_with_apps
from src.ui.left_nav_dock import _LeftNavTree, _NODE_TYPE_ROLE, _NODE_PAYLOAD_ROLE
from src.ui.main_window import PhotoGalleryMainWindow
from src.ui.ribbon_toolbar import RibbonToolbar


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_ribbon_responsive_button_compaction(qapp: QApplication) -> None:
    """Verify Authentic WLPG Priority Collapse Order across Wide, Medium, Narrow tiers."""
    ribbon = RibbonToolbar()
    ribbon.show()

    # 1. Wide tier (>= 1220px)
    ribbon.resize(1280, 115)
    ribbon._apply_responsive_layout(1280)
    qapp.processEvents()
    for btn in ribbon._responsive_large_buttons:
        assert btn.toolButtonStyle() == Qt.ToolButtonTextUnderIcon
    for btn in ribbon.home_manage_rot_delete:
        assert btn.toolButtonStyle() == Qt.ToolButtonTextBesideIcon
    for btn in ribbon.home_qf_buttons:
        assert btn.toolButtonStyle() == Qt.ToolButtonTextBesideIcon

    # 2. Authentic WLPG Medium Tier (920px on Home tab):
    # - Batch people tag and Descriptive tag have full un-elided text ("..." not in btn.text())
    # - Rotate left / Rotate right / Delete and Months / Rated / Flagged are in Qt.ToolButtonIconOnly mode.
    # - Organize (batch, desc, caption, geotag, rate, flag) and Select all / copy / paste stay Expanded!
    ribbon.resize(920, 115)
    ribbon._apply_responsive_layout(920)
    qapp.processEvents()

    assert "..." not in ribbon.btn_batch_people.text()
    assert ribbon.btn_batch_people.text() == "Batch people tag"
    assert "..." not in ribbon.btn_tag_desc.text()
    assert ribbon.btn_tag_desc.text() == "Descriptive tag"

    for btn in ribbon.home_manage_rot_delete:
        assert btn.toolButtonStyle() == Qt.ToolButtonIconOnly
    for btn in ribbon.home_qf_buttons:
        assert btn.toolButtonStyle() == Qt.ToolButtonIconOnly

    for btn in ribbon.home_manage_select_copy_paste:
        assert btn.toolButtonStyle() == Qt.ToolButtonTextBesideIcon
    for btn in ribbon.home_org_geotag_rate_flag:
        assert btn.toolButtonStyle() == Qt.ToolButtonTextBesideIcon

    # 3. Narrow tier (< 740px)
    ribbon.resize(700, 115)
    ribbon._apply_responsive_layout(700)
    qapp.processEvents()
    for btn in ribbon.home_org_batch_desc_caption:
        assert btn.toolButtonStyle() == Qt.ToolButtonIconOnly

    # 4. Zero Overlap check from 1400px down to 700px:
    # Quick find face gallery geometry().right() is strictly < Months button's geometry().left()
    for w in (1400, 1280, 1150, 1050, 950, 920, 850, 780, 700):
        ribbon.resize(w, 115)
        ribbon._apply_responsive_layout(w)
        ribbon.updateGeometry()
        ribbon.layout().activate()
        qapp.processEvents()
        fc_right = ribbon.home_face_carousel.geometry().right()
        months_left = ribbon.btn_qf_months.geometry().left()
        assert fc_right < months_left, f"Overlap at width={w}: carousel right ({fc_right}) >= months left ({months_left})"

    ribbon.close()


def test_ribbon_tab_header_double_click_minimize(qapp: QApplication) -> None:
    """Verify double-clicking ribbon tab headers minimizes and restores the ribbon."""
    ribbon = RibbonToolbar()
    ribbon.show()

    assert not ribbon._ribbon_collapsed
    assert ribbon.collapse_btn.text() == "▲"

    # Simulate double-clicking tab bar at index 0 (Home)
    ribbon.tab_widget.tabBarDoubleClicked.emit(0)
    assert ribbon._ribbon_collapsed
    assert ribbon.collapse_btn.text() == "▼"

    # Double click again to restore
    ribbon.tab_widget.tabBarDoubleClicked.emit(0)
    assert not ribbon._ribbon_collapsed
    assert ribbon.collapse_btn.text() == "▲"

    # Single click when collapsed restores ribbon
    ribbon._toggle_ribbon_collapse()
    assert ribbon._ribbon_collapsed
    ribbon.tab_widget.tabBarClicked.emit(1)
    assert not ribbon._ribbon_collapsed

    ribbon.close()


def test_left_nav_tree_drop_signal_no_index_error(qapp: QApplication, tmp_path: Path) -> None:
    """Verify _LeftNavTree.dropEvent emits files_dropped without IndexError."""
    tree = _LeftNavTree()
    tree.show()

    emitted_payloads = []

    def _on_dropped(folder_id: int, folder_path: str, file_paths: list, is_copy: bool) -> None:
        emitted_payloads.append((folder_id, folder_path, file_paths, is_copy))

    tree.files_dropped.connect(_on_dropped)

    # Populate dummy folder item
    target_dir = str(tmp_path / "TargetFolder")
    os.makedirs(target_dir, exist_ok=True)
    item = QTreeWidgetItem(tree)
    item.setText(0, "TargetFolder")
    item.setToolTip(0, target_dir)
    item.setData(0, _NODE_TYPE_ROLE, "folder")
    item.setData(0, _NODE_PAYLOAD_ROLE, 42)

    # Construct drop event on item
    mime_data = QMimeData()
    sample_file = tmp_path / "sample.jpg"
    sample_file.write_bytes(b"\xFF\xD8\xFF\xE0\x00\x10JFIF")
    mime_data.setUrls([QUrl.fromLocalFile(str(sample_file))])

    # Position event over target folder item
    visual_rect = tree.visualItemRect(item)
    drop_point = visual_rect.center()

    drop_event = QDropEvent(
        drop_point,
        Qt.MoveAction,
        mime_data,
        Qt.LeftButton,
        Qt.NoModifier,
    )

    # Execute dropEvent - must not raise IndexError: Signature not found
    tree.dropEvent(drop_event)

    assert len(emitted_payloads) == 1
    fid, fpath, fpaths, is_copy = emitted_payloads[0]
    assert fid == 42
    assert os.path.normpath(fpath) == os.path.normpath(target_dir)
    assert any(os.path.normpath(p) == os.path.normpath(sample_file) for p in fpaths)
    assert is_copy is False

    tree.close()


def test_main_window_handles_external_and_catalog_dropped_files(
    qapp: QApplication, tmp_path: Path
) -> None:
    """Verify PhotoGalleryMainWindow._on_files_dropped_on_folder handles external and catalog files."""
    engine = DualDatabaseEngine(tmp_path / "catalog.db", tmp_path / "thumbs.db")
    repo = CatalogRepository(engine)

    src_dir = tmp_path / "SrcDir"
    dst_dir = tmp_path / "DstDir"
    src_dir.mkdir()
    dst_dir.mkdir()

    src_fid = repo.upsert_folder(str(src_dir), "SrcDir", None, 1700000000)
    dst_fid = repo.upsert_folder(str(dst_dir), "DstDir", None, 1700000000)

    # 1. Existing catalog item
    cat_file = src_dir / "catalog_img.jpg"
    cat_file.write_bytes(b"\xFF\xD8\xFF\xE0\x00\x10JFIF")
    ids = repo.bulk_upsert_media(
        [
            {
                "folder_id": src_fid,
                "filename": "catalog_img.jpg",
                "ext": ".jpg",
                "size": 100,
                "mtime": 1700000000,
                "is_video": 0,
            }
        ]
    )
    cat_mid = ids[0]

    # 2. External uncataloged file
    ext_dir = tmp_path / "External"
    ext_dir.mkdir()
    ext_file = ext_dir / "external_photo.jpg"
    ext_file.write_bytes(b"\xFF\xD8\xFF\xE0\x00\x10JFIF")

    win = PhotoGalleryMainWindow(engine, start_xmp_daemon=False)

    # Drop both files onto dst_fid (move)
    win._on_files_dropped_on_folder(
        dst_fid,
        str(dst_dir),
        [str(cat_file), str(ext_file)],
        is_copy=False,
    )

    # Verify catalog file moved
    assert not cat_file.exists()
    assert (dst_dir / "catalog_img.jpg").exists()

    # Verify external file moved
    assert not ext_file.exists()
    assert (dst_dir / "external_photo.jpg").exists()

    # Verify DB record updated for catalog item
    detail = repo.get_media_detail(cat_mid)
    assert detail is not None
    assert detail["folder_id"] == dst_fid

    win.close()


def test_windows_open_with_apps_query() -> None:
    """Verify get_windows_open_with_apps returns a list without crashing on Windows."""
    apps = get_windows_open_with_apps("jpg")
    assert isinstance(apps, list)
    # Each entry should be a tuple of (name, command)
    for item in apps:
        assert isinstance(item, tuple)
        assert len(item) == 2
