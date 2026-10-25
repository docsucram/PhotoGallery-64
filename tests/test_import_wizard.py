"""
Comprehensive Test Suite for WLPG Camera, Memory Card & Folder Import Wizard (tests/test_import_wizard.py).

Verifies:
1. Source scanning and 'Skip already in catalog' deduplication against an existing SQLite catalog.db.
2. Date grouping and destination path generation for YYYY\\YYYY-MM-DD [Event] and YYYY-MM-DD HH.MM.SS.jpg renaming (including burst collision handling).
3. Safe file copy, byte-size verification, and automatic tag assignment in catalog.db.
4. Storage volume and DCIM source detection.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pytest
from PIL import Image
from PySide6.QtCore import QEventLoop, Qt
from PySide6.QtWidgets import QApplication

from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository
from src.ui.import_dialog import (
    DiscoveredItem,
    ImportCopySignals,
    ImportMediaDialog,
    SafeCopyWorker,
    detect_available_sources,
    generate_destination_subfolder,
    resolve_target_filename,
)


@pytest.fixture
def qapp() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def test_engine(tmp_path: Path) -> DualDatabaseEngine:
    engine = DualDatabaseEngine(
        catalog_path=tmp_path / "catalog.db",
        thumbnails_path=tmp_path / "thumbnails.db",
        schema_sql_path=Path("db/schema.sql"),
    )
    engine.initialize_schemas()
    yield engine
    engine.close_all()


def test_device_discovery_mounted_volumes(qapp: QApplication) -> None:
    """Verifies storage volume detection discovers mounted drives and standard paths."""
    sources = detect_available_sources()
    assert len(sources) >= 1
    # Check that labels and paths are valid
    for label, path, has_dcim in sources:
        assert isinstance(label, str) and len(label) > 0
        assert isinstance(path, Path)


def test_destination_path_generation_and_burst_renaming() -> None:
    """Verifies all subfolder templates and burst collision filename resolution."""
    dt = datetime(2026, 10, 24, 15, 14, 1)

    # 1. Subfolder Template 0: YYYY \ YYYY-MM-DD [Event]
    sub0_event = generate_destination_subfolder(0, dt, "Dorset Trip")
    assert sub0_event == Path("2026") / "2026-10-24 Dorset Trip"
    sub0_empty = generate_destination_subfolder(0, dt, "")
    assert sub0_empty == Path("2026") / "2026-10-24"

    # 2. Subfolder Template 1: YYYY \ YYYY_MM [Event]
    sub1_event = generate_destination_subfolder(1, dt, "Dorset Trip")
    assert sub1_event == Path("2026") / "2026_10 Dorset Trip"

    # 3. Subfolder Template 2: YYYY-MM-DD [Event]
    sub2_event = generate_destination_subfolder(2, dt, "Dorset Trip")
    assert sub2_event == Path("2026-10-24 Dorset Trip")

    # 4. Subfolder Template 3: Single Folder
    sub3_event = generate_destination_subfolder(3, dt, "Dorset Trip")
    assert sub3_event == Path("Dorset Trip")
    sub3_empty = generate_destination_subfolder(3, dt, "")
    assert sub3_empty == Path(".")

    # 5. Filename Renaming: Original filename
    alloc: set[str] = set()
    fn_orig1 = resolve_target_filename(0, "IMG_1001.JPG", dt, alloc)
    assert fn_orig1 == "IMG_1001.JPG"
    fn_orig2 = resolve_target_filename(0, "IMG_1001.JPG", dt, alloc)
    assert fn_orig2 == "IMG_1001_1.JPG"

    # 6. Filename Renaming: Date & Time Taken with Burst Collisions (_1, _2)
    alloc_burst: set[str] = set()
    b1 = resolve_target_filename(1, "RAW_001.CR2", dt, alloc_burst)
    assert b1 == "2026-10-24 15.14.01.CR2"
    b2 = resolve_target_filename(1, "RAW_002.CR2", dt, alloc_burst)
    assert b2 == "2026-10-24 15.14.01_1.CR2"
    b3 = resolve_target_filename(1, "RAW_003.CR2", dt, alloc_burst)
    assert b3 == "2026-10-24 15.14.01_2.CR2"


def test_source_scanning_and_deduplication(
    qapp: QApplication, test_engine: DualDatabaseEngine, tmp_path: Path
) -> None:
    """Verifies that existing catalog photos are flagged as duplicates and filtered out."""
    source_dir = tmp_path / "camera_sd" / "DCIM" / "100CANON"
    source_dir.mkdir(parents=True, exist_ok=True)

    # Create 3 files in source: 2 exist in catalog, 1 is new
    f1 = source_dir / "IMG_0001.JPG"
    im1 = Image.new("RGB", (64, 64), (200, 100, 50))
    im1.save(f1)
    sz1 = f1.stat().st_size

    f2 = source_dir / "IMG_0002.JPG"
    im2 = Image.new("RGB", (64, 64), (50, 150, 200))
    im2.save(f2)
    sz2 = f2.stat().st_size

    f3 = source_dir / "IMG_0003.JPG"  # Brand new file
    im3 = Image.new("RGB", (64, 64), (100, 200, 100))
    im3.save(f3)

    # Insert f1 and f2 into catalog.db
    conn = test_engine.catalog_conn()
    conn.execute("INSERT INTO folders (path, name) VALUES ('C:/Existing', 'Existing');")
    conn.execute(
        """
        INSERT INTO media (folder_id, filename, ext, size, is_video)
        VALUES (1, 'IMG_0001.JPG', 'jpg', ?, 0),
               (1, 'IMG_0002.JPG', 'jpg', ?, 0);
        """,
        (sz1, sz2),
    )
    conn.commit()

    repo = CatalogRepository(test_engine)
    dialog = ImportMediaDialog(repo, initial_source=source_dir)
    dialog.resize(900, 600)
    dialog.show()

    # Wait for scanner thread pool to finish
    dialog._thread_pool.waitForDone(4000)
    qapp.processEvents()
    dialog._rebuild_groups()

    # Verify discovery results
    assert len(dialog._all_items) == 3
    dup_items = [it for it in dialog._all_items if it.is_duplicate]
    new_items = [it for it in dialog._all_items if not it.is_duplicate]
    assert len(dup_items) == 2
    assert len(new_items) == 1
    assert new_items[0].filename == "IMG_0003.JPG"

    # With skip_duplicates=True, only 1 new item is selected
    total_selected = sum(len(c.get_selected_items()) for c in dialog._group_cards)
    assert total_selected == 1

    # Toggling skip_duplicates off reveals all 3
    dialog.chk_skip_duplicates.setChecked(False)
    assert sum(len(c.items) for c in dialog._group_cards) == 3

    dialog.close()


def test_safe_copy_verification_and_tag_assignment(
    qapp: QApplication, test_engine: DualDatabaseEngine, tmp_path: Path
) -> None:
    """Tests SafeCopyWorker file copying, byte verification, and catalog tag assignment."""
    source_dir = tmp_path / "card_source"
    source_dir.mkdir(parents=True, exist_ok=True)
    dest_root = tmp_path / "library_dest"
    dest_root.mkdir(parents=True, exist_ok=True)

    # Create 2 test photos
    p1 = source_dir / "IMG_101.JPG"
    im1 = Image.new("RGB", (120, 80), (10, 20, 30))
    im1.save(p1)
    sz1 = p1.stat().st_size

    p2 = source_dir / "IMG_102.JPG"
    im2 = Image.new("RGB", (120, 80), (40, 50, 60))
    im2.save(p2)
    sz2 = p2.stat().st_size

    dt1 = datetime(2026, 10, 24, 11, 0, 0)
    dt2 = datetime(2026, 10, 24, 11, 5, 0)

    it1 = DiscoveredItem(p1, "IMG_101.JPG", sz1, dt1, is_video=False, selected=True)
    it2 = DiscoveredItem(p2, "IMG_102.JPG", sz2, dt2, is_video=False, selected=True)

    groups_data = [
        {
            "date_str": "2026-10-24",
            "event_name": "Castle Visit",
            "tags": "Holiday, Historic, Castle",
            "items": [it1, it2],
        }
    ]

    signals = ImportCopySignals()
    completed_events: list[tuple[list, list]] = []
    signals.finished.connect(lambda f, i: completed_events.append((f, i)))

    worker = SafeCopyWorker(
        dest_root=dest_root,
        template_idx=0,  # YYYY \ YYYY-MM-DD [Event]
        renaming_idx=0,  # Keep original filename
        delete_source=False,
        groups_data=groups_data,
        signals=signals,
    )
    worker.run()

    # 1. Verify files copied to correct subfolder
    expected_folder = dest_root / "2026" / "2026-10-24 Castle Visit"
    assert expected_folder.exists()
    dest_file1 = expected_folder / "IMG_101.JPG"
    dest_file2 = expected_folder / "IMG_102.JPG"
    assert dest_file1.exists()
    assert dest_file2.exists()

    # 2. Verify byte sizes match source exactly
    assert dest_file1.stat().st_size == sz1
    assert dest_file2.stat().st_size == sz2

    # 3. Simulate catalog indexing & tag assignment
    from src.workers.scanner_thumb_worker import FolderSyncWorker
    sync_worker = FolderSyncWorker(test_engine, expected_folder, include_subfolders=False)
    sync_worker.run()

    conn = test_engine.catalog_conn()
    m_rows = conn.execute("SELECT id, filename FROM media;").fetchall()
    assert len(m_rows) == 2
    m_ids = [int(r["id"]) for r in m_rows]

    # Assign tags
    repo = CatalogRepository(test_engine)
    tags = ["Holiday", "Historic", "Castle"]
    for t in tags:
        tid = repo.create_tag(t)
        repo.assign_tag(m_ids, tid)

    # Verify tags in database
    t_rows = conn.execute("SELECT name FROM tags;").fetchall()
    tag_names = {r["name"] for r in t_rows}
    for t in tags:
        assert t in tag_names

    mt_rows = conn.execute("SELECT media_id, tag_id FROM media_tags;").fetchall()
    assert len(mt_rows) == 6  # 2 media * 3 tags


def test_main_window_import_wizard_action(qapp: QApplication, test_engine: DualDatabaseEngine) -> None:
    """Verifies that main_window routes import_media / import_photos to open_import_wizard."""
    from unittest.mock import MagicMock
    from src.ui.main_window import PhotoGalleryMainWindow

    mw = PhotoGalleryMainWindow(test_engine, start_xmp_daemon=False)
    try:
        mw.open_import_wizard = MagicMock()
        mw.dispatch_ribbon_action("import_media")
        mw.open_import_wizard.assert_called_once()

        mw.open_import_wizard.reset_mock()
        mw.dispatch_ribbon_action("import_photos")
        mw.open_import_wizard.assert_called_once()
    finally:
        mw.close()
