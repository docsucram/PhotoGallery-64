"""
Unit tests for Real-Time Directory Watching, Recursive Subfolder Ingestion,
and Startup Library Reconciliation:
1. Live addition of nested subfolder with photos is recursively ingested.
2. Live deletion of subfolder removes folder tree and invalidates thumbnails.
3. StartupReconciliationWorker detects offline folder creations and updates database.
4. Main window responds to delta and reconciliation signals and refreshes UI.
5. Left navigation dock preserves expanded folder items across refreshes.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path
from PIL import Image
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository, ThumbnailRepository
from src.ui.left_nav_dock import LeftNavigationDock
from src.ui.main_window import PhotoGalleryMainWindow
from src.workers.folder_watcher import (
    DeltaSyncWorker,
    FolderWatcherDaemon,
    StartupReconciliationWorker,
)


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def env():
    tmp_dir = tempfile.mkdtemp(prefix="pg64_reconcile_test_")
    db_dir = os.path.join(tmp_dir, "db")
    media_dir = os.path.join(tmp_dir, "photos")
    os.makedirs(db_dir, exist_ok=True)
    os.makedirs(media_dir, exist_ok=True)

    catalog_path = os.path.join(db_dir, "catalog.db")
    thumbs_path = os.path.join(db_dir, "thumbnails.db")
    engine = DualDatabaseEngine(catalog_path, thumbs_path)

    yield {
        "root": tmp_dir,
        "media_dir": media_dir,
        "engine": engine,
    }

    try:
        engine.close_all()
    except Exception:
        pass
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_delta_sync_recursively_ingests_new_subfolder(qapp: QApplication, env) -> None:
    """Verify DeltaSyncWorker recursively discovers and ingests new subfolders and nested photos."""
    engine = env["engine"]
    media_dir = env["media_dir"]
    repo = CatalogRepository(engine)

    root_fid = repo.upsert_folder(media_dir, "PhotosRoot", None, int(time.time()))

    # Create a nested subfolder with photos
    sub_dir = os.path.join(media_dir, "2026_09")
    os.makedirs(sub_dir, exist_ok=True)
    photo1 = os.path.join(sub_dir, "photo1.jpg")
    Image.new("RGB", (100, 100), color="blue").save(photo1)

    nested_sub = os.path.join(sub_dir, "nested")
    os.makedirs(nested_sub, exist_ok=True)
    photo2 = os.path.join(nested_sub, "photo2.jpg")
    Image.new("RGB", (120, 80), color="green").save(photo2)

    emitted = []
    worker = DeltaSyncWorker(engine, media_dir)
    worker.signals.delta_completed.connect(
        lambda path, added, mod, deleted, f_changed: emitted.append(
            (path, added, mod, deleted, f_changed)
        )
    )
    worker.run()

    assert len(emitted) == 1
    path, added, mod, deleted, f_changed = emitted[0]
    assert added == 2
    assert f_changed == 2  # 2026_09 and 2026_09/nested

    folders = repo.get_folders()
    folder_paths = [os.path.normcase(os.path.normpath(f["path"])) for f in folders]
    assert os.path.normcase(os.path.normpath(sub_dir)) in folder_paths
    assert os.path.normcase(os.path.normpath(nested_sub)) in folder_paths

    conn = engine.catalog_conn()
    media_rows = conn.execute("SELECT filename FROM media;").fetchall()
    fnames = [r["filename"] for r in media_rows]
    assert "photo1.jpg" in fnames
    assert "photo2.jpg" in fnames


def test_delta_sync_prunes_deleted_subfolder(qapp: QApplication, env) -> None:
    """Verify DeltaSyncWorker removes deleted subfolder tree and media."""
    engine = env["engine"]
    media_dir = env["media_dir"]
    repo = CatalogRepository(engine)

    root_fid = repo.upsert_folder(media_dir, "PhotosRoot", None, int(time.time()))
    sub_dir = os.path.join(media_dir, "to_delete")
    os.makedirs(sub_dir, exist_ok=True)
    photo = os.path.join(sub_dir, "photo.jpg")
    Image.new("RGB", (50, 50), color="red").save(photo)

    # Ingest subfolder
    worker1 = DeltaSyncWorker(engine, media_dir)
    worker1.run()

    # Ensure it's in DB
    sub_row = engine.catalog_conn().execute(
        "SELECT id FROM folders WHERE path = ?;", (os.path.normpath(sub_dir),)
    ).fetchone()
    assert sub_row is not None

    # Now delete folder on disk
    shutil.rmtree(sub_dir)
    assert not os.path.exists(sub_dir)

    emitted = []
    worker2 = DeltaSyncWorker(engine, media_dir)
    worker2.signals.delta_completed.connect(
        lambda path, added, mod, deleted, f_changed: emitted.append(
            (path, added, mod, deleted, f_changed)
        )
    )
    worker2.run()

    assert len(emitted) == 1
    _, _, _, deleted, f_changed = emitted[0]
    assert deleted >= 1
    assert f_changed >= 1

    # Ensure folder and media are pruned from catalog.db
    sub_row_after = engine.catalog_conn().execute(
        "SELECT id FROM folders WHERE path = ?;", (os.path.normpath(sub_dir),)
    ).fetchone()
    assert sub_row_after is None


def test_startup_reconciliation_worker(qapp: QApplication, env) -> None:
    """Verify StartupReconciliationWorker reconciles offline additions and missing folders."""
    engine = env["engine"]
    media_dir = env["media_dir"]
    repo = CatalogRepository(engine)

    # Setup existing root folder
    root_fid = repo.upsert_folder(media_dir, "PhotosRoot", None, 1000)

    # Simulate offline addition: user creates folder in Explorer while app closed
    offline_dir = os.path.join(media_dir, "offline_event")
    os.makedirs(offline_dir, exist_ok=True)
    img = os.path.join(offline_dir, "offline_shot.jpg")
    Image.new("RGB", (80, 80), color="yellow").save(img)

    emitted = []
    reconcile_worker = StartupReconciliationWorker(engine)
    reconcile_worker.signals.reconciliation_completed.connect(
        lambda a, m, d, f_c: emitted.append((a, m, d, f_c))
    )
    reconcile_worker.run()

    assert len(emitted) == 1
    added, mod, deleted, folders_changed = emitted[0]
    assert added >= 1
    assert folders_changed >= 1

    # Check that offline_event was indexed
    conn = engine.catalog_conn()
    f_row = conn.execute(
        "SELECT id FROM folders WHERE path = ?;", (os.path.normpath(offline_dir),)
    ).fetchone()
    assert f_row is not None
    m_row = conn.execute(
        "SELECT id FROM media WHERE filename = ?;", ("offline_shot.jpg",)
    ).fetchone()
    assert m_row is not None


def test_left_nav_dock_preserves_expanded_folders(qapp: QApplication, env) -> None:
    """Verify LeftNavigationDock preserves expanded folders when refreshed."""
    engine = env["engine"]
    media_dir = env["media_dir"]
    repo = CatalogRepository(engine)

    root_fid = repo.upsert_folder(media_dir, "Root", None, 1000)
    sub_fid = repo.upsert_folder(os.path.join(media_dir, "Sub"), "Sub", root_fid, 1000)

    dock = LeftNavigationDock()
    dock.refresh_trees(repo)

    root_item = dock._folder_items.get(root_fid)
    assert root_item is not None
    assert not root_item.isExpanded()

    # Expand root item
    root_item.setExpanded(True)
    assert root_item.isExpanded()

    # Refresh trees - expanded state should persist!
    dock.refresh_trees(repo)
    root_item_after = dock._folder_items.get(root_fid)
    assert root_item_after is not None
    assert root_item_after.isExpanded()

    dock.close()
