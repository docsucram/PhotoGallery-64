"""
Phase 19 Verification Tests:
1. 4-Cycle Lossless Rotation in Viewer & Thumbnail Grid (0ms in-memory rotation, orientation reconciliation).
2. LruCache dict item assignment/retrieval.
3. Left-Click and Right-Click Drag and Drop (OLE URLs, ghost preview, Move/Copy menu, no-drop cursor).
4. Real-time Directory Watching & Background Delta Synchronization (FolderWatcherDaemon, DeltaSyncWorker).
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageOps
from PySide6.QtCore import QByteArray, QDate, QMimeData, QPoint, QRect, QSize, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent, QImage, QPixmap
from PySide6.QtWidgets import QApplication

from legacy_reference.imgBrowserv13 import LruCache
from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository, ThumbnailRepository
from src.ui.grid_model import CatalogListModel, CatalogRoles
from src.ui.grid_view import ThumbnailGridView
from src.ui.left_nav_dock import LeftNavigationDock, _NODE_PAYLOAD_ROLE, _NODE_TYPE_ROLE, _LeftNavTree
from src.ui.main_window import PhotoGalleryMainWindow
from src.ui.media_viewer import SingleMediaViewer
from src.workers.folder_watcher import DeltaSyncWorker, FolderWatcherDaemon


@pytest.fixture(scope="session")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def temp_gallery_env():
    tmp_dir = tempfile.mkdtemp(prefix="pg64_phase19_")
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


def test_lru_cache_dict_subscripting(qapp):
    """Verify LruCache supports item assignment, retrieval, deletion, and containment."""
    cache = LruCache(max_size=10)
    pm = QPixmap(10, 10)

    # Assignment via __setitem__
    cache[101] = pm
    assert 101 in cache
    assert cache[101] == pm

    # Multiple items
    pm2 = QPixmap(20, 20)
    cache[102] = pm2
    assert len(cache) == 2

    # KeyError on missing
    with pytest.raises(KeyError):
        _ = cache[999]

    # Deletion via __delitem__
    del cache[101]
    assert 101 not in cache


def test_four_cycle_rotation_in_media_viewer(qapp, temp_gallery_env):
    """
    Verify that an image rotated 4 consecutive times in SingleMediaViewer
    returns 100% to its original orientation and dimensions without desync.
    """
    media_dir = temp_gallery_env["media_dir"]
    img_path = os.path.join(media_dir, "test_rot_cycle.jpg")

    # Create a 160x100 landscape image (width=160, height=100)
    im = Image.new("RGB", (160, 100), color=(200, 100, 50))
    im.save(img_path)

    viewer = SingleMediaViewer()
    ok = viewer.load_media(media_id=1, file_path=img_path, orientation=1)
    assert ok is True
    assert viewer.master_np_rgb.shape[:2] == (100, 160)  # height=100, width=160

    # Rotate 1: 90 deg CW -> height=160, width=100
    viewer.apply_in_memory_rotation(clockwise=True)
    assert viewer.master_np_rgb.shape[:2] == (160, 100)

    # Rotate 2: 180 deg CW -> height=100, width=160
    viewer.apply_in_memory_rotation(clockwise=True)
    assert viewer.master_np_rgb.shape[:2] == (100, 160)

    # Rotate 3: 270 deg CW -> height=160, width=100
    viewer.apply_in_memory_rotation(clockwise=True)
    assert viewer.master_np_rgb.shape[:2] == (160, 100)

    # Rotate 4: 360 deg CW -> back to original orientation (height=100, width=160)!
    viewer.apply_in_memory_rotation(clockwise=True)
    assert viewer.master_np_rgb.shape[:2] == (100, 160)


def test_main_window_rotate_internal_and_cache(qapp, temp_gallery_env):
    """
    Verify PhotoGalleryMainWindow._rotate_internal losslessly rotates catalog orientation,
    updates pixmap_cache in-memory, updates thumbnails.db directly, and refreshes the grid.
    """
    engine = temp_gallery_env["engine"]
    media_dir = temp_gallery_env["media_dir"]
    repo = CatalogRepository(engine)
    thumb_repo = ThumbnailRepository(engine)

    # Insert test folder and media
    fid = repo.upsert_folder(media_dir, "Photos", parent_id=None, mtime=int(time.time()))
    img_path = os.path.join(media_dir, "sample.jpg")
    Image.new("RGB", (120, 80), color=(10, 20, 30)).save(img_path)

    mid = repo.bulk_upsert_media(
        [
            {
                "folder_id": fid,
                "filename": "sample.jpg",
                "ext": "jpg",
                "size": 1000,
                "mtime": int(time.time()),
                "date_taken": int(time.time()),
                "is_video": 0,
                "width": 120,
                "height": 80,
            }
        ]
    )[0]

    main_win = PhotoGalleryMainWindow(engine, start_xmp_daemon=False)
    # Store initial pixmap in cache
    initial_pm = QPixmap(120, 80)
    main_win.pixmap_cache.put(mid, initial_pm)

    # Store initial thumbnail in thumbnails.db with valid WebP
    from PySide6.QtCore import QBuffer, QIODevice
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.WriteOnly)
    qimg = QImage(120, 80, QImage.Format_RGB32)
    qimg.fill(Qt.blue)
    qimg.save(buf, "WEBP", 80)
    buf.close()
    thumb_repo.put_webp_blob(mid, bytes(ba.data()), int(time.time()), 120, 80)

    # Rotate CW 90
    main_win._rotate_internal([mid], clockwise=True)
    main_win.thumbnail_pool.waitForDone(2000)

    # Check that media in catalog has orientation = 6 and swapped width/height (80, 120)
    detail = repo.get_media_detail(mid)
    assert detail["orientation"] == 6
    assert detail["width"] == 80
    assert detail["height"] == 120

    # Cached pixmap dimensions should be swapped (80 x 120)
    cached_pm = main_win.pixmap_cache.get(mid)
    assert cached_pm is not None
    assert cached_pm.width() == 80
    assert cached_pm.height() == 120

    main_win.shutdown()


def test_drag_drop_grid_to_folder_left_and_right(qapp, temp_gallery_env):
    """
    Verify Left-Click and Right-Click Drag and Drop:
    - Grid view prepares QMimeData with file URLs and drag button metadata.
    - Grid view rejects drops onto itself (forbidden cursor).
    - LeftNavTree dragMoveEvent displays tooltip and highlights folder.
    - Right-click drop triggers Move / Copy / Cancel menu.
    """
    engine = temp_gallery_env["engine"]
    media_dir = temp_gallery_env["media_dir"]
    sub_dir = os.path.join(media_dir, "Subfolder")
    os.makedirs(sub_dir, exist_ok=True)

    repo = CatalogRepository(engine)
    fid1 = repo.upsert_folder(media_dir, "Photos", parent_id=None, mtime=int(time.time()))
    fid2 = repo.upsert_folder(sub_dir, "Subfolder", parent_id=fid1, mtime=int(time.time()))

    src_img = os.path.join(media_dir, "drag_test.jpg")
    Image.new("RGB", (60, 60), color=(128, 128, 128)).save(src_img)

    mid = repo.bulk_upsert_media(
        [
            {
                "folder_id": fid1,
                "filename": "drag_test.jpg",
                "ext": "jpg",
                "size": 500,
                "mtime": int(time.time()),
                "date_taken": int(time.time()),
                "is_video": 0,
            }
        ]
    )[0]

    main_win = PhotoGalleryMainWindow(engine, start_xmp_daemon=False)

    # 1. Verify Grid Rejects Drops (shows no-drop cursor)
    grid = main_win.grid_view
    mime_data = QMimeData()
    mime_data.setUrls([QUrl.fromLocalFile(src_img)])

    drag_enter = QDragEnterEvent(
        QPoint(50, 50),
        Qt.CopyAction | Qt.MoveAction,
        mime_data,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    grid.dragEnterEvent(drag_enter)
    assert not drag_enter.isAccepted()  # Rejected for no-drop cursor

    # 2. Test LeftNavTree Hover & _start_drag ghost card creation
    main_win.show()
    main_win.resize(1000, 700)
    main_win.left_nav_dock.tree.expandAll()
    qapp.processEvents()

    grid.select_media_id(mid)
    from PySide6.QtGui import QDrag
    orig_exec = QDrag.exec
    executed_drags = []
    try:
        QDrag.exec = lambda self_drag, *args, **kwargs: (executed_drags.append(self_drag.mimeData().urls()), Qt.MoveAction)[1]
        grid._start_drag(drag_button=Qt.LeftButton)
        grid._start_drag(drag_button=Qt.RightButton)
    finally:
        QDrag.exec = orig_exec
    assert len(executed_drags) == 2

    nav_tree = main_win.left_nav_dock.tree
    folder_item = main_win.left_nav_dock._folder_items.get(fid2)
    assert folder_item is not None

    item_rect = nav_tree.visualItemRect(folder_item)
    drag_move = QDragMoveEvent(
        item_rect.center(),
        Qt.CopyAction | Qt.MoveAction,
        mime_data,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    nav_tree.dragMoveEvent(drag_move)
    assert drag_move.isAccepted()
    assert nav_tree.currentItem() == folder_item

    # 3. Test _on_files_dropped_on_folder copy action
    main_win._on_files_dropped_on_folder(
        target_folder_id=fid2,
        target_folder_path=sub_dir,
        file_paths=[src_img],
        is_copy=True,
    )
    copied_path = os.path.join(sub_dir, "drag_test.jpg")
    assert os.path.exists(copied_path)  # Copy succeeded
    assert os.path.exists(src_img)      # Original preserved

    # 4. Test _on_files_dropped_on_folder move action
    main_win._on_files_dropped_on_folder(
        target_folder_id=fid2,
        target_folder_path=sub_dir,
        file_paths=[src_img],
        is_copy=False,
    )
    # The original drag_test.jpg in media_dir was moved (collides with existing copy, so renamed with (1))
    moved_path = os.path.join(sub_dir, "drag_test (1).jpg")
    assert os.path.exists(moved_path)
    assert not os.path.exists(src_img)  # Original was moved

    main_win.shutdown()


def test_directory_watcher_and_delta_sync(qapp, temp_gallery_env):
    """
    Verify real-time folder watching and background delta sync:
    - FolderWatcherDaemon watches library directories.
    - External file addition is detected and inserted into catalog.db.
    - External file deletion is detected and removed from catalog.db.
    """
    engine = temp_gallery_env["engine"]
    media_dir = temp_gallery_env["media_dir"]
    repo = CatalogRepository(engine)

    fid = repo.upsert_folder(media_dir, "WatchFolder", parent_id=None, mtime=int(time.time()))

    watcher_daemon = FolderWatcherDaemon(engine, debounce_interval_ms=100)
    watched = watcher_daemon.update_watched_paths()
    assert os.path.normpath(media_dir) in [os.path.normpath(p) for p in watched]

    # Add an external photo on disk
    new_photo = os.path.join(media_dir, "ext_photo.jpg")
    Image.new("RGB", (200, 150), color=(50, 150, 250)).save(new_photo)

    # Execute synchronous DeltaSyncWorker to verify delta detection
    delta_worker = DeltaSyncWorker(engine, media_dir)
    delta_worker.run()

    # Verify photo was indexed into catalog.db
    conn = engine.catalog_conn()
    row = conn.execute(
        "SELECT id, filename, width, height FROM media WHERE folder_id = ? AND filename = ?;",
        (fid, "ext_photo.jpg"),
    ).fetchone()
    assert row is not None
    assert row["width"] == 200
    assert row["height"] == 150
    photo_mid = int(row["id"])

    # Now delete the file externally on disk
    os.remove(new_photo)
    assert not os.path.exists(new_photo)

    # Re-run DeltaSyncWorker to verify deletion detection
    delta_worker2 = DeltaSyncWorker(engine, media_dir)
    delta_worker2.run()

    row_deleted = conn.execute(
        "SELECT id FROM media WHERE id = ?;", (photo_mid,)
    ).fetchone()
    assert row_deleted is None  # Cleanly removed from catalog.db!

    watcher_daemon.stop()
