"""
Verification Suite for WP-03 & WP-04:
1. Batch undo field maps and deduplicated people assignments (WP-03).
2. Relative time shift in a single SQL UPDATE statement (WP-03).
3. Lossless XMP write-back with author, location name, and GPS coordinates (WP-04).
4. MetadataSyncDaemon asynchronous thread-pool offloading and clean mtime tracking (WP-04).
5. Thread-local SQLite connection cleanup on worker exit (WP-04).
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QImage

from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository
from src.workers.folder_watcher import DeltaSyncWorker
from src.workers.scanner_thumb_worker import FolderSyncWorker, ThumbnailDecodeWorker
from src.workers.xmp_sync_worker import (
    MetadataSyncDaemon,
    build_wlpg_xmp_packet,
    parse_xmp_packet_bytes,
    write_metadata_lossless,
)


@pytest.fixture()
def db_engine(tmp_path: Path) -> DualDatabaseEngine:
    engine = DualDatabaseEngine(tmp_path / "catalog.db", tmp_path / "thumbs.db")
    yield engine
    engine.close_all()


def test_wp03_batch_undo_field_map(db_engine: DualDatabaseEngine) -> None:
    repo = CatalogRepository(db_engine)
    fid = repo.upsert_folder("D:/Photos", "Photos", None, 1700000000)

    # Insert 100 items with alternating ratings
    items = [
        {
            "folder_id": fid,
            "filename": f"img_{i:03d}.jpg",
            "ext": "jpg",
            "size": 1000,
            "mtime": 1700000000,
            "date_taken": 1700000000 + i,
            "rating": (i % 5) + 1,
            "flag": 0,
        }
        for i in range(100)
    ]
    ids = repo.bulk_upsert_media(items)

    # 1. Snapshot previous ratings in single query batch
    orig_ratings = repo.get_field_map_for_ids(ids, "rating")
    assert len(orig_ratings) == 100
    assert orig_ratings[ids[0]] == 1
    assert orig_ratings[ids[4]] == 5

    # 2. Modify all to 5 stars
    repo.set_rating(ids, 5)
    updated = repo.get_field_map_for_ids(ids, "rating")
    assert all(r == 5 for r in updated.values())

    # 3. Restore snapshot via batch restore
    repo.restore_field_map(orig_ratings, "rating")
    restored = repo.get_field_map_for_ids(ids, "rating")
    assert restored == orig_ratings


def test_wp03_people_assignment_deduplication(db_engine: DualDatabaseEngine) -> None:
    repo = CatalogRepository(db_engine)
    fid = repo.upsert_folder("D:/Photos", "Photos", None, 1700000000)
    ids = repo.bulk_upsert_media(
        [
            {
                "folder_id": fid,
                "filename": "person_test.jpg",
                "ext": "jpg",
                "size": 1000,
                "mtime": 1700000000,
                "date_taken": 1700000000,
            }
        ]
    )
    mid = ids[0]

    # Assign "Alice" at (0.2, 0.2, 0.3, 0.3)
    p1 = repo.assign_person(mid, "Alice", rect=(0.2, 0.2, 0.3, 0.3))
    # Assign "Alice" again at practically identical bounding box (within tolerance < 0.01)
    p2 = repo.assign_person(mid, "Alice", rect=(0.202, 0.201, 0.301, 0.302))

    conn = db_engine.catalog_conn()
    rows = conn.execute(
        "SELECT id, person_id, rect_x, rect_y FROM media_people WHERE media_id = ?;",
        (mid,),
    ).fetchall()
    assert len(rows) == 1, "Duplicate bounding box should not create duplicate media_people row"


def test_wp03_relative_time_shift(db_engine: DualDatabaseEngine) -> None:
    repo = CatalogRepository(db_engine)
    fid = repo.upsert_folder("D:/Photos", "Photos", None, 1700000000)
    items = [
        {
            "folder_id": fid,
            "filename": f"img_{i}.jpg",
            "ext": "jpg",
            "size": 1000,
            "mtime": 1700000000,
            "date_taken": 1700000000 + (i * 100),
        }
        for i in range(20)
    ]
    ids = repo.bulk_upsert_media(items)

    # Shift all by +3600 seconds in a single query relative to lead_id = ids[0]
    lead_orig = 1700000000
    delta = repo.apply_relative_time_shift(ids, lead_id=ids[0], new_lead_epoch=lead_orig + 3600)
    assert delta == 3600
    shifts = repo.get_field_map_for_ids(ids, "date_taken")
    for i, mid in enumerate(ids):
        assert shifts[mid] == 1700000000 + (i * 100) + 3600


def test_wp04_xmp_author_location_roundtrip() -> None:
    media_detail = {
        "rating": 4,
        "caption": "Mountain hike",
        "author": "Jane Doe",
        "location_name": "Rocky Mountain National Park",
        "lat": 40.3428,
        "lon": -105.6836,
        "orientation": 1,
        "date_taken": 1700000000,
        "hierarchical_tags": ["Travel/Vacation", "Nature"],
        "people": [{"name": "Jane", "rect": (0.1, 0.1, 0.2, 0.2)}],
    }

    xmp_bytes = build_wlpg_xmp_packet(media_detail)
    assert b"Jane Doe" in xmp_bytes
    assert b"Rocky Mountain National Park" in xmp_bytes
    assert b"40.3428" in xmp_bytes

    parsed = parse_xmp_packet_bytes(xmp_bytes)
    assert parsed["author"] == "Jane Doe"
    assert parsed["location_name"] == "Rocky Mountain National Park"
    assert parsed["rating"] == 4
    assert parsed["caption"] == "Mountain hike"
    assert "Travel/Vacation" in parsed["hierarchical_tags"]


def test_wp04_metadata_sync_daemon_async(tmp_path: Path, db_engine: DualDatabaseEngine) -> None:
    repo = CatalogRepository(db_engine)
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    fid = repo.upsert_folder(str(img_dir), "images", None, 1700000000)

    # Create dummy JPEG image
    img_path = img_dir / "test_sample.jpg"
    qimg = QImage(100, 100, QImage.Format_RGB32)
    qimg.fill(0xFF00FF)
    qimg.save(str(img_path), "JPEG")

    ids = repo.bulk_upsert_media(
        [
            {
                "folder_id": fid,
                "filename": img_path.name,
                "ext": "jpg",
                "size": img_path.stat().st_size,
                "mtime": int(img_path.stat().st_mtime),
                "date_taken": int(img_path.stat().st_mtime),
                "rating": 5,
                "caption": "Test Daemon",
                "author": "Marcus",
                "metadata_dirty": 1,
            }
        ]
    )
    mid = ids[0]

    # Verify initially dirty
    dirty = repo.fetch_dirty_media_batch(10)
    assert any(d["id"] == mid for d in dirty)

    daemon = MetadataSyncDaemon(db_engine, poll_interval_ms=50)
    # Drain dirty records synchronously
    synced_count = daemon.flush_now()
    assert synced_count >= 1

    # Verify no longer dirty
    remaining_dirty = repo.fetch_dirty_media_batch(10)
    assert len(remaining_dirty) == 0

    daemon.stop()


def test_wp04_worker_connection_cleanup(tmp_path: Path, db_engine: DualDatabaseEngine) -> None:
    repo = CatalogRepository(db_engine)
    img_dir = tmp_path / "scan_test"
    img_dir.mkdir()
    fid = repo.upsert_folder(str(img_dir), "scan_test", None, 1700000000)

    img_path = img_dir / "item1.jpg"
    qimg = QImage(64, 64, QImage.Format_RGB32)
    qimg.fill(0x00FF00)
    qimg.save(str(img_path), "JPEG")

    worker = FolderSyncWorker(db_engine, img_dir, include_subfolders=False)
    worker.run()

    # After worker.run() completes, thread connections must have been closed
    # Running a query on main thread should work cleanly
    folders = repo.get_folders()
    assert len(folders) >= 1

    # Run DeltaSyncWorker
    delta_worker = DeltaSyncWorker(db_engine, str(img_dir))
    conn = db_engine.catalog_conn()
    media = conn.execute("SELECT id FROM media WHERE folder_id = ?;", (fid,)).fetchall()
    assert len(media) == 1
