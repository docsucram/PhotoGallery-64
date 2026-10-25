"""
Test Suite for Production-Readiness Safeguards (tests/test_production_safeguards.py).

Verifies:
1. Pause File Write-Back (Safe Mode):
   - MetadataSyncDaemon in paused mode keeps metadata_dirty = 1 queued without touching disk.
   - Unpausing flushes normally to disk.
2. Online SQLite Backup & Query Planner Statistics:
   - PRAGMA optimize runs on connection close.
   - ANALYZE populates sqlite_stat1 query planner statistics.
   - backup_catalog() creates a consistent, validated SQLite backup snapshot using sqlite3.Connection.backup().
3. Camera RAW Format Isolation:
   - Proprietary RAW extensions (.raf, .cr2, .cr3, .nef, .arw, .dng, .orf, .rw2) write XMP changes
     strictly to .xmp sidecars without modifying RAW container bytes.
4. Status Bar Worker Monitor & Controls:
   - Live worker monitor formatting and Pause Workers toggle button behavior.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication

from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository, ThumbnailRepository
from src.ui.main_window import PhotoGalleryMainWindow
from src.workers.scanner_thumb_worker import IMAGE_EXT_SET, RAW_EXT_SET
from src.workers.xmp_sync_worker import (
    MetadataSyncDaemon,
    RAW_EXT_SET as WORKER_RAW_EXT_SET,
    read_xmp_from_file,
    write_metadata_lossless,
)


@pytest.fixture(scope="session")
def qapp() -> QApplication:
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


def test_pause_file_write_back_safe_mode(tmp_path: Path, test_engine: DualDatabaseEngine) -> None:
    """Verifies that Safe Mode pause prevents disk file writes while keeping metadata_dirty = 1 queued."""
    repo = CatalogRepository(test_engine)

    img_dir = tmp_path / "Photos"
    img_dir.mkdir()
    img_file = img_dir / "safe_test.jpg"
    Image.new("RGB", (100, 100), color="blue").save(img_file, "JPEG")

    fid = repo.upsert_folder(str(img_dir), "Photos", None, 1700000000)
    mid = repo.bulk_upsert_media([
        {
            "folder_id": fid,
            "filename": img_file.name,
            "ext": "jpg",
            "size": img_file.stat().st_size,
            "mtime": 1700000000,
            "date_taken": 1700000000,
            "rating": 0,
            "flag": 0,
        }
    ])[0]

    orig_mtime = img_file.stat().st_mtime
    orig_bytes = img_file.read_bytes()

    # Rate photo to 5 stars (sets metadata_dirty = 1)
    repo.set_rating([mid], 5)
    assert repo.count_dirty_media() == 1

    daemon = MetadataSyncDaemon(test_engine, poll_interval_ms=500)
    assert not daemon.is_paused()

    # Enable Safe Mode
    daemon.set_paused(True)
    assert daemon.is_paused()

    # Attempt to flush while paused: should do nothing
    synced_cnt = daemon.flush_now(force=False)
    assert synced_cnt == 0

    # Verify disk file is 100% untouched
    assert img_file.stat().st_mtime == orig_mtime
    assert img_file.read_bytes() == orig_bytes
    assert repo.count_dirty_media() == 1

    # Unpause and flush
    daemon.set_paused(False)
    assert not daemon.is_paused()

    synced_cnt = daemon.flush_now()
    assert synced_cnt == 1
    assert repo.count_dirty_media() == 0

    # Verify metadata was now written
    xmp_meta = read_xmp_from_file(img_file)
    assert xmp_meta.get("rating") == 5


def test_online_sqlite_backup_and_analyze(tmp_path: Path, test_engine: DualDatabaseEngine) -> None:
    """Verifies PRAGMA optimize, ANALYZE query planner statistics, and online catalog.db backup."""
    repo = CatalogRepository(test_engine)
    fid = repo.upsert_folder("D:/Photos", "Photos", None, 1700000000)
    repo.bulk_upsert_media([
        {
            "folder_id": fid,
            "filename": f"photo_{i:04d}.jpg",
            "ext": "jpg",
            "size": 1000 + i,
            "mtime": 1700000000 + i,
            "date_taken": 1700000000 + i,
            "rating": (i % 5) + 1,
            "flag": i % 2,
        }
        for i in range(50)
    ])

    # 1. Run ANALYZE and verify sqlite_stat1 is generated
    test_engine.analyze_catalog()
    cat_conn = test_engine.catalog_conn()
    stat_row = cat_conn.execute("SELECT COUNT(*) AS cnt FROM sqlite_stat1;").fetchone()
    assert stat_row is not None
    assert int(stat_row["cnt"]) > 0

    # 2. Test online backup using sqlite3.Connection.backup()
    backup_file = tmp_path / "backups" / "catalog_snapshot.db"
    test_engine.backup_catalog(backup_file)
    assert backup_file.exists()
    assert backup_file.stat().st_size > 0

    # Verify integrity and row count of backup database
    b_conn = sqlite3.connect(str(backup_file))
    b_conn.row_factory = sqlite3.Row
    integrity = b_conn.execute("PRAGMA integrity_check;").fetchone()
    assert integrity[0] == "ok"
    cnt = b_conn.execute("SELECT COUNT(*) AS cnt FROM media;").fetchone()["cnt"]
    assert cnt == 50
    b_conn.close()

    # 3. Verify close_thread_connections and close_all run PRAGMA optimize without errors
    test_engine.close_thread_connections()
    test_engine.close_all()


def test_raw_camera_extensions_route_to_sidecar_only(tmp_path: Path) -> None:
    """Verifies that all RAW extensions write to .xmp sidecars without modifying RAW container bytes."""
    test_extensions = [".raf", ".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2"]

    for ext in test_extensions:
        assert ext in WORKER_RAW_EXT_SET
        assert ext.lstrip(".") in RAW_EXT_SET
        assert ext.lstrip(".") in IMAGE_EXT_SET

        raw_file = tmp_path / f"test_shot{ext}"
        dummy_raw_bytes = f"PROPRIETARY_RAW_CONTAINER_BYTES_{ext}".encode("ascii") * 10
        raw_file.write_bytes(dummy_raw_bytes)

        orig_st = raw_file.stat()

        media_detail = {
            "id": 100,
            "folder_path": str(tmp_path),
            "filename": raw_file.name,
            "rating": 4,
            "caption": f"Landscape shot in {ext}",
            "tags": [{"name": "Landscape", "full_path": "Travel/Landscape"}],
            "people": [],
            "orientation": 1,
            "date_taken": 1700000000,
            "size": len(dummy_raw_bytes),
        }

        # Write metadata
        res_mtime = write_metadata_lossless(media_detail)
        assert res_mtime is not None

        # Verify RAW container was NEVER modified (byte-for-byte identical)
        assert raw_file.read_bytes() == dummy_raw_bytes
        assert raw_file.stat().st_size == orig_st.st_size

        # Verify .xmp sidecar was created and contains the metadata
        sidecar_file = tmp_path / f"test_shot{ext}.xmp"
        assert sidecar_file.exists()
        sidecar_content = sidecar_file.read_text(encoding="utf-8")
        assert "xmp:Rating>4" in sidecar_content or "xmp:Rating=\"4\"" in sidecar_content
        assert "Landscape" in sidecar_content


def test_main_window_safeguards_ui(qapp: QApplication, test_engine: DualDatabaseEngine) -> None:
    """Verifies MainWindow Safe Mode write-back toggle, worker pause, and monitor widget."""
    win = PhotoGalleryMainWindow(test_engine, start_xmp_daemon=False)
    win.show()

    # 1. Initial monitor state
    assert hasattr(win, "worker_monitor_widget")
    assert hasattr(win, "lbl_worker_monitor")
    assert hasattr(win, "btn_pause_workers")
    win._update_worker_monitor()
    assert "Scanning: Idle" in win.lbl_worker_monitor.toolTip() or "Scanning: Idle" in win.lbl_worker_monitor.text()
    assert "Faces: Idle" in win.lbl_worker_monitor.toolTip() or "Faces: Idle" in win.lbl_worker_monitor.text()
    assert "Sync: Idle" in win.lbl_worker_monitor.toolTip() or "Sync: Idle" in win.lbl_worker_monitor.text()

    # 2. Toggle Safe Mode (pause write-back)
    assert not win.write_back_paused
    win.toggle_pause_write_back()
    assert win.write_back_paused
    assert win.xmp_daemon.is_paused()
    assert win.ribbon.act_pause_write_back.isChecked()

    win._update_worker_monitor()
    assert "Write-Back Paused" in win.lbl_worker_monitor.text()

    # Toggle back to normal
    win.toggle_pause_write_back()
    assert not win.write_back_paused
    assert not win.xmp_daemon.is_paused()
    assert not win.ribbon.act_pause_write_back.isChecked()

    # 3. Toggle Pause Workers button
    assert not win._workers_paused
    win.toggle_pause_all_workers()
    assert win._workers_paused
    assert win.btn_pause_workers.text() in ("Resume Workers", "▶")
    assert "Resume" in win.btn_pause_workers.toolTip() or win.btn_pause_workers.text() == "▶"
    assert win.xmp_daemon.is_paused()

    win.toggle_pause_all_workers()
    assert not win._workers_paused
    assert win.btn_pause_workers.text() in ("Pause Workers", "⏸")
    assert not win.xmp_daemon.is_paused()

    win.shutdown()
