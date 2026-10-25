"""
Phase 2 Verification Suite: High-Speed Media Scanner, WebP Thumbnailer & Background XMP Daemon.
Run via: pytest tests/test_phase2_workers.py -v
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pytest
from PIL import Image
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from src.db.connection import DualDatabaseEngine
from src.db.repository import (
    CatalogRepository,
    FilterQuerySpec,
    ThumbnailRepository,
)
from src.workers.scanner_thumb_worker import (
    CancellationFlag,
    FolderSyncWorker,
    LruCache,
    ThumbnailDecodeWorker,
    create_filmstrip_for_qimage,
    extract_media_metadata,
)
from src.workers.xmp_sync_worker import (
    MetadataSyncDaemon,
    build_wlpg_xmp_packet,
    read_xmp_from_file,
)


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app  # type: ignore[return-value]


@pytest.fixture()
def db_engine(tmp_path: Path) -> DualDatabaseEngine:
    engine = DualDatabaseEngine(tmp_path / "catalog.db", tmp_path / "thumbnails.db")
    yield engine
    engine.close_all()


def _extract_jpeg_sos_to_eoi(jpeg_bytes: bytes) -> bytes:
    """Returns the compressed scan data starting from SOS (FF DA) to EOF for lossless verification."""
    sos_idx = jpeg_bytes.find(b"\xff\xda")
    assert sos_idx != -1, "JPEG SOS marker not found"
    return jpeg_bytes[sos_idx:]


def test_lru_cache_eviction_and_resizing() -> None:
    cache = LruCache(max_size=3)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    # Touch 'a' so 'b' becomes oldest
    assert cache.get("a") == 1
    cache.put("d", 4)
    assert "b" not in cache
    assert "a" in cache and "c" in cache and "d" in cache

    # Shrink max_size dynamically
    cache.max_size = 2
    assert len(cache) == 2
    assert "c" not in cache


def test_two_stage_folder_scanner_and_incremental_skip(
    qapp, db_engine: DualDatabaseEngine, tmp_path: Path
) -> None:
    photos_dir = tmp_path / "Pictures"
    sub_dir = photos_dir / "1957_Malta"
    backup_dir = photos_dir / ".OriginalFiles"
    sub_dir.mkdir(parents=True)
    backup_dir.mkdir(parents=True)

    # Create sample JPEGs
    img1_path = photos_dir / "landscape.jpg"
    Image.new("RGB", (1200, 800), color=(40, 120, 200)).save(img1_path, "JPEG", quality=90)

    img2_path = sub_dir / "scan410a.jpg"
    Image.new("RGB", (800, 1200), color=(180, 180, 180)).save(img2_path, "JPEG", quality=90)

    # Hidden backup file in .OriginalFiles must be ignored by scanner
    ignored_path = backup_dir / "backup_copy.jpg"
    Image.new("RGB", (400, 400), color=(255, 0, 0)).save(ignored_path, "JPEG")

    stage1_counts: list[int] = []
    finished_results: list[tuple[int, bool]] = []

    worker = FolderSyncWorker(db_engine, photos_dir, include_subfolders=True)
    worker.signals.stage1_completed.connect(lambda c: stage1_counts.append(c))
    worker.signals.finished.connect(lambda cnt, ok: finished_results.append((cnt, ok)))
    worker.run()

    assert stage1_counts == [2]
    assert finished_results == [(2, True)]

    repo = CatalogRepository(db_engine)
    records = repo.query_media_records(FilterQuerySpec(arrange_by="name"))
    assert len(records) == 2
    assert [r.filename for r in records] == ["landscape.jpg", "scan410a.jpg"]
    assert records[0].width == 1200 and records[0].height == 800
    assert records[1].width == 800 and records[1].height == 1200

    # Second scan with no file changes should enrich 0 files (incremental mtime skip)
    finished_second: list[tuple[int, bool]] = []
    worker2 = FolderSyncWorker(db_engine, photos_dir, include_subfolders=True)
    worker2.signals.finished.connect(lambda cnt, ok: finished_second.append((cnt, ok)))
    worker2.run()
    assert finished_second == [(0, True)]


def test_thumbnail_decode_worker_webp_storage_and_filmstrip(
    qapp, db_engine: DualDatabaseEngine, tmp_path: Path
) -> None:
    img_path = tmp_path / "highres.jpg"
    Image.new("RGB", (1600, 1200), color=(20, 160, 90)).save(img_path, "JPEG", quality=92)
    mtime = int(img_path.stat().st_mtime)

    decoded_list: list[tuple[int, QImage]] = []
    worker = ThumbnailDecodeWorker(
        db_engine, media_id=101, file_path=img_path, mtime=mtime, is_video=False
    )
    worker.signals.thumbnail_ready.connect(lambda mid, qimg: decoded_list.append((mid, qimg)))
    worker.run()

    assert len(decoded_list) == 1
    mid, qimg = decoded_list[0]
    assert mid == 101
    assert not qimg.isNull()
    assert qimg.width() == 256
    assert qimg.height() == 192  # 4:3 aspect ratio preserved

    # Verify WebP blob was persisted in thumbnails.db
    thumb_repo = ThumbnailRepository(db_engine)
    blob = thumb_repo.get_webp_blob(101, expected_mtime=mtime)
    assert blob is not None
    assert blob[:4] == b"RIFF" and blob[8:12] == b"WEBP"

    # Verify thread-safe QImage filmstrip compositor for video frames
    raw_frame = QImage(640, 360, QImage.Format_RGB888)
    raw_frame.fill(QColor(80, 100, 160))
    filmstrip_qimg = create_filmstrip_for_qimage(raw_frame, target_size=256)
    assert not filmstrip_qimg.isNull()
    assert filmstrip_qimg.width() == 256


def test_lossless_xmp_sync_daemon_and_roundtrip_rescan(
    qapp, db_engine: DualDatabaseEngine, tmp_path: Path
) -> None:
    photo_dir = tmp_path / "Library"
    photo_dir.mkdir()
    jpg_path = photo_dir / "alfred_1957.jpg"
    Image.new("RGB", (1000, 1400), color=(120, 110, 100)).save(jpg_path, "JPEG", quality=95)

    original_scan_bytes = _extract_jpeg_sos_to_eoi(jpg_path.read_bytes())

    # Scan folder into catalog.db
    FolderSyncWorker(db_engine, photo_dir, include_subfolders=False).run()

    repo = CatalogRepository(db_engine)
    records = repo.query_media_records(FilterQuerySpec())
    assert len(records) == 1
    mid = records[0].id

    # Apply WLPG metadata edits on UI thread (marks metadata_dirty = 1):
    # 1. Rating = 5 stars
    # 2. Caption = "Outside St Lukes by Times of Malta"
    # 3. Relative Date Taken = 14 October 1957 22:09:45 UTC
    # 4. Lossless rotation CW (orientation 1 -> 6)
    # 5. Hierarchical tag = "PeopleTags/Family/Craig"
    # 6. Person face tag = "Alfred John Craig" with rect (0.25, 0.15, 0.10, 0.12)
    oct_1957 = int(
        (
            datetime.datetime(1957, 10, 14, 22, 9, 45, tzinfo=datetime.timezone.utc)
            - datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
        ).total_seconds()
    )
    repo.set_rating([mid], 5)
    repo.set_caption([mid], "Outside St Lukes by Times of Malta")
    repo.apply_relative_time_shift([mid], lead_id=mid, new_lead_epoch=oct_1957)
    repo.rotate_lossless([mid], clockwise=True)
    tag_id = repo.create_tag("PeopleTags/Family/Craig")
    repo.assign_tag([mid], tag_id)
    repo.assign_person(mid, "Alfred John Craig", rect=(0.25, 0.15, 0.10, 0.12))

    assert len(repo.fetch_dirty_media_batch()) == 1

    # Flush background XMP daemon
    daemon = MetadataSyncDaemon(db_engine)
    synced_count = daemon.flush_now()
    assert synced_count == 1
    assert repo.fetch_dirty_media_batch() == []

    # Verify JPEG compressed scan data (FF DA .. FF D9) is 100% untouched (lossless!)
    after_scan_bytes = _extract_jpeg_sos_to_eoi(jpg_path.read_bytes())
    assert after_scan_bytes == original_scan_bytes

    # Verify XMP packet read back from disk
    parsed_xmp = read_xmp_from_file(jpg_path)
    assert parsed_xmp["rating"] == 5
    assert parsed_xmp["orientation"] == 6
    assert parsed_xmp["date_taken"] == oct_1957
    assert parsed_xmp["caption"] == "Outside St Lukes by Times of Malta"
    assert "PeopleTags/Family/Craig" in parsed_xmp["hierarchical_tags"]
    assert len(parsed_xmp["people"]) == 1
    assert parsed_xmp["people"][0]["name"] == "Alfred John Craig"
    assert parsed_xmp["people"][0]["rect"] == pytest.approx((0.25, 0.15, 0.10, 0.12))

    # Verify video XMP packet never emits MP:RegionInfo face regions
    video_detail = {
        "id": 999,
        "is_video": 1,
        "rating": 4,
        "orientation": 1,
        "caption": "Family video",
        "date_taken": 1700000000,
        "tags": [{"name": "Craig", "full_path": "PeopleTags/Family/Craig"}],
        "people": [{"name": "Alfred John Craig", "rect_x": 0.2, "rect_y": 0.2, "rect_w": 0.1, "rect_h": 0.1}],
    }
    video_xmp = build_wlpg_xmp_packet(video_detail)
    assert b"MP:RegionInfo" not in video_xmp


def test_exif_rating_and_xpkeywords_extraction(tmp_path: Path) -> None:
    """Verify that extract_media_metadata correctly extracts EXIF Rating and Windows XPKeywords."""
    from PIL import Image

    test_img_path = tmp_path / "test_exif_meta.jpg"
    img = Image.new("RGB", (100, 100), color=(200, 150, 100))
    exif = img.getexif()
    exif[18246] = 4  # EXIF Rating = 4 stars
    exif[40094] = "Holiday;Sunset;Family".encode("utf-16le")  # Windows XPKeywords
    exif[40091] = "A lovely sunset".encode("utf-16le")        # XPTitle
    img.save(test_img_path, format="JPEG", exif=exif)

    meta = extract_media_metadata(test_img_path, is_video=False)
    assert meta["rating"] == 4
    assert "Holiday" in meta["hierarchical_tags"]
    assert "Sunset" in meta["hierarchical_tags"]
    assert "Family" in meta["hierarchical_tags"]
    assert meta["caption"] == "A lovely sunset"

