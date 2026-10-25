import sys
from pathlib import Path
import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication

from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository, ThumbnailRepository
from src.workers.scanner_thumb_worker import IGNORED_DIR_NAMES
from src.utils.scan_logger import log_scan_error
from src.workers.xmp_sync_worker import MetadataSyncDaemon, write_metadata_lossless


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


@pytest.fixture
def temp_dbs(tmp_path):
    cat_path = tmp_path / "catalog.db"
    thumb_path = tmp_path / "thumbnails.db"
    engine = DualDatabaseEngine(cat_path, thumb_path)
    yield engine
    engine.close_all()


def test_ignored_dir_names():
    assert ".originalfiles" in IGNORED_DIR_NAMES
    assert "originalfiles" in IGNORED_DIR_NAMES
    assert ".original_files" in IGNORED_DIR_NAMES


def test_scan_errors_log(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log_scan_error("TestWorker", "broken_image.jpg", "Corrupt JPEG SOS marker")
    log_file = tmp_path / "logs" / "scan_errors.log"
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "[TestWorker]" in content
    assert "broken_image.jpg" in content
    assert "Corrupt JPEG SOS marker" in content


def test_multi_resolution_ico():
    ico_path = Path("resources/icons/photogallery_live.ico")
    assert ico_path.exists(), "ICO icon file must exist"
    with Image.open(ico_path) as img:
        sizes = img.info.get("sizes")
        assert (16, 16) in sizes
        assert (24, 24) in sizes
        assert (32, 32) in sizes
        assert (48, 48) in sizes
        assert (64, 64) in sizes
        assert (128, 128) in sizes
        assert (256, 256) in sizes


def test_metadata_sync_daemon_stop_requested(qapp, temp_dbs):
    daemon = MetadataSyncDaemon(temp_dbs)
    daemon.start()
    assert daemon._running is True
    assert daemon.stop_requested.is_set() is False
    daemon.stop()
    assert daemon._running is False
    assert daemon.stop_requested.is_set() is True


def test_lossless_jpeg_app1_splice(tmp_path):
    # Create valid synthetic minimal JPEG
    # SOI (FF D8) + APP0 (FF E0 00 10 ...) + SOS (FF DA ...) + EOI (FF D9)
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    jpg_path = tmp_path / "test_lossless.jpg"
    img.save(jpg_path, format="JPEG")

    original_bytes = jpg_path.read_bytes()
    # Find SOS marker (0xFFDA)
    sos_idx = original_bytes.find(b"\xff\xda")
    assert sos_idx != -1
    scan_data_orig = original_bytes[sos_idx:]

    # Prepare item payload
    item = {
        "id": 1,
        "folder_path": str(tmp_path),
        "filename": "test_lossless.jpg",
        "rating": 5,
        "caption": "Pristine Test",
        "orientation": 1,
        "date_taken": 1700000000,
        "tags": ["Landscape", "Live"],
        "people": [],
    }
    mtime = write_metadata_lossless(item)
    assert mtime is not None

    patched_bytes = jpg_path.read_bytes()
    # Verify APP1 XMP header was injected
    assert b"http://ns.adobe.com/xap/1.0/" in patched_bytes
    assert b"Pristine Test" in patched_bytes

    # Verify compressed DCT scan data after SOS is 100% byte-identical (NO re-encoding!)
    patched_sos_idx = patched_bytes.find(b"\xff\xda")
    assert patched_sos_idx != -1
    scan_data_patched = patched_bytes[patched_sos_idx:]
    assert scan_data_orig == scan_data_patched, "JPEG scan data must be 100% byte-identical"


def test_app_title_photo_gallery_live(qapp, temp_dbs):
    from src.ui.main_window import PhotoGalleryMainWindow
    win = PhotoGalleryMainWindow(temp_dbs, start_xmp_daemon=False)
    assert win.windowTitle() == "Photo Gallery Live"
    win.close()


def test_delete_and_move_with_original_files_and_xmp(tmp_path, temp_dbs, qapp):
    from src.ui.main_window import PhotoGalleryMainWindow, _MoveFilesUndoCommand

    # Set up source folder with image, .OriginalFiles backup, and .xmp sidecar
    src_dir = tmp_path / "Source"
    src_dir.mkdir(exist_ok=True)
    dst_dir = tmp_path / "Destination"
    dst_dir.mkdir(exist_ok=True)

    img_file = src_dir / "pic.jpg"
    img_file.write_bytes(b"\xff\xd8\xff\xd9")

    orig_backup_dir = src_dir / ".OriginalFiles"
    orig_backup_dir.mkdir(exist_ok=True)
    backup_file = orig_backup_dir / "pic.jpg"
    backup_file.write_bytes(b"\xff\xd8\xff\xd9")

    xmp_file = src_dir / "pic.jpg.xmp"
    xmp_file.write_text("<xmp>test</xmp>", encoding="utf-8")

    # Insert into DB
    conn = temp_dbs.catalog_conn()
    with temp_dbs.write_lock:
        conn.execute("INSERT INTO folders (id, name, path) VALUES (1, 'Source', ?);", (str(src_dir),))
        conn.execute("INSERT INTO folders (id, name, path) VALUES (2, 'Destination', ?);", (str(dst_dir),))
        conn.execute(
            "INSERT INTO media (id, folder_id, filename, ext, size, mtime, is_video, original_backup_path) "
            "VALUES (10, 1, 'pic.jpg', 'jpg', 100, 1700000000, 0, ?);",
            (str(backup_file),),
        )
        conn.commit()

    win = PhotoGalleryMainWindow(temp_dbs, start_xmp_daemon=False)

    # Test Move Command
    moves = [{
        "media_id": 10,
        "src_path": str(img_file),
        "dst_path": str(dst_dir / "pic.jpg"),
        "src_folder_id": 1,
        "dst_folder_id": 2,
        "src_filename": "pic.jpg",
        "dst_filename": "pic.jpg",
    }]
    cmd = _MoveFilesUndoCommand(win, moves, "Destination")
    cmd.redo()

    # Verify moved to destination including .OriginalFiles and .xmp
    assert (dst_dir / "pic.jpg").exists()
    assert (dst_dir / ".OriginalFiles" / "pic.jpg").exists()
    assert (dst_dir / "pic.jpg.xmp").exists()
    assert not img_file.exists()
    assert not backup_file.exists()
    assert not xmp_file.exists()

    # Test Delete
    win._selected_ids = [10]
    win.delete_selection_to_recycle_bin()

    assert not (dst_dir / "pic.jpg").exists()
    assert not (dst_dir / ".OriginalFiles" / "pic.jpg").exists()
    assert not (dst_dir / "pic.jpg.xmp").exists()

    win.close()
