import sys
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository
from src.utils.scan_logger import install_global_crash_logger


def test_crash_logger(tmp_path):
    log_dir = tmp_path / "test_logs"
    install_global_crash_logger(log_dir)

    try:
        raise RuntimeError("Synthetic test error for crash log")
    except RuntimeError:
        exc_type, exc_val, exc_tb = sys.exc_info()
        sys.excepthook(exc_type, exc_val, exc_tb)

    crash_log = log_dir / "crash.log"
    assert crash_log.exists()
    content = crash_log.read_text(encoding="utf-8")
    assert "UNCAUGHT EXCEPTION" in content
    assert "RuntimeError: Synthetic test error for crash log" in content


def test_person_header_bar_resets_view_mode(tmp_path):
    app = QApplication.instance() or QApplication([])

    cat_path = tmp_path / "catalog.db"
    thumb_path = tmp_path / "thumbnails.db"
    engine = DualDatabaseEngine(cat_path, thumb_path)
    engine.initialize_schemas()
    repo = CatalogRepository(engine)
    pid = repo.upsert_person("Test Person")

    from src.ui.person_header_bar import PersonHeaderBar
    bar = PersonHeaderBar(engine)

    # Set mode to crops
    bar._set_view_mode("crops")
    bar.btn_face_crops.setChecked(True)
    assert bar.current_view_mode == "crops"
    assert bar.btn_face_crops.isChecked() is True

    # Loading person must reset view mode to "photos" and check btn_full_photos
    bar.load_person(pid)
    assert bar.current_view_mode == "photos"
    assert bar.btn_full_photos.isChecked() is True
    assert bar.btn_face_crops.isChecked() is False


def test_merge_people_robustness(tmp_path):
    cat_path = tmp_path / "catalog.db"
    thumb_path = tmp_path / "thumbnails.db"
    engine = DualDatabaseEngine(cat_path, thumb_path)
    engine.initialize_schemas()
    repo = CatalogRepository(engine)

    p1 = repo.upsert_person("Source Person")
    p2 = repo.upsert_person("Target Person")

    fid = repo.upsert_folder("D:/Photos", "Photos", None, 0)
    mids = repo.bulk_upsert_media([{
        "folder_id": fid,
        "filename": "img1.jpg",
        "ext": "jpg",
        "size": 100,
        "mtime": 1000,
    }])
    mid = mids[0]

    # Assign both persons to the same photo
    f1 = repo.assign_person(mid, "Source Person", rect=(0.1, 0.1, 0.2, 0.2))
    f2 = repo.assign_person(mid, "Target Person", rect=(0.5, 0.5, 0.2, 0.2))

    # Add rejections for both
    conn = engine.catalog_conn()
    conn.execute("INSERT INTO face_rejections (face_id, person_id) VALUES (?, ?);", (f1, p1))
    conn.execute("INSERT INTO face_rejections (face_id, person_id) VALUES (?, ?);", (f2, p2))
    conn.commit()

    # Merge p1 into p2
    repo.merge_people(p1, p2)

    # p1 should be deleted from people table
    all_p = repo.get_all_people()
    p_ids = [p["id"] for p in all_p]
    assert p1 not in p_ids
    assert p2 in p_ids

    # Duplicate assignments for p2 on mid should be deduplicated
    faces = repo.get_people_for_media(mid)
    p2_faces = [f for f in faces if f.get("person_id") == p2]
    assert len(p2_faces) == 1


def test_open_file_on_startup_and_get_media_by_path(tmp_path):
    app = QApplication.instance() or QApplication([])

    cat_path = tmp_path / "catalog.db"
    thumb_path = tmp_path / "thumbnails.db"
    engine = DualDatabaseEngine(cat_path, thumb_path)
    engine.initialize_schemas()
    repo = CatalogRepository(engine)

    img_dir = tmp_path / "Photos"
    img_dir.mkdir()
    img_file = img_dir / "sample.jpg"

    # Create dummy image bytes
    from PIL import Image
    im = Image.new("RGB", (100, 100), color=(255, 0, 0))
    im.save(str(img_file), "JPEG")

    # Initially not in catalog
    assert repo.get_media_id_by_file_path(img_file) is None

    # Fast ingest via main window method
    from src.ui.main_window import PhotoGalleryMainWindow
    win = PhotoGalleryMainWindow(engine=engine, start_xmp_daemon=False)
    try:
        ok = win.open_file_on_startup(img_file)
        assert ok is True
        # Verify media was inserted and viewer opened
        mid = repo.get_media_id_by_file_path(img_file)
        assert mid is not None
        assert win.media_viewer.current_media_id == mid
    finally:
        win.close()


def test_left_nav_dock_people_collapsed_by_default(tmp_path):
    app = QApplication.instance() or QApplication([])
    from src.ui.left_nav_dock import LeftNavigationDock

    cat_path = tmp_path / "catalog.db"
    thumb_path = tmp_path / "thumbnails.db"
    engine = DualDatabaseEngine(cat_path, thumb_path)
    engine.initialize_schemas()
    repo = CatalogRepository(engine)
    repo.upsert_person("Alice")

    dock = LeftNavigationDock()
    # Initially collapsed by default
    assert dock.root_people.isExpanded() is False

    # After refresh_trees, stays collapsed
    dock.refresh_trees(repo)
    assert dock.root_people.isExpanded() is False

    # When user expands, stays expanded on subsequent refresh
    dock.root_people.setExpanded(True)
    dock.refresh_trees(repo)
    assert dock.root_people.isExpanded() is True


def test_merge_people_dialog_typing_autocomplete_and_preview(tmp_path):
    app = QApplication.instance() or QApplication([])
    from src.ui.people_directory_view import MergePeopleDialog

    candidates = [
        {"id": 10, "name": "Bob Jones", "photo_count": 8, "thumbnail_blob": None},
        {"id": 20, "name": "Charlie Brown", "photo_count": 3, "thumbnail_blob": None},
    ]

    dlg = MergePeopleDialog(
        source_person_name="Alice Smith",
        candidates=candidates,
        source_person_id=1,
        source_photo_count=12,
    )

    # Initial state: button disabled, no target selected
    assert dlg.btn_action.isEnabled() is False
    assert dlg.target_person_id() is None

    # Clicking line edit opens suggestions popup
    dlg.txt_target.show_completion_popup()
    assert dlg.txt_target.completer() is not None

    # User types "Bob"
    dlg.txt_target.setText("Bob Jones")
    assert dlg.btn_action.isEnabled() is True
    assert dlg.btn_action.text() == "Merge"
    assert dlg.target_person_id() == 10
    assert dlg.is_rename() is False

    # User types brand new name "David Clark"
    dlg.txt_target.setText("David Clark")
    assert dlg.btn_action.isEnabled() is True
    assert dlg.btn_action.text() == "Rename"
    assert dlg.is_rename() is True
    assert dlg.target_person_name() == "David Clark"

    # User types source person name (cannot merge into self)
    dlg.txt_target.setText("Alice Smith")
    assert dlg.btn_action.isEnabled() is False


def test_person_header_bar_name_autocomplete(tmp_path):
    app = QApplication.instance() or QApplication([])
    from src.ui.person_header_bar import PersonHeaderBar

    cat_path = tmp_path / "catalog.db"
    thumb_path = tmp_path / "thumbnails.db"
    engine = DualDatabaseEngine(cat_path, thumb_path)
    engine.initialize_schemas()
    repo = CatalogRepository(engine)
    p1 = repo.upsert_person("Alice")
    p2 = repo.upsert_person("Bob")
    p3 = repo.upsert_person("Charlie")

    bar = PersonHeaderBar(engine)
    bar.load_person(p1)

    # txt_name has completer populated with other people
    comp = bar.txt_name.completer()
    assert comp is not None
    # Bob and Charlie in model, Alice excluded
    model = comp.model()
    names = [model.index(i, 0).data() for i in range(model.rowCount())]
    assert "Bob" in names
    assert "Charlie" in names
    assert "Alice" not in names
