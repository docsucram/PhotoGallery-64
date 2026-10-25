import json
import os
import shutil
from pathlib import Path
import pytest
from PySide6.QtCore import Qt, QUrl, QMimeData, QByteArray, QPoint
from PySide6.QtGui import QMouseEvent, QKeyEvent, QPixmap
from PySide6.QtWidgets import QApplication, QMessageBox, QCompleter

from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository, ThumbnailRepository
from src.workers.scanner_thumb_worker import IMAGE_EXT_SET
from src.ui.main_window import PhotoGalleryMainWindow
from src.ui.right_tag_dock import TagAndCaptionDock
from src.ui.map_view import _clusters_to_json


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def test_env(tmp_path):
    engine = DualDatabaseEngine(
        catalog_path=tmp_path / "catalog.db",
        thumbnails_path=tmp_path / "thumbnails.db",
        schema_sql_path=Path("db/schema.sql"),
    )
    engine.initialize_schemas()
    repo = CatalogRepository(engine)

    f1 = tmp_path / "FolderA"
    f2 = tmp_path / "FolderB"
    f1.mkdir()
    f2.mkdir()

    f1_id = repo.upsert_folder(str(f1), "FolderA", None, 1000)
    f2_id = repo.upsert_folder(str(f2), "FolderB", None, 1000)

    # Create dummy images
    img1 = f1 / "photo1.jpg"
    img1.write_bytes(b"dummy_image_1")
    m1_id = repo.bulk_upsert_media([
        {
            "folder_id": f1_id,
            "filename": "photo1.jpg",
            "ext": "jpg",
            "size": 13,
            "mtime": 1000,
            "date_taken": 1000,
            "rating": 0,
            "flag": 0,
        }
    ])[0]

    img2 = f1 / "photo2.jpg"
    img2.write_bytes(b"dummy_image_2")
    m2_id = repo.bulk_upsert_media([
        {
            "folder_id": f1_id,
            "filename": "photo2.jpg",
            "ext": "jpg",
            "size": 13,
            "mtime": 1000,
            "date_taken": 1000,
            "rating": 0,
            "flag": 0,
        }
    ])[0]

    t1_id = repo.create_tag("Nature")
    repo.assign_tag([m1_id], t1_id)

    return {
        "engine": engine,
        "repo": repo,
        "f1": f1,
        "f2": f2,
        "f1_id": f1_id,
        "f2_id": f2_id,
        "m1_id": m1_id,
        "m2_id": m2_id,
        "t1_id": t1_id,
        "tmp_path": tmp_path,
    }


def test_remove_folder_refreshes_immediately(qapp, test_env, monkeypatch):
    """Verifies that removing a folder from the library executes cleanly and refreshes immediately without AttributeError."""
    repo = test_env["repo"]
    win = PhotoGalleryMainWindow(test_env["engine"], start_xmp_daemon=False)
    win.show()

    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)

    # Remove FolderA
    win._on_remove_folder_from_library(test_env["f1_id"], str(test_env["f1"]), "FolderA")

    # Verify FolderA is gone from the database and left tree immediately
    folders = repo.get_folders()
    folder_ids = [f["id"] for f in folders]
    assert test_env["f1_id"] not in folder_ids
    assert test_env["f1_id"] not in win.left_nav_dock._folder_items

    win.close()


def test_tag_typing_recent_and_filtered_popup(qapp, test_env):
    """Verifies that clicking/focusing descriptive tag input shows recent tags in UnfilteredPopupCompletion mode, and typing switches to filtered PopupCompletion over all tags."""
    dock = TagAndCaptionDock()
    dock.show()
    dock.set_autocomplete_lists(["Nature", "Places/Brunei", "Family", "Architecture"], [])
    dock.txt_add_tag.set_recent_tags(["Family", "Nature"])
    dock.bind_selection([test_env["m1_id"]], {"count": 1})

    added_tags = []
    dock.tag_added.connect(added_tags.append)

    # Focus via Ctrl+T helper: when empty, completer shows recent tags in UnfilteredPopupCompletion
    dock.focus_descriptive_tag_input()
    assert dock.txt_add_tag.text() == ""
    assert dock.txt_add_tag._completer.completionMode() == QCompleter.UnfilteredPopupCompletion
    assert dock.txt_add_tag._model.stringList() == ["Family", "Nature"]

    # As user types, switches to filtered PopupCompletion over all tags
    dock.txt_add_tag.setText("Bru")
    dock.txt_add_tag.textEdited.emit("Bru")
    assert dock.txt_add_tag._completer.completionMode() == QCompleter.PopupCompletion
    assert "Places/Brunei" in dock.txt_add_tag._model.stringList()

    # Simulate user selecting from completer
    dock._on_tag_completer_activated("Places/Brunei")
    assert "Places/Brunei" in added_tags
    assert dock.txt_add_tag.text() == ""
    assert dock.txt_add_tag._recent_tags[0] == "Places/Brunei"

    dock.close()


def test_ctrl_t_focuses_descriptive_tag_box(qapp, test_env):
    """Verifies that focus_descriptive_tag_box (bound to Ctrl+T) opens the info pane and focuses the descriptive tag input."""
    win = PhotoGalleryMainWindow(test_env["engine"], start_xmp_daemon=False)
    win.show()
    win._selected_ids = [test_env["m1_id"]]
    win.right_tag_dock.bind_selection([test_env["m1_id"]], {"count": 1})

    win.focus_descriptive_tag_box()
    assert win.right_tag_dock.isVisible()
    assert win.right_tag_dock.mode_stack.currentIndex() == 0

    win.close()


def test_tag_assign_and_remove_sets_dirty(qapp, test_env):
    """Verifies that assigning or removing tags flags media metadata_dirty = 1."""
    repo = test_env["repo"]
    m1_id = test_env["m1_id"]
    m2_id = test_env["m2_id"]

    # Clear dirty flag first
    with repo.engine.write_lock:
        repo.engine.catalog_conn().execute("UPDATE media SET metadata_dirty = 0;")
        repo.engine.catalog_conn().commit()

    assert repo.count_dirty_media() == 0

    # Assign tag to m2
    repo.assign_tag([m2_id], test_env["t1_id"])
    assert repo.count_dirty_media() == 1

    # Remove tag from m1
    repo.remove_tag([m1_id], test_env["t1_id"])
    assert repo.count_dirty_media() == 2


def test_cut_copy_paste_explorer_flow(qapp, test_env):
    """Verifies that Cut, Copy, and Paste work like Explorer to move/copy photos between folders."""
    win = PhotoGalleryMainWindow(test_env["engine"], start_xmp_daemon=False)
    win.show()

    # Select m1 in FolderA
    win._selected_ids = [test_env["m1_id"]]
    win.left_nav_dock._folder_id = test_env["f1_id"]

    # 1. Copy m1
    win.copy_selection_to_clipboard()
    clipboard = QApplication.clipboard()
    mime = clipboard.mimeData()
    if not mime or not mime.hasUrls():
        mime = win._clipboard_mime
    assert mime.hasUrls()
    assert bytes(mime.data("Preferred DropEffect")).startswith(b"\x05")

    # Select FolderB in left nav and paste
    win.left_nav_dock._folder_id = test_env["f2_id"]
    win.paste_from_clipboard()

    # Both folders should have photo1.jpg (copy)
    assert (test_env["f1"] / "photo1.jpg").exists()
    assert (test_env["f2"] / "photo1.jpg").exists()

    # 2. Cut photo2 from FolderA
    win._selected_ids = [test_env["m2_id"]]
    win.left_nav_dock._folder_id = test_env["f1_id"]
    win.cut_selection_to_clipboard()
    mime = clipboard.mimeData()
    if not mime or not mime.hasUrls():
        mime = win._clipboard_mime
    assert bytes(mime.data("Preferred DropEffect")).startswith(b"\x02")

    # Paste into FolderB
    win.left_nav_dock._folder_id = test_env["f2_id"]
    win.paste_from_clipboard()

    # photo2.jpg should be moved from FolderA to FolderB
    assert not (test_env["f1"] / "photo2.jpg").exists()
    assert (test_env["f2"] / "photo2.jpg").exists()

    win.close()


def test_drag_and_drop_move_and_copy_to_folder(qapp, test_env):
    """Verifies that dropping files onto a folder in LeftNavDock moves or copies them on disk and updates catalog.db."""
    repo = test_env["repo"]
    win = PhotoGalleryMainWindow(test_env["engine"], start_xmp_daemon=False)
    win.show()

    src_photo1 = str(test_env["f1"] / "photo1.jpg")
    src_photo2 = str(test_env["f1"] / "photo2.jpg")

    # 1. Copy photo1 to FolderB via drop signal (simulating Ctrl+Left-Drag or Right-Drag -> Copy here)
    win.left_nav_dock.files_dropped_on_folder.emit(
        test_env["f2_id"], "", [src_photo1], True
    )
    assert (test_env["f1"] / "photo1.jpg").exists()
    assert (test_env["f2"] / "photo1.jpg").exists()

    # 2. Move photo2 to FolderB via drop signal (simulating Left-Drag or Right-Drag -> Move here)
    win.left_nav_dock.files_dropped_on_folder.emit(
        test_env["f2_id"], str(test_env["f2"]), [src_photo2], False
    )
    assert not (test_env["f1"] / "photo2.jpg").exists()
    assert (test_env["f2"] / "photo2.jpg").exists()

    # Verify media row m2_id now points to f2_id in catalog.db
    row = repo.engine.catalog_conn().execute(
        "SELECT folder_id FROM media WHERE id = ?", (test_env["m2_id"],)
    ).fetchone()
    assert row["folder_id"] == test_env["f2_id"]

    win.close()


def test_map_view_clusters_to_json_strips_qpixmap(qapp):
    """Verifies that _clusters_to_json strips non-serializable QPixmap objects before json.dumps."""
    pix = QPixmap(32, 32)
    clusters = [
        {
            "lat": 51.5074,
            "lon": -0.1278,
            "count": 3,
            "rep_media_id": 42,
            "media_ids": [42, 43, 44],
            "thumb_b64": "dGVzdA==",
            "pixmap": pix,
        }
    ]
    serialized = _clusters_to_json(clusters)
    parsed = json.loads(serialized)
    assert len(parsed) == 1
    assert parsed[0]["rep_media_id"] == 42
    assert "pixmap" not in parsed[0]

