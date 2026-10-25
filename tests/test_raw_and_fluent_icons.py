import pytest
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon

from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository, FilterQuerySpec, RAW_EXTENSIONS
from src.ui.main_window import PhotoGalleryMainWindow
import src.ui.icon_provider as icon_provider


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_raw_media_type_filter(tmp_path: Path) -> None:
    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    engine = DualDatabaseEngine(
        catalog_path=db_dir / "catalog.db",
        thumbnails_path=db_dir / "thumbnails.db",
    )
    repo = CatalogRepository(engine)

    f_id = repo.upsert_folder(str(tmp_path), "Photos", None, 1000)

    # Insert standard JPEG, video, and RAW files
    repo.bulk_upsert_media([
        {"folder_id": f_id, "filename": "sample.jpg", "size": 100, "mtime": 100, "is_video": 0},
        {"folder_id": f_id, "filename": "clip.mp4", "size": 200, "mtime": 100, "is_video": 1},
        {"folder_id": f_id, "filename": "raw_photo.cr2", "size": 300, "mtime": 100, "is_video": 0},
        {"folder_id": f_id, "filename": "sony.arw", "size": 400, "mtime": 100, "is_video": 0},
        {"folder_id": f_id, "filename": "nikon.nef", "size": 500, "mtime": 100, "is_video": 0},
    ])

    # All media
    all_recs = repo.query_media_records(FilterQuerySpec(media_type="all"))
    assert len(all_recs) == 5

    # Video only
    video_recs = repo.query_media_records(FilterQuerySpec(media_type="video"))
    assert len(video_recs) == 1
    assert video_recs[0].filename == "clip.mp4"

    # RAW only
    raw_recs = repo.query_media_records(FilterQuerySpec(media_type="raw"))
    assert len(raw_recs) == 3
    raw_names = {r.filename for r in raw_recs}
    assert raw_names == {"raw_photo.cr2", "sony.arw", "nikon.nef"}


def test_fluent_and_aero_icon_provider(qapp: QApplication) -> None:
    # 1. Test Classic Aero Theme
    icon_provider.set_icon_theme("wlpg_aero")
    assert icon_provider.get_icon_theme() == "wlpg_aero"
    ico_aero = icon_provider.get_icon("folder", 22)
    assert isinstance(ico_aero, QIcon)
    assert not ico_aero.isNull()

    # 2. Test Microsoft Fluent UI Theme
    icon_provider.set_icon_theme("fluent")
    assert icon_provider.get_icon_theme() == "fluent"
    ico_fluent = icon_provider.get_icon("folder", 22)
    assert isinstance(ico_fluent, QIcon)
    assert not ico_fluent.isNull()

    # Specific fluent keys
    assert not icon_provider.get_icon("delete", 22).isNull()
    assert not icon_provider.get_icon("star", 22).isNull()
    assert not icon_provider.get_icon("crop", 22).isNull()

    # 3. Test Minimal Theme
    icon_provider.set_icon_theme("minimal")
    assert icon_provider.get_icon_theme() == "minimal"
    ico_min = icon_provider.get_icon("folder", 22)
    assert isinstance(ico_min, QIcon)
    assert not ico_min.isNull()

    # Reset back to wlpg_aero default
    icon_provider.set_icon_theme("wlpg_aero")


def test_theme_and_icon_settings_persistence(qapp: QApplication, tmp_path: Path) -> None:
    from PySide6.QtCore import QSettings

    settings = QSettings("PhotoGallery64", "PhotoGallery64")
    settings.setValue("theme", "modern")
    settings.setValue("icon_theme", "fluent")

    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    engine = DualDatabaseEngine(
        catalog_path=db_dir / "catalog.db",
        thumbnails_path=db_dir / "thumbnails.db",
    )

    window = PhotoGalleryMainWindow(engine=engine, start_xmp_daemon=False)
    window._restore_app_state(force=True)
    # Verify restored settings
    assert window.current_theme == "modern"
    assert window.current_icon_theme == "fluent"
    assert icon_provider.get_icon_theme() == "fluent"

    # Reset back to default
    settings.setValue("theme", "wlpg_aero")
    settings.setValue("icon_theme", "wlpg_aero")
    icon_provider.set_icon_theme("wlpg_aero")
    window.shutdown()


def test_deduplicate_media_people_and_exclude_same_photo_suggestions(tmp_path: Path) -> None:
    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    engine = DualDatabaseEngine(
        catalog_path=db_dir / "catalog.db",
        thumbnails_path=db_dir / "thumbnails.db",
    )
    repo = CatalogRepository(engine)

    f_id = repo.upsert_folder(str(tmp_path), "Photos", None, 1000)
    repo.bulk_upsert_media([
        {"folder_id": f_id, "filename": "portrait.jpg", "size": 100, "mtime": 100, "is_video": 0, "orientation": 6}
    ])
    mid = repo.query_media_records(FilterQuerySpec())[0].id
    pid = repo.upsert_person("Test Child")

    with engine.write_lock:
        conn = engine.catalog_conn()
        # Insert 1 true face with embedding, and 2 rotated duplicate phantoms without embedding
        conn.execute(
            """
            INSERT INTO media_people (media_id, person_id, state, is_manual, rect_x, rect_y, rect_w, rect_h, embedding)
            VALUES (?, ?, 1, 1, 0.50, 0.28, 0.30, 0.24, ?);
            """,
            (mid, pid, b"fake_embedding" * 32),
        )
        conn.execute(
            """
            INSERT INTO media_people (media_id, person_id, state, is_manual, rect_x, rect_y, rect_w, rect_h, embedding)
            VALUES (?, ?, 1, 1, 0.48, 0.50, 0.24, 0.30, NULL);
            """,
            (mid, pid),
        )
        conn.execute(
            """
            INSERT INTO media_people (media_id, person_id, state, is_manual, rect_x, rect_y, rect_w, rect_h, embedding)
            VALUES (?, ?, 1, 1, 0.02, 0.29, 0.15, 0.11, NULL);
            """,
            (mid, pid),
        )
        # Also insert an unassigned detection on the same image (state = 0)
        cur = conn.execute(
            """
            INSERT INTO media_people (media_id, person_id, state, is_manual, rect_x, rect_y, rect_w, rect_h, embedding)
            VALUES (?, NULL, 0, 0, 0.02, 0.29, 0.15, 0.11, ?)
            RETURNING id;
            """,
            (mid, b"fake_embedding" * 32),
        )
        unassigned_id = int(cur.fetchone()[0])
        conn.commit()

    # 1. Deduplication should remove the 2 phantoms and retain the true face
    del_cnt = repo.deduplicate_media_people(mid)
    assert del_cnt == 2

    # Check remaining confirmed rows
    detail = repo.get_media_detail(mid)
    assert len(detail["confirmed_people"]) == 1
    assert detail["confirmed_people"][0]["rect"] == (0.50, 0.28, 0.30, 0.24)

    # 2. Verify get_rejected_face_pairs excludes suggesting Test Child for unassigned_id
    rejected = repo.get_rejected_face_pairs()
    assert (unassigned_id, pid) in rejected
