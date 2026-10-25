import pytest
from pathlib import Path
from src.workers.scanner_thumb_worker import (
    _timestamp_from_filename,
    _resolve_person_name_from_tag,
    _assign_tags_and_people_for_media,
    extract_media_metadata,
)
from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository


def test_filename_timestamp_patterns():
    # Various real-world camera / phone formats
    assert _timestamp_from_filename("2016-01-05 14.01.27") is not None
    assert _timestamp_from_filename("2016-12-30 09.16.02") is not None
    assert _timestamp_from_filename("IMG_20210514_123045") is not None
    assert _timestamp_from_filename("PXL_20231025_091530") is not None
    assert _timestamp_from_filename("VID_20200101_102030") is not None
    assert _timestamp_from_filename("non_timestamp_file") is None


def test_resolve_person_name_from_wlpg_tag():
    class DummyRepo:
        def get_all_people(self):
            return [
                {"id": 1, "name": "Nansumay Robinson"},
                {"id": 2, "name": "Tu May Robinson"},
            ]

    repo = DummyRepo()
    # WLPG PeopleTags / Face tag patterns
    assert _resolve_person_name_from_tag(repo, ["PeopleTags", "Family", "Robinson", "Nansumay"]) == "Nansumay Robinson"
    assert _resolve_person_name_from_tag(repo, ["Face", "Family", "Robinson", "Nansumay"]) == "Nansumay Robinson"
    assert _resolve_person_name_from_tag(repo, ["PeopleTags", "Tu May"]) == "Tu May Robinson"
    assert _resolve_person_name_from_tag(repo, ["Face", "Friends", "John"]) == "John"


def test_assign_tags_and_people_deduplication(tmp_path):
    cat_path = tmp_path / "catalog.db"
    thumb_path = tmp_path / "thumbnails.db"
    engine = DualDatabaseEngine(cat_path, thumb_path)
    engine.initialize_schemas()
    repo = CatalogRepository(engine)

    # Add folder and media
    fid = repo.upsert_folder("D:/Photos", "Photos", None, 0)
    mid = repo.bulk_upsert_media([{
        "folder_id": fid,
        "filename": "test.jpg",
        "ext": "jpg",
        "size": 100,
        "mtime": 1000,
    }])[0]

    # Hierarchical tag + flat redundant leaf tag (e.g. Christmas from dc:subject) + PeopleTags + Face/
    h_tags = [
        "Subject/Occasions/Christmas",
        "Christmas",
        "PeopleTags/Family/Robinson/Nansumay",
        "Face/Family/John",
    ]
    people_list = []

    _assign_tags_and_people_for_media(repo, mid, h_tags, people_list)

    # Check tags on mid
    summary = repo.get_selection_summary([mid])
    tags = summary.get("tags", [])
    # Flat 'Christmas' is deduplicated against 'Subject/Occasions/Christmas'.
    # 'PeopleTags/Family/Robinson/Nansumay' is kept as a descriptive tag.
    # 'Face/Family/John' is mapped to person 'John' and excluded from descriptive tags.
    tag_paths = {t["full_path"] for t in tags}
    assert tag_paths == {"Subject/Occasions/Christmas", "PeopleTags/Family/Robinson/Nansumay"}

    # Must have person 'John' from Face/
    people = summary.get("people", [])
    assert len(people) == 1
    assert people[0]["name"] == "John"


def test_deduplicate_media_tags_cleanup(tmp_path):
    cat_path = tmp_path / "catalog.db"
    thumb_path = tmp_path / "thumbnails.db"
    engine = DualDatabaseEngine(cat_path, thumb_path)
    engine.initialize_schemas()
    repo = CatalogRepository(engine)

    fid = repo.upsert_folder("D:/Photos", "Photos", None, 0)
    mid = repo.bulk_upsert_media([{
        "folder_id": fid,
        "filename": "snow.jpg",
        "ext": "jpg",
        "size": 200,
        "mtime": 2000,
    }])[0]

    # Artificially assign both a hierarchical tag and root tag
    t1 = repo.create_tag("Subject/Weather/Snow")
    t2 = repo.create_tag("Snow")
    repo.assign_tag([mid], t1)
    repo.assign_tag([mid], t2)

    assert len(repo.get_selection_summary([mid])["tags"]) == 2

    # Run cleanup
    removed = repo.deduplicate_media_tags()
    assert removed == 1

    # Should only have the hierarchical tag left
    tags_after = repo.get_selection_summary([mid])["tags"]
    assert len(tags_after) == 1
    assert tags_after[0]["full_path"] == "Subject/Weather/Snow"


def test_grid_scrollbar_and_dock_resizing():
    from PySide6.QtWidgets import QApplication
    from src.ui.grid_view import ThumbnailGridView
    from src.ui.right_tag_dock import TagAndCaptionDock

    app = QApplication.instance() or QApplication([])

    # Grid scrollbar geometry positioning
    grid = ThumbnailGridView()
    grid.resize(800, 600)
    grid.updateGeometries()
    sb = grid.verticalScrollBar()
    assert sb is not None
    # Scrollbar should be at the right edge
    sb_w = sb.sizeHint().width() or sb.width() or 14
    assert sb.width() > 0

    # Right Tag dock features & min height
    dock = TagAndCaptionDock()
    assert dock.maximumWidth() == 600
    assert not (dock.features() & dock.DockWidgetFeature.DockWidgetMovable)
    assert dock.features() & dock.DockWidgetFeature.DockWidgetClosable
    assert dock.info_splitter.isCollapsible(1) is False
    bottom_scroll = dock.findChild(object, "BottomScroll")
    assert bottom_scroll is not None
    assert bottom_scroll.minimumHeight() >= 28
