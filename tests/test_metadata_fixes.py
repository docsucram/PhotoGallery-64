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

    # Hierarchical tag + flat redundant leaf tag (e.g. Christmas from dc:subject) + WLPG PeopleTag
    h_tags = [
        "Subject/Occasions/Christmas",
        "Christmas",
        "PeopleTags/Family/Robinson/Nansumay",
    ]
    people_list = []

    _assign_tags_and_people_for_media(repo, mid, h_tags, people_list)

    # Check tags on mid
    summary = repo.get_selection_summary([mid])
    tags = summary.get("tags", [])
    # Must have only 1 tag ('Christmas' with path 'Subject/Occasions/Christmas'), not duplicate
    assert len(tags) == 1
    assert tags[0]["name"] == "Christmas"
    assert tags[0]["full_path"] == "Subject/Occasions/Christmas"

    # Must have person 'Nansumay Robinson'
    people = summary.get("people", [])
    assert len(people) == 1
    assert people[0]["name"] == "Nansumay Robinson"


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
