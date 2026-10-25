"""
Phase 1 Verification Suite: Dual-Database Engine & Sub-10ms Repository Layer.
Run via: pytest tests/test_phase1_db.py -v
"""

from __future__ import annotations

import datetime
import time
from pathlib import Path

import pytest

from src.db.connection import DualDatabaseEngine
from src.db.repository import (
    CatalogRepository,
    FilterQuerySpec,
    ThumbnailRepository,
)


@pytest.fixture()
def db_engine(tmp_path: Path) -> DualDatabaseEngine:
    cat_db = tmp_path / "catalog.db"
    thumb_db = tmp_path / "thumbnails.db"
    engine = DualDatabaseEngine(cat_db, thumb_db)
    yield engine
    engine.close_all()


def test_wal_and_schema_initialization(db_engine: DualDatabaseEngine) -> None:
    cat_conn = db_engine.catalog_conn()
    thumb_conn = db_engine.thumbnails_conn()

    cat_mode = cat_conn.execute("PRAGMA journal_mode;").fetchone()[0]
    thumb_mode = thumb_conn.execute("PRAGMA journal_mode;").fetchone()[0]
    assert cat_mode.lower() == "wal"
    assert thumb_mode.lower() == "wal"

    cat_tables = {
        r[0]
        for r in cat_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ).fetchall()
    }
    assert {
        "folders",
        "media",
        "tags",
        "tag_closure",
        "media_tags",
        "people",
        "media_people",
        "media_rtree",
    }.issubset(cat_tables)

    thumb_tables = {
        r[0]
        for r in thumb_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ).fetchall()
    }
    assert "thumbnails" in thumb_tables


def test_folder_hierarchy_and_media_upsert(db_engine: DualDatabaseEngine) -> None:
    repo = CatalogRepository(db_engine)
    root_id = repo.upsert_folder("D:/Pictures", "Pictures", None, 1700000000)
    sub_id = repo.upsert_folder("D:/Pictures/1957", "1957", root_id, 1700000100)
    repo.set_folder_favourite(sub_id, True)

    folders = repo.get_folders()
    assert len(folders) == 2
    assert any(f["id"] == sub_id and f["is_favourite"] == 1 for f in folders)

    oct_1957 = int(
        datetime.datetime(1957, 10, 14, 22, 9, 45, tzinfo=datetime.timezone.utc).timestamp()
    )
    jan_2026 = int(
        datetime.datetime(2026, 1, 2, 12, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
    )

    ids = repo.bulk_upsert_media(
        [
            {
                "folder_id": root_id,
                "filename": "root_photo.jpg",
                "ext": "jpg",
                "size": 102400,
                "mtime": 1700000000,
                "date_taken": jan_2026,
                "rating": 4,
                "width": 1920,
                "height": 1080,
            },
            {
                "folder_id": sub_id,
                "filename": "scan410a.jpg",
                "ext": "jpg",
                "size": 371712,
                "mtime": 1700000100,
                "date_taken": oct_1957,
                "rating": 5,
                "width": 1760,
                "height": 2413,
                "caption": "Outside St Lukes by Times of Malta",
            },
        ]
    )
    assert len(ids) == 2

    # Query root with subfolders -> 2 items
    all_in_root = repo.query_media_records(
        FilterQuerySpec(folder_id=root_id, include_subfolders=True)
    )
    assert len(all_in_root) == 2

    # Query root without subfolders -> 1 item
    only_root = repo.query_media_records(
        FilterQuerySpec(folder_id=root_id, include_subfolders=False)
    )
    assert len(only_root) == 1
    assert only_root[0].filename == "root_photo.jpg"

    # Date filtering (1957 -> October -> 14)
    oct_records = repo.query_media_records(
        FilterQuerySpec(date_year=1957, date_month=10, date_day=14)
    )
    assert len(oct_records) == 1
    assert oct_records[0].filename == "scan410a.jpg"
    assert oct_records[0].group_key == "October 1957"

    rating_counts = repo.get_rating_counts()
    assert rating_counts[5] == 1
    assert rating_counts[4] == 1

    date_tree = repo.get_date_tree_counts()
    assert len(date_tree) == 2
    assert date_tree[0]["day_label"] == "2 Jan, Friday"
    assert date_tree[1]["day_label"] == "14 Oct, Monday"


def test_tag_closure_boolean_filtering_and_reparenting(
    db_engine: DualDatabaseEngine,
) -> None:
    repo = CatalogRepository(db_engine)
    fid = repo.upsert_folder("D:/Photos", "Photos", None, 1700000000)
    m_ids = repo.bulk_upsert_media(
        [
            {"folder_id": fid, "filename": "img1.jpg", "ext": "jpg", "size": 100, "mtime": 10},
            {"folder_id": fid, "filename": "img2.jpg", "ext": "jpg", "size": 100, "mtime": 20},
            {"folder_id": fid, "filename": "img3.jpg", "ext": "jpg", "size": 100, "mtime": 30},
            {"folder_id": fid, "filename": "untagged.jpg", "ext": "jpg", "size": 100, "mtime": 40},
        ]
    )

    t_craig = repo.create_tag("PeopleTags/Family/Craig")
    t_marcus = repo.create_tag("PeopleTags/Family/Robinson/Marcus")
    t_malta = repo.create_tag("Place/Europe/Malta")

    # Fetch ancestor IDs
    tree = {t["full_path"]: t["id"] for t in repo.get_tag_tree()}
    t_people_root = tree["PeopleTags"]
    t_family = tree["PeopleTags/Family"]
    t_place = tree["Place"]

    # Assign tags:
    # img1: Craig + Malta
    # img2: Marcus + Malta
    # img3: Craig only
    repo.assign_tag([m_ids[0], m_ids[2]], t_craig)
    repo.assign_tag([m_ids[1]], t_marcus)
    repo.assign_tag([m_ids[0], m_ids[1]], t_malta)

    # 1. Ancestor query: selecting 'PeopleTags/Family' returns img1, img2, img3
    res_family = repo.query_media_records(
        FilterQuerySpec(include_tag_ids=(t_family,))
    )
    assert {r.id for r in res_family} == {m_ids[0], m_ids[1], m_ids[2]}

    # 2. Boolean AND: 'PeopleTags/Family' AND 'Place' -> img1, img2
    res_and = repo.query_media_records(
        FilterQuerySpec(
            include_tag_ids=(t_family, t_place),
            tag_boolean_mode="AND",
        )
    )
    assert {r.id for r in res_and} == {m_ids[0], m_ids[1]}

    # 3. Boolean AND + Exclude (NOT): 'PeopleTags' AND NOT 'Place' -> img3
    res_not = repo.query_media_records(
        FilterQuerySpec(
            include_tag_ids=(t_people_root,),
            exclude_tag_ids=(t_place,),
        )
    )
    assert [r.id for r in res_not] == [m_ids[2]]

    # 4. Untagged only -> untagged.jpg
    res_untagged = repo.query_media_records(FilterQuerySpec(untagged_only=True))
    assert [r.id for r in res_untagged] == [m_ids[3]]

    # 5. Test tag reparenting trigger (move 'Robinson' under root)
    t_robinson = tree["PeopleTags/Family/Robinson"]
    repo.move_tag(t_robinson, None)
    updated_tree = {t["full_path"]: t["id"] for t in repo.get_tag_tree()}
    assert "Robinson/Marcus" in updated_tree


def test_people_static_image_only_face_regions_and_dirty_triggers(
    db_engine: DualDatabaseEngine,
) -> None:
    repo = CatalogRepository(db_engine)
    fid = repo.upsert_folder("D:/Media", "Media", None, 100)
    img_id, vid_id = repo.bulk_upsert_media(
        [
            {"folder_id": fid, "filename": "photo.jpg", "ext": "jpg", "is_video": 0, "mtime": 100},
            {"folder_id": fid, "filename": "clip.mp4", "ext": "mp4", "is_video": 1, "mtime": 100},
        ]
    )

    # Clear initial dirty state
    repo.mark_media_clean([img_id, vid_id])
    assert repo.fetch_dirty_media_batch() == []

    # Assign person with face rect to static photo AND video
    p_id = repo.assign_person(img_id, "Alfred John Craig", rect=(0.25, 0.15, 0.10, 0.12))
    repo.assign_person(vid_id, "Alfred John Craig", rect=(0.50, 0.50, 0.20, 0.20))

    img_detail = repo.get_media_detail(img_id)
    vid_detail = repo.get_media_detail(vid_id)

    # Static image retains MP:RegionInfo rectangle; video zeroes out rectangle
    assert img_detail["people"][0]["rect_x"] == pytest.approx(0.25)
    assert vid_detail["people"][0]["rect_x"] == 0.0
    assert vid_detail["people"][0]["rect_w"] == 0.0

    # Both items should now be marked metadata_dirty = 1 by trg_media_people_dirty_insert
    dirty_batch = repo.fetch_dirty_media_batch()
    assert {d["id"] for d in dirty_batch} == {img_id, vid_id}

    # Filter by person
    person_Filtered = repo.query_media_records(FilterQuerySpec(person_ids=(p_id,)))
    assert {r.id for r in person_Filtered} == {img_id, vid_id}


def test_rtree_spatial_triggers_and_map_queries(db_engine: DualDatabaseEngine) -> None:
    repo = CatalogRepository(db_engine)
    fid = repo.upsert_folder("D:/Geo", "Geo", None, 100)
    valletta_id, london_id = repo.bulk_upsert_media(
        [
            {
                "folder_id": fid,
                "filename": "valletta.jpg",
                "ext": "jpg",
                "lat": 35.8989,
                "lon": 14.5146,
                "location_name": "Valletta, Malta",
                "mtime": 100,
            },
            {
                "folder_id": fid,
                "filename": "london.jpg",
                "ext": "jpg",
                "lat": 51.5074,
                "lon": -0.1278,
                "location_name": "London, UK",
                "mtime": 100,
            },
        ]
    )

    # Bounding box around Mediterranean (Malta)
    malta_hits = repo.query_map_clusters(35.0, 36.5, 14.0, 15.0)
    assert [h["id"] for h in malta_hits] == [valletta_id]

    # Grid filter with bbox
    bbox_records = repo.query_media_records(
        FilterQuerySpec(bbox=(50.0, 52.0, -1.0, 1.0))
    )
    assert [r.id for r in bbox_records] == [london_id]


def test_relative_change_time_taken_math(db_engine: DualDatabaseEngine) -> None:
    repo = CatalogRepository(db_engine)
    fid = repo.upsert_folder("D:/TimeShift", "TimeShift", None, 100)
    t0 = 1700000000
    m1, m2, m3 = repo.bulk_upsert_media(
        [
            {"folder_id": fid, "filename": "a.jpg", "ext": "jpg", "date_taken": t0, "mtime": t0},
            {"folder_id": fid, "filename": "b.jpg", "ext": "jpg", "date_taken": t0 + 60, "mtime": t0},
            {"folder_id": fid, "filename": "c.jpg", "ext": "jpg", "date_taken": t0 + 300, "mtime": t0},
        ]
    )

    # Shift lead item m1 forward by +1 hour (+3600 seconds)
    delta = repo.apply_relative_time_shift([m1, m2, m3], lead_id=m1, new_lead_epoch=t0 + 3600)
    assert delta == 3600

    assert repo.get_media_detail(m1)["date_taken"] == t0 + 3600
    assert repo.get_media_detail(m2)["date_taken"] == t0 + 60 + 3600
    assert repo.get_media_detail(m3)["date_taken"] == t0 + 300 + 3600
    assert repo.get_media_detail(m2)["metadata_dirty"] == 1


def test_lossless_rotation_and_30_day_backup_retention(
    db_engine: DualDatabaseEngine,
) -> None:
    repo = CatalogRepository(db_engine)
    fid = repo.upsert_folder("D:/Edits", "Edits", None, 100)
    mid1, mid2 = repo.bulk_upsert_media(
        [
            {
                "folder_id": fid,
                "filename": "portrait.jpg",
                "ext": "jpg",
                "width": 1000,
                "height": 1500,
                "orientation": 1,
                "mtime": 100,
            },
            {
                "folder_id": fid,
                "filename": "old_edit.jpg",
                "ext": "jpg",
                "width": 800,
                "height": 600,
                "orientation": 1,
                "mtime": 100,
            },
        ]
    )

    # Rotate clockwise 90 deg: orientation 1 -> 6, width & height swapped
    orient_map = repo.rotate_lossless([mid1], clockwise=True)
    assert orient_map[mid1] == 6
    d1 = repo.get_media_detail(mid1)
    assert (d1["width"], d1["height"]) == (1500, 1000)
    assert d1["metadata_dirty"] == 1

    # Record pixel-edit backups: mid1 edited today, mid2 edited 31 days ago
    now = 1800000000
    repo.record_pixel_edit_backup(mid1, "D:/Edits/.OriginalFiles/portrait.jpg", edited_at=now)
    repo.record_pixel_edit_backup(
        mid2, "D:/Edits/.OriginalFiles/old_edit.jpg", edited_at=now - (31 * 86400)
    )

    expired = repo.prune_expired_backups(retention_days=30, now_epoch=now)
    assert expired == ["D:/Edits/.OriginalFiles/old_edit.jpg"]
    assert repo.get_media_detail(mid2)["original_backup_path"] is None
    assert (
        repo.get_media_detail(mid1)["original_backup_path"]
        == "D:/Edits/.OriginalFiles/portrait.jpg"
    )


def test_thumbnail_repository_webp_blobs(db_engine: DualDatabaseEngine) -> None:
    thumbs = ThumbnailRepository(db_engine)
    fake_webp = b"RIFF\x10\x00\x00\x00WEBPVP8 "
    thumbs.put_webp_blob(media_id=42, webp_blob=fake_webp, mtime=1700000500)

    assert thumbs.get_webp_blob(42, expected_mtime=1700000500) == fake_webp
    # Stale mtime returns None
    assert thumbs.get_webp_blob(42, expected_mtime=1700000999) is None
    assert thumbs.get_webp_blobs_batch([42])[42] == fake_webp

    thumbs.invalidate([42])
    assert thumbs.get_webp_blob(42) is None


def test_100k_scale_sub_10ms_query_benchmark(db_engine: DualDatabaseEngine) -> None:
    """Verifies that SQLite indexed filtering & tag_closure joins execute in < 10ms on 100,000 items."""
    repo = CatalogRepository(db_engine)
    fid = repo.upsert_folder("D:/MassiveLibrary", "MassiveLibrary", None, 1700000000)

    t_family = repo.create_tag("PeopleTags/Family")
    t_craig = repo.create_tag("PeopleTags/Family/Craig")
    t_holiday = repo.create_tag("Events/Holiday")

    conn = db_engine.catalog_conn()
    base_ts = 1600000000
    # Fast synthetic bulk insert of 100,000 media rows
    with db_engine.write_lock:
        conn.executemany(
            """
            INSERT INTO media (folder_id, filename, ext, size, mtime, date_taken, rating, flag, width, height, is_video)
            VALUES (?, ?, 'jpg', 204800, ?, ?, ?, ?, 1920, 1080, 0);
            """,
            [
                (
                    fid,
                    f"DSC_{i:06d}.jpg",
                    base_ts + i,
                    base_ts + (i * 60),
                    5 if i % 50 == 0 else (i % 5),
                    1 if i % 200 == 0 else 0,
                )
                for i in range(1, 100_001)
            ],
        )
        # Tag 500 items with 'PeopleTags/Family/Craig' and 250 of those with 'Events/Holiday'
        conn.executemany(
            "INSERT INTO media_tags (media_id, tag_id) VALUES (?, ?);",
            [(i * 200, t_craig) for i in range(1, 501)]
            + [(i * 400, t_holiday) for i in range(1, 251)],
        )
        conn.commit()

    # Warm query planner
    repo.query_media_records(
        FilterQuerySpec(include_tag_ids=(t_family, t_holiday), tag_boolean_mode="AND")
    )

    t_start = time.perf_counter()
    results = repo.query_media_records(
        FilterQuerySpec(
            include_tag_ids=(t_family, t_holiday),
            tag_boolean_mode="AND",
            arrange_by="rating",
        )
    )
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    assert len(results) == 250
    assert elapsed_ms < 10.0, f"Expected < 10ms query time on 100k catalog, got {elapsed_ms:.2f}ms"
