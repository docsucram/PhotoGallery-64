"""
Verification Suite for WP-09:
Tier 1 & Tier 2 Face Tagging Foundation in CatalogRepository & ThumbnailRepository.

Tests:
1. assign_person sets state=1, is_manual=1, deduplicates overlapping regions, updates confirmed_count.
2. Data migration fixes legacy state=0 rows to state=1, is_manual=1 and recalculates confirmed_count.
3. insert_detected_faces inserts state=0, is_manual=0 with ArcFace embedding and avoids duplicate regions.
4. State transitions:
   - confirm_faces (0 -> 1) updates person_id and confirmed_count.
   - reject_face_suggestion resets suggested_person_id and inserts rejection pair into face_rejections.
   - ignore_face sets state = -2 ("Don't show again").
5. face_rejections persistence and get_rejected_face_pairs.
6. update_suggested_matches and get_pending_face_suggestions.
7. get_unassigned_face_embeddings, get_people_centroids, update_person_centroid.
8. ThumbnailRepository face thumbnail operations: put, get, batch get/put, invalidate.
9. get_media_detail and get_selection_summary filter mp.state = 1 (ignoring state 0, -1, -2).
10. Chunked batch operations (>900 IDs) without SQLite variable limit errors.
11. Verification that _media_meta_lookup cache in main_window is cleared on sync callbacks.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication

from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository, ThumbnailRepository


@pytest.fixture()
def db_engine(tmp_path: Path) -> DualDatabaseEngine:
    engine = DualDatabaseEngine(tmp_path / "catalog.db", tmp_path / "thumbs.db")
    yield engine
    engine.close_all()


def _dummy_embedding(val: float = 0.5) -> bytes:
    """Creates a 512-dimensional float32 embedding vector blob (2048 bytes)."""
    return struct.pack("512f", *([val] * 512))


def test_wp09_assign_person_state_and_deduplication(db_engine: DualDatabaseEngine) -> None:
    repo = CatalogRepository(db_engine)
    fid = repo.upsert_folder("D:/Photos", "Photos", None, 1700000000)
    items = [
        {
            "folder_id": fid,
            "filename": "face1.jpg",
            "ext": "jpg",
            "size": 1000,
            "mtime": 1700000000,
            "date_taken": 1700000000,
            "rating": 0,
            "flag": 0,
        }
    ]
    ids = repo.bulk_upsert_media(items)
    mid = ids[0]

    # Assign Person A with rect (0.2, 0.2, 0.3, 0.3)
    p_a_id = repo.upsert_person("Alice")
    res_p_a = repo.assign_person(mid, p_a_id, rect=(0.2, 0.2, 0.3, 0.3))
    assert res_p_a == p_a_id

    conn = db_engine.catalog_conn()
    mp_rows = conn.execute(
        "SELECT id, person_id, state, is_manual FROM media_people WHERE media_id = ?;", (mid,)
    ).fetchall()
    assert len(mp_rows) == 1
    mp_id1 = mp_rows[0]["id"]
    assert mp_rows[0]["person_id"] == p_a_id
    assert mp_rows[0]["state"] == 1
    assert mp_rows[0]["is_manual"] == 1

    person_row = conn.execute("SELECT confirmed_count FROM people WHERE id = ?;", (p_a_id,)).fetchone()
    assert person_row["confirmed_count"] == 1

    # Overlapping assignment on almost identical region (within tolerance 0.02)
    # Assigning Person B to the same face box should update person_id, not duplicate
    p_b_id = repo.upsert_person("Bob")
    res_p_b = repo.assign_person(mid, p_b_id, rect=(0.205, 0.205, 0.295, 0.295))
    assert res_p_b == p_b_id

    mp_rows2 = conn.execute(
        "SELECT id, person_id, state, is_manual FROM media_people WHERE media_id = ?;", (mid,)
    ).fetchall()
    assert len(mp_rows2) == 1  # Reused existing region, did not duplicate
    assert mp_rows2[0]["id"] == mp_id1
    assert mp_rows2[0]["person_id"] == p_b_id

    # Alice should have 0 confirmed count, Bob should have 1
    p_a = conn.execute("SELECT confirmed_count FROM people WHERE id = ?;", (p_a_id,)).fetchone()
    p_b = conn.execute("SELECT confirmed_count FROM people WHERE id = ?;", (p_b_id,)).fetchone()
    assert p_a["confirmed_count"] == 0
    assert p_b["confirmed_count"] == 1


def test_wp09_data_migration_legacy_state(tmp_path: Path) -> None:
    cat_path = tmp_path / "mig_cat.db"
    thm_path = tmp_path / "mig_thm.db"

    # Create catalog manually simulating older DB state
    engine = DualDatabaseEngine(cat_path, thm_path)
    repo = CatalogRepository(engine)
    fid = repo.upsert_folder("D:/Photos", "Photos", None, 1700000000)
    mid = repo.bulk_upsert_media([
        {"folder_id": fid, "filename": "mig.jpg", "ext": "jpg", "size": 100, "mtime": 1700000000, "date_taken": 1700000000, "rating": 0, "flag": 0}
    ])[0]
    pid = repo.upsert_person("Charlie")

    # Artificially insert legacy row with state = 0, is_manual = 0, suggested_person_id = NULL
    conn = engine.catalog_conn()
    with engine.write_lock:
        conn.execute(
            """
            INSERT INTO media_people (media_id, person_id, state, is_manual, rect_x, rect_y, rect_w, rect_h)
            VALUES (?, ?, 0, 0, 0.1, 0.1, 0.2, 0.2);
            """,
            (mid, pid),
        )
        conn.execute("UPDATE people SET confirmed_count = 0 WHERE id = ?;", (pid,))
        conn.commit()

    engine.close_all()

    # Re-open DualDatabaseEngine: _migrate_catalog_schema will run
    engine2 = DualDatabaseEngine(cat_path, thm_path)
    conn2 = engine2.catalog_conn()
    row = conn2.execute("SELECT state, is_manual FROM media_people WHERE media_id = ? AND person_id = ?;", (mid, pid)).fetchone()
    assert row["state"] == 1
    assert row["is_manual"] == 1

    p_row = conn2.execute("SELECT confirmed_count FROM people WHERE id = ?;", (pid,)).fetchone()
    assert p_row["confirmed_count"] == 1
    engine2.close_all()


def test_wp09_insert_detected_faces_and_embeddings(db_engine: DualDatabaseEngine) -> None:
    repo = CatalogRepository(db_engine)
    fid = repo.upsert_folder("D:/Photos", "Photos", None, 1700000000)
    mid = repo.bulk_upsert_media([
        {"folder_id": fid, "filename": "det.jpg", "ext": "jpg", "size": 100, "mtime": 1700000000, "date_taken": 1700000000, "rating": 0, "flag": 0}
    ])[0]

    emb1 = _dummy_embedding(0.123)
    emb2 = _dummy_embedding(0.456)

    faces = [
        {"rect": (0.1, 0.1, 0.2, 0.2), "embedding": emb1},
        {"rect": (0.6, 0.6, 0.25, 0.25), "embedding": emb2},
    ]

    inserted_ids = repo.insert_detected_faces(mid, faces)
    assert len(inserted_ids) == 2

    conn = db_engine.catalog_conn()
    rows = conn.execute("SELECT id, state, is_manual FROM media_people WHERE media_id = ? ORDER BY id ASC;", (mid,)).fetchall()
    assert len(rows) == 2
    assert rows[0]["state"] == 0
    assert rows[0]["is_manual"] == 0

    # Verify embeddings are queryable
    unassigned = repo.get_unassigned_face_embeddings()
    assert len(unassigned) >= 2
    unassigned_ids = [u["id"] for u in unassigned]
    assert inserted_ids[0] in unassigned_ids
    assert inserted_ids[1] in unassigned_ids

    # Dedup check: re-inserting same face box on same media should return existing IDs and avoid duplicate rows
    inserted_again = repo.insert_detected_faces(mid, faces)
    assert inserted_again == inserted_ids
    total_count = conn.execute("SELECT COUNT(*) FROM media_people WHERE media_id = ?;", (mid,)).fetchone()[0]
    assert total_count == 2


def test_wp09_state_transitions_and_rejections(db_engine: DualDatabaseEngine) -> None:
    repo = CatalogRepository(db_engine)
    fid = repo.upsert_folder("D:/Photos", "Photos", None, 1700000000)
    mid = repo.bulk_upsert_media([
        {"folder_id": fid, "filename": "state_test.jpg", "ext": "jpg", "size": 100, "mtime": 1700000000, "date_taken": 1700000000, "rating": 0, "flag": 0}
    ])[0]

    f_ids = repo.insert_detected_faces(mid, [
        {"rect": (0.1, 0.1, 0.2, 0.2), "embedding": _dummy_embedding(0.1)},
        {"rect": (0.4, 0.4, 0.2, 0.2), "embedding": _dummy_embedding(0.2)},
        {"rect": (0.7, 0.7, 0.2, 0.2), "embedding": _dummy_embedding(0.3)},
    ])
    f1, f2, f3 = f_ids

    p1 = repo.upsert_person("David")
    p2 = repo.upsert_person("Emma")

    # 1. Update suggested matches
    repo.update_suggested_matches([(f1, p1, 0.92), (f2, p2, 0.85)])
    pending = repo.get_pending_face_suggestions(limit=10)
    assert len(pending) == 2
    assert {p["face_id"] for p in pending} == {f1, f2}

    # 2. Confirm f1 as p1
    repo.confirm_faces([f1], p1)
    conn = db_engine.catalog_conn()
    r1 = conn.execute("SELECT person_id, state, suggested_person_id FROM media_people WHERE id = ?;", (f1,)).fetchone()
    assert r1["person_id"] == p1
    assert r1["state"] == 1
    assert r1["suggested_person_id"] is None

    p1_row = conn.execute("SELECT confirmed_count FROM people WHERE id = ?;", (p1,)).fetchone()
    assert p1_row["confirmed_count"] == 1

    # 3. Reject f2 suggestion (suggested Emma)
    repo.reject_face_suggestion(f2, p2)
    r2 = conn.execute("SELECT person_id, state, suggested_person_id FROM media_people WHERE id = ?;", (f2,)).fetchone()
    assert r2["state"] == 0
    assert r2["suggested_person_id"] is None

    rejected_pairs = repo.get_rejected_face_pairs()
    assert (f2, p2) in rejected_pairs

    # 4. Ignore f3 ("Don't show again" -> state = -2)
    repo.ignore_face(f3)
    r3 = conn.execute("SELECT state, person_id FROM media_people WHERE id = ?;", (f3,)).fetchone()
    assert r3["state"] == -2
    assert r3["person_id"] is None


def test_wp09_people_centroids(db_engine: DualDatabaseEngine) -> None:
    repo = CatalogRepository(db_engine)
    pid = repo.upsert_person("Frank")

    # Initially centroid is None
    centroids = repo.get_people_centroids()
    assert pid not in centroids

    emb = _dummy_embedding(0.777)
    repo.update_person_centroid(pid, emb)

    centroids2 = repo.get_people_centroids()
    assert pid in centroids2
    assert centroids2[pid] == emb


def test_wp09_thumbnail_repository_face_ops(db_engine: DualDatabaseEngine) -> None:
    thumb_repo = ThumbnailRepository(db_engine)

    fake_webp_1 = b"RIFF....WEBPVP8L...face1"
    fake_webp_2 = b"RIFF....WEBPVP8L...face2"
    fake_webp_3 = b"RIFF....WEBPVP8L...face3"

    thumb_repo.put_face_thumbnail(101, fake_webp_1)
    retrieved = thumb_repo.get_face_thumbnail(101)
    assert retrieved == fake_webp_1

    # Batch put
    thumb_repo.put_face_thumbnails_batch([(102, fake_webp_2), (103, fake_webp_3)])
    batch_res = thumb_repo.get_face_thumbnails_batch([101, 102, 103, 999])
    assert len(batch_res) == 3
    assert batch_res[101] == fake_webp_1
    assert batch_res[102] == fake_webp_2
    assert batch_res[103] == fake_webp_3
    assert 999 not in batch_res

    # Invalidate
    thumb_repo.invalidate_face_thumbnails([102])
    assert thumb_repo.get_face_thumbnail(102) is None
    assert thumb_repo.get_face_thumbnail(101) == fake_webp_1


def test_wp09_media_detail_and_summary_filter_state(db_engine: DualDatabaseEngine) -> None:
    repo = CatalogRepository(db_engine)
    fid = repo.upsert_folder("D:/Photos", "Photos", None, 1700000000)
    mid = repo.bulk_upsert_media([
        {"folder_id": fid, "filename": "detail_filter.jpg", "ext": "jpg", "size": 100, "mtime": 1700000000, "date_taken": 1700000000, "rating": 0, "flag": 0}
    ])[0]

    p_conf = repo.upsert_person("Confirmed User")
    p_sugg = repo.upsert_person("Suggested User")

    # Insert one confirmed face (state = 1)
    repo.assign_person(mid, p_conf, rect=(0.1, 0.1, 0.2, 0.2))

    # Insert one unconfirmed/suggested face (state = 0)
    conn = db_engine.catalog_conn()
    with db_engine.write_lock:
        conn.execute(
            """
            INSERT INTO media_people (media_id, person_id, state, is_manual, rect_x, rect_y, rect_w, rect_h)
            VALUES (?, ?, 0, 0, 0.5, 0.5, 0.2, 0.2);
            """,
            (mid, p_sugg),
        )
        conn.commit()

    detail = repo.get_media_detail(mid)
    people_in_detail = [p["name"] for p in detail["people"]]
    assert "Confirmed User" in people_in_detail
    assert "Suggested User" not in people_in_detail

    summary = repo.get_selection_summary([mid])
    people_in_summary = [p["name"] for p in summary["people"]]
    assert "Confirmed User" in people_in_summary
    assert "Suggested User" not in people_in_summary


def test_wp09_chunked_batch_execution(db_engine: DualDatabaseEngine) -> None:
    """Verifies that queries with > 900 IDs work seamlessly using _iter_id_chunks."""
    thumb_repo = ThumbnailRepository(db_engine)

    # Put 1000 face thumbnails
    entries = [(i, f"face_{i}".encode("utf-8")) for i in range(1, 1001)]
    thumb_repo.put_face_thumbnails_batch(entries)

    ids = list(range(1, 1001))
    retrieved = thumb_repo.get_face_thumbnails_batch(ids)
    assert len(retrieved) == 1000

    thumb_repo.invalidate_face_thumbnails(ids)
    retrieved_after = thumb_repo.get_face_thumbnails_batch(ids)
    assert len(retrieved_after) == 0


def test_wp09_media_meta_lookup_invalidation_hook(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verifies that _media_meta_lookup in PhotoGalleryMainWindow is emptied when sync/watcher events fire."""
    import sys
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    from src.ui.main_window import PhotoGalleryMainWindow

    engine = DualDatabaseEngine(tmp_path / "win_cat.db", tmp_path / "win_thm.db")
    win = PhotoGalleryMainWindow(engine, start_xmp_daemon=False)

    # Populate cache artificially
    win._media_meta_lookup[123] = (1700000000, 1)
    assert len(win._media_meta_lookup) == 1

    # Trigger on_sync_finished
    win.on_sync_finished(0, True)
    assert len(win._media_meta_lookup) == 0

    # Populate again and test _on_folder_watcher_delta
    win._media_meta_lookup[456] = (1700000000, 1)
    assert len(win._media_meta_lookup) == 1
    win._on_folder_watcher_delta("D:/Photos", 1, 0, 0, 0)
    assert len(win._media_meta_lookup) == 0

    win.close()
    engine.close_all()
