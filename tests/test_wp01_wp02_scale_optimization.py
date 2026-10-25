"""
Verification Suite for WP-01 & WP-02:
1. Parameter chunking for selections exceeding SQLite variable limit (>32,766 items).
2. SQL-side date tree bucket aggregation and spatial grid clustering.
3. Sub-10ms range-based virtual grid selection (Ctrl+A without 400k QModelIndex allocations).
4. O(1) group selected counts and binary-searched index_for_media_id.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from PySide6.QtCore import QItemSelection, QItemSelectionRange, QModelIndex
from PySide6.QtWidgets import QApplication

from legacy_reference.imgBrowserv13 import LruCache
from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository, FilterQuerySpec, MediaItemRecord, ThumbnailRepository
from src.ui.grid_model import CatalogListModel, CatalogRoles


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture()
def db_engine(tmp_path: Path) -> DualDatabaseEngine:
    engine = DualDatabaseEngine(tmp_path / "catalog.db", tmp_path / "thumbs.db")
    yield engine
    engine.close_all()


def test_wp01_large_selection_parameter_chunking(db_engine: DualDatabaseEngine) -> None:
    """Verify operations on >35,000 items succeed without 'too many SQL variables'."""
    repo = CatalogRepository(db_engine)
    thumb_repo = ThumbnailRepository(db_engine)
    fid = repo.upsert_folder("D:/Photos", "Photos", None, 1700000000)

    # Insert 35,000 media items in batches of 5,000
    all_ids: list[int] = []
    for batch_i in range(7):
        rows = [
            {
                "folder_id": fid,
                "filename": f"img_{batch_i}_{i:04d}.jpg",
                "ext": "jpg",
                "size": 1000,
                "mtime": 1700000000,
                "date_taken": 1700000000 + i,
                "rating": 1,
                "flag": 0,
            }
            for i in range(5000)
        ]
        all_ids.extend(repo.bulk_upsert_media(rows))

    assert len(all_ids) == 35000

    # 1. get_selection_summary with 35,000 items
    t0 = time.perf_counter()
    summary = repo.get_selection_summary(all_ids)
    elapsed_summary = (time.perf_counter() - t0) * 1000.0
    assert summary["count"] == 35000
    assert summary["common_rating"] == 1
    assert elapsed_summary < 500.0

    # 2. set_rating on 35,000 items
    repo.set_rating(all_ids, 5)
    summary_rated = repo.get_selection_summary(all_ids[:10])
    assert summary_rated["common_rating"] == 5

    # 3. set_flag on 35,000 items
    repo.set_flag(all_ids, 1)
    summary_flagged = repo.get_selection_summary(all_ids[:10])
    assert summary_flagged["common_flag"] == 1

    # 4. set_caption and set_author on 35,000 items
    repo.set_caption(all_ids, "Summer Vacation")
    repo.set_author(all_ids, "Alice")
    detail = repo.get_media_detail(all_ids[0])
    assert detail["caption"] == "Summer Vacation"
    assert detail["author"] == "Alice"

    # 5. apply_relative_time_shift on 35,000 items
    shift_s = repo.apply_relative_time_shift(all_ids, all_ids[0], 1700003600)
    assert shift_s == 3600

    # 6. assign_tag and remove_tag on 35,000 items
    tid = repo.create_tag("Events/Holiday")
    repo.assign_tag(all_ids, tid)
    summary_tagged = repo.get_selection_summary(all_ids[:500])
    assert any(t["id"] == tid for t in summary_tagged["tags"])
    repo.remove_tag(all_ids, tid)

    # 7. rotate_lossless on 35,000 items
    rot_map = repo.rotate_lossless(all_ids, clockwise=True)
    assert len(rot_map) == 35000
    assert rot_map[all_ids[0]] == 6

    # 8. ThumbnailRepository batch operations on 35,000 items
    blobs = thumb_repo.get_webp_blobs_batch(all_ids)
    assert isinstance(blobs, dict)
    thumb_repo.invalidate(all_ids)


def test_wp01_date_tree_aggregation_and_covering_index(db_engine: DualDatabaseEngine) -> None:
    """Verify SQL-side day bucket aggregation and query_media_records order optimization."""
    repo = CatalogRepository(db_engine)
    fid = repo.upsert_folder("D:/Photos", "Photos", None, 1700000000)

    # Insert items with timestamps across 3 distinct days, including pre-1970
    ts_1955 = -460000000  # 1955
    ts_2020 = 1580000000
    ts_2024 = 1710000000

    repo.bulk_upsert_media(
        [
            {"folder_id": fid, "filename": "old.jpg", "ext": "jpg", "size": 100, "mtime": 100, "date_taken": ts_1955},
            {"folder_id": fid, "filename": "mid.jpg", "ext": "jpg", "size": 100, "mtime": 100, "date_taken": ts_2020},
            {"folder_id": fid, "filename": "new.jpg", "ext": "jpg", "size": 100, "mtime": 100, "date_taken": ts_2024},
        ]
    )

    t0 = time.perf_counter()
    tree = repo.get_date_tree_counts()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 50.0
    assert len(tree) == 3
    years = [item["year"] for item in tree]
    assert 1955 in years
    assert 2020 in years
    assert 2024 in years

    # Query with month grouping and date sort
    spec = FilterQuerySpec(group_by="month", sort_by="date")
    records = repo.query_media_records(spec)
    assert len(records) == 3
    assert records[0].date_taken == ts_2024
    assert records[2].date_taken == ts_1955


def test_wp02_range_based_selection_and_group_counts(qapp: QApplication) -> None:
    """Verify range-based selection extraction in O(ranges) time and O(1) group selected counts."""
    cache = LruCache(500)
    model = CatalogListModel(cache)

    group_pool = [f"Group {g}" for g in range(5)]
    records = [
        MediaItemRecord(
            id=i + 1,
            folder_id=1,
            filename=f"img_{i}.jpg",
            size=1000,
            mtime=1700000000,
            date_taken=1700000000 + i,
            rating=0,
            flag=0,
            width=1920,
            height=1080,
            orientation=1,
            is_video=0,
            caption="",
            group_key=group_pool[i // 1000],
        )
        for i in range(5000)
    ]

    model.set_records(records, group_headers_enabled=True)
    assert model.total_media_count == 5000

    # Simulate Ctrl+A: a single selection range spanning all virtual rows
    total_vrows = model.rowCount()
    sel_range = QItemSelectionRange(model.index(0, 0), model.index(total_vrows - 1, 0))
    q_sel = QItemSelection()
    q_sel.append(sel_range)

    t0 = time.perf_counter()
    selected_ids = model.get_media_ids_for_selection(q_sel)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert len(selected_ids) == 5000
    assert elapsed_ms < 10.0, f"Range extraction took {elapsed_ms:.2f}ms (budget <10ms)"

    # Set selected IDs on model -> updates _group_selected_counts
    model.set_selected_media_ids(selected_ids)

    # Verify GroupSelectedCountRole on each group header is O(1) and exact
    for g_idx, g in enumerate(model._groups):
        hdr_idx = model.index(g.vrow_header, 0)
        count_sel = model.data(hdr_idx, CatalogRoles.GroupSelectedCountRole)
        assert count_sel == 1000

    # Verify binary search index_for_media_id
    mid_to_find = 2500
    idx_found = model.index_for_media_id(mid_to_find)
    assert idx_found.isValid()
    assert model.data(idx_found, CatalogRoles.MediaIdRole) == mid_to_find

    # Verify batched update_items_metadata
    model.update_items_metadata([1, 2, 3], rating=4)
    idx_1 = model.index_for_media_id(1)
    assert model.data(idx_1, CatalogRoles.RatingRole) == 4
