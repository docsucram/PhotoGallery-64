"""
Unit tests for Phase 24:
1. Search By Box 'Search All' behavior:
   - Does not hide the search box or remove active filters (people, tags, ratings, dates, etc.).
   - Removes only the folder filter and searches across all folders.
2. World Map View Presentation & Interactions:
   - Circular pins render thumbnail images and badges when count > 1.
   - Clicking a cluster with 1 item opens SingleMediaViewer directly.
   - Clicking a cluster with multiple items filters the gallery to that location and displays an active location chip.
   - 'Filter Gallery to Map Area' keeps the search scope bar visible, shows an active location chip, and filters by bbox.
   - Removing the location chip clears the filter and re-runs the query.
"""

from __future__ import annotations

import base64
import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository, FilterQuerySpec, ThumbnailRepository
from src.ui.left_nav_dock import LeftNavigationDock
from src.ui.main_window import PhotoGalleryMainWindow
from src.ui.map_view import WorldMapView, _InteractiveMapCanvas, cluster_gps_points
from src.ui.search_scope_bar import FilterChip, SearchScopeBar


@pytest.fixture(scope="session")
def qapp():
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["GALLERY_NO_WEBENGINE"] = "1"
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def dual_db(tmp_path: Path):
    cat_path = tmp_path / "catalog.db"
    thm_path = tmp_path / "thumbnails.db"
    return DualDatabaseEngine(cat_path, thm_path)


def _insert_test_folder(engine: DualDatabaseEngine, path: str, name: str) -> int:
    conn = engine.catalog_conn()
    with engine.write_lock:
        cur = conn.execute(
            "INSERT INTO folders (path, name) VALUES (?, ?);",
            (path, name),
        )
        conn.commit()
        return cur.lastrowid


def _insert_test_media(
    engine: DualDatabaseEngine,
    fid: int,
    fname: str,
    lat: float | None = None,
    lon: float | None = None,
    location_name: str | None = None,
    rating: int = 0,
) -> int:
    conn = engine.catalog_conn()
    ext = Path(fname).suffix.lower()
    with engine.write_lock:
        cur = conn.execute(
            """
            INSERT INTO media (folder_id, filename, ext, size, mtime, width, height, is_video, lat, lon, location_name, rating)
            VALUES (?, ?, ?, 1024, 1600000000, 1920, 1080, 0, ?, ?, ?, ?);
            """,
            (fid, fname, ext, lat, lon, location_name, rating),
        )
        mid = cur.lastrowid
        conn.commit()
    return mid


def test_search_all_preserves_other_filters_and_does_not_hide_bar(qapp, dual_db):
    """Verifies that clicking 'Search all' removes ONLY the folder filter and keeps the bar open."""
    cat_repo = CatalogRepository(dual_db)
    thm_repo = ThumbnailRepository(dual_db)

    # Insert two folders and media
    f1 = _insert_test_folder(dual_db, "C:/Photos/Vacation", "Vacation")
    f2 = _insert_test_folder(dual_db, "C:/Photos/Work", "Work")

    # Insert media
    m1 = _insert_test_media(dual_db, f1, "beach.jpg", rating=5)
    m2 = _insert_test_media(dual_db, f2, "office.jpg", rating=5)
    m3 = _insert_test_media(dual_db, f1, "night.jpg", rating=2)

    win = PhotoGalleryMainWindow(dual_db, start_xmp_daemon=False)
    win.catalog_repo = cat_repo
    win.thumb_repo = thm_repo

    # Set active folder to Vacation (f1)
    win.left_nav_dock._folder_id = f1
    win.left_nav_dock._folder_name = "Vacation"
    # Set rating filter to 5 stars
    win.left_nav_dock.select_ratings([5])

    # Scope bar should be visible (not hidden) due to rating filter
    win._update_scope_bar_visibility()
    assert not win.scope_bar.isHidden()
    assert not win.scope_bar.btn_search_all.isHidden()

    # Query before Search All: only 1 photo (beach.jpg in Vacation with 5 stars)
    records = win.rebuild_and_run_query()
    assert len(records) == 1
    assert records[0].id == m1

    # User clicks "Search all"
    win._on_search_all_requested()

    # Scope bar must still be open (not hidden)
    assert not win.scope_bar.isHidden()
    # Folder filter should be None (All folders)
    assert win.left_nav_dock._folder_id is None
    # Rating filter must still be intact (5 stars)
    assert 5 in win.left_nav_dock._ratings

    # Query after Search All: both beach.jpg and office.jpg (5 stars across all folders)
    records_after = win.rebuild_and_run_query()
    assert len(records_after) == 2
    record_ids = {r.id for r in records_after}
    assert record_ids == {m1, m2}

    win.close()


def test_map_cluster_single_click_opens_viewer(qapp, dual_db):
    """Verifies that clicking a single-item cluster opens SingleMediaViewer directly."""
    cat_repo = CatalogRepository(dual_db)
    f1 = _insert_test_folder(dual_db, "C:/Photos/Tokyo", "Tokyo")
    m1 = _insert_test_media(dual_db, f1, "tokyo_tower.jpg", lat=35.6586, lon=139.7454, location_name="Tokyo Tower")

    win = PhotoGalleryMainWindow(dual_db, start_xmp_daemon=False)
    win.open_single_viewer = MagicMock()

    # Single-item cluster clicked
    win._on_map_cluster_clicked([m1])
    win.open_single_viewer.assert_called_once_with(m1)

    win.open_single_viewer.reset_mock()
    win._on_map_cluster_clicked_data([m1], "Tokyo Tower")
    win.open_single_viewer.assert_called_once_with(m1)

    win.close()


def test_map_cluster_multi_click_filters_gallery_and_shows_location_chip(qapp, dual_db):
    """Verifies that clicking a multi-item cluster filters the gallery to that location with an active chip."""
    cat_repo = CatalogRepository(dual_db)
    thm_repo = ThumbnailRepository(dual_db)
    f1 = _insert_test_folder(dual_db, "C:/Photos/Paris", "Paris")
    m1 = _insert_test_media(dual_db, f1, "eiffel1.jpg", lat=48.8584, lon=2.2945, location_name="Eiffel Tower")
    m2 = _insert_test_media(dual_db, f1, "eiffel2.jpg", lat=48.8585, lon=2.2946, location_name="Eiffel Tower")
    m3 = _insert_test_media(dual_db, f1, "louvre.jpg", lat=48.8606, lon=2.3376, location_name="Louvre")

    win = PhotoGalleryMainWindow(dual_db, start_xmp_daemon=False)
    win.show_grid_view()

    # Multi-item cluster clicked for Eiffel Tower (m1, m2)
    win._on_map_cluster_clicked_data([m1, m2], "Eiffel Tower")

    # Switched to grid view
    assert win.central_stack.currentWidget() == win.grid_view
    # Scope bar is not hidden and shows location label
    assert not win.scope_bar.isHidden()
    assert "Eiffel Tower" in (win._active_map_location_label or "")

    # Query filtered to m1 and m2
    records = win.rebuild_and_run_query()
    assert len(records) == 2
    assert {r.id for r in records} == {m1, m2}

    # Location chip is rendered in chips layout
    chips = [
        win.scope_bar.chips_layout.itemAt(i).widget()
        for i in range(win.scope_bar.chips_layout.count())
        if win.scope_bar.chips_layout.itemAt(i).widget() is not None
    ]
    location_chips = [c for c in chips if isinstance(c, FilterChip)]
    assert len(location_chips) >= 1

    # User clicks remove on location chip
    win._on_clear_location_filter()
    assert win._active_map_media_ids is None
    assert win._active_map_location_label is None

    # Gallery resets to show all items
    records_all = win.rebuild_and_run_query()
    assert len(records_all) == 3

    win.close()


def test_filter_gallery_to_map_area_button(qapp, dual_db):
    """Verifies that 'Filter Gallery to Map Area' applies bounding box and keeps search bar visible."""
    cat_repo = CatalogRepository(dual_db)
    f1 = _insert_test_folder(dual_db, "C:/Photos/GPS", "GPS")
    m_paris = _insert_test_media(dual_db, f1, "paris.jpg", lat=48.85, lon=2.35)
    m_ny = _insert_test_media(dual_db, f1, "nyc.jpg", lat=40.71, lon=-74.00)

    win = PhotoGalleryMainWindow(dual_db, start_xmp_daemon=False)

    # Filter to Europe bbox (lat 35..60, lon -10..30)
    win.on_map_bbox_filter_requested(35.0, 60.0, -10.0, 30.0)

    # Scope bar is visible with Location chip
    assert not win.scope_bar.isHidden()
    assert win._active_bbox == (35.0, 60.0, -10.0, 30.0)
    assert win._active_map_location_label is not None

    records = win.rebuild_and_run_query()
    assert len(records) == 1
    assert records[0].id == m_paris

    # Clear location filter
    win._on_clear_location_filter()
    assert win._active_bbox is None

    win.close()


def test_map_view_cluster_thumbnails(qapp, dual_db):
    """Verifies that WorldMapView enriches clusters with WebP thumbnails."""
    cat_repo = CatalogRepository(dual_db)
    thm_repo = ThumbnailRepository(dual_db)
    f1 = _insert_test_folder(dual_db, "C:/Photos/Rome", "Rome")
    m1 = _insert_test_media(dual_db, f1, "colosseum.jpg", lat=41.8902, lon=12.4922, location_name="Colosseum")

    # Generate a dummy 1x1 WebP blob
    from PIL import Image
    import io
    im = Image.new("RGB", (32, 32), color="blue")
    buf = io.BytesIO()
    im.save(buf, format="WEBP")
    fake_webp = buf.getvalue()
    thm_repo.put_webp_blob(m1, fake_webp, 1600000000)

    map_view = WorldMapView()
    map_view.bind_repository(cat_repo, thm_repo)

    clusters = map_view.set_viewport_bbox(40.0, 43.0, 10.0, 15.0)
    assert len(clusters) == 1
    c = clusters[0]
    assert c["count"] == 1
    assert "thumb_url" in c
    assert c["thumb_url"].startswith("data:image/webp;base64,")
    assert "pixmap" in c
    assert not c["pixmap"].isNull()

    map_view.close()
