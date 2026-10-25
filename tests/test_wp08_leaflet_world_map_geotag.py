"""
Comprehensive unit test suite for Work Package WP-08:
- Interactive Leaflet World Map View (QWebEngineView + QWebChannel + Offline Fallback)
- Geotag Coordinate Lookup & Offline Gazetteer (src/utils/geotag_resolver.py)
- SQLite R*Tree (media_rtree) spatial synchronization & clustering
- Undo/Redo for Geotag edits in PhotoGalleryMainWindow
"""

import json
import os
import sqlite3
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository
from src.ui.map_view import (
    HAS_WEBENGINE,
    MapBridge,
    WorldMapView,
    _InteractiveMapCanvas,
    cluster_gps_points,
    generate_leaflet_html,
)
from src.utils.geotag_resolver import (
    OFFLINE_GAZETTEER,
    parse_coordinates,
    resolve_location_query,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def temp_db_engine(tmp_path: Path):
    cat_path = tmp_path / "catalog.db"
    thm_path = tmp_path / "thumbnails.db"
    engine = DualDatabaseEngine(cat_path, thm_path)

    # Insert sample folder and media
    conn = engine.catalog_conn()
    conn.execute(
        "INSERT INTO folders (path, name) VALUES ('C:/Photos', 'Photos');"
    )
    folder_id = conn.execute("SELECT id FROM folders WHERE path = 'C:/Photos';").fetchone()[0]

    conn.execute(
        """
        INSERT INTO media (folder_id, filename, ext, size, mtime, date_taken, rating, flag)
        VALUES (?, 'photo1.jpg', 'jpg', 1024, 1700000000, 1700000000, 5, 1);
        """,
        (folder_id,),
    )
    conn.execute(
        """
        INSERT INTO media (folder_id, filename, ext, size, mtime, date_taken, rating, flag)
        VALUES (?, 'photo2.jpg', 'jpg', 2048, 1700000001, 1700000001, 4, 0);
        """,
        (folder_id,),
    )
    conn.execute(
        """
        INSERT INTO media (folder_id, filename, ext, size, mtime, date_taken, rating, flag)
        VALUES (?, 'photo3.jpg', 'jpg', 4096, 1700000002, 1700000002, 3, 0);
        """,
        (folder_id,),
    )
    conn.commit()

    yield engine
    engine.close_all()


# -------------------------------------------------------------------------
# 1. Geotag Coordinate & Gazetteer Resolver Tests
# -------------------------------------------------------------------------
def test_parse_coordinates_decimal():
    assert parse_coordinates("51.5074, -0.1278") == (51.5074, -0.1278)
    assert parse_coordinates("-33.8688, 151.2093") == (-33.8688, 151.2093)
    assert parse_coordinates("48.8566 -2.3522") == (48.8566, -2.3522)


def test_parse_coordinates_cardinal():
    assert parse_coordinates("51.5074° N, 0.1278° W") == (51.5074, -0.1278)
    assert parse_coordinates("33.8688 S, 151.2093 E") == (-33.8688, 151.2093)


def test_parse_coordinates_out_of_bounds():
    assert parse_coordinates("95.0, 10.0") is None
    assert parse_coordinates("10.0, 185.0") is None
    assert parse_coordinates("not coordinates") is None
    assert parse_coordinates("") is None


def test_resolve_location_query_offline_gazetteer():
    lat, lon, name = resolve_location_query("London", allow_network=False)
    assert lat is not None and abs(lat - 51.5074) < 0.01
    assert lon is not None and abs(lon - (-0.1278)) < 0.01
    assert "London" in name

    lat, lon, name = resolve_location_query("Paris", allow_network=False)
    assert lat is not None and abs(lat - 48.8566) < 0.01
    assert "Paris" in name

    lat, lon, name = resolve_location_query("Tokyo", allow_network=False)
    assert lat is not None and abs(lat - 35.6762) < 0.01
    assert "Tokyo" in name

    lat, lon, name = resolve_location_query("Sydney", allow_network=False)
    assert lat is not None and abs(lat - (-33.8688)) < 0.01
    assert "Sydney" in name


def test_resolve_location_query_custom_place_fallback():
    lat, lon, name = resolve_location_query("Grandma's Summer Cottage", allow_network=False)
    assert lat is None
    assert lon is None
    assert name == "Grandma's Summer Cottage"


def test_resolve_location_query_empty():
    lat, lon, name = resolve_location_query("", allow_network=False)
    assert lat is None and lon is None and name == ""


# -------------------------------------------------------------------------
# 2. Database Geotag & SQLite R*Tree Spatial Synchronization Tests
# -------------------------------------------------------------------------
def test_set_geotag_updates_media_and_rtree(temp_db_engine):
    repo = CatalogRepository(temp_db_engine)
    conn = temp_db_engine.catalog_conn()

    # Assign London geotag to photo 1 and photo 2
    repo.set_geotag([1, 2], 51.5074, -0.1278, "London, United Kingdom")

    # Verify media table columns
    row1 = conn.execute("SELECT lat, lon, location_name, metadata_dirty FROM media WHERE id = 1;").fetchone()
    assert abs(float(row1["lat"]) - 51.5074) < 1e-4
    assert abs(float(row1["lon"]) - (-0.1278)) < 1e-4
    assert row1["location_name"] == "London, United Kingdom"
    assert row1["metadata_dirty"] == 1

    # Verify SQLite media_rtree virtual table was populated by triggers
    rtree_rows = conn.execute("SELECT * FROM media_rtree WHERE id IN (1, 2);").fetchall()
    assert len(rtree_rows) == 2
    for r in rtree_rows:
        assert abs(float(r["min_lat"]) - 51.5074) < 1e-4
        assert abs(float(r["max_lon"]) - (-0.1278)) < 1e-4

    # Query clusters inside UK bounding box
    clusters = repo.query_map_clusters(50.0, 55.0, -5.0, 2.0, grid_degrees=1.0)
    assert len(clusters) >= 1
    top_cluster = clusters[0]
    assert top_cluster["count"] == 2
    assert 1 in top_cluster["media_ids"]
    assert 2 in top_cluster["media_ids"]

    # Clear geotag from photo 1
    repo.set_geotag([1], None, None, None)
    cleared_row = conn.execute("SELECT lat, lon, location_name FROM media WHERE id = 1;").fetchone()
    assert cleared_row["lat"] is None
    assert cleared_row["lon"] is None
    assert cleared_row["location_name"] is None

    # Verify photo 1 was removed from media_rtree
    rtree_1 = conn.execute("SELECT * FROM media_rtree WHERE id = 1;").fetchone()
    assert rtree_1 is None

    # Photo 2 still remains in media_rtree
    rtree_2 = conn.execute("SELECT * FROM media_rtree WHERE id = 2;").fetchone()
    assert rtree_2 is not None


def test_get_geotag_map_and_restore(temp_db_engine):
    repo = CatalogRepository(temp_db_engine)
    repo.set_geotag([1], 48.8566, 2.3522, "Paris, France")
    repo.set_geotag([2], 35.6762, 139.6503, "Tokyo, Japan")

    geotag_map = repo.get_geotag_map([1, 2])
    assert 1 in geotag_map and 2 in geotag_map
    assert abs(geotag_map[1][0] - 48.8566) < 1e-4
    assert geotag_map[1][2] == "Paris, France"
    assert abs(geotag_map[2][0] - 35.6762) < 1e-4

    # Change them
    repo.set_geotag([1, 2], None, None, None)
    assert repo.get_geotag_map([1])[1][0] is None

    # Restore from snapshot
    repo.restore_geotag_map(geotag_map)
    restored = repo.get_geotag_map([1, 2])
    assert abs(restored[1][0] - 48.8566) < 1e-4
    assert restored[1][2] == "Paris, France"
    assert abs(restored[2][0] - 35.6762) < 1e-4


# -------------------------------------------------------------------------
# 3. Leaflet HTML Generation & Map Bridge Tests
# -------------------------------------------------------------------------
def test_generate_leaflet_html():
    clusters = [
        {"lat": 51.5074, "lon": -0.1278, "count": 12, "label": "London", "media_ids": [1, 2, 3]},
        {"lat": 48.8566, "lon": 2.3522, "count": 5, "label": "Paris", "media_ids": [4, 5]},
    ]
    html = generate_leaflet_html(clusters)
    assert "<!DOCTYPE html>" in html
    assert "leaflet.js" in html
    assert "updateClusters" in html
    assert "setMapBounds" in html
    assert "mapBridge" in html
    assert "London" in html
    assert "Paris" in html


def test_map_bridge_signals(qapp):
    bridge = MapBridge()
    clicked_ids = []
    bounds_received = []

    bridge.cluster_clicked.connect(clicked_ids.extend)
    bridge.bounds_changed.connect(lambda s, n, w, e: bounds_received.append((s, n, w, e)))

    bridge.onClusterClicked(json.dumps([42, 99]))
    assert clicked_ids == [42, 99]

    bridge.onBoundsChanged(10.5, 20.5, -30.0, 40.0)
    assert len(bounds_received) == 1
    assert bounds_received[0] == (10.5, 20.5, -30.0, 40.0)


# -------------------------------------------------------------------------
# 4. WorldMapView Widget Tests
# -------------------------------------------------------------------------
def test_world_map_view_widget(qapp, temp_db_engine):
    repo = CatalogRepository(temp_db_engine)
    repo.set_geotag([1, 2], 51.5074, -0.1278, "London, United Kingdom")
    repo.set_geotag([3], 40.7128, -74.0060, "New York, NY")

    map_view = WorldMapView()
    map_view.bind_repository(repo)
    map_view.resize(800, 600)
    map_view.show()
    qapp.processEvents()

    # Query whole world
    clusters = map_view.set_viewport_bbox(-85.0, 85.0, -180.0, 180.0)
    assert len(clusters) >= 2
    assert map_view.cluster_list.count() == len(clusters)
    assert "geotagged items" in map_view.lbl_cluster_summary.text()

    # Test toggling map engine
    if HAS_WEBENGINE:
        assert map_view.btn_toggle_engine.isEnabled()
        map_view._toggle_map_engine()
        assert map_view.map_stack.currentWidget() == map_view.canvas
        assert "Vector Offline" in map_view.btn_toggle_engine.text()
        map_view._toggle_map_engine()
        assert map_view.map_stack.currentWidget() == map_view.web_view
        assert "Leaflet Web" in map_view.btn_toggle_engine.text()

    # Test clicking cluster list item
    clicked_clusters = []
    map_view.cluster_clicked.connect(clicked_clusters.append)
    first_item = map_view.cluster_list.item(0)
    map_view._on_cluster_list_clicked(first_item)
    assert len(clicked_clusters) == 1
    assert isinstance(clicked_clusters[0], list) and len(clicked_clusters[0]) > 0

    # Test filter to bbox signal
    emitted_bboxes = []
    map_view.filter_to_bbox_requested.connect(
        lambda min_la, max_la, min_lo, max_lo: emitted_bboxes.append((min_la, max_la, min_lo, max_lo))
    )
    map_view.btn_filter_grid.click()
    assert len(emitted_bboxes) == 1
    assert emitted_bboxes[0] == (-85.0, 85.0, -180.0, 180.0)

    map_view.close()


def test_geotag_undo_command(qapp, temp_db_engine):
    from PySide6.QtGui import QUndoStack
    from src.ui.main_window import _GeotagUndoCommand

    repo = CatalogRepository(temp_db_engine)
    repo.set_geotag([1], 51.5074, -0.1278, "London, United Kingdom")

    undo_stack = QUndoStack()

    class StubWin:
        def __init__(self, r):
            self.catalog_repo = r
            self._selected_ids = [1]
            self.right_tag_dock = type("Dock", (), {"bind_selection": lambda *a: None})()
            self.central_stack = type("Stack", (), {"currentWidget": lambda *a: None})()
            self.map_view = None

    stub = StubWin(repo)
    cmd = _GeotagUndoCommand(stub, [1], 48.8566, 2.3522, "Paris, France")
    undo_stack.push(cmd)

    # After push/redo: Photo 1 should have Paris
    m = repo.get_geotag_map([1])
    assert abs(m[1][0] - 48.8566) < 1e-4
    assert m[1][2] == "Paris, France"

    # Undo: Photo 1 should revert to London
    undo_stack.undo()
    m_undo = repo.get_geotag_map([1])
    assert abs(m_undo[1][0] - 51.5074) < 1e-4
    assert m_undo[1][2] == "London, United Kingdom"

    # Redo: Photo 1 should be Paris again
    undo_stack.redo()
    m_redo = repo.get_geotag_map([1])
    assert abs(m_redo[1][0] - 48.8566) < 1e-4
    assert m_redo[1][2] == "Paris, France"
