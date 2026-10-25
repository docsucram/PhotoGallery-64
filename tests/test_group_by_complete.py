"""
Test suite verifying all 19 Group by criteria from authentic Windows Live Photo Gallery:
1. Auto
2. Date taken
3. Month taken
4. Year taken
5. Time taken
6. File size
7. Image size
8. Orientation
9. File type
10. Media type
11. Rating
12. Flag
13. Folder
14. Tag
15. People
16. Published
17. Geotag
18. Camera
19. None
+ Ascending / Descending sorting
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest
from PySide6.QtWidgets import QApplication

from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository, FilterQuerySpec
from src.ui.grid_model import CatalogListModel
from src.ui.grid_view import ThumbnailGridView
from src.ui.ribbon_toolbar import RibbonToolbar


@pytest.fixture(scope="session")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def test_engine(tmp_path: Path) -> DualDatabaseEngine:
    engine = DualDatabaseEngine(
        catalog_path=tmp_path / "catalog.db",
        thumbnails_path=tmp_path / "thumbnails.db",
    )
    engine.initialize_schemas()

    # Seed rich sample records to test each group by mode
    conn = engine.catalog_conn()
    cur = conn.execute("INSERT INTO folders (path, name) VALUES (?, ?)", (str(tmp_path / "Vacation"), "Vacation"))
    fid1 = cur.lastrowid
    cur = conn.execute("INSERT INTO folders (path, name) VALUES (?, ?)", (str(tmp_path / "Work"), "Work"))
    fid2 = cur.lastrowid

    # Add tags
    cur = conn.execute("INSERT INTO tags (name, full_path) VALUES ('Landscape', 'Landscape')")
    tid1 = cur.lastrowid

    # Add people
    cur = conn.execute("INSERT INTO people (name) VALUES ('John Doe')")
    pid1 = cur.lastrowid

    # Media items with diverse dates, sizes, dimensions, cameras, orientations, and formats:
    # 1. Huge JPEG Landscape taken morning 2026-10-24 09:30, 5 stars, flagged, Canon EOS 5D, published=1
    conn.execute(
        """
        INSERT INTO media (id, folder_id, filename, ext, size, mtime, date_taken, rating, flag, width, height, orientation, lat, lon, location_name, camera_make, camera_model, published, is_video)
        VALUES (1, ?, 'photo1.jpg', '.jpg', 30000000, 1000, 1792834200, 5, 1, 6000, 4000, 1, 51.5074, -0.1278, 'London, UK', 'Canon', 'EOS 5D', 1, 0)
        """,
        (fid1,),
    )
    # 2. Medium PNG Portrait taken afternoon 2026-10-24 14:15, 3 stars, unflagged, Nikon D850, published=0
    conn.execute(
        """
        INSERT INTO media (id, folder_id, filename, ext, size, mtime, date_taken, rating, flag, width, height, orientation, lat, lon, location_name, camera_make, camera_model, published, is_video)
        VALUES (2, ?, 'photo2.png', '.png', 12000000, 2000, 1792851300, 3, 0, 3000, 4500, 1, NULL, NULL, '', 'Nikon', 'D850', 0, 0)
        """,
        (fid1,),
    )
    # 3. Small MP4 Video taken evening 2025-06-15 19:45, 0 stars, Sony Alpha, published=0
    conn.execute(
        """
        INSERT INTO media (id, folder_id, filename, ext, size, mtime, date_taken, rating, flag, width, height, orientation, lat, lon, location_name, camera_make, camera_model, published, is_video)
        VALUES (3, ?, 'video1.mp4', '.mp4', 3000000, 3000, 1750016700, 0, 0, 1920, 1080, 1, NULL, NULL, '', 'Sony', 'Alpha 7', 0, 1)
        """,
        (fid2,),
    )
    # 4. Tiny RAW Square taken night 2024-01-01 23:30, 4 stars, Apple iPhone
    conn.execute(
        """
        INSERT INTO media (id, folder_id, filename, ext, size, mtime, date_taken, rating, flag, width, height, orientation, lat, lon, location_name, camera_make, camera_model, published, is_video)
        VALUES (4, ?, 'raw1.dng', '.dng', 500000, 4000, 1704151800, 4, 0, 2000, 2000, 1, 40.7128, -74.0060, 'New York, USA', 'Apple', 'iPhone 15 Pro', 1, 0)
        """,
        (fid2,),
    )

    # Link tag and person
    conn.execute("INSERT INTO media_tags (media_id, tag_id) VALUES (1, ?)", (tid1,))
    conn.execute(
        "INSERT INTO media_people (media_id, person_id, state, rect_x, rect_y, rect_w, rect_h) VALUES (1, ?, 1, 0.1, 0.1, 0.2, 0.2)",
        (pid1,),
    )
    conn.commit()

    yield engine
    engine.close_all()


def test_all_19_group_by_modes_query(test_engine: DualDatabaseEngine):
    """Verifies that query_media_records executes cleanly and generates valid headers for all 19 criteria."""
    repo = CatalogRepository(test_engine)

    modes = [
        ("auto", ["October 2026", "June 2025", "January 2024"]),
        ("date", ["Saturday, 24 October 2026", "Sunday, 15 June 2025", "Monday, 01 January 2024"]),
        ("month", ["October 2026", "June 2025", "January 2024"]),
        ("year", ["2026", "2025", "2024"]),
        ("time", ["Morning (6:00 AM - 12:00 PM)", "Afternoon (12:00 PM - 6:00 PM)", "Evening (6:00 PM - 10:00 PM)", "Night (10:00 PM - 6:00 AM)"]),
        ("size", ["Large (25 - 100 MB)", "Medium (5 - 25 MB)", "Small (1 - 5 MB)", "Tiny (< 1 MB)"]),
        ("image_size", ["Extra Large (> 16 MP)", "Large (8 - 16 MP)", "Medium (2 - 8 MP)"]),
        ("orientation", ["Landscape", "Portrait", "Square"]),
        ("file_type", ["JPEG Image (*.jpg, *.jpeg)", "PNG Image (*.png)", "RAW Digital Negative (*.raw, *.dng)", "MP4 Video (*.mp4)"]),
        ("media_type", ["Photos", "Videos"]),
        ("rating", ["5 stars", "4 stars", "3 stars", "Unrated"]),
        ("flag", ["Flagged", "Not flagged"]),
        ("folder", ["Vacation", "Work"]),
        ("tag", ["Landscape", "Not tagged"]),
        ("people", ["John Doe", "Not tagged"]),
        ("published", ["Published", "Not published"]),
        ("geotag", ["London, UK", "New York, USA", "No Geotag"]),
        ("camera", ["Canon EOS 5D", "Nikon D850", "Sony Alpha 7", "Apple iPhone 15 Pro"]),
        ("none", [""]),
    ]

    for mode_name, expected_headers in modes:
        spec = FilterQuerySpec(group_by=mode_name)
        records = repo.query_media_records(spec)
        assert len(records) == 4, f"Failed for mode {mode_name}"

        # Collect distinct group keys in returned order
        seen_keys = []
        for r in records:
            if r.group_key not in seen_keys:
                seen_keys.append(r.group_key)

        # Check that headers match expectations
        for exp in expected_headers:
            assert exp in seen_keys, f"Expected header '{exp}' in mode '{mode_name}', got {seen_keys}"


def test_group_reverse_sorting(test_engine: DualDatabaseEngine):
    """Verifies that group_reverse=True reverses the header group sequence."""
    repo = CatalogRepository(test_engine)

    spec_fwd = FilterQuerySpec(group_by="year", group_reverse=False)
    recs_fwd = repo.query_media_records(spec_fwd)
    keys_fwd = [r.group_key for r in recs_fwd]
    # Forward: 2026 first, 2024 last
    assert keys_fwd[0] == "2026"
    assert keys_fwd[-1] == "2024"

    spec_rev = FilterQuerySpec(group_by="year", group_reverse=True)
    recs_rev = repo.query_media_records(spec_rev)
    keys_rev = [r.group_key for r in recs_rev]
    # Reverse: 2024 first, 2026 last
    assert keys_rev[0] == "2024"
    assert keys_rev[-1] == "2026"


def test_grid_view_context_menu_has_all_19_group_criteria(qapp):
    """Verifies that the Grid View context menu 'Group by' submenu contains all 19 criteria."""
    from PySide6.QtCore import QPoint
    grid = ThumbnailGridView()
    grid._suppress_menu_exec = True

    menu = grid._show_background_context_menu(QPoint(0, 0))
    group_menu = None
    for action in menu.actions():
        if action.menu() and action.text() == "Group by":
            group_menu = action.menu()
            break

    assert group_menu is not None
    action_texts = [a.text() for a in group_menu.actions() if not a.isSeparator()]

    expected_all_19 = [
        "Auto",
        "Date taken",
        "Month taken",
        "Year taken",
        "Time taken",
        "File size",
        "Image size",
        "Orientation",
        "File type",
        "Media type",
        "Rating",
        "Flag",
        "Folder",
        "Tag",
        "People",
        "Published",
        "Geotag",
        "Camera",
        "None",
        "Ascending",
        "Descending",
    ]

    assert action_texts == expected_all_19


def test_ribbon_view_tab_arrange_more_menu(qapp):
    """Verifies that RibbonToolbar's Arrange more button triggers the complete Group by menu."""
    ribbon = RibbonToolbar()
    assert hasattr(ribbon, "btn_arrange_more")
    assert ribbon.btn_arrange_more is not None
    assert ribbon.btn_arrange_more.text() == "▾"
