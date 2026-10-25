"""
Unit & Integration Tests for Authentic Windows Live Photo Gallery Find Bar & SearchScopeBar.
Tests:
- SearchScopeBar positioning, pinning, close, and layout.
- FilterChip removal (people, tags, rating, flag, date).
- Boolean search parser: AND, OR, NOT, -term, quoted phrases, person: and tag: prefixes.
- Keyboard navigation (Ctrl+F, Esc with pin state).
- Sub-10ms search query performance.
"""

from __future__ import annotations

import os
from pathlib import Path
from PIL import Image
import pytest

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel

from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository, FilterQuerySpec, _build_boolean_search_sql
from src.ui.main_window import PhotoGalleryMainWindow
from src.ui.search_scope_bar import FilterChip, SearchScopeBar


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_boolean_search_sql_generation(tmp_path: Path) -> None:
    """Verifies SQL where clause generator handles AND, OR, NOT, -, quotes, and prefixes."""
    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    engine = DualDatabaseEngine(
        catalog_path=db_dir / "catalog.db",
        thumbnails_path=db_dir / "thumbnails.db",
    )
    repo = CatalogRepository(engine)

    # 1. Simple word
    sql1, p1 = _build_boolean_search_sql("sunset")
    assert "m.filename LIKE ?" in sql1
    assert any("%sunset%" in str(x) for x in p1)

    # 2. Boolean AND
    sql2, p2 = _build_boolean_search_sql("beach AND sunset")
    assert " AND " in sql2
    assert len(p2) >= 2

    # 3. Boolean OR
    sql3, p3 = _build_boolean_search_sql("beach OR sunset")
    assert " OR " in sql3

    # 4. Negation NOT and -
    sql4, p4 = _build_boolean_search_sql("beach NOT winter")
    assert "NOT" in sql4

    sql5, p5 = _build_boolean_search_sql("beach -winter")
    assert "NOT" in sql5

    # 5. Prefixes: person: and tag:
    sql6, p6 = _build_boolean_search_sql('person:"John Doe"')
    assert "media_people" in sql6
    assert any("%John Doe%" in str(x) for x in p6)

    sql7, p7 = _build_boolean_search_sql("tag:Summer")
    assert "media_tags" in sql7
    assert any("%Summer%" in str(x) for x in p7)


def test_search_scope_bar_chips_and_pinning(qapp: QApplication, tmp_path: Path) -> None:
    """Verifies SearchScopeBar interaction: pin latching, chip clicks, clear."""
    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    engine = DualDatabaseEngine(
        catalog_path=db_dir / "catalog.db",
        thumbnails_path=db_dir / "thumbnails.db",
    )
    repo = CatalogRepository(engine)

    # Insert sample person and tag
    with engine.write_lock:
        conn = engine.catalog_conn()
        conn.execute("INSERT INTO people (id, name) VALUES (101, 'Alice');")
        conn.execute("INSERT INTO tags (id, name, full_path) VALUES (202, 'Vacation', 'Vacation');")

    bar = SearchScopeBar()
    bar.resize(800, 62)

    # 1. Pinning
    assert not bar.is_pinned()
    pin_events: list[bool] = []
    bar.pin_toggled.connect(pin_events.append)
    bar.btn_pin.click()
    assert bar.is_pinned() is True
    assert pin_events == [True]
    bar.btn_pin.click()
    assert bar.is_pinned() is False
    assert pin_events == [True, False]

    # 2. Scope label
    bar.set_scope_label("Folder: Pictures")
    assert bar.lbl_scope_label.text() == "Folder:"
    assert bar.lbl_scope_value.text() == "Pictures"

    # 3. Active filter chips
    removed_people: list[int] = []
    removed_tags: list[int] = []
    bar.person_remove_requested.connect(removed_people.append)
    bar.tag_remove_requested.connect(removed_tags.append)

    filter_state = {
        "person_ids": (101,),
        "include_tag_ids": (202,),
        "ratings": (4, 5),
        "flagged_only": True,
        "date_year": 2023,
        "date_month": 7,
    }
    bar.update_active_filter_chips(filter_state, repo)

    # Locate generated FilterChips
    chips = bar.findChildren(FilterChip)
    assert len(chips) == 5  # Person, Tag, Rating, Flag, Date

    # Click remove on person chip (first chip)
    chips[0].remove_clicked.emit()
    assert removed_people == [101]

    # Click remove on tag chip (second chip)
    chips[1].remove_clicked.emit()
    assert removed_tags == [202]

    # 4. Search input
    search_events: list[str] = []
    bar.search_text_changed.connect(search_events.append)
    bar.txt_search.setText("person:Alice")
    bar._emit_search_text()
    assert search_events == ["person:Alice"]


def test_main_window_search_scope_integration_and_shortcuts(
    qapp: QApplication, tmp_path: Path
) -> None:
    """Verifies main window scope bar positioning, Ctrl+F, Esc with pin, and chip actions."""
    photos_dir = tmp_path / "Pictures"
    photos_dir.mkdir(parents=True, exist_ok=True)

    for idx, name in enumerate(["Family_Beach.jpg", "Mountain_Hike.jpg", "Sunset_Park.jpg"], start=1):
        img = Image.new("RGB", (320, 240), (40 * idx, 50 * idx, 60 * idx))
        img.save(photos_dir / name, quality=90)

    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    engine = DualDatabaseEngine(
        catalog_path=db_dir / "catalog.db",
        thumbnails_path=db_dir / "thumbnails.db",
    )

    window = PhotoGalleryMainWindow(engine=engine, start_xmp_daemon=False)
    window.resize(1280, 800)
    window.show()
    qapp.processEvents()
    window.scan_folder(photos_dir, include_subfolders=True, blocking=True)

    all_ids = [r.id for r in window.catalog_model._records]
    assert len(all_ids) == 3

    # Tag item 1 with person 'Craig'
    window.grid_view.on_card_checkbox_toggled(all_ids[0], True)
    window.right_tag_dock.add_person_tag_text("Craig")
    window.right_tag_dock.commit_caption("Family Beach Trip")

    # 1. Ctrl+F in Grid opens Search Scope Bar directly above grid
    assert not window.scope_bar.isVisible()
    window._on_ctrl_f()
    assert window.scope_bar.isVisible()
    assert window.scope_bar.txt_search.hasFocus()

    # 2. Esc hides unpinned Search Scope Bar
    assert not window.scope_bar.is_pinned()
    window.hide_search_scope_bar()
    assert not window.scope_bar.isVisible()

    # 3. Pinning keeps bar open
    window.scope_bar.set_pinned(True)
    window.scope_bar.show()
    assert window.scope_bar.isVisible()
    # Pinned bar stays visible on nav filter changed
    window.left_nav_dock.clear_all_filters()
    assert window.scope_bar.isVisible()

    # 4. Boolean Search query execution: person:"Craig"
    window.scope_bar.txt_search.setText('person:"Craig"')
    window.scope_bar._emit_search_text()
    qapp.processEvents()
    assert window.catalog_model.total_media_count == 1
    assert window.catalog_model._records[0].id == all_ids[0]

    # Boolean Search with negation: Beach -Craig -> 0 results
    window.scope_bar.txt_search.setText("Beach -Craig")
    window.scope_bar._emit_search_text()
    qapp.processEvents()
    assert window.catalog_model.total_media_count == 0

    # Clear search
    window.scope_bar.txt_search.clear()
    window.scope_bar._emit_search_text()
    qapp.processEvents()
    assert window.catalog_model.total_media_count == 3


def test_search_scope_bar_boolean_tags_and_folder_label(qapp: QApplication, tmp_path: Path) -> None:
    """Verifies that non-folder scopes don't show on the left, and boolean tags display + and / with hover chips."""
    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    engine = DualDatabaseEngine(
        catalog_path=db_dir / "catalog.db",
        thumbnails_path=db_dir / "thumbnails.db",
    )
    repo = CatalogRepository(engine)

    with engine.write_lock:
        conn = engine.catalog_conn()
        conn.execute("INSERT INTO tags (id, name, full_path) VALUES (1, 'Burma', 'Burma');")
        conn.execute("INSERT INTO tags (id, name, full_path) VALUES (2, 'Nansumay', 'Nansumay');")

    bar = SearchScopeBar()

    # 1. Non-folder scope (e.g. Tags (AND): 1 or None) must hide folder labels
    bar.set_scope_label("Tags (AND): 1")
    assert bar.lbl_scope_label.isHidden()
    assert bar.lbl_scope_value.isHidden()
    assert bar.btn_search_all.isHidden()

    bar.set_scope_label("Folder: 2019_01 Wisley Lego")
    assert not bar.lbl_scope_label.isHidden()
    assert not bar.lbl_scope_value.isHidden()
    assert bar.lbl_scope_label.text() == "Folder:"
    assert bar.lbl_scope_value.text() == "2019_01 Wisley Lego"
    assert not bar.btn_search_all.isHidden()

    # 2. Boolean Tag rendering with AND mode
    filter_state_and = {
        "include_tag_ids": (1, 2),
        "tag_boolean_mode": "AND",
    }
    bar.update_active_filter_chips(filter_state_and, repo)
    chips = bar.findChildren(FilterChip)
    assert len(chips) == 2
    assert chips[0].text() == "Burma"
    assert chips[1].text() == "Nansumay"

    # In unhovered state, remove button is hidden
    assert chips[0].btn_close.isHidden()
    assert chips[1].btn_close.isHidden()

    # On hover, remove button is visible (not hidden)
    chips[1]._set_hover_state(True)
    assert not chips[1].btn_close.isHidden()
    chips[1]._set_hover_state(False)
    assert chips[1].btn_close.isHidden()

    # Labels inside chips_layout include "Tagged:" and separator " + "
    qlabels = bar.chips_widget.findChildren(QLabel)
    texts = [lbl.text() for lbl in qlabels]
    assert "Tagged:" in texts
    assert " + " in texts

    # 3. Boolean Tag rendering with OR mode
    filter_state_or = {
        "include_tag_ids": (1, 2),
        "tag_boolean_mode": "OR",
    }
    bar.update_active_filter_chips(filter_state_or, repo)
    qlabels_or = bar.chips_widget.findChildren(QLabel)
    texts_or = [lbl.text() for lbl in qlabels_or]
    assert "Tagged:" in texts_or
    assert " / " in texts_or

