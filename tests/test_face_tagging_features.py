"""
Test suite for enhanced face tagging, batch people tagging, and context menus:
1. TagSomeoneDialog autocomplete, filtering, selection, and new name entry.
2. Unconfirmed face suggestions in RightTagDock with 'Name?' and WLPG context menu options.
3. BatchPeopleView toggle to view previously tagged faces.
4. "Rescan for faces" in context menus (ThumbnailGridView and SingleMediaViewer).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PySide6.QtCore import QItemSelectionModel, QPoint, Qt
from PySide6.QtWidgets import QApplication, QMenu

from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository, ThumbnailRepository
from src.ui.batch_people_view import BatchPeopleView, FaceCardRecord, _PersonGroupSectionWidget
from src.ui.grid_view import ThumbnailGridView
from src.ui.media_viewer import SingleMediaViewer
from src.ui.right_tag_dock import TagAndCaptionDock, _PersonTagCardWidget
from src.ui.tag_someone_dialog import TagSomeoneDialog


@pytest.fixture
def qapp() -> QApplication:
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
        schema_sql_path=Path("db/schema.sql"),
    )
    engine.initialize_schemas()
    return engine


def test_tag_someone_dialog_filtering_and_selection(qapp: QApplication) -> None:
    people = [
        {"id": 1, "name": "Alice Smith"},
        {"id": 2, "name": "Bob Jones"},
        {"id": 3, "name": "Charlie Brown"},
    ]
    dlg = TagSomeoneDialog(people, title="Tag someone")
    assert dlg.list_widget.count() == 3

    # Filter to "Bob"
    dlg.txt_search.setText("Bob")
    assert dlg.list_widget.count() == 1
    assert dlg.list_widget.item(0).text() == "Bob Jones"

    # Select the item
    item = dlg.list_widget.item(0)
    dlg._on_item_clicked(item)
    assert dlg._selected_name == "Bob Jones"
    assert dlg._selected_id == 2


def test_tag_someone_dialog_new_name_entry(qapp: QApplication) -> None:
    people = [{"id": 1, "name": "Alice"}]
    dlg = TagSomeoneDialog(people, title="Tag someone")
    dlg.txt_search.setText("New Person")
    dlg._on_return_pressed()
    assert dlg._selected_name == "New Person"
    assert dlg._selected_id == -1


def test_unconfirmed_card_in_right_tag_dock(qapp: QApplication) -> None:
    card = _PersonTagCardWidget(
        person_id=42,
        person_name="Nansumay Robinson",
        is_unconfirmed=True,
        face_id=101,
    )
    # Must show 'Name?'
    assert card.lbl_name.text() == "Nansumay Robinson?"

    # Context menu must have Confirm and Don't show suggestion
    card._suppress_menu_exec = True
    menu = card._exec_menu(QPoint(0, 0))
    action_texts = [a.text() for a in menu.actions()]
    assert "Confirm" in action_texts
    assert "Tag as someone else..." in action_texts
    assert "Don't show suggestion" in action_texts

    # Test Confirm trigger
    confirmed = []
    card.confirm_clicked.connect(lambda pid, fid: confirmed.append((pid, fid)))
    for a in menu.actions():
        if a.text() == "Confirm":
            a.trigger()
            break
    assert confirmed == [(42, 101)]

    # Test Reject trigger
    rejected = []
    card.reject_suggestion_clicked.connect(lambda pid, fid: rejected.append((pid, fid)))
    for a in menu.actions():
        if a.text() == "Don't show suggestion":
            a.trigger()
            break
    assert rejected == [(42, 101)]


def test_confirmed_card_in_right_tag_dock(qapp: QApplication) -> None:
    card = _PersonTagCardWidget(
        person_id=10,
        person_name="Confirmed Person",
        is_unconfirmed=False,
    )
    assert card.lbl_name.text() == "Confirmed Person"
    card._suppress_menu_exec = True
    menu = card._exec_menu(QPoint(0, 0))
    action_texts = [a.text() for a in menu.actions()]
    assert "Confirm" not in action_texts
    assert "Remove tag" in action_texts
    assert "Tag as someone else..." in action_texts


def test_batch_people_view_toggle_and_context_menu(qapp: QApplication, test_engine: DualDatabaseEngine) -> None:
    repo = CatalogRepository(test_engine)
    bpv = BatchPeopleView(test_engine)
    assert hasattr(bpv, "btn_toggle_tagged")
    assert bpv.btn_toggle_tagged.text() == "Show tagged faces"

    # Toggle it
    bpv.btn_toggle_tagged.click()
    assert bpv._show_tagged_faces is True
    assert bpv.btn_toggle_tagged.text() == "Hide tagged faces"

    # Context menu on section widget
    from legacy_reference.imgBrowserv13 import LruCache
    rec = FaceCardRecord(
        face_id=99,
        media_id=123,
        suggested_person_id=5,
        suggested_name="Test Person",
        confidence=0.9,
        rect_x=0.1,
        rect_y=0.1,
        rect_w=0.2,
        rect_h=0.2,
        filename="test.jpg",
        folder_path="C:/photos",
    )
    sec = _PersonGroupSectionWidget(
        person_id=5,
        person_name="Test Person",
        confirmed_count=10,
        records=[rec],
        thumb_repo=ThumbnailRepository(test_engine),
        pixmap_cache=LruCache(10),
        parent=bpv,
    )
    sec._suppress_menu_exec = True
    sec.list_view.selectionModel().select(sec.model.index(0), QItemSelectionModel.Select)
    menu = sec._show_context_menu(QPoint(0, 0))
    action_texts = [a.text() for a in menu.actions()]
    assert "View photo in single image view" in action_texts
    assert "Tag as..." in action_texts


def test_rescan_faces_in_grid_and_viewer_context_menus(qapp: QApplication, test_engine: DualDatabaseEngine) -> None:
    grid = ThumbnailGridView()
    grid._suppress_menu_exec = True
    menu = grid._show_card_context_menu(QPoint(0, 0), media_id=1, is_video=False)
    action_texts = [a.text() for a in menu.actions()]
    assert "Rescan for faces" in action_texts

    viewer = SingleMediaViewer()
    viewer.current_media_id = 1
    # Check that viewer has _show_viewer_context_menu
    assert hasattr(viewer, "_show_viewer_context_menu")
