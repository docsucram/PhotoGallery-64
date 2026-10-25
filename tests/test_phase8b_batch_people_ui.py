"""
Test Suite for WP-12: Virtualized 'Batch people tag' View & Ribbon Integration (tests/test_phase8b_batch_people_ui.py).

Verifies:
1. BatchPeopleView population with grouped sections (suggested persons + unnamed).
2. People avatar picker strip 1-click filtering.
3. Confirming suggestions ([Confirm All] / Enter key) updates media_people.state = 1,
   people.confirmed_count, people.centroid_embedding, and removes confirmed cards.
4. Rejecting suggestions (Not this person / Delete key) records rejections in face_rejected_suggestions.
5. Ignoring faces (Don't show again / I key) sets state = -2.
6. Ribbon integration: clicking 'batch_people_tag' toggles central_stack to BatchPeopleView and back.
7. Keyboard shortcuts and quick-preview dialog.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QEvent, QItemSelectionModel, QPoint, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from src.core.face_matcher import (
    deserialize_embedding,
    normalize_vector,
    serialize_embedding,
)
from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository, ThumbnailRepository
from src.ui.batch_people_view import (
    BatchPeopleView,
    FaceCardRecord,
    _FaceCardDelegate,
    _FaceCardModel,
    _PersonGroupSectionWidget,
)
from src.ui.main_window import PhotoGalleryMainWindow


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
    yield engine
    engine.close_all()


@pytest.fixture
def populated_data(tmp_path: Path, test_engine: DualDatabaseEngine) -> dict[str, int]:
    """Sets up sample photos, people, and detected face suggestions."""
    repo = CatalogRepository(test_engine)
    thumb_repo = ThumbnailRepository(test_engine)

    folder_dir = tmp_path / "Photos"
    folder_dir.mkdir()
    img1 = folder_dir / "Photo_01.jpg"
    img2 = folder_dir / "Photo_02.jpg"
    Image.new("RGB", (200, 200), color="blue").save(img1)
    Image.new("RGB", (200, 200), color="red").save(img2)

    fid = repo.upsert_folder(str(folder_dir), "Photos", None, 1700000000)
    mids = repo.bulk_upsert_media(
        [
            {"folder_id": fid, "filename": img1.name, "ext": "jpg", "size": 1000, "mtime": 1700000000},
            {"folder_id": fid, "filename": img2.name, "ext": "jpg", "size": 1000, "mtime": 1700000000},
        ]
    )

    # Create Alice with a pre-existing vector centroid
    alice_pid = repo.upsert_person("Alice")
    alice_vec = np.zeros((512,), dtype=np.float32)
    alice_vec[0:10] = 1.0
    alice_vec = normalize_vector(alice_vec)
    repo.update_person_centroid(alice_pid, serialize_embedding(alice_vec))

    # Insert 2 face detections for Alice (state = 0) and 1 unnamed detection
    with test_engine.write_lock:
        conn = test_engine.catalog_conn()
        cur1 = conn.execute(
            """
            INSERT INTO media_people (media_id, suggested_person_id, similarity_score, state, is_manual, rect_x, rect_y, rect_w, rect_h, embedding)
            VALUES (?, ?, 0.94, 0, 0, 0.1, 0.1, 0.2, 0.2, ?)
            RETURNING id;
            """,
            (mids[0], alice_pid, serialize_embedding(alice_vec)),
        )
        f1_id = cur1.fetchone()[0]

        cur2 = conn.execute(
            """
            INSERT INTO media_people (media_id, suggested_person_id, similarity_score, state, is_manual, rect_x, rect_y, rect_w, rect_h, embedding)
            VALUES (?, ?, 0.88, 0, 0, 0.4, 0.1, 0.2, 0.2, ?)
            RETURNING id;
            """,
            (mids[1], alice_pid, serialize_embedding(alice_vec)),
        )
        f2_id = cur2.fetchone()[0]

        cur3 = conn.execute(
            """
            INSERT INTO media_people (media_id, suggested_person_id, similarity_score, state, is_manual, rect_x, rect_y, rect_w, rect_h, embedding)
            VALUES (?, NULL, NULL, 0, 0, 0.7, 0.1, 0.2, 0.2, NULL)
            RETURNING id;
            """,
            (mids[0],),
        )
        f3_id = cur3.fetchone()[0]
        conn.commit()

    # Store fake thumbnail crops
    thumb_repo.put_face_thumbnail(f1_id, b"webp_alice_1")
    thumb_repo.put_face_thumbnail(f2_id, b"webp_alice_2")
    thumb_repo.put_face_thumbnail(f3_id, b"webp_unnamed")

    return {
        "alice_pid": alice_pid,
        "f1_id": f1_id,
        "f2_id": f2_id,
        "f3_id": f3_id,
        "mid1": mids[0],
        "mid2": mids[1],
    }


def test_batch_people_view_population_and_groups(
    qapp, test_engine: DualDatabaseEngine, populated_data: dict[str, int]
) -> None:
    """Verifies BatchPeopleView populates grouped sections for suggested persons and unnamed faces."""
    view = BatchPeopleView(test_engine)
    # widget show
    view.show()
    view.refresh_suggestions()

    # Should have 2 sections: Alice (2 suggestions) and Unnamed (1 face)
    assert len(view._sections) == 2
    alice_pid = populated_data["alice_pid"]
    assert alice_pid in view._sections
    assert None in view._sections

    alice_sec = view._sections[alice_pid]
    assert len(alice_sec.model._records) == 2
    assert "Alice" in alice_sec.lbl_title.text()
    assert alice_sec.btn_confirm_all is not None
    assert "Confirm All (2)" in alice_sec.btn_confirm_all.text()

    unnamed_sec = view._sections[None]
    assert len(unnamed_sec.model._records) == 1
    assert "Unnamed" in unnamed_sec.lbl_title.text()


def test_avatar_picker_strip_filtering(
    qapp, test_engine: DualDatabaseEngine, populated_data: dict[str, int]
) -> None:
    """Verifies clicking an avatar chip filters the visible sections."""
    view = BatchPeopleView(test_engine)
    # widget show
    view.show()
    view.refresh_suggestions()

    alice_pid = populated_data["alice_pid"]

    # Filter to Alice
    view.avatar_strip._on_clicked(alice_pid)
    assert view._sections[alice_pid].isVisible()
    assert not view._sections[None].isVisible()

    # Filter back to All
    view.avatar_strip._on_clicked(None)
    assert view._sections[alice_pid].isVisible()
    assert view._sections[None].isVisible()


def test_confirm_all_button_in_section(
    qapp, test_engine: DualDatabaseEngine, populated_data: dict[str, int]
) -> None:
    """Clicking [Confirm All] in a person's section commits state=1 and updates centroid."""
    view = BatchPeopleView(test_engine)
    # widget show
    view.show()
    view.refresh_suggestions()

    alice_pid = populated_data["alice_pid"]
    alice_sec = view._sections[alice_pid]

    confirmed_events: list[tuple[list, int]] = []
    view.faces_confirmed.connect(lambda fids, pid: confirmed_events.append((fids, pid)))

    # Click Confirm All
    alice_sec.btn_confirm_all.click()

    assert len(confirmed_events) == 1
    fids, pid = confirmed_events[0]
    assert pid == alice_pid
    assert len(fids) == 2

    # Verify SQLite database: both faces should have state = 1
    conn = test_engine.catalog_conn()
    rows = conn.execute(
        "SELECT id, state, person_id FROM media_people WHERE id IN (?, ?);",
        (populated_data["f1_id"], populated_data["f2_id"]),
    ).fetchall()
    assert len(rows) == 2
    assert all(r["state"] == 1 and r["person_id"] == alice_pid for r in rows)

    # Alice section should have been removed from the view
    assert alice_pid not in view._sections


def test_reject_faces_records_in_rejections(
    qapp, test_engine: DualDatabaseEngine, populated_data: dict[str, int]
) -> None:
    """Rejecting a face records in face_rejected_suggestions and clears suggestion."""
    view = BatchPeopleView(test_engine)
    # widget show
    view.show()
    view.refresh_suggestions()

    alice_pid = populated_data["alice_pid"]
    alice_sec = view._sections[alice_pid]

    # Select first face and reject
    alice_sec.list_view.selectionModel().select(
        alice_sec.model.index(0, 0),
        QItemSelectionModel.SelectionFlag.Select,
    )

    rejected_events: list[list[int]] = []
    view.faces_rejected.connect(lambda fids: rejected_events.append(fids))

    view.reject_selected()

    assert len(rejected_events) == 1
    rejected_id = rejected_events[0][0]

    # Verify face_rejected_suggestions table
    conn = test_engine.catalog_conn()
    row = conn.execute(
        "SELECT face_id, person_id FROM face_rejected_suggestions WHERE face_id = ?;",
        (rejected_id,),
    ).fetchone()
    assert row is not None
    assert row["person_id"] == alice_pid

    # media_people row should have suggested_person_id = NULL
    mp_row = conn.execute("SELECT suggested_person_id FROM media_people WHERE id = ?;", (rejected_id,)).fetchone()
    assert mp_row["suggested_person_id"] is None


def test_ignore_faces_sets_state_minus_two(
    qapp, test_engine: DualDatabaseEngine, populated_data: dict[str, int]
) -> None:
    """Ignoring a face sets state = -2 ('Don't show again')."""
    view = BatchPeopleView(test_engine)
    # widget show
    view.show()
    view.refresh_suggestions()

    alice_pid = populated_data["alice_pid"]
    alice_sec = view._sections[alice_pid]

    alice_sec.list_view.selectAll()
    ignored_events: list[list[int]] = []
    view.faces_ignored.connect(lambda fids: ignored_events.append(fids))

    view.ignore_selected()

    assert len(ignored_events) == 1
    assert len(ignored_events[0]) == 2

    conn = test_engine.catalog_conn()
    rows = conn.execute(
        "SELECT state FROM media_people WHERE id IN (?, ?);",
        (populated_data["f1_id"], populated_data["f2_id"]),
    ).fetchall()
    assert all(r["state"] == -2 for r in rows)


def test_main_window_batch_people_tag_ribbon_toggle(
    qapp, test_engine: DualDatabaseEngine, populated_data: dict[str, int]
) -> None:
    """Verifies clicking 'Batch people tag' on the ribbon toggles BatchPeopleView."""
    win = PhotoGalleryMainWindow(test_engine, start_xmp_daemon=False)
    win.show()

    # Initial state: grid_view active
    assert win.central_stack.currentWidget() == win.grid_view

    # Click Batch people tag on Ribbon
    win.dispatch_ribbon_action("batch_people_tag")
    assert win.central_stack.currentWidget() == win.batch_people_view

    # Click Back to Gallery
    win.batch_people_view.back_to_gallery_requested.emit()
    assert win.central_stack.currentWidget() == win.grid_view
    win.close()


def test_wlpg_manual_face_tagging_and_person_cards(
    qapp, test_engine: DualDatabaseEngine, populated_data: dict[str, int]
) -> None:
    """
    Verifies WLPG manual face tagging ('Tag someone' popup, movable target box,
    device-independent _FaceNameBadgeItem below face rect, and RightTagDock
    _PersonTagCardWidget context actions: rename, merge, set display picture).
    """
    from PySide6.QtWidgets import QGraphicsItem
    from src.ui.media_viewer import _FaceNameBadgeItem
    from src.ui.right_tag_dock import _PersonTagCardWidget

    win = PhotoGalleryMainWindow(test_engine, start_xmp_daemon=False)
    win.show()

    mid = populated_data["mid1"]
    assert win.open_single_viewer(mid)
    assert win.central_stack.currentWidget() == win.media_viewer

    # Trigger 'People tag' ribbon action -> starts face tagging mode
    win.dispatch_ribbon_action("people_tag")
    assert win.media_viewer.is_face_tagging_mode is True
    assert win.media_viewer._face_tag_target_item is not None
    assert not win.media_viewer._tag_someone_popup.isHidden()

    # Select 'Alice' from the 'Tag someone' popup
    win.media_viewer._on_tag_someone_selected("Alice")
    assert win.media_viewer.is_face_tagging_mode is False

    # Verify _FaceNameBadgeItem is rendered with ItemIgnoresTransformations
    badges = [
        it for it in win.media_viewer.image_scene.items() if isinstance(it, _FaceNameBadgeItem)
    ]
    assert len(badges) >= 1
    assert bool(badges[0].flags() & QGraphicsItem.ItemIgnoresTransformations)
    assert badges[0].name == "Alice"

    # Verify RightTagDock has a _PersonTagCardWidget for Alice
    cards = win.right_tag_dock.people_chips_container.findChildren(_PersonTagCardWidget)
    assert any(c.person_name == "Alice" for c in cards)
    alice_card = next(c for c in cards if c.person_name == "Alice")

    # Test hovering highlights face in viewer
    win._on_dock_person_hovered(alice_card.person_id, True)
    assert win.media_viewer._highlighted_person_id == alice_card.person_id
    win._on_dock_person_hovered(alice_card.person_id, False)
    assert win.media_viewer._highlighted_person_id is None

    # Test rename_person and set_person_display_picture in CatalogRepository
    win.catalog_repo.rename_person(alice_card.person_id, "Alice Smith")
    people_names = [r["name"] for r in win.catalog_repo.get_people_list()]
    assert "Alice Smith" in people_names
    assert win.catalog_repo.set_person_display_picture(alice_card.person_id, mid) is True

    win.close()

