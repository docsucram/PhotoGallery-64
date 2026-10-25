"""
Verification suite for Immich / Google Photos inspired People Navigation:
1. People Directory View (Section 1 Named People & Section 2 Unnamed Clusters)
2. Inline cluster naming ('Add a name...')
3. Real-time search/filtering in People Directory
4. Person Header Bar with avatar, editable name, photo count, and Google Photos-style suggestions banner
5. View mode toggle: Full Photos vs Face Crops
6. LeftNavigationDock 'People' root routing vs individual person routing
"""

import struct
from pathlib import Path
from PIL import Image
import numpy as np
import pytest
from PySide6.QtCore import Qt, QPoint
from PySide6.QtWidgets import QApplication

from src.core.face_matcher import serialize_embedding, normalize_vector
from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository, ThumbnailRepository
from src.ui.left_nav_dock import LeftNavigationDock
from src.ui.people_directory_view import PeopleDirectoryView, _NamedPersonCard, _UnnamedClusterCard
from src.ui.person_header_bar import PersonHeaderBar, make_circular_avatar


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def db_engine(tmp_path: Path):
    engine = DualDatabaseEngine(tmp_path / "catalog.db", tmp_path / "thumbs.db")
    yield engine
    engine.close_all()


def _create_test_media(repo: CatalogRepository, filename: str) -> int:
    fid = repo.upsert_folder("D:/Photos", "Photos", None, 1700000000)
    items = [
        {
            "folder_id": fid,
            "filename": filename,
            "ext": "jpg",
            "size": 1000,
            "mtime": 1700000000,
            "date_taken": 1700000000,
            "rating": 0,
            "flag": 0,
        }
    ]
    ids = repo.bulk_upsert_media(items)
    return ids[0]


def test_people_directory_data_repository(db_engine):
    """Verifies get_people_directory_data returns named people with counts & unnamed clusters."""
    repo = CatalogRepository(db_engine)
    conn = db_engine.catalog_conn()

    m1 = _create_test_media(repo, "p1.jpg")
    m2 = _create_test_media(repo, "p2.jpg")

    p_alice = repo.upsert_person("Alice")
    p_bob = repo.upsert_person("Bob")

    # Alice has 2 confirmed photos
    conn.execute(
        "INSERT INTO media_people (media_id, person_id, state, rect_x, rect_y, rect_w, rect_h) "
        "VALUES (?, ?, 1, 0.1, 0.1, 0.2, 0.2);",
        (m1, p_alice),
    )
    conn.execute(
        "INSERT INTO media_people (media_id, person_id, state, rect_x, rect_y, rect_w, rect_h) "
        "VALUES (?, ?, 1, 0.1, 0.1, 0.2, 0.2);",
        (m2, p_alice),
    )
    # Alice has 1 pending suggestion
    conn.execute(
        "INSERT INTO media_people (media_id, state, suggested_person_id, similarity_score, rect_x, rect_y, rect_w, rect_h) "
        "VALUES (?, 0, ?, 0.85, 0.4, 0.4, 0.2, 0.2);",
        (m2, p_alice),
    )

    # Bob has 1 confirmed photo, 0 suggestions
    conn.execute(
        "INSERT INTO media_people (media_id, person_id, state, rect_x, rect_y, rect_w, rect_h) "
        "VALUES (?, ?, 1, 0.5, 0.5, 0.2, 0.2);",
        (m1, p_bob),
    )
    conn.commit()

    data = repo.get_people_directory_data()
    named = data["named_people"]
    assert len(named) >= 2

    alice_data = next(p for p in named if p["id"] == p_alice)
    assert alice_data["name"] == "Alice"
    assert alice_data["photo_count"] == 2
    assert alice_data["suggestion_count"] == 1

    bob_data = next(p for p in named if p["id"] == p_bob)
    assert bob_data["photo_count"] == 1
    assert bob_data["suggestion_count"] == 0


def test_people_directory_view_population_and_search(qapp, db_engine):
    """Verifies PeopleDirectoryView populates cards and filters by name in real-time."""
    repo = CatalogRepository(db_engine)
    m1 = _create_test_media(repo, "alice.jpg")
    m2 = _create_test_media(repo, "carol.jpg")

    p1 = repo.upsert_person("Alice")
    p2 = repo.upsert_person("Carol")
    repo.assign_person(m1, p1, rect=(0.1, 0.1, 0.2, 0.2))
    repo.assign_person(m2, p2, rect=(0.2, 0.2, 0.2, 0.2))

    view = PeopleDirectoryView(db_engine)
    view.resize(800, 600)
    view.refresh_directory()

    assert len(view._all_named_people) == 2
    assert view.lbl_total_count.text() == "(2)"

    # Filter for "car"
    view.txt_search.setText("car")
    # Named grid should show 1 card (Carol)
    named_cards = [
        view.grid_named.itemAt(i).widget()
        for i in range(view.grid_named.count())
        if view.grid_named.itemAt(i).widget()
    ]
    assert len(named_cards) == 1
    assert isinstance(named_cards[0], _NamedPersonCard)
    assert named_cards[0].person_name == "Carol"

    # Clear filter
    view.txt_search.setText("")
    assert view.grid_named.count() == 2

    # Scrollbar policy & button aesthetics verification
    assert view.scroll_area.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert not view.btn_review_suggestions.icon().isNull()
    assert "👥" not in view.btn_review_suggestions.text()
    assert "↻" not in view.btn_refresh.text()


def test_inline_naming_unnamed_cluster(qapp, db_engine):
    """Verifies that naming an unnamed cluster creates a person, confirms faces, and updates centroids."""
    repo = CatalogRepository(db_engine)
    conn = db_engine.catalog_conn()
    m1 = _create_test_media(repo, "mystery1.jpg")
    m2 = _create_test_media(repo, "mystery2.jpg")
    m3 = _create_test_media(repo, "mystery3.jpg")

    # Insert 3 unassigned faces with identical dummy embeddings (forming a cluster)
    vec = np.zeros((512,), dtype=np.float32)
    vec[5] = 1.0
    blob = serialize_embedding(vec)

    fids = []
    for mid in (m1, m2, m3):
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO media_people (media_id, state, embedding, rect_x, rect_y, rect_w, rect_h) "
            "VALUES (?, 0, ?, 0.1, 0.1, 0.2, 0.2);",
            (mid, blob),
        )
        fids.append(cur.lastrowid)
    conn.commit()

    view = PeopleDirectoryView(db_engine)
    view.refresh_directory()
    assert len(view._all_unnamed_clusters) >= 1

    # Simulate user typing "Daniel" in the inline text field and pressing Enter
    view._on_name_cluster("Daniel", fids)

    # Check Daniel was created and confirmed
    people = repo.get_people_list()
    daniel = next((p for p in people if p["name"] == "Daniel"), None)
    assert daniel is not None
    assert daniel["confirmed_count"] == 3

    # Check rows in media_people are now state = 1
    confirmed_rows = conn.execute(
        f"SELECT state, person_id FROM media_people WHERE id IN ({','.join('?' for _ in fids)});",
        fids,
    ).fetchall()
    assert all(r["state"] == 1 and r["person_id"] == daniel["id"] for r in confirmed_rows)


def test_person_header_bar_and_suggestions_banner(qapp, db_engine):
    """Verifies PersonHeaderBar displays person info, view toggle, and inline suggestions."""
    repo = CatalogRepository(db_engine)
    conn = db_engine.catalog_conn()

    m1 = _create_test_media(repo, "eva1.jpg")
    m2 = _create_test_media(repo, "eva2.jpg")
    p_eva = repo.upsert_person("Eva")

    # 1 confirmed face
    repo.assign_person(m1, p_eva, rect=(0.1, 0.1, 0.2, 0.2))

    # 1 pending suggestion for Eva
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO media_people (media_id, state, suggested_person_id, similarity_score, rect_x, rect_y, rect_w, rect_h) "
        "VALUES (?, 0, ?, 0.88, 0.3, 0.3, 0.2, 0.2);",
        (m2, p_eva),
    )
    sug_fid = cur.lastrowid
    conn.commit()

    header = PersonHeaderBar(db_engine)
    header.load_person(p_eva)

    assert header.current_person_name == "Eva"
    assert "1 photo" in header.lbl_photo_count.text()
    assert header.banner_frame.isVisible() is True
    assert "Eva has 1 unconfirmed face suggestion" in header.lbl_banner_title.text()

    # Test quick confirm on the inline card
    header._on_quick_confirm(sug_fid)
    row = conn.execute("SELECT state, person_id FROM media_people WHERE id = ?;", (sug_fid,)).fetchone()
    assert row["state"] == 1
    assert row["person_id"] == p_eva

    # Banner should hide once all suggestions are handled
    assert header.banner_frame.isVisible() is False


def test_view_mode_toggle_in_person_header_bar(qapp, db_engine):
    """Verifies view mode toggle emits 'photos' vs 'crops'."""
    header = PersonHeaderBar(db_engine)
    modes = []
    header.view_mode_changed.connect(modes.append)

    header.btn_face_crops.click()
    assert modes[-1] == "crops"

    header.btn_full_photos.click()
    assert modes[-1] == "photos"


def test_left_nav_dock_people_signals(qapp):
    """Verifies clicking 'People' root emits people_directory_requested."""
    dock = LeftNavigationDock()
    received = []
    dock.people_directory_requested.connect(lambda: received.append(True))

    # Trigger click on root_people item
    dock._on_item_clicked(dock.root_people, 0)
    assert len(received) == 1


def test_delete_unused_tags(db_engine):
    """Verifies delete_unused_tags deletes unassigned tags and preserves assigned ones."""
    repo = CatalogRepository(db_engine)
    m1 = _create_test_media(repo, "tagged_pic.jpg")

    tid_used = repo.create_tag("Vacation/Hawaii")
    tid_unused = repo.create_tag("Temporary/OldTag")
    tid_face = repo.create_tag("Face|Family|Robinson|Marcus")

    repo.assign_tag([m1], tid_used)

    deleted_count = repo.delete_unused_tags()
    assert deleted_count >= 2

    # Verify Hawaii still exists
    conn = db_engine.catalog_conn()
    row = conn.execute("SELECT id FROM tags WHERE full_path = 'Vacation/Hawaii';").fetchone()
    assert row is not None

    # Verify unused tags are deleted
    assert conn.execute("SELECT id FROM tags WHERE id = ?;", (tid_unused,)).fetchone() is None
    assert conn.execute("SELECT id FROM tags WHERE id = ?;", (tid_face,)).fetchone() is None


def test_unnamed_person_click_and_merge_dialog(qapp, db_engine):
    """Verifies clicking unnamed card emits signal and _MergeConfirmDialog accepts/rejects."""
    from src.ui.people_directory_view import _MergeConfirmDialog, _UnnamedPersonCard

    card = _UnnamedPersonCard(
        cluster_name="Person Group 1",
        face_ids=[101, 102],
        face_count=2,
        thumb_blob=None,
        existing_names=["Alice", "Bob"],
    )
    clicked_data = []
    card.clicked.connect(lambda fids, tb: clicked_data.append(fids))

    # Mouse click outside the line edit emits clicked
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtCore import QEvent, QPointF
    ev = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(5, 5), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    card.mousePressEvent(ev)
    assert clicked_data == [[101, 102]]

    # Test _MergeConfirmDialog
    dlg = _MergeConfirmDialog(None, "Alice", None)
    assert dlg.windowTitle() == "Merge People"
    dlg.close()


def test_navigation_history_people_directory_and_back(qapp, tmp_path):
    """Verifies browser-style back/forward navigation between People Directory and Person Timeline."""
    from src.ui.main_window import PhotoGalleryMainWindow
    from PIL import Image

    photos_dir = tmp_path / "Pictures"
    photos_dir.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (200, 200), (60, 80, 120))
    img.save(photos_dir / "Nav_Test.jpg", quality=90)

    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    engine = DualDatabaseEngine(
        catalog_path=db_dir / "catalog.db",
        thumbnails_path=db_dir / "thumbnails.db",
    )

    window = PhotoGalleryMainWindow(engine=engine, start_xmp_daemon=False)
    window.scan_folder(photos_dir, include_subfolders=True, blocking=True)

    # 1. Open People Directory
    window.open_people_directory_view()
    assert window.central_stack.currentWidget() == window.people_directory_view
    assert window.ribbon.btn_back.isEnabled()

    # 2. Add and navigate to person "Nansumay Robinson"
    p_id = window.catalog_repo.upsert_person("Nansumay Robinson")
    window._on_directory_person_selected(p_id)

    assert window.central_stack.currentWidget() == window.grid_view
    assert window.person_header_bar.current_person_name == "Nansumay Robinson"
    assert window.person_header_bar.txt_name.text() == "Nansumay Robinson"
    assert window.person_header_bar.txt_name.cursorPosition() == 0
    assert window.person_header_bar.txt_name.alignment() & Qt.AlignmentFlag.AlignLeft
    assert window.person_header_bar.txt_name.minimumWidth() >= 160

    # 3. Press Back -> Must return to People Directory!
    window.navigate_history_back()
    assert window.central_stack.currentWidget() == window.people_directory_view
    assert window.ribbon.btn_forward.isEnabled()

    # 4. Press Forward -> Returns to Nansumay Robinson timeline!
    window.navigate_history_forward()
    assert window.central_stack.currentWidget() == window.grid_view
    assert window.person_header_bar.current_person_name == "Nansumay Robinson"

    window.shutdown()


def test_person_header_bar_wlpg_presentation_and_confirm_all(qapp, db_engine):
    """Verifies square face crops, WLPG banner header, and [Confirm all] functionality."""
    repo = CatalogRepository(db_engine)
    conn = db_engine.catalog_conn()

    m1 = _create_test_media(repo, "person_base.jpg")
    m2 = _create_test_media(repo, "person_sug1.jpg")
    m3 = _create_test_media(repo, "person_sug2.jpg")
    pid = repo.upsert_person("Marcus Robinson")

    repo.assign_person(m1, pid, rect=(0.1, 0.1, 0.2, 0.2))

    # Add 2 suggestions
    conn.execute(
        "INSERT INTO media_people (media_id, state, suggested_person_id, similarity_score, rect_x, rect_y, rect_w, rect_h) "
        "VALUES (?, 0, ?, 0.85, 0.2, 0.2, 0.2, 0.2);",
        (m2, pid),
    )
    sug1 = conn.execute("SELECT last_insert_rowid();").fetchone()[0]

    conn.execute(
        "INSERT INTO media_people (media_id, state, suggested_person_id, similarity_score, rect_x, rect_y, rect_w, rect_h) "
        "VALUES (?, 0, ?, 0.89, 0.3, 0.3, 0.2, 0.2);",
        (m3, pid),
    )
    sug2 = conn.execute("SELECT last_insert_rowid();").fetchone()[0]
    conn.commit()

    header = PersonHeaderBar(db_engine)
    header.load_person(pid)

    assert header.banner_frame.isVisible() is True
    assert "Are these Marcus Robinson?" in header.lbl_banner_title.text()
    assert header.btn_confirm_all.isVisible() is True
    assert "2 suggestions" in header.lbl_banner_count.text()

    # Cards in strip should be _QuickFaceSuggestionCard with 66x66 square bounds
    from src.ui.person_header_bar import _QuickFaceSuggestionCard
    cards = [
        header.strip_layout.itemAt(i).widget()
        for i in range(header.strip_layout.count() - 1)
        if isinstance(header.strip_layout.itemAt(i).widget(), _QuickFaceSuggestionCard)
    ]
    assert len(cards) == 2
    assert cards[0].width() == 66 and cards[0].height() == 66

    # Test [✓ Confirm all] button
    header.btn_confirm_all.click()

    # Check both rows are confirmed (state = 1)
    rows = conn.execute(
        f"SELECT state, person_id FROM media_people WHERE id IN ({sug1}, {sug2});"
    ).fetchall()
    assert all(r["state"] == 1 and r["person_id"] == pid for r in rows)

    # Banner should now be hidden
    assert header.banner_frame.isVisible() is False
    assert header.sizeHint().height() == 64
    assert not hasattr(header, "btn_edit_name")
    assert header.txt_name.alignment() & Qt.AlignmentFlag.AlignLeft


def test_people_directory_card_dimensions_and_square_crops(qapp, db_engine):
    """Verifies ~20% larger portraits (130px), 162px card width, and square face crop encoding."""
    from src.ui.people_directory_view import _NamedPersonCard, _UnnamedPersonCard
    from src.core.face_engine import FaceEngine
    from PIL import Image
    import io

    # 1. Check card dimensions
    named_card = _NamedPersonCard(1, "Marcus Robinson", 10, 0, None)
    assert named_card.width() == 162
    assert named_card.height() == 198
    assert named_card.avatar_container.width() == 130
    assert named_card.avatar_container.height() == 130

    unnamed_card = _UnnamedPersonCard("Group 1", [1, 2], 2, None)
    assert unnamed_card.width() == 162
    assert unnamed_card.height() == 198
    assert unnamed_card.lbl_avatar.width() == 130
    assert unnamed_card.txt_name.width() == 146

    # 2. Check FaceEngine square crop encoding on tall rectangle (no stretching!)
    dummy_img = np.zeros((400, 300, 3), dtype=np.uint8)
    dummy_img[50:250, 80:200] = [180, 150, 120]  # Tall 200x120 region
    # rect: x=0.267, y=0.125, w=0.4, h=0.5
    crop_webp = FaceEngine.crop_and_encode_webp(
        dummy_img,
        rect_x=80 / 300.0,
        rect_y=50 / 400.0,
        rect_w=120 / 300.0,
        rect_h=200 / 400.0,
        target_size=(128, 128),
    )
    assert len(crop_webp) > 0
    pil_crop = Image.open(io.BytesIO(crop_webp))
    assert pil_crop.size == (128, 128)


def test_dock_person_confirm_and_reject_suggestion(db_engine):
    """Verifies confirming and rejecting suggestions from Tag & Caption dock and repo alias."""
    cat_repo = CatalogRepository(db_engine)
    m_id = _create_test_media(cat_repo, "dock_test.jpg")
    pid = cat_repo.upsert_person("Marcus Robinson")

    # Insert an unconfirmed suggestion (state = 0, suggested_person_id = pid)
    with db_engine.write_lock:
        conn = db_engine.catalog_conn()
        cur = conn.execute(
            """
            INSERT INTO media_people (media_id, state, suggested_person_id, similarity_score, rect_x, rect_y, rect_w, rect_h)
            VALUES (?, 0, ?, 0.85, 0.1, 0.1, 0.3, 0.3);
            """,
            (m_id, pid),
        )
        face_id = cur.lastrowid
        conn.commit()

    # 1. Test confirmation via alias confirm_face_suggestion
    cat_repo.confirm_face_suggestion(face_id, pid)

    row = db_engine.catalog_conn().execute(
        "SELECT state, person_id, suggested_person_id FROM media_people WHERE id = ?;",
        (face_id,),
    ).fetchone()
    assert row["state"] == 1
    assert row["person_id"] == pid
    assert row["suggested_person_id"] is None

    p_row = db_engine.catalog_conn().execute(
        "SELECT confirmed_count FROM people WHERE id = ?;", (pid,)
    ).fetchone()
    assert p_row["confirmed_count"] == 1

    # 2. Test rejection
    with db_engine.write_lock:
        conn = db_engine.catalog_conn()
        cur = conn.execute(
            """
            INSERT INTO media_people (media_id, state, suggested_person_id, similarity_score, rect_x, rect_y, rect_w, rect_h)
            VALUES (?, 0, ?, 0.72, 0.5, 0.5, 0.2, 0.2);
            """,
            (m_id, pid),
        )
        face_id2 = cur.lastrowid
        conn.commit()

    cat_repo.reject_face_suggestion(face_id2, pid)
    rejections = cat_repo.get_face_rejections()
    assert face_id2 in rejections
    assert pid in rejections[face_id2]


def test_update_face_rect_and_display_picture(tmp_path: Path, db_engine):
    """Verifies update_face_rect and set_person_display_picture_from_face."""
    cat_repo = CatalogRepository(db_engine)
    thm_repo = ThumbnailRepository(db_engine)

    img_dir = tmp_path / "photos"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_path = img_dir / "portrait.jpg"
    img = Image.new("RGB", (400, 400), color=(120, 150, 180))
    img.save(img_path)

    fid = cat_repo.upsert_folder(str(img_dir), "photos", None, 1700000000)
    items = [
        {
            "folder_id": fid,
            "filename": "portrait.jpg",
            "ext": "jpg",
            "size": 1000,
            "mtime": 1700000000,
            "date_taken": 1700000000,
            "rating": 0,
            "flag": 0,
        }
    ]
    m_id = cat_repo.bulk_upsert_media(items)[0]
    pid = cat_repo.upsert_person("Alice")

    with db_engine.write_lock:
        conn = db_engine.catalog_conn()
        cur = conn.execute(
            """
            INSERT INTO media_people (media_id, person_id, state, is_manual, rect_x, rect_y, rect_w, rect_h)
            VALUES (?, ?, 1, 1, 0.1, 0.1, 0.2, 0.2);
            """,
            (m_id, pid),
        )
        face_id = cur.lastrowid
        conn.commit()

    # Update face rect
    new_rect = (0.25, 0.25, 0.45, 0.45)
    ok = cat_repo.update_face_rect(face_id, new_rect, thumb_repo=thm_repo)
    assert ok is True

    row = db_engine.catalog_conn().execute(
        "SELECT rect_x, rect_y, rect_w, rect_h FROM media_people WHERE id = ?;", (face_id,)
    ).fetchone()
    assert abs(row["rect_x"] - 0.25) < 1e-4
    assert abs(row["rect_w"] - 0.45) < 1e-4

    # Set as display picture from face
    ok_dp = cat_repo.set_person_display_picture_from_face(pid, face_id)
    assert ok_dp is True

    p_row = db_engine.catalog_conn().execute(
        "SELECT thumbnail_blob FROM people WHERE id = ?;", (pid,)
    ).fetchone()
    assert p_row["thumbnail_blob"] is not None
    assert len(p_row["thumbnail_blob"]) > 0


def test_face_crop_context_menu_items(qapp):
    """Verifies the Face Crop context menu presents all required face actions."""
    from PySide6.QtCore import QPoint
    from src.ui.grid_view import ThumbnailGridView, MediaCardDelegate

    grid = ThumbnailGridView()
    delegate = MediaCardDelegate(grid)
    grid.setItemDelegate(delegate)
    grid.card_delegate = delegate
    grid.card_delegate.set_face_crop_person_id(42)
    grid._suppress_menu_exec = True

    # Invoke face crop menu
    menu = grid._show_face_crop_context_menu(QPoint(10, 10), media_id=1, person_id=42)
    action_texts = [a.text() for a in menu.actions() if a.text()]

    assert "Set as display picture" in action_texts
    assert any("Not this person" in t for t in action_texts)
    assert any("Not a face" in t for t in action_texts)
    assert any("Tag as someone else..." in t for t in action_texts)
    assert "View full photo" in action_texts
    assert "Open file location" in action_texts


def test_single_viewer_face_adjust_mode(qapp):
    """Verifies SingleMediaViewer entering and exiting face bounding box adjustment mode."""
    from src.ui.media_viewer import SingleMediaViewer

    viewer = SingleMediaViewer()
    viewer.show()

    # Create dummy image
    test_img = Image.new("RGB", (600, 600), color=(200, 200, 200))
    viewer.current_pil_image = test_img
    viewer.current_media_id = 99
    viewer.current_is_video = False

    region_data = {
        "face_id": 5,
        "person_id": 10,
        "name": "Marcus",
        "rect_x": 0.2,
        "rect_y": 0.2,
        "rect_w": 0.3,
        "rect_h": 0.3,
    }

    # Test context menu creation
    viewer._suppress_menu_exec = True
    menu = viewer.show_face_region_context_menu(QPoint(50, 50), region_data, 10)
    action_texts = [a.text() for a in menu.actions() if a.text()]
    assert "Adjust bounding box" in action_texts
    assert "Tag as someone else..." in action_texts
    assert "Not this person" in action_texts
    assert "Not a face" in action_texts
    assert "Set as display picture" in action_texts

    # Start adjust mode
    started = viewer.start_face_adjust_mode(region_data, face_id=5, person_id=10)
    assert started is True
    assert viewer.is_face_adjust_mode is True
    assert viewer._face_adjust_box_item is not None
    assert not viewer._adjust_toolbar.isHidden()

    # Stop adjust mode
    viewer.stop_face_adjust_mode()
    assert viewer.is_face_adjust_mode is False
    assert viewer._face_adjust_box_item is None
    assert viewer._adjust_toolbar.isHidden()



