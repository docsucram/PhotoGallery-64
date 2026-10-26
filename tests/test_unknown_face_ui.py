"""
Tests for Universal Tooltip Styling, View Ribbon Zoom Group Sizing,
and Detected Unknown & Suggested Face Tagging Overhaul.
"""

from __future__ import annotations

from pathlib import Path
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QLineEdit, QPushButton, QSizePolicy, QToolTip

from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository, ThumbnailRepository
from src.ui.main_window import PhotoGalleryMainWindow
from src.ui.media_viewer import SingleMediaViewer, _FaceHoverRegionItem, _FaceNameBadgeItem
from src.ui.right_tag_dock import TagAndCaptionDock, _PersonTagCardWidget, _UnknownFaceCardWidget
from src.ui.ribbon_toolbar import RibbonToolbar, _RibbonGroupBox
from src.utils.scan_logger import install_universal_tooltip_filter


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    install_universal_tooltip_filter(app)
    return app


@pytest.fixture
def test_engine(tmp_path: Path):
    engine = DualDatabaseEngine(
        catalog_path=tmp_path / "catalog.db",
        thumbnails_path=tmp_path / "thumbnails.db",
        schema_sql_path=Path("db/schema.sql"),
    )
    engine.initialize_schemas()
    return engine


# -------------------------------------------------------------------------
# 1. Universal Tooltip Fix Tests
# -------------------------------------------------------------------------
def test_universal_tooltip_filter(qapp):
    """Verifies that universal tooltip filter intercepts tooltips and enforces white/blue Aero theme."""
    filt = install_universal_tooltip_filter(qapp)
    assert filt is not None

    btn = QPushButton("Test Button")
    btn.setStyleSheet("QPushButton { background: black; color: white; }")
    btn.setToolTip("Test Tooltip Text")
    btn.show()
    qapp.processEvents()

    # Trigger tooltip
    QToolTip.showText(QPoint(50, 50), "Test Tooltip Text", btn)
    qapp.processEvents()

    tooltips = [
        w for w in qapp.topLevelWidgets()
        if w.objectName() == "qtooltip_label" or w.windowType() == Qt.WindowType.ToolTip
    ]
    if tooltips:
        tip = tooltips[0]
        # Ensure palette has white tooltip base and dark text
        pal = tip.palette()
        assert pal.color(QPalette.ColorRole.ToolTipBase).name().lower() in ("#ffffff", "#ffffe1")
        assert pal.color(QPalette.ColorRole.ToolTipText).name().lower() in ("#1e395b", "#000000", "#111111")
    QToolTip.hideText()
    btn.close()


# -------------------------------------------------------------------------
# 2. View Ribbon Zoom Group Gap Tests
# -------------------------------------------------------------------------
def test_ribbon_zoom_group_size_policy_and_shrink(qapp):
    """Verifies that _RibbonGroupBox and grp_zoom shrink properly in Grid mode without residual gaps."""
    ribbon = RibbonToolbar()
    ribbon.tab_widget.setCurrentWidget(ribbon.tab_view)
    grp_zoom = ribbon.grp_zoom
    assert grp_zoom.sizePolicy().horizontalPolicy() == QSizePolicy.Maximum

    # Grid mode initially: only zoom_in and zoom_out visible
    ribbon.set_active_mode("grid")
    qapp.processEvents()
    assert not ribbon.btn_zoom_in.isHidden()
    assert not ribbon.btn_zoom_out.isHidden()
    assert ribbon.btn_zoom_actual.isHidden()
    assert ribbon.btn_zoom_fit.isHidden()
    grid_width = grp_zoom.sizeHint().width()

    # Switch to viewer mode: 4 buttons visible
    ribbon.set_active_mode("viewer")
    qapp.processEvents()
    assert not ribbon.btn_zoom_actual.isHidden()
    assert not ribbon.btn_zoom_fit.isHidden()
    viewer_width = grp_zoom.sizeHint().width()
    assert viewer_width > grid_width

    # Switch back to grid mode: should contract back without leaving large viewer width
    ribbon.set_active_mode("grid")
    qapp.processEvents()
    assert ribbon.btn_zoom_actual.isHidden()
    assert ribbon.btn_zoom_fit.isHidden()
    assert grp_zoom.sizeHint().width() == grid_width
    ribbon.close()


# -------------------------------------------------------------------------
# 3. Database Query Tests: Detected Unknown & Suggested Faces (state >= 0)
# -------------------------------------------------------------------------
def test_get_media_detail_includes_all_state_ge_zero_faces(test_engine):
    """Verifies that get_media_detail queries confirmed (1), suggested (0), and unknown (0) faces, excluding -1."""
    repo = CatalogRepository(test_engine)
    conn = test_engine.catalog_conn()

    # Insert test folder and media
    f_cur = conn.execute("INSERT INTO folders (path, name) VALUES ('C:/Photos', 'Photos') RETURNING id;")
    fid = f_cur.fetchone()[0]
    m_cur = conn.execute(
        "INSERT INTO media (folder_id, filename, size, mtime, ext) VALUES (?, 'photo.jpg', 1024, 1000, '.jpg') RETURNING id;",
        (fid,),
    )
    mid = m_cur.fetchone()[0]

    # Create known people
    pid_alice = repo.upsert_person("Alice")
    pid_bob = repo.upsert_person("Bob")

    # 1. Confirmed face: Alice (state = 1, person_id = pid_alice)
    conn.execute(
        "INSERT INTO media_people (media_id, person_id, state, rect_x, rect_y, rect_w, rect_h) VALUES (?, ?, 1, 0.1, 0.1, 0.2, 0.2);",
        (mid, pid_alice),
    )
    # 2. Suggested face: Bob (state = 0, suggested_person_id = pid_bob)
    conn.execute(
        "INSERT INTO media_people (media_id, suggested_person_id, state, similarity_score, rect_x, rect_y, rect_w, rect_h) VALUES (?, ?, 0, 0.88, 0.35, 0.1, 0.2, 0.2);",
        (mid, pid_bob),
    )
    # 3. Detected unknown face: unassigned (state = 0, person_id = NULL, suggested_person_id = NULL)
    conn.execute(
        "INSERT INTO media_people (media_id, person_id, suggested_person_id, state, rect_x, rect_y, rect_w, rect_h) VALUES (?, NULL, NULL, 0, 0.6, 0.1, 0.2, 0.2);",
        (mid,),
    )
    # 4. Rejected face: false positive (state = -1) - should NOT be included
    conn.execute(
        "INSERT INTO media_people (media_id, person_id, state, rect_x, rect_y, rect_w, rect_h) VALUES (?, NULL, -1, 0.8, 0.1, 0.1, 0.1);",
        (mid,),
    )
    conn.commit()

    detail = repo.get_media_detail(mid)
    face_regions = detail.get("face_regions", [])
    assert len(face_regions) == 3

    # Check confirmed
    confirmed = detail.get("confirmed_people", [])
    assert len(confirmed) == 1
    assert confirmed[0]["person_name"] == "Alice"
    assert confirmed[0]["state"] == 1

    # Check unconfirmed suggestion
    unconfirmed = detail.get("unconfirmed_people", [])
    assert len(unconfirmed) == 1
    assert unconfirmed[0]["person_name"] == "Bob"
    assert unconfirmed[0]["state"] == 0

    # Check unknown face
    unknown = detail.get("unknown_people", [])
    assert len(unknown) == 1
    assert unknown[0]["person_name"] is None
    assert unknown[0]["state"] == 0
    assert unknown[0]["rect_x"] == 0.6


def test_assign_and_dismiss_face_region(test_engine):
    """Verifies assign_face_to_person and dismiss_face_region repository functions."""
    repo = CatalogRepository(test_engine)
    conn = test_engine.catalog_conn()

    f_cur = conn.execute("INSERT INTO folders (path, name) VALUES ('C:/Photos', 'Photos') RETURNING id;")
    fid = f_cur.fetchone()[0]
    m_cur = conn.execute(
        "INSERT INTO media (folder_id, filename, size, mtime, ext) VALUES (?, 'img2.jpg', 2048, 2000, '.jpg') RETURNING id;",
        (fid,),
    )
    mid = m_cur.fetchone()[0]

    # Create detected unknown face
    cur = conn.execute(
        "INSERT INTO media_people (media_id, person_id, state, rect_x, rect_y, rect_w, rect_h) VALUES (?, NULL, 0, 0.2, 0.2, 0.2, 0.2) RETURNING id;",
        (mid,),
    )
    face_id = cur.fetchone()[0]
    conn.commit()

    # Assign face to Charlie
    pid_charlie = repo.assign_face_to_person(face_id, "Charlie")
    assert pid_charlie > 0

    detail = repo.get_media_detail(mid)
    assert len(detail["confirmed_people"]) == 1
    assert detail["confirmed_people"][0]["person_name"] == "Charlie"
    assert detail["confirmed_people"][0]["state"] == 1
    assert detail["metadata_dirty"] == 1

    # Now create another face and dismiss it
    cur2 = conn.execute(
        "INSERT INTO media_people (media_id, person_id, state, rect_x, rect_y, rect_w, rect_h) VALUES (?, NULL, 0, 0.5, 0.5, 0.1, 0.1) RETURNING id;",
        (mid,),
    )
    face_id2 = cur2.fetchone()[0]
    conn.commit()

    repo.dismiss_face_region(face_id2)
    detail2 = repo.get_media_detail(mid)
    # The dismissed face (state = -1) must be excluded
    fids = [r["face_id"] for r in detail2["face_regions"]]
    assert face_id2 not in fids


# -------------------------------------------------------------------------
# 4. RightTagDock UI Tests
# -------------------------------------------------------------------------
def test_right_tag_dock_renders_unknown_and_suggested_cards(qapp, test_engine):
    """Verifies that RightTagDock renders _UnknownFaceCardWidget for unknown faces and _PersonTagCardWidget with ✓/✗ for suggestions."""
    dock = TagAndCaptionDock()
    dock.set_autocomplete_lists(["Tag1"], ["Alice", "Bob", "Charlie"])

    summary = {
        "count": 1,
        "face_regions": [
            {
                "face_id": 101,
                "person_id": 1,
                "person_name": "Alice",
                "state": 1,
                "is_unconfirmed": False,
            },
            {
                "face_id": 102,
                "person_id": 2,
                "person_name": "Bob",
                "state": 0,
                "is_unconfirmed": True,
            },
            {
                "face_id": 103,
                "person_id": None,
                "person_name": None,
                "state": 0,
                "is_unconfirmed": True,
            },
        ],
    }

    hovered_events = []
    dock.face_region_hovered.connect(lambda reg_id: hovered_events.append(reg_id))

    dock.bind_selection([1], summary)
    qapp.processEvents()

    # Check child widgets in people_chips_container
    chips = dock.people_chips_container.findChildren(_PersonTagCardWidget)
    assert len(chips) == 2  # Alice and Bob

    # Alice (confirmed): no inline ✕ button; removed via menu options
    chip_alice = [c for c in chips if c.person_name == "Alice"][0]
    assert not hasattr(chip_alice, "btn_del")
    assert hasattr(chip_alice, "btn_menu")
    assert not hasattr(chip_alice, "btn_confirm")

    # Bob (suggestion): has ✓ confirm and ✗ reject buttons
    chip_bob = [c for c in chips if c.person_name == "Bob"][0]
    assert hasattr(chip_bob, "btn_confirm")
    assert hasattr(chip_bob, "btn_reject")

    # Unknown face: _UnknownFaceCardWidget
    unknown_chips = dock.people_chips_container.findChildren(_UnknownFaceCardWidget)
    assert len(unknown_chips) == 1
    u_chip = unknown_chips[0]
    assert u_chip.face_id == 103
    assert u_chip.txt_who.placeholderText() == "Who is this?"
    assert hasattr(u_chip, "btn_del")

    dock.close()


# -------------------------------------------------------------------------
# 5. SingleMediaViewer Canvas Overlay Tests
# -------------------------------------------------------------------------
def test_media_viewer_canvas_overlays_and_highlighting(qapp, tmp_path: Path):
    """Verifies that SingleMediaViewer creates overlays for all state >= 0 faces and highlight_face_region works."""
    viewer = SingleMediaViewer()
    
    # Create test image
    img_path = tmp_path / "canvas_test.jpg"
    from PIL import Image
    im = Image.new("RGB", (400, 300), color=(180, 200, 220))
    im.save(img_path)

    viewer.load_media(media_id=5, file_path=img_path)
    qapp.processEvents()

    people_regions = [
        {
            "face_id": 201,
            "person_id": 1,
            "person_name": "Alice",
            "state": 1,
            "rect_x": 0.1,
            "rect_y": 0.1,
            "rect_w": 0.2,
            "rect_h": 0.2,
        },
        {
            "face_id": 202,
            "person_id": 2,
            "person_name": "Bob",
            "state": 0,
            "rect_x": 0.4,
            "rect_y": 0.1,
            "rect_w": 0.2,
            "rect_h": 0.2,
        },
        {
            "face_id": 203,
            "person_id": None,
            "person_name": None,
            "state": 0,
            "rect_x": 0.7,
            "rect_y": 0.1,
            "rect_w": 0.2,
            "rect_h": 0.2,
        },
    ]

    viewer.render_face_regions(people_regions)
    qapp.processEvents()

    badges = [it for it in viewer._face_overlay_items if isinstance(it, _FaceNameBadgeItem)]
    assert len(badges) == 3

    # Check badge names
    badge_texts = [b.name for b in badges]
    assert "Alice" in badge_texts
    assert "Bob" in badge_texts
    assert "Who is this?" in badge_texts

    # Test highlight_face_region
    viewer.highlight_face_region(203)  # Highlight unknown face
    qapp.processEvents()

    # The matching badge and box should be visible
    matching_badges = [b for b in badges if b.data(1) == 203]
    assert len(matching_badges) == 1
    assert matching_badges[0].isVisible()
    assert matching_badges[0].highlighted

    # Non-matching should be hidden
    non_matching = [b for b in badges if b.data(1) != 203]
    assert all(not b.isVisible() for b in non_matching)

    # When un-highlighting (region_id = None), all face overlays must hide
    viewer.highlight_face_region(None)
    qapp.processEvents()
    assert all(not b.isVisible() for b in badges)

    # Test unknown face hoverLeaveEvent hides face overlays when mouse leaves
    sensors = [it for it in viewer._face_overlay_items if isinstance(it, _FaceHoverRegionItem)]
    unk_sensor = [s for s in sensors if s.region_data.get("face_id") == 203][0]
    
    # Enter hover
    unk_sensor.hoverEnterEvent(None)
    assert any(it.isVisible() for it in unk_sensor.related_items)

    # Leave hover -> grace timer starts, and _do_leave_hide hides
    unk_sensor.hoverLeaveEvent(None)
    assert unk_sensor._leave_timer.isActive()
    unk_sensor._do_leave_hide()
    assert all(not it.isVisible() for it in unk_sensor.related_items)

    # Test right-click over face sensor triggers show_face_region_context_menu
    viewer._suppress_menu_exec = True
    menu = viewer.show_face_region_context_menu(QPoint(100, 100), unk_sensor.region_data, None)
    menu_actions = [a.text() for a in menu.actions()]
    assert "Tag person..." in menu_actions
    assert "Ignore face" in menu_actions
    assert "Not a face" in menu_actions

    viewer.close()
