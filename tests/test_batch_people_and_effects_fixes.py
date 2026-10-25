"""
Tests for Ribbon Effects whitespace/tooltip fixes, People Tag context menu actions,
and Batch People Tag distinct styling + Esc/Back navigation.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QStyleOptionViewItem

from legacy_reference.imgBrowserv13 import LruCache
from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository, ThumbnailRepository
from src.ui.batch_people_view import (
    BatchPeopleView,
    FaceCardRecord,
    _FaceCardDelegate,
    _FaceCardModel,
)
from src.ui.main_window import PhotoGalleryMainWindow
from src.ui.ribbon_toolbar import _RibbonEffectsGallery


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


def test_ribbon_effects_gallery_spacing_and_tooltip(qapp: QApplication) -> None:
    gallery = _RibbonEffectsGallery()
    gallery.resize(450, 60)
    gallery.show()

    # Verify CARD_STRIDE is 49 (48px button + 1px spacing)
    assert gallery.CARD_STRIDE == 49

    # Verify stylesheet sets clean Aero tooltip (not black)
    ss = gallery.styleSheet()
    assert "QToolTip" in ss
    assert "background-color: #FFFFFF" in ss
    assert "color: #1E395B" in ss

    # Check button positions: buttons should be positioned contiguously
    # in cards_layout without artificial stretching between them
    btn_list = list(gallery.buttons.values())
    assert len(btn_list) >= 6
    p0 = btn_list[0].pos().x()
    p1 = btn_list[1].pos().x()
    assert (p1 - p0) == 49


def test_face_card_distinct_styling_confirmed_vs_suggestion(qapp: QApplication) -> None:
    mock_thumb_repo = MagicMock(spec=ThumbnailRepository)
    mock_thumb_repo.get_face_thumbnail.return_value = None
    cache = LruCache(10)

    delegate = _FaceCardDelegate(mock_thumb_repo, cache)

    # Confirmed record
    rec_conf = FaceCardRecord(
        face_id=1,
        media_id=10,
        suggested_person_id=5,
        suggested_name="Alice",
        confidence=0.95,
        rect_x=0.1,
        rect_y=0.1,
        rect_w=0.2,
        rect_h=0.2,
        filename="alice.jpg",
        folder_path="C:/Photos",
        is_confirmed=True,
    )

    # Suggestion record
    rec_sugg = FaceCardRecord(
        face_id=2,
        media_id=10,
        suggested_person_id=5,
        suggested_name="Alice",
        confidence=0.88,
        rect_x=0.3,
        rect_y=0.1,
        rect_w=0.2,
        rect_h=0.2,
        filename="alice2.jpg",
        folder_path="C:/Photos",
        is_confirmed=False,
    )

    img = QImage(300, 300, QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    painter = QPainter(img)

    opt = QStyleOptionViewItem()
    opt.rect = QRect(0, 0, 110, 140)

    # Paint confirmed
    model = _FaceCardModel([rec_conf])
    idx_conf = model.index(0, 0)
    delegate.paint(painter, opt, idx_conf)

    # Paint suggestion
    model2 = _FaceCardModel([rec_sugg])
    idx_sugg = model2.index(0, 0)
    opt.rect = QRect(120, 0, 110, 140)
    delegate.paint(painter, opt, idx_sugg)

    painter.end()
    assert rec_conf.is_confirmed is True
    assert rec_sugg.is_confirmed is False


def test_right_dock_context_menu_handlers(qapp: QApplication, test_engine: DualDatabaseEngine) -> None:
    repo = CatalogRepository(test_engine)
    p_id = repo.upsert_person("Bob Smith")

    mw = PhotoGalleryMainWindow(test_engine, start_xmp_daemon=False)
    try:
        mw.left_nav_dock.select_person = MagicMock()
        mw.rebuild_and_run_query = MagicMock()
        mw.open_batch_people_view = MagicMock()

        # 1. Search for photos of this person
        mw._on_dock_person_search_requested(p_id, "Bob Smith")
        mw.left_nav_dock.select_person.assert_called_once_with(p_id)
        mw.rebuild_and_run_query.assert_called_once()

        # 2. View all suggestions (with single or multiple args)
        mw._on_dock_person_view_suggestions_requested(p_id)
        mw.open_batch_people_view.assert_called_once_with(person_id=p_id)
    finally:
        mw.close()


def test_batch_people_nav_and_esc_back_to_view(qapp: QApplication, test_engine: DualDatabaseEngine, tmp_path: Path) -> None:
    # Create sample media file
    photo_file = tmp_path / "test.jpg"
    im = Image.new("RGB", (200, 200), (120, 150, 180))
    im.save(photo_file)

    conn = test_engine.catalog_conn()
    cur = conn.execute("INSERT INTO folders (path, name) VALUES (?, ?)", (str(tmp_path), tmp_path.name))
    folder_id = cur.lastrowid
    conn.execute(
        """
        INSERT INTO media (id, folder_id, filename, ext, size, width, height, is_video, rating)
        VALUES (1, ?, 'test.jpg', '.jpg', 1024, 200, 200, 0, 0);
        """,
        (folder_id,),
    )
    conn.commit()

    mw = PhotoGalleryMainWindow(test_engine, start_xmp_daemon=False)
    try:
        # Initially in grid view
        assert mw.central_stack.currentWidget() == mw.grid_view

        # Open BatchPeopleView
        mw.open_batch_people_view(person_id=1)
        assert mw.central_stack.currentWidget() == mw.batch_people_view

        # Click through to open a photo in single media viewer
        ok = mw.open_single_viewer(1)
        assert ok is True
        assert mw.central_stack.currentWidget() == mw.media_viewer
        assert mw._prev_central_widget == mw.batch_people_view

        # Pressing Esc or calling show_grid_view() should return to batch_people_view
        mw.show_grid_view()
        assert mw.central_stack.currentWidget() == mw.batch_people_view
        assert mw.lbl_sync_status.text() == "Batch people tag view"

        # Esc from batch_people_view should close batch view back to grid
        mw.close_batch_people_view()
        assert mw.central_stack.currentWidget() == mw.grid_view

        # QAT Back button should navigate history back to batch_people_view
        mw.navigate_history_back()
        assert mw.central_stack.currentWidget() == mw.batch_people_view
    finally:
        mw.close()
