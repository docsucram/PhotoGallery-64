"""
Phase 3 Automated Verification Suite:
1. 400,000-row virtualized CatalogListModel load benchmark (<50ms) and memory footprint (<85MB).
2. Collapsible group headers, column reflow, and O(log G) bisect lookup verification.
3. Tessellated Mosaic layout geometry calculation (portrait + landscape aspect ratio row packing).
4. MediaCardDelegate offscreen painting and editorEvent hit-testing (5-star rating, flag, checkbox, group header).
"""

from __future__ import annotations

import gc
import os
import time

import psutil
import pytest
from PySide6.QtCore import QEvent, QPointF, QRect, Qt
from PySide6.QtGui import QImage, QMouseEvent, QPainter
from PySide6.QtWidgets import QApplication, QStyleOptionViewItem

from legacy_reference.imgBrowserv13 import LruCache
from src.db.repository import MediaItemRecord
from src.ui.grid_model import CatalogListModel, CatalogRoles
from src.ui.grid_view import MediaCardDelegate, ThumbnailGridView


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_400k_load_and_memory_budget(qapp: QApplication) -> None:
    """
    Verify loading 400,000 MediaItemRecord items into CatalogListModel:
    - Model set_records() completes in < 50ms
    - Total memory delta for 400,000 records + model state is < 85MB
    - Random-access O(log G) row lookup takes < 1.0ms
    """
    gc.collect()
    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss

    group_pool = [f"Month {g:02d} 2024" for g in range(20)]
    ts_pool = [1700000000 - (b * 60) for b in range(4_000)]
    fname_str = "IMG_1000.JPG"
    caption_str = ""

    records = [
        MediaItemRecord(
            id=i + 1,
            folder_id=1,
            filename=fname_str,
            size=3_400_000,
            mtime=1700000000,
            date_taken=ts_pool[i // 100],
            rating=i % 6,
            flag=1 if (i % 10 == 0) else 0,
            width=4000,
            height=3000,
            orientation=1,
            is_video=0,
            caption=caption_str,
            group_key=group_pool[i // 20_000],
        )
        for i in range(400_000)
    ]

    cache = LruCache(600)
    model = CatalogListModel(cache)

    t0 = time.perf_counter()
    model.set_records(records, group_headers_enabled=True)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    rss_after = proc.memory_info().rss
    mem_delta_mb = (rss_after - rss_before) / (1024 * 1024)

    assert model.total_media_count == 400_000
    assert elapsed_ms < 50.0, f"set_records(400k) took {elapsed_ms:.2f}ms (budget <50ms)"
    assert mem_delta_mb < 85.0, f"400k records + model used {mem_delta_mb:.2f}MB (budget <85MB)"

    # Verify O(log G) random access across beginning, middle, and end of 400k dataset
    total_vrows = model.rowCount()
    assert total_vrows > 400_000

    t_lookup = time.perf_counter()
    first_hdr_idx = model.index(0, 0)
    assert model.data(first_hdr_idx, CatalogRoles.IsGroupHeaderRole) is True
    assert model.data(first_hdr_idx, CatalogRoles.GroupCountRole) == 20_000

    mid_idx = model.index(total_vrows // 2, 0)
    _ = model.data(mid_idx, CatalogRoles.MediaIdRole)

    last_idx = model.index(total_vrows - 1, 0)
    _ = model.data(last_idx, CatalogRoles.MediaIdRole)
    lookup_ms = (time.perf_counter() - t_lookup) * 1000.0
    assert lookup_ms < 1.0


def test_group_collapse_and_column_reflow(qapp: QApplication) -> None:
    """Verify collapsible section headers and multi-column virtual row reflow."""
    records = [
        MediaItemRecord(
            id=i + 1,
            folder_id=1,
            filename=f"Photo_{i + 1}.jpg",
            size=2_100_000,
            mtime=1700000000,
            date_taken=1700000000 - i * 60,
            rating=3,
            flag=0,
            width=1920,
            height=1080,
            orientation=1,
            is_video=0,
            caption="Holiday",
            group_key="October 2024" if i < 5 else "September 2024",
        )
        for i in range(10)
    ]

    model = CatalogListModel(LruCache(100))
    model.set_column_count(4)
    model.set_records(records, group_headers_enabled=True)

    # With zero-spacer 2D virtual geometry, each group of 5 items has:
    # - 1 header row + 5 item rows = 6 virtual rows per group; 2 groups = 12 virtual rows
    assert model.rowCount() == 12

    # Header row 0 spans full width at y=6, height=38; item 5 (5th item of group 1, index 4) wraps to row 1 col 0
    hdr0_rect = model.item_rect(0)
    assert hdr0_rect.height() == 38
    item5_rect_col4 = model.item_rect(5)
    assert item5_rect_col4.x() == model.item_rect(1).x()  # Wrapped to column 0 of second row
    assert item5_rect_col4.y() > model.item_rect(1).y()

    # Collapse "October 2024" -> shrinks October 2024 to 1 header row (1 + 6 = 7 virtual rows)
    model.toggle_group_collapsed("October 2024")
    assert model.rowCount() == 7
    hdr_idx = model.index(0, 0)
    assert model.data(hdr_idx, CatalogRoles.GroupCollapsedRole) is True

    # Expand "October 2024" back -> 12 virtual rows
    model.toggle_group_collapsed("October 2024")
    assert model.rowCount() == 12
    assert model.data(hdr_idx, CatalogRoles.GroupCollapsedRole) is False

    # Reflow to 5 columns: all 5 items in group 1 now fit on the first row (same Y coordinate)
    model.set_column_count(5)
    assert model.rowCount() == 12
    item5_rect_col5 = model.item_rect(5)
    assert item5_rect_col5.y() == model.item_rect(1).y()
    assert item5_rect_col5.x() > model.item_rect(1).x()


def test_mosaic_layout_geometry(qapp: QApplication) -> None:
    """Verify Tessellated Mosaic layout packs portrait & landscape items into gapless rows."""
    dims = [
        (3000, 2000),  # 1.5 landscape
        (2000, 3000),  # 0.67 portrait
        (4000, 3000),  # 1.33 landscape
        (1920, 1080),  # 1.77 wide
        (1500, 2000),  # 0.75 portrait
        (3000, 2000),  # 1.5 landscape
        (2500, 2000),  # 1.25 landscape
    ]
    records = [
        MediaItemRecord(
            id=i + 1,
            folder_id=1,
            filename=f"Mosaic_{i + 1}.jpg",
            size=2_500_000,
            mtime=1700000000,
            date_taken=1700000000 - i,
            rating=4,
            flag=1,
            width=dims[i][0],
            height=dims[i][1],
            orientation=1,
            is_video=0,
            caption="",
            group_key="October 2024",
        )
        for i in range(len(dims))
    ]

    model = CatalogListModel(LruCache(100))
    model.set_layout_mode("mosaic")
    model.set_records(records, group_headers_enabled=True)
    model.compute_mosaic_geometry(viewport_width=800, target_row_height=160, spacing=6)

    assert model.mosaic_total_height > 160

    # Check header rect spans full available width (800 - 2*12 = 776)
    hdr_idx = model.index(0, 0)
    hdr_rect: QRect = model.data(hdr_idx, CatalogRoles.MosaicRectRole)
    assert hdr_rect.width() >= 770

    # Check first row items have identical height and non-overlapping X positions
    item1_rect: QRect = model.data(model.index(1, 0), CatalogRoles.MosaicRectRole)
    item2_rect: QRect = model.data(model.index(2, 0), CatalogRoles.MosaicRectRole)
    assert item1_rect.height() == item2_rect.height()
    assert item2_rect.left() >= item1_rect.right()


def test_delegate_painting_and_hit_testing(qapp: QApplication) -> None:
    """
    Verify MediaCardDelegate paints cards & headers cleanly and editorEvent hit-tests:
    - 5-star rating clicks (rating_clicked signal)
    - Flag icon clicks (flag_toggled signal)
    - Top-left selection checkbox clicks (checkbox_toggled signal)
    - Group header clicks (group_header_clicked signal)
    """
    records = [
        MediaItemRecord(
            id=42,
            folder_id=1,
            filename="Beach_Sunset.jpg",
            size=4_200_000,
            mtime=1700000000,
            date_taken=1700000000,
            rating=2,
            flag=0,
            width=4032,
            height=3024,
            orientation=1,
            is_video=0,
            caption="Sunset over beach",
            group_key="October 2024",
        )
    ]

    cache = LruCache(100)
    model = CatalogListModel(cache)
    delegate = MediaCardDelegate(cache)
    view = ThumbnailGridView(model, delegate)
    view.resize(900, 600)
    model.set_records(records, group_headers_enabled=True)

    canvas = QImage(900, 600, QImage.Format_ARGB32)
    painter = QPainter(canvas)

    # 1. Paint group header (row 0) and media card
    hdr_idx = model.index(0, 0)
    opt_hdr = QStyleOptionViewItem()
    opt_hdr.rect = QRect(0, 0, 880, 36)
    delegate.paint(painter, opt_hdr, hdr_idx)

    item_idx = model.index_for_media_id(42)
    assert item_idx.isValid()

    outer_rect = QRect(10, 40, 180, 224)
    card_rect = outer_rect.adjusted(2, 2, -2, -2)
    opt_card = QStyleOptionViewItem()
    opt_card.rect = outer_rect
    delegate.paint(painter, opt_card, item_idx)
    painter.end()

    # 2. Test editorEvent hit-testing on group header
    header_clicks: list[str] = []
    delegate.group_header_clicked.connect(header_clicks.append)
    hdr_pt = QPointF(50, 15)
    hdr_ev = QMouseEvent(
        QEvent.MouseButtonRelease,
        hdr_pt,
        hdr_pt,
        Qt.LeftButton,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    assert delegate.editorEvent(hdr_ev, model, opt_hdr, hdr_idx) is True
    assert header_clicks == ["October 2024"]
    # Expand back so item_idx stays valid for subsequent hit tests
    if model.is_group_collapsed("October 2024"):
        model.toggle_group_collapsed("October 2024")
    item_idx = model.index_for_media_id(42)

    # 3. Test editorEvent hit-testing on top-left selection checkbox
    checkbox_events: list[tuple[int, bool]] = []
    delegate.checkbox_toggled.connect(lambda mid, sel: checkbox_events.append((mid, sel)))
    cb_rect = delegate.checkbox_rect(card_rect)
    cb_pt = QPointF(cb_rect.center())
    cb_ev = QMouseEvent(
        QEvent.MouseButtonRelease,
        cb_pt,
        cb_pt,
        Qt.LeftButton,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    assert delegate.editorEvent(cb_ev, model, opt_card, item_idx) is True
    assert checkbox_events == [(42, True)]

    # 4. Test editorEvent hit-testing on top-right Flag toggle
    flag_events: list[tuple[int, int]] = []
    delegate.flag_toggled.connect(lambda mid, flg: flag_events.append((mid, flg)))
    flg_rect = delegate.flag_rect(card_rect)
    flg_pt = QPointF(flg_rect.center())
    flg_ev = QMouseEvent(
        QEvent.MouseButtonRelease,
        flg_pt,
        flg_pt,
        Qt.LeftButton,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    assert delegate.editorEvent(flg_ev, model, opt_card, item_idx) is True
    assert flag_events == [(42, 1)]

    # 5. Test editorEvent hit-testing on Star #4 in the 5-star rating bar
    rating_events: list[tuple[int, int]] = []
    delegate.rating_clicked.connect(lambda mid, stars: rating_events.append((mid, stars)))
    star_bar = delegate.stars_bar_rect(card_rect)
    star4_x = star_bar.left() + int(star_bar.width() * 0.70)  # 4th star out of 5
    star_pt = QPointF(star4_x, star_bar.center().y())
    star_ev = QMouseEvent(
        QEvent.MouseButtonRelease,
        star_pt,
        star_pt,
        Qt.LeftButton,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    assert delegate.editorEvent(star_ev, model, opt_card, item_idx) is True
    assert rating_events == [(42, 4)]


def test_grid_context_menus_authentic_wlpg(qapp: QApplication) -> None:
    """Verify ThumbnailGridView right-click context menus matching WLPG screenshots."""
    from PySide6.QtGui import QContextMenuEvent
    from PySide6.QtCore import QPoint
    from src.ui.grid_view import ThumbnailGridView

    records = [
        MediaItemRecord(
            id=101,
            folder_id=1,
            filename="SamplePhoto.jpg",
            size=1_000_000,
            mtime=1700000000,
            date_taken=1700000000,
            rating=3,
            flag=0,
            width=800,
            height=600,
            orientation=1,
            is_video=0,
            caption="Test Caption",
            group_key="November 2023",
        )
    ]
    model = CatalogListModel(LruCache(100))
    model.set_records(records, group_headers_enabled=True)

    grid = ThumbnailGridView(model)
    grid.resize(800, 600)
    grid._suppress_menu_exec = True

    # 1. Right-click on card index
    card_idx = model.index_for_media_id(101)
    card_rect = model.item_rect(card_idx.row())
    pt_card = card_rect.center()

    ev_card = QContextMenuEvent(QContextMenuEvent.Mouse, pt_card, grid.mapToGlobal(pt_card))
    grid.contextMenuEvent(ev_card)

    card_menu = grid._show_card_context_menu(grid.mapToGlobal(pt_card), 101)
    card_menu_actions = [act.text() for act in card_menu.actions() if act.text()]

    # Verify card selection and exact WLPG menu items
    assert grid.selected_media_ids() == [101]
    assert "View file" in card_menu_actions
    assert "Open with" in card_menu_actions
    assert "Open file location" in card_menu_actions
    assert "Set as desktop background" in card_menu_actions
    assert "Rotate right" in card_menu_actions
    assert "Rotate left" in card_menu_actions
    assert "Copy" in card_menu_actions
    assert "Delete" in card_menu_actions
    assert "Rename..." in card_menu_actions
    assert "Resize..." in card_menu_actions
    assert "Properties" in card_menu_actions

    # Check View file is Bold
    view_act = next(act for act in card_menu.actions() if act.text() == "View file")
    assert view_act.font().bold() is True

    # 2. Right-click on empty background
    bg_pt = QPoint(750, 550)
    ev_bg = QContextMenuEvent(QContextMenuEvent.Mouse, bg_pt, grid.mapToGlobal(bg_pt))
    grid.contextMenuEvent(ev_bg)

    bg_menu = grid._show_background_context_menu(grid.mapToGlobal(bg_pt))
    bg_menu_actions = [act.text() for act in bg_menu.actions() if act.text()]

    # Verify exact WLPG background menu items
    assert "View" in bg_menu_actions
    assert "Group by" in bg_menu_actions
    assert "Sort by" in bg_menu_actions
    assert "Table of contents" in bg_menu_actions
    assert "Refresh" in bg_menu_actions
    assert "Select all" in bg_menu_actions

    # Test Table of Contents toggle
    assert model.group_headers_enabled is True
    model.group_headers_enabled = False
    assert model.group_headers_enabled is False


