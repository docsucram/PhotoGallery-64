"""
Test Suite for Video Player, Immersive Mode (F4), Search Scope Media Type,
Find Ribbon Flagged Menu, Slideshow Button, and Slideshow Image Loading.
"""

from pathlib import Path
import numpy as np
import cv2
import pytest
from PySide6.QtWidgets import QApplication, QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtMultimedia import QMediaPlayer

from src.ui.media_viewer import SingleMediaViewer
from src.ui.ribbon_toolbar import RibbonToolbar, make_wlpg_icon
from src.ui.search_scope_bar import SearchScopeBar
from src.ui.slideshow_view import FullscreenSlideshowDialog
from src.db.repository import MediaItemRecord


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


def test_video_player_icon_and_volume(qapp, tmp_path):
    viewer = SingleMediaViewer()
    
    # 1. Play/Pause button must be icon-only (no text)
    assert viewer.btn_play_pause.text() == ""
    assert not viewer.btn_play_pause.icon().isNull()

    # Test playback state changes icon
    viewer._on_playback_state_changed(QMediaPlayer.PlayingState)
    assert not viewer.btn_play_pause.icon().isNull()
    viewer._on_playback_state_changed(QMediaPlayer.PausedState)
    assert not viewer.btn_play_pause.icon().isNull()

    # 2. Volume control
    assert hasattr(viewer, "btn_volume")
    assert hasattr(viewer, "slider_volume")
    assert viewer.slider_volume.value() == 100
    assert not viewer.btn_volume.icon().isNull()

    # Changing volume updates audio output
    viewer.slider_volume.setValue(40)
    assert viewer.audio_output.volume() == pytest.approx(0.4, 0.05)

    # Mute toggle
    viewer._toggle_mute()
    assert viewer.audio_output.isMuted() is True
    viewer._toggle_mute()
    assert viewer.audio_output.isMuted() is False
    viewer.close()


def test_search_scope_bar_media_type(qapp):
    scope_bar = SearchScopeBar()
    
    # Simulate filter state with media_type="video"
    state = {
        "media_type": "video",
        "person_ids": (),
        "include_tag_ids": (),
        "ratings": (),
        "flagged_only": False,
    }
    
    class DummyRepo:
        pass
        
    scope_bar.update_active_filter_chips(state, DummyRepo())
    
    # Check that a label "Media type:" was added
    labels = [c.text() for c in scope_bar.chips_widget.findChildren(QLabel)]
    assert "Media type:" in labels

    # Simulate clearing media type
    cleared = []
    scope_bar.media_type_clear_requested.connect(lambda: cleared.append(True))
    scope_bar.media_type_clear_requested.emit()
    assert len(cleared) == 1
    scope_bar.close()


def test_find_ribbon_flagged_menu_and_slideshow_arrow(qapp):
    ribbon = RibbonToolbar()
    
    # 1. Find Ribbon Flagged button has a menu
    assert hasattr(ribbon, "btn_find_flagged")
    menu = ribbon.btn_find_flagged.menu()
    assert menu is not None
    action_texts = [act.text() for act in menu.actions()]
    assert "All items" in action_texts
    assert "Flagged" in action_texts
    assert "Not flagged" in action_texts

    # 2. Slideshow button arrow underneath (InstantPopup)
    assert ribbon.btn_home_slideshow.popupMode() == ribbon.btn_home_slideshow.ToolButtonPopupMode.InstantPopup
    assert ribbon.btn_view_slideshow.popupMode() == ribbon.btn_view_slideshow.ToolButtonPopupMode.InstantPopup

    # Slideshow menu has "Play slide show (F12)" and theme options
    ss_menu = ribbon.btn_home_slideshow.menu()
    assert ss_menu is not None
    ss_actions = [act.text() for act in ss_menu.actions()]
    assert any("Play slide show" in a for a in ss_actions)
    assert any("Pan and Zoom" in a for a in ss_actions)
    ribbon.close()


def test_slideshow_image_fallback_loading(qapp, tmp_path):
    # Create test synthetic image
    img = np.full((200, 300, 3), 100, dtype=np.uint8)
    p = tmp_path / "test_slide.jpg"
    cv2.imwrite(str(p), img)

    rec = MediaItemRecord(
        id=1,
        folder_id=1,
        filename="test_slide.jpg",
        size=1000,
        mtime=123456,
        date_taken=123456,
        rating=0,
        flag=0,
        width=300,
        height=200,
        orientation=1,
        is_video=0,
        caption="Test Slide",
        group_key="2026-10",
    )
    class MockRepo:
        def get_folder_by_id(self, fid):
            return {"path": str(tmp_path)}
        def get_thumbnail_blob(self, mid):
            return None

    dlg = FullscreenSlideshowDialog(records=[rec], start_index=0, catalog_repo=MockRepo())
    assert dlg._current_pixmap is not None
    assert not dlg._current_pixmap.isNull()
    dlg.close()


def test_immersive_mode_f4(qapp):
    from unittest.mock import MagicMock
    from src.ui.main_window import PhotoGalleryMainWindow

    # Mock PhotoGalleryMainWindow components without spinning up full DB
    mw = PhotoGalleryMainWindow.__new__(PhotoGalleryMainWindow)
    mw.left_nav_dock = MagicMock()
    mw.left_nav_dock.isVisible.return_value = True
    mw.right_tag_dock = MagicMock()
    mw.right_tag_dock.isVisible.return_value = True
    mw.media_viewer = MagicMock()
    mw.media_viewer.fix_pane.isVisible.return_value = True
    
    mock_status_bar = MagicMock()
    mock_status_bar.isVisible.return_value = True
    mw.statusBar = MagicMock(return_value=mock_status_bar)
    
    mw.ribbon = MagicMock()
    mw.ribbon.top_header.isVisible.return_value = True
    mw.ribbon.qat_container.isVisible.return_value = True
    mw.ribbon.btn_exit_immersive = MagicMock()
    mw.ribbon._ribbon_collapsed = False

    mw.scope_bar = MagicMock()
    mw.scope_bar.isVisible.return_value = True

    mw.resizeDocks = MagicMock()

    # 1. Toggle ON immersive mode
    PhotoGalleryMainWindow.toggle_immersive_viewer_mode(mw)
    assert getattr(mw, "_is_immersive_mode") is True
    mw.left_nav_dock.hide.assert_called_once()
    mw.right_tag_dock.hide.assert_called_once()
    mw.media_viewer.fix_pane.hide.assert_called_once()
    mock_status_bar.hide.assert_called_once()
    mw.ribbon.top_header.hide.assert_called_once()
    mw.ribbon.qat_container.hide.assert_called_once()
    mw.scope_bar.hide.assert_called_once()
    mw.ribbon.set_ribbon_collapsed.assert_called_with(True)
    mw.ribbon.btn_exit_immersive.show.assert_called_once()

    # 2. Toggle OFF immersive mode (restores previously visible components)
    PhotoGalleryMainWindow.toggle_immersive_viewer_mode(mw)
    assert getattr(mw, "_is_immersive_mode") is False
    mw.left_nav_dock.show.assert_called_once()
    mw.right_tag_dock.show.assert_called_once()
    mw.media_viewer.fix_pane.show.assert_called_once()
    mock_status_bar.show.assert_called_once()
    mw.ribbon.top_header.show.assert_called_once()
    mw.ribbon.qat_container.show.assert_called_once()
    mw.scope_bar.show.assert_called_once()
    mw.ribbon.set_ribbon_collapsed.assert_called_with(False)
    mw.ribbon.btn_exit_immersive.hide.assert_called_once()


def test_ribbon_exit_immersive_pill(qapp):
    ribbon = RibbonToolbar()
    assert hasattr(ribbon, "btn_exit_immersive")
    assert ribbon.btn_exit_immersive.isHidden()
    assert "Exit Immersive" in ribbon.btn_exit_immersive.text()
    
    # Test action triggered signal when clicked
    actions_emitted = []
    ribbon.action_triggered.connect(actions_emitted.append)
    ribbon.btn_exit_immersive.click()
    assert "toggle_immersive" in actions_emitted


def test_creation_saved_does_not_add_desktop_to_library(tmp_path):
    from unittest.mock import MagicMock
    from src.ui.main_window import PhotoGalleryMainWindow

    mw = PhotoGalleryMainWindow.__new__(PhotoGalleryMainWindow)
    mw.catalog_repo = MagicMock()
    # User's library only contains Pictures/Photos
    lib_folder = tmp_path / "Pictures" / "Photos"
    lib_folder.mkdir(parents=True, exist_ok=True)
    mw.catalog_repo.get_folders.return_value = [{"id": 1, "path": str(lib_folder)}]

    mw.scan_folder = MagicMock()
    mw.rebuild_and_run_query = MagicMock()
    mw._select_file_in_grid = MagicMock()
    mw.lbl_sync_status = MagicMock()

    # Case A: Saved to Desktop (External folder)
    desktop_folder = tmp_path / "Desktop"
    desktop_folder.mkdir(parents=True, exist_ok=True)
    desktop_file = desktop_folder / "panorama_output.jpg"
    desktop_file.write_text("dummy")

    assert mw._is_path_in_library(desktop_folder) is False
    mw._on_creation_saved(desktop_file, "panorama")
    
    # scan_folder must NOT be called for external folder (prevents adding Desktop to library)
    mw.scan_folder.assert_not_called()
    mw.lbl_sync_status.setText.assert_called_with("✓ Saved panorama to: panorama_output.jpg")

    # Case B: Saved inside Library
    internal_file = lib_folder / "collage_output.jpg"
    internal_file.write_text("dummy")

    assert mw._is_path_in_library(lib_folder) is True
    mw._on_creation_saved(internal_file, "collage")
    mw.scan_folder.assert_called_once_with(lib_folder, include_subfolders=False)


def test_effects_flower_and_alignment(qapp):
    from src.ui.ribbon_toolbar import make_default_variation_base_image, make_effect_preview_icon
    
    img = make_default_variation_base_image(84, 54)
    assert img.shape == (54, 84, 3)
    assert img.dtype == np.uint8
    
    # Rich color variation across the photographic flower image
    assert len(np.unique(img.reshape(-1, 3), axis=0)) > 500

    icon_bw = make_effect_preview_icon("bw")
    assert not icon_bw.isNull()


def test_people_dropdown_height_calculation(qapp):
    from src.ui.ribbon_toolbar import _WLPGPeopleMenuWidget
    from PySide6.QtWidgets import QMenu, QScrollArea

    menu = QMenu()
    # Test with 25 people (top 10 + 15 other people = 2 rows of other people)
    people = [{"id": i, "name": f"Person {i}", "media_count": i} for i in range(25)]
    w = _WLPGPeopleMenuWidget(menu, people)
    
    # Find scroll area
    scroll = w.findChild(QScrollArea)
    assert scroll is not None
    # For 2 other rows, height must be at least 200px to avoid bottom row cutoff
    assert scroll.height() >= 200


def test_sepia_arbitrary_dimensions_and_stride_safety(qapp):
    from PySide6.QtGui import QPixmap, QPainter, QColor
    from src.ui.slideshow_view import FullscreenSlideshowDialog
    
    # Exact non-4-byte-aligned dimensions that triggered the user's crash: width=811, height=1440
    pix = QPixmap(811, 1440)
    pix.fill(QColor(120, 160, 200))
    
    dlg = FullscreenSlideshowDialog(records=[], start_index=0)
    sepia_pix = dlg._get_sepia_pixmap(999, pix)
    
    assert sepia_pix is not None
    assert not sepia_pix.isNull()
    assert sepia_pix.width() == 811
    assert sepia_pix.height() == 1440
    dlg.close()


def test_ribbon_home_rate_button_dropdown(qapp):
    from PySide6.QtWidgets import QToolButton
    ribbon = RibbonToolbar()
    btn_rate = ribbon.btn_home_rate
    
    # Must be instant popup so clicking it shows the dropdown menu immediately
    assert btn_rate.popupMode() == QToolButton.InstantPopup
    assert "Rate" in (getattr(btn_rate, "original_text", "") or btn_rate.text())
    menu = btn_rate.menu()
    assert menu is not None
    actions = [a.text() for a in menu.actions() if a.text()]
    assert any("5 stars" in a for a in actions)
    assert any("Clear rating" in a for a in actions)


def test_slideshow_split_button_adjacent_layout(qapp):
    ribbon = RibbonToolbar()
    btn_home = ribbon.btn_home_slideshow
    
    assert hasattr(btn_home, "container_widget")
    assert hasattr(btn_home, "btn_bottom")
    
    # Top button has projector icon and plays slideshow (F12)
    assert "F12" in btn_home.toolTip()
    emitted = []
    ribbon.action_triggered.connect(emitted.append)
    btn_home.click()
    assert "slide_show" in emitted
    
    # Bottom button has "Slide\nshow ▾" text and opens the style menu
    btn_bot = btn_home.btn_bottom
    assert "Slide" in btn_bot.text()
    assert "▾" in btn_bot.text()
    assert btn_bot.menu() is not None



