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
    mw.ribbon.qat_container.isVisible.return_value = True
    mw.ribbon._ribbon_collapsed = False

    mw.resizeDocks = MagicMock()

    # 1. Toggle ON immersive mode
    PhotoGalleryMainWindow.toggle_immersive_viewer_mode(mw)
    assert getattr(mw, "_is_immersive_mode") is True
    mw.left_nav_dock.hide.assert_called_once()
    mw.right_tag_dock.hide.assert_called_once()
    mw.media_viewer.fix_pane.hide.assert_called_once()
    mock_status_bar.hide.assert_called_once()
    mw.ribbon.qat_container.hide.assert_called_once()
    mw.ribbon.set_ribbon_collapsed.assert_called_with(True)

    # 2. Toggle OFF immersive mode (restores previously visible components)
    PhotoGalleryMainWindow.toggle_immersive_viewer_mode(mw)
    assert getattr(mw, "_is_immersive_mode") is False
    mw.left_nav_dock.show.assert_called_once()
    mw.right_tag_dock.show.assert_called_once()
    mw.media_viewer.fix_pane.show.assert_called_once()
    mock_status_bar.show.assert_called_once()
    mw.ribbon.qat_container.show.assert_called_once()
    mw.ribbon.set_ribbon_collapsed.assert_called_with(False)

