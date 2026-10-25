"""
Phase 13 Tests: Video Right-Click Context Menu, White Background, and Autoplay on Open.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtWidgets import QApplication

from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository
from src.ui.grid_model import CatalogListModel
from src.ui.grid_view import MediaCardDelegate, ThumbnailGridView
from src.ui.main_window import PhotoGalleryMainWindow
from src.ui.media_viewer import SingleMediaViewer


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_grid_view_video_vs_photo_context_menu(qapp: QApplication, tmp_path: Path) -> None:
    """Verify video card context menu differs from photo card context menu."""
    engine = DualDatabaseEngine(tmp_path / "catalog.db", tmp_path / "thumbs.db")
    repo = CatalogRepository(engine)

    fid = repo.upsert_folder("D:/Media", "Media", None, 1700000000)
    ids = repo.bulk_upsert_media(
        [
            {
                "folder_id": fid,
                "filename": "Photo.jpg",
                "ext": ".jpg",
                "size": 1000,
                "mtime": 1700000000,
                "is_video": 0,
            },
            {
                "folder_id": fid,
                "filename": "Clip.mp4",
                "ext": ".mp4",
                "size": 5000000,
                "mtime": 1700000000,
                "is_video": 1,
            },
        ]
    )
    photo_id, video_id = ids[0], ids[1]

    win = PhotoGalleryMainWindow(engine, start_xmp_daemon=False)
    win.rebuild_and_run_query()

    # Suppress modal menu.exec for testing
    win.grid_view._suppress_menu_exec = True

    # 1. Video Context Menu
    video_menu = win.grid_view._show_card_context_menu(QPoint(100, 100), video_id, is_video=True)
    video_action_texts = [act.text() for act in video_menu.actions() if not act.isSeparator()]

    assert "Play" in video_action_texts
    assert "View file" not in video_action_texts
    assert "Rotate right" not in video_action_texts
    assert "Rotate left" not in video_action_texts
    assert "Set as desktop background" not in video_action_texts
    assert "Resize..." not in video_action_texts
    assert "Open with" in video_action_texts
    assert "Copy" in video_action_texts
    assert "Delete" in video_action_texts
    assert "Rename..." in video_action_texts
    assert "Properties" in video_action_texts

    # Submenu of Open with for video should NOT include Paint
    open_with_actions = [act.text() for act in video_menu._open_with_menu.actions()]
    assert "System default player" in open_with_actions
    assert "Paint" not in open_with_actions
    assert "Choose another app..." in open_with_actions

    # 2. Photo Context Menu
    photo_menu = win.grid_view._show_card_context_menu(QPoint(100, 100), photo_id, is_video=False)
    photo_action_texts = [act.text() for act in photo_menu.actions() if not act.isSeparator()]

    assert "View file" in photo_action_texts
    assert "Play" not in photo_action_texts
    assert "Rotate right" in photo_action_texts
    assert "Rotate left" in photo_action_texts
    assert "Set as desktop background" in photo_action_texts
    assert "Resize..." in photo_action_texts

    open_with_photo = [act.text() for act in photo_menu._open_with_menu.actions()]
    assert "Paint" in open_with_photo

    win.close()


def test_single_media_viewer_video_white_background_and_autoplay(
    qapp: QApplication, tmp_path: Path
) -> None:
    """Verify video player in default theme has white background and autoplays on open."""
    viewer = SingleMediaViewer()
    viewer._suppress_menu_exec = True

    # 1. Background color of video container & widget must be white
    assert "background-color: #FFFFFF" in viewer.video_container.styleSheet()
    assert viewer.video_widget.palette().color(QPalette.Window).name().lower() == "#ffffff"

    # 2. Create dummy test video file
    dummy_vid = tmp_path / "sample.mp4"
    dummy_vid.write_bytes(b"\x00" * 1024)

    # 3. Load video -> must trigger autoplay
    ok = viewer.load_media(media_id=99, file_path=dummy_vid, is_video=True)
    assert ok is True
    assert viewer.current_is_video is True
    assert viewer.media_stack.currentIndex() == 1
    assert viewer.btn_play_pause.text() == "⏸ Pause"

    # 4. Context Menu in single video view
    ctx_menu = viewer._show_video_context_menu(QPoint(50, 50))
    action_texts = [act.text() for act in ctx_menu.actions() if not act.isSeparator()]
    assert any("Pause" in t or "Play" in t for t in action_texts)
    assert "Stop" in action_texts
    assert any("Mute" in t or "Unmute" in t for t in action_texts)
    assert "Open with" in action_texts
    assert "Open file location" in action_texts
    assert "Properties" in action_texts

    # 5. Stop video
    viewer.stop_video_playback()
    assert viewer.media_player.playbackState() != QMediaPlayer.PlayingState
    assert viewer.btn_play_pause.text() == "▶ Play"

    viewer.close()
