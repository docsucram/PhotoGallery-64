import os
import shutil
import tempfile
from pathlib import Path
from PIL import Image

import pytest
from PySide6.QtWidgets import QApplication, QLineEdit, QMenu
from PySide6.QtCore import Qt, QPoint

from src.db.connection import DualDatabaseEngine
from src.ui.main_window import PhotoGalleryMainWindow


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def test_app_env(qapp: QApplication, tmp_path: Path):
    photos_dir = tmp_path / "Pictures"
    photos_dir.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (300, 200), (100, 150, 200))
    img.save(photos_dir / "Photo_01.jpg", quality=90)

    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    engine = DualDatabaseEngine(
        catalog_path=db_dir / "catalog.db",
        thumbnails_path=db_dir / "thumbnails.db",
    )
    window = PhotoGalleryMainWindow(engine=engine, start_xmp_daemon=False)
    window.scan_folder(photos_dir, include_subfolders=True, blocking=True)
    window.show()
    yield {"window": window, "photos_dir": photos_dir, "engine": engine}
    window.close()
    engine.close_thread_connections()


def test_revert_to_original_greying_out(qapp: QApplication, test_app_env: dict):
    win: PhotoGalleryMainWindow = test_app_env["window"]
    records = win.catalog_model._records
    assert len(records) > 0
    mid = records[0].id

    # 1. In Grid View with unedited item selected: Revert button must be greyed out
    win.show_grid_view()
    win.grid_view.select_media_id(mid)
    win.ribbon.tab_widget.setCurrentWidget(win.ribbon.tab_edit)
    assert not win.ribbon.btn_edit_revert.isEnabled(), "Revert should be disabled in Grid View for unedited photos"

    # 2. In Single Viewer for unedited photo: Revert button must be greyed out
    win.open_single_viewer(mid)
    assert not win.ribbon.btn_edit_revert.isEnabled(), "Revert should be disabled in Single View for unedited photos"
    assert not win.media_viewer.btn_revert.isEnabled(), "Fix pane revert button must be disabled"

    # 3. Apply an edit (e.g. black and white effect): Revert button must become enabled
    win.media_viewer.apply_preset_effect("bw")
    assert win.ribbon.btn_edit_revert.isEnabled(), "Revert must be enabled after applying an edit"
    assert win.media_viewer.btn_revert.isEnabled(), "Fix pane revert button must be enabled after applying an edit"

    # 4. Revert: Revert button must become disabled again
    win.media_viewer.revert_to_original()
    assert not win.ribbon.btn_edit_revert.isEnabled(), "Revert must be disabled after reverting"
    assert not win.media_viewer.btn_revert.isEnabled(), "Fix pane revert must be disabled after reverting"


def test_zoom_keyboard_shortcuts(qapp: QApplication, test_app_env: dict, monkeypatch: pytest.MonkeyPatch):
    win: PhotoGalleryMainWindow = test_app_env["window"]
    win.show_grid_view()
    win.slider_status_zoom.setValue(150)

    # Initial zoom slider value
    initial_zoom = win.slider_status_zoom.value()

    # Test zoom in (+)
    win._on_zoom_in_shortcut()
    assert win.slider_status_zoom.value() > initial_zoom

    # Test zoom out (-)
    win._on_zoom_out_shortcut()
    assert win.slider_status_zoom.value() == initial_zoom

    # Focus guard: typing in QLineEdit should not trigger zoom shortcut
    le = QLineEdit(win)
    monkeypatch.setattr(QApplication, "focusWidget", lambda: le)
    win._on_zoom_in_shortcut()
    # Should not have changed because QLineEdit had focus
    assert win.slider_status_zoom.value() == initial_zoom


def test_status_bar_item_count_isolation(qapp: QApplication, test_app_env: dict):
    win: PhotoGalleryMainWindow = test_app_env["window"]

    # Verify initial item count
    win.lbl_status_counts.setText("519 items, 1 selected")
    assert win.lbl_status_counts.text() == "519 items, 1 selected"
    assert win.lbl_status_counts.isVisible()
    assert win.worker_monitor_widget.isVisible()

    # Trigger a sync message via statusBar().showMessage()
    win.statusBar().showMessage("✓ Synced '2019_01 Wisley Lego': ~11", 4000)

    # Item count and worker monitor MUST REMAIN VISIBLE AND SEPARATE
    assert win.lbl_status_counts.text() == "519 items, 1 selected"
    assert win.lbl_status_counts.isVisible()
    assert win.worker_monitor_widget.isVisible()

    # Message must be placed into lbl_sync_status without overlapping
    assert "Synced" in win.lbl_sync_status.text()
    assert win.sep_status_msg.isVisible()

    # Clear message
    win.statusBar().clearMessage()
    assert win.lbl_sync_status.text() == ""
    assert not win.sep_status_msg.isVisible()
    assert win.lbl_status_counts.isVisible()


def test_folder_context_menu_open_folder_location(qapp: QApplication, test_app_env: dict, monkeypatch: pytest.MonkeyPatch):
    win: PhotoGalleryMainWindow = test_app_env["window"]
    dock = win.left_nav_dock
    assert hasattr(dock, "_open_folder_location")

    # 1. Test direct invocation of _open_folder_location with subprocess.Popen mock
    opened_commands = []
    import subprocess
    monkeypatch.setattr(subprocess, "Popen", lambda cmd: opened_commands.append(cmd))
    dock._open_folder_location(str(test_app_env["photos_dir"]))
    assert len(opened_commands) == 1
    assert "explorer.exe" in opened_commands[0]
    assert str(test_app_env["photos_dir"]) in opened_commands[0]

    # 2. Test context menu on folder item has "Open folder location"
    folder_item = next(
        (itm for itm in dock._folder_items.values() if os.path.normcase(itm.toolTip(0)) == os.path.normcase(str(test_app_env["photos_dir"]))),
        None
    )
    assert folder_item is not None

    menu = dock._build_context_menu(folder_item)
    actions = [a.text() for a in menu.actions()]
    assert "Open folder location" in actions
