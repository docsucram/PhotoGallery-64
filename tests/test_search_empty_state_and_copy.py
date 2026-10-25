"""
Unit and integration tests for:
1. WLPG empty search state ("Try searching all photos and videos" and "No items match your current search.")
2. EXIF DateTimeOriginal taking priority over IFD0 DateTime modification date
3. Edit Ribbon Manage group snug fit (elimination of gap after 'Make a copy')
4. Enhanced 'Make a copy' metadata cloning and viewer continuity
"""

import sys
from datetime import datetime
from pathlib import Path
import pytest
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QMainWindow

from src.ui.grid_view import EmptyStateWidget, ThumbnailGridView
from src.ui.ribbon_toolbar import RibbonToolbar
from src.workers.scanner_thumb_worker import extract_media_metadata, _parse_iso_or_exif_datetime
from PIL import Image, ExifTags


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_empty_state_widget_logic(qapp):
    """Verifies that EmptyStateWidget displays 'Try searching all photos and videos' when scoped with no results."""
    w = EmptyStateWidget()

    # 1. Has items -> hidden
    w.set_state(is_empty=False, is_search=True, is_scoped=True)
    assert not w.isVisible()

    # 2. Empty with scoped search -> shows link + no match label
    w.set_state(is_empty=True, is_search=True, is_scoped=True)
    assert w.isVisible()
    assert w.btn_try_all.isVisible()
    assert w.lbl_message.text() == "No items match your current search."

    # Verify clicking link emits signal
    emitted = []
    w.search_all_clicked.connect(lambda: emitted.append(True))
    w.btn_try_all.click()
    assert len(emitted) == 1

    # 3. Empty with global search (not scoped) -> link hidden, label visible
    w.set_state(is_empty=True, is_search=True, is_scoped=False)
    assert w.isVisible()
    assert not w.btn_try_all.isVisible()
    assert w.lbl_message.text() == "No items match your current search."

    # 4. Empty browse (not a search) -> link hidden, generic empty text
    w.set_state(is_empty=True, is_search=False, is_scoped=True)
    assert w.isVisible()
    assert not w.btn_try_all.isVisible()
    assert "There are no pictures or videos" in w.lbl_message.text()


def test_exif_datetime_original_priority(tmp_path: Path):
    """
    Verifies that when a file has a DateTime in IFD0 (e.g. edited in 2011)
    and a DateTimeOriginal in SubIFD (e.g. taken in 2010),
    ScannerThumbWorker prioritizes DateTimeOriginal (2010).
    """
    img_path = tmp_path / "fuji_test.jpg"
    im = Image.new("RGB", (100, 100), color=(180, 200, 220))

    exif = im.getexif()
    # Tag 306 is DateTime in IFD0 (e.g. modified 2011:05:04 22:52:04)
    exif[306] = "2011:05:04 22:52:04"

    # Exif SubIFD has DateTimeOriginal (taken 2010:01:07 09:30:16)
    sub_ifd = exif.get_ifd(ExifTags.IFD.Exif)
    sub_ifd[36867] = "2010:01:07 09:30:16"  # DateTimeOriginal
    sub_ifd[36868] = "2010:01:07 09:30:16"  # DateTimeDigitized

    im.save(str(img_path), exif=exif)

    meta = extract_media_metadata(img_path, is_video=False)

    assert meta["date_taken"] is not None
    dt = datetime.fromtimestamp(meta["date_taken"])
    # Must be 2010-01-07 (Date Taken), NOT 2011-05-04 (DateTime modification date)
    assert dt.year == 2010
    assert dt.month == 1
    assert dt.day == 7


def test_ribbon_edit_tab_manage_no_gap(qapp):
    """Verifies that in single image viewer mode, the Edit tab Manage group has no excess gap after 'Make a copy'."""
    win = QMainWindow()
    ribbon = RibbonToolbar()
    win.setCentralWidget(ribbon)
    win.resize(1280, 200)
    win.show()

    # Initial Grid mode
    qapp.processEvents()
    initial_w = ribbon.grp_edit_manage.width()

    # Switch to Single Viewer mode
    ribbon.set_active_mode("viewer")
    qapp.processEvents()

    grp = ribbon.grp_edit_manage
    assert grp.isVisible()
    assert not ribbon.col_grid_manage.isVisible()
    assert ribbon.col_common_manage.isVisible()
    assert ribbon.btn_edit_make_copy.isVisible()

    # Total width of grp_edit_manage should fit snuggly around col_common_manage + btn_edit_make_copy
    content_w = grp.content_widget.width()
    expected_w = ribbon.col_common_manage.width() + ribbon.btn_edit_make_copy.width() + 10
    # Must not have a huge 30-60px void
    assert content_w <= expected_w + 5
    win.close()
