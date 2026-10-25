"""
Test suite verifying:
1. Fully-featured Print Pictures dialog (templates, paper aspect, page calculation, fit-to-frame).
2. QAT customization without 'Order prints' and with toggleable Print button.
3. File menu Print item and Ctrl+P shortcut.
4. Zoom in (+) and Zoom out (-) keyboard shortcuts functioning without ambiguity.
"""

from pathlib import Path
import pytest
import numpy as np
from PySide6.QtCore import Qt, QSize, QEvent
from PySide6.QtGui import QImage, QColor, QKeyEvent, QShortcut, QKeySequence
from PySide6.QtWidgets import QApplication

from src.ui.print_dialog import PrintPicturesDialog
from src.ui.ribbon_toolbar import RibbonToolbar
from src.ui.media_viewer import SingleMediaViewer


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_print_pictures_dialog_templates_and_pages(qapp):
    """Verifies PrintPicturesDialog layout templates, pagination, copies, and frame fitting."""
    # Create 3 test images
    images = []
    for i in range(3):
        qimg = QImage(300, 200, QImage.Format.Format_RGB888)
        qimg.fill(QColor(i * 50, 100, 150))
        images.append((f"photo_{i}.jpg", qimg))

    dlg = PrintPicturesDialog(images)

    # 1. Dialog title and window icon
    assert dlg.windowTitle() == "Print Pictures"
    assert not dlg.windowIcon().isNull()

    # 2. Template options present
    expected_layouts = ["full_page", "two_5x7", "one_8x10", "two_4x6", "four_wallet", "contact_sheet"]
    for l_id in expected_layouts:
        assert l_id in dlg.template_cards

    # 3. Full page: 3 images, 1 per page = 3 pages
    dlg._select_layout("full_page")
    assert dlg.active_layout_id == "full_page"
    assert dlg.current_page_index == 0
    assert "1 of 3 pages" in dlg.lbl_page_counter.text()

    # Navigate forward and backward
    assert dlg.btn_next_page.isEnabled()
    dlg._next_page()
    assert dlg.current_page_index == 1
    assert "2 of 3 pages" in dlg.lbl_page_counter.text()
    assert dlg.btn_prev_page.isEnabled()
    dlg._prev_page()
    assert dlg.current_page_index == 0

    # 4. Two 5x7: 3 images, 2 per page = 2 pages
    dlg._select_layout("two_5x7")
    assert "1 of 2 pages" in dlg.lbl_page_counter.text()

    # 5. Four wallet: 3 images, 4 per page = 1 page
    dlg._select_layout("four_wallet")
    assert "1 of 1 page" in dlg.lbl_page_counter.text()

    # 6. Contact sheet: 3 images, 35 per page = 1 page
    dlg._select_layout("contact_sheet")
    assert "1 of 1 page" in dlg.lbl_page_counter.text()

    # 7. Copies adjustment: 2 copies of each picture = 6 items
    dlg.spn_copies.setValue(2)
    assert dlg.copies_per_picture == 2
    # In two_5x7, 6 items with 2 per page = 3 pages
    dlg._select_layout("two_5x7")
    assert "3 pages" in dlg.lbl_page_counter.text()

    # 8. Fit picture to frame toggle
    assert dlg.chk_fit.isChecked()
    dlg.chk_fit.setChecked(False)
    assert not dlg.fit_to_frame
    dlg.chk_fit.setChecked(True)
    assert dlg.fit_to_frame

    # 9. Paper size aspect ratios
    dlg.cmb_paper_size.setCurrentText("Letter")
    assert dlg.get_current_paper_aspect() == (8.5, 11.0)
    dlg.cmb_paper_size.setCurrentText("A4")
    assert dlg.get_current_paper_aspect() == (210.0, 297.0)
    dlg.cmb_paper_size.setCurrentText("4 x 6 in")
    assert dlg.get_current_paper_aspect() == (4.0, 6.0)


def test_qat_order_prints_removed_and_print_button_present(qapp):
    """Verifies that 'Order prints' is removed from QAT, and Print button is available."""
    ribbon = RibbonToolbar()

    # 1. 'Order prints' must NOT be in the QAT menu
    assert not hasattr(ribbon, "act_qat_order")

    # 2. 'Print' QAT button exists
    assert hasattr(ribbon, "btn_print")
    assert hasattr(ribbon, "act_qat_print")
    assert ribbon.btn_print.isHidden()

    # Toggling act_qat_print shows/hides btn_print
    ribbon.act_qat_print.setChecked(True)
    assert not ribbon.btn_print.isHidden()
    ribbon.act_qat_print.setChecked(False)
    assert ribbon.btn_print.isHidden()

    # 3. File menu has Print action
    file_actions = [a.text() for a in ribbon.btn_file.menu().actions()]
    assert any("Print" in text for text in file_actions)


def test_zoom_shortcuts_no_ambiguity_and_zoom_out_works(qapp, tmp_path):
    """Verifies that '-' and '+' zoom shortcuts operate without QShortcutMap ambiguity."""
    viewer = SingleMediaViewer()
    dummy = np.zeros((400, 600, 3), dtype=np.uint8)
    dummy[:] = [120, 140, 160]
    viewer.master_np_rgb = dummy
    viewer.current_is_video = False
    viewer.show()

    # Initial zoom factor
    initial_zoom = viewer.graphics_view.zoom_level
    assert initial_zoom > 0.0

    # 1. Send '+' KeyPress event -> Zoom In
    ev_plus = QKeyEvent(QEvent.Type.KeyPress, Qt.Key_Plus, Qt.NoModifier, "+")
    handled = viewer.eventFilter(viewer, ev_plus)
    assert handled is True
    assert viewer.graphics_view.zoom_level > initial_zoom

    zoomed_in = viewer.graphics_view.zoom_level

    # 2. Send '-' KeyPress event -> Zoom Out
    ev_minus = QKeyEvent(QEvent.Type.KeyPress, Qt.Key_Minus, Qt.NoModifier, "-")
    handled_minus = viewer.eventFilter(viewer, ev_minus)
    assert handled_minus is True
    assert viewer.graphics_view.zoom_level < zoomed_in

    # 3. Verify MainWindow registered shortcuts are distinct without duplicates
    from src.db.connection import DualDatabaseEngine
    from src.ui.main_window import PhotoGalleryMainWindow

    db_dir = tmp_path / "db_zoom"
    db_dir.mkdir(parents=True, exist_ok=True)
    engine = DualDatabaseEngine(
        catalog_path=db_dir / "catalog.db",
        thumbnails_path=db_dir / "thumbnails.db",
    )

    mw = PhotoGalleryMainWindow(engine=engine, start_xmp_daemon=False)
    seq_strings = [sc.key().toString() for sc in mw._zoom_shortcuts]

    # There should be NO duplicate '-' sequences in _zoom_shortcuts (which caused the ambiguous shortcut bug)
    minus_count = seq_strings.count("-")
    assert minus_count == 1, f"Expected exactly 1 '-' shortcut sequence, found {minus_count}: {seq_strings}"
    assert "Ctrl+P" in seq_strings
    mw.close()
