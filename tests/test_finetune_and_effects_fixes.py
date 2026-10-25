import pytest
import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QFrame, QLabel

from src.imaging.opencv_engine import PhotoAdjustmentRecipe, render_adjusted_image
from src.ui.media_viewer import SingleMediaViewer
from src.ui.right_tag_dock import TagAndCaptionDock, HistogramWidget
from src.ui.ribbon_toolbar import RibbonToolbar, _RibbonEffectsGallery


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_no_black_lines_or_scrollbars(qapp):
    viewer = SingleMediaViewer()
    assert viewer.graphics_view.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    assert viewer.graphics_view.verticalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    assert viewer.graphics_view.frameShape() == QFrame.NoFrame
    assert viewer.media_stack.frameShape() == QFrame.NoFrame


def test_histogram_handles_and_levels(qapp):
    hist = HistogramWidget()
    assert hist.height() in (88, 94)
    assert hist.black_point == 0
    assert hist.white_point == 255

    received = []
    hist.points_changed.connect(lambda bp, wp: received.append((bp, wp)))

    # Set mock data
    arr = np.linspace(0, 255, 1000, dtype=np.uint8)
    hist.set_histogram_data(arr)
    assert any(hist._counts)

    # Click on black point handle (near x=6, y=70)
    press_ev = QMouseEvent(
        QMouseEvent.MouseButtonPress,
        QPoint(6, 70),
        Qt.LeftButton,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    hist.mousePressEvent(press_ev)
    assert hist._drag_handle == "black"

    # Drag to x=50
    move_ev = QMouseEvent(
        QMouseEvent.MouseMove,
        QPoint(50, 70),
        Qt.LeftButton,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    hist.mouseMoveEvent(move_ev)
    assert hist.black_point > 0
    assert len(received) > 0

    # Release
    rel_ev = QMouseEvent(
        QMouseEvent.MouseButtonRelease,
        QPoint(50, 70),
        Qt.LeftButton,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    hist.mouseReleaseEvent(rel_ev)
    assert hist._drag_handle is None

    # Double click resets
    dbl_ev = QMouseEvent(
        QMouseEvent.MouseButtonDblClick,
        QPoint(20, 20),
        Qt.LeftButton,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    hist.mouseDoubleClickEvent(dbl_ev)
    assert hist.black_point == 0
    assert hist.white_point == 255


def test_levels_processing_in_opencv_engine():
    # Test level stretching
    base = np.array([[[50, 50, 50], [100, 100, 100], [200, 200, 200]]], dtype=np.uint8)
    rec_default = PhotoAdjustmentRecipe()
    out_default = render_adjusted_image(base, rec_default)
    assert np.array_equal(out_default, base)

    # Black point at 60 (clips 50 to 0)
    rec_bp = PhotoAdjustmentRecipe(black_point=60.0)
    out_bp = render_adjusted_image(base, rec_bp)
    assert out_bp[0, 0, 0] == 0
    assert out_bp[0, 1, 0] > 0


def test_fine_tune_panel_labels_reset_and_no_analyze(qapp):
    dock = TagAndCaptionDock()
    dock.set_mode("fine_tune")

    # 1. Analyze button must NOT exist
    assert not hasattr(dock, "btn_analyze")
    all_buttons = dock.findChildren(type(dock.btn_close_dock))
    analyze_btns = [b for b in all_buttons if b.text() == "Analyze"]
    assert len(analyze_btns) == 0

    # 2. Reset to default and Hold to Compare at bottom
    assert hasattr(dock, "btn_reset_default")
    assert hasattr(dock, "btn_hold_compare")
    assert dock.btn_reset_default.text() == "Reset to default"
    assert dock.btn_hold_compare.text() == "Hold to Compare (Original)"

    # 3. Change sliders, test value readouts and reset
    dock.ft_slider_brightness.setValue(25)
    dock.ft_slider_contrast.setValue(-15)
    dock.ft_slider_straighten.setValue(5)
    dock.histogram_widget.black_point = 30

    reset_emitted = []
    dock.fine_tune_reset_requested.connect(lambda: reset_emitted.append(True))
    dock.btn_reset_default.click()

    assert dock.ft_slider_brightness.value() == 0
    assert dock.ft_slider_contrast.value() == 0
    assert dock.ft_slider_straighten.value() == 0
    assert dock.histogram_widget.black_point == 0
    assert dock.histogram_widget.white_point == 255
    assert len(reset_emitted) == 1


def test_hold_to_compare_instant_and_no_change_when_default(qapp):
    viewer = SingleMediaViewer()
    img = np.full((50, 50, 3), 128, dtype=np.uint8)
    viewer.master_np_rgb = img.copy()
    viewer.master_pil_image = Image.fromarray(img)
    viewer.current_pil_image = viewer.master_pil_image.copy()
    viewer._render_current_recipe()

    # When recipe is default, original pixmap equals adjusted pixmap
    pm_before = viewer.pixmap_item.pixmap()
    viewer.set_compare_mode(True)
    pm_during = viewer.pixmap_item.pixmap()
    assert pm_before.toImage() == pm_during.toImage()

    viewer.set_compare_mode(False)
    pm_after = viewer.pixmap_item.pixmap()
    assert pm_before.toImage() == pm_after.toImage()

    # When recipe is changed, compare mode shows original
    viewer.current_recipe.brightness = 50.0
    viewer._render_current_recipe()
    pm_edited = viewer.pixmap_item.pixmap()
    assert pm_edited.toImage() != pm_before.toImage()

    viewer.set_compare_mode(True)
    pm_compared = viewer.pixmap_item.pixmap()
    assert pm_compared.toImage() == pm_before.toImage()

    viewer.set_compare_mode(False)
    assert viewer.pixmap_item.pixmap().toImage() == pm_edited.toImage()


def test_ribbon_effects_gallery_scrolling_and_menu(qapp):
    gallery = _RibbonEffectsGallery()
    gallery.resize(180, 60)
    gallery.show()

    # Cards must have fixed size (48x56)
    for eff_id, btn in gallery.buttons.items():
        assert btn.width() == 48
        assert btn.height() == 56

    # Navigation buttons exist
    assert hasattr(gallery, "btn_up")
    assert hasattr(gallery, "btn_dn")
    assert hasattr(gallery, "btn_more")

    # Down scroll
    initial_idx = gallery._scroll_index
    gallery.btn_dn.click()
    assert gallery._scroll_index >= initial_idx

    # Up scroll
    if gallery._scroll_index > 0:
        gallery.btn_up.click()
        assert gallery._scroll_index < initial_idx + 1

    # Menu trigger
    selected_effects = []
    gallery.effect_selected.connect(lambda eff: selected_effects.append(eff))
    gallery._select_from_menu("sepia")
    assert "sepia" in selected_effects
    assert gallery.buttons["sepia"].isChecked()


def test_effects_gallery_size_hint_and_expansion(qapp):
    from src.ui.ribbon_toolbar import _RibbonGroupBox, _RibbonEffectsGallery
    from PySide6.QtWidgets import QWidget, QHBoxLayout
    from pathlib import Path

    gallery = _RibbonEffectsGallery()
    # Size hint must be wide enough to show all 8 effects plus controls (>= 400px)
    assert gallery.sizeHint().width() >= 400
    # Minimum size hint must allow compressed ribbon (< 100px)
    assert gallery.minimumSizeHint().width() <= 80

    # Test in layout with RibbonGroupBox
    win = QWidget()
    h = QHBoxLayout(win)
    grp = _RibbonGroupBox("Effects")
    gal = _RibbonEffectsGallery(grp)
    grp.content_layout.addWidget(gal)
    h.addWidget(grp)
    h.addStretch(1)
    win.resize(1200, 100)
    win.show()
    qapp.processEvents()

    # Under full window width, gallery must expand to show all cards
    assert gal.width() >= 400
    assert gal.viewport_widget.width() >= 390


def test_recipe_copy_preserves_black_and_white_points():
    rec = PhotoAdjustmentRecipe(black_point=45.0, white_point=215.0, brightness=20.0)
    c = rec.copy()
    assert c.black_point == 45.0
    assert c.white_point == 215.0
    assert c.brightness == 20.0


def test_editing_engine_specification_doc_exists():
    from pathlib import Path
    doc_path = Path("editing_engine.md")
    assert doc_path.exists()
    content = doc_path.read_text(encoding="utf-8")
    assert "Red Eye Reduction" in content
    assert "Digital Noise Reduction" in content
    assert "Clarity" in content
    assert "Photographic Preset Effects Engine" in content
    assert "Inscribed-Rectangle Auto-Straightening" in content
    assert "Classic Chrome" in content
    assert "Warm Portrait" in content
    assert "Vivid Landscape" in content
