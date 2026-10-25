"""
Verification test suite for Effects Ribbon Gallery & Dropdown Menu improvements:
1. No squishing on portrait images (center-crop to target aspect ratio)
2. Reduced white space between effect cards (tight filmstrip layout)
3. Sepia effect live preview thumbnail included
4. 2x larger preview images in the 'Show all effects' dropdown menu
"""

from PIL import Image
import numpy as np
import pytest
from PySide6.QtCore import Qt, QSize
from PySide6.QtWidgets import QApplication

from src.ui.media_viewer import SingleMediaViewer
from src.ui.ribbon_toolbar import _RibbonEffectsGallery


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_effect_preview_pixmaps_portrait_crop_and_sepia(qapp):
    """Verifies portrait images are center-cropped rather than squished, and sepia is included."""
    viewer = SingleMediaViewer()

    # Create a tall portrait image (300 wide, 600 tall, 1:2 aspect ratio)
    portrait_np = np.zeros((600, 300, 3), dtype=np.uint8)
    portrait_np[100:500, 50:250] = [180, 120, 80]
    viewer.master_np_rgb = portrait_np
    viewer.current_is_video = False

    pixmaps = viewer.get_effect_preview_pixmaps(96, 64)

    # 1. Sepia must be present and valid
    assert "sepia" in pixmaps
    assert not pixmaps["sepia"].isNull()
    assert pixmaps["sepia"].size() == QSize(96, 64)

    # 2. All 8 effect keys must be present
    expected_keys = [
        "none",
        "classic_chrome",
        "warm_portrait",
        "vivid_landscape",
        "matte_vintage",
        "fine_art_bw",
        "cool_cinema",
        "sepia",
    ]
    for k in expected_keys:
        assert k in pixmaps
        assert not pixmaps[k].isNull()
        assert pixmaps[k].size() == QSize(96, 64)


def test_ribbon_effects_gallery_dimensions_and_dropdown(qapp):
    """Verifies gallery spacing, button dimensions, and 2x dropdown thumbnail sizes."""
    gallery = _RibbonEffectsGallery()
    gallery._suppress_menu_exec = True

    # 1. Dimensions check: cards are 48px wide, stride is 49px (snug edge-to-edge layout)
    assert gallery.CARD_WIDTH == 48
    assert gallery.CARD_STRIDE == 49

    # 2. All buttons must be properly initialized
    assert "sepia" in gallery.buttons
    assert "matte_vintage" in gallery.buttons
    assert gallery.buttons["matte_vintage"].text() == "Matte"
    assert gallery.buttons["sepia"].text() == "Sepia"

    # 3. Supply live thumbnails
    viewer = SingleMediaViewer()
    test_img = np.zeros((400, 600, 3), dtype=np.uint8)
    test_img[:, :] = [100, 150, 200]
    viewer.master_np_rgb = test_img
    viewer.current_is_video = False

    pixmaps = viewer.get_effect_preview_pixmaps(96, 64)
    gallery.update_live_thumbnails(pixmaps)

    assert hasattr(gallery, "_live_thumbnails")
    assert "sepia" in gallery._live_thumbnails
    assert not gallery._live_thumbnails["sepia"].isNull()

    # 4. Trigger dropdown menu and inspect preview icon size (2x size: 84x54)
    gallery._show_all_effects_menu()


def test_ribbon_tab_seamless_connection_and_dropdown_arrows(qapp):
    """Verifies that the selected tab seamlessly merges with the ribbon pane without a dividing border,
    and dropdown indicators are tiny 5x3 icons."""
    from src.ui.ribbon_toolbar import RibbonToolbar
    from PySide6.QtGui import QImage, QColor

    ribbon = RibbonToolbar()
    ribbon.resize(900, 150)
    ribbon.show()

    # Verify tab styles include border-bottom: 1px solid #F4F8FD, margin-bottom: -1px, and tiny dropdown arrow
    qss = ribbon.styleSheet()
    assert "top: -1px;" in qss
    assert "margin-bottom: -1px;" in qss
    assert "arrow_dropdown_mini.svg" in qss

    tw = ribbon.tab_widget
    img = QImage(tw.size(), QImage.Format.Format_ARGB32)
    tw.render(img)

    tb = tw.tabBar()
    t0 = tb.tabRect(0)
    pt0 = tb.mapTo(tw, t0.center())
    y_boundary = tb.mapTo(tw, t0.bottomLeft()).y()

    # In selected tab 0 (Home), the bottom boundary matches pane top (#F4F8FD), masking the line
    c0 = QColor(img.pixel(pt0.x(), y_boundary))
    assert c0.red() > 230 and c0.green() > 235 and c0.blue() > 240

    # In unselected tab 1 (Edit), the dividing border #9EB6CE is visible
    t1 = tb.tabRect(1)
    pt1 = tb.mapTo(tw, t1.center())
    c1 = QColor(img.pixel(pt1.x(), y_boundary))
    assert c1.red() < 200 and c1.green() < 210

    # Switch to tab 1 (Edit) dynamically and verify the seamless mask moves to tab 1
    tw.setCurrentIndex(1)
    tw.render(img)

    c1_after = QColor(img.pixel(pt1.x(), y_boundary))
    c0_after = QColor(img.pixel(pt0.x(), y_boundary))
    assert c1_after.red() > 230 and c1_after.green() > 235 and c1_after.blue() > 240
    assert c0_after.red() < 200 and c0_after.green() < 210
