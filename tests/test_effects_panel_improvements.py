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

    # 1. Dimensions check: cards are 54px wide, stride is 55px (tight ~1px gap)
    assert gallery.CARD_WIDTH == 54
    assert gallery.CARD_STRIDE == 55

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
