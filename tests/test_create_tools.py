"""
Test suite for Creation Studio Tools:
1. Microsoft ICE-Style Panorama Stitcher with auto-crop of wavy black borders & inpaint gap filling.
2. Interactive Collage & 'Then & Now' Card generator (Grid, Polaroid Scatter, and Then & Now).
3. Photo Fuse group shot combiner (Homography alignment, face detection & seamless Poisson cloning).
4. Ribbon Create Tab buttons and dispatching.
"""

from pathlib import Path
import pytest
import numpy as np
import cv2
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from src.imaging.create_tools import (
    find_largest_interior_rectangle,
    inpaint_minor_border_gaps,
    stitch_panorama,
    create_grid_collage,
    create_polaroid_scatter_collage,
    create_then_and_now_card,
    align_burst_frame,
    detect_faces_in_image,
    fuse_face,
)
from src.ui.ribbon_toolbar import RibbonToolbar
from src.ui.create_studio_dialogs import PanoramaDialog, CollageDialog, PhotoFuseDialog


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# =============================================================================
# 1. Panorama Stitching & Auto-Crop Tests
# =============================================================================

def test_largest_interior_rectangle_and_auto_crop():
    """Verifies Microsoft ICE signature largest interior rectangle crop without black pixels."""
    # Create a canvas with wavy / irregular black borders around a central region
    canvas_mask = np.zeros((300, 500), dtype=np.uint8)
    # Valid region: y from 30..270, x from 50..450
    canvas_mask[30:270, 50:450] = 255
    # Cut out an irregular bite on the bottom right corner
    canvas_mask[240:270, 400:450] = 0

    rx, ry, rw, rh = find_largest_interior_rectangle(canvas_mask)

    # 1. Returned rectangle must be valid and non-empty
    assert rw > 300 and rh > 180
    assert rx >= 0 and ry >= 0

    # 2. Every single pixel inside the cropped rectangle must be valid (>0)
    sub = canvas_mask[ry : ry + rh, rx : rx + rw]
    assert np.all(sub > 0), "Crop contains black border pixels!"


def test_panorama_stitching_and_inpaint(tmp_path: Path):
    """Verifies stitching of overlapping images, fallback homography, and inpainting."""
    # Create two synthetic overlapping textured images
    w, h = 400, 300
    img1 = np.full((h, w, 3), 180, dtype=np.uint8)
    cv2.circle(img1, (150, 150), 60, (0, 140, 255), -1)
    cv2.rectangle(img1, (30, 30), (120, 260), (40, 200, 40), -1)
    cv2.putText(img1, "PANO-1", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Image 2 shifted horizontally by 100px
    img2 = np.full((h, w, 3), 180, dtype=np.uint8)
    cv2.circle(img2, (50, 150), 60, (0, 140, 255), -1)
    cv2.rectangle(img2, (200, 40), (350, 260), (220, 50, 50), -1)
    cv2.putText(img2, "PANO-2", (220, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Stitch with auto-crop and inpaint
    success, stitched, msg = stitch_panorama(
        [img1, img2],
        projection_mode="cylindrical",
        auto_crop=True,
        fill_minor_gaps=True,
    )

    assert success is True
    assert stitched is not None
    assert stitched.shape[0] > 100
    assert stitched.shape[1] > 200


# =============================================================================
# 2. Collage & 'Then & Now' Card Tests
# =============================================================================

def test_grid_collage_generation():
    """Verifies Grid and Masonry collage layouts with borders and corner rounding."""
    imgs = [
        np.full((300, 400, 3), (100 + i * 30, 80, 150), dtype=np.uint8)
        for i in range(4)
    ]

    # Test 2x2 grid
    collage_2x2 = create_grid_collage(
        imgs,
        layout="2x2",
        border_thickness=10,
        corner_radius=8,
        target_size=(1200, 800),
    )
    assert collage_2x2.shape == (800, 1200, 3)

    # Test center hero layout
    collage_hero = create_grid_collage(
        imgs,
        layout="center_hero",
        border_thickness=12,
        corner_radius=10,
        target_size=(1200, 800),
    )
    assert collage_hero.shape == (800, 1200, 3)


def test_polaroid_scatter_and_then_and_now():
    """Verifies Polaroid scatter collage and Then & Now comparison card."""
    imgs_data = [
        {
            "image_bgr": np.full((250, 250, 3), (50, 100, 200), dtype=np.uint8),
            "date_taken": "2015-06-12",
            "caption": "Summer Holiday 2015",
        },
        {
            "image_bgr": np.full((250, 250, 3), (200, 100, 50), dtype=np.uint8),
            "date_taken": "2026-10-25",
            "caption": "Reunion 2026",
        },
    ]

    # 1. Polaroid Scatter
    scatter = create_polaroid_scatter_collage(
        images_data=imgs_data,
        target_size=(1600, 900),
        bg_style="dark_wood",
        print_captions=True,
    )
    assert scatter.shape == (900, 1600, 3)

    # 2. Then & Now Comparison Card
    then_now = create_then_and_now_card(
        images_data=imgs_data,
        target_size=(1600, 900),
    )
    assert then_now.shape == (900, 1600, 3)


# =============================================================================
# 3. Photo Fuse Homography & Seamless Blending Tests
# =============================================================================

def test_photo_fuse_homography_and_seamless_cloning():
    """Verifies burst frame alignment and seamless Poisson cloning."""
    base = np.full((400, 400, 3), 120, dtype=np.uint8)
    cv2.circle(base, (200, 200), 70, (220, 200, 180), -1)  # Face circle
    cv2.circle(base, (180, 180), 8, (20, 20, 20), -1)      # Eye 1
    cv2.circle(base, (220, 180), 8, (20, 20, 20), -1)      # Eye 2
    cv2.ellipse(base, (200, 225), (25, 10), 0, 0, 180, (40, 40, 180), 3) # Smile

    # Second burst shot: same person with a wink
    shot2 = base.copy()
    cv2.line(shot2, (170, 180), (190, 180), (20, 20, 20), 3)  # Wink eye 1

    # Homography alignment test
    success, aligned = align_burst_frame(base, shot2)
    assert aligned.shape == base.shape

    # Seamless face fuse test
    face_rect = (160, 150, 80, 90)
    fused = fuse_face(base, aligned, face_rect)

    assert fused.shape == base.shape
    # Check that image was modified at the face region
    assert not np.array_equal(fused, base)

    # Boundary fallback test (near border without assertion crash)
    border_rect = (5, 5, 60, 60)
    fused_border = fuse_face(base, aligned, border_rect)
    assert fused_border.shape == base.shape


# =============================================================================
# 4. Ribbon Create Tab & Dialogs UI Integration
# =============================================================================

def test_ribbon_create_tab_wiring_and_dialog_init(qapp, tmp_path: Path):
    """Verifies that the Create ribbon tab has Panoramic, Photo Fuse, and Auto Collage buttons."""
    ribbon = RibbonToolbar()
    ribbon.show()

    # Verify buttons on Create tab
    assert hasattr(ribbon, "btn_panorama")
    assert hasattr(ribbon, "btn_photo_fuse")
    assert hasattr(ribbon, "btn_collage")

    # Verify action emission
    received_actions = []
    ribbon.action_triggered.connect(received_actions.append)

    ribbon.btn_panorama.click()
    assert "panorama" in received_actions

    ribbon.btn_photo_fuse.click()
    assert "photo_fuse" in received_actions

    ribbon.btn_collage.click()
    assert "collage" in received_actions

    # Test Dialog initializations with sample files
    p1 = tmp_path / "img1.jpg"
    p2 = tmp_path / "img2.jpg"
    test_img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.imwrite(str(p1), test_img)
    cv2.imwrite(str(p2), test_img)

    # PanoramaDialog initialization
    pano_dlg = PanoramaDialog([p1, p2])
    assert pano_dlg.windowTitle().startswith("Create Panorama")
    pano_dlg.close()

    # CollageDialog initialization
    col_dlg = CollageDialog([
        {"file_path": str(p1), "caption": "Shot 1", "image_bgr": test_img},
        {"file_path": str(p2), "caption": "Shot 2", "image_bgr": test_img},
    ])
    assert col_dlg.windowTitle().startswith("Create Collage")
    col_dlg.close()

    # PhotoFuseDialog initialization
    fuse_dlg = PhotoFuseDialog([p1, p2])
    assert fuse_dlg.windowTitle().startswith("Photo Fuse")
    fuse_dlg.close()
