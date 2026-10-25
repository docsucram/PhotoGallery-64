"""
Test suite for Creative & Markup Tools:
1. Google Photos-Style 'Color Pop' mask estimation, bilateral brush snapping, and B&W background contrast.
2. Portrait Depth Bokeh and foreground face fill light.
3. 3D Cinematic Motion faux-3D parallax frame rendering & video encoder.
4. Privacy Redaction (Mosaic pixelation & Gaussian blur) and markup flattening.
5. Viewer markup mode & Studio dialog initializations.
"""

from pathlib import Path
import pytest
import numpy as np
import cv2
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

from src.imaging.creative_effects import (
    estimate_subject_mask,
    apply_color_pop,
    refine_mask_brush,
    apply_portrait_bokeh_and_fill_light,
    inpaint_disocclusion_background,
    render_3d_cinematic_motion_frame,
    render_3d_cinematic_motion_video,
    apply_pixelate,
    apply_blur_redact,
)
from src.ui.markup_overlay import MarkupElement, flatten_markup_onto_image, MarkupFloatingToolbar
from src.ui.create_studio_dialogs import ColorPopDialog, PortraitBokehDialog, CinematicMotionDialog
from src.ui.media_viewer import SingleMediaViewer


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# =============================================================================
# 1. Color Pop Tests
# =============================================================================

def test_color_pop_mask_and_bilateral_brush():
    """Verifies subject mask estimation, bilateral edge snapping, and B&W background rendering."""
    # Create test image with distinct foreground circle on light background
    h, w = 300, 300
    img = np.full((h, w, 3), 220, dtype=np.uint8)  # Light gray background
    cv2.circle(img, (150, 150), 55, (40, 80, 230), -1)  # Red/orange foreground

    # 1. Mask estimation
    mask = estimate_subject_mask(img, face_boxes=[(120, 110, 60, 60)])
    assert mask.shape == (h, w)
    assert np.max(mask) > 0.5
    # Center should be foreground
    assert mask[150, 150] > 0.4
    # Corner should be background
    assert mask[10, 10] < 0.3

    # 2. Bilateral brush snapping
    # Paint "Make B&W" on the corner
    refined = refine_mask_brush(mask, img, (10, 10), radius=20, mode="make_bw")
    assert refined[10, 10] == 0.0

    # Paint "Keep Color" on the center
    refined = refine_mask_brush(refined, img, (150, 150), radius=20, mode="keep_color")
    assert refined[150, 150] > 0.8

    # 3. Apply Color Pop
    pop = apply_color_pop(img, refined, bg_contrast=1.2)
    assert pop.shape == (h, w, 3)

    # In the background corner, BGR channels should be equal or very close (B&W)
    corner_px = pop[10, 10]
    assert abs(int(corner_px[0]) - int(corner_px[1])) <= 2
    assert abs(int(corner_px[1]) - int(corner_px[2])) <= 2

    # In foreground center, colors should remain vibrant
    center_px = pop[150, 150]
    assert center_px[2] > center_px[0] + 50


# =============================================================================
# 2. Portrait Bokeh & Fill Light Tests
# =============================================================================

def test_portrait_bokeh_and_fill_light():
    """Verifies portrait depth lens bokeh and subject fill light."""
    h, w = 250, 250
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Background texture: grid pattern with sharp edges
    img[::10, :] = 255
    img[:, ::10] = 255
    # Foreground subject
    cv2.circle(img, (125, 125), 45, (100, 120, 140), -1)

    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, (125, 125), 45, 1.0, -1)

    result = apply_portrait_bokeh_and_fill_light(
        img,
        mask,
        bokeh_radius=15,
        fill_light=0.30,
    )

    assert result.shape == (h, w, 3)
    # Background sharp grid lines should be blurred (no pure 255 adjacent to 0)
    bg_sample = result[20:30, 20:30]
    assert np.std(bg_sample) < np.std(img[20:30, 20:30])

    # Subject center should be brightened by fill light
    assert int(result[125, 125, 0]) >= int(img[125, 125, 0])


# =============================================================================
# 3. 3D Cinematic Motion Tests
# =============================================================================

def test_3d_cinematic_motion_rendering(tmp_path: Path):
    """Verifies background disocclusion inpainting and 3D parallax video frame generation."""
    h, w = 240, 320
    img = np.full((h, w, 3), 150, dtype=np.uint8)
    cv2.rectangle(img, (100, 60), (220, 200), (40, 160, 220), -1)

    mask = np.zeros((h, w), dtype=np.float32)
    mask[60:200, 100:220] = 1.0

    # 1. Inpaint background behind subject
    bg_clean = inpaint_disocclusion_background(img, mask)
    assert bg_clean.shape == (h, w, 3)

    # 2. Render parallax frame at t = 0.5
    frame = render_3d_cinematic_motion_frame(
        foreground_bgr=img,
        background_bgr=bg_clean,
        subject_mask=mask,
        t=0.5,
        canvas_size=(320, 240),
    )
    assert frame.shape == (240, 320, 3)

    # 3. Short video render test
    out_mp4 = tmp_path / "cinematic_test.mp4"
    success = render_3d_cinematic_motion_video(
        img,
        out_mp4,
        duration_sec=0.2,  # 6 frames for fast test
        fps=30,
        resolution=(320, 240),
        subject_mask=mask,
    )
    assert success is True
    assert out_mp4.exists()
    assert out_mp4.stat().st_size > 100


# =============================================================================
# 4. Privacy Redaction & Markup Flattening Tests
# =============================================================================

def test_privacy_redact_and_markup_flatten(qapp):
    """Verifies mosaic pixelate, blur, arrow callout, magnifier, and text flattening."""
    h, w = 300, 400
    base = np.full((h, w, 3), 100, dtype=np.uint8)
    cv2.putText(base, "SECRET 123", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 1. Standalone pixelate and blur redaction
    pix = apply_pixelate(base, (40, 70, 200, 40), block_size=16)
    assert pix.shape == base.shape
    assert not np.array_equal(pix, base)

    blur = apply_blur_redact(base, (40, 70, 200, 40), radius=20)
    assert blur.shape == base.shape

    # 2. Markup elements list
    elements = [
        MarkupElement(kind="pixelate", rect=QRectF(30, 60, 120, 50)),
        MarkupElement(kind="arrow", start_pt=QPointF(20, 20), end_pt=QPointF(80, 80), color=QColor(255, 0, 0), stroke_width=4),
        MarkupElement(kind="magnifier", rect=QRectF(180, 150, 80, 80), zoom_factor=2.5),
        MarkupElement(kind="text", rect=QRectF(50, 200, 120, 30), text="Callout Note", color=QColor(255, 255, 255)),
    ]

    flattened = flatten_markup_onto_image(base, elements)
    assert flattened.shape == base.shape
    # Pixelated area should be modified
    assert not np.array_equal(flattened[60:110, 30:150], base[60:110, 30:150])


# =============================================================================
# 5. UI Integration & Dialog Tests
# =============================================================================

def test_single_media_viewer_markup_mode_and_dialogs(qapp, tmp_path: Path):
    """Verifies SingleMediaViewer markup mode toggling, toolbar presence, and Creative dialogs."""
    viewer = SingleMediaViewer()
    dummy = np.zeros((300, 400, 3), dtype=np.uint8)
    viewer.master_np_rgb = dummy
    viewer.show()

    # Verify markup toolbar exists
    assert hasattr(viewer, "markup_toolbar")
    assert isinstance(viewer.markup_toolbar, MarkupFloatingToolbar)

    # Toggle markup mode
    viewer.set_markup_mode(True)
    assert viewer.is_markup_mode is True
    assert viewer.markup_toolbar.isVisible()

    viewer.set_markup_mode(False)
    assert viewer.is_markup_mode is False
    assert not viewer.markup_toolbar.isVisible()

    # Test Studio Dialogs initialization
    test_img = np.full((200, 200, 3), 150, dtype=np.uint8)
    p = tmp_path / "sample.jpg"
    cv2.imwrite(str(p), test_img)

    dlg_pop = ColorPopDialog(test_img, p)
    assert dlg_pop.windowTitle().startswith("Colour Pop") or dlg_pop.windowTitle().startswith("Color Pop")
    dlg_pop.close()

    dlg_bokeh = PortraitBokehDialog(test_img, p)
    assert dlg_bokeh.windowTitle().startswith("Portrait Depth")
    dlg_bokeh.close()

    dlg_cinematic = CinematicMotionDialog(test_img, p)
    assert dlg_cinematic.windowTitle().startswith("3D Cinematic")
    dlg_cinematic.close()


def test_color_pop_interactive_tools_and_undo_redo(qapp, tmp_path: Path, monkeypatch):
    """Verifies ColorPopDialog canvas interaction, painting, undo/redo, invert, and save without crash."""
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtCore import QEvent

    test_img = np.full((300, 400, 3), 120, dtype=np.uint8)
    test_img[100:200, 150:250] = [220, 50, 50]
    p = tmp_path / "interactive_pop.jpg"
    cv2.imwrite(str(p), test_img)

    dlg = ColorPopDialog(test_img, p)
    dlg.show()
    qapp.processEvents()

    assert dlg.view is not None
    assert len(dlg.undo_stack) >= 1
    initial_undo_count = len(dlg.undo_stack)

    # 1. Simulate mouse painting on canvas (Press -> Move -> Release)
    canvas = dlg.view
    press_event = QMouseEvent(QEvent.MouseButtonPress, QPoint(100, 100), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    canvas.mousePressEvent(press_event)
    move_event = QMouseEvent(QEvent.MouseMove, QPoint(150, 120), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    canvas.mouseMoveEvent(move_event)
    release_event = QMouseEvent(QEvent.MouseButtonRelease, QPoint(150, 120), Qt.LeftButton, Qt.NoButton, Qt.NoModifier)
    canvas.mouseReleaseEvent(release_event)

    # Stroke should be pushed to undo stack
    assert len(dlg.undo_stack) > initial_undo_count
    assert dlg.btn_undo.isEnabled()

    # 2. Test Undo & Redo
    dlg._undo_mask()
    qapp.processEvents()
    assert dlg.btn_redo.isEnabled()
    dlg._redo_mask()
    qapp.processEvents()

    # 3. Test changing tools (B&W brush vs Color brush)
    dlg._set_brush_mode("make_bw")
    assert dlg.brush_mode == "make_bw"
    dlg._set_brush_mode("keep_color")
    assert dlg.brush_mode == "keep_color"

    # 4. Test slider adjustments
    dlg.sld_radius.setValue(45)
    assert dlg.brush_radius == 45
    dlg.sld_contrast.setValue(125)
    assert abs(dlg.bg_contrast - 1.25) < 1e-4

    # 5. Test Invert Mask and Auto Detect
    dlg._invert_mask()
    qapp.processEvents()
    dlg._auto_detect_mask()
    qapp.processEvents()

    # 6. Test Hold to Compare
    dlg._show_original()
    qapp.processEvents()
    dlg._restore_color_pop()
    qapp.processEvents()

    # 7. Test Save / Apply with new file prompt
    out_save_path = tmp_path / "interactive_pop_colourPop.jpg"
    monkeypatch.setattr(
        "src.ui.create_studio_dialogs.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(out_save_path), "JPEG Image (*.jpg *.jpeg)"),
    )
    dlg._save_photo()
    assert out_save_path.exists()
    assert dlg.saved_file_path == out_save_path

    dlg.close()


def test_portrait_bokeh_anatomical_mask_and_normalized_conv():
    """
    Verifies that estimate_subject_mask with face coordinates:
    - Eliminates tree/car halo beside head (mask == 0.0 outside head at eye/ear level).
    - Eliminates bottom jacket horizontal seam (mask > 0.8 all the way to bottom edge).
    """
    h, w = 1000, 1000
    img = np.full((h, w, 3), 210, dtype=np.uint8)
    # Face and standing body
    img[280:460, 520:630] = [150, 180, 220]
    img[460:1000, 420:720] = [40, 40, 40]

    fx, fy, fw, fh = int(w * 0.52), int(h * 0.28), int(w * 0.11), int(h * 0.18)
    mask = estimate_subject_mask(img, [(fx, fy, fw, fh)])

    # Beside head (left & right at eye/ear level) must be clean background
    assert mask[int(h * 0.35), int(w * 0.35)] == 0.0
    assert mask[int(h * 0.35), int(w * 0.70)] == 0.0
    # Lower jacket near bottom of photo (y = 96%) must be foreground
    assert mask[int(h * 0.96), int(w * 0.57)] > 0.8

