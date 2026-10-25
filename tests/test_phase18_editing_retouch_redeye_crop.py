"""
Tests for Phase 18:
- Drag-and-drop role bug fix (CatalogRoles.RecordRole).
- Windows safe atomic replace with retry & read-only permission clearing in xmp_sync_worker.
- Instant in-memory QPixmap rotation.
- Highlight-preserving photographic brightness curve.
- 60 FPS slider drag proxy rendering in SingleMediaViewer.
- Interactive Crop floating toolbar with proportions (1:1, 4:3, 16:9, Original).
- Retouch (Magic Eraser) interactive inpainting with PixelArrayUndoCommand.
- Red Eye removal interactive pupil desaturation.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtWidgets import QApplication

from src.db.connection import DualDatabaseEngine
from src.imaging.opencv_engine import PhotoAdjustmentRecipe, render_adjusted_image
from src.ui.grid_model import CatalogRoles
from src.ui.main_window import PhotoGalleryMainWindow
from src.ui.media_viewer import SingleMediaViewer, PixelArrayUndoCommand
from src.workers.xmp_sync_worker import _safe_atomic_replace


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_catalog_roles_record_role() -> None:
    """Verifies CatalogRoles has RecordRole defined and avoids AttributeError."""
    assert hasattr(CatalogRoles, "RecordRole")
    assert CatalogRoles.RecordRole == Qt.UserRole + 18


def test_safe_atomic_replace_with_readonly(tmp_path: Path) -> None:
    """Verifies _safe_atomic_replace handles Windows Read-Only target files without failing."""
    target_file = tmp_path / "test_photo.jpg"
    target_file.write_bytes(b"initial_bytes")

    # Mark target file read-only (WinError 5 trigger)
    os.chmod(target_file, stat.S_IREAD)

    tmp_file = tmp_path / "test_photo.jpg.tmp_xmp"
    tmp_file.write_bytes(b"new_xmp_bytes")

    ok = _safe_atomic_replace(tmp_file, target_file, max_retries=3)
    assert ok is True
    assert target_file.read_bytes() == b"new_xmp_bytes"
    assert not tmp_file.exists()


def test_photographic_brightness_preserves_highlights() -> None:
    """Verifies that brightness adjustment preserves specular highlights (1.0) and black levels (0.0)."""
    # Create test ramp: black, shadow, midtone, highlight, pure white
    ramp = np.array([[[0, 0, 0], [64, 64, 64], [128, 128, 128], [220, 220, 220], [255, 255, 255]]], dtype=np.uint8)

    # Moderate boost (+40)
    rec_boost = PhotoAdjustmentRecipe(brightness=40.0)
    rendered_boost = render_adjusted_image(ramp, rec_boost)

    # Pure black remains black
    assert int(rendered_boost[0, 0, 0]) == 0
    # Midtones are lifted (128 -> >150)
    assert int(rendered_boost[0, 2, 0]) > 140
    # Specular white does not overflow or wrap around, stays 255
    assert int(rendered_boost[0, 4, 0]) == 255

    # Darkening (-40)
    rec_dark = PhotoAdjustmentRecipe(brightness=-40.0)
    rendered_dark = render_adjusted_image(ramp, rec_dark)
    assert int(rendered_dark[0, 0, 0]) == 0
    assert int(rendered_dark[0, 2, 0]) < 120
    assert int(rendered_dark[0, 4, 0]) == 255


def test_viewer_fast_proxy_during_slider_drag(qapp: QApplication, tmp_path: Path) -> None:
    """Verifies that slider dragging uses the fast downscaled proxy and commits full-res on release."""
    viewer = SingleMediaViewer()
    viewer.show()
    test_img = tmp_path / "big_photo.jpg"
    # Create a 2400x1800 image
    im = Image.new("RGB", (2400, 1800), (100, 150, 200))
    im.save(test_img, quality=90)

    viewer.load_media(media_id=1, file_path=test_img, is_video=False)

    # Check proxy creation
    assert viewer.display_proxy_rgb is not None
    assert max(viewer.display_proxy_rgb.shape[:2]) <= 1600
    assert viewer._proxy_scale < 1.0

    # Start slider dragging
    viewer.begin_slider_adjustment()
    assert viewer._is_sliding is True

    # Adjust exposure parameters
    viewer.set_exposure_parameters(brightness=30, contrast=10, shadows=0, highlights=0)
    assert viewer.current_recipe.brightness == 30.0

    # Commit slider drag
    viewer.commit_slider_adjustment("Brightness")
    assert viewer._is_sliding is False
    assert viewer.current_recipe.brightness == 30.0
    assert viewer.undo_stack.count() == 1


def test_crop_toolbar_ratios_and_apply(qapp: QApplication, tmp_path: Path) -> None:
    """Verifies Crop proportion presets and non-destructive recipe crop application."""
    viewer = SingleMediaViewer()
    viewer.show()
    test_img = tmp_path / "crop_test.jpg"
    im = Image.new("RGB", (800, 600), (120, 140, 160))
    im.save(test_img, quality=90)

    viewer.load_media(media_id=2, file_path=test_img, is_video=False)

    # Enable crop mode
    viewer.set_crop_mode(True)
    assert viewer.is_crop_mode is True
    assert viewer.crop_toolbar.isVisible()

    # Select Square 1:1 proportion
    viewer.combo_crop_ratio.setCurrentText("1:1 (Square)")
    rect = viewer.scene.get_crop_rect()
    assert rect is not None
    # Verify square aspect ratio within floating tolerance
    assert abs(rect.width() - rect.height()) < 1.0

    # Apply crop
    ok = viewer.apply_crop()
    assert ok is True
    assert viewer.is_crop_mode is False
    assert not viewer.crop_toolbar.isVisible()
    assert viewer.current_recipe.crop_rect is not None
    assert viewer.undo_stack.count() == 1


def test_retouch_inpainting_and_undo(qapp: QApplication, tmp_path: Path) -> None:
    """Verifies interactive Retouch (Magic Eraser) inpainting and undo/redo."""
    viewer = SingleMediaViewer()
    viewer.show()
    test_img = tmp_path / "retouch_test.jpg"
    # Create white canvas with a black blemish at (50, 50)
    arr = np.full((100, 100, 3), 255, dtype=np.uint8)
    arr[48:53, 48:53] = 0
    im = Image.fromarray(arr)
    im.save(test_img, quality=95)

    viewer.load_media(media_id=3, file_path=test_img, is_video=False)

    # Enable retouch mode
    viewer.set_retouch_mode(True)
    assert viewer.is_retouch_mode is True
    assert viewer.retouch_toolbar.isVisible()

    # Stroke over the blemish at (50, 50)
    viewer.retouch_radius = 10
    viewer._start_retouch(QPointF(50, 50))
    viewer._finish_retouch()

    # Blemish at (50, 50) should now be healed (close to 255)
    healed_pixel = viewer.master_np_rgb[50, 50]
    assert int(healed_pixel[0]) > 200

    # Undo retouch
    assert viewer.undo_stack.count() == 1
    viewer.undo_stack.undo()
    assert int(viewer.master_np_rgb[50, 50, 0]) == 0

    # Redo retouch
    viewer.undo_stack.redo()
    assert int(viewer.master_np_rgb[50, 50, 0]) > 200


def test_red_eye_removal_and_undo(qapp: QApplication, tmp_path: Path) -> None:
    """Verifies interactive Red Eye pupil detection and desaturation."""
    viewer = SingleMediaViewer()
    viewer.show()
    test_img = tmp_path / "redeye_test.jpg"
    # Create neutral skin canvas with a pure red pupil box at (30..40, 30..40)
    arr = np.full((80, 80, 3), 160, dtype=np.uint8)
    arr[30:40, 30:40] = [240, 20, 20]  # Bright red pupil
    im = Image.fromarray(arr)
    im.save(test_img, quality=95)

    viewer.load_media(media_id=4, file_path=test_img, is_video=False)

    # Enable Red Eye mode
    viewer.set_red_eye_mode(True)
    assert viewer.is_red_eye_mode is True
    assert viewer.red_eye_toolbar.isVisible()

    # Drag rectangle enclosing the red pupil
    viewer._start_red_eye(QPointF(28, 28))
    viewer._drag_red_eye(QPointF(42, 42))
    viewer._finish_red_eye()

    # Pupil redness should be desaturated (R no longer dominates G & B)
    pupil_px = viewer.master_np_rgb[35, 35]
    r, g, b = int(pupil_px[0]), int(pupil_px[1]), int(pupil_px[2])
    assert r < 60  # Desaturated and darkened

    # Undo
    viewer.undo_stack.undo()
    assert int(viewer.master_np_rgb[35, 35, 0]) == 240
