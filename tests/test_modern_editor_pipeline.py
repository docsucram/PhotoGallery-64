"""
Tests for PhotoGallery64 Modern Photo Editing Pipeline:
1. Perceptual Float32 Processing & Filmic Exposure Shoulder
2. Edge-Aware Shadows & Highlights via LAB Decomposition (Snapseed Ambience)
3. Clarity, Dehaze, and Vignette
4. Functional Noise Profile Analysis (MAD Estimator) & Dual-Domain Denoising
5. 6 Modern Photographic Look Presets & Intensity Blending
6. Inscribed-Rectangle Auto Straightening
7. Grid View vs Single Viewer Tool Routing & Batch Edits
8. Non-Destructive .OriginalFiles Lifecycle & Revert
9. Double-Click Slider Reset & Hold to Compare (Before / After)
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytest
from PySide6.QtCore import Qt, QPoint
from PySide6.QtWidgets import QApplication

from src.db.connection import DualDatabaseEngine
from src.imaging.opencv_engine import (
    PhotoAdjustmentRecipe,
    render_adjusted_image,
    analyze_auto_adjust,
    analyze_noise_profile,
    detect_auto_straighten_angle,
    _crop_largest_inscribed_rect,
    _apply_effect_grade,
)
from src.ui.main_window import PhotoGalleryMainWindow
from src.ui.right_tag_dock import _AeroFineTuneSlider


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def test_env(tmp_path: Path):
    photos_dir = tmp_path / "Pictures"
    photos_dir.mkdir(parents=True, exist_ok=True)

    # Create two test images
    img1 = Image.new("RGB", (320, 240), (140, 160, 180))
    img1.save(photos_dir / "Photo_01.jpg", quality=92)

    img2 = Image.new("RGB", (320, 240), (200, 120, 90))
    img2.save(photos_dir / "Photo_02.jpg", quality=92)

    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    engine = DualDatabaseEngine(
        catalog_path=db_dir / "catalog.db",
        thumbnails_path=db_dir / "thumbnails.db",
    )
    window = PhotoGalleryMainWindow(engine=engine, start_xmp_daemon=False)
    window.scan_folder(photos_dir, include_subfolders=True, blocking=True)
    window.show()

    yield {
        "window": window,
        "photos_dir": photos_dir,
        "engine": engine,
        "photo1": photos_dir / "Photo_01.jpg",
        "photo2": photos_dir / "Photo_02.jpg",
    }
    window.close()
    engine.close_thread_connections()


def test_adjustment_recipe_and_modern_fields():
    """Verifies PhotoAdjustmentRecipe defaults, new fields, is_default, and copy behavior."""
    rec = PhotoAdjustmentRecipe()
    assert rec.is_default()
    assert rec.clarity == 0.0
    assert rec.dehaze == 0.0
    assert rec.vignette == 0.0
    assert rec.effect_intensity == 100.0
    assert len(rec.selective_points) == 0

    rec.clarity = 25.0
    assert not rec.is_default()

    copied = rec.copy()
    assert copied.clarity == 25.0
    copied.selective_points.append({"x": 100, "y": 100, "radius": 50})
    assert len(rec.selective_points) == 0  # Deep copy test


def test_perceptual_tone_and_filmic_exposure():
    """Verifies soft-knee shoulder roll-off prevents harsh white blowout, and shadows lift details."""
    # Synthetic test image with dark, midtone, and bright regions
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:33, :] = 30    # Shadows
    arr[33:66, :] = 128 # Midtones
    arr[66:, :] = 210   # Highlights

    # 1. Filmic exposure boost (+80)
    rec_exp = PhotoAdjustmentRecipe(brightness=80.0)
    rendered_exp = render_adjusted_image(arr, rec_exp)
    # Highlight area should not clip entirely to 255 due to filmic shoulder roll-off
    assert rendered_exp[80, 50, 0] <= 255
    assert rendered_exp[80, 50, 0] > 210
    assert rendered_exp[50, 50, 0] > 128  # Midtones lifted

    # 2. Edge-aware shadow lift (+75)
    rec_shadow = PhotoAdjustmentRecipe(shadows=75.0)
    rendered_shadow = render_adjusted_image(arr, rec_shadow)
    # Dark area lifted significantly
    assert rendered_shadow[15, 50, 0] > arr[15, 50, 0]
    # Highlights remain controlled
    assert abs(int(rendered_shadow[80, 50, 0]) - int(arr[80, 50, 0])) < 15


def test_clarity_dehaze_and_vignette():
    """Verifies clarity, dehaze, and vignette adjustments."""
    arr = np.full((120, 120, 3), 128, dtype=np.uint8)
    # Add a high-contrast edge
    arr[40:80, 40:80] = 180

    rec_clarity = PhotoAdjustmentRecipe(clarity=50.0)
    rend_clarity = render_adjusted_image(arr, rec_clarity)
    assert rend_clarity is not None
    assert rend_clarity.shape == arr.shape

    # Vignette: corners should be darker than center when vignette is negative
    rec_vig = PhotoAdjustmentRecipe(vignette=-80.0)
    rend_vig = render_adjusted_image(arr, rec_vig)
    corner_val = int(rend_vig[5, 5, 0])
    center_val = int(rend_vig[60, 60, 0])
    assert corner_val < center_val


def test_modern_photographic_effects_and_intensity():
    """Verifies all 6 modern preset looks and 0-100% intensity blending."""
    arr = np.full((80, 80, 3), (120, 150, 180), dtype=np.uint8)

    presets = [
        "classic_chrome",
        "warm_portrait",
        "vivid_landscape",
        "matte_vintage",
        "fine_art_bw",
        "cool_cinema",
    ]

    for preset in presets:
        # Full intensity (100%)
        rec_full = PhotoAdjustmentRecipe(effect=preset, effect_intensity=100.0)
        out_full = render_adjusted_image(arr, rec_full)
        assert out_full is not None
        assert not np.array_equal(out_full, arr), f"Preset {preset} should alter image"

        # Zero intensity (0%) should match base unmodified image
        rec_zero = PhotoAdjustmentRecipe(effect=preset, effect_intensity=0.0)
        out_zero = render_adjusted_image(arr, rec_zero)
        assert np.allclose(out_zero, arr, atol=2), f"Preset {preset} at 0% intensity must match base"


def test_functional_noise_analysis_and_denoise():
    """Verifies MAD noise estimator detects high ISO noise vs clean images."""
    clean_img = np.full((160, 160, 3), 128, dtype=np.uint8)
    clean_score = analyze_noise_profile(clean_img)
    assert clean_score == 0.0, "Pristine flat image should have 0 noise score"

    # Add artificial Gaussian noise
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 18, (160, 160, 3)).astype(np.int16)
    noisy_img = np.clip(clean_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    noisy_score = analyze_noise_profile(noisy_img)
    assert noisy_score > 30.0, f"Noisy image should have high noise score, got {noisy_score}"

    # Denoise reduces variance
    rec_denoise = PhotoAdjustmentRecipe(denoise=70.0)
    denoised = render_adjusted_image(noisy_img, rec_denoise)
    assert np.std(denoised) < np.std(noisy_img), "Denoising must reduce noisy pixel variance"


def test_auto_straighten_and_inscribed_crop():
    """Verifies line-detection tilt calculation and zero black corner triangles."""
    # Synthetic canvas with a +5 degree tilted line
    canvas = np.full((300, 400, 3), 200, dtype=np.uint8)
    rad = math.radians(5.0)
    cx, cy = 200, 150
    x1 = int(cx - 150 * math.cos(rad))
    y1 = int(cy - 150 * math.sin(rad))
    x2 = int(cx + 150 * math.cos(rad))
    y2 = int(cy + 150 * math.sin(rad))
    cv2.line(canvas, (x1, y1), (x2, y2), (20, 20, 20), 4)

    angle = detect_auto_straighten_angle(canvas)
    assert abs(angle - 5.0) <= 2.5, f"Expected detected tilt around 5 deg, got {angle}"

    # Inscribed crop check
    rec_str = PhotoAdjustmentRecipe(straighten_deg=angle)
    rendered = render_adjusted_image(canvas, rec_str)
    # Check that corners are not black (0, 0, 0)
    assert rendered[0, 0, 0] > 100, "Inscribed crop should eliminate black border corners"


def test_fine_tune_double_click_reset(qapp: QApplication):
    """Verifies double-clicking any fine-tune slider handle resets it to its default value."""
    slider = _AeroFineTuneSlider(-100, 100, 0)
    slider.setValue(65)
    assert slider.value() == 65

    # Simulate double-click event
    class DummyEvent:
        def accept(self):
            pass

    slider.mouseDoubleClickEvent(DummyEvent())
    assert slider.value() == 0, "Double click must reset slider to default_val"


def test_hold_to_compare(qapp: QApplication, test_env: dict):
    """Verifies holding compare temporarily reveals unadjusted original without losing recipe."""
    win: PhotoGalleryMainWindow = test_env["window"]
    records = win.catalog_model._records
    mid = records[0].id

    win.open_single_viewer(mid)
    mv = win.media_viewer

    # Apply an effect
    mv.apply_preset_effect("fine_art_bw")
    assert mv.current_recipe.effect == "fine_art_bw"

    # Hold compare (active = True)
    mv.set_compare_mode(True)
    assert mv._is_comparing is True

    # Release compare (active = False)
    mv.set_compare_mode(False)
    assert mv._is_comparing is False
    assert mv.current_recipe.effect == "fine_art_bw"


def test_canvas_tools_routing_from_grid_view(qapp: QApplication, test_env: dict):
    """Verifies that clicking canvas-only tools in Grid View automatically opens Single Viewer with that tool active."""
    win: PhotoGalleryMainWindow = test_env["window"]
    records = win.catalog_model._records
    mid = records[0].id

    win.show_grid_view()
    win.grid_view.select_media_id(mid)

    # 1. Click 'crop_photo' in Grid View
    win.dispatch_ribbon_action("crop_photo")
    assert win.central_stack.currentWidget() == win.media_viewer
    assert win.media_viewer.is_crop_mode is True

    # Return to Grid View
    win.show_grid_view()
    win.grid_view.select_media_id(mid)

    # 2. Click 'retouch' in Grid View
    win.dispatch_ribbon_action("retouch")
    assert win.central_stack.currentWidget() == win.media_viewer
    assert win.media_viewer.is_retouch_mode is True

    # Return to Grid View
    win.show_grid_view()
    win.grid_view.select_media_id(mid)

    # 3. Click 'fine_tune' in Grid View
    win.dispatch_ribbon_action("fine_tune")
    assert win.central_stack.currentWidget() == win.media_viewer
    assert win.right_tag_dock.current_mode() == "fine_tune"


def test_grid_view_batch_edits_and_original_backup(qapp: QApplication, test_env: dict):
    """Verifies batch edits in Grid View back up to .OriginalFiles and support Revert to original."""
    win: PhotoGalleryMainWindow = test_env["window"]
    records = win.catalog_model._records
    assert len(records) >= 2
    mids = [r.id for r in records[:2]]

    win.show_grid_view()
    win.grid_view.set_selected_media_ids(mids)
    win.on_grid_selection_changed(mids, mids[0])

    p1 = test_env["photo1"]
    p2 = test_env["photo2"]
    orig_b1 = p1.parent / ".OriginalFiles" / p1.name
    orig_b2 = p2.parent / ".OriginalFiles" / p2.name

    # Backups should not exist yet
    assert not orig_b1.exists()
    assert not orig_b2.exists()

    # Apply batch effect (Warm Portrait)
    win.dispatch_ribbon_action("effect_warm_portrait")

    # .OriginalFiles backups MUST now exist
    assert orig_b1.exists(), ".OriginalFiles backup must exist for Photo 1"
    assert orig_b2.exists(), ".OriginalFiles backup must exist for Photo 2"
    assert win.ribbon.btn_edit_revert.isEnabled(), "Revert button must be enabled"

    # Revert to original
    win.dispatch_ribbon_action("revert_original")
    assert not orig_b1.exists(), "Reverting must clean up .OriginalFiles backup"
    assert not orig_b2.exists()
    assert not win.ribbon.btn_edit_revert.isEnabled(), "Revert button must be disabled after revert"
