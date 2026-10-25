"""
Tests for Phase 10: High-Fidelity OpenCV & NumPy Photo Adjustment Engine,
Auto-Adjust, Auto-Straighten, Recipe Undo/Redo, and Non-Destructive Revert.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication

from src.imaging.opencv_engine import (
    PhotoAdjustmentRecipe,
    analyze_auto_adjust,
    compute_histogram_data,
    detect_auto_straighten_angle,
    render_adjusted_image,
)
from src.ui.media_viewer import RecipeUndoCommand, SingleMediaViewer
from src.ui.right_tag_dock import TagAndCaptionDock


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def sample_rgb_image() -> np.ndarray:
    """Creates a synthetic RGB test image with an artificial horizon line."""
    h, w = 240, 320
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Sky (blue gradient)
    for y in range(h // 2):
        img[y, :] = [80 + y // 3, 130 + y // 3, 220]
    # Ground (green gradient)
    for y in range(h // 2, h):
        img[y, :] = [30, 140 - (y - h // 2) // 4, 40]
    return img


def test_recipe_dataclass():
    recipe = PhotoAdjustmentRecipe()
    assert recipe.is_default() is True

    recipe.brightness = 15.0
    assert recipe.is_default() is False

    copy_rec = recipe.copy()
    assert copy_rec == recipe
    copy_rec.contrast = 20.0
    assert copy_rec != recipe


def test_render_exposure_and_contrast(sample_rgb_image):
    recipe = PhotoAdjustmentRecipe(brightness=25.0, contrast=30.0, shadows=15.0, highlights=-10.0)
    adjusted = render_adjusted_image(sample_rgb_image, recipe)

    assert adjusted.shape == sample_rgb_image.shape
    assert adjusted.dtype == np.uint8
    # Adjusted image should have different mean luminance
    assert not np.array_equal(adjusted, sample_rgb_image)


def test_render_color_temperature_tint(sample_rgb_image):
    # Warm amber shift (increases Red, decreases Blue)
    recipe_warm = PhotoAdjustmentRecipe(temperature=50.0)
    warm = render_adjusted_image(sample_rgb_image, recipe_warm)
    assert warm.shape == sample_rgb_image.shape

    # Cool blue shift (decreases Red, increases Blue)
    recipe_cool = PhotoAdjustmentRecipe(temperature=-50.0)
    cool = render_adjusted_image(sample_rgb_image, recipe_cool)
    assert cool.shape == sample_rgb_image.shape

    # Red channel in warm must be higher than in cool
    assert warm[:, :, 0].mean() > cool[:, :, 0].mean()
    # Blue channel in cool must be higher than in warm
    assert cool[:, :, 2].mean() > warm[:, :, 2].mean()

    # Saturation boost
    recipe_sat = PhotoAdjustmentRecipe(saturation=50.0)
    sat_boost = render_adjusted_image(sample_rgb_image, recipe_sat)
    assert sat_boost.shape == sample_rgb_image.shape


def test_render_preset_effects(sample_rgb_image):
    # Black and white
    bw = render_adjusted_image(sample_rgb_image, PhotoAdjustmentRecipe(effect="bw"))
    # In B&W, R, G, and B channels must match
    assert np.allclose(bw[:, :, 0], bw[:, :, 1], atol=1)
    assert np.allclose(bw[:, :, 1], bw[:, :, 2], atol=1)

    # Sepia
    sepia = render_adjusted_image(sample_rgb_image, PhotoAdjustmentRecipe(effect="sepia"))
    assert sepia.shape == sample_rgb_image.shape
    # Sepia R > G > B
    assert sepia[:, :, 0].mean() >= sepia[:, :, 1].mean()


def test_render_straighten_and_crop(sample_rgb_image):
    # Straighten with auto-crop (prevents black triangles)
    recipe = PhotoAdjustmentRecipe(straighten_deg=3.5)
    straightened = render_adjusted_image(sample_rgb_image, recipe)
    assert straightened.ndim == 3
    assert straightened.shape[0] > 0 and straightened.shape[1] > 0

    # Explicit crop rectangle
    crop_rec = PhotoAdjustmentRecipe(crop_rect=(20, 30, 200, 180))
    cropped = render_adjusted_image(sample_rgb_image, crop_rec)
    assert cropped.shape[1] == (200 - 20)
    assert cropped.shape[0] == (180 - 30)


def test_analyze_auto_adjust(sample_rgb_image):
    # Underexposed dark image
    dark = (sample_rgb_image * 0.3).astype(np.uint8)
    recipe = analyze_auto_adjust(dark)
    assert isinstance(recipe, PhotoAdjustmentRecipe)
    # Underexposed image should receive positive brightness shift
    assert recipe.brightness > 0.0


def test_detect_auto_straighten_angle(sample_rgb_image):
    angle = detect_auto_straighten_angle(sample_rgb_image)
    assert isinstance(angle, float)
    # Artificial horizon line is horizontal (angle should be near 0)
    assert abs(angle) < 5.0


def test_compute_histogram_data(sample_rgb_image):
    hist_r, hist_g, hist_b, hist_l = compute_histogram_data(sample_rgb_image)
    assert len(hist_r) == 256
    assert len(hist_g) == 256
    assert len(hist_b) == 256
    assert len(hist_l) == 256
    assert 0.0 <= max(hist_r.max(), hist_g.max(), hist_b.max(), hist_l.max()) <= 1.0


def test_viewer_recipe_undo_redo(qapp, sample_rgb_image, tmp_path):
    viewer = SingleMediaViewer()
    test_img_path = tmp_path / "test.jpg"
    Image.fromarray(sample_rgb_image).save(test_img_path)

    viewer.load_media(1, test_img_path, is_video=False)
    assert viewer.master_np_rgb is not None
    assert viewer.current_recipe.is_default()

    # Apply Auto Adjust
    viewer.apply_auto_adjust()
    assert not viewer.current_recipe.is_default()
    assert viewer.undo_stack.canUndo()

    # Undo
    viewer.undo_stack.undo()
    assert viewer.current_recipe.is_default()
    assert viewer.undo_stack.canRedo()

    # Redo
    viewer.undo_stack.redo()
    assert not viewer.current_recipe.is_default()


def test_viewer_save_and_revert_original(qapp, sample_rgb_image, tmp_path):
    viewer = SingleMediaViewer()
    test_img_path = tmp_path / "sample.jpg"
    Image.fromarray(sample_rgb_image).save(test_img_path)

    viewer.load_media(42, test_img_path, is_video=False)

    # Make adjustments
    viewer.set_exposure_parameters(brightness=30, contrast=20, shadows=10, highlights=0)
    assert viewer.has_unsaved_adjustments()

    # Save with backup
    backup_path = viewer.save_current_image_with_backup()
    assert backup_path is not None
    assert Path(backup_path).exists()
    assert not viewer.has_unsaved_adjustments()

    # Revert to original
    reverted = viewer.revert_to_original()
    assert reverted is True
    assert not Path(backup_path).exists()  # Backup consumed
    assert viewer.current_recipe.is_default()


def test_dock_sync_recipe(qapp):
    dock = TagAndCaptionDock()
    dock.set_mode("fine_tune")

    recipe = PhotoAdjustmentRecipe(
        brightness=22.0,
        contrast=-15.0,
        shadows=10.0,
        highlights=-8.0,
        temperature=35.0,
        tint=-12.0,
        saturation=40.0,
        straighten_deg=3.0,
        sharpen=55.0,
        denoise=30.0,
    )
    dock.sync_recipe(recipe)

    assert dock.ft_slider_brightness.value() == 22
    assert dock.ft_slider_contrast.value() == -15
    assert dock.ft_slider_shadows.value() == 10
    assert dock.ft_slider_highlights.value() == -8
    assert dock.ft_slider_temp.value() == 35
    assert dock.ft_slider_tint.value() == -12
    assert dock.ft_slider_sat.value() == 40
    assert dock.ft_slider_straighten.value() == 3
    assert dock.ft_slider_sharpen.value() == 55
    assert dock.ft_slider_denoise.value() == 30
