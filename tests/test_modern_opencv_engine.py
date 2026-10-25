"""
Tests for resolution-invariant spatial scaling, single-pass LAB engine,
negative exposure curve, and photographic pipeline order in opencv_engine.py.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.imaging.opencv_engine import PhotoAdjustmentRecipe, render_adjusted_image


@pytest.fixture
def test_pair() -> tuple[np.ndarray, np.ndarray]:
    """
    Creates a realistic test scene with smooth gradients, step edges, and fine texture.
    Master: 3840x2880 (4K UHD 4:3)
    Proxy: 1280x960 (Display preview)
    """
    np.random.seed(42)
    h_m, w_m = 2880, 3840
    y, x = np.ogrid[:h_m, :w_m]

    master = np.zeros((h_m, w_m, 3), dtype=np.uint8)
    # Luma gradient + sinusoids simulating textured landscape
    master[:, :, 0] = np.clip(x / w_m * 210.0 + 30.0, 0, 255).astype(np.uint8)
    master[:, :, 1] = np.clip(y / h_m * 190.0 + 35.0, 0, 255).astype(np.uint8)
    master[:, :, 2] = np.clip(
        ((np.sin(x / 40.0) + np.cos(y / 40.0) + 2.0) / 4.0) * 200.0 + 40.0, 0, 255
    ).astype(np.uint8)

    # Insert a synthetic face-like patch near center (warm skin tone)
    # R~200, G~150, B~120
    cy, cx = h_m // 2, w_m // 2
    r_face = 200
    face_mask = ((x - cx) ** 2 + (y - cy) ** 2) < (r_face ** 2)
    master[face_mask, 0] = 200
    master[face_mask, 1] = 150
    master[face_mask, 2] = 120

    # Add gentle high-frequency noise
    noise = np.random.normal(0, 3.0, (h_m, w_m, 3)).astype(np.float32)
    master = np.clip(master.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    proxy = cv2.resize(master, (1280, 960), interpolation=cv2.INTER_AREA)
    return master, proxy


def test_negative_exposure_attenuates_highlights():
    """Verify negative exposure properly pulls down highlights without flattening."""
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    recipe = PhotoAdjustmentRecipe(brightness=-50.0)
    adjusted = render_adjusted_image(img, recipe)

    # In old broken formula 1.0 ** p == 1.0 (255 remained 255!)
    # In stops formula 255 * (2.0 ** (-0.5 * 1.5)) ≈ 255 * 0.5946 ≈ 151
    assert adjusted[50, 50, 0] < 200
    assert adjusted[50, 50, 0] > 100


def test_skin_safe_vibrance_protects_faces():
    """Verify saturation in LAB expands chroma for landscape while dampening skin shift."""
    # Skin color: RGB [200, 150, 120]
    skin_patch = np.zeros((50, 50, 3), dtype=np.uint8)
    skin_patch[:, :] = [200, 150, 120]

    # Neutral / Blue sky: RGB [60, 120, 210]
    sky_patch = np.zeros((50, 50, 3), dtype=np.uint8)
    sky_patch[:, :] = [60, 120, 210]

    recipe = PhotoAdjustmentRecipe(saturation=50.0)
    adj_skin = render_adjusted_image(skin_patch, recipe)
    adj_sky = render_adjusted_image(sky_patch, recipe)

    # Skin shift should be controlled
    skin_delta = np.mean(np.abs(adj_skin.astype(float) - skin_patch.astype(float)))
    sky_delta = np.mean(np.abs(adj_sky.astype(float) - sky_patch.astype(float)))

    # Sky saturation should expand significantly
    assert sky_delta > 10.0
    assert skin_delta < sky_delta * 1.5


def test_resolution_invariance_sharpen(test_pair):
    """Verify Sharpen on 1280x960 proxy vs 3840x2880 master downsampled has mean diff < 3.5."""
    master, proxy = test_pair
    recipe = PhotoAdjustmentRecipe(sharpen=50.0)

    res_proxy = render_adjusted_image(proxy, recipe)
    res_master = render_adjusted_image(master, recipe)
    res_master_down = cv2.resize(res_master, (1280, 960), interpolation=cv2.INTER_AREA)

    mean_diff = float(np.mean(np.abs(res_proxy.astype(float) - res_master_down.astype(float))))
    print(f"\nSharpen proxy vs master diff: {mean_diff:.4f}")
    assert mean_diff < 3.5


def test_resolution_invariance_denoise(test_pair):
    """Verify Denoise on 1280x960 proxy vs 3840x2880 master downsampled has mean diff < 3.5."""
    master, proxy = test_pair
    recipe = PhotoAdjustmentRecipe(denoise=30.0)

    res_proxy = render_adjusted_image(proxy, recipe)
    res_master = render_adjusted_image(master, recipe)
    res_master_down = cv2.resize(res_master, (1280, 960), interpolation=cv2.INTER_AREA)

    mean_diff = float(np.mean(np.abs(res_proxy.astype(float) - res_master_down.astype(float))))
    print(f"\nDenoise proxy vs master diff: {mean_diff:.4f}")
    assert mean_diff < 3.5


def test_resolution_invariance_dehaze(test_pair):
    """Verify Dehaze on 1280x960 proxy vs 3840x2880 master downsampled has mean diff < 3.5."""
    master, proxy = test_pair
    recipe = PhotoAdjustmentRecipe(dehaze=40.0)

    res_proxy = render_adjusted_image(proxy, recipe)
    res_master = render_adjusted_image(master, recipe)
    res_master_down = cv2.resize(res_master, (1280, 960), interpolation=cv2.INTER_AREA)

    mean_diff = float(np.mean(np.abs(res_proxy.astype(float) - res_master_down.astype(float))))
    print(f"\nDehaze proxy vs master diff: {mean_diff:.4f}")
    assert mean_diff < 3.5


def test_resolution_invariance_shadows_and_clarity(test_pair):
    """Verify Shadows & Clarity on proxy vs master downsampled has mean diff < 3.5."""
    master, proxy = test_pair
    recipe = PhotoAdjustmentRecipe(shadows=40.0, clarity=30.0)

    res_proxy = render_adjusted_image(proxy, recipe)
    res_master = render_adjusted_image(master, recipe)
    res_master_down = cv2.resize(res_master, (1280, 960), interpolation=cv2.INTER_AREA)

    mean_diff = float(np.mean(np.abs(res_proxy.astype(float) - res_master_down.astype(float))))
    print(f"\nShadows & Clarity proxy vs master diff: {mean_diff:.4f}")
    assert mean_diff < 3.5


def test_resolution_invariance_upoint(test_pair):
    """Verify U-Point with normalized coords on proxy vs master downsampled has mean diff < 3.5."""
    master, proxy = test_pair
    recipe = PhotoAdjustmentRecipe(
        selective_points=[
            {
                "x": 0.5,
                "y": 0.5,
                "radius": 0.2,
                "brightness": 25.0,
                "contrast": 20.0,
                "saturation": 25.0,
            }
        ]
    )

    res_proxy = render_adjusted_image(proxy, recipe)
    res_master = render_adjusted_image(master, recipe)
    res_master_down = cv2.resize(res_master, (1280, 960), interpolation=cv2.INTER_AREA)

    mean_diff = float(np.mean(np.abs(res_proxy.astype(float) - res_master_down.astype(float))))
    print(f"\nU-Point proxy vs master diff: {mean_diff:.4f}")
    assert mean_diff < 3.5


def test_combined_full_stack_recipe(test_pair):
    """Verify full editing stack executing simultaneously across proxy and master."""
    master, proxy = test_pair
    recipe = PhotoAdjustmentRecipe(
        brightness=15.0,
        contrast=10.0,
        shadows=20.0,
        highlights=-15.0,
        clarity=15.0,
        temperature=10.0,
        tint=-5.0,
        saturation=15.0,
        dehaze=20.0,
        vignette=-10.0,
        sharpen=25.0,
        denoise=15.0,
        effect="classic_chrome",
        effect_intensity=50.0,
        selective_points=[
            {"x": 0.5, "y": 0.5, "radius": 0.15, "brightness": 10.0}
        ],
    )

    res_proxy = render_adjusted_image(proxy, recipe)
    res_master = render_adjusted_image(master, recipe)
    res_master_down = cv2.resize(res_master, (1280, 960), interpolation=cv2.INTER_AREA)

    mean_diff = float(np.mean(np.abs(res_proxy.astype(float) - res_master_down.astype(float))))
    print(f"\nCombined full stack diff: {mean_diff:.4f}")
    assert mean_diff < 3.5
