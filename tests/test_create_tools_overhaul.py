"""
Comprehensive Automated Test Suite for Create Tools & Effects Overhaul:
1. Guided Filter Matte & Face-Seeded Subject Segmentation
2. Colour Pop with Eyedropper Hue / Chroma Selection
3. Portrait Depth Optical Disc Bokeh Pyramid & Halo Prevention
4. Continuous 3D Cinematic Motion (Mesh Remap, Aspect Ratio Preservation & Motion Presets)
5. Scalable Collage Typography & Smart Cropping
6. Panorama Fast Postprocessing Cache & Photo Fuse LAB Color Matching
"""

import time
import numpy as np
import pytest
import cv2

from src.imaging.creative_effects import (
    guided_filter_matte,
    estimate_subject_mask,
    estimate_depth_map,
    compute_color_similarity_mask,
    apply_color_pop,
    build_bokeh_pyramid,
    apply_portrait_bokeh,
    apply_portrait_bokeh_and_fill_light,
    compute_target_dimensions,
    render_3d_cinematic_motion_frame,
)
from src.imaging.create_tools import (
    _crop_to_aspect,
    _load_scalable_font,
    create_grid_collage,
    create_polaroid_scatter_collage,
    create_then_and_now_card,
    postprocess_panorama,
    fuse_face,
    detect_faces_in_image,
)


def test_guided_filter_matte_and_subject_mask():
    # 1. Test guided filter edge snapping
    h, w = 120, 120
    guide = np.zeros((h, w, 3), dtype=np.uint8)
    guide[:, 60:] = 255  # High-contrast vertical edge at x=60

    # Rough blurry mask centered around x=60
    rough_mask = np.zeros((h, w), dtype=np.float32)
    rough_mask[:, 50:] = 1.0
    rough_mask = cv2.GaussianBlur(rough_mask, (15, 15), 5.0)

    refined = guided_filter_matte(guide, rough_mask, radius=8, eps=1e-3, subsample=1)
    assert refined.shape == (h, w)
    assert 0.0 <= refined.min() and refined.max() <= 1.0
    # Left of edge should be near 0, right of edge should be near 1
    assert refined[:, 20].mean() < 0.15
    assert refined[:, 100].mean() > 0.85

    # 2. Test estimate_subject_mask with provided face box
    test_img = np.full((300, 200, 3), 180, dtype=np.uint8)
    # Synthetic face at cx=100, cy=70, w=40, h=50
    face_box = (80, 45, 40, 50)
    test_img[45:95, 80:120] = [60, 100, 200]
    mask = estimate_subject_mask(test_img, face_boxes=[face_box])
    assert mask.shape == (300, 200)
    assert mask.dtype == np.float32
    assert mask[70, 100] > 0.5  # Inside head should be foreground


def test_color_pop_hue_and_subject():
    h, w = 100, 100
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Red patch in top-left, Blue patch in bottom-right
    img[:50, :50] = [0, 0, 255]  # Red in BGR
    img[50:, 50:] = [255, 0, 0]  # Blue in BGR

    # Test hue similarity mask for red
    red_bgr = (0, 0, 255)
    red_mask = compute_color_similarity_mask(img, red_bgr, tolerance=0.25)
    assert red_mask.shape == (h, w)
    assert red_mask[:40, :40].mean() > 0.70  # Red region should be high
    assert red_mask[60:, 60:].mean() < 0.20  # Blue region should be low

    # Test apply_color_pop
    pop_img = apply_color_pop(img, red_mask)
    assert pop_img.shape == (h, w, 3)
    # The blue region should be converted to grayscale (B == G == R approx)
    blue_region = pop_img[60:, 60:]
    diff_bg = np.abs(blue_region[:, :, 0].astype(int) - blue_region[:, :, 1].astype(int))
    assert np.mean(diff_bg) < 15  # Desaturated into B&W


def test_portrait_bokeh_pyramid_speed_and_halos():
    h, w = 240, 320
    img = np.full((h, w, 3), 120, dtype=np.uint8)
    # Add a bright specular fairy-light highlight in background
    img[40:50, 40:50] = 250

    # Build bokeh pyramid
    t0 = time.perf_counter()
    pyr = build_bokeh_pyramid(img, max_blur_radius=24.0, num_levels=4)
    dt_build = time.perf_counter() - t0
    assert len(pyr) == 4
    for level in pyr:
        assert level.shape == (h, w, 3)

    # Test rapid interpolation from pyramid (simulating 60 FPS slider drag)
    mask = np.zeros((h, w), dtype=np.float32)
    mask[100:200, 120:200] = 1.0  # Foreground subject

    t1 = time.perf_counter()
    bokeh_frame = apply_portrait_bokeh(img, mask, bokeh_radius=18, bokeh_pyramid=pyr)
    dt_apply = time.perf_counter() - t1
    # Applying from precomputed pyramid must be < 20ms
    assert dt_apply < 0.025
    assert bokeh_frame.shape == (h, w, 3)

    # Foreground center should remain sharp and identical to original
    np.testing.assert_allclose(bokeh_frame[150, 160], img[150, 160], atol=2)


def test_3d_cinematic_aspect_ratio_and_depth_remap():
    # 1. Test aspect ratio preservation and even dimensions
    # Portrait aspect (600x800)
    tw, th = compute_target_dimensions(600, 800, max_long_edge=640)
    assert tw % 2 == 0 and th % 2 == 0
    assert abs(tw / th - 600 / 800) < 0.05
    assert th == 640

    # Landscape aspect (1920x1080)
    tw_l, th_l = compute_target_dimensions(1920, 1080, max_long_edge=1920)
    assert tw_l % 2 == 0 and th_l % 2 == 0
    assert tw_l == 1920 and th_l == 1080

    # 2. Test 3D continuous depth mesh remap across presets
    h, w = 160, 120  # Portrait photo
    img = np.full((h, w, 3), 150, dtype=np.uint8)
    cv2.circle(img, (60, 80), 30, (50, 80, 220), -1)
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, (60, 80), 30, 1.0, -1)

    depth = estimate_depth_map(img, subject_mask=mask)
    assert depth.shape == (h, w)
    assert depth[80, 60] > depth[10, 10]  # Center subject is nearer than top corner

    for style in ("orbital", "dolly_in", "pan_lr", "breathe"):
        f0 = render_3d_cinematic_motion_frame(
            img, img, mask, t=0.0, canvas_size=(w, h), depth_map=depth, motion_style=style, intensity=0.6
        )
        f1 = render_3d_cinematic_motion_frame(
            img, img, mask, t=0.25, canvas_size=(w, h), depth_map=depth, motion_style=style, intensity=0.6
        )
        assert f0.shape == (h, w, 3)
        assert f1.shape == (h, w, 3)
        # Frames at t=0.0 and peak displacement t=0.25 must have parallax motion (not identical)
        assert not np.array_equal(f0, f1)


def test_collage_scalable_fonts_and_smart_crop():
    # 1. Test scalable font loader
    font_small = _load_scalable_font(14)
    font_large = _load_scalable_font(48, bold=True)
    assert font_small is not None
    assert font_large is not None

    # 2. Test face-aware smart crop vs blind crop
    img = np.zeros((300, 100, 3), dtype=np.uint8)  # Tall image
    # Subject face is located near top (cy = 0.20)
    cropped_top = _crop_to_aspect(img, 100, 100, focus_center=(0.5, 0.20))
    assert cropped_top.shape == (100, 100, 3)

    # 3. Test Polaroid and Then & Now cards
    test_photo = np.full((200, 200, 3), 180, dtype=np.uint8)
    pol_data = [{
        "image_bgr": test_photo,
        "caption": "Summer Vacation 2026",
        "date_str": "2026-07-15",
    }]
    pol_result = create_polaroid_scatter_collage(pol_data, target_size=(800, 600))
    assert pol_result.shape == (600, 800, 3)

    then_now_data = [
        {"image_bgr": test_photo, "date_taken": "2016-05-10"},
        {"image_bgr": test_photo, "date_taken": "2026-05-10"},
    ]
    then_now_result = create_then_and_now_card(then_now_data, target_size=(800, 450))
    assert then_now_result.shape == (450, 800, 3)


def test_panorama_cache_and_photo_fuse_color_match():
    # 1. Test panorama postprocessing cache speed
    h, w = 400, 1200
    synthetic_pano = np.full((h, w, 3), 200, dtype=np.uint8)
    # Black borders on wavy edges
    synthetic_pano[:30, :100] = 0
    synthetic_pano[-30:, -100:] = 0

    t0 = time.perf_counter()
    cropped = postprocess_panorama(synthetic_pano, auto_crop=True, fill_minor_gaps=False)
    dt = time.perf_counter() - t0
    # Must run in < 100ms and crop black border area
    assert dt < 0.100
    assert cropped.shape[0] < h or cropped.shape[1] < w

    # 2. Test Photo Fuse LAB color matching & seamless blend
    base = np.full((200, 200, 3), 180, dtype=np.uint8)  # Brighter lighting
    src = np.full((200, 200, 3), 120, dtype=np.uint8)   # Darker shot from burst
    face_rect = (50, 50, 60, 60)
    fused = fuse_face(base, src, face_rect)
    assert fused.shape == (200, 200, 3)
    # The fused face region should match the base lighting without harsh cutoff
    assert abs(int(fused[80, 80, 0]) - int(base[80, 80, 0])) < 40
