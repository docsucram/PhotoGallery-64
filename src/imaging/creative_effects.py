"""
Creative Effects Engine:
1. Google Photos-Style 'Color Pop' (Subject Color Splash) with bilateral edge-aware brush snapping.
2. Portrait Depth Bokeh (spatially varying lens blur) & Foreground Fill Light.
3. '3D Cinematic Motion' (Faux-3D Parallax MP4 video generator with background disocclusion inpainting).
4. Privacy Redaction (Mosaic pixelation and Gaussian blur).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
import cv2
import numpy as np


# =============================================================================
# 1. Google Photos-Style "Color Pop" (Subject Color Splash)
# =============================================================================

def estimate_subject_mask(
    image_bgr: np.ndarray,
    face_boxes: list[tuple[int, int, int, int]] | None = None,
) -> np.ndarray:
    """
    Estimates a smooth float32 mask in [0.0, 1.0] isolating the main subject/person:
    - Uses face boxes to initialize foreground bounding region (extended to torso).
    - Uses GrabCut segmentation followed by guided / bilateral smoothing for hair and soft edges.
    """
    if image_bgr is None or image_bgr.size == 0:
        return np.zeros((10, 10), dtype=np.float32)

    h, w = image_bgr.shape[:2]

    # Downsample for fast GrabCut if image is large
    scale = max(1.0, max(h, w) / 500.0)
    sw = int(w / scale)
    sh = int(h / scale)
    small_bgr = cv2.resize(image_bgr, (sw, sh), interpolation=cv2.INTER_AREA)

    # 1. Determine bounding box for subject
    rect_to_use: tuple[int, int, int, int] | None = None
    if face_boxes:
        # Scale face box
        fx, fy, fw, fh = face_boxes[0]
        s_fx = int(fx / scale)
        s_fy = int(fy / scale)
        s_fw = int(fw / scale)
        s_fh = int(fh / scale)

        # Extend face downwards to include neck, shoulders, and torso
        bx = max(2, s_fx - int(s_fw * 0.6))
        by = max(2, s_fy - int(s_fh * 0.2))
        bw = min(sw - bx - 4, int(s_fw * 2.2))
        bh = min(sh - by - 4, int(s_fh * 3.6))
        if bw > 20 and bh > 20:
            rect_to_use = (bx, by, bw, bh)

    if rect_to_use is None:
        # Default center crop: center 60% of image
        margin_x = int(sw * 0.18)
        margin_y = int(sh * 0.12)
        rect_to_use = (margin_x, margin_y, sw - 2 * margin_x, sh - 2 * margin_y)

    # Run GrabCut
    gc_mask = np.zeros((sh, sw), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(
            small_bgr,
            gc_mask,
            rect_to_use,
            bgd_model,
            fgd_model,
            iterCount=2,
            mode=cv2.GC_INIT_WITH_RECT,
        )
        binary_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)
    except Exception:
        # Elliptical fallback mask
        binary_mask = np.zeros((sh, sw), dtype=np.float32)
        rx, ry, rw, rh = rect_to_use
        cv2.ellipse(
            binary_mask,
            (rx + rw // 2, ry + rh // 2),
            (rw // 2, rh // 2),
            0,
            0,
            360,
            1.0,
            -1,
        )

    # Upscale to original resolution
    full_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_LINEAR)
    # Apply bilateral smoothing / soft feathering
    smoothed = cv2.GaussianBlur(full_mask, (15, 15), 0)
    return np.clip(smoothed, 0.0, 1.0).astype(np.float32)


def apply_color_pop(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    bg_contrast: float = 1.15,
    bg_brightness: float = 0.0,
) -> np.ndarray:
    """
    Produces the signature Color Pop effect:
    - Subject (where mask=1.0) remains in full vibrant color.
    - Background (where mask=0.0) is converted to crisp high-contrast B&W.
    """
    if image_bgr is None or mask is None:
        return image_bgr

    # Background B&W with contrast adjustment
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_adj = np.clip((gray - 128.0) * bg_contrast + 128.0 + bg_brightness, 0.0, 255.0)
    bw_bgr = cv2.cvtColor(gray_adj.astype(np.uint8), cv2.COLOR_GRAY2BGR).astype(np.float32)

    fg_bgr = image_bgr.astype(np.float32)
    m3 = mask[:, :, None]

    blended = fg_bgr * m3 + bw_bgr * (1.0 - m3)
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def refine_mask_brush(
    mask: np.ndarray,
    image_bgr: np.ndarray,
    brush_center: tuple[int, int],
    radius: int = 24,
    mode: str = "keep_color",
) -> np.ndarray:
    """
    Edge-aware bilateral brush for interactive mask touch-up.
    Snaps to color edges so the user does not have to paint carefully along contours.
    """
    if mask is None or image_bgr is None:
        return mask

    h, w = mask.shape[:2]
    cx, cy = brush_center
    r = max(4, radius)

    x1 = max(0, cx - r)
    y1 = max(0, cy - r)
    x2 = min(w, cx + r + 1)
    y2 = min(h, cy + r + 1)

    if x2 <= x1 or y2 <= y1:
        return mask

    roi_img = image_bgr[y1:y2, x1:x2].astype(np.float32)
    sample_col = image_bgr[min(h - 1, max(0, cy)), min(w - 1, max(0, cx))].astype(np.float32)

    # Spatial Gaussian weight
    ys, xs = np.ogrid[y1 - cy : y2 - cy, x1 - cx : x2 - cx]
    dist_sq = xs * xs + ys * ys
    spatial_w = np.exp(-dist_sq / (2.0 * (r * 0.5) ** 2))
    spatial_w[dist_sq > r * r] = 0.0

    # Color difference weight (bilateral edge snapping)
    color_diff = np.linalg.norm(roi_img - sample_col, axis=2)
    color_w = np.exp(-color_diff / 45.0)

    brush_weight = (spatial_w * color_w).astype(np.float32)

    updated = mask.copy()
    if mode == "keep_color":
        updated[y1:y2, x1:x2] = np.maximum(updated[y1:y2, x1:x2], brush_weight)
    else:  # "make_bw"
        updated[y1:y2, x1:x2] = np.minimum(updated[y1:y2, x1:x2], 1.0 - brush_weight)

    return np.clip(updated, 0.0, 1.0).astype(np.float32)


# =============================================================================
# 2. Portrait Depth Bokeh & Foreground Fill Light
# =============================================================================

def apply_portrait_bokeh_and_fill_light(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    bokeh_radius: int = 16,
    fill_light: float = 0.22,
) -> np.ndarray:
    """
    Renders DSLR-quality Portrait Bokeh:
    - Blurs background behind subject using spatially smooth lens blur.
    - Applies a subtle 'Face / Subject Fill Light' lifting shadows and midtones on the subject.
    """
    if image_bgr is None or mask is None:
        return image_bgr

    h, w = image_bgr.shape[:2]

    # 1. Background lens blur (disk kernel simulation with Gaussian)
    ksize = max(3, bokeh_radius * 2 + 1)
    if ksize % 2 == 0:
        ksize += 1
    bg_blurred = cv2.GaussianBlur(image_bgr, (ksize, ksize), bokeh_radius * 0.6)

    # 2. Subject Fill Light: lift midtones without clipping highlights
    fg = image_bgr.copy()
    if fill_light > 0.01:
        hsv = cv2.cvtColor(fg, cv2.COLOR_BGR2HSV).astype(np.float32)
        v = hsv[:, :, 2]
        # Soft-knee fill light formula
        v_lift = v + fill_light * (255.0 - v) * (v / 255.0)
        hsv[:, :, 2] = np.clip(v_lift, 0.0, 255.0)
        fg = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 3. Composite subject over blurred background
    m3 = mask[:, :, None]
    blended = fg.astype(np.float32) * m3 + bg_blurred.astype(np.float32) * (1.0 - m3)
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


# =============================================================================
# 3. 3D Cinematic Motion (Faux-3D Parallax Video Generator)
# =============================================================================

def inpaint_disocclusion_background(
    image_bgr: np.ndarray,
    subject_mask: np.ndarray,
) -> np.ndarray:
    """
    Inpaints the background behind the subject so parallax virtual camera moves
    reveal realistic background content rather than black holes.
    """
    h, w = image_bgr.shape[:2]
    # Dilate subject mask by 12px for clean boundary coverage
    bin_mask = (subject_mask > 0.35).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dilated_mask = cv2.dilate(bin_mask, kernel)

    # Inpaint background using Telea
    # If image is large, downsample inpaint for performance and upsample
    scale = max(1.0, max(h, w) / 600.0)
    if scale > 1.2:
        sw, sh = int(w / scale), int(h / scale)
        small_img = cv2.resize(image_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(dilated_mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
        inpainted_small = cv2.inpaint(small_img, small_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        return cv2.resize(inpainted_small, (w, h), interpolation=cv2.INTER_LINEAR)

    return cv2.inpaint(image_bgr, dilated_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)


def render_3d_cinematic_motion_frame(
    foreground_bgr: np.ndarray,
    background_bgr: np.ndarray,
    subject_mask: np.ndarray,
    t: float,  # Normalized time in [0.0, 1.0]
    canvas_size: tuple[int, int] = (1920, 1080),
) -> np.ndarray:
    """
    Renders a single frame of the 3D cinematic virtual camera parallax motion:
    - Executes virtual camera dolly/pan: foreground moves slightly faster than background.
    """
    tw, th = canvas_size
    h, w = foreground_bgr.shape[:2]

    # Virtual camera trajectory (smooth sinusoidal dolly + slow push-in zoom)
    # Camera drifts right-to-left and gently zooms in by 4%
    pan_x = math.sin(t * math.pi) * 25.0
    pan_y = math.sin(t * math.pi * 0.5) * 8.0
    zoom = 1.0 + 0.04 * t

    # 1. Background transform (far layer moves with 0.35 parallax factor)
    bg_scale = zoom * 1.02
    bg_dx = pan_x * 0.35
    bg_dy = pan_y * 0.35

    M_bg = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 0.0, bg_scale)
    M_bg[0, 2] += bg_dx + (tw - w) / 2.0
    M_bg[1, 2] += bg_dy + (th - h) / 2.0
    warped_bg = cv2.warpAffine(background_bgr, M_bg, (tw, th), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # 2. Foreground subject transform (near layer moves with 1.0 parallax factor)
    fg_scale = zoom * 1.05
    fg_dx = pan_x * 1.0
    fg_dy = pan_y * 1.0

    M_fg = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 0.0, fg_scale)
    M_fg[0, 2] += fg_dx + (tw - w) / 2.0
    M_fg[1, 2] += fg_dy + (th - h) / 2.0
    warped_fg = cv2.warpAffine(foreground_bgr, M_fg, (tw, th), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    warped_mask = cv2.warpAffine(subject_mask, M_fg, (tw, th), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # 3. Composite foreground over background
    m3 = warped_mask[:, :, None]
    composite = warped_fg.astype(np.float32) * m3 + warped_bg.astype(np.float32) * (1.0 - m3)
    return np.clip(composite, 0.0, 255.0).astype(np.uint8)


def render_3d_cinematic_motion_video(
    image_bgr: np.ndarray,
    output_path: Path | str,
    duration_sec: float = 4.0,
    fps: int = 30,
    resolution: tuple[int, int] = (1920, 1080),
    subject_mask: np.ndarray | None = None,
    progress_callback: Any | None = None,
) -> bool:
    """
    Renders a 3D Cinematic Motion .mp4 video from a single photo.
    """
    if image_bgr is None:
        return False

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if subject_mask is None:
        subject_mask = estimate_subject_mask(image_bgr)

    # Inpaint background behind subject
    bg_clean = inpaint_disocclusion_background(image_bgr, subject_mask)

    total_frames = int(duration_sec * fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_p), fourcc, float(fps), resolution)

    if not writer.isOpened():
        return False

    try:
        for frame_idx in range(total_frames):
            t = frame_idx / float(max(1, total_frames - 1))
            frame = render_3d_cinematic_motion_frame(
                foreground_bgr=image_bgr,
                background_bgr=bg_clean,
                subject_mask=subject_mask,
                t=t,
                canvas_size=resolution,
            )
            writer.write(frame)
            if progress_callback:
                progress_callback(int((frame_idx + 1) * 100 / total_frames))
    finally:
        writer.release()

    return out_p.exists() and out_p.stat().st_size > 1000


# =============================================================================
# 4. Privacy Redaction (Mosaic Pixelate & Gaussian Blur)
# =============================================================================

def apply_pixelate(
    image_bgr: np.ndarray,
    rect: tuple[int, int, int, int],
    block_size: int = 16,
) -> np.ndarray:
    """Applies heavy privacy mosaic pixelation over rect (x, y, w, h)."""
    if image_bgr is None:
        return image_bgr

    h, w = image_bgr.shape[:2]
    rx, ry, rw, rh = rect
    x1, y1 = max(0, rx), max(0, ry)
    x2, y2 = min(w, rx + rw), min(h, ry + rh)
    if x2 <= x1 or y2 <= y1:
        return image_bgr

    result = image_bgr.copy()
    roi = result[y1:y2, x1:x2]
    bs = max(4, block_size)

    # Downscale and upscale with nearest neighbor
    small_w = max(1, (x2 - x1) // bs)
    small_h = max(1, (y2 - y1) // bs)
    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

    result[y1:y2, x1:x2] = pixelated
    return result


def apply_blur_redact(
    image_bgr: np.ndarray,
    rect: tuple[int, int, int, int],
    radius: int = 25,
) -> np.ndarray:
    """Applies heavy Gaussian privacy blur over rect (x, y, w, h)."""
    if image_bgr is None:
        return image_bgr

    h, w = image_bgr.shape[:2]
    rx, ry, rw, rh = rect
    x1, y1 = max(0, rx), max(0, ry)
    x2, y2 = min(w, rx + rw), min(h, ry + rh)
    if x2 <= x1 or y2 <= y1:
        return image_bgr

    result = image_bgr.copy()
    roi = result[y1:y2, x1:x2]
    ksize = max(3, radius * 2 + 1)
    if ksize % 2 == 0:
        ksize += 1

    blurred = cv2.GaussianBlur(roi, (ksize, ksize), 0)
    result[y1:y2, x1:x2] = blurred
    return result
