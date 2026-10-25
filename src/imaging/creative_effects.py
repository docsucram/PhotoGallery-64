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
    - When face boxes are available, uses an anatomical trimap:
      * Head Ellipse with inner GC_FGD core and outer GC_PR_FGD.
      * Strict background GC_BGD above shoulder level outside head bounds (eliminates tree/car halo).
      * Torso/shoulder polygon extending all the way down to bottom edge y = H - 1 (eliminates bottom seam).
      * Central spine column marked as GC_FGD.
    - Uses GrabCut segmentation followed by morphological cleanup and edge-preserving bilateral filtering.
    """
    if image_bgr is None or image_bgr.size == 0:
        return np.zeros((10, 10), dtype=np.float32)

    h, w = image_bgr.shape[:2]

    # Downsample for fast GrabCut if image is large
    scale = max(1.0, max(h, w) / 500.0)
    sw = max(10, int(w / scale))
    sh = max(10, int(h / scale))
    small_bgr = cv2.resize(image_bgr, (sw, sh), interpolation=cv2.INTER_AREA)

    # Initialize trimap: start with cv2.GC_BGD everywhere
    gc_mask = np.full((sh, sw), cv2.GC_BGD, dtype=np.uint8)

    if face_boxes:
        fx, fy, fw, fh = face_boxes[0]
        s_fx = fx / scale
        s_fy = fy / scale
        s_fw = fw / scale
        s_fh = fh / scale

        cx = int(s_fx + s_fw * 0.5)
        cy = int(s_fy + s_fh * 0.48)
        rx = max(4, int(s_fw * 0.65))
        ry = max(4, int(s_fh * 0.72))

        # Outer head ellipse marked as GC_PR_FGD
        cv2.ellipse(gc_mask, (cx, cy), (int(1.15 * rx), int(1.15 * ry)), 0, 0, 360, cv2.GC_PR_FGD, -1)
        # Inner head ellipse marked as GC_FGD
        cv2.ellipse(gc_mask, (cx, cy), (max(2, int(0.75 * rx)), max(2, int(0.75 * ry))), 0, 0, 360, cv2.GC_FGD, -1)

        # Above shoulder level (y < fy + fh * 1.05), everything outside 1.20*rx MUST remain cv2.GC_BGD
        shoulder_y = min(sh, max(0, int(s_fy + s_fh * 1.05)))
        outside_head = None
        if shoulder_y > 0:
            y_coords, x_coords = np.indices((shoulder_y, sw))
            outside_head = np.abs(x_coords - cx) > (1.20 * rx)
            gc_mask[:shoulder_y, :][outside_head] = cv2.GC_BGD

        # Construct anatomical shoulder/torso polygon starting at neck (y = fy + fh * 0.90, width fw * 0.9)
        # curving to shoulders (y = fy + fh * 1.45, width fw * 3.0) and extending all the way to bottom edge y = sh - 1
        neck_y = min(sh - 1, max(0, int(s_fy + s_fh * 0.90)))
        shld_y = min(sh - 1, max(neck_y + 1, int(s_fy + s_fh * 1.45)))
        torso_pts = np.array([
            [int(cx - s_fw * 0.45), neck_y],
            [int(cx + s_fw * 0.45), neck_y],
            [int(cx + s_fw * 1.50), shld_y],
            [int(cx + s_fw * 1.70), sh - 1],
            [int(cx - s_fw * 1.70), sh - 1],
            [int(cx - s_fw * 1.50), shld_y],
        ], dtype=np.int32)
        torso_pts[:, 0] = np.clip(torso_pts[:, 0], 0, sw - 1)
        torso_pts[:, 1] = np.clip(torso_pts[:, 1], 0, sh - 1)
        cv2.fillPoly(gc_mask, [torso_pts], cv2.GC_PR_FGD)

        # Central spine column (width = fw * 1.6 down to y = sh - 1) marked as GC_FGD
        spine_x1 = max(0, int(cx - s_fw * 0.8))
        spine_x2 = min(sw, int(cx + s_fw * 0.8))
        spine_y1 = min(sh - 1, max(0, int(s_fy + s_fh * 1.0)))
        if spine_x2 > spine_x1 and spine_y1 < sh:
            gc_mask[spine_y1:sh, spine_x1:spine_x2] = cv2.GC_FGD

        # Re-enforce head inner ellipse as GC_FGD
        cv2.ellipse(gc_mask, (cx, cy), (max(2, int(0.75 * rx)), max(2, int(0.75 * ry))), 0, 0, 360, cv2.GC_FGD, -1)
        # Re-enforce above shoulder outside head as GC_BGD
        if shoulder_y > 0 and outside_head is not None:
            gc_mask[:shoulder_y, :][outside_head] = cv2.GC_BGD
    else:
        # Default center crop: center 60% ellipse
        cx, cy = sw // 2, int(sh * 0.55)
        rx, ry = int(sw * 0.30), int(sh * 0.42)
        cv2.ellipse(gc_mask, (cx, cy), (rx, ry), 0, 0, 360, cv2.GC_PR_FGD, -1)
        cv2.ellipse(gc_mask, (cx, cy), (int(rx * 0.5), int(ry * 0.5)), 0, 0, 360, cv2.GC_FGD, -1)

    # Run GrabCut with GC_INIT_WITH_MASK
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(
            small_bgr,
            gc_mask,
            None,
            bgd_model,
            fgd_model,
            iterCount=3,
            mode=cv2.GC_INIT_WITH_MASK,
        )
        binary_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)
    except Exception:
        binary_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)

    # Upscale to full camera resolution
    full_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_LINEAR)

    # Morphological open/close to eliminate isolated pinholes
    mask_u8 = (full_mask * 255.0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)

    # Bilateral and soft Gaussian feathering
    mask_bilat = cv2.bilateralFilter(mask_u8.astype(np.float32) / 255.0, d=7, sigmaColor=0.2, sigmaSpace=5)
    smoothed = cv2.GaussianBlur(mask_bilat, (7, 7), 1.8)
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
    Edge-aware bilateral brush for interactive mask touch-up:
    - Guaranteed responsive painting with smooth radial falloff.
    - Bilateral edge affinity prevents overshooting across strong color boundaries.
    """
    if mask is None or image_bgr is None:
        return mask

    h, w = mask.shape[:2]
    cx, cy = brush_center
    r = max(4, int(radius))

    x1 = max(0, cx - r)
    y1 = max(0, cy - r)
    x2 = min(w, cx + r + 1)
    y2 = min(h, cy + r + 1)

    if x2 <= x1 or y2 <= y1:
        return mask

    roi_img = image_bgr[y1:y2, x1:x2].astype(np.float32)
    sample_col = image_bgr[min(h - 1, max(0, cy)), min(w - 1, max(0, cx))].astype(np.float32)

    # Radial distance and smooth falloff
    ys, xs = np.ogrid[y1 - cy : y2 - cy, x1 - cx : x2 - cx]
    dist = np.sqrt(xs * xs + ys * ys)
    spatial_w = np.clip(1.0 - dist / float(r), 0.0, 1.0)
    spatial_w = spatial_w * spatial_w * (3.0 - 2.0 * spatial_w)

    # Bilateral edge factor (color difference to center point)
    color_diff = np.linalg.norm(roi_img - sample_col, axis=2)
    edge_factor = np.exp(-color_diff / 75.0)

    # Combined alpha ensuring responsive painting with edge-aware feather
    brush_alpha = np.clip(spatial_w * (0.45 + 0.55 * edge_factor), 0.0, 1.0).astype(np.float32)

    updated = mask.copy()
    if mode == "keep_color":
        updated[y1:y2, x1:x2] = updated[y1:y2, x1:x2] * (1.0 - brush_alpha) + 1.0 * brush_alpha
    else:  # "make_bw"
        updated[y1:y2, x1:x2] = updated[y1:y2, x1:x2] * (1.0 - brush_alpha) + 0.0 * brush_alpha

    return np.clip(updated, 0.0, 1.0).astype(np.float32)


# =============================================================================
# 2. Portrait Depth Bokeh & Foreground Fill Light
# =============================================================================

def apply_subject_fill_light(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    fill_light: float = 0.0,
) -> np.ndarray:
    """
    Applies gentle foreground fill light lifting midtones without blowing out highlights:
    - Guided by a heavily feathered subject mask to eliminate harsh boundary seams.
    - Default fill_light = 0.0.
    """
    if image_bgr is None or mask is None or fill_light <= 0.01:
        return image_bgr

    h, w = image_bgr.shape[:2]
    # Heavily feathered mask to avoid hard seams
    sigma_fill = max(15.0, max(w, h) * 0.04)
    feather_mask = cv2.GaussianBlur(mask, (0, 0), sigma_fill)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    v = hsv[:, :, 2]
    # Soft-knee fill light formula
    v_lift = v + fill_light * (255.0 - v) * (v / 255.0)
    hsv[:, :, 2] = np.clip(v * (1.0 - feather_mask) + v_lift * feather_mask, 0.0, 255.0)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_portrait_bokeh(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    bokeh_radius: int = 16,
) -> np.ndarray:
    """
    Renders DSLR-quality Portrait Bokeh with optical lens simulation:
    - Vertical ground-plane depth gradient so ground plane stays naturally sharp.
    - Normalized convolution to prevent dark hair/jacket bleeding into snow/background.
    - Specular highlight boost in linear space (gamma = 2.2) for crisp bokeh discs.
    """
    if image_bgr is None or mask is None:
        return image_bgr

    h, w = image_bgr.shape[:2]

    # 1. Ground plane depth gradient
    y_coords = np.arange(h, dtype=np.float32).reshape(h, 1)
    ground_fg = np.clip((y_coords / float(h) - 0.75) / 0.25, 0.0, 1.0) * 0.35
    effective_mask = np.clip(mask + ground_fg, 0.0, 1.0)

    # 2. Linearize BGR (gamma = 2.2) and specular highlight boost
    bgr_norm = image_bgr.astype(np.float32) / 255.0
    bgr_linear = np.power(bgr_norm, 2.2)

    # Specular boost for L > 220
    lum = 0.114 * image_bgr[:, :, 0] + 0.587 * image_bgr[:, :, 1] + 0.299 * image_bgr[:, :, 2]
    highlight_factor = np.clip((lum - 220.0) / 35.0, 0.0, 1.0)[..., None]
    bgr_linear = bgr_linear * (1.0 + highlight_factor * 1.5)

    # 3. Normalized convolution
    W_bg = np.clip(1.0 - effective_mask, 1e-3, 1.0)
    premult = bgr_linear * W_bg[..., None]

    ksize = max(3, int(bokeh_radius * 2 + 1))
    if ksize % 2 == 0:
        ksize += 1
    sigma = max(1.0, bokeh_radius * 0.65)

    blurred_premult = cv2.GaussianBlur(premult, (ksize, ksize), sigma)
    blurred_weight = cv2.GaussianBlur(W_bg, (ksize, ksize), sigma)[..., None]
    bg_linear = blurred_premult / np.maximum(blurred_weight, 1e-4)

    # De-linearize to sRGB
    bg_bgr = np.clip(np.power(np.clip(bg_linear, 0.0, 1.0), 1.0 / 2.2) * 255.0, 0.0, 255.0)

    # 4. Composite foreground subject over blurred background
    m3 = mask[..., None]
    composite = image_bgr.astype(np.float32) * m3 + bg_bgr * (1.0 - m3)
    return np.clip(composite, 0.0, 255.0).astype(np.uint8)


def apply_portrait_bokeh_and_fill_light(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    bokeh_radius: int = 16,
    fill_light: float = 0.0,
) -> np.ndarray:
    """Combines portrait depth lens blur and foreground fill light."""
    fg = apply_subject_fill_light(image_bgr, mask, fill_light=fill_light)
    return apply_portrait_bokeh(fg, mask, bokeh_radius=bokeh_radius)


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
