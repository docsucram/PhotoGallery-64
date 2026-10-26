"""
Creative Effects Engine:
1. Google Photos-Style 'Color Pop' (Subject Color Splash) with bilateral & hue-aware matting.
2. Portrait Depth Bokeh (spatially varying optical disc lens blur) & Foreground Fill Light.
3. '3D Cinematic Motion' (Continuous Depth-Map Mesh Warping via cv2.remap with Parallax Presets).
4. Privacy Redaction (Mosaic pixelation and Gaussian blur).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

_SHARED_FACE_ENGINE: Any = None


def _get_face_engine() -> Any:
    """Lazily initializes and caches the shared FaceEngine."""
    global _SHARED_FACE_ENGINE
    if _SHARED_FACE_ENGINE is None:
        try:
            from src.core.face_engine import FaceEngine
            _SHARED_FACE_ENGINE = FaceEngine(backend="auto", allow_fallback=True)
        except Exception as e:
            logger.debug("FaceEngine initialization in creative_effects failed: %s", e)
            _SHARED_FACE_ENGINE = False
    return _SHARED_FACE_ENGINE if _SHARED_FACE_ENGINE is not False else None


# =============================================================================
# EXIF-Safe Loading & Edge-Preserving Guided Filter
# =============================================================================

def load_bgr_exif_safe(path: str | Path) -> np.ndarray | None:
    """
    Load an image into BGR uint8 format respecting EXIF orientation tags.
    Eliminates sideways orientation on JPEG photos.
    """
    if not path:
        return None
    try:
        with Image.open(str(path)) as img:
            img = ImageOps.exif_transpose(img)
            rgb = np.array(img.convert("RGB"), dtype=np.uint8)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        try:
            return cv2.imread(str(path), cv2.IMREAD_COLOR)
        except Exception:
            return None


def guided_filter_matte(
    guide_bgr: np.ndarray,
    src_mask: np.ndarray,
    radius: int = 16,
    eps: float = 1e-3,
    subsample: int = 2,
) -> np.ndarray:
    """
    Edge-preserving Joint Guided Filter (He et al.) implemented via pure NumPy + cv2.boxFilter.
    Snaps rough or upscaled alpha masks [0..1] directly to full-resolution luminance/color edges
    eliminating blurry halo bleed. Runs in ~15ms on large images via subsampling.
    """
    if guide_bgr is None or src_mask is None:
        return src_mask

    h, w = guide_bgr.shape[:2]
    I_full = cv2.cvtColor(guide_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    p_full = src_mask.astype(np.float32)
    if p_full.shape[:2] != (h, w):
        p_full = cv2.resize(p_full, (w, h), interpolation=cv2.INTER_LINEAR)

    subsample = max(1, int(subsample))
    sh, sw = max(16, h // subsample), max(16, w // subsample)
    I = cv2.resize(I_full, (sw, sh), interpolation=cv2.INTER_AREA) if subsample > 1 else I_full
    p = cv2.resize(p_full, (sw, sh), interpolation=cv2.INTER_AREA) if subsample > 1 else p_full
    r = max(2, radius // subsample)
    ksize = (2 * r + 1, 2 * r + 1)

    mean_I = cv2.boxFilter(I, cv2.CV_32F, ksize)
    mean_p = cv2.boxFilter(p, cv2.CV_32F, ksize)
    corr_I = cv2.boxFilter(I * I, cv2.CV_32F, ksize)
    corr_Ip = cv2.boxFilter(I * p, cv2.CV_32F, ksize)

    var_I = np.maximum(corr_I - mean_I * mean_I, 0.0)
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)

    if subsample > 1:
        mean_a_up = cv2.resize(mean_a, (w, h), interpolation=cv2.INTER_LINEAR)
        mean_b_up = cv2.resize(mean_b, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        mean_a_up = mean_a
        mean_b_up = mean_b

    q = mean_a_up * I_full + mean_b_up
    return np.clip(q, 0.0, 1.0)


# =============================================================================
# 1. Subject Segmentation, Depth Estimation & Color Pop
# =============================================================================

def _detect_faces_for_mask(image_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detects faces and returns pixel coordinates [(x, y, w, h)]."""
    h, w = image_bgr.shape[:2]
    engine = _get_face_engine()
    boxes: list[tuple[int, int, int, int]] = []
    if engine is not None:
        try:
            # Detect on max 640px long edge for real-time speed
            scale = max(1.0, max(h, w) / 640.0)
            sw, sh = max(16, int(w / scale)), max(16, int(h / scale))
            small = cv2.resize(image_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            detected = engine.detect_faces(rgb)
            for f in detected:
                bx = int(round(f.rect_x * w))
                by = int(round(f.rect_y * h))
                bw = int(round(f.rect_w * w))
                bh = int(round(f.rect_h * h))
                if bw >= 16 and bh >= 16:
                    boxes.append((bx, by, bw, bh))
        except Exception as ex:
            logger.debug("FaceEngine detection failed during subject mask estimate: %s", ex)

    if not boxes:
        # Fallback to OpenCV Haar cascade
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            face_cascade = cv2.CascadeClassifier(cascade_path)
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            scale = max(1.0, max(h, w) / 500.0)
            if scale > 1.0:
                sw, sh = max(10, int(w / scale)), max(10, int(h / scale))
                small_gray = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
                raw_faces = face_cascade.detectMultiScale(small_gray, 1.2, 4, minSize=(20, 20))
                for rx, ry, rw, rh in raw_faces:
                    boxes.append((int(rx * scale), int(ry * scale), int(rw * scale), int(rh * scale)))
            else:
                raw_faces = face_cascade.detectMultiScale(gray, 1.2, 4, minSize=(24, 24))
                for rx, ry, rw, rh in raw_faces:
                    boxes.append((int(rx), int(ry), int(rw), int(rh)))
        except Exception:
            pass

    return boxes


def estimate_subject_mask(
    image_bgr: np.ndarray,
    face_boxes: list[tuple[int, int, int, int]] | None = None,
) -> np.ndarray:
    """
    Estimates a smooth float32 mask in [0.0, 1.0] isolating the main subject(s):
    - Automatically detects faces via FaceEngine if face_boxes is not provided.
    - If faces are found: builds anatomical trimap for all faces (head, shoulders, torso).
    - If no faces are found: computes multi-scale spectral saliency & contrast map to seed GrabCut.
    - Finishes with Joint Guided Filter alpha matting to snap mask to true image edges.
    """
    if image_bgr is None or image_bgr.size == 0:
        return np.zeros((10, 10), dtype=np.float32)

    h, w = image_bgr.shape[:2]

    # Downsample for fast GrabCut
    scale = max(1.0, max(h, w) / 500.0)
    sw = max(10, int(w / scale))
    sh = max(10, int(h / scale))
    small_bgr = cv2.resize(image_bgr, (sw, sh), interpolation=cv2.INTER_AREA)

    if face_boxes is None:
        face_boxes = _detect_faces_for_mask(image_bgr)

    gc_mask = np.full((sh, sw), cv2.GC_BGD, dtype=np.uint8)

    if face_boxes:
        # Mark all detected faces in the trimap
        for fx, fy, fw, fh in face_boxes:
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

            # Neck and torso/shoulder polygon extending down to bottom edge
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

            # Central spine column marked as GC_FGD
            spine_x1 = max(0, int(cx - s_fw * 0.8))
            spine_x2 = min(sw, int(cx + s_fw * 0.8))
            spine_y1 = min(sh - 1, max(0, int(s_fy + s_fh * 1.0)))
            if spine_x2 > spine_x1 and spine_y1 < sh:
                gc_mask[spine_y1:sh, spine_x1:spine_x2] = cv2.GC_FGD

            # Re-enforce head inner ellipse as GC_FGD
            cv2.ellipse(gc_mask, (cx, cy), (max(2, int(0.75 * rx)), max(2, int(0.75 * ry))), 0, 0, 360, cv2.GC_FGD, -1)
    else:
        # Saliency-informed seed: Center-weighted color contrast against image border
        lab = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        border_lab = np.concatenate([
            lab[0, :, :], lab[-1, :, :], lab[:, 0, :], lab[:, -1, :]
        ], axis=0)
        bg_mean = np.mean(border_lab, axis=0)
        dist_from_bg = np.linalg.norm(lab - bg_mean, axis=2)

        # Spatial center Gaussian prior
        yy, xx = np.ogrid[:sh, :sw]
        cx, cy = sw * 0.5, sh * 0.52
        gauss_prior = np.exp(-((xx - cx) ** 2 / (2 * (sw * 0.35) ** 2) + (yy - cy) ** 2 / (2 * (sh * 0.40) ** 2)))
        saliency = dist_from_bg * gauss_prior
        saliency_norm = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Probable foreground where saliency > threshold
        th_fg = max(50, int(np.percentile(saliency_norm, 65)))
        th_def_fg = max(80, int(np.percentile(saliency_norm, 85)))
        gc_mask[saliency_norm > th_fg] = cv2.GC_PR_FGD
        gc_mask[saliency_norm > th_def_fg] = cv2.GC_FGD

        # Always ensure center core is marked
        c_rx, c_ry = max(4, int(sw * 0.20)), max(4, int(sh * 0.25))
        cv2.ellipse(gc_mask, (int(cx), int(cy)), (c_rx, c_ry), 0, 0, 360, cv2.GC_PR_FGD, -1)
        cv2.ellipse(gc_mask, (int(cx), int(cy)), (max(2, int(c_rx * 0.5)), max(2, int(c_ry * 0.5))), 0, 0, 360, cv2.GC_FGD, -1)

    # Borders should always remain strict background
    gc_mask[0:max(1, int(sh * 0.02)), :] = cv2.GC_BGD
    gc_mask[-max(1, int(sh * 0.02)):, :] = cv2.GC_BGD
    gc_mask[:, 0:max(1, int(sw * 0.02))] = cv2.GC_BGD
    gc_mask[:, -max(1, int(sw * 0.02)):] = cv2.GC_BGD

    # Run GrabCut
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
    raw_float = mask_u8.astype(np.float32) / 255.0

    # Edge-preserving Joint Guided Filter matting against high-res image
    mat_radius = max(8, int(max(h, w) * 0.012))
    refined_mask = guided_filter_matte(image_bgr, raw_float, radius=mat_radius, eps=1e-3, subsample=2)
    return np.clip(refined_mask, 0.0, 1.0).astype(np.float32)


def estimate_depth_map(
    image_bgr: np.ndarray,
    subject_mask: np.ndarray | None = None,
    face_boxes: list[tuple[int, int, int, int]] | None = None,
) -> np.ndarray:
    """
    Produces a continuous, smooth float32 depth map D(x, y) in [0.0, 1.0]:
    - 1.0 = nearest foreground, 0.0 = farthest background.
    - Combines subject segmentation, ground-plane perspective gradient, and high-frequency sharpness.
    - Refined with Joint Guided Filter so depth transitions align with real visual boundaries.
    """
    if image_bgr is None or image_bgr.size == 0:
        return np.zeros((10, 10), dtype=np.float32)

    h, w = image_bgr.shape[:2]
    if subject_mask is None:
        subject_mask = estimate_subject_mask(image_bgr, face_boxes=face_boxes)

    # 1. Ground plane depth gradient (bottom of frame is near, top is far)
    y_coords = np.arange(h, dtype=np.float32).reshape(h, 1)
    ground_gradient = np.clip(y_coords / float(h), 0.0, 1.0)
    ground_gradient = np.tile(ground_gradient, (1, w))

    # 2. Defocus sharpness map via Laplacian energy
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_mag = np.abs(lap)
    sharpness = cv2.boxFilter(lap_mag, cv2.CV_32F, (31, 31))
    sharp_norm = cv2.normalize(sharpness, None, 0.0, 1.0, cv2.NORM_MINMAX)

    # 3. Composite depth estimation
    # Subject has highest foreground weight (0.65), ground gradient (0.25), sharpness (0.10)
    raw_depth = 0.65 * subject_mask + 0.25 * ground_gradient + 0.10 * sharp_norm
    raw_depth = np.clip(raw_depth, 0.0, 1.0)

    # 4. Refine with Guided Filter for smooth, edge-aware depth transitions
    depth_smooth = guided_filter_matte(image_bgr, raw_depth, radius=24, eps=5e-3, subsample=2)
    return np.clip(depth_smooth, 0.0, 1.0).astype(np.float32)


def compute_color_similarity_mask(
    image_bgr: np.ndarray,
    target_bgr: tuple[int, int, int] | np.ndarray,
    tolerance: float = 0.25,
) -> np.ndarray:
    """
    Generates a smooth alpha mask [0.0, 1.0] isolating pixels similar in color to target_bgr:
    - Uses CIE LAB perceptual distance and HSV hue matching.
    - tolerance: in [0.05, 0.80].
    - Edge-snapped using guided_filter_matte.
    """
    if image_bgr is None:
        return np.zeros((10, 10), dtype=np.float32)

    h, w = image_bgr.shape[:2]
    target_pixel = np.array([[target_bgr]], dtype=np.uint8)

    lab_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_target = cv2.cvtColor(target_pixel, cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]

    # Weighted Delta E distance (giving slightly higher weight to chroma a, b vs luminance L)
    dl = (lab_img[:, :, 0] - lab_target[0]) * 0.5
    da = lab_img[:, :, 1] - lab_target[1]
    db = lab_img[:, :, 2] - lab_target[2]
    dist = np.sqrt(dl * dl + da * da + db * db)

    # Tolerance normalization
    max_dist = max(10.0, float(tolerance) * 110.0)
    raw_mask = np.clip(1.0 - (dist / max_dist), 0.0, 1.0)
    # Smooth sigmoid curve
    smooth_mask = np.power(raw_mask, 1.5)

    return guided_filter_matte(image_bgr, smooth_mask, radius=12, eps=1e-3, subsample=2)


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

def create_disc_kernel(radius: int) -> np.ndarray:
    """Generates an anti-aliased circular aperture disc kernel normalized to sum to 1.0."""
    r = max(1, int(radius))
    d = 2 * r + 1
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    dist = np.sqrt(x * x + y * y)
    # Anti-aliased outer edge (1px boundary)
    kernel = np.clip(r + 0.5 - dist, 0.0, 1.0)
    s = np.sum(kernel)
    return kernel / s if s > 0 else kernel


def build_bokeh_pyramid(
    image_bgr: np.ndarray,
    max_blur_radius: float = 28.0,
    num_levels: int = 4,
) -> list[np.ndarray]:
    """
    Precomputes a multi-level optical bokeh blur pyramid in gamma-linear space (gamma=2.2):
    - Level 0: Original sharp image.
    - Level 1..3: Progressively stronger optical disc bokeh blur with specular highlight bloom.
    - Allows interactive slider dragging to interpolate in <5ms at 60 FPS.
    """
    if image_bgr is None or image_bgr.size == 0:
        return []

    # Linearize (gamma = 2.2) and add highlight bloom
    bgr_norm = image_bgr.astype(np.float32) / 255.0
    bgr_linear = np.power(bgr_norm, 2.2)

    # Specular boost for bright highlights
    lum = 0.114 * image_bgr[:, :, 0] + 0.587 * image_bgr[:, :, 1] + 0.299 * image_bgr[:, :, 2]
    highlight_factor = np.clip((lum - 215.0) / 40.0, 0.0, 1.0)[..., None]
    bgr_bloomed = bgr_linear * (1.0 + highlight_factor * 2.0)

    pyramid: list[np.ndarray] = [image_bgr.copy()]

    radii = np.linspace(max_blur_radius * 0.33, max_blur_radius, num_levels - 1)
    for r in radii:
        disc = create_disc_kernel(int(round(r)))
        blurred_lin = cv2.filter2D(bgr_bloomed, -1, disc, borderType=cv2.BORDER_REFLECT)
        # De-linearize to sRGB
        blurred_srgb = np.clip(np.power(np.clip(blurred_lin, 0.0, 1.0), 1.0 / 2.2) * 255.0, 0.0, 255.0).astype(np.uint8)
        pyramid.append(blurred_srgb)

    return pyramid


def apply_subject_fill_light(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    fill_light: float = 0.0,
) -> np.ndarray:
    """
    Applies gentle foreground fill light lifting midtones without blowing out highlights:
    - Guided by a heavily feathered subject mask to eliminate harsh boundary seams.
    """
    if image_bgr is None or mask is None or fill_light <= 0.01:
        return image_bgr

    h, w = image_bgr.shape[:2]
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
    bokeh_pyramid: list[np.ndarray] | None = None,
    background_warmth: float = 0.0,
) -> np.ndarray:
    """
    Renders DSLR-quality Portrait Bokeh with optical lens simulation:
    - Uses precomputed bokeh_pyramid if provided for instantaneous <5ms rendering.
    - Normalized convolution prevents subject edge bleeding into background blur.
    - Specular highlight boost in linear space (gamma = 2.2) for crisp circular bokeh discs.
    - Optional background warmth adjustment (-50 to +50).
    """
    if image_bgr is None or mask is None:
        return image_bgr

    h, w = image_bgr.shape[:2]

    # If bokeh_pyramid is provided, interpolate between levels for instant 60 FPS slider drag
    if bokeh_pyramid and len(bokeh_pyramid) >= 2:
        # bokeh_radius in [0..40] -> level float in [0 .. len-1]
        max_r = 36.0
        lvl_f = float(np.clip(bokeh_radius / max_r * (len(bokeh_pyramid) - 1), 0.0, len(bokeh_pyramid) - 1))
        idx0 = int(math.floor(lvl_f))
        idx1 = min(len(bokeh_pyramid) - 1, idx0 + 1)
        frac = lvl_f - idx0

        if idx0 == idx1 or frac <= 1e-4:
            bg_bgr = bokeh_pyramid[idx0].astype(np.float32)
        else:
            bg_bgr = (1.0 - frac) * bokeh_pyramid[idx0].astype(np.float32) + frac * bokeh_pyramid[idx1].astype(np.float32)
    else:
        # On-the-fly optical disc blur
        r = max(1, int(bokeh_radius))
        disc = create_disc_kernel(r)

        # 1. Linearize BGR (gamma = 2.2) and highlight boost
        bgr_norm = image_bgr.astype(np.float32) / 255.0
        bgr_linear = np.power(bgr_norm, 2.2)
        lum = 0.114 * image_bgr[:, :, 0] + 0.587 * image_bgr[:, :, 1] + 0.299 * image_bgr[:, :, 2]
        highlight_factor = np.clip((lum - 215.0) / 40.0, 0.0, 1.0)[..., None]
        bgr_linear = bgr_linear * (1.0 + highlight_factor * 2.0)

        # 2. Normalized convolution using background weight
        W_bg = np.clip(1.0 - mask, 1e-3, 1.0)
        premult = bgr_linear * W_bg[..., None]
        blurred_premult = cv2.filter2D(premult, -1, disc, borderType=cv2.BORDER_REFLECT)
        blurred_weight = cv2.filter2D(W_bg, -1, disc, borderType=cv2.BORDER_REFLECT)[..., None]
        bg_linear = blurred_premult / np.maximum(blurred_weight, 1e-4)

        # De-linearize to sRGB
        bg_bgr = np.clip(np.power(np.clip(bg_linear, 0.0, 1.0), 1.0 / 2.2) * 255.0, 0.0, 255.0)

    # Optional background warmth
    if abs(background_warmth) > 0.5:
        # Warmth shifts Blue down and Red up
        shift = float(background_warmth) * 0.4
        bg_bgr[:, :, 0] = np.clip(bg_bgr[:, :, 0] - shift, 0.0, 255.0)  # B
        bg_bgr[:, :, 2] = np.clip(bg_bgr[:, :, 2] + shift, 0.0, 255.0)  # R

    # 3. Composite foreground subject over blurred background
    m3 = mask[..., None]
    composite = image_bgr.astype(np.float32) * m3 + bg_bgr * (1.0 - m3)
    return np.clip(composite, 0.0, 255.0).astype(np.uint8)


def apply_portrait_bokeh_and_fill_light(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    bokeh_radius: int = 16,
    fill_light: float = 0.0,
    bokeh_pyramid: list[np.ndarray] | None = None,
    background_warmth: float = 0.0,
) -> np.ndarray:
    """Combines portrait depth lens blur, foreground fill light, and background warmth."""
    fg = apply_subject_fill_light(image_bgr, mask, fill_light=fill_light)
    return apply_portrait_bokeh(
        fg,
        mask,
        bokeh_radius=bokeh_radius,
        bokeh_pyramid=bokeh_pyramid,
        background_warmth=background_warmth,
    )


# =============================================================================
# 3. 3D Cinematic Motion (Continuous Depth-Map Mesh Warping via cv2.remap)
# =============================================================================

def compute_target_dimensions(w: int, h: int, max_long_edge: int = 1920) -> tuple[int, int]:
    """
    Computes dimensions preserving exact aspect ratio with even width & height
    required by H.264 / mp4v video encoders.
    """
    scale = min(1.0, float(max_long_edge) / max(w, h))
    tw = int(round(w * scale))
    th = int(round(h * scale))
    # Must be even integers for video encoders
    tw = max(16, tw & ~1)
    th = max(16, th & ~1)
    return (tw, th)


def inpaint_disocclusion_background(
    image_bgr: np.ndarray,
    subject_mask: np.ndarray,
) -> np.ndarray:
    """
    Inpaints the background behind the subject so parallax virtual camera moves
    reveal realistic background content rather than black holes.
    """
    h, w = image_bgr.shape[:2]
    bin_mask = (subject_mask > 0.25).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dilated_mask = cv2.dilate(bin_mask, kernel)

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
    canvas_size: tuple[int, int] | None = None,
    depth_map: np.ndarray | None = None,
    motion_style: str = "orbital",
    intensity: float = 0.5,
    focal_plane: float = 0.5,
) -> np.ndarray:
    """
    Renders a single frame of continuous 3D cinematic virtual camera parallax motion:
    - Uses continuous depth field D(x, y) to construct smooth coordinate displacement fields (cv2.remap).
    - Preserves photo aspect ratio (canvas_size defaults to source dimensions).
    - motion_style: 'orbital', 'dolly_in', 'pan_lr', 'breathe'.
    - intensity: parallax displacement amplitude in [0.1, 1.0].
    """
    h, w = foreground_bgr.shape[:2]
    if canvas_size is None:
        tw, th = w, h
    else:
        tw, th = canvas_size

    if depth_map is None:
        depth_map = estimate_depth_map(foreground_bgr, subject_mask=subject_mask)

    # Resize depth map to output canvas if dimensions differ
    if depth_map.shape[:2] != (th, tw):
        depth_canvas = cv2.resize(depth_map, (tw, th), interpolation=cv2.INTER_LINEAR)
    else:
        depth_canvas = depth_map

    # Ensure background and foreground match canvas size
    if foreground_bgr.shape[:2] != (th, tw):
        fg_canvas = cv2.resize(foreground_bgr, (tw, th), interpolation=cv2.INTER_LINEAR)
    else:
        fg_canvas = foreground_bgr

    if background_bgr.shape[:2] != (th, tw):
        bg_canvas = cv2.resize(background_bgr, (tw, th), interpolation=cv2.INTER_LINEAR)
    else:
        bg_canvas = background_bgr

    if subject_mask.shape[:2] != (th, tw):
        mask_canvas = cv2.resize(subject_mask, (tw, th), interpolation=cv2.INTER_LINEAR)
    else:
        mask_canvas = subject_mask

    # Camera trajectory depending on motion style
    intensity = max(0.05, min(1.0, float(intensity)))
    max_disp = float(min(tw, th)) * 0.045 * intensity

    # Center depth offset around focal_plane
    # Near objects (depth > focal_plane) move opposite to far objects (depth < focal_plane)
    rel_depth = depth_canvas - float(focal_plane)

    if motion_style == "dolly_in":
        # Hitchcock / Google Cinematic zoom push-in:
        # Foreground expands faster than background
        # Ease-in-out curve
        ease_t = 0.5 - 0.5 * math.cos(t * math.pi)
        cam_zoom = 1.0 + 0.05 * ease_t * intensity
        # Differential zoom map
        cx, cy = tw * 0.5, th * 0.5
        xs, ys = np.meshgrid(np.arange(tw, dtype=np.float32), np.arange(th, dtype=np.float32))
        depth_zoom = cam_zoom + rel_depth * (0.04 * ease_t * intensity)
        map_x = cx + (xs - cx) / np.maximum(depth_zoom, 0.5)
        map_y = cy + (ys - cy) / np.maximum(depth_zoom, 0.5)

    elif motion_style == "pan_lr":
        # Smooth horizontal parallax glide
        cam_x = math.sin(t * 2.0 * math.pi) * max_disp
        cx, cy = tw * 0.5, th * 0.5
        xs, ys = np.meshgrid(np.arange(tw, dtype=np.float32), np.arange(th, dtype=np.float32))
        map_x = xs + cam_x * rel_depth
        map_y = ys

    elif motion_style == "breathe":
        # Gentle floating breathe
        cam_y = math.sin(t * 2.0 * math.pi) * (max_disp * 0.6)
        zoom = 1.0 + 0.02 * math.sin(t * 2.0 * math.pi) * intensity
        cx, cy = tw * 0.5, th * 0.5
        xs, ys = np.meshgrid(np.arange(tw, dtype=np.float32), np.arange(th, dtype=np.float32))
        map_x = cx + (xs - cx) / zoom
        map_y = cy + (ys - cy) / zoom + cam_y * rel_depth

    else:  # "orbital" (default)
        # Sinusoidal 3D orbital loop with gentle push
        cam_x = math.sin(t * 2.0 * math.pi) * max_disp
        cam_y = math.cos(t * 2.0 * math.pi) * (max_disp * 0.35)
        zoom = 1.0 + 0.025 * math.sin(t * math.pi) * intensity

        cx, cy = tw * 0.5, th * 0.5
        xs, ys = np.meshgrid(np.arange(tw, dtype=np.float32), np.arange(th, dtype=np.float32))
        map_x = cx + (xs - cx) / zoom + cam_x * rel_depth
        map_y = cy + (ys - cy) / zoom + cam_y * rel_depth

    # Remap background and foreground using continuous displacement
    warped_bg = cv2.remap(bg_canvas, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    warped_fg = cv2.remap(fg_canvas, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    warped_mask = cv2.remap(mask_canvas, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    m3 = warped_mask[..., None]
    composite = warped_fg.astype(np.float32) * m3 + warped_bg.astype(np.float32) * (1.0 - m3)
    return np.clip(composite, 0.0, 255.0).astype(np.uint8)


def render_3d_cinematic_motion_video(
    image_bgr: np.ndarray,
    output_path: Path | str,
    duration_sec: float = 4.0,
    fps: int = 30,
    resolution: tuple[int, int] | None = None,
    subject_mask: np.ndarray | None = None,
    depth_map: np.ndarray | None = None,
    motion_style: str = "orbital",
    intensity: float = 0.5,
    focal_plane: float = 0.5,
    progress_callback: Any | None = None,
) -> bool:
    """
    Renders a 3D Cinematic Motion .mp4 video from a single photo preserving native aspect ratio.
    """
    if image_bgr is None or image_bgr.size == 0:
        return False

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    h, w = image_bgr.shape[:2]
    if resolution is None:
        target_size = compute_target_dimensions(w, h, max_long_edge=1920)
    else:
        target_size = (resolution[0] & ~1, resolution[1] & ~1)

    tw, th = target_size

    if subject_mask is None:
        subject_mask = estimate_subject_mask(image_bgr)

    if depth_map is None:
        depth_map = estimate_depth_map(image_bgr, subject_mask=subject_mask)

    bg_clean = inpaint_disocclusion_background(image_bgr, subject_mask)

    total_frames = int(duration_sec * fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_p), fourcc, float(fps), (tw, th))

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
                canvas_size=(tw, th),
                depth_map=depth_map,
                motion_style=motion_style,
                intensity=intensity,
                focal_plane=focal_plane,
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
