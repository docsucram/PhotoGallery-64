# PhotoGallery64 Image Editing Engine: Technical Architecture & Specification

## 1. Architectural Overview & Design Philosophy

`PhotoGallery64` implements a hybrid non-destructive parametric editing architecture with deterministic, lossless pixel pipeline execution built on NumPy and OpenCV (`src/imaging/opencv_engine.py`, `src/ui/media_viewer.py`, `src/ui/right_tag_dock.py`).

The design satisfies four core operational principles:
1. **Interactive Fluidity (Zero-Latency Preview):** During interactive slider dragging or mouse movement, operations execute against an adaptive, downscaled display proxy (`display_proxy_rgb`) at $>60\text{ FPS}$.
2. **Lossless Master Commit:** When adjustments are released or committed, the identical parametric recipe executes against the full-resolution master array (`master_np_rgb`) in 32-bit floating point precision.
3. **Pristine Non-Destructive Lifecycle:** Destructive file modifications on disk back up the original, untouched asset into an adjacent `.OriginalFiles/` directory, preserving all initial EXIF, XMP, MakerNotes, and GPS blocks.
4. **Metadata & Face Region Coherence:** Any spatial transformation (rotation, straightening, cropping) automatically remaps normalized face tag bounding boxes (`[0.0, 1.0]`) and flags metadata as dirty for background synchronization.

```
                           +----------------------------+
                           | Master File on Disk (JPEG) |
                           +--------------+-------------+
                                          |
                      +-------------------+-------------------+
                      |                                       |
                      v                                       v
         +-------------------------+             +-------------------------+
         | Master Array (RGB uint8)|             |  Display Proxy (1280px) |
         +------------+------------+             +------------+------------+
                      |                                       |
                      |  Full-Res Commit                      |  Live Slider Scrubbing
                      |  (res_scale = max/1280)               |  (res_scale = 1.0)
                      v                                       v
         +-------------------------------------------------------------+
         | Strict Photographic Execution Pipeline:                     |
         | 1. Denoise (Dual-domain bilateral, scaled spatial radius)   |
         | 2. Dehaze (Dark channel prior + bilateral map refinement)   |
         | 3. Levels & Filmic Exposure (Stops + shadow toe lift)       |
         | 4. Unified CIE LAB Pass:                                    |
         |    - Base/Detail separation (L_base, L_detail)              |
         |    - White Balance (B* Temp, A* Tint)                       |
         |    - Shadows/Highlights + Ambience micro-contrast           |
         |    - Midtone Clarity on L_detail                            |
         |    - Filmic S-curve Contrast on L                           |
         |    - U-Point Selective Control Points ([0..1] normalized)   |
         |    - Skin-Safe Vibrance (LAB warm skin locus, zero HSV)     |
         |    - Halo-free Luminance Unsharp Mask (dead-zone threshold) |
         | 5. Creative Preset Blend (0..100% intensity)                |
         | 6. Aspect-Aware Elliptical Vignette                         |
         | 7. Straighten & Crop (Bicubic rotation + inscribed crop)    |
         +------------------------------+------------------------------+
                                        |
                      +-----------------+-----------------+
                      |                                   |
                      v                                   v
         +-------------------------+         +-------------------------+
         |  Lossless Master Cache  |         |  Real-Time Viewport Pix |
         +-------------------------+         +-------------------------+
```

---

## 2. Parametric Recipe Stack (`PhotoAdjustmentRecipe`)

All image modifications (except direct pixel repairs like Red Eye and Retouch) are parameterized as a serializable, non-destructive dataclass:

```python
@dataclass
class PhotoAdjustmentRecipe:
    # Tone & Exposure
    brightness: float = 0.0          # -100.0 .. +100.0 (Filmic shoulder roll-off)
    contrast: float = 0.0            # -100.0 .. +100.0 (Filmic S-curve on L*)
    shadows: float = 0.0             # -100.0 .. +100.0 (Edge-aware shadow lift/crush)
    highlights: float = 0.0          # -100.0 .. +100.0 (Highlight recovery)
    clarity: float = 0.0             # -100.0 .. +100.0 (Midtone local contrast)
    black_point: float = 0.0         # 0.0 .. 255.0 (Histogram shadow level clip)
    white_point: float = 255.0       # 0.0 .. 255.0 (Histogram highlight level clip)

    # Color & Atmospheric
    temperature: float = 0.0         # -100.0 .. +100.0 (Cool Blue <-> Warm Amber)
    tint: float = 0.0                # -100.0 .. +100.0 (Green <-> Magenta)
    saturation: float = 0.0          # -100.0 .. +100.0 (Skin-safe vibrance)
    dehaze: float = 0.0              # -100.0 .. +100.0 (Dark channel prior recovery)
    vignette: float = 0.0            # -100.0 .. +100.0 (Elliptical corner shading)

    # Detail & Geometry
    straighten_deg: float = 0.0      # -15.0 .. +15.0 degrees
    sharpen: float = 0.0             # 0.0 .. 100.0 (Luminance unsharp masking)
    denoise: float = 0.0             # 0.0 .. 100.0 (Dual-domain bilateral filter)
    crop_rect: tuple[int, ...] | None = None  # (left, top, right, bottom)

    # Creative Presets & U-Point Selective Adjustments
    effect: str = "none"             # Preset ID
    effect_intensity: float = 100.0  # 0.0 .. 100.0 (% blend with base)
    selective_points: list[dict] = field(default_factory=list)
```

---

## 3. Mathematical & Algorithmic Specifications

### 3.1. Red Eye Reduction
Flash photography in low ambient light causes retro-reflection from the vascularized retina through a dilated pupil, resulting in bright, unnatural crimson circular anomalies.

#### Detection & Isolation Pipeline:
1. **Bounding Box Saliency Filtering:**
   Given a user-selected rectangle, the local pixel patch is converted to floating-point RGB:
   $$\text{gb\_sum} = G + B + 1.0$$
   $$\text{r\_ratio} = \frac{R}{\text{gb\_sum}}, \quad \text{r\_diff} = R - \max(G, B)$$
   A pixel is marked as candidate pupil tissue if:
   $$\text{is\_red} = (R > 70.0) \land (\text{r\_ratio} \ge 1.25) \land (\text{r\_diff} \ge 30.0)$$
   (With dynamic fallback to $R > 60.0$, $\text{r\_ratio} \ge 1.15$, $\text{r\_diff} \ge 20.0$ for darker pupils).

2. **Connected Component Analysis (Spatial Clustering):**
   Applies `cv2.connectedComponentsWithStats` with 8-way connectivity over the candidate binary mask. Evaluates the centroid distance of each connected component to the center of the selection box:
   $$d_i = \sqrt{(c_{y, i} - \text{center}_y)^2 + (c_{x, i} - \text{center}_x)^2}$$
   The component closest to the click center with $\text{area} \ge 2\text{ px}$ is isolated as the true pupil.

3. **Morphological Closing:**
   Applies a $3 \times 3$ elliptical structuring element ($\text{MORPH\_CLOSE}$) to bridge tiny specular corneal glints (the natural reflection of the light source) so they are preserved:
   $$\text{mask}_{closed} = (\text{mask} \oplus B) \ominus B$$

4. **Smooth Alpha Feathering:**
   Applies a normalized Gaussian blur ($k \in \{3, 5\}$, $\sigma = 1.0$) across the component perimeter to generate a continuous blending alpha $\alpha \in [0.0, 1.0]$.

5. **Natural Tone Replacement:**
   Replaces the affected red pupil channel with a neutral, desaturated dark tone derived from the existing green/blue channels:
   $$\text{target\_dark} = \text{clip}(\min(G, B) \times 0.55, 12.0, 42.0)$$
   $$\text{patch}_{final} = \text{patch}_{orig} \times (1.0 - \alpha) + \text{target\_dark} \times \alpha$$
   This prevents the "black hole" artifact common in naive red eye tools and retains authentic corneal highlights.

---

### 3.2. Digital Noise Reduction (Dual-Domain Bilateral Filtering)

High ISO values and small sensor photosites introduce both high-frequency luminance grain (salt-and-pepper) and low-frequency chrominance blotchiness (color mottling).

#### 1. Noise Profile Analysis (MAD Wavelet Estimation)
To measure digital sensor noise without user intervention, the engine analyzes the fine-scale Laplacian gradient:
$$\nabla^2 I = I * \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$
The noise standard deviation $\hat{\sigma}$ is estimated via the Median Absolute Deviation (MAD):
$$\hat{\sigma} = \frac{\text{median}(|\nabla^2 I - \text{median}(\nabla^2 I)|)}{0.6745}$$

#### 2. Filtering Architecture & Resolution Scaling
To ensure identical noise suppression between the $1280\text{px}$ display proxy and full-resolution camera master, spatial diameters and standard deviations scale dynamically:
$$\text{res\_scale} = \frac{\max(W, H)}{1280.0}$$
$$d_{chroma} = \max(3, \text{round}(7 \cdot \text{res\_scale}) \mid 1), \quad \sigma_{space, chroma} = 7.0 \cdot \text{res\_scale}$$
$$d_{luma} = \max(3, \text{round}(5 \cdot \text{res\_scale}) \mid 1), \quad \sigma_{space, luma} = 5.0 \cdot \text{res\_scale}$$
- **Luminance Channel ($L^*$):**
  Uses edge-preserving bilateral smoothing ($\sigma_{color} = 0.45 \cdot \text{denoise}$) to clear salt-and-pepper grain without softening edges.
- **Chrominance Channels ($A^*, B^*$):**
  Applies wide-kernel bilateral smoothing ($\sigma_{color} = 0.85 \cdot \text{denoise}$) to eliminate chromatic blotches.

---

### 3.3. Clarity (Midtone Local Contrast & Structure)

Unlike global contrast (which stretches the entire tonal range from pure black to pure white), **Clarity** enhances contrast exclusively within the midtones, emphasizing texture (skin pores, fabric weaves, foliage, masonry) without blowing highlights or crushing shadow detail.

#### Implementation:
1. **CIE LAB Luminance Decomposition:**
   Converts RGB to CIE $L^*a^*b^*$ and isolates luminance channel $L \in [0.0, 255.0]$.
2. **Frequency Separation (Resolution Scaled):**
   Separates $L$ into low-frequency base illumination ($L_{base}$) and high-frequency textural detail ($L_{detail}$) using an edge-preserving Gaussian kernel:
   $$\sigma_{base} = \text{clip}\left(\frac{\min(H, W)}{90.0}, 3.0 \cdot \text{res\_scale}, 14.0 \cdot \text{res\_scale}\right)$$
   $$L_{base} = G_{\sigma_{base}} * L, \quad L_{detail} = L - L_{base}$$
3. **Midtone Bell-Curve Weighting:**
   Constructs a parabolic weighting mask peaked at normalized midtone value $0.5$ ($L \approx 128$):
   $$w_{mid} = \text{clip}\left(1.0 - 4.0 \cdot \left(\frac{L_{base}}{255.0} - 0.5\right)^2, 0.0, 1.0\right)$$
4. **Detail Modulation:**
   Modulates local detail exclusively where $w_{mid} > 0$:
   $$L_{detail}' = L_{detail} \cdot \left(1.0 + 0.45 \cdot \frac{\text{clarity}}{100.0} \cdot w_{mid}\right)$$
   $$L' = \text{clip}(L_{base} + L_{detail}', 0.0, 255.0)$$

---

### 3.4. Filmic Exposure & Negative Exposure Curve

Naive digital exposure multipliers ($I' = I \times 2^{\Delta EV}$) clip abruptly at $255.0$, creating harsh, posterized digital halos around bright skies and light bulbs.

#### Filmic Roll-Off Formulation:
1. Normalize pixel values to $f \in [0.0, 1.0]$.
2. For positive exposure boost ($b > 0$):
   $$f_{scaled} = f \cdot (1.0 + 1.4 \cdot b)$$
   A soft-knee compression threshold is set at $knee = 0.65$. For values exceeding $knee$:
   $$\Delta_{over} = \max(0.0, f_{scaled} - knee)$$
   $$f_{compressed} = knee + \frac{\Delta_{over}}{1.0 + 1.8 \cdot \Delta_{over}}$$
   This asymptotic curve rolls off specular highlights smoothly towards $1.0$, emulating the shoulder density curve of analog silver-halide photographic film.
3. For negative exposure reduction ($b < 0$):
   Attenuates highlights and midtones proportionally in stops:
   $$f_{scaled} = f \cdot 2^{1.5 \cdot b}$$
   To prevent deep shadows and low tones from crushing to pitch black, a gentle photographic toe lift is blended:
   $$toe = 0.05 \cdot |b| \cdot f \cdot (1.0 - f)$$
   $$f_{final} = \text{clip}(f_{scaled} + toe, 0.0, 1.0)$$
   This guarantees highlight recovery down from $1.0$ while retaining open shadow detail.

---

### 3.5. Shadows & Highlights Recovery (Snapseed Ambience)

Shadow and highlight adjustments operate in perceptual CIE LAB space to decouple luminance correction from color hue.

#### Asymmetric Transfer Functions:
- **Shadow Recovery ($s > 0$):**
  Constructs a power-weighted shadow mask targeting dark areas:
  $$\text{mask}_{shadow} = \text{clip}\left(1.0 - \frac{L_{base} / 255.0}{0.65}, 0.0, 1.0\right)^{1.8}$$
  $$L_{base}' = L_{base} + s \cdot \text{mask}_{shadow} \cdot (145.0 - L_{base}) \cdot 0.72$$
  To prevent lifted shadows from appearing hazy or washed-out, an **Ambience Boost** simultaneously elevates local micro-contrast:
  $$L_{detail}' = L_{detail} \cdot (1.0 + 0.35 \cdot s \cdot \text{mask}_{shadow})$$
- **Highlight Recovery ($h < 0$):**
  Identifies overexposed highlight regions above $L_{base} / 255.0 = 0.42$:
  $$\text{mask}_{highlight} = \text{clip}\left(\frac{L_{base} / 255.0 - 0.42}{0.58}, 0.0, 1.0\right)^{1.5}$$
  $$L_{base}' = L_{base} + h \cdot \text{mask}_{highlight} \cdot (L_{base} - 100.0) \cdot 0.65$$

---

### 3.6. Levels & Histogram Black/White Point Stretching

Allows setting shadow clipping (black point $bp \in [0, 255]$) and highlight clipping (white point $wp \in [0, 255]$):
$$f_{out} = \text{clip}\left(\frac{f_{in} - (bp / 255.0)}{(wp / 255.0) - (bp / 255.0)}, 0.0, 1.0\right)$$
- If $bp > 0$, pixels below $bp$ map to deep black ($0.0$), increasing punch and contrast.
- If $wp < 255$, pixels above $wp$ map to pure white ($1.0$), brightening dull or hazy captures.

---

### 3.7. Color Temperature, Tint, and Skin-Safe Vibrance in CIE LAB

1. **White Balance (Temperature & Tint):**
   Evaluated as orthogonal chromatic shifts directly in CIE $L^*a^*b^*$:
   - **Temperature ($B^*$ Axis):** Shifts between Cool Cyan/Blue (negative) and Warm Amber/Yellow (positive):
     $$B^{*\prime} = \text{clip}(B^* + 26.0 \cdot \frac{T}{100.0}, 0.0, 255.0)$$
   - **Tint ($A^*$ Axis):** Shifts between Green (negative) and Magenta (positive):
     $$A^{*\prime} = \text{clip}(A^* + 20.0 \cdot \frac{\text{tint}}{100.0}, 0.0, 255.0)$$

2. **Skin-Safe Vibrance directly in CIE LAB (No HSV Conversion):**
   Global saturation boosts often turn human faces orange or sunburned. The engine computes chroma $C = \sqrt{(A^* - 128)^2 + (B^* - 128)^2}$ and evaluates distance from the warm human skin locus ($A^* \approx 146, B^* \approx 152$):
   $$\text{skin\_dist} = \sqrt{\left(\frac{A^* - 146.0}{14.0}\right)^2 + \left(\frac{B^* - 152.0}{18.0}\right)^2}$$
   $$\text{skin\_mask} = \text{clip}(1.0 - \text{skin\_dist}, 0.0, 1.0)$$
   An inverse-chroma weighting factor prioritizes muted tones while dampening skin adjustments:
   $$\text{vibrance\_weight} = \left(1.0 - \left(\frac{C}{128.0}\right)^{0.8}\right) \cdot (1.0 - 0.6 \cdot \text{skin\_mask})$$
   $$\text{scale} = 1.0 + \frac{sat}{100.0} \cdot (0.8 + 0.6 \cdot \text{vibrance\_weight})$$
   $$A^{*\prime} = \text{clip}(128.0 + (A^* - 128.0) \cdot \text{scale}, 0.0, 255.0)$$
   $$B^{*\prime} = \text{clip}(128.0 + (B^* - 128.0) \cdot \text{scale}, 0.0, 255.0)$$

---

### 3.8. Atmospheric Dehaze

Haze, fog, and smoke scatter ambient light according to the Koschmieder atmospheric attenuation model:
$$I(x) = J(x) t(x) + A (1 - t(x))$$
Where $I$ is observed image, $J$ is scene radiance (cleared image), $t$ is transmission medium, and $A$ is atmospheric light.

#### Recovery & Halo Elimination:
1. Compute Dark Channel Prior:
   $$I_{dark}(x) = \min_{c \in \{R, G, B\}} I^c(x)$$
2. Smooth $I_{dark}$ with a resolution-scaled box filter ($k = \max(3, \text{round}(15 \cdot \text{res\_scale}))$, made odd) to eliminate localized dark objects.
3. Estimate atmospheric light $A$ from the 99th percentile luminance ($A \ge 100.0$).
4. Estimate raw transmission map:
   $$t_{raw}(x) = \text{clip}\left(1.0 - 0.75 \cdot \frac{\text{dehaze}}{100.0} \cdot \frac{I_{dark\_smooth}(x)}{A}, 0.25, 1.0\right)$$
5. **Bilateral Transmission Map Refinement:**
   Refines $t_{raw}(x)$ via a fast bilateral filter ($d = \max(3, \text{round}(7 \cdot \text{res\_scale}) \mid 1), \sigma_c = 0.15, \sigma_s = 7.0 \cdot \text{res\_scale}$) to eliminate $15 \times 15$ boundary halos around high-contrast horizons and tree branches.
6. Recover clear radiance:
   $$J(x) = \frac{I(x) - A}{t(x)} + A$$

---

### 3.9. Halo-Free Luminance Sharpening

To prevent dark/light fringe halos around high-contrast edges, sharpening is evaluated strictly on the luminance plane $L$ using a resolution-scaled Gaussian Unsharp Mask with dead-zone soft thresholding:
$$\sigma_{sharp} = 1.2 \cdot \text{res\_scale}$$
$$L_{blur} = G_{\sigma_{sharp}} * L, \quad d = L - L_{blur}$$
$$d_{soft} = \text{sign}(d) \cdot \max(0.0, |d| - 3.0)$$
$$L_{sharp} = \text{clip}\left(L + 1.5 \cdot \frac{\text{sharpen}}{100.0} \cdot d_{soft}, 0.0, 255.0\right)$$
The $\pm 3.0$ dead-zone prevents sharpening 1px camera sensor noise while delivering crisp, halo-free edge definition.

---

### 3.10. Analytical Inscribed-Rectangle Auto-Straightening

When rotating an image by angle $\theta \in [-15^\circ, +15^\circ]$ to correct a tilted horizon, black boundary triangles appear at the four corners. Rather than forcing the user to manually crop the image, the engine analytically computes the **maximum inscribed rectangle** having the original aspect ratio $R = W_0 / H_0$:

```
+---------------------------------------+
|  /\                               /\  |
| /  \                             /  \ |
|/    +---------------------------+    \|
|     |                           |     |
|     |  Inscribed Crop Region    |     |
|     |                           |     |
|\    +---------------------------+    /|
| \  /                             \  / |
|  \/                               \/  |
+---------------------------------------+
```

For rotation angle $\alpha = |\text{radians}(\theta)|$:
$$W_{crop} = \frac{W_0 \cos\alpha - H_0 \sin\alpha}{\cos(2\alpha)}$$
$$H_{crop} = \frac{H_0 \cos\alpha - W_0 \sin\alpha}{\cos(2\alpha)}$$
The image is rotated via bilinear interpolation around its center point and cropped to $(W_{crop}, H_{crop})$, guaranteeing that zero black border pixels or transparent corners ever appear.

---

### 3.11. Selective Control Points (U-Point Bilateral Engine)

Allows localized adjustments to specific objects or faces without manual masks:
1. User clicks point $P_0 = (x_0, y_0)$ with radius $R$. Supports normalized $[0.0, 1.0]$ coordinates across both display preview proxy and full-resolution master:
   $$x_{px} = x_0 \cdot W, \quad y_{px} = y_0 \cdot H, \quad R_{px} = R \cdot \max(W, H)$$
2. Target color $C_0 = (L_0, A_0, B_0)$ is sampled at $P_0$.
3. For every pixel $P = (x, y)$ with color $C = (L, A, B)$:
   - **Spatial Weight:**
     $$w_s = \text{clip}\left(1.0 - \frac{\sqrt{(x - x_{px})^2 + (y - y_{px})^2}}{R_{px}}, 0.0, 1.0\right)^2$$
   - **Photometric Color Weight:**
     $$\Delta E = \sqrt{(L - L_0)^2 + (A - A_0)^2 + (B - B_0)^2}$$
     $$w_c = \exp\left(-\left(\frac{\Delta E}{28.0}\right)^2\right)$$
   - **Combined Influence:**
     $$W(x, y) = w_s \cdot w_c$$
4. Adjustments (brightness, contrast, saturation) are modulated by $W(x, y)$, restricting modifications strictly to the selected subject (e.g., brightening a single face or saturating a blue shirt while leaving the background untouched).

---

## 4. Photographic Preset Effects Engine

The engine provides 7 filmic emulsion profiles calibrated against real photographic print stocks, plus clean baseline:

| Effect ID | Display Name | Emulation Target | Technical Formulation |
| :--- | :--- | :--- | :--- |
| `none` | **None** | Unadjusted Baseline | Straight pass-through of base adjustments. |
| `classic_chrome` | **Classic Chrome** | Fujifilm Classic Chrome / Kodachrome | Saturation reduced by 18%; midtone contrast boosted (+15%); warm amber tint on highlights; subtle teal shift in shadows. |
| `warm_portrait` | **Warm Portrait** | Kodak Portra 160 / 400 | Golden highlight glow ($B^* +8$); shadow lift ($L^* +6$); skin tone vibrance protection; softened micro-contrast. |
| `vivid_landscape` | **Vivid Landscape** | Fujifilm Velvia 50 | High chromatic saturation (+38%); boosted sky cyan/blues ($B^* -10$); deep, rich blacks ($bp = 10$); high clarity (+25%). |
| `matte_vintage` | **Matte Vintage** | 1970s Matte Paper Print | Raised black point ($L_{black} = 22.0$); highlight compression ($L_{white} = 240.0$); warm sepia-yellow cross-fade; reduced micro-contrast. |
| `fine_art_bw` | **Fine Art B&W** | Fujifilm Neopan Acros 100 | Panchromatic spectral weighting: $L = 0.299R + 0.587G + 0.114B$; steep filmic S-curve; deep shadow blacks; micro-contrast enhancement (+30%). |
| `cool_cinema` | **Cool Cinema** | Modern 35mm Hollywood Teal & Orange | Shadow split-toning: teal cyan ($A^* -10, B^* -14$); highlight split-toning: warm amber ($A^* +6, B^* +18$); filmic contrast. |
| `sepia` | **Sepia** | 19th-Century Silver Bromide Print | Complete panchromatic desaturation followed by warm duotone matrix multiply: $R \times 1.15, G \times 0.98, B \times 0.80$ with lifted shadow blacks. |

All effects support smooth intensity blending from $0\%$ to $100\%$:
$$I_{blended} = I_{base} \times (1.0 - \frac{\text{intensity}}{100.0}) + I_{effect} \times \frac{\text{intensity}}{100.0}$$

---

## 5. Non-Destructive File Lifecycle & Safety Guardrails

### 5.1. The `.OriginalFiles/` Backup Protocol
Before any destructive file overwrite occurs on disk (e.g. from the viewer or batch ribbon operations):
1. The target file's parent folder is inspected for a `.OriginalFiles/` subfolder.
2. If `.OriginalFiles/<filename>` does not already exist, the original source file is losslessly copied into `.OriginalFiles/`.
3. If `.OriginalFiles/<filename>` already exists, it is **never overwritten**—preserving the true camera original permanently.
4. When the user clicks **Revert to original**, the file from `.OriginalFiles/` is copied back over the working file, and any temporary cache is invalidated.

### 5.2. Metadata & Face Coordinate Healing
When cropping or rotating an image:
1. Normalized face bounding boxes $[x, y, w, h] \in [0.0, 1.0]$ are re-projected through the inverse transformation matrix to remain anchored to the person's physical features.
2. The image's cached thumbnail in `thumbnails.db` is purged.
3. The catalog database row is flagged with `metadata_dirty = 1` so background scanner workers rewrite updated EXIF/XMP tags losslessly without recompressing image pixels.
