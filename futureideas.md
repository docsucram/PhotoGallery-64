\# PhotoGallery64 — Post-WLPG Innovations \& Modern Editor Roadmap (`FUTURE\_IDEAS.md`)



This document tracks Phase 2 enhancements that bring 15 years of photography, computer vision, and UX innovations (2012–2026) into `PhotoGallery64` while keeping the fast, familiar Windows Live Photo Gallery layout and 100% local/offline execution.



\---



\## Part 1: Modernising the Single-Image Editor ("Snapseed + Lightroom Grade" Inline Editing)



\### 1.1 Perceptual Tone \& Color Engine (`src/imaging/opencv\_engine.py`)

Replace basic 8-bit linear sliders with a 32-bit float perceptual pipeline operating in CIE LAB / Oklab and Linear sRGB:

\- \*\*Filmic Exposure \& Highlight Roll-Off:\*\* Use a sigmoid/filmic toe-and-shoulder curve so boosting exposure never clips skin tones or skies to harsh digital white (`255, 255, 255`).

\- \*\*Edge-Aware Shadows \& Highlights (Clarity / Ambience):\*\*

&#x20; - Decompose luminance $L$ into a smooth base layer $L\_{\\text{base}}$ (via fast Guided Filter / Bilateral Grid) and a detail layer $L\_{\\text{detail}} = L - L\_{\\text{base}}$.

&#x20; - Lift shadows and compress highlights on $L\_{\\text{base}}$, then re-add $L\_{\\text{detail}}$ so lifted shadows retain full micro-contrast ("Snapseed Ambience" slider).

\- \*\*Structure / Micro-Contrast \& Dehaze:\*\*

&#x20; - \*Structure:\* Boost mid-frequency local contrast (`sigma=3..15px`) while masking out skin tones (detected via CIE `a/b` skin locus) so portraits don't look gritty.

&#x20; - \*Dehaze:\* Dark-channel prior atmospheric haze removal slider (`0..100%`) for hazy landscapes and window glare.

\- \*\*Chroma-First Denoise \& Smart Sharpen:\*\*

&#x20; - Heavy bilateral/wavelet smoothing on CIE `a` and `b` color channels (eliminates ugly red/blue high-ISO color speckles immediately) paired with gentle edge-masked luminance denoising and halo-free unsharp masking.

\- \*\*Vignette \& Grain:\*\*

&#x20; - Smooth aspect-aware elliptical vignette slider (`-100` dark corners to `+100` white vintage border) with adjustable feathering, plus realistic organic film grain.



\### 1.2 Snapseed-Style "Control Points" (U-Point Selective Editing)

Allow local adjustments without manual lasso or brush masking:

\- \*\*UX:\*\* User clicks \*\*"Selective (U-Point)"\*\* in the Edit Ribbon and clicks any point `(x\_0, y\_0)` on the photo (e.g., a shadowed face, a dull sky, or a red jumper).

\- \*\*Interactive Pin:\*\* Displays a draggable pin with an outer radius ring $R$ and on-canvas vertical gesture sliders for \*\*Brightness (`B`), Contrast (`C`), Saturation (`S`), and Warmth (`W`)\*\*.

\- \*\*Edge-Aware Math:\*\*

&#x20; - Sample reference color $(L\_0, a\_0, b\_0)$ at `(x\_0, y\_0)`.

&#x20; - Compute a real-time weight map $W(x, y)$ combining spatial Gaussian falloff $d\_{\\text{spatial}}$ and color/luminance similarity $d\_{\\text{LAB}}$:

&#x20;   $$W(x, y) = \\exp\\left(-\\frac{(x - x\_0)^2 + (y - y\_0)^2}{2 R^2}\\right) \\cdot \\exp\\left(-\\frac{(L - L\_0)^2 + (a - a\_0)^2 + (b - b\_0)^2}{2 \\sigma\_c^2}\\right)$$

&#x20; - Apply the user's `B/C/S/W` adjustments multiplied by $W(x, y)$. Edges (like a roofline against the sky) are respected automatically!

\- \*\*Radial \& Linear Graduated Filters:\*\*

&#x20; - Draggable horizon gradient (for darkening skies) and elliptical spotlight filter.



\### 1.3 Modern Film Emulation \& 3D LUT Presets (With Intensity Slider)

Replace WLPG's dated 2012 effects strip with modern photographic color grades, each with a `0–100%` opacity blend slider:

\- \*\*Film Stocks:\*\* \*Fuji Classic Chrome\*, \*Fuji Acros (High-Contrast B\&W)\*, \*Kodak Portra 400 (Warm Skin)\*, \*Kodachrome 64\*, \*Ilford HP5\*.

\- \*\*Modern Grades:\*\* \*Cinematic Teal \& Orange\*, \*Matte Vintage Fade\*, \*Crisp Daylight\*, \*Golden Hour\*, \*High-Key Portrait\*.

\- \*\*Implementation:\*\* Fast trilinear interpolation over compact `17x17x17` or `33x33x33` `.cube` LUT arrays in NumPy (`<8 ms` on a 2K preview).



\### 1.4 Robust Auto-Straighten \& 4-Point Perspective / Keystone Correction

\- \*\*Auto-Horizon Straighten:\*\* Detect line segments via OpenCV `createLineSegmentDetector()` (LSD); weight long segments within `±15°` of horizontal and vertical (`90° ± 15°`) using RANSAC median voting to find the true horizon angle, then analytically compute the maximum inscribed axis-aligned crop rectangle so zero black corners ever appear.

\- \*\*Vertical / Horizontal Perspective Sliders \& 4-Corner Pin:\*\* Fix converging vertical lines on buildings or flatten photos of framed pictures/documents/whiteboards taken at an angle (`cv2.getPerspectiveTransform`).



\### 1.5 Quick Markup, Text, Callouts \& Privacy Redaction Tool

Built directly into `SingleMediaViewer` (`QGraphicsScene` vector layer flattened on save, backed up via `.OriginalFiles`):

\- \*\*Text Tool:\*\* Draggable, rotatable text labels with modern font picker, outline/drop-shadow, or semi-transparent pill background.

\- \*\*Shapes \& Arrows:\*\* Crisp vector arrows, rectangles, circles, and magnifier loupe callouts (great for annotating screenshots or instructions).

\- \*\*Privacy Blur / Pixelate Brush:\*\* Drag a box or brush over a license plate, address, or bystander face to apply instant mosaic pixelation or heavy Gaussian blur before sharing.



\### 1.6 Local AI Editing Power Tools (`onnxruntime-directml`, 100% Offline)

\- \*\*Magic Eraser (LaMa Inpainting ONNX, \~50 MB):\*\*

&#x20; - Replace basic `cv2.INPAINT\_TELEA` on larger brush strokes with a quantized LaMa ONNX model. Erases power lines, photobombers, bins, and wall blemishes with texture-aware synthesis in `<250 ms` on GPU.

\- \*\*Depth-Based Portrait Blur \& Relighting (`Depth-Anything-V2-Small` ONNX, \~25 MB):\*\*

&#x20; - Generates a per-pixel depth map in `<150 ms`. Enables a \*\*"Background Blur (Bokeh)"\*\* slider and \*\*"Foreground Fill Light"\*\* slider on any photo (even 1950s scans or non-portrait camera shots).

\- \*\*1-Click Background Cutout / Sticker (`RMBG-1.4` / `ModNet` ONNX):\*\*

&#x20; - Isolate the main subject to a transparent PNG or copy straight to the Windows clipboard for pasting into documents/presentations.

\- \*\*Old Scan Restoration \& 2x Super-Resolution (`Real-ESRGAN-x2` / `GFPGAN` ONNX):\*\*

&#x20; - Dedicated \*\*"Restore Archival Scan"\*\* button for low-res early-2000s digital photos or 1950s–1980s family prints (removes JPEG block artifacts, dust scratches, and gently restores facial clarity).



\---



\## Part 2: 15 Years of Library \& DAM Innovations (400,000-Photo Scale)



\### 2.1 Semantic Natural-Language Search (Local MobileCLIP / SigLIP2 ONNX)

\- \*\*Capability:\*\* Search by visual concept (`"red car"`, `"snowy garden"`, `"playing board games"`, `"church interior"`, `"plate of food"`) with zero manual tags required.

\- \*\*Architecture:\*\*

&#x20; - Background worker feeds existing `256x256` WebP thumbnails from `thumbnails.db` into `MobileCLIP-S1` image encoder (`onnxruntime-directml` at \~150 images/sec).

&#x20; - Stores L2-normalized `512 x float16` vectors (`1 KB` per photo = `\~400 MB` total for 400,000 photos) in a memory-mapped binary file (`clip\_embeddings.bin`) + SQLite index mapping.

&#x20; - Text queries run through the MobileCLIP text encoder in `<10 ms` and rank all 400,000 photos via a single matrix dot product.

\- \*\*"Find Visually Similar" Right-Click Action:\*\* Right-click any photo $\\to$ \*"Find similar photos"\* (computes cosine similarity against that photo's CLIP embedding in `<5 ms`).



\### 2.2 Perceptual Duplicate Finder \& "Best of Burst" Culling

\- \*\*64-Bit Perceptual Hash (`dHash` + `pHash`):\*\*

&#x20; - Stored as an `INTEGER` column (`phash`) in `media` during initial thumbnail generation (zero extra disk reads).

&#x20; - Instant duplicate/near-duplicate detection across different resolutions, WhatsApp re-compressions, or crops.

\- \*\*Smart Burst Stacking \& Auto-Pick Best Shot:\*\*

&#x20; - Groups photos taken within `< 2.0 seconds` of each other with similar `pHash`.

&#x20; - Scores each frame in the burst using \*\*Laplacian sharpness variance\*\* (on the detected face regions first, or whole image) + \*\*Eye-Open aspect ratio\*\* from facial landmarks, automatically badging the sharpest, non-blinking shot with a gold "Best Pick" star.



\### 2.3 Offline OCR Text-in-Photo Search (SQLite FTS5)

\- \*\*Capability:\*\* Instantly find photos by text visible inside the image (street signs, book covers, certificates, receipts, whiteboards, wine labels, screenshots).

\- \*\*Architecture:\*\*

&#x20; - Background worker runs Windows native `Windows.Media.Ocr` (or `RapidOCR` ONNX) on thumbnails/images containing high-frequency text edges.

&#x20; - Inserts recognized text into an SQLite \*\*FTS5 virtual table\*\* (`CREATE VIRTUAL TABLE media\_ocr\_fts USING fts5(media\_id UNINDEXED, ocr\_text)`).

&#x20; - Integrates seamlessly into the main top-right Search box (`"Search tags, captions, filenames, or text in photos..."`).



\### 2.4 Offline Reverse Geocoding (Automatic Country / County / Town Hierarchy)

\- \*\*Capability:\*\* Converts raw GPS `(lat, lon)` coordinates into human-readable hierarchical place names (`United Kingdom -> Surrey -> Claygate` or `Malta -> Valletta`) completely offline, with zero API keys or rate limits.

\- \*\*Architecture:\*\*

&#x20; - Uses an embedded offline GeoNames KD-Tree (`reverse\_geocoder` / SQLite spatial lookup table).

&#x20; - Populates a collapsible \*\*Places\*\* tree in the Left Navigation Dock (`Country -> State/County -> City/Town`) alongside `Date taken` and `Descriptive tags`.



\### 2.5 Smart Albums (Saved Live Boolean Filters) \& "On This Day" Memories

\- \*\*Smart Albums:\*\* Save any combination of Left-Tree filters (e.g., `Person = "Nansumay" AND Rating >= 4 AND Year = 2025`) as a persistent Smart Album in the Left Navigation Dock that updates live as new photos are added.

\- \*\*"On This Day" / Rediscover Bar:\*\* Optional toggleable header strip in Grid View surfacing high-rated photos taken on today's calendar day (`DD MMM`) across previous years (`1 year ago`, `5 years ago`, `10 years ago`, `50+ years ago`).



\### 2.6 Modernised "Create" Tab Workflows

\- \*\*Interactive Collage Studio:\*\* Multi-template grid, Polaroid scatter, and masonry layouts on a live `QGraphicsScene` where photos can be dragged between slots, panned/zoomed within their frames, given custom background textures/margins, and exported at 300 DPI print resolution.

\- \*\*Photo Fuse (Best-Face Group Shot Combiner):\*\* Aligns 2–6 burst photos of a group using `cv2.findHomography` + ECC alignment; lets the user click any person's face to cycle through the aligned faces from the other shots and seamlessly Poisson-blend (`cv2.seamlessClone`) the chosen smile into the master photo.

\- \*\*Hardware-Accelerated Slideshow Composer + MP4 Export:\*\* Smooth 60 FPS Ken Burns pan-and-zoom (centered automatically on detected face coordinates so heads are never cropped off during zoom!), cross-fades, vintage film overlays, background audio track crossfading, and 1-click export to a standalone `1080p`/`4K` `.mp4` video file.

