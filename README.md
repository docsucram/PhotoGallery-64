# PhotoGallery 64

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/GUI-PySide6%20%2F%20Qt6-green.svg)](https://wiki.qt.io/Qt_for_Python)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

**PhotoGallery 64** is a high-performance, 64-bit desktop photo management and editing workstation inspired by the classic **Windows Live Photo Gallery (WLPG)** and elevated with modern computer vision, biometric face clustering, interactive world map geotagging, and non-destructive image processing.

---

## Key Features

### 🎨 Authentic WLPG Ribbon Experience
- **Classic Fluent & Aero Styling:** Faithful recreation of the Windows Live Photo Gallery ribbon tabs (*Home*, *Edit*, *Find*, *Create*, *View*) with responsive compaction and Quick Access Toolbar (QAT).
- **Effects Gallery:** Live preview carousel and scrollable dropdown (*Black and white*, *Sepia*, *Warm*, *Cool*, *Chrome*, *Vivid*, *Matte*, and *B&W filter effects*).
- **Browser-Style Navigation History:** Seamless back and forward navigation stack with mouse thumb button (`BackButton`, `ForwardButton`) and keyboard shortcut (`Alt+Left` / `Alt+Right`) support.

### 👤 Biometric Face Recognition & People Directory
- **InsightFace & ArcFace AI Backend:** SCRFD anatomical face detector paired with 512-dimensional ArcFace vector embeddings running on ONNX Runtime (CPU or DirectML GPU acceleration).
- **Immich / Google Photos Inspired People Directory:** Visual directory grid with 130px high-resolution portrait cards, live search filtering, and smart elision.
- **Recurring Unnamed Clusters:** Groups unknown face embeddings (`>= 3` faces) with inline *"Add a name..."* pills and instant merge confirmation dialogs.
- **WLPG Inline Verification Strip:** Single-row horizontal square face thumbnail strip on person timelines with 1-click `[✓ Confirm all]` and hover controls.

### 🛠 Modern 60 FPS Editing Engine
- **Non-Destructive Pipeline:** All edits are stored in recipes and applied non-destructively; original master files are safely backed up in `.OriginalFiles`.
- **Resolution-Invariant Processing:** Identical visual output between 1280px interactive proxy previews and full-resolution master exports.
- **Fine-Tune Adjustments:**
  - Interactive RGB/Luminance histogram with draggable black and white point clipping markers.
  - Soft-threshold Gaussian unsharp masking.
  - Spatial bilateral filter denoising & dark-channel prior transmission dehazing.
  - Clarity, vibrancy, color temperature, and exposure adjustments.
- **Retouch & Red-Eye Tools:** Multi-radius Poisson healing brush and pupil-masked red-eye removal.

### 🗺 Interactive World Map Geotagging
- Embedded OpenStreetMap Leaflet integration with hardware-accelerated interactive maps.
- Click-to-pin coordinate tagging, reverse-geocoded location reverse search, and bounding-box spatial clustering.

### ⚡ Dual-Database Architecture
- **Catalog Database (`catalog.db`):** High-speed SQLite storing media records, EXIF metadata, hierarchical tags, and ArcFace embeddings.
- **Thumbnails Database (`thumbnails.db`):** Dedicated blob storage caching multi-scale WebP thumbnail pyramids and face crops for instantaneous scrolling.

---

## Installation

### Prerequisites
- Python 3.11, 3.12, or 3.13 (64-bit)
- Git

### Setup
```bash
# 1. Clone repository
git clone https://github.com/docsucram/PhotoGallery-64.git
cd PhotoGallery-64

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows (pwsh / cmd)

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the Application

```bash
python main.py
```

---

## Testing

PhotoGallery 64 includes a test suite covering database operations, face recognition pipelines, imaging algorithms, and UI controls:

```bash
python -m pytest tests -v
```

---

## Architecture & Documentation

- [`ARCHITECTURE.md`](ARCHITECTURE.md): Deep-dive into database schema, thread pools, and worker lifecycle.
- [`editing_engine.md`](editing_engine.md): Technical documentation of image processing math, denoise, dehaze, and tone recipes.
- [`SPEC.md`](SPEC.md): Functional and technical specifications.

---

## License

This project is licensed under the MIT License.
