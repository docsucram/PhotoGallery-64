# PhotoGallery64 — 8-Phase Modular Build Plan (`TASKS.md`)

Each phase is strictly scoped to **1–2 implementation files** and verified by an isolated `pytest` test suite before moving to the next phase.

---

## Phase 1: Dual-Database Engine & Sub-10ms Repository Layer
- **Files Touched:**
  1. `src/db/connection.py`
  2. `src/db/repository.py`
  *(Test suite: `tests/test_phase1_db.py`)*
- **Scope & Deliverables:**
  - Implement `DualDatabaseEngine` managing thread-local SQLite `WAL` connections (`PRAGMA synchronous = NORMAL`) for `catalog.db` and `thumbnails.db`, initializing DDL/indexes/triggers from `db/schema.sql`.
  - Implement `CatalogRepository` and `ThumbnailRepository`:
    - Folder & media bulk upsert with `mtime` diffing.
    - Hierarchical tag creation (`PeopleTags/Family/Craig`) verifying `trg_tags_after_insert` populates `tag_closure`.
    - Boolean tag filtering (`AND`, `OR`, `NOT` via `tag_closure` + `media_tags`), Date Taken tree aggregation (`Year` -> `Month` -> `Day`), People filtering, Rating filtering, and inline text search.
    - Spatial bounding-box queries via `media_rtree` (`trg_media_rtree_insert` / `update` / `delete`).
    - Relative timestamp adjustment (`apply_relative_time_shift`) applying $\Delta t = T_{\text{lead\_new}} - T_{\text{lead\_original}}$ to all selected media IDs and setting `metadata_dirty = 1`.
    - Non-destructive edit backup tracking (`record_pixel_edit_backup`, `clear_pixel_edit_backup`, `prune_expired_backups(retention_days=30)`).
- **Verification Command:**
  ```powershell
  pytest tests/test_phase1_db.py -v
  ```
- **Acceptance Criteria:**
  - All triggers (`tag_closure`, `media_rtree`, `metadata_dirty`) and 30-day backup expiration pruning pass automated assertions.
  - Benchmark test with 100,000+ synthetic rows executes Boolean `tag_closure` (`AND`/`OR`/`NOT`) and `rating`/`date_taken` sorted queries in `< 10 ms`.

---

## Phase 2: High-Speed Media Scanner, WebP Thumbnailer & Background XMP Daemon
- **Files Touched:**
  1. `src/workers/scanner_thumb_worker.py`
  2. `src/workers/xmp_sync_worker.py`
  *(Test suite: `tests/test_phase2_workers.py`)*
- **Scope & Deliverables:**
  - Implement `CancellationFlag` and thread-safe `LruCache` (reused from `legacy_reference/imgBrowserv13.py`).
  - Implement two-stage progressive `FolderSyncWorker(QRunnable)`:
    - Stage 1 (Fast Pass): `os.scandir` recursive walk diffing `mtime` against `catalog.db` at ~50k files/sec.
    - Stage 2 (Enrichment Pass): Batched (5,000 rows/tx) EXIF/XMP extraction (`DateTimeOriginal`, dimensions, orientation, GPS, hierarchical keywords, and static-image-only `MP:RegionInfo` face rectangles).
  - Implement `ThumbnailDecodeWorker(QRunnable)`:
    - Fast image scaling via `QImageReader` (`setAutoTransform(True)`, `setScaledSize(QSize(256, 256))`) encoded to 256x256 WebP blobs (`~8–12 KB`) stored in `thumbnails.db`.
    - Smart video frame extraction via OpenCV (reusing `_is_frame_too_dark` at 15% / 50% / 1000ms and `_get_or_create_filmstrip_overlay` from `legacy_reference/imgBrowserv13.py`).
  - Implement `MetadataSyncDaemon` draining `metadata_dirty = 1` rows in the background (`pyexiv2` primary, persistent `exiftool -stay_open` fallback), writing compliant XMP (`lr:hierarchicalSubject`, `MicrosoftPhoto:LastKeywordXMP`, `dc:subject`, plus `MP:RegionInfo` for static images `is_video = 0` only) and lossless EXIF `Orientation` / `DateTimeOriginal`, then clearing `metadata_dirty = 0`.
- **Verification Command:**
  ```powershell
  pytest tests/test_phase2_workers.py -v
  ```
- **Acceptance Criteria:**
  - Scanning a sample directory populates `catalog.db` and generates valid WebP blobs in `thumbnails.db`.
  - Modifying tags, ratings, or `date_taken` in `catalog.db` (`metadata_dirty = 1`) and running `MetadataSyncDaemon.flush_now()` writes XMP/EXIF metadata to a temporary JPEG copy losslessly and resets `metadata_dirty = 0`.

---

## Phase 3: Virtualized 500k Catalog Model & Polaroid / Mosaic Delegate
- **Files Touched:**
  1. `src/ui/grid_model.py`
  2. `src/ui/grid_view.py`
  *(Test suite: `tests/test_phase3_grid.py`)*
- **Scope & Deliverables:**
  - Implement `CatalogListModel(QAbstractListModel)`:
    - Holds only compact `MediaItemRecord` tuples in memory (`~48 MB` for 400k items), never `QListWidgetItem` or `QPixmap`.
    - Supports collapsible section headers grouped by active arrange criteria (e.g. `v 5 stars (1810 items, 1 selected)` or `v October 1957 (340 items)`).
    - Supports both **WLPG Classic Uniform Grid (`"wlpg_grid"`)** and **Justified Tessellated Mosaic (`"mosaic"`)** geometry calculation (`compute_mosaic_geometry` packing portrait and landscape aspect ratios into gapless equal-height rows).
  - Implement `MediaCardDelegate(QStyledItemDelegate)`:
    - Renders WLPG rounded polaroid card with subtle border and drop shadow (in `"wlpg_grid"` mode) or gapless tessellated tile (in `"mosaic"` mode).
    - Top-left hover-activated multi-selection checkbox.
    - Top-right hover-activated Flag pin icon.
    - Bottom interactive 5-star rating widget directly on the card + configurable `Show details` text lines.
    - Hit-testing in `editorEvent()` emitting `rating_clicked`, `flag_toggled`, `checkbox_toggled`, and `group_header_clicked`.
  - Implement `ThumbnailGridView(QListView)` configured with `setViewMode(QListView.IconMode)`, `setUniformItemSizes(True)` (in Classic Grid mode), Mosaic layout switcher, and scroll/resize viewport range calculation (`visible_media_ids`).
- **Verification Command:**
  ```powershell
  pytest tests/test_phase3_grid.py -v
  ```
- **Acceptance Criteria:**
  - Loading 400,000 synthetic `MediaItemRecord` items into `CatalogListModel` + `ThumbnailGridView` completes in `< 50 ms` and consumes `< 85 MB` of additional RSS memory.
  - Simulated clicks on card star regions, flag pin, checkbox, and collapsible group headers emit the exact expected signals.

---

## Phase 4: Left Unified Filter Tree & Top WLPG Ribbon / Quick Find Bar
- **Files Touched:**
  1. `src/ui/left_nav_dock.py`
  2. `src/ui/ribbon_toolbar.py`
  *(Test suite: `tests/test_phase4_nav_ribbon.py`)*
- **Scope & Deliverables:**
  - Implement `LeftNavigationDock`:
    - Unified filter tree matching `docs/reference_ui/main_ui.png` and `find_ui.png`:
      - **Folders:** Directory tree with Favourites menu (reusing `FavouritesMenu` logic) and `Include Subfolders` toggle.
      - **Date Taken:** Hierarchical `Year` -> `Month` -> `Day, Weekday` (`1 Jan, Thursday`) with item counts.
      - **Descriptive Tags:** Multi-level nested tree (`PeopleTags/Family/Craig`) with tri-state/modifier checkboxes (Checked = Include `AND`/`OR`, Unchecked = Neutral, Right-click / Alt-check = Exclude `NOT`, plus `Not tagged` node).
      - **People** & **Ratings** (`5 stars` .. `1 star`, `Unrated`) filter nodes.
  - Implement `RibbonToolbar`:
    - Tabbed ribbon (`Home`, `Edit`, `Find`, `Create`, `View`) matching `docs/reference_ui/view_tab.png` and `find_ui.png`.
    - `View` tab: `Arrange list` group (`Date`, `Geotag`, `Person`, `Name`, `Tag`, `Published`, `Folder`, `Media type`, `Rating`), `Reverse sort`, `Show details` toggles (`Rating`, `Caption`, `Flags`, `All details`, `File name`, `File size`, `Date taken`, `Image size`, `Date modified`), `Zoom`, `Slide show`, and `Tag and caption pane` toggle.
    - `Home` / `Find` tab: `Quick find` horizontal face avatar carousel + inline `Search by:` scope header above the grid (`Folder: Pictures (Marcus) | Search all` + search box with clear `x`).
- **Verification Command:**
  ```powershell
  pytest tests/test_phase4_nav_ribbon.py -v
  ```
- **Acceptance Criteria:**
  - Selecting folders, dates, included/excluded (`NOT`) hierarchical tags, or clicking a Quick Find face avatar produces the exact `FilterQuerySpec` expected by `CatalogRepository`.

---

## Phase 5: Right "Tag and Caption Pane" & Relative "Change Time Taken" Dialog
- **Files Touched:**
  1. `src/ui/right_tag_dock.py`
  2. `src/ui/dialogs.py`
  *(Test suite: `tests/test_phase5_dock_dialogs.py`)*
- **Scope & Deliverables:**
  - Implement `TagAndCaptionDock`:
    - Matches `docs/reference_ui/main_ui.png` right pane and dynamically adapts to single or multi-selection.
    - **People tags:** Face crop thumbnail + name + `+ Add people tag` with autocomplete.
    - **Geotag:** Location display + `+ Add geotag`.
    - **Caption:** Multi-line editable text box committing on focus-out / Enter.
    - **Descriptive tags:** Blue clickable/removable tag chips `(x)` + `+ Add descriptive tag` with hierarchical autocomplete.
    - **Information Summary (EXIF):** Filename, Date taken (clickable to open Change Time Taken), Size, clickable 5-star Rating, clickable Flag, Dimensions, Published, Camera, Author, Exposure, Aperture, Focal length, ISO, Latitude, Longitude (using EXIF formatters from `legacy_reference/imgBrowserv13.py`).
  - Implement `ChangeTimeTakenDialog` (and `AppSettingsDialog`) in `src/ui/dialogs.py`:
    - Exact visual layout of `docs/reference_ui/adjust_time.png`: header explanation text, live status label (*"No time adjustment has been made."* -> *"Adjustment: +01:00:00 applied to N files"*), `QTimeEdit` (`HH:mm:ss`), `QDateEdit` (`d MMMM yyyy` with calendar popup), and `Current time:` label of the lead selected item.
- **Verification Command:**
  ```powershell
  pytest tests/test_phase5_dock_dialogs.py -v
  ```
- **Acceptance Criteria:**
  - Multi-selection in `TagAndCaptionDock` displays shared tags/ratings and applies edits across all selected IDs.
  - `ChangeTimeTakenDialog` computes exact $\Delta t = T_{\text{lead\_new}} - T_{\text{lead\_original}}$ (including negative and multi-day shifts) and updates all selected items in `catalog.db` relative to their individual original timestamps.

---

## Phase 6: Single Media Viewer, "Fix" Adjustments & World Map View
- **Files Touched:**
  1. `src/ui/media_viewer.py`
  2. `src/ui/map_view.py`
  *(Test suite: `tests/test_phase6_viewer_map.py`)*
- **Scope & Deliverables:**
  - Implement `SingleMediaViewer` in `src/ui/media_viewer.py`:
    - Reuses and modernizes `ImageGraphicsView`, `CropScene`, `EditCommand(QUndoCommand)`, `ClickableVideoWidget`, and `MouseOverlay` from `legacy_reference/imgBrowserv13.py`.
    - Implements contextual `Edit` / `Fix` operations matching `docs/reference_ui/fullsize and edit_ui.png`: Auto adjust, anchored Crop, Straighten, Noise reduction, Color/Exposure sliders, preset effect tiles (B&W, Sepia, Cool, Warm, Vignette), `Revert to original`, and `Close file`.
  - Implement `WorldMapView` in `src/ui/map_view.py`:
    - `QWebEngineView` rendering a self-contained local Leaflet + Supercluster map page.
    - Queries `CatalogRepository.query_map_clusters(min_lat, max_lat, min_lon, max_lon)` backed by `media_rtree` as the map viewport moves, and filters/opens photos when map markers are clicked.
- **Verification Command:**
  ```powershell
  pytest tests/test_phase6_viewer_map.py -v
  ```
- **Acceptance Criteria:**
  - `ImageGraphicsView` zoom modes (`fit`, `fill`, `100%`, `custom`), `CropScene` clamped rect drawing, and `QUndoStack` `EditCommand` undo/redo pass unit tests.
  - `WorldMapView` bounding-box updates return clustered GPS points from `media_rtree` in `< 5 ms`.

---

## Phase 7: Main Window Coordinator, Status Bar, Aero Theme & Entrypoint
- **Files Touched:**
  1. `src/ui/main_window.py`
  2. `main.py`
  *(Test suite: `tests/test_phase7_main_window.py`)*
- **Scope & Deliverables:**
  - Implement `PhotoGalleryMainWindow(QMainWindow)` in `src/ui/main_window.py`:
    - Coordinates central `QStackedWidget` (0: `ThumbnailGridView`, 1: `SingleMediaViewer`, 2: `WorldMapView`), `RibbonToolbar`, `LeftNavigationDock`, `TagAndCaptionDock`, and WLPG bottom status bar (`"393881 items, 1 selected"`, Rotate Left/Right, Delete, Slideshow/Fullscreen, Grid/Viewer toggle, and Zoom/Thumbnail size slider).
    - Manages `scanner_pool`, `thumbnail_pool`, and `MetadataSyncDaemon` lifecycle and clean shutdown.
    - Implements WLPG keyboard navigation (`Enter` to open viewer / apply crop, `Esc` to exit crop -> fullscreen -> single viewer -> grid, `F11` fullscreen, `Delete` to Recycle Bin via `send2trash`, `Ctrl+Wheel` thumbnail/viewer zoom).
  - Implement `main.py` loading `assets/themes/wlpg_aero.qss` (plus `dark.qss` / `modern.qss` switchers) and launching `PhotoGalleryMainWindow`.
- **Verification Command:**
  ```powershell
  pytest tests/test_phase7_main_window.py -v
  ```
- **Acceptance Criteria:**
  - Full end-to-end test suite (`pytest -v`) passes across all 7 phases: scanning, virtualized grid display, Boolean tag filtering, inline star/flag click, relative `Change Time Taken`, single-image viewer edit/undo, and clean thread-pool teardown.

---

## Phase 8: Automated Face Detection & Batch People Tagging
Modeled on Windows Live Photo Gallery (`docs/reference_ui/batch_people_tag.jpg`) with Picasa-grade confirmation speed.

### Phase 8A: Database Schema, Vector Storage & In-Memory NumPy Vector Matcher
- **Files Touched:**
  1. `src/db/repository.py`
  2. `src/core/face_matcher.py`
  *(Test suite: `tests/test_phase8a_face_matcher.py`)*
- **Scope & Deliverables:**
  - Database schema & repository methods:
    - `people`: `avatar_media_people_id`, `centroid_embedding` (512xfloat32 normalized vector = 2048 bytes), `confirmed_count`.
    - `media_people`: `suggested_person_id`, `similarity_score`, `state` (`0` = unassigned/suggested, `1` = confirmed, `-1` = ignored), `embedding` (512-d ArcFace vector).
    - `face_rejected_suggestions`: `(face_id, person_id, PRIMARY KEY(face_id, person_id))` to permanently prevent "Not this person" rejections from reappearing.
    - `face_thumbnails` (`thumbnails.db`): `(face_id INTEGER PRIMARY KEY, crop_webp BLOB NOT NULL)` storing pre-cropped 96x96 WebP thumbnails (~2–3 KB).
  - Implement `FaceVectorMatcher` (`src/core/face_matcher.py`):
    - Caches unassigned face vectors ($N \times 512$ float32) and people centroids ($M \times 512$ float32) in RAM.
    - **Zero-Lag Confirmation Loop (< 5ms):**
      1. Given confirmed face IDs for person $P$:
         - Updates `media_people` (`person_id = P, state = 1`) and `media` (`metadata_dirty = 1`) on the UI thread.
         - Incrementally updates Person $P$'s centroid: $C_{\text{new}} = \text{normalize}\left(\frac{N \cdot C_{\text{old}} + \sum v_{\text{confirmed}}}{N + k}\right)$ and persists $C_{\text{new}}$ to `people.centroid_embedding`.
         - Runs a single vectorized matrix multiply (`unassigned_embeddings @ C_new`) in `< 3 ms`, filtering out `face_rejected_suggestions`.
         - Immediately returns qualified suggestions to the UI without disk I/O.
- **Verification Command:**
  ```powershell
  pytest tests/test_phase8a_face_matcher.py -v
  ```
- **Acceptance Criteria:**
  - Automated benchmark with 10,000 synthetic 512-d vectors verifies centroid incremental updating and vectorized similarity search completes in `< 5 ms`.
  - Rejection persistence tests confirm rejected face-person pairs are never re-suggested.

---

### Phase 8B: "Batch People Tag" Virtualised UI View & Keyboard Ergonomics [COMPLETED]
- **Files Touched:**
  1. `src/ui/batch_people_view.py`
  2. `src/ui/ribbon_toolbar.py`
  3. `src/ui/main_window.py`
  *(Test suite: `tests/test_phase8b_batch_people_ui.py`)*
- **Scope & Deliverables:**
  - Implement `BatchPeopleView(QWidget)` in `src/ui/batch_people_view.py`:
    - Triggered via `Batch people tag` button in Home/Find ribbon or context menus.
    - Top Ribbon contextual mode: `Actions` group (`Select all`, `Confirm`, `Not this person`, `Tag as...`, `Don't show again`), plus horizontal scrollable `People` avatar picker strip.
    - Dismissible top banner: *"As you act on these suggestions, the faces will be removed from the list below, and the confirmed faces will be tagged in your photos."*
    - Virtualised `QListView` with `FaceTileDelegate`: 96x96 tiles (`face_thumbnails.crop_webp`), rubber-band multi-selection, hover `[✓]` (Confirm) and `[✗]` (Not this person) corner buttons, and similarity percentage badge.
    - Collapsible section headers: `Person Name? (28 suggestions, 25 selected, 20400 confirmed)` with one-click `[Confirm Selected]` button.
    - Right-click context menu: `Confirm`, `Not this person`, `Tag as...`, `Don't show again`, `View full photo`.
    - Ergonomic keyboard shortcuts:
      - `Enter`: Confirm selected.
      - `Delete` / `Backspace`: Not this person.
      - `I`: Don't show again (`state = -1`).
      - `Space`: Non-modal full-photo quick-preview centered on face bounding box for clothing/event disambiguation.
- **Verification Command:**
  ```powershell
  pytest tests/test_phase8b_batch_people_ui.py -v
  ```
- **Acceptance Criteria:**
  - Virtualised list renders 5,000 face suggestions with 60 FPS scrolling and low memory footprint (<20 MB RAM).
  - All keyboard shortcuts and click interactions execute instant visual removals and batch state updates.

---

### Phase 8C: Background ONNX InsightFace Worker & XMP Region Write-Back [COMPLETED]
- **Files Touched:**
  1. `src/core/face_engine.py`
  2. `src/workers/face_worker.py`
  3. `src/workers/xmp_sync_worker.py`
  4. `src/ui/main_window.py`
  *(Test suites: `tests/test_face_worker.py`, `tests/test_face_matcher.py`)*
- **Scope & Deliverables:**
  - Implement `FaceEngine` (`src/core/face_engine.py`):
    - Wraps InsightFace SCRFD face detector + ArcFace 512-d feature extractor via `onnxruntime` with CPU / DirectML execution provider.
    - `extract_crop_and_embed_rect`: extracts 512-d vector and 96x96 WebP crop for existing rectangle.
    - `detect_and_embed`: scans image for all faces, returning rectangles, embeddings, and 96x96 WebP crops.
  - Implement `FaceDetectionWorker` (`src/workers/face_worker.py`):
    - Runs in background `face_pool` (`maxThreadCount = 1`, `QThread.LowPriority`).
    - Stage 1: Prioritized backfill for existing WLPG `MP:RegionInfo` tags (`state = 1`), generating 96x96 crops in `thumbnails.db` and building initial person centroids.
    - Stage 2: Background scanning of untagged photos, inserting unassigned face regions (`state = 0`) into `media_people` and crops into `face_thumbnails`.
    - Stage 3: Auto-triggers `FaceVectorMatcher` to match unassigned faces against centroids.
  - Update `MetadataSyncDaemon` (`src/workers/xmp_sync_worker.py`):
    - Writes confirmed faces (`state = 1`) as compliant `Xmp.MP.RegionInfo` and `Xmp.mwg-rs.Regions` metadata to disk asynchronously without blocking the UI.
- **Verification Command:**
  ```powershell
  pytest tests/test_phase8c_face_worker.py -v
  ```
- **Acceptance Criteria:**
  - Existing photos with `MP:RegionInfo` have embeddings and 96x96 crops backfilled into databases.
  - Confirmed faces in Batch People Tag are written back to image files on disk as standard XMP face regions readable by Windows Live Photo Gallery.
