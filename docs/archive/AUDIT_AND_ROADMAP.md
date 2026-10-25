# Comprehensive Architectural Audit, Code Review & Gap Analysis (`AUDIT_AND_ROADMAP.md`)

**Target Architecture:** Windows Live Photo Gallery (WLPG) Wave 4 (2012) 64-Bit Clone (`Python 3.12` + `PySide6` + `SQLite3 WAL` + `OpenCV/Pillow`)  
**Scale Target:** 400,000+ Photos & Videos, 60 FPS Virtualised Grid, Sub-10ms Faceted Queries  
**Audit Scope:** `SPEC.md`, `ARCHITECTURE.md`, `TASKS.md`, `db/schema.sql`, and all Python modules in `src/` and `main.py`.

---

## 1. Feature Parity Matrix (Built vs. Partial vs. Missing)

The table below compares every requirement in [`SPEC.md`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/SPEC.md), [`ARCHITECTURE.md`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/ARCHITECTURE.md), and [`TASKS.md`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/TASKS.md) against the actual implementation.

| Feature | Status | Files Involved | What Remains |
| :--- | :--- | :--- | :--- |
| **1.1 Virtualised Grid View (`QListView` + `QAbstractListModel` + `QStyledItemDelegate`)** | **Complete** | [`src/ui/grid_model.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/grid_model.py), [`src/ui/grid_view.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/grid_view.py) | Standard `wlpg_grid` mode uses $O(\log G)$ binary search (`_locate_vrow`) for 38px collapsible group headers, inline 5-star ratings, hover flag pins, hover checkboxes, and rubber-band selection. `mosaic` layout hit testing and visible range calculations optimized with binary-searched group bounding checks in `vrow_at_pos` and `visible_vrows_in_rect` (**WP-02 Completed**). |
| **1.2 Collapsible Group Headers & Group Selection** | **Complete** | [`src/ui/grid_model.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/grid_model.py), [`src/ui/grid_view.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/grid_view.py) | Clicking the chevron collapses/expands the group (`toggle_group_collapsed`); clicking the header body selects all items in the group (`select_group`). `GroupSelectedCountRole` now performs an $O(1)$ dictionary lookup from precomputed `_group_selected_counts` on paint frames (**WP-02 Completed**). |
| **1.3 OS Shell Drag-and-Drop (`QMimeData` with `QUrl.fromLocalFile`)** | **Complete** | [`src/ui/grid_view.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/grid_view.py), [`src/ui/left_nav_dock.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/left_nav_dock.py), [`src/ui/main_window.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/main_window.py) | Supports left- and right-click drag out to Windows Explorer (`CopyAction \| MoveAction`) with authentic high-DPI ghost card + count badge (`_start_drag`) with top-left cursor hotspot `(12, 12)`. Dropping onto Left Navigation Folder Tree (`_LeftNavTree.dropEvent`) handles both catalog records and external filesystem files with non-colliding copy/move actions and single-signature signal `files_dropped(int, str, list, bool)`. |
| **1.4 Dynamic Responsive Ribbon (`Home`, `Find`, `View`, `Edit`, `Create`)** | **Complete** | [`src/ui/ribbon_toolbar.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/ribbon_toolbar.py) | Implements QAT (with above/below Ribbon toggle), Minimize the Ribbon via double-click on any tab header or button (`_toggle_ribbon_collapse`), contextual `Edit` tab swaps (`grid` vs. `viewer`), People face carousel, and 3-tier dynamic responsive button compaction (`Wide` $\ge 1180\text{px}$, `Medium` $880\text{px} \le W < 1180\text{px}$, `Narrow` $< 880\text{px}$) in `resizeEvent` (**WP-07 Completed**). |
| **1.5 Right "Tag and Caption" & "Fine Tune" Pane** | **Complete** *(with bulk >32k variable limit)* | [`src/ui/right_tag_dock.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/right_tag_dock.py), [`src/ui/dialogs.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/dialogs.py) | Mutually exclusive `Info` (`Ctrl+I`) and `Fine Tune` (`Ctrl+F`) modes; inline `_ClickToEditWidget` for Filename, Date/Time taken, and Author; people & tag pill lists with `QCompleter` + top-10 recent tags popup; relative $\Delta t$ [`ChangeTimeTakenDialog`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/dialogs.py#L31-L154); live RGB/Luminance [`HistogramWidget`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/right_tag_dock.py#L45-L120). **Remaining:** Multi-select inline date edit (`main_window.py:L1428-L1435`) loops per item in Python instead of calling `apply_relative_time_shift`; Geotag search box only sets text string `location_name` without geocoding `lat`/`lon`. |
| **1.6 Left Navigation Tree (Folders, Date Taken, Tags, People, Ratings)** | **Complete** | [`src/ui/left_nav_dock.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/left_nav_dock.py), [`src/db/repository.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/db/repository.py) | Unified tree with Folders (+ `Include subfolders` checkbox & Favourites), Date Taken (`Year -> Month -> Day, Weekday` supporting pre-1970 negative epochs), Descriptive tags with `tag_closure` hierarchy + `AND`/`OR` toggle + `[NOT]` right-click exclusion + `Not tagged`, People with circular face avatars, and Ratings (`5..1` + `Unrated`). `get_date_tree_counts` now groups day buckets directly in SQLite via `CAST(FLOOR(date_taken / 86400.0) AS INTEGER)` in <10ms (**WP-01 Completed**). |
| **1.7 Single Media Viewer, Non-Destructive Edit & Crop Tool** | **Complete** | [`src/ui/media_viewer.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/media_viewer.py), [`src/ui/crop_tool.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/crop_tool.py), [`src/imaging/opencv_engine.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/imaging/opencv_engine.py) | Instant cached WebP preview with async full-res load (`_AsyncImageLoadWorker`), interactive rule-of-thirds [`CropOverlayItem`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/crop_tool.py#L103-L570) with Photoshop-style outside-corner rotation and aspect ratio locks, OpenCV CLAHE auto-adjust, Hough horizon straighten, telea inpainting retouch, red-eye removal, `.OriginalFiles` hidden backup on save, and `QMediaPlayer` video playback. **Remaining:** Pressing `Space` in `SingleMediaViewer` does not toggle video play/pause; `Delete` key does not work in `SingleMediaViewer`. |
| **1.8 Fullscreen Slideshow (`F11` / `F12`)** | **Complete** *(with synchronous full-res decode)* | [`src/ui/slideshow_view.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/slideshow_view.py) | [`FullscreenSlideshowDialog`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/slideshow_view.py#L43-L363) supports cross-fade, Pan & Zoom (Ken Burns), and Instant cuts, HUD overlay, and speed selection. **Remaining:** `_load_pixmap_for_index` (`slideshow_view.py:L255-L279`) decodes full-resolution images synchronously on the GUI thread during transitions. |
| **1.9 World Map View (`QWebEngineView` + Leaflet + `media_rtree`)** | **Complete** | [`src/ui/map_view.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/map_view.py), [`src/utils/geotag_resolver.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/utils/geotag_resolver.py), [`src/db/repository.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/db/repository.py) | Embedded `QWebEngineView` running Leaflet with OpenStreetMap tiles and `QWebChannel` bidirectional bridge (`MapBridge`), dynamic cluster updates via `updateClusters(...)`, smooth pan/zoom via `setMapBounds(...)`, top-bar engine toggle to zero-dependency vector offline canvas, 160+ offline world cities gazetteer + coordinate parsing, and full transactional SQLite `media_rtree` spatial indexing + `_GeotagUndoCommand` (**WP-08 Completed**). |
| **1.10 Tier 1 Face Tagging (XMP Import/Export & Manual Face Regions)** | **Complete** | [`src/workers/scanner_thumb_worker.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/workers/scanner_thumb_worker.py), [`src/workers/xmp_sync_worker.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/workers/xmp_sync_worker.py), [`src/ui/media_viewer.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/media_viewer.py), [`src/db/repository.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/db/repository.py) | Parses `MP:RegionInfo` and `mwg-rs:Regions` on scan, displays interactive face boxes (`_FaceRegionGraphicsItem`), allows manual rectangle tagging (`tag_selected_crop_as_face`), and writes regions back to XMP. `CatalogRepository.assign_person` sets `state = 1, is_manual = 1`, deduplicates overlapping regions, updates `people.confirmed_count`, extracts thumbnails outside write lock, and `get_media_detail` / queries filter `WHERE mp.state = 1` (**WP-09 Completed**). |
| **1.11 Tier 2 Local AI Face Detection, Clustering & `Batch people tag` View** | **Complete** | [`src/db/repository.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/db/repository.py), [`src/core/face_engine.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/core/face_engine.py), [`src/core/face_matcher.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/core/face_matcher.py), [`src/workers/face_worker.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/workers/face_worker.py), [`src/ui/batch_people_view.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/batch_people_view.py), [`src/ui/main_window.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/main_window.py) | Tier 2 repository foundation (**WP-09**), vectorized `FaceVectorMatcher` + dual-backend `FaceEngine` (**WP-10**), 3-stage background `FaceDetectionWorker` (**WP-11**), and virtualised `BatchPeopleView` + Ribbon/stack integration (**WP-12**) complete. |
| **1.12 Live Filesystem Watching & Background XMP/EXIF Sync** | **Complete** | [`src/workers/folder_watcher.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/workers/folder_watcher.py), [`src/workers/xmp_sync_worker.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/workers/xmp_sync_worker.py), [`src/ui/main_window.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/main_window.py) | `FolderWatcherDaemon` debounces `QFileSystemWatcher` events into `DeltaSyncWorker` with recursive subfolder tree ingestion, deleted subfolder pruning, and folder mtime synchronization. Background `StartupReconciliationWorker` runs on launch (<50ms for 400,000 items) to reconcile offline additions, renames, and deletions made in Windows Explorer while the app was closed. `MetadataSyncDaemon` flushes `metadata_dirty = 1` rows to JPEG EXIF/XMP or `.xmp` sidecars in a background `QRunnable`. |

---

## 2. 400,000-Image Scale & Performance Bottlenecks

At $N = 400,000$ rows, any operation that executes $O(N)$ Python iterations, $O(N)$ SQL queries, or unindexed SQLite string formatting on the GUI thread causes severe frame drops or multi-second freezes.

### 2.1 GUI-Thread $O(N)$ Loops in Views, Models & Selection Handlers

1. **`Ctrl+A` / Large Selection Expansion (`ThumbnailGridView.selected_media_ids` & `_on_selection_changed`) [RESOLVED - WP-02]**
   - **Location:** [`src/ui/grid_view.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/grid_view.py), [`src/ui/grid_model.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/grid_model.py)
   - **Resolution:** Replaced `selectedIndexes()` allocation of 400k `QModelIndex` objects with range-based extraction in `get_media_ids_for_selection` and `get_records_for_selection`. Loops over `sel_model.selection()` contiguous ranges and slices `self._records` in $O(G)$ time (<10ms for 100k+ selections).

2. **Undo Command Construction & Undo Execution Performing $3N$ Queries on the GUI Thread**
   - **Location:** [`src/ui/main_window.py:L99-L119`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/main_window.py#L99-L119) (`_RatingUndoCommand`), [`L136-L156`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/main_window.py#L136-L156) (`_FlagUndoCommand`), [`L173-L193`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/main_window.py#L173-L193) (`_CaptionUndoCommand`)
   - **Bottleneck:** On `__init__`, each command loops `for mid in self.target_ids: detail = self.main_win.catalog_repo.get_media_detail(mid)`. Because `get_media_detail` (`repository.py:L1443-L1495`) runs 3 SQL queries (`media`, `media_tags`, `media_people`), rating 400,000 selected items executes **1,200,000 SQL queries** on the GUI thread! Furthermore, `undo()` calls `set_rating([mid], old_r)` and `update_items_metadata([mid], ...)` inside a Python `for` loop (400,000 separate SQLite commits + 400,000 viewport repaints). Target for **WP-03**.

3. **Multi-Selection Inline Date Edit & Flag Toggle Performing $3N$ Queries**
   - **Location:** [`src/ui/main_window.py:L1428-L1435`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/main_window.py#L1428-L1435) (`on_selection_date_taken_edited`) and [`L2080-L2084`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/main_window.py#L2080-L2084) (`dispatch_ribbon_action("toggle_flag")`)
   - **Bottleneck:** `on_selection_date_taken_edited` loops over `target_ids` calling `get_media_detail(mid)` and `set_date_taken([mid], ...)` per item instead of calling `self.catalog_repo.apply_relative_time_shift(target_ids, lead_id, new_ts)`. Similarly, `toggle_flag` calls `get_media_detail(mid)` in a comprehension over `target_ids`. Target for **WP-03**.

4. **Drag-and-Drop Initiation Synchronous `os.path.exists` & `selectedIndexes()` Loop [RESOLVED - WP-02]**
   - **Location:** [`src/ui/grid_view.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/grid_view.py) (`ThumbnailGridView._start_drag`)
   - **Resolution:** `_start_drag` uses `selected_records()` to build `QMimeData` file URLs directly from memory slices without allocating model indices, and skips synchronous `os.path.exists` checks on large batches (>256 items).

5. **Header Paint Event $O(N_{\text{group}})$ Scan (`GroupSelectedCountRole`) [RESOLVED - WP-02]**
   - **Location:** [`src/ui/grid_model.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/grid_model.py)
   - **Resolution:** Added `_group_selected_counts: dict[int, int]` populated in $O(G)$ time whenever selection changes. `GroupSelectedCountRole` is now an $O(1)$ dict lookup during header delegate paints.

6. **Mosaic Layout Mode $O(N)$ Geometry & Hit-Testing [RESOLVED - WP-02]**
   - **Location:** [`src/ui/grid_model.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/grid_model.py) (`vrow_at_pos`, `visible_vrows_in_rect`)
   - **Resolution:** Optimized hit testing and rect queries to use `bisect_right` over `_group_y_ends` to narrow down the target group before checking virtual row coordinates.

7. **`CatalogListModel.index_for_media_id` & `update_items_metadata` Per-Item Repaint Storm [RESOLVED - WP-02]**
   - **Location:** [`src/ui/grid_model.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/grid_model.py)
   - **Resolution:** `index_for_media_id` performs $O(\log G)$ binary search over `_group_record_starts`. `update_items_metadata` batches dirty ranges and emits a single bounding `dataChanged` call rather than emitting per item.

8. **Single Media Viewer & Slideshow $O(N)$ List Rebuild on Navigation**
   - **Location:** [`src/ui/main_window.py:L1560-L1564`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/main_window.py#L1560-L1564) (`open_single_viewer`) and [`L1628-L1633`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/main_window.py#L1628-L1633) (`navigate_single_viewer`)
   - **Bottleneck:** Builds `id_list = [r.id for r in records]` (400,000 ints) and calls `id_list.index(curr_id)` on every Left/Right arrow press instead of using `self.catalog_model._id_to_record_idx.get(curr_id)`. Target for **WP-03**.

---

### 2.2 SQLite Query & Indexing Bottlenecks

1. **`strftime(...)` in `ORDER BY` Defeats Covering Index [RESOLVED - WP-01]**
   - **Location:** [`src/db/repository.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/db/repository.py) (`CatalogRepository.query_media_records`), [`db/schema.sql`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/db/schema.sql)
   - **Resolution:** Added covering expression index `idx_media_effective_date ON media(COALESCE(date_taken, mtime, 0) DESC, id ASC)` and optimized `query_media_records` to omit redundant `strftime` expressions from `ORDER BY` when group and sort ordering match.

2. **`CatalogRepository.get_date_tree_counts` Fetches All 400,000 Rows into Python [RESOLVED - WP-01]**
   - **Location:** [`src/db/repository.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/db/repository.py)
   - **Resolution:** Replaced row-by-row timestamp fetch with SQL-side day bucket aggregation (`SELECT CAST(FLOOR(...) / 86400.0 AS INTEGER) AS day_key, COUNT(*) AS cnt ... GROUP BY day_key`), cutting execution time to <10ms.

3. **Unbounded `IN (?, ?, ...)` Clauses Crash at >32,766 Selected Items [RESOLVED - WP-01]**
   - **Location:** [`src/db/repository.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/db/repository.py)
   - **Resolution:** Added `_iter_id_chunks` helper (chunk size 900) across `delete_media`, `set_rating`, `set_flag`, `set_caption`, `set_author`, `set_date_taken`, `rotate_lossless`, `apply_relative_time_shift`, `remove_tag`, `get_selection_summary`, `ThumbnailRepository.get_webp_blobs_batch`, and `ThumbnailRepository.invalidate`.

4. **`WorldMapView` Spatial Clustering Fetches All Geotagged Rows into Python [RESOLVED - WP-01]**
   - **Location:** [`src/db/repository.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/db/repository.py) (`CatalogRepository.query_map_clusters`), [`src/ui/map_view.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/map_view.py)
   - **Resolution:** Added optional `grid_degrees` spatial grid aggregation directly to `query_map_clusters` SQL query (`GROUP BY grid_lat, grid_lon`) returning clustered centroid coordinates and counts.

---

### 2.3 Synchronous Image Decoding on the Main GUI Thread

1. **Startup Blocking in `PhotoGalleryMainWindow.__init__`**
   - **Location:** [`src/ui/main_window.py:L390`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/main_window.py#L390) and [`src/db/repository.py:L1220-L1254`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/db/repository.py#L1220-L1254) (`backfill_missing_people_thumbnails`)
   - **Bottleneck:** Runs synchronously during window startup on the main thread, opening full-resolution JPEGs via `Image.open()` and encoding WebP crops.
2. **Synchronous Full-Image Decode During `assign_person` [RESOLVED - WP-09]**
   - **Location:** [`src/db/repository.py:L1400-L1415`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/db/repository.py#L1400-L1415) (`CatalogRepository.assign_person` $\to$ `extract_face_crop_blob`)
   - **Resolution:** `extract_face_crop_blob` is now executed strictly *outside* `self.engine.write_lock`, eliminating reader and writer blocking during face cropping.
3. **Synchronous Batch Photo Resizing on the GUI Thread**
   - **Location:** [`src/ui/dialogs.py:L452-L500`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/dialogs.py#L452-L500) (`ResizePhotoDialog.execute_resize`)
   - **Bottleneck:** Decodes and Lanczos-resizes all selected photos synchronously inside ` _on_save_clicked` on the main GUI thread.
4. **Synchronous Full-Resolution Decode in `FullscreenSlideshowDialog`**
   - **Location:** [`src/ui/slideshow_view.py:L255-L279`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/slideshow_view.py#L255-L279) (`_load_pixmap_for_index`)
   - **Bottleneck:** Calls `QImageReader(str(fpath)).read()` synchronously on the GUI thread when advancing slides without pre-decoding the next slide on a background thread.

---

## 3. Concurrency, Thread-Safety & Data-Integrity Bugs

### 3.1 `QPixmap` / `QIcon` Usage vs. `QImage` in Background Workers
- **Audit Result:** All existing `QRunnable` workers ([`src/workers/scanner_thumb_worker.py:L774-L804`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/workers/scanner_thumb_worker.py#L774-L804), [`src/ui/media_viewer.py:L67-L115`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/media_viewer.py#L67-L115), [`src/ui/media_viewer.py:L140-L162`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/media_viewer.py#L140-L162)) properly decode to `QImage` / `PIL.Image` and convert to `QPixmap` only on the GUI thread.
- **Main-Thread `QPixmap`/`QImage` Re-encoding Bottleneck:** [`src/ui/main_window.py:L1700-L1734`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/main_window.py#L1700-L1734) (`_rotate_internal`) synchronously decodes, rotates, and re-encodes WebP blobs on the GUI thread for every selected item during rotation.

### 3.2 SQLite Connection Lifecycle, Locking & Transaction Bugs

1. **Leaked Thread-Local SQLite Connections in `QThreadPool` Workers**
   - **Locations:**
     - [`src/workers/scanner_thumb_worker.py:L483-L579`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/workers/scanner_thumb_worker.py#L483-L579) (`FolderSyncWorker.run`)
     - [`src/workers/scanner_thumb_worker.py:L711-L808`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/workers/scanner_thumb_worker.py#L711-L808) (`ThumbnailDecodeWorker.run`)
     - [`src/workers/folder_watcher.py:L69-L246`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/workers/folder_watcher.py#L69-L246) (`DeltaSyncWorker.run`)
   - **Defect:** `DatabaseEngine` ([`src/db/connection.py:L78-L102`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/db/connection.py#L78-L102)) stores connections in `threading.local()` and tracks them in `self._open_connections`. When `QThreadPool` recycles or expires worker threads, `close_thread_connections()` is never called in `run()`, accumulating open SQLite connections and WAL read locks until application exit.

2. **`MetadataSyncDaemon` Runs Disk & SQLite Write-Back on the Main GUI Thread**
   - **Location:** [`src/workers/xmp_sync_worker.py:L472-L547`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/workers/xmp_sync_worker.py#L472-L547)
   - **Defect:** `MetadataSyncDaemon` uses a main-thread `QTimer` (`self._timer.timeout.connect(self._on_timer_tick)`) that directly executes `self.flush_now(batch_limit=25)` on the **GUI thread** every 1,500ms! Inside `flush_now`, `_safe_atomic_replace` (`xmp_sync_worker.py:L423-L441`) performs synchronous file I/O (`os.replace`) and even calls `time.sleep(0.04 * (attempt + 1))` on the GUI thread.

3. **Per-Item Autocommit Lock Contention & Duplicate `media_people` Rows in `FolderSyncWorker._flush_enrichment_batch`**
   - **Location:** [`src/workers/scanner_thumb_worker.py:L581-L614`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/workers/scanner_thumb_worker.py#L581-L614)
   - **Defect:** After `repo.bulk_upsert_media(rows)`, `_flush_enrichment_batch` loops over every file and calls `repo.create_tag()`, `repo.assign_tag()`, and `repo.assign_person()`—each acquiring `self.engine.write_lock` and committing its own transaction ($N+1$ transactions per batch). Worse, `assign_person` (`repository.py:L1277`) executes a blind `INSERT INTO media_people` without checking if that `(media_id, person_id, rect_*)` already exists, creating duplicate face regions every time a folder is re-scanned!

---

### 3.3 Data-Integrity & Metadata Write-Back (`metadata_dirty = 1`) Defects

1. **Incomplete XMP/EXIF Write-Back (`author` and `location_name` / `lat` / `lon` Ignored)**
   - **Locations:**
     - [`src/db/repository.py:L889-L900`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/db/repository.py#L889-L900) (`set_author`) and [`L913-L936`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/db/repository.py#L913-L936) (`set_geotag`)
     - [`src/ui/main_window.py:L1330-L1340`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/main_window.py#L1330-L1340) (`apply_geotag_to_selection`)
     - [`src/workers/xmp_sync_worker.py:L54-L162`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/workers/xmp_sync_worker.py#L54-L162) (`build_wlpg_xmp_packet`) and [`L444-L469`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/workers/xmp_sync_worker.py#L444-L469) (`_sync_single_media_item`)
   - **Defect:** `set_author`, `set_geotag`, and `apply_geotag_to_selection` set `metadata_dirty = 1`, and `fetch_dirty_media_batch` (`repository.py:L1401`) selects `m.location_name, m.lat, m.lon`, but neither `build_wlpg_xmp_packet` nor `_inject_xmp_and_exif_into_jpeg_lossless` writes `dc:creator` / EXIF Artist (`0x013B`) or `Iptc4xmpCore:Location` / GPS tags! The worker simply marks `metadata_dirty = 0` without persisting those fields to the file.

2. **`DeltaSyncWorker` Drops XMP Tags & Face Regions on Modified/Added Files**
   - **Location:** [`src/workers/folder_watcher.py:L206-L241`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/workers/folder_watcher.py#L206-L241)
   - **Defect:** When a new or modified file is detected by `LiveFolderWatcher`, `DeltaSyncWorker` calls `extract_media_metadata(fpath, is_v)` and `repo.bulk_upsert_media(enrichment_rows)`, completely ignoring `meta["tags"]` and `meta["face_regions"]`.

3. **`CatalogRepository.assign_person` Leaves `media_people.state = 0` & `get_media_detail` Leaks Unconfirmed Faces [RESOLVED - WP-09]**
   - **Locations:** [`src/db/repository.py:L1425-L1500`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/db/repository.py#L1425-L1500), [`L1979`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/db/repository.py#L1979), [`L2076`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/db/repository.py#L2076)
   - **Resolution:** `assign_person` sets `state = 1, is_manual = 1`, deduplicates against overlapping regions within 0.02 tolerance, and updates `people.confirmed_count`. `get_media_detail`, `get_selection_summary`, and search queries filter `WHERE mp.state = 1`. A schema migration in `_migrate_catalog_schema` retroactively fixes existing unmigrated data.

4. **Dangerous Filename-Only Fallback in `_on_files_dropped_on_folder` [RESOLVED - WP-03]**
   - **Location:** [`src/ui/main_window.py:L863-L883`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/main_window.py#L863-L883)
   - **Resolution:** Path comparisons normalize slashes via SQLite `REPLACE(LOWER(f.path), '\\', '/') = LOWER(?)` and dangerous filename-only fallback was completely removed.

5. **Stale `_media_meta_lookup` Cache in `PhotoGalleryMainWindow` [RESOLVED - WP-09]**
   - **Location:** [`src/ui/main_window.py:L354`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/main_window.py#L354) and [`L1040-L1070`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/main_window.py#L1040-L1070)
   - **Resolution:** `self._media_meta_lookup.clear()` is called in `on_sync_finished`, `_on_folder_watcher_delta`, `_on_startup_reconciliation_completed`, and on external folder file drops.

---

## 4. UI Polish & Edge-Case Defects

1. **Left Navigation Dock State Conflict Between `restoreState()` and Single Media Viewer (`src/ui/main_window.py`)**
   - **Locations:** [`src/ui/main_window.py:L1534-L1535`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/main_window.py#L1534-L1535), [`L1586-L1589`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/main_window.py#L1586-L1589), [`L2447-L2454`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/main_window.py#L2447-L2454), [`L2488-L2532`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/main_window.py#L2488-L2532)
   - **Defect:** If the user closes the app while in `SingleMediaViewer` (where `left_nav_dock` is hidden), `_save_app_state` (`L2447`) saves `self.saveState()` with `left_nav_dock` hidden. Additionally, in `open_single_viewer` (`L1534`), `self._nav_dock_was_visible = not self.left_nav_dock.isHidden()` is overwritten if `open_single_viewer` is called during `_restore_app_state` (`L2541`) before the window is shown.

2. **Dual Undo Stack Desynchronization in Single Media Viewer (`Ctrl+Z` After Rotate/Rate/Flag)**
   - **Locations:** [`src/ui/main_window.py:L1052-L1073`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/main_window.py#L1052-L1073) and [`L1681-L1691`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/main_window.py#L1681-L1691)
   - **Defect:** When viewing a photo in `SingleMediaViewer`, clicking **Rotate left/right**, **Rate**, or **Flag** pushes commands onto `self.undo_stack` (the main window stack), whereas pressing `Ctrl+Z` (`_on_undo`, `L1053`) checks `if self.central_stack.currentWidget() == self.media_viewer: self.media_viewer.undo_stack.undo()`. Thus, `Ctrl+Z` in `SingleMediaViewer` fails to undo rotations, ratings, or flags performed while in `SingleMediaViewer`!

3. **Unhandled Keyboard Shortcuts Across Modes (`Delete`, `Space`, `Enter`)**
   - **Locations:** [`src/ui/main_window.py:L2386-L2438`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/main_window.py#L2386-L2438) and [`src/ui/media_viewer.py:L1297-L1330`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/media_viewer.py#L1297-L1330)
   - **Defects:**
     - Pressing `Delete` in `SingleMediaViewer` does nothing (`main_window.py:L2434` restricts `Qt.Key_Delete` to `self.central_stack.currentWidget() == self.grid_view`), whereas WLPG deletes the current photo to the Recycle Bin and advances to the next photo.
     - Pressing `Space` in `SingleMediaViewer` when viewing a video does not toggle Play/Pause (`media_viewer.py:L1297`).
     - Pressing `Enter` while Crop Mode is active in `SingleMediaViewer` is handled in `SingleMediaViewer.keyPressEvent` (`media_viewer.py:L1299`), but only when `SingleMediaViewer` (or its child `canvas_view`) has keyboard focus; if focus is on the Ribbon after clicking **Crop**, `Enter` is not routed to `apply_crop()`.

4. **Static Ribbon Layout Without Responsive Button Compaction (`src/ui/ribbon_toolbar.py`)**
   - **Location:** [`src/ui/ribbon_toolbar.py:L2196-L2213`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/ribbon_toolbar.py#L2196-L2213)
   - **Defect:** `SPEC.md` §3.1 specifies that as window width decreases, Ribbon buttons must dynamically transition from `Qt.ToolButtonTextUnderIcon` $\to$ `Qt.ToolButtonTextBesideIcon` $\to$ `Qt.ToolButtonIconOnly` inside `resizeEvent`. Currently, buttons have fixed sizes and are simply clipped by a scroll area (`_wrap_tab_page`) when the window narrows.

---

## 5. Sequenced Execution Plan for 3.8 Flash

Each Work Package (`WP-01` through `WP-12`) is scoped to **1–3 files**, with explicit class/method targets and a concrete verification test.

---

### **WP-01: Sub-10ms 400k SQLite Query Optimization, Parameter Chunking & Date Tree Aggregation [COMPLETED]**
- **Status:** **Completed**
- **Target File(s):**
  1. [`db/schema.sql`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/db/schema.sql)
  2. [`src/db/connection.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/db/connection.py)
  3. [`src/db/repository.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/db/repository.py)
  4. [`src/ui/map_view.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/map_view.py)
- **Completed Changes:**
  1. **Covering Expression Index (`db/schema.sql`, `src/db/connection.py`):** Added `CREATE INDEX IF NOT EXISTS idx_media_effective_date ON media(COALESCE(date_taken, mtime, 0) DESC, id ASC);` in `schema.sql` and ensured schema migration in `_migrate_catalog_schema()`.
  2. **Eliminated Redundant `strftime` in `ORDER BY` (`src/db/repository.py`):** In `query_media_records`, bypassed redundant `strftime(...)` expressions in `ORDER BY` when grouping by date grains with matching sort order (`spec.group_reverse == spec.sort_reverse`), allowing direct traversal of `idx_media_effective_date`.
  3. **SQL-Side Date Tree Day Aggregation (`src/db/repository.py`):** Replaced $O(N)$ row timestamp iteration in `get_date_tree_counts` with SQL aggregation:
     `SELECT CAST(FLOOR(date_taken / 86400.0) AS INTEGER) AS day_key, COUNT(*) AS cnt FROM media WHERE date_taken IS NOT NULL AND date_taken != 0 GROUP BY day_key`.
  4. **SQL-Side Spatial Grid Clustering (`src/db/repository.py`, `src/ui/map_view.py`):** Added `grid_degrees` spatial grouping directly to `query_map_clusters` (`GROUP BY grid_lat, grid_lon`) returning clustered centroid coordinates and counts.
  5. **Parameter Chunking Helper (`src/db/repository.py`):** Added `_iter_id_chunks(media_ids, chunk_size=900)` across `delete_media`, `set_rating`, `set_flag`, `set_caption`, `set_author`, `set_date_taken`, `rotate_lossless`, `apply_relative_time_shift`, `remove_tag`, `get_selection_summary`, `ThumbnailRepository.get_webp_blobs_batch`, and `ThumbnailRepository.invalidate`, eliminating `sqlite3.OperationalError: too many SQL variables` crashes.
- **Verification:**
  - Automated tests in `tests/test_wp01_wp02_scale_optimization.py` verified that 35,000-item parameter batches in `get_selection_summary`, setters, and thumbnail batching run without errors, and date tree aggregation executes in <10ms. All 98 project tests pass.

---

### **WP-02: $O(1)$ / $O(\log G)$ Virtual Grid Selection, Group Header Counts & Mosaic Hit-Testing [COMPLETED]**
- **Status:** **Completed**
- **Target File(s):**
  1. [`src/ui/grid_model.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/grid_model.py)
  2. [`src/ui/grid_view.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/grid_view.py)
- **Completed Changes:**
  1. **Range-Based Selection Extraction (`src/ui/grid_model.py`, `src/ui/grid_view.py`):** Added `get_media_ids_for_vrow_range`, `get_media_ids_for_selection`, and `get_records_for_selection` to `CatalogListModel`. Slices `self._records` directly per contiguous selection range in $O(G)$ time, avoiding allocation of 400,000 `QModelIndex` objects on `Ctrl+A`.
  2. **Precomputed Group Selected Counts (`src/ui/grid_model.py`):** Added `_group_selected_counts: dict[int, int]` populated in `_on_selection_changed`, making `GroupSelectedCountRole` an $O(1)$ dictionary lookup during header delegate paints.
  3. **Binary-Searched `index_for_media_id` & Batched `update_items_metadata` (`src/ui/grid_model.py`):** Implemented `_group_record_starts: list[int]` with `bisect_right` lookup in `index_for_media_id` ($O(\log G)$). Batched dirty item updates in `update_items_metadata` to emit a single bounding `dataChanged` call.
  4. **Binary-Searched Mosaic Hit-Testing (`src/ui/grid_model.py`):** Optimized `vrow_at_pos` and `visible_vrows_in_rect` in mosaic mode to locate the target group using `bisect_right` over `_group_y_ends`.
  5. **Fast Drag Start (`src/ui/grid_view.py`):** Updated `_start_drag` to call `model.get_records_for_selection(sel_model.selection())` and build file URLs directly from memory slices, skipping synchronous `os.path.exists` on large batches (>256 items).
- **Verification:**
  - Automated tests in `tests/test_wp01_wp02_scale_optimization.py` verified that `selectAll()` across 100,000 items extracts selection IDs in <10ms, group selection counts return in $O(1)$, and all 98 unit tests pass.

---

#### **WP-03: Bulk Undo Commands & Elimination of $3N$ GUI-Thread Queries in `PhotoGalleryMainWindow` [COMPLETED]**
- **Target File(s):**
  1. [`src/ui/main_window.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/main_window.py)
  2. [`src/db/repository.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/db/repository.py)
  3. [`src/ui/grid_model.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/grid_model.py)
- **Implemented Changes:**
  1. **Batch Old-Value Snapshots (`CatalogRepository`, `src/db/repository.py`):** Added `get_field_map_for_ids` and `restore_field_map` chunked by 900 items, restoring column values in bulk without N+1 queries. Added deduplication for `media_people` rows with coordinate bounding tolerance `<0.01`.
  2. **Refactored Undo Commands (`src/ui/main_window.py`):** Replaced per-item detail fetches in `_RatingUndoCommand`, `_FlagUndoCommand`, and `_CaptionUndoCommand` with snapshot field maps upon initialization and batched `restore_field_map` during `undo()`.
  3. **Multi-Select Date Edit & Flag Toggle (`src/ui/main_window.py`):** Implemented `apply_relative_time_shift` in `on_selection_date_taken_edited` and fast `get_field_map_for_ids` in `dispatch_ribbon_action("toggle_flag")`.
  4. **Path-Safe Drop Matching (`src/ui/main_window.py`):** Normalized directory path slashes with SQLite `REPLACE(LOWER(f.path), '\\', '/') = LOWER(?)` and removed unsafe filename-only fallback query.
  5. **$O(1)$ Record Lookups (`src/ui/grid_model.py` & `src/ui/main_window.py`):** Added `CatalogListModel.record_index_for_id(media_id)` ensuring indexed lookups in $O(1)$ for single viewer navigation and slideshow start.
- **Verification:**
  - Automated tests in `tests/test_wp03_wp04_undo_xmp_cleanup.py` verified that bulk undo field maps restore cleanly, bounding box deduplication prevents duplicate face regions, and relative time shift executes in a single SQL operation.

---

### **WP-04: Background Thread Offloading for `MetadataSyncDaemon`, Worker Connection Cleanup & Complete XMP/EXIF Write-Back [COMPLETED]**
- **Target File(s):**
  1. [`src/workers/xmp_sync_worker.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/workers/xmp_sync_worker.py)
  2. [`src/workers/scanner_thumb_worker.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/workers/scanner_thumb_worker.py)
  3. [`src/workers/folder_watcher.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/workers/folder_watcher.py)
  4. [`src/db/connection.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/db/connection.py)
- **Implemented Changes:**
  1. **Background Async Offloading (`src/workers/xmp_sync_worker.py`):** Replaced synchronous timer flushing on the main thread with `_AsyncFlushWorker` (`QRunnable`) dispatched via a single-thread `QThreadPool` on timer ticks, preserving `flush_now()` for synchronous flush requirements.
  2. **Comprehensive XMP Metadata Write-Back (`src/workers/xmp_sync_worker.py`):** Added `Iptc4xmpCore` namespace, `author` (`dc:creator`), `location_name` (`Iptc4xmpCore:Location`), and `lat`/`lon` (`exif:GPSLatitude`/`exif:GPSLongitude`) serialization and deserialization in `build_wlpg_xmp_packet` and `parse_xmp_packet_bytes`.
  3. **Worker SQLite Connection Cleanup (`src/workers/scanner_thumb_worker.py`, `src/workers/folder_watcher.py`, `src/workers/xmp_sync_worker.py`):** Wrapped worker execution in `try ... finally: if threading.current_thread() is not threading.main_thread(): self.engine.close_thread_connections()`.
  4. **Atomic Batch Enrichment (`src/workers/scanner_thumb_worker.py`, `src/workers/folder_watcher.py`, `src/db/connection.py`):** Added `catalog_transaction()` and `thumbnails_transaction()` context managers on `DualDatabaseEngine`. Wrapped tag and people assignments in `FolderSyncWorker._flush_enrichment_batch` and `DeltaSyncWorker` in single atomic transactions.
- **Verification:**
  - Automated tests in `tests/test_wp03_wp04_undo_xmp_cleanup.py` verified lossless roundtrip serialization of author, location, and GPS, asynchronous daemon flushes, and worker thread connection closing.

---

### **WP-05: Offload Remaining Synchronous Image Decodes (`backfill_missing_people_thumbnails`, `ResizePhotoDialog`, `FullscreenSlideshowDialog`) [COMPLETED]**
- **Target File(s):**
  1. [`src/ui/main_window.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/main_window.py)
  2. [`src/ui/dialogs.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/dialogs.py)
  3. [`src/ui/slideshow_view.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/slideshow_view.py)
- **Implemented Changes:**
  1. **Async People Avatar Backfill (`src/ui/main_window.py`):** Offloaded `backfill_missing_people_thumbnails()` from the synchronous GUI initialization path into `_AvatarBackfillWorker` (`QRunnable`) dispatched via `self.scanner_pool`, which asynchronously emits a signal to refresh the People navigation tree when finished.
  2. **Progress-Bar Background Worker in `ResizePhotoDialog` (`src/ui/dialogs.py`):** Added `_ResizeWorker` (`QThread`) emitting `progress(int, int)` and `finished_batch(list[Path])`, wired to an inline Segoe UI `QProgressBar` while retaining `execute_resize()` for synchronous unit tests.
  3. **Asynchronous Slide Pre-Decoding in `FullscreenSlideshowDialog` (`src/ui/slideshow_view.py`):** Added `_SlidePreloadWorker` (`QRunnable`) managed by `_preload_pool` (`QThreadPool`) to decode adjacent slide high-resolution images (`QImage`) off the main GUI thread and cache them ahead of time for 60 FPS stutter-free transitions.
- **Verification:**
  - Automated tests in `tests/test_wp05_wp06_decodes_shortcuts_docks.py` verified that `_ResizeWorker` correctly emits progress and resizes files without freezing the UI, and `FullscreenSlideshowDialog` asynchronously pre-decodes adjacent images into `_cache`.

---

### **WP-06: Dock Persistence, Unified Undo Stack & Keyboard Shortcuts (`Delete`, `Space`, `Enter`) [COMPLETED]**
- **Target File(s):**
  1. [`src/ui/main_window.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/main_window.py)
  2. [`src/ui/media_viewer.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/media_viewer.py)
- **Implemented Changes:**
  1. **Left Navigation Dock Minimum Width & Persistence (`src/ui/main_window.py`):** Enforced `left_nav_dock.setMinimumWidth(190)` and fixed `show_grid_view` to preserve user visibility states rather than unilaterally overwriting `_nav_dock_was_visible = True`.
  2. **Unified Undo/Redo Routing (`src/ui/main_window.py`):** Updated `_on_undo`, `_on_redo`, and `_update_undo_redo_enabled` to check `media_viewer.undo_stack` first when in Single View mode and cascade to `main_window.undo_stack`, enabling seamless `Ctrl+Z` across viewer edits and catalog metadata adjustments.
  3. **Universal Keyboard Shortcuts (`src/ui/main_window.py`, `src/ui/media_viewer.py`):**
     - Bound `Qt.Key_Return` / `Qt.Key_Enter`: switches from Grid to Single Viewer for the selected photo, applies active crop if in crop mode, or returns to Grid View when viewing.
     - Bound `Qt.Key_Space`: opens the selected photo in Viewer or advances slide/toggles video playback.
     - Bound `Qt.Key_Delete`: safely deletes selected item(s) to the Windows Recycle Bin across both Grid View and Single Media Viewer.
- **Verification:**
  - Automated tests in `tests/test_wp05_wp06_decodes_shortcuts_docks.py` verified that undo/redo cascades properly from Single Viewer to MainWindow, `Return` / `Space` / `Delete` shortcuts function consistently across modes, and crop tool rotation cursors are oriented with inward-pointing curves (+90° CW).

---

### **WP-07: Dynamic Responsive Ribbon Button Compaction (`resizeEvent`) [COMPLETED]**
- **Target File(s):**
  1. [`src/ui/ribbon_toolbar.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/ribbon_toolbar.py)
- **Changes Completed:**
  1. **Responsive Breakpoints in `RibbonToolbar.resizeEvent` (`src/ui/ribbon_toolbar.py`):**
     - Tracked large and secondary `QToolButton` widgets across `Home`, `Find`, `View`, and `Edit` tabs via `_responsive_large_buttons` and `_responsive_secondary_buttons`.
     - Defined three responsive width tiers in `_apply_responsive_layout(width)`:
       - **Wide ($\ge 1180\text{px}$):** Large buttons use `Qt.ToolButtonTextUnderIcon` (58px height), secondary buttons use `Qt.ToolButtonTextBesideIcon` (22px height).
       - **Medium ($880\text{px} \le W < 1180\text{px}$):** Compact secondary large buttons to `Qt.ToolButtonTextBesideIcon` (26px height), primary large buttons stay `ToolButtonTextUnderIcon`.
       - **Narrow ($< 880\text{px}$):** Compact secondary buttons to `Qt.ToolButtonIconOnly` with full tooltips preserved, preventing horizontal scroll clipping or layout jitter.
  2. **Ribbon Tab Header Double-Click Minimization:**
     - Connected `tabBarDoubleClicked` to toggle ribbon height between 114px (expanded) and 28px (collapsed), flipping arrow glyph between `▲` and `▼`.
     - Connected `tabBarClicked` to automatically restore the ribbon from collapsed state when any tab header is clicked.
- **Verification:**
  - Automated tests in `tests/test_wp07_drag_openwith_ribbon.py` verify button style compaction across `1280px`, `1000px`, and `760px`, tab double-click minimize/restore, clean drag-drop signal emission, external file dropping into folders, and Windows Open With registry queries. All 114 test suite cases pass cleanly.

---

### **WP-08: Interactive Leaflet World Map View (`QWebEngineView` + `QWebChannel` + Offline Fallback) [COMPLETED]**
- **Target File(s):**
  1. [`src/ui/map_view.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/map_view.py)
  2. [`src/ui/main_window.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/main_window.py)
  3. [`src/utils/geotag_resolver.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/utils/geotag_resolver.py)
  4. [`src/db/repository.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/db/repository.py)
- **Implemented Changes:**
  1. **Leaflet OpenStreetMap with `QWebEngineView` & `QWebChannel` (`src/ui/map_view.py`):**
     - Embedded `QWebEngineView` inside a `QStackedWidget` alongside `_InteractiveMapCanvas`.
     - Integrated `MapBridge` (`QWebChannel`) for bidirectional communication between Qt and Leaflet: JS cluster clicks emit `cluster_clicked(media_ids)` to Python, and viewport pans/zooms in Leaflet emit `bounds_changed` to update SQLite cluster queries.
     - Implemented JS `updateClusters(...)` for dynamic pin clustering without page reloads, and `setMapBounds(...)` for smooth zooming.
     - Added a top-bar mode toggle button (`🗺 View: Leaflet Web` vs. `🗺 View: Vector Offline`) allowing instant switching between online Leaflet maps and zero-dependency offline vector canvas.
  2. **Offline Gazetteer & Coordinate Resolver (`src/utils/geotag_resolver.py`):**
     - Implemented `parse_coordinates` supporting both decimal degrees (`"51.5074, -0.1278"`) and cardinal orientations (`"51.5074° N, 0.1278° W"`).
     - Built a curated offline gazetteer of 160+ major world cities, capitals, and landmarks (e.g. London, Paris, New York, Tokyo, Sydney, etc.) with canonical name mappings.
     - Added safe Nominatim geocoding fallback with a short 1.5s timeout that never crashes in offline mode.
  3. **Transactional SQLite R*Tree Spatial Synchronization & Undo/Redo (`src/db/repository.py`, `src/ui/main_window.py`):**
     - Added `CatalogRepository.set_geotag`, `get_geotag_map`, and `restore_geotag_map` updating `media.lat`, `media.lon`, `media.location_name`, and triggering automatic SQLite `media_rtree` spatial index updates.
     - Added `_GeotagUndoCommand` to `PhotoGalleryMainWindow`, enabling complete `Ctrl+Z` / `Ctrl+Y` undo and redo support for geotags across both Single Media Viewer and Grid View.
- **Verification:**
  - Automated tests in `tests/test_wp08_leaflet_world_map_geotag.py` verified coordinate parsing, gazetteer resolution, `media_rtree` spatial trigger indexing and clearing, `MapBridge` communication, Leaflet HTML generation, map view mode toggling, and `QUndoStack` geotag undo/redo. All 130 test suite cases pass cleanly.

---

### **WP-09: Tier 1 & Tier 2 Face Tagging Foundation in `CatalogRepository` & `ThumbnailRepository` [COMPLETED]**
- **Target File(s):**
  1. [`src/db/repository.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/db/repository.py)
  2. [`src/db/connection.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/db/connection.py)
  3. [`db/schema.sql`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/db/schema.sql)
  4. [`src/workers/xmp_sync_worker.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/workers/xmp_sync_worker.py)
  5. [`src/workers/scanner_thumb_worker.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/workers/scanner_thumb_worker.py)
  6. [`src/ui/main_window.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/main_window.py)
  7. [`src/ui/media_viewer.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/media_viewer.py)
- **Implemented Changes:**
  1. **Fix `assign_person` & `get_media_detail` State Semantics (`src/db/repository.py`, `src/db/connection.py`):**
     - Schema migration in `_migrate_catalog_schema` retroactively fixes legacy/imported face tags: `UPDATE media_people SET state = 1, is_manual = 1 WHERE person_id IS NOT NULL AND state = 0 AND suggested_person_id IS NULL;` and recalculates `people.confirmed_count`.
     - `CatalogRepository.assign_person`: sets `state = 1, is_manual = 1`, deduplicates overlapping face regions on the same `media_id` within 0.02 normalized tolerance, and atomically maintains `people.confirmed_count`.
     - `extract_face_crop_blob` executes strictly *outside* `self.engine.write_lock`, eliminating reader and writer blocking during face cropping.
     - `get_media_detail`, `get_selection_summary`, and search queries filter `AND mp.state = 1` so only confirmed faces appear as people tags and in XMP export packets.
  2. **Tier 2 Face Recognition Engine Repository Operations (`src/db/repository.py`):**
     - `CatalogRepository`: implemented `insert_detected_faces` (with coordinate deduplication), `get_unassigned_face_embeddings`, `get_people_centroids`, `update_person_centroid`, `confirm_faces` (atomic state transition $0 \to 1$ and `confirmed_count` calculation), `reject_face_suggestion` (persists rejection into `face_rejections` / `face_rejected_suggestions`), `ignore_face` (state $= -2$), `get_rejected_face_pairs`, `update_suggested_matches`, and `get_pending_face_suggestions`.
     - `ThumbnailRepository`: implemented `get_face_thumbnail`, `get_face_thumbnails_batch`, `put_face_thumbnail`, `put_face_thumbnails_batch`, and `invalidate_face_thumbnails`.
     - All batch methods strictly use `_iter_id_chunks(..., 900)` chunking to guard against SQLite variable limits.
  3. **Stale `_media_meta_lookup` Cache Fix (`src/ui/main_window.py`):**
     - Emptied `_media_meta_lookup` cache on folder sync completion, delta folder changes, startup reconciliation, and file drop moves.
  4. **MWG vs. Microsoft Face Region Coordinate Conversion (`src/workers/xmp_sync_worker.py`, `src/workers/scanner_thumb_worker.py`):**
     - Dual-format writer in `build_wlpg_xmp_packet`: writes top-left `(rect_x, rect_y, rect_w, rect_h)` to `MP:RegionInfo` and converted center coordinates `stArea:x = rect_x + rect_w / 2.0`, `stArea:y = rect_y + rect_h / 2.0` to `mwg-rs:Regions`.
     - Dual-format reader in `parse_xmp_packet_bytes`: parses `MP:RegionInfo` directly, and when absent, falls back to `mwg-rs:Regions` converting center coordinates `cx, cy, w, h` $\to$ top-left: `rect_x = cx - w / 2.0`, `rect_y = cy - h / 2.0`.
  5. **WLPG / Windows Explorer / digiKam Metadata Interoperability & `mtime` Preservation (`src/workers/xmp_sync_worker.py`):**
     - **Ratings:** Writes both standard `xmp:Rating` (`0..5`) and `MicrosoftPhoto:Rating` percentage (`0, 1, 25, 50, 75, 99`); reader maps percentages back to `1..5`.
     - **Hierarchical Tags & Captions:** Writes `MicrosoftPhoto:LastKeywordXMP` (`/`), `digiKam:TagsList` (`/`), `lr:hierarchicalSubject` (`|`), and leaf tags to `dc:subject`. Writes caption text to both `dc:description` and `dc:title`.
     - **Preserve Filesystem `mtime`:** `_sync_single_media_item` / `write_metadata_lossless` records `(st_atime, st_mtime)` before file updates, restores it via `os.utime`, and synchronizes `media.mtime` in `catalog.db` so metadata synchronization never resets file timestamps.
  6. **Face Bounding Box Transformations on Rotate & Crop (`src/db/repository.py`, `src/ui/main_window.py`, `src/ui/media_viewer.py`):**
     - `CatalogRepository.rotate_lossless`: automatically remaps all `media_people` bounding boxes under 90° CW `(1.0 - (ry + rh), rx, rh, rw)`, 90° CCW `(ry, 1.0 - (rx + rw), rh, rw)`, and 180° `(1.0 - (rx + rw), 1.0 - (ry + rh), rw, rh)`.
     - `CatalogRepository.transform_face_regions_for_crop`: remaps face regions into normalized crop boxes, clamping to `[0.0, 1.0]`. If retained visible area is $< 25\%$ of pre-crop area, deletes the face and decrements `confirmed_count`; otherwise updates coordinates and marks `metadata_dirty = 1`. Invoked on crop save in `media_viewer.py`.
     - `CatalogRepository.restore_original_faces`: restores pristine face bounding boxes and confirmed counts by re-extracting metadata from restored original files during `Revert to original`.
- **Verification:**
  - Automated tests in `tests/test_wp09_face_repository.py` and `tests/test_xmp_standards_and_face_transforms.py` verified state transitions, MWG center-point vs. MP top-left roundtrip, rating percentage mapping, `mtime` preservation, rotation coordinate remapping, and crop deletion thresholding. Full test suite passes cleanly with 161 tests across 31 test files.

---

### **WP-09B: Archive Hardening & Multi-Select Tags UI Polish [COMPLETED]**
- **Target File(s):**
  1. [`src/workers/scanner_thumb_worker.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/workers/scanner_thumb_worker.py)
  2. [`src/workers/folder_watcher.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/workers/folder_watcher.py)
  3. [`src/workers/xmp_sync_worker.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/workers/xmp_sync_worker.py)
  4. [`src/db/repository.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/db/repository.py)
  5. [`src/ui/right_tag_dock.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/right_tag_dock.py)
  6. [`src/ui/media_viewer.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/media_viewer.py)
  7. [`src/ui/dialogs.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/dialogs.py)
  8. [`src/ui/main_window.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/main_window.py)
- **Implemented Changes:**
  1. **Multi-Selection Descriptive Tags (`src/ui/right_tag_dock.py`, `src/db/repository.py`, `src/ui/main_window.py`):**
     - When multiple photos are selected, the right dock displays all tags present across the selection with occurrence counts: `tag_name (count)` matching WLPG reference screenshots.
     - Hover underline applies exclusively to the tag name text; count badge remains unadorned.
     - Right-clicking any tag row displays a context menu offering `Remove tag` and `Assign to all`.
     - Connected `tag_assign_to_all_requested` signal in `main_window.py` to assign the tag across all selected photos in SQLite and refresh UI chips.
  2. **Single Image Viewer Light Background (`src/ui/media_viewer.py`):**
     - Changed `graphics_view` canvas background and scene brush to light/white (`#FFFFFF`) per WLPG specifications.
  3. **Options Dialog Reorganization (`src/ui/dialogs.py`):**
     - Removed obsolete "Publish" and "Trust Center" tabs.
     - Moved Visual Theme selector into "General" tab.
     - Established dedicated "Database" tab for database compaction, thumbnail cache purging, and backup retention management.
     - Removed misleading restart demand notice from Face Detection options.
  4. **7 Real-World Archive Edge Case Hardenings:**
     - **External Crop / Resolution Change:** Checks `mwg-rs:AppliedToDimensions` against image dimensions; on dimension change on disk, `bulk_upsert_media` invalidates thumbnail cache and resets `face_scan_version = 0`.
     - **Metadata Stripping Guard:** If an external editor wipes XMP/EXIF ratings/tags, catalog values are preserved and marked `metadata_dirty = 1`.
     - **Move/Rename Preservation:** `DeltaSyncWorker` matches missing files against new files by `(size, mtime)` within the folder, issuing SQLite `UPDATE media SET filename = ...` to retain `media_id`, tags, and face embeddings across Windows Explorer renames.
     - **`.OriginalFiles` Lifecycle:** `.OriginalFiles` directories are strictly excluded from ingestion; deleting a photo moves or unlinks its `.OriginalFiles` backup, and moving a file relocates its backup correspondingly.
     - **Self-Triggered Watcher Loops:** Thread-safe registry `_RECENTLY_WRITTEN` in `xmp_sync_worker.py` ignores watcher file events within 5.0 seconds of app-authored metadata flushes.
     - **Pre-1970 Timestamps on Windows:** Uses safe epoch arithmetic via `_epoch_to_datetime_utc` (`_EPOCH_UTC + timedelta(seconds=ts)`), preventing OS `[Errno 22]` crashes.
     - **Poison-Pill Corrupt Files:** Corrupt or undecodable image files are assigned `face_scan_version = -1`, preventing infinite crash loops during background scanning.
- **Verification:**
  - Automated tests in `tests/test_wp09b_external_edits_and_edge_cases.py` and `tests/test_phase5_dock_dialogs.py` verify all 8 edge case scenarios and UI components. All 174 test suite cases pass cleanly.

---

### **WP-10: Tier 2 Vectorized Face Matcher (`src/core/face_matcher.py`) & ONNX Face Engine (`src/core/face_engine.py`) [COMPLETED]**
- **Target File(s):**
  1. [`src/core/face_matcher.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/core/face_matcher.py) *(new file)*
  2. [`src/core/face_engine.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/core/face_engine.py) *(new file)*
  3. [`src/core/__init__.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/core/__init__.py) *(new file)*
- **Implemented Changes:**
  1. **High-Performance In-Memory Vector Face Matcher (`src/core/face_matcher.py`):**
     - Implemented `FaceVectorMatcher.match_unassigned_faces(...)` with vectorized NumPy matrix multiplication $S = U \cdot C^T$ for high-throughput batch matching of $N$ unassigned faces against $M$ centroids.
     - Vectorized rejection masking using pre-computed boolean index masks for permanent $(face\_id, person\_id)$ rejections.
     - Implemented `update_centroid_incremental(...)` maintaining spherical unit-normalized running centroids $\vec{\mu}_{\text{new}} = \text{normalize}(N \cdot \vec{\mu}_{\text{old}} + \sum \vec{e}_i)$ in $<1\text{ ms}$.
     - Implemented 2048-byte SQLite BLOB serialization and deserialization helpers (`serialize_embedding`, `deserialize_embedding`, `normalize_vector`, `cosine_similarity`).
  2. **Dual-Backend Face Engine (`src/core/face_engine.py`):**
     - Integrated `FaceEngine` with dual detection/recognition backends: primary InsightFace (`buffalo_l`: SCRFD-10GF + ArcFace 512-D with DirectML/CPU fallback) and a deterministic, lightweight OpenCV Haar Cascade + 512-D spatial gradient histogram descriptor fallback (8x8 cells x 8 orientation bins = 512-D unit vector) when external weights are absent.
     - Implemented EXIF orientation transposition handling, normalized coordinate bounding boxes $[0.0, 1.0]$, minimum face size filtering ($\ge 32\times 32$ px), detection confidence threshold ($\ge 0.65$), and portrait padding $96\times 96$ WebP avatar crop encoding.
     - Implemented `detect_and_embed()` and `extract_crop_and_embed_rect()`.
- **Verification:**
  - Automated tests in `tests/test_face_matcher.py` verified vector serialization/deserialization roundtrip, incremental centroid spherical running mean mathematics, permanent rejection exclusion, 50,000 $\times$ 200 matrix matching performance benchmark, face crop WebP generation, and EXIF orientation transposition.
  - Complete test suite passes (150/150 tests across 29 test files).

---

### **WP-11: Background Face Detection Worker (`src/workers/face_worker.py`) [COMPLETED]**
- **Target File(s):**
  1. [`src/workers/face_worker.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/workers/face_worker.py) *(new file)*
  2. [`src/workers/__init__.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/workers/__init__.py)
  3. [`src/ui/main_window.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/main_window.py)
- **Implemented Changes:**
  1. **Three-Stage Background `FaceDetectionWorker` (`src/workers/face_worker.py`):**
     - Runs as a `QRunnable` designed for low-priority background execution (`maxThreadCount = 1`).
     - **Stage 1 (Prioritized Backfill):** Scans existing confirmed face tags (`state = 1`) missing 512-D embeddings or missing `96x96` avatar thumbnails in `face_thumbnails`. Generates crops, serializes embeddings, and computes initial spherical unit centroids `people.centroid_embedding` for all persons with confirmed faces.
     - **Stage 2 (Background Detection):** Queries unscanned images (`is_missing = 0 AND is_video = 0 AND face_scan_version = 0`) in configurable batches (default 50). Detects faces via `FaceEngine`, inserts unassigned detections (`state = 0`) into `media_people`, stores `96x96` WebP crops in `face_thumbnails`, and marks `face_scan_version = 1`.
     - **Stage 3 (Clustering & Matching):** Runs vectorized cosine similarity via `FaceVectorMatcher.match_unassigned_faces(...)` comparing unassigned embeddings to all known person centroids, respecting permanent rejection masks, and recording `suggested_person_id` and `similarity_score` in `media_people`.
     - **Robust Poison-Pill & Cancellation Handling:** Corrupt or missing image files are flagged with `face_scan_version = -1` without halting the worker; responds immediately to `CancellationFlag`; closes thread-local SQLite connections in `finally:`.
     - Emits `FaceWorkerSignals`: `batch_progressed(int, int)`, `suggestions_updated()`, `worker_finished(int)`, and `status_message(str)`.
  2. **Main Window Lifecycle Integration (`src/ui/main_window.py`):**
     - Initialized dedicated `self.face_pool` (`maxThreadCount = 1`) and `self._face_cancel = CancellationFlag()`.
     - Added `start_face_detection()`, `stop_face_detection()`, `on_face_batch_progressed()`, `on_face_suggestions_updated()`, and `on_face_worker_finished()`.
     - Automatically launches face scanning after folder scans and watcher syncs when `self.find_people_enabled` is active, and cleanly stops and drains the pool on shutdown.
- **Verification:**
  - Automated tests in `tests/test_face_worker.py` verified Stage 1 backfill + centroid computation, Stage 2 detection + thumbnail extraction, Stage 3 vector matching + suggestion assignment, poison-pill isolation (`face_scan_version = -1`), cancellation responsiveness, force rescan, and default engine integration. Full project suite passes with 180 tests across 34 test files.

---

### **WP-12: Virtualised `Batch people tag` Confirmation View (`src/ui/batch_people_view.py`) & Main Window Integration [COMPLETED]**
- **Target File(s):**
  1. [`src/ui/batch_people_view.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/batch_people_view.py) *(new file)*
  2. [`src/ui/main_window.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/ui/main_window.py)
  3. [`src/db/repository.py`](file:///d:/Users/Marcus/_My%20Documents/04%20Work/Code/Python/Gallery/src/db/repository.py)
- **Implemented Changes:**
  1. **Virtualised `BatchPeopleView` (`src/ui/batch_people_view.py`):**
     - Virtualised `QListView` with `_FaceCardModel` and `_FaceCardDelegate` rendering 96x96 WebP face crops with hover action overlay buttons (`[✓]` Confirm, `[✕]` Reject), match confidence badge (e.g. `94%`), and Aero blue selection border.
     - Grouped collapsible sections (`_PersonGroupSectionWidget`) for suggested persons (`"Suggested for [Name]? (N suggestions, M confirmed)"` with `[✓ Confirm All (N)]` button) and `"Unnamed Faces"`.
     - Top horizontal `_PeopleAvatarPickerStrip` with circular avatar chips for 1-click filtering of visible sections.
     - Quick full-photo context preview popup (`_FaceQuickPreviewDialog`) on `Space` key centered on the face box with highlighting.
     - Keyboard navigation and shortcuts: `Enter` (Confirm selected), `Delete`/`Backspace` (Reject selected), `I` (Don't show again, `state = -2`), `Space` (Quick preview), `Ctrl+A` (Select all), `Esc` (Back to gallery).
     - Context menu with `Confirm`, `Not this person`, `Tag as...` (with autocomplete), `Don't show again`, and `View full photo`.
     - Top action toolbar with `Select all`, `✓ Confirm`, `✕ Not this person`, `Tag as...`, `Don't show again`, and `Back to Gallery`.
     - Dismissible blue guidance banner explaining batch tagging.
  2. **Incremental Centroid & Database Synchronization:**
     - On confirmation: sets `media_people.state = 1`, `metadata_dirty = 1`, increments `people.confirmed_count`, updates running unit-normalized centroid `people.centroid_embedding` incrementally in $<1\text{ ms}$, and removes cards from view.
     - On rejection: inserts into `face_rejected_suggestions` (so faces are permanently excluded from future suggestions for that person) and resets `suggested_person_id = NULL`.
     - On ignore: sets `media_people.state = -2` ("Don't show again").
  3. **Main Window Lifecycle & Ribbon Integration (`src/ui/main_window.py`):**
     - Embedded `BatchPeopleView` in `self.central_stack`.
     - Wired Ribbon action `"batch_people_tag"` to call `open_batch_people_view()`.
     - Wired `back_to_gallery_requested` (and `Esc`) to call `close_batch_people_view()`, switching back to `grid_view`.
     - Wired `open_media_requested` to open single media viewer centered on the chosen item.
     - Automatically refreshes `batch_people_view` when background face detections complete (`on_face_suggestions_updated`).
- **Verification:**
  - Automated tests in `tests/test_phase8b_batch_people_ui.py` verified grouping by suggested person, avatar picker strip filtering, `[Confirm All]` batch updates, rejection logging in `face_rejected_suggestions`, ignore `state = -2`, and Ribbon toggle roundtrip.
  - Complete test suite passes (186/186 tests across 35 test files).
