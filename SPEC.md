# PhotoGallery64 — System Specification (WLPG 64-bit Clone)

## 1. Project Goal
A high-performance 64-bit Windows 11 desktop replacement for Windows Live Photo Gallery (WLPG) built in Python 3.12+ and PySide6, capable of handling **500,000+ images and videos** with sub-10ms UI query and filtering responsiveness.

---

## 2. Core Architecture & Scale Requirements (400k Items)
1. **Virtualised Item Presentation:**
   - STRICT RULE: Never use `QListWidget` or `QTableWidget`.
   - Grid View MUST use `QListView` (`setViewMode(IconMode)`, `setUniformItemSizes(True)`) backed by a custom `QAbstractListModel` and `QStyledItemDelegate`.
   - The model holds only numeric IDs and metadata tuples in memory (~40–80 MB RAM for 400k items).
   - `QPixmap` objects are strictly generated on-demand for visible viewport cells via a thread-safe LRU memory cache.
2. **Two-Database SQLite Engine (WAL Mode, `PRAGMA synchronous = NORMAL`):**
   - **`catalog.db`**:
     - `folders`: (`id`, `path` UNIQUE, `name`, `parent_id`, `mtime`)
     - `media`: (`id`, `folder_id`, `filename`, `ext`, `size`, `mtime`, `date_taken` INTEGER indexed, `rating` INTEGER DEFAULT 0, `flag` INTEGER DEFAULT 0, `width`, `height`, `lat`, `lon`, `caption`, `is_video`, `metadata_dirty` INTEGER DEFAULT 0)
     - `tags`: (`id`, `parent_id`, `name`, `full_path` UNIQUE)
     - `tag_closure`: (`ancestor_id`, `descendant_id`, `depth`) for <5ms Boolean tag tree filtering (`AND`, `OR`, `NOT`).
     - `media_tags`: (`media_id`, `tag_id`, PRIMARY KEY(`media_id`, `tag_id`))
     - `people`: (`id`, `name`, `avatar_media_people_id`, `centroid_embedding` 512xfloat32 vector BLOB, `confirmed_count`, `thumbnail_blob`)
     - `media_people`: (`id`, `media_id`, `person_id`, `suggested_person_id`, `similarity_score`, `state` [-1=ignored, 0=unassigned/suggested, 1=confirmed], `rect_x`, `rect_y`, `rect_w`, `rect_h`, `embedding` 512xfloat32 ArcFace BLOB)
     - `face_rejected_suggestions`: (`face_id`, `person_id`, PRIMARY KEY(`face_id`, `person_id`)) preventing rejected suggestions from re-appearing on centroid shifts.
     - `media_rtree`: SQLite R*Tree virtual table (`id`, `min_lat`, `max_lat`, `min_lon`, `max_lon`) for instant World Map bounding-box queries.
   - **`thumbnails.db`**:
     - `thumbnails`: (`media_id` INTEGER PRIMARY KEY, `webp_blob` BLOB, `mtime` INTEGER, `width`, `height`) storing 256x256 WebP blobs (~8–12 KB each) to isolate BLOB I/O from catalog queries.
     - `face_thumbnails`: (`face_id` INTEGER PRIMARY KEY, `crop_webp` BLOB) storing pre-cropped 96x96 WebP face thumbnails (~2–3 KB) for instant Batch People Tag grid rendering.
3. **High-Speed Decoding:**
   - Generate thumbnails using `QImageReader` with `setAutoTransform(True)` and `setScaledSize()` (leveraging DCT sub-band fast scaling in C++), compressed to WebP.
4. **Lossless Edits & Background Metadata Write-Back:**
   - Any tag, rating, flag, caption, or Date Taken adjustment writes to `catalog.db` immediately on the UI thread (`metadata_dirty = 1`) for instantaneous UI updates.
   - A background daemon worker drains `metadata_dirty` items, writing compliant XMP (`lr:hierarchicalSubject`, `MicrosoftPhoto:LastKeywordXMP`, `MP:RegionInfo`, `dc:subject`) to file via `pyexiv2` / `ExifTool`.
   - Rotations modify EXIF orientation losslessly without touching image pixel data or stripping XMP blocks.

---

## 3. UI Layout & Component Specifications (Matching WLPG Screenshots)

### 3.1 Top Ribbon / Action Toolbar
- **Tabbed Interface:** `Home`, `Edit`, `Find`, `Create`, `View`.
- **View Tab:**
  - *Arrange List Group:* Buttons to group/sort by Date, Geotag, Person, Name, Tag, Published, Folder, Media Type, Rating.
  - *Reverse Sort:* Toggle button to flip direction.
  - *Show Details Dropdown:* Toggle overlays for Filename, File size, Date taken, Image dimensions, Rating, Caption.
  - *Tag & Caption Pane Toggle:* Shows/hides the right dock.
- **Find Tab / Quick Find Face Carousel:**
  - Horizontal bar displaying circular avatars of tagged people with names underneath.
  - Clicking an avatar immediately filters the grid to photos containing that person.
- **Search Header (Above Grid):**
  - Path / Scope indicator: e.g. `Folder: Pictures (Marcus) | Search all`
  - Inline search text box with clear (X) button.

### 3.2 Left Navigation Dock (Collapsible Tree View)
- **Additions Over Original WLPG to Preserve (Keep in Architecture):**
  1. **People Section:** Dedicated alphabetic hierarchy of tagged individuals with face avatars and direct click-to-filter.
  2. **Ratings Section:** Dedicated 5 stars down to 1 star + Unrated filter nodes with instant item counts.
  3. **Subfolders Checkbox:** Option to recursively include/exclude nested media within folder selections.
- **Unified Filter Tree:**
  - **Folders:** Directory tree with "Include Subfolders" option and Favourites.
  - **Date Taken:** Grouped tree `Year` -> `Month` -> `Day, Weekday` (e.g. `1 Jan, Thursday`), showing item count per node.
  - **Descriptive Tags:** Hierarchical tree with multi-level nesting (`PeopleTags/Family/Craig`) and checkboxes:
    - Checked: Include tag (`AND` or `OR` mode selectable).
    - Unchecked: Neutral.
    - Right-click / Alt-check: Exclude tag (`NOT`).
  - **People:** Alphabetic list of tagged individuals with photo count badges.
  - **Ratings:** 5 stars down to 1 star, plus "Unrated".

### 3.3 Center Area (`QStackedWidget`)
1. **Thumbnail Grid View (`QListView`):**
   - **Collapsible Section Headers:** Groups items by active sort criteria (e.g., `v 5 stars (1570 items, 12 selected)` or `v October 1957 (340 items)`).
   - **Card Delegate Styling:**
     - Rounded polaroid-style card with subtle border and drop shadow.
     - Top-left: Hover-activated checkbox for multi-selection.
     - Top-right: Hover-activated Flag pin icon.
     - Bottom: Interactive 5-star rating widget directly on the card.
2. **Single Media / Viewer & Fix Tab View:**
   - Full-resolution viewport with smooth zoom/pan and anchored crop rectangle (`ImageGraphicsView` & `CropScene` from `legacy_reference/viewer_extract.py`).
   - Integrated video playback (`ClickableVideoWidget` with filmstrip composite).
   - Inline "Fix" adjustments in top ribbon: Auto adjust, Crop, Retouch, Straighten, Noise reduction, Color/Exposure balance, and preset filter tiles (B&W, Sepia, Cool, Warm).
3. **World Map View:**
   - `QWebEngineView` running local Leaflet + Supercluster querying `catalog.db`'s `media_rtree` for the active map bounding box.

### 3.4 Right Dock ("Tag and Caption Pane")
- Dynamically updates based on current single or multi-selection:
  1. **People Tags:** Shows face preview thumbnails + person display name + clickable `+ Add people tag`.
  2. **Geotag:** Displays location name + `+ Add geotag`.
  3. **Caption:** Editable multi-line text input field.
  4. **Descriptive Tags:** Clickable blue chip tags with `(x)` to remove + `+ Add descriptive tag` with auto-complete.
  5. **Information Summary (EXIF):**
     - Filename, Date taken, Size, Rating (clickable stars), Flag (clickable).
     - Dimensions, Camera, Author, Exposure, Aperture, Focal length, ISO, Latitude, Longitude.

---

## 4. Key Feature Logic: Relative "Change Time Taken" Dialog
- **Dialog UI:**
  - Matches `docs/reference_ui/adjust_time.jpg`.
  - Header: *"Change the time stamp for selected photos and videos. To adjust the time stamp, select the time you'd like to change the selected file to. If you have multiple files selected, time changes will be relative to the time indicated."*
  - Controls: `QTimeEdit` (HH:mm:ss) and `QDateEdit` (d MMMM yyyy with calendar popup).
  - Current Time Label: Displays exact `date_taken` of the primary / lead selected item.
  - Status Label: Displays *"No time adjustment has been made."* initially; updates live to show offset (e.g., `Adjustment: +01:00:00 applied to 12 files`).
- **Mathematical Logic:**
  - Let $T_{\text{lead\_original}}$ be the `date_taken` timestamp (in seconds) of the lead selected item.
  - Let $T_{\text{lead\_new}}$ be the timestamp configured in the dialog.
  - Compute $\Delta t = T_{\text{lead\_new}} - T_{\text{lead\_original}}$.
  - Apply to all selected IDs:
    ```sql
    UPDATE media
    SET date_taken = date_taken + :delta_seconds,
        metadata_dirty = 1
    WHERE id IN (:selected_ids);
    ```
  - Background worker immediately queues these IDs to write the new timestamp into `DateTimeOriginal` / `XMP` on disk.

---

## 5. "Batch People Tag" & Automated Face Recognition Specification
Modeled on Windows Live Photo Gallery (`docs/reference_ui/batch_people_tag.jpg`) with Picasa-grade instant confirmation responsiveness.

### 5.1 Trigger & Ribbon Context Mode
- **Entry Points:**
  - `Batch people tag` button on the Home and Find tabs.
  - Context menu / tag pane action on any person node.
- **Top Ribbon Context Bar ("Batch People Tag" Mode):**
  - Replaces or overlays standard ribbon with contextual `Actions` group:
    - **Select all (`Ctrl+A`)**: Selects all suggestions in the active section or entire view.
    - **Confirm (`Enter`)**: Confirms currently selected face suggestions for the suggested person.
    - **Not this person (`Delete` / `Backspace`)**: Rejects suggestions for this person, adding to `face_rejected_suggestions`.
    - **Tag as...**: Autocomplete dropdown to re-assign selected faces to a different existing person or create a new person.
    - **Don't show again (`I`)**: Sets face `state = -1` (ignored, e.g. background crowd faces, statues, posters).
  - **People Avatar Picker Strip:**
    - Horizontal scrollable strip of tagged people with circular avatars and confirmed photo counts, allowing the user to filter suggestions to a specific person or view "All suggestions".

### 5.2 Top Notification / Info Banner
- Dismissible subtle notification banner placed directly above the suggestion grid:
  - Text: *"As you act on these suggestions, the faces will be removed from the list below, and the confirmed faces will be tagged in your photos."*
  - Right-hand `[✕]` dismiss button with preference persisted in settings.

### 5.3 Grouped Suggestion Sections (`BatchPeopleView` / Virtualised `QListView`)
- **Virtualised Presentation:**
  - Implemented as a virtualised `QListView` (or dedicated group delegate) holding only lightweight `FaceSuggestionRecord` tuples.
  - Memory budget: < 20 MB RAM for 50,000 pending face suggestions.
  - Renders pre-cropped 96x96 WebP thumbnails directly from `thumbnails.db`'s `face_thumbnails` table.
- **Collapsible Section Headers:**
  - Header Title Format: `Person Name? (28 suggestions, 25 selected, 20400 confirmed)`
  - Header Action: Prominent one-click `[Confirm Selected]` button that confirms all checked/selected faces in that section immediately without having to use the top ribbon.
- **Face Card Delegate (96x96 Tiles):**
  - Clean square polaroid-style tile displaying the 96x96 face crop (`face_thumbnails.crop_webp`).
  - Selection border: Blue Aero accent border when selected.
  - Multi-selection support: Rubber-band marquee selection (`Ctrl`/`Shift` + click/drag).
  - **Hover Overlay Controls:**
    - Top-left: Selection checkbox.
    - Top-right: Quick one-click `[✓]` (Confirm) and `[✗]` (Not this person) circular action buttons appearing smoothly on mouse hover.
    - Bottom: Subtle similarity match percentage badge (e.g. `94%`).

### 5.4 Context Menu & Keyboard Ergonomics
- **Right-Click Context Menu (on single or multi-selection):**
  - `Confirm` (`Enter`)
  - `Not this person` (`Delete` / `Backspace`)
  - `Tag as... ▸` (autocomplete popup of people)
  - `Don't show again` (`I`)
  - `View full photo` (`Space`)
- **Keyboard Shortcuts:**
  - `Enter`: Confirm all selected faces.
  - `Delete` or `Backspace`: Mark selected faces as "Not this person" (persisting to `face_rejected_suggestions`).
  - `I`: "Don't show again" (`state = -1`).
  - `Space`: **Instant Full-Photo Quick-Preview Popover**. Shows the full photo in a non-modal overlay centered on the face box with context (e.g. child clothing, event setting, companion people) to easily disambiguate difficult/ambiguous faces without leaving the batch workflow.
  - `Ctrl+A`: Select all visible suggestions.
  - `Esc`: Dismiss quick-preview or exit Batch People Tag view back to main grid.