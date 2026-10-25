"""
Phase 6 Automated Verification Suite:
1. SingleMediaViewer zoom modes ('fit', 'fill', '100', 'custom'), CropScene clamped rect crop,
   EditCommand QUndoStack undo/redo, '.OriginalFiles' backup & Revert to original, and static-image face tagging.
2. WorldMapView R*Tree spatial bounding-box query and clustering (<5ms).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from PIL import Image
from PySide6.QtCore import QRectF
from PySide6.QtWidgets import QApplication

from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository
from src.ui.map_view import WorldMapView, generate_leaflet_html
from src.ui.media_viewer import SingleMediaViewer


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_single_media_viewer_zoom_crop_and_undo_stack(
    qapp: QApplication, tmp_path: Path
) -> None:
    """
    Verify SingleMediaViewer:
    - ImageGraphicsView zoom modes ('fit', 'fill', '100', 'custom')
    - CropScene rubber-band crop + EditCommand QUndoStack undo/redo
    - Auto adjust, Exposure/Contrast, B&W effect
    - Non-destructive '.OriginalFiles' backup stash & 'Revert to original'
    - Static-image face region tagging (and rejection on videos)
    """
    img_path = tmp_path / "sample.jpg"
    orig_img = Image.new("RGB", (400, 300), (80, 140, 220))
    orig_img.save(img_path, quality=95)

    viewer = SingleMediaViewer()
    viewer.resize(800, 600)
    assert viewer.load_media(
        media_id=101,
        file_path=img_path,
        is_video=False,
        people_regions=[{"name": "Craig", "rect_x": 0.1, "rect_y": 0.1, "rect_w": 0.2, "rect_h": 0.2}],
    )
    assert viewer.current_pil_image is not None
    assert viewer.current_pil_image.size == (400, 300)

    # 1. Zoom modes
    viewer.set_zoom_mode("100")
    assert viewer.graphics_view.zoom_level == pytest.approx(1.0)
    viewer.set_zoom_mode("fill")
    assert viewer.current_zoom_mode == "fill"
    viewer.set_zoom_mode("fit")
    assert viewer.current_zoom_mode == "fit"

    # 2. CropScene crop + Undo/Redo
    viewer.set_crop_mode(True)
    viewer.scene.crop_rect_item.setRect(QRectF(50, 40, 200, 150))
    viewer.scene.crop_rect_item.show()
    assert viewer.apply_crop() is True
    assert viewer.current_pil_image.size == (200, 150)

    viewer.undo_stack.undo()
    assert viewer.current_pil_image.size == (400, 300)

    viewer.undo_stack.redo()
    assert viewer.current_pil_image.size == (200, 150)

    # 3. Auto adjust, Exposure, B&W effect + Undo
    viewer.apply_auto_adjust()
    viewer.apply_exposure_contrast(brightness=20, contrast=15)
    viewer.apply_preset_effect("bw")
    assert viewer.undo_stack.count() == 4

    # 4. Save with '.OriginalFiles' backup and Revert to original
    saved_events: list[tuple[int, str]] = []
    viewer.pixel_edit_saved.connect(lambda mid, bpath: saved_events.append((mid, bpath)))
    backup_path = viewer.save_current_image_with_backup()
    assert backup_path is not None
    assert Path(backup_path).exists()
    assert saved_events == [(101, backup_path)]

    # Verify edited file on disk is 200x150
    with Image.open(img_path) as disk_edited:
        assert disk_edited.size == (200, 150)

    # Revert to original -> restores 400x300 from .OriginalFiles
    assert viewer.revert_to_original(backup_path) is True
    assert viewer.current_pil_image.size == (400, 300)
    with Image.open(img_path) as disk_restored:
        assert disk_restored.size == (400, 300)

    # 5. Static-image face region tagging vs Video rejection
    face_tags: list[tuple[int, str, float, float, float, float]] = []
    viewer.face_region_tagged.connect(
        lambda mid, name, rx, ry, rw, rh: face_tags.append((mid, name, rx, ry, rw, rh))
    )
    viewer.scene.crop_rect_item.setRect(QRectF(40, 30, 80, 90))
    viewer.scene.crop_rect_item.show()
    assert viewer.tag_selected_crop_as_face("Craig") is True
    assert len(face_tags) == 1
    assert face_tags[0][0] == 101
    assert face_tags[0][1] == "Craig"
    assert face_tags[0][2] == pytest.approx(40 / 400)
    assert face_tags[0][3] == pytest.approx(30 / 300)

    # Video mode must ignore face region tagging
    viewer.current_is_video = True
    viewer.scene.crop_rect_item.setRect(QRectF(40, 30, 80, 90))
    viewer.scene.crop_rect_item.show()
    assert viewer.tag_selected_crop_as_face("Craig") is False


def test_world_map_view_rtree_spatial_clustering_under_5ms(
    qapp: QApplication, tmp_path: Path
) -> None:
    """
    Verify WorldMapView queries media_rtree and clusters GPS points in <5ms
    as the bounding box moves, and emits cluster_clicked when a cluster is selected.
    """
    engine = DualDatabaseEngine(tmp_path / "catalog.db", tmp_path / "thumbs.db")
    repo = CatalogRepository(engine)
    fid = repo.upsert_folder("D:/Travel", "Travel", None, 1700000000)

    ids = repo.bulk_upsert_media(
        [
            {
                "folder_id": fid,
                "filename": "london_1.jpg",
                "ext": ".jpg",
                "size": 2_000_000,
                "mtime": 1700000000,
                "date_taken": 1700000000,
                "lat": 51.5074,
                "lon": -0.1278,
                "location_name": "London, UK",
            },
            {
                "folder_id": fid,
                "filename": "london_2.jpg",
                "ext": ".jpg",
                "size": 2_000_000,
                "mtime": 1700000000,
                "date_taken": 1700000000,
                "lat": 51.5090,
                "lon": -0.1250,
                "location_name": "London, UK",
            },
            {
                "folder_id": fid,
                "filename": "tokyo_1.jpg",
                "ext": ".jpg",
                "size": 2_000_000,
                "mtime": 1700000000,
                "date_taken": 1700000000,
                "lat": 35.6762,
                "lon": 139.6503,
                "location_name": "Tokyo, Japan",
            },
        ]
    )

    map_view = WorldMapView()
    map_view.resize(800, 500)
    map_view.bind_repository(repo)

    # Warm up SQLite cursor then time bounding-box R*Tree lookup + clustering for London
    _ = map_view.set_viewport_bbox(-85.0, 85.0, -180.0, 180.0, repo)
    assert len(map_view.current_clusters) == 2

    t0 = time.perf_counter()
    london_clusters = map_view.set_viewport_bbox(50.0, 53.0, -2.0, 1.0, repo)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert elapsed_ms < 5.0, f"WorldMapView R*Tree bbox query took {elapsed_ms:.2f}ms (budget <5ms)"
    assert len(london_clusters) == 1
    assert london_clusters[0]["count"] == 2
    assert set(london_clusters[0]["media_ids"]) == {ids[0], ids[1]}

    # Verify clicking cluster list item emits cluster_clicked
    clicked_clusters: list[list[int]] = []
    map_view.cluster_clicked.connect(clicked_clusters.append)
    first_item = map_view.cluster_list.item(0)
    map_view._on_cluster_list_clicked(first_item)
    assert len(clicked_clusters) == 1
    assert set(clicked_clusters[0]) == {ids[0], ids[1]}

    # Verify Leaflet HTML generator includes cluster data
    html = generate_leaflet_html(london_clusters)
    assert "London, UK" in html

    engine.close_all()
