"""
Phase 5 Automated Verification Suite:
1. TagAndCaptionDock single and multi-selection binding, shared tag chips, EXIF display, and edit signals.
2. ChangeTimeTakenDialog relative Δt = T_lead_new - T_lead_original calculation (including pre-1970 dates)
   and batch relative shift application in CatalogRepository.
3. AppSettingsDialog 30-day backup retention and maintenance triggers.
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from src.db.connection import DualDatabaseEngine
from src.db.repository import _EPOCH_UTC, CatalogRepository, _epoch_to_datetime_utc
from src.ui.dialogs import AppSettingsDialog, ChangeTimeTakenDialog
from src.ui.right_tag_dock import TagAndCaptionDock


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_tag_and_caption_dock_single_and_multi_selection(
    qapp: QApplication, tmp_path: Path
) -> None:
    """
    Verify TagAndCaptionDock adapts to single and multi-selection, shows shared tags,
    and emits rating, flag, caption, tag, person, and Change Time Taken signals.
    """
    engine = DualDatabaseEngine(tmp_path / "catalog.db", tmp_path / "thumbs.db")
    repo = CatalogRepository(engine)

    fid = repo.upsert_folder("D:/Photos", "Photos", None, 1700000000)
    ids = repo.bulk_upsert_media(
        [
            {
                "folder_id": fid,
                "filename": "IMG_01.jpg",
                "ext": ".jpg",
                "size": 3_500_000,
                "mtime": 1700000000,
                "date_taken": 1700000000,
                "rating": 4,
                "flag": 1,
                "width": 4000,
                "height": 3000,
                "orientation": 1,
                "camera_make": "Canon",
                "camera_model": "EOS R5",
                "exposure_time": "1/250",
                "f_number": 2.8,
                "focal_length": 50.0,
                "iso": 400,
                "caption": "First photo",
            },
            {
                "folder_id": fid,
                "filename": "IMG_02.jpg",
                "ext": ".jpg",
                "size": 2_500_000,
                "mtime": 1700000060,
                "date_taken": 1700000060,
                "rating": 4,
                "flag": 0,
                "width": 4000,
                "height": 3000,
                "orientation": 1,
                "caption": "Second photo",
            },
        ]
    )

    t_shared = repo.create_tag("Events/Holiday")
    t_only1 = repo.create_tag("Style/Portrait")
    repo.assign_tag(ids, t_shared)
    repo.assign_tag([ids[0]], t_only1)
    repo.assign_person(ids[0], "Craig", (0.2, 0.2, 0.3, 0.3))

    dock = TagAndCaptionDock()
    dock.set_autocomplete_lists(["Events/Holiday", "Style/Portrait"], ["Craig", "Sarah"])

    # 1. Single-selection binding
    summary_single = repo.get_selection_summary([ids[0]])
    dock.bind_selection([ids[0]], summary_single)
    assert dock.lbl_selection_header.text() == "1 item selected"
    assert dock.lbl_filename.text() == "IMG_01.jpg"
    assert "Canon EOS R5" in dock.lbl_camera.text()
    assert dock.tags_chips_layout.count() == 2
    assert dock.people_chips_layout.count() == 1

    # 2. Multi-selection binding -> all tags across selected items appear with counts
    summary_multi = repo.get_selection_summary(ids)
    dock.bind_selection(ids, summary_multi)
    assert dock.lbl_selection_header.text() == "2 items selected"
    assert dock.lbl_filename.text() == "Multiple files (2)"
    assert dock.tags_chips_layout.count() == 2

    # 3. Interactive signals from Right Dock applied to multi-selection
    ratings_emitted: list[int] = []
    flags_emitted: list[int] = []
    captions_emitted: list[str] = []
    tags_added: list[str] = []
    people_added: list[str] = []
    time_requests: list[int] = []

    dock.rating_edited.connect(ratings_emitted.append)
    dock.flag_edited.connect(flags_emitted.append)
    dock.caption_committed.connect(captions_emitted.append)
    dock.tag_added.connect(tags_added.append)
    dock.person_added.connect(people_added.append)
    dock.change_time_requested.connect(lambda: time_requests.append(1))

    dock.star_buttons[4].click()  # Click 5th star (rate 5)
    assert ratings_emitted == [5]
    repo.set_rating(ids, 5)

    dock.btn_flag.click()
    assert flags_emitted == [1]

    dock.commit_caption("Summer Trip 2024")
    assert captions_emitted == ["Summer Trip 2024"]
    repo.set_caption(ids, "Summer Trip 2024")

    dock.add_descriptive_tag_text("Nature/Sunset")
    assert tags_added == ["Nature/Sunset"]

    dock.add_person_tag_text("Sarah")
    assert people_added == ["Sarah"]

    dock.btn_date_taken.click()
    assert time_requests == [1]

    # Verify Splitter & Empty State (media_1792167382404_9d7d42ec.png)
    assert dock.info_splitter is not None
    dock.set_total_view_count(1722)
    dock.bind_selection([], {})
    assert not dock.lbl_view_items_count.isHidden()
    assert "1722 items in this view" in dock.lbl_view_items_count.text()
    assert dock.info_form_frame.isHidden()

    # When 1 item selected, form is shown and empty state count is hidden
    dock.bind_selection([ids[0]], repo.get_selection_summary([ids[0]]))
    assert dock.lbl_view_items_count.isHidden()
    assert not dock.info_form_frame.isHidden()

    engine.close_all()


def test_change_time_taken_dialog_relative_delta_math(
    qapp: QApplication, tmp_path: Path
) -> None:
    """
    Verify ChangeTimeTakenDialog computes Δt = T_lead_new - T_lead_original (including pre-1970 dates)
    and shifts all selected media items by the exact relative offset.
    """
    engine = DualDatabaseEngine(tmp_path / "catalog.db", tmp_path / "thumbs.db")
    repo = CatalogRepository(engine)
    fid = repo.upsert_folder("D:/Archive", "Archive", None, 1700000000)

    # 3 photos taken 15 minutes (900s) apart on 14 October 1957: 10:00, 10:15, 10:30
    base_1957 = int(
        (
            datetime.datetime(1957, 10, 14, 10, 0, 0, tzinfo=datetime.timezone.utc)
            - _EPOCH_UTC
        ).total_seconds()
    )
    ids = repo.bulk_upsert_media(
        [
            {
                "folder_id": fid,
                "filename": "seq_01.jpg",
                "ext": ".jpg",
                "size": 1_000_000,
                "mtime": 1700000000,
                "date_taken": base_1957,
            },
            {
                "folder_id": fid,
                "filename": "seq_02.jpg",
                "ext": ".jpg",
                "size": 1_000_000,
                "mtime": 1700000000,
                "date_taken": base_1957 + 900,
            },
            {
                "folder_id": fid,
                "filename": "seq_03.jpg",
                "ext": ".jpg",
                "size": 1_000_000,
                "mtime": 1700000000,
                "date_taken": base_1957 + 1800,
            },
        ]
    )

    dlg = ChangeTimeTakenDialog(
        lead_filename="seq_01.jpg",
        lead_date_taken=base_1957,
        selected_count=3,
    )
    assert dlg.delta_seconds() == 0
    assert "No time adjustment has been made" in dlg.lbl_status.text()

    # Shift lead photo forward by +1 hour 30 minutes (+5400s) -> 11:30:00 on 14 Oct 1957
    dlg.set_new_datetime(1957, 10, 14, 11, 30, 0)
    assert dlg.delta_seconds() == 5400
    assert "+01:30:00 applied to 3 files" in dlg.lbl_status.text()

    applied_delta = repo.apply_relative_time_shift(ids, ids[0], dlg.new_lead_timestamp())
    assert applied_delta == 5400

    # Verify each item preserved its 15-minute relative spacing and is marked metadata_dirty = 1
    d1 = repo.get_media_detail(ids[0])
    d2 = repo.get_media_detail(ids[1])
    d3 = repo.get_media_detail(ids[2])

    assert _epoch_to_datetime_utc(d1["date_taken"]).strftime("%Y-%m-%d %H:%M:%S") == "1957-10-14 11:30:00"
    assert _epoch_to_datetime_utc(d2["date_taken"]).strftime("%Y-%m-%d %H:%M:%S") == "1957-10-14 11:45:00"
    assert _epoch_to_datetime_utc(d3["date_taken"]).strftime("%Y-%m-%d %H:%M:%S") == "1957-10-14 12:00:00"
    assert d1["metadata_dirty"] == 1
    assert d2["metadata_dirty"] == 1
    assert d3["metadata_dirty"] == 1

    engine.close_all()


def test_app_settings_dialog_options(qapp: QApplication) -> None:
    """Verify AppSettingsDialog retention days, theme selection, and maintenance flags."""
    dlg = AppSettingsDialog(retention_days=30, current_theme="wlpg_aero")
    assert dlg.retention_days() == 30
    assert dlg.selected_theme() == "wlpg_aero"

    dlg.spin_retention.setValue(45)
    dlg.cmb_theme.setCurrentIndex(2)  # Dark Studio
    dlg.btn_compact_db.click()
    dlg.btn_prune_backups.click()

    assert dlg.compact_requested is True
    assert dlg.prune_backups_requested is True


def test_resize_photo_dialog_scaling_and_collision(
    qapp: QApplication, tmp_path: Path
) -> None:
    """Verify ResizePhotoDialog presets, Custom dimensions, Lanczos scaling, and (2) collision avoidance."""
    from PIL import Image
    from src.ui.dialogs import ResizePhotoDialog

    # 1. Create a dummy test image 2000x1000
    src_dir = tmp_path / "photos"
    src_dir.mkdir()
    img_path = src_dir / "landscape.jpg"
    img = Image.new("RGB", (2000, 1000), color=(100, 150, 200))
    img.save(img_path)

    dlg = ResizePhotoDialog([img_path])
    assert dlg.windowTitle() == "Photo Gallery"
    assert dlg.cmb_size.currentText() == "Medium: 1024"
    assert dlg.txt_dim.text() == "1024"

    # 2. Selecting "Smaller: 640" updates dimension text
    dlg.cmb_size.setCurrentIndex(0)
    assert dlg.txt_dim.text() == "640"

    # 3. Typing custom dimension switches combo to "Custom"
    dlg.txt_dim.setText("800")
    assert dlg.cmb_size.currentText() == "Custom"

    # 4. Target output directory
    out_dir = tmp_path / "resized"
    dlg.txt_save_path.setText(str(out_dir))

    # 5. Execute resize
    generated = dlg.execute_resize()
    assert len(generated) == 1
    assert generated[0].exists()
    assert generated[0].name == "landscape.jpg"

    with Image.open(generated[0]) as res_img:
        w, h = res_img.size
        assert max(w, h) == 800
        assert w == 800
        assert h == 400

    # 6. Re-executing resize with same target directory triggers WLPG collision avoidance: "landscape (2).jpg"
    gen2 = dlg.execute_resize()
    assert len(gen2) == 1
    assert gen2[0].name == "landscape (2).jpg"
    assert gen2[0].exists()

