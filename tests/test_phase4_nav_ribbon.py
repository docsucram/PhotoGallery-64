"""
Phase 4 Automated Verification Suite:
1. LeftNavigationDock unified tree population (Folders, Date Taken including 1957, Hierarchical Tags, People, Ratings).
2. Boolean AND/OR/NOT tag filter state -> FilterQuerySpec -> CatalogRepository query integration.
3. RibbonToolbar Arrange List, Show Details overlays, Quick Find Face Carousel, Layout Mode switcher, and Search Scope bar.
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtWidgets import QApplication

from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository
from src.ui.left_nav_dock import LeftNavigationDock
from src.ui.ribbon_toolbar import RibbonToolbar


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_left_nav_dock_trees_and_boolean_tag_filters(
    qapp: QApplication, tmp_path: Path
) -> None:
    """
    Verify LeftNavigationDock populates all 5 filter hierarchies from CatalogRepository
    and generates exact FilterQuerySpec objects for Folder, Date, AND/OR/NOT Tag, Person, and Rating filters.
    """
    engine = DualDatabaseEngine(tmp_path / "catalog.db", tmp_path / "thumbs.db")
    repo = CatalogRepository(engine)

    f_root = repo.upsert_folder("D:/Photos", "Photos", None, 1700000000)
    f_sub = repo.upsert_folder("D:/Photos/1957", "1957", f_root, 1700000000)

    import datetime
    epoch_1957 = int(
        (
            datetime.datetime(1957, 10, 14, 14, 43, 0, tzinfo=datetime.timezone.utc)
            - datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
        ).total_seconds()
    )
    media_ids = repo.bulk_upsert_media(
        [
            {
                "folder_id": f_sub,
                "filename": "scan410a.jpg",
                "ext": ".jpg",
                "size": 1_200_000,
                "mtime": 1700000000,
                "date_taken": epoch_1957,
                "rating": 5,
                "flag": 1,
            },
            {
                "folder_id": f_root,
                "filename": "family_pic.jpg",
                "ext": ".jpg",
                "size": 2_400_000,
                "mtime": 1700000000,
                "date_taken": 1728993600,
                "rating": 4,
                "flag": 0,
            },
            {
                "folder_id": f_root,
                "filename": "draft_outtake.jpg",
                "ext": ".jpg",
                "size": 1_800_000,
                "mtime": 1700000000,
                "date_taken": 1728993600,
                "rating": 0,
                "flag": 0,
            },
        ]
    )

    tag_craig = repo.create_tag("PeopleTags/Family/Craig")
    tag_family = repo.create_tag("PeopleTags/Family")
    tag_draft = repo.create_tag("Status/Draft")

    repo.assign_tag([media_ids[0], media_ids[1]], tag_craig)
    repo.assign_tag([media_ids[1]], tag_Draft := tag_draft)

    pid_craig = repo.assign_person(media_ids[0], "Craig", (0.1, 0.1, 0.2, 0.2))

    nav = LeftNavigationDock()
    nav.refresh_trees(repo)

    # Verify tree sections populated
    assert nav.root_folders.childCount() == 1
    assert nav.root_dates.childCount() == 2  # 2024 and 1957
    assert nav.root_people.childCount() == 1
    assert nav.root_ratings.childCount() == 6

    filter_signals: list[int] = []
    nav.filter_changed.connect(lambda: filter_signals.append(1))

    # 1. Include parent tag "PeopleTags/Family" AND Exclude "Status/Draft" (NOT)
    nav.set_tag_filter_state(tag_family, "include")
    nav.set_tag_filter_state(tag_Draft, "exclude")

    spec = nav.build_filter_spec()
    assert spec.include_tag_ids == (tag_family,)
    assert spec.exclude_tag_ids == (tag_Draft,)
    assert spec.tag_boolean_mode == "AND"

    matched = repo.query_media_records(spec)
    assert [m.filename for m in matched] == ["scan410a.jpg"]

    # 2. Test Untagged only filter
    nav.clear_all_filters()
    nav.set_untagged_only(True)
    matched_untagged = repo.query_media_records(nav.build_filter_spec())
    assert [m.filename for m in matched_untagged] == ["draft_outtake.jpg"]

    # 3. Test 1957 Date filter
    nav.clear_all_filters()
    nav.select_date(1957, 10, 14)
    matched_1957 = repo.query_media_records(nav.build_filter_spec())
    assert [m.filename for m in matched_1957] == ["scan410a.jpg"]

    # 4. Test Person & Rating filter
    nav.clear_all_filters()
    nav.select_person(pid_craig)
    nav.select_ratings([5])
    spec_pr = nav.build_filter_spec()
    assert spec_pr.person_ids == (pid_craig,)
    assert spec_pr.ratings == (5,)
    assert [m.filename for m in repo.query_media_records(spec_pr)] == ["scan410a.jpg"]
    assert len(filter_signals) >= 5

    engine.close_all()


def test_ribbon_toolbar_arrange_details_and_quick_find(qapp: QApplication) -> None:
    """
    Verify RibbonToolbar signals and state transitions:
    - Quick Find Face Carousel avatar click -> person_quick_find_selected
    - View tab Arrange List + Reverse sort -> arrange_changed
    - View tab Show Details checkboxes -> details_overlay_changed
    - Layout mode toggle ('wlpg_grid' <-> 'mosaic') -> layout_mode_changed
    - Search text input -> search_text_changed
    - Viewer mode -> switches to Edit tab and hides scope bar
    """
    ribbon = RibbonToolbar()

    # 1. Quick Find Face Carousel
    people_data = [
        {"id": 10, "name": "Craig", "media_count": 42, "thumbnail_blob": None},
        {"id": 20, "name": "Sarah", "media_count": 19, "thumbnail_blob": None},
    ]
    ribbon.populate_face_carousel(people_data)

    selected_people: list[int] = []
    ribbon.person_quick_find_selected.connect(selected_people.append)
    ribbon.home_face_carousel.click_person(10)
    assert selected_people == [10]

    # 2. Arrange List & Reverse Sort
    arrange_events: list[tuple[str, bool]] = []
    ribbon.arrange_changed.connect(lambda m, rev: arrange_events.append((m, rev)))
    ribbon.set_arrange_by("rating", reverse_sort=True)
    assert arrange_events[-1] == ("rating", True)
    assert ribbon.current_arrange_state() == ("rating", True)

    # 3. Show Details checkboxes
    detail_events: list[set[str]] = []
    ribbon.details_overlay_changed.connect(detail_events.append)
    ribbon.set_detail_flag("filename", True)
    ribbon.set_detail_flag("date_taken", True)
    assert "filename" in ribbon.current_visible_details()
    assert "date_taken" in ribbon.current_visible_details()
    assert "filename" in detail_events[-1]

    # 4. Layout Mode Switcher (WLPG Grid vs Mosaic)
    layout_events: list[str] = []
    ribbon.layout_mode_changed.connect(layout_events.append)
    ribbon.btn_layout_mosaic.click()
    assert layout_events == ["mosaic"]

    # 5. Inline Search Box & Scope Label
    ribbon.set_scope_label("Folder: Pictures (Marcus)")
    assert ribbon.lbl_scope.text() == "Folder: Pictures (Marcus)"

    search_events: list[str] = []
    ribbon.search_text_changed.connect(search_events.append)
    ribbon.set_search_text("scan410a", immediate=True)
    assert search_events == ["scan410a"]

    # 6. Active Mode transitions (Grid <-> Single Viewer Edit tab)
    ribbon.set_active_mode("viewer")
    assert ribbon.tab_widget.currentWidget() == ribbon.tab_edit
    assert not ribbon.scope_bar.isVisibleTo(ribbon)

    ribbon.set_active_mode("grid")
    assert ribbon.tab_widget.currentWidget() == ribbon.tab_home
    assert ribbon.scope_bar.isVisibleTo(ribbon)


def test_ribbon_find_tab_filters_and_scope_bar(qapp: QApplication) -> None:
    """
    Verify complete Find tab controls, popup menus, and scope bar toggle:
    - Date taken: Dates, Months, Years dropdowns emit date_filter_selected
    - People: Carousel + and/or toggle buttons emit person_boolean_mode_changed
    - Rated: 5-to-1 stars + and higher / and lower / only modes emit rating_filter_selected
    - Tags: tags dropdown menu with top tags, all tags, not tagged emit tag signals
    - File details: Flagged, Edited, Media type emit respective filter signals
    - Text search: binoculars button emits text_search_toggle_requested
    - Scope bar: close button hides scope bar and clears text
    """
    ribbon = RibbonToolbar()
    ribbon.show()

    # 1. Date taken dropdown menus
    date_events: list[tuple[Any, Any, Any]] = []
    ribbon.date_filter_selected.connect(lambda y, m, d: date_events.append((y, m, d)))

    # Trigger Today action from Dates menu
    dates_menu = ribbon.btn_find_dates.menu()
    assert dates_menu is not None
    today_act = [a for a in dates_menu.actions() if a.text() == "Today"][0]
    today_act.trigger()
    assert len(date_events) == 1
    now = datetime.datetime.now()
    assert date_events[-1] == (now.year, now.month, now.day)

    # Trigger March from Months menu
    months_menu = ribbon.btn_find_months.menu()
    assert months_menu is not None
    march_act = [a for a in months_menu.actions() if a.text() == "March"][0]
    march_act.trigger()
    assert date_events[-1] == (None, 3, None)

    # Trigger 2024 from Years menu
    years_menu = ribbon.btn_find_years.menu()
    assert years_menu is not None
    yr2024_act = [a for a in years_menu.actions() if a.text() == "2024"][0]
    yr2024_act.trigger()
    assert date_events[-1] == (2024, None, None)

    # 2. People Group (+ and / / or)
    people_bool_events: list[str] = []
    ribbon.person_boolean_mode_changed.connect(people_bool_events.append)
    assert not ribbon.is_people_and_mode()
    ribbon.btn_people_and.click()
    assert ribbon.is_people_and_mode()
    assert people_bool_events[-1] == "AND"
    ribbon.btn_people_or.click()
    assert not ribbon.is_people_and_mode()
    assert people_bool_events[-1] == "OR"

    # 3. Rated Group (Stars + and higher / and lower / only)
    rating_events: list[tuple[int, ...]] = []
    ribbon.rating_filter_selected.connect(lambda r: rating_events.append(tuple(r)))

    rated_menu = ribbon.btn_find_rated.menu()
    assert rated_menu is not None
    star3_act = [a for a in rated_menu.actions() if "3 stars" in a.text()][0]

    # Mode 1: ↑ and higher (default) -> 3, 4, 5
    assert ribbon.btn_rated_higher.isChecked()
    star3_act.trigger()
    assert rating_events[-1] == (3, 4, 5)

    # Mode 2: ↓ and lower -> 1, 2, 3
    ribbon.btn_rated_lower.click()
    assert rating_events[-1] == (1, 2, 3)

    # Mode 3: = only -> 3
    ribbon.btn_rated_only.click()
    assert rating_events[-1] == (3,)

    # Unrated -> 0
    unrated_act = [a for a in rated_menu.actions() if "Unrated" in a.text()][0]
    unrated_act.trigger()
    assert rating_events[-1] == (0,)

    # Clear -> ()
    clear_act = [a for a in rated_menu.actions() if "Clear" in a.text()][0]
    clear_act.trigger()
    assert rating_events[-1] == ()

    # 4. Tags Group
    tag_events: list[tuple[Any, str]] = []
    ribbon.tag_filter_selected.connect(lambda tid, mode: tag_events.append((tid, mode)))
    untagged_events: list[bool] = []
    ribbon.untagged_filter_selected.connect(untagged_events.append)

    tag_tree_mock = [
        {"id": 101, "name": "Beach", "full_path": "Vacation/Beach", "media_count": 50},
        {"id": 102, "name": "Mountains", "full_path": "Vacation/Mountains", "media_count": 10},
    ]
    ribbon.populate_tags_menu(tag_tree_mock)
    tag_menu_acts = ribbon.menu_find_tags.actions()
    beach_act = [a for a in tag_menu_acts if a.text() == "Beach"][0]
    beach_act.trigger()
    assert tag_events[-1] == (101, "include")

    not_tagged_act = [a for a in tag_menu_acts if "Not tagged" in a.text()][0]
    not_tagged_act.trigger()
    assert untagged_events[-1] is True

    # 5. File details Group
    flag_events: list[bool] = []
    ribbon.flagged_filter_selected.connect(flag_events.append)
    ribbon.btn_find_flagged.click()
    assert flag_events[-1] is True

    edit_events: list[bool] = []
    ribbon.edited_filter_selected.connect(edit_events.append)
    ribbon.btn_find_edited.click()
    assert edit_events[-1] is True

    media_type_events: list[str] = []
    ribbon.media_type_filter_selected.connect(media_type_events.append)
    mt_menu = ribbon.btn_find_media_type.menu()
    assert mt_menu is not None
    photos_act = [a for a in mt_menu.actions() if "Photos only" in a.text()][0]
    photos_act.trigger()
    assert media_type_events[-1] == "image"

    # 6. Text search binoculars button
    text_search_triggered: list[bool] = []
    ribbon.text_search_toggle_requested.connect(lambda: text_search_triggered.append(True))
    ribbon.btn_find_text_search.click()
    assert len(text_search_triggered) == 1

    # 7. Scope Bar toggle and close button
    ribbon.show_search_scope()
    assert ribbon.scope_bar.isVisible()
    ribbon.set_search_text("sunset")
    assert ribbon.txt_search.text() == "sunset"

    # Click ✕ close button
    ribbon.btn_close_scope.click()
    assert not ribbon.scope_bar.isVisible()
    assert ribbon.txt_search.text() == ""


def test_ribbon_view_tab_grid_and_single_mode(qapp: QApplication) -> None:
    """
    Verify View tab structure, 4x2 arrange matrix, details checkboxes, and adaptive single view mode:
    - 4x2 matrix buttons (Date, Geotag, Person, Name, Tag, Published, Folder, Media type) + Rating + Reverse sort
    - Show details checkboxes & large buttons
    - Single view mode hides Home & Find tabs, hides Arrange/Details/Layout, shows 1:1 and Fit to window
    - Grid view mode restores Home & Find tabs and restores Arrange/Details/Layout
    """
    ribbon = RibbonToolbar()
    ribbon.show()

    # 1. Verify 4x2 Arrange matrix buttons exist and function
    arrange_events: list[tuple[str, bool]] = []
    ribbon.arrange_changed.connect(lambda m, rev: arrange_events.append((m, rev)))

    assert "date" in ribbon._arrange_buttons
    assert "geotag" in ribbon._arrange_buttons
    assert "person" in ribbon._arrange_buttons
    assert "name" in ribbon._arrange_buttons
    assert "tag" in ribbon._arrange_buttons
    assert "published" in ribbon._arrange_buttons
    assert "folder" in ribbon._arrange_buttons
    assert "media_type" in ribbon._arrange_buttons
    assert "rating" in ribbon._arrange_buttons

    ribbon._arrange_buttons["name"].click()
    assert arrange_events[-1] == ("name", False)

    ribbon.btn_reverse_sort.click()
    assert arrange_events[-1] == ("name", True)

    # 2. Verify Show details checkboxes and large buttons
    detail_events: list[set[str]] = []
    ribbon.details_overlay_changed.connect(detail_events.append)

    assert "rating" in ribbon._detail_checkboxes
    assert "caption" in ribbon._detail_checkboxes
    assert "flag" in ribbon._detail_checkboxes
    assert "filename" in ribbon._detail_checkboxes
    assert "size" in ribbon._detail_checkboxes
    assert "date_taken" in ribbon._detail_checkboxes
    assert "dimensions" in ribbon._detail_checkboxes
    assert "date_modified" in ribbon._detail_checkboxes

    # All details checkbox checks all 5 sub-checkboxes
    ribbon.chk_all_details.setChecked(True)
    assert ribbon.chk_filename.isChecked()
    assert ribbon.chk_filesize.isChecked()
    assert ribbon.chk_date_taken.isChecked()
    assert ribbon.chk_dimensions.isChecked()
    assert ribbon.chk_date_modified.isChecked()

    # 3. Enter Single Viewer mode
    ribbon.set_active_mode("viewer")
    # Home (0) and Find (2) tabs are hidden
    assert not ribbon.tab_widget.tabBar().isTabVisible(0)
    assert ribbon.tab_widget.tabBar().isTabVisible(1)  # Edit
    assert not ribbon.tab_widget.tabBar().isTabVisible(2)
    assert ribbon.tab_widget.tabBar().isTabVisible(3)  # Create
    assert ribbon.tab_widget.tabBar().isTabVisible(4)  # View

    # In View tab, arrange, details, layout are hidden
    ribbon.tab_widget.setCurrentWidget(ribbon.tab_view)
    assert not ribbon.grp_arrange.isVisible()
    assert not ribbon.grp_details.isVisible()
    assert not ribbon.grp_layout.isVisible()
    assert not ribbon.btn_toggle_nav_pane.isVisible()
    # Zoom shows Actual size and Fit to window
    assert ribbon.btn_zoom_actual.isVisible()
    assert ribbon.btn_zoom_fit.isVisible()

    # 4. Return to Grid mode
    ribbon.set_active_mode("grid")
    assert ribbon.tab_widget.tabBar().isTabVisible(0)  # Home restored
    assert ribbon.tab_widget.tabBar().isTabVisible(2)  # Find restored
    ribbon.tab_widget.setCurrentWidget(ribbon.tab_view)
    assert ribbon.grp_arrange.isVisible()
    assert ribbon.grp_details.isVisible()
    assert ribbon.grp_layout.isVisible()
    assert ribbon.btn_toggle_tag_pane.isVisible()
    assert not ribbon.btn_zoom_actual.isVisible()
    assert not ribbon.btn_zoom_fit.isVisible()

