"""
Unit & Integration tests for People search scope chips, Home ribbon Quick find (Months/Rated),
large button menu arrow positioning, and 10-face carousel expansion.
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel, QToolButton

from src.ui.search_scope_bar import SearchScopeBar, FilterChip
from src.ui.ribbon_toolbar import RibbonToolbar, _FaceCarouselWidget


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_search_scope_bar_special_people_chip(qapp) -> None:
    """Verifies that special people search labels appear as chips in the Search by box."""
    bar = SearchScopeBar()
    bar.show()
    bar.set_scope_label("Folder: Vacations")
    assert not bar.lbl_scope_label.isHidden()
    assert bar.lbl_scope_value.text() == "Vacations"
    assert not bar.btn_search_all.isHidden()

    cleared = []
    bar.special_people_clear_requested.connect(lambda: cleared.append(True))

    # Update with special people filter "All tagged or detected"
    class DummyRepo:
        pass

    bar.update_active_filter_chips(
        filter_state={},
        catalog_repo=DummyRepo(),
        special_people_label="All tagged or detected",
    )

    # Verify People label and FilterChip are present
    people_labels = [w for w in bar.chips_widget.findChildren(QLabel) if w.text() == "People:"]
    assert len(people_labels) == 1

    chips = bar.chips_widget.findChildren(FilterChip)
    assert len(chips) == 1
    assert chips[0].text() == "All tagged or detected"

    # Simulate remove click on chip
    chips[0].btn_close.click()
    assert len(cleared) == 1

    # Verify "All detected but not tagged"
    bar.update_active_filter_chips(
        filter_state={},
        catalog_repo=DummyRepo(),
        special_people_label="All detected but not tagged",
    )
    qapp.processEvents()
    chips2 = [c for c in bar.chips_widget.findChildren(FilterChip) if c.isVisible()]
    assert len(chips2) == 1
    assert chips2[0].text() == "All detected but not tagged"


def test_home_ribbon_months_and_rated_instant_popup(qapp) -> None:
    """Verifies Months and Rated in Home ribbon have InstantPopup and emit filter signals."""
    ribbon = RibbonToolbar()
    ribbon.show()

    assert ribbon.btn_qf_months.popupMode() == QToolButton.InstantPopup
    assert ribbon.btn_qf_rated.popupMode() == QToolButton.InstantPopup
    assert ribbon.btn_qf_months.menu() is not None
    assert ribbon.btn_qf_rated.menu() is not None

    month_emits = []
    date_emits = []
    rated_emits = []

    ribbon.filter_month_selected.connect(lambda m: month_emits.append(m))
    ribbon.date_filter_selected.connect(lambda y, m, d: date_emits.append((y, m, d)))
    ribbon.rating_filter_selected.connect(lambda r: rated_emits.append(r))

    # Trigger a month action (e.g. October = 10)
    month_actions = [a for a in ribbon.btn_qf_months.menu().actions() if a.text() == "October"]
    assert len(month_actions) == 1
    month_actions[0].trigger()

    assert 10 in month_emits
    assert (None, 10, None) in date_emits

    # Trigger a rated action (e.g. 4 stars or higher)
    rated_actions = [a for a in ribbon.btn_qf_rated.menu().actions() if a.text().endswith("or higher") and a.text().count("\u2605") == 4]
    assert len(rated_actions) == 1
    rated_actions[0].trigger()

    assert (4, 5) in rated_emits

    ribbon.close()


def test_large_ribbon_button_arrow_below_icon(qapp) -> None:
    """Verifies that large buttons in the ribbon have their dropdown arrow positioned at bottom center."""
    ribbon = RibbonToolbar()
    qss = ribbon.styleSheet()

    # The QSS for LargeRibbonToolBtn and LargeRibbonBtn should specify bottom center
    assert "QToolButton#LargeRibbonToolBtn::menu-indicator" in qss
    assert "subcontrol-position: bottom center" in qss
    ribbon.close()


def test_face_carousel_expands_to_10_faces(qapp) -> None:
    """Verifies that the face carousel expands up to 10 faces wide when room allows."""
    ribbon = RibbonToolbar()
    carousel = ribbon.home_face_carousel

    # Populate 15 dummy people
    dummy_people = [
        {"id": i, "name": f"Person {i}", "media_count": 10 - i, "thumbnail_blob": None}
        for i in range(15)
    ]
    carousel.populate(dummy_people)

    # Test set_visible_faces up to 10
    carousel.set_visible_faces(10)
    assert carousel._visible_faces == 10
    assert carousel.scroll.width() == 10 * 56
    assert carousel.width() == 10 * 56 + 19

    # Test responsive layout at 1600px width -> expands to 10 faces
    ribbon._apply_responsive_layout(1600)
    assert ribbon.home_face_carousel._visible_faces == 10
    assert ribbon.find_face_carousel._visible_faces == 10

    # Test responsive collapse at narrow widths
    ribbon._apply_responsive_layout(950)
    assert ribbon.home_face_carousel._visible_faces <= 3

    ribbon.close()
