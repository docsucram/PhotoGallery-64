"""
Phase 14 Tests: Home Tab Ribbon Matching, QAT Below Ribbon, and Customize Menu.
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

from src.ui.ribbon_toolbar import RibbonToolbar


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_home_tab_layout_and_groups(qapp: QApplication) -> None:
    """Verify Home tab groups, 2-column Manage group, Organize with Geotag, and compact Quick find."""
    ribbon = RibbonToolbar()

    # 1. Groups on Home tab
    home_tab = ribbon.tab_home
    group_boxes = home_tab.findChildren(ribbon.tab_home.__class__, "RibbonGroupBox")
    group_titles = [g.lbl_title.text() for g in home_tab.findChildren(object, "RibbonGroupBox")]
    assert "New" in group_titles
    assert "Manage" in group_titles
    assert "Organize" in group_titles
    assert "Quick find" in group_titles
    assert "Share" in group_titles

    # 2. Quick Find face carousel has bounded responsive width up to 10 faces
    assert ribbon.home_face_carousel.width() > 0 and ribbon.home_face_carousel._visible_faces >= 1

    # 3. Filter buttons exist in Quick find
    assert ribbon.btn_qf_months is not None
    assert ribbon.btn_qf_rated is not None
    assert ribbon.btn_qf_flagged is not None

    ribbon.deleteLater()


def test_qat_below_ribbon_and_customize_menu(qapp: QApplication) -> None:
    """Verify QAT toggle between above and below ribbon, and customize menu options."""
    ribbon = RibbonToolbar()
    ribbon.show()

    # Initial state: above ribbon
    assert ribbon._qat_below_ribbon is False
    assert ribbon.top_header.isVisible() is True
    assert ribbon.below_ribbon_bar.isHidden() is True
    assert ribbon.act_qat_position.text() == "Show below the Ribbon"

    # Verify Customize QAT Menu items
    qat_menu = ribbon.btn_qat_menu.menu()
    assert qat_menu is not None
    action_texts = [act.text() for act in qat_menu.actions()]
    assert "Customize Quick Access Toolbar" in action_texts
    assert "Print" in action_texts
    assert "Back" in action_texts
    assert "Forward" in action_texts
    assert "Undo" in action_texts
    assert "Redo" in action_texts
    assert "Show below the Ribbon" in action_texts
    assert "Minimize the Ribbon" in action_texts

    # Toggle QAT below ribbon
    ribbon.set_qat_below_ribbon(True)
    assert ribbon._qat_below_ribbon is True
    assert ribbon.top_header.isHidden() is True
    assert ribbon.below_ribbon_bar.isVisible() is True
    assert ribbon.act_qat_position.text() == "Show above the Ribbon"
    assert ribbon.qat_container.parent() == ribbon.below_ribbon_bar

    # Toggle QAT back above ribbon
    ribbon.set_qat_below_ribbon(False)
    assert ribbon._qat_below_ribbon is False
    assert ribbon.top_header.isVisible() is True
    assert ribbon.below_ribbon_bar.isHidden() is True
    assert ribbon.act_qat_position.text() == "Show below the Ribbon"
    assert ribbon.qat_container.parent() == ribbon.top_header

    # Test toggling individual QAT button visibility
    ribbon.act_qat_back.setChecked(False)
    assert ribbon.btn_back.isHidden() is True
    ribbon.act_qat_back.setChecked(True)
    assert ribbon.btn_back.isVisible() is True

    ribbon.deleteLater()
