import pytest
import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QPointF, QEvent
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QToolButton, QGraphicsSceneMouseEvent

from src.ui.crop_tool import CropScene
from src.ui.media_viewer import SingleMediaViewer, _FaceHoverRegionItem
from src.ui.right_tag_dock import TagAndCaptionDock, _NoHorizontalScrollArea
from src.ui.ribbon_toolbar import RibbonToolbar


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_crop_scene_disabled_by_default(qapp):
    scene = CropScene()
    scene.setSceneRect(0, 0, 400, 300)
    assert not scene.crop_enabled
    assert not scene.crop_rect_item.isVisible()

    # Simulate mouse press outside when crop is not enabled
    event = QGraphicsSceneMouseEvent(QEvent.GraphicsSceneMousePress)
    event.setButton(Qt.LeftButton)
    event.setScenePos(QPointF(50, 50))
    scene.mousePressEvent(event)
    assert not scene.crop_rect_item.isVisible()
    assert scene.get_crop_rect() is None

    # Now enable crop
    scene.set_crop_mode(True)
    assert scene.crop_enabled
    assert scene.crop_rect_item.isVisible()

    # Disable crop again
    scene.set_crop_mode(False)
    assert not scene.crop_enabled
    assert not scene.crop_rect_item.isVisible()


def test_face_boxes_hidden_in_edit_modes(qapp):
    viewer = SingleMediaViewer()
    assert not viewer.is_in_edit_mode()

    dummy_faces = [
        {"person_id": 1, "name": "Alice", "rect_x": 0.2, "rect_y": 0.2, "rect_w": 0.2, "rect_h": 0.2, "is_unconfirmed": False}
    ]
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    viewer.master_np_rgb = img.copy()
    viewer.master_pil_image = Image.fromarray(img)
    viewer.current_pil_image = viewer.master_pil_image.copy()
    viewer._update_display_proxy()
    viewer.render_face_regions(dummy_faces)

    assert len(viewer._face_overlay_items) > 0
    # Before edit mode, hovering over the sensor reveals related items
    sensor = next(it for it in viewer._face_overlay_items if isinstance(it, _FaceHoverRegionItem))
    sensor.hoverEnterEvent(None)
    for it in sensor.related_items:
        assert it.isVisible()

    # Enter Crop Mode
    viewer.set_crop_mode(True)
    assert viewer.is_in_edit_mode()
    for it in sensor.related_items:
        assert not it.isVisible()

    # While in edit mode, hover must not reveal face box
    sensor.hoverEnterEvent(None)
    for it in sensor.related_items:
        assert not it.isVisible()

    viewer.set_crop_mode(False)
    assert not viewer.is_in_edit_mode()

    # Enter Retouch Mode
    viewer.set_retouch_mode(True)
    assert viewer.is_in_edit_mode()
    sensor.hoverEnterEvent(None)
    for it in sensor.related_items:
        assert not it.isVisible()
    viewer.set_retouch_mode(False)

    # Enter Red Eye Mode
    viewer.set_red_eye_mode(True)
    assert viewer.is_in_edit_mode()
    sensor.hoverEnterEvent(None)
    for it in sensor.related_items:
        assert not it.isVisible()
    viewer.set_red_eye_mode(False)


def test_red_eye_algorithm_safe_on_skin_and_fixes_pupil(qapp):
    viewer = SingleMediaViewer()
    # Create image: 100x100 with human skin tone (R=180, G=130, B=110)
    # and a red pupil in the center (radius 6, R=220, G=45, B=40)
    img = np.full((100, 100, 3), (180, 130, 110), dtype=np.uint8)
    for y in range(43, 58):
        for x in range(43, 58):
            if (y - 50) ** 2 + (x - 50) ** 2 <= 6**2:
                img[y, x] = [220, 45, 40]

    viewer.master_np_rgb = img.copy()
    viewer.master_pil_image = Image.fromarray(img)
    viewer.current_pil_image = viewer.master_pil_image.copy()
    viewer._update_display_proxy()
    viewer.set_red_eye_mode(True)

    # 1. Selection over skin only (no red pupil) -> must remain 100% untouched
    viewer._start_red_eye(QPointF(5, 5))
    viewer._drag_red_eye(QPointF(25, 25))
    viewer._finish_red_eye()

    skin_patch = viewer.master_np_rgb[5:25, 5:25]
    # Verify skin has NOT changed to green or olive
    assert np.array_equal(skin_patch[10, 10], [180, 130, 110])
    assert skin_patch[10, 10, 0] == 180
    assert skin_patch[10, 10, 1] == 130
    assert skin_patch[10, 10, 2] == 110

    # 2. Selection over red pupil
    viewer._start_red_eye(QPointF(40, 40))
    viewer._drag_red_eye(QPointF(60, 60))
    viewer._finish_red_eye()

    pupil_center = viewer.master_np_rgb[50, 50]
    # Must be neutral dark pupil (R <= 60, G <= 60, B <= 60)
    assert pupil_center[0] < 60
    assert pupil_center[1] < 60
    assert pupil_center[2] < 60
    # Must NOT turn green (green cannot be substantially greater than red)
    assert abs(int(pupil_center[0]) - int(pupil_center[1])) <= 5
    # Non-pupil skin around the box (e.g. (41, 41)) must remain skin tone
    corner_skin = viewer.master_np_rgb[41, 41]
    assert corner_skin[0] >= 170
    assert corner_skin[1] >= 120


def test_right_dock_no_horizontal_scroll(qapp):
    dock = TagAndCaptionDock()
    scroll_areas = dock.findChildren(_NoHorizontalScrollArea)
    assert len(scroll_areas) >= 2
    for sa in scroll_areas:
        assert sa.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
        assert not sa.horizontalScrollBar().isEnabled()

        event_left = QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Left, Qt.NoModifier)
        sa.keyPressEvent(event_left)
        assert not event_left.isAccepted()
        assert sa.horizontalScrollBar().value() == 0

        event_right = QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Right, Qt.NoModifier)
        sa.keyPressEvent(event_right)
        assert not event_right.isAccepted()
        assert sa.horizontalScrollBar().value() == 0


def test_ribbon_exposure_and_color_instant_popup(qapp):
    ribbon = RibbonToolbar()
    assert ribbon.btn_edit_color.popupMode() == QToolButton.InstantPopup
    assert ribbon.btn_edit_exposure.popupMode() == QToolButton.InstantPopup
