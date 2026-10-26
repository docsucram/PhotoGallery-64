import math
from pathlib import Path
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication

from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository
from src.ui.media_viewer import SingleMediaViewer
from src.ui.slideshow_view import FullscreenSlideshowDialog, SlideshowHud, make_slideshow_theme_icon
from src.ui.create_studio_dialogs import PortraitBokehDialog
import numpy as np


def test_crop_toolbar_ratio_swap_and_custom_inputs():
    app = QApplication.instance() or QApplication([])
    viewer = SingleMediaViewer()
    # Create dummy image for viewer
    from PIL import Image
    im = Image.new("RGB", (400, 300), color=(100, 150, 200))
    viewer.current_pil_image = im

    viewer.show()
    # Enter crop mode
    viewer.set_crop_mode(True)
    assert not viewer.crop_toolbar.isHidden()

    # 1. Select 4:3
    viewer.combo_crop_ratio.setCurrentText("4:3")
    assert viewer.txt_crop_w.text() == "4"
    assert viewer.txt_crop_h.text() == "3"
    assert abs(viewer.scene.crop_rect_item.aspect_ratio() - (4.0 / 3.0)) < 1e-4

    # 2. Click swap button ⇄
    viewer.btn_crop_swap.click()
    assert viewer.txt_crop_w.text() == "3"
    assert viewer.txt_crop_h.text() == "4"
    assert abs(viewer.scene.crop_rect_item.aspect_ratio() - (3.0 / 4.0)) < 1e-4
    assert viewer.combo_crop_ratio.currentText() == "3:4"

    # 3. Swap back to 4:3
    viewer.btn_crop_swap.click()
    assert viewer.txt_crop_w.text() == "4"
    assert viewer.txt_crop_h.text() == "3"
    assert abs(viewer.scene.crop_rect_item.aspect_ratio() - (4.0 / 3.0)) < 1e-4
    assert viewer.combo_crop_ratio.currentText() == "4:3"

    # 4. Custom text inputs: type 16 and 10
    viewer.txt_crop_w.setText("16")
    viewer.txt_crop_h.setText("10")
    viewer._on_crop_custom_text_edited()
    assert abs(viewer.scene.crop_rect_item.aspect_ratio() - 1.6) < 1e-4
    assert viewer.combo_crop_ratio.currentText() == "Custom"

    # 5. Clear inputs -> unconstrained / freeform
    viewer.txt_crop_w.setText("")
    viewer._on_crop_custom_text_edited()
    assert viewer.scene.crop_rect_item.aspect_ratio() is None

    viewer.set_crop_mode(False)


def test_portrait_blur_dialog_defaults_to_fit_to_window(tmp_path):
    app = QApplication.instance() or QApplication([])

    # Create dummy image BGR
    img_bgr = np.zeros((600, 800, 3), dtype=np.uint8)
    img_bgr[:] = (128, 128, 128)
    dummy_file = tmp_path / "portrait.jpg"
    from PIL import Image
    Image.fromarray(img_bgr).save(str(dummy_file))

    dlg = PortraitBokehDialog(image_bgr=img_bgr, file_path=dummy_file)
    dlg.resize(1000, 700)
    assert dlg._is_fit_mode is True

    # Show dialog
    dlg.show()
    app.processEvents()

    # View should be fitted to window (not 2% or 3%)
    current_scale = dlg.view.transform().m11()
    assert current_scale > 0.50  # Fits window properly (> 50%)
    dlg.close()


def test_slideshow_polaroid_rename_and_scattering_collage():
    app = QApplication.instance() or QApplication([])

    # 1. Theme option renamed to Polaroid
    theme_ids = [tid for tid, _ in SlideshowHud.THEME_OPTIONS]
    assert "polaroid" in theme_ids
    assert "contemporary" not in theme_ids

    # 2. Theme icon generator handles both polaroid and legacy contemporary
    icon_p = make_slideshow_theme_icon("polaroid")
    assert not icon_p.isNull()
    icon_c = make_slideshow_theme_icon("contemporary")
    assert not icon_c.isNull()

    # 3. FullscreenSlideshowDialog switches contemporary to polaroid
    from src.db.repository import MediaItemRecord
    dummy_records = [
        MediaItemRecord(
            id=100 + i,
            folder_id=1,
            filename=f"photo_{i}.jpg",
            size=1000,
            mtime=1000,
            date_taken=1000,
            rating=0,
            flag=0,
            width=200,
            height=150,
            orientation=1,
            is_video=0,
            caption="",
            group_key="",
        )
        for i in range(8)
    ]
    view = FullscreenSlideshowDialog(records=dummy_records)
    view.set_theme("contemporary")
    assert view.current_theme == "polaroid"
    view.set_theme("polaroid")
    assert view.current_theme == "polaroid"

    # 4. Polaroid scattering collage stack stores up to 6 cards with scatter positions
    for i in range(8):
        pm = QPixmap(200, 150)
        pm.fill(Qt.blue)
        mid = 100 + i
        view._cache[mid] = pm
        view.current_index = i
        view._load_slide(i)

    # Max 6 cards retained in stack
    assert len(view._contemporary_stack) == 6
    # Each card has organic normalized scatter coordinates
    for card in view._contemporary_stack:
        assert 0.15 <= card["norm_x"] <= 0.85
        assert 0.15 <= card["norm_y"] <= 0.85
        assert abs(card["tilt"]) <= 15.0

    view.close()
