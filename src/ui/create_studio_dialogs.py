"""
Creation Studio Dialogs:
Authentic Windows Live Photo Gallery Create Tab Studio Dialogs:
1. PanoramaDialog — Microsoft ICE-Style Panorama Stitcher with auto-crop of wavy black borders & gap inpainting.
2. CollageDialog — Interactive QGraphicsView Collage & 'Then & Now' Studio (Grid, Polaroid Scatter, and Then & Now).
3. PhotoFuseDialog — Best-Face Group Shot Combiner with burst homography alignment and seamless Poisson face cloning.
"""

from __future__ import annotations

import datetime
import math
from pathlib import Path
from typing import Any
import cv2
import numpy as np

from PySide6.QtCore import Qt, QThread, Signal, QRectF, QPointF, QPoint, QTimer
from PySide6.QtGui import (
    QImage,
    QPixmap,
    QColor,
    QIcon,
    QPainter,
    QPen,
    QBrush,
    QFont,
    QCursor,
)
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QCheckBox,
    QSlider,
    QSpinBox,
    QColorDialog,
    QProgressBar,
    QMessageBox,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsEllipseItem,
    QGraphicsTextItem,
    QScrollArea,
    QWidget,
    QFrame,
    QFileDialog,
    QButtonGroup,
    QToolButton,
)

from src.imaging.create_tools import (
    stitch_panorama,
    create_grid_collage,
    create_polaroid_scatter_collage,
    create_then_and_now_card,
    align_burst_frame,
    detect_faces_in_image,
    fuse_face,
)
from src.imaging.creative_effects import (
    estimate_subject_mask,
    apply_color_pop,
    refine_mask_brush,
    apply_portrait_bokeh_and_fill_light,
    render_3d_cinematic_motion_video,
)
from src.ui.ribbon_toolbar import make_wlpg_icon


# =============================================================================
# Helper Utilities
# =============================================================================

def _cv2_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    """Converts a BGR numpy array to QPixmap safely without dangling memory."""
    if img_bgr is None or img_bgr.size == 0:
        return QPixmap()
    if len(img_bgr.shape) == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    elif img_bgr.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
    h, w = img_bgr.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def _qcolor_to_bgr(color: QColor) -> tuple[int, int, int]:
    return color.blue(), color.green(), color.red()


def prompt_save_new_file(
    parent: QWidget,
    base_file_path: Path | None,
    suffix: str,
    ext: str = ".jpg",
    filter_str: str = "JPEG Image (*.jpg *.jpeg);;All Files (*.*)",
) -> Path | None:
    """
    Prompts the user to save the newly created item:
    - Makes it crystal clear that a new file is being created.
    - Suggests a default filename based on the first photo's stem + suffix (e.g. DSC3125_colourPop.jpg).
    - Opens a standard Save File dialog rooted in the source directory.
    """
    if base_file_path:
        stem = base_file_path.stem
        src_dir = base_file_path.parent
    else:
        stem = "Photo"
        src_dir = Path.home()

    default_name = f"{stem}_{suffix}{ext}"
    initial_path = (src_dir / default_name).as_posix()

    selected_path, _ = QFileDialog.getSaveFileName(
        parent,
        f"Save New File ({suffix})",
        initial_path,
        filter_str,
    )
    if not selected_path:
        return None
    return Path(selected_path)


# =============================================================================
# 1. Panorama Stitcher Dialog (Microsoft ICE-Style)
# =============================================================================

class PanoramaWorker(QThread):
    """Background worker for multi-image stitching."""
    progress_updated = Signal(str)
    stitching_finished = Signal(bool, object, str)

    def __init__(
        self,
        images_bgr: list[np.ndarray],
        projection_mode: str,
        auto_crop: bool,
        fill_minor_gaps: bool,
    ) -> None:
        super().__init__()
        self.images_bgr = images_bgr
        self.projection_mode = projection_mode
        self.auto_crop = auto_crop
        self.fill_minor_gaps = fill_minor_gaps

    def run(self) -> None:
        self.progress_updated.emit("Analyzing feature keypoints & camera trajectories...")
        success, result_bgr, msg = stitch_panorama(
            self.images_bgr,
            projection_mode=self.projection_mode,
            auto_crop=self.auto_crop,
            fill_minor_gaps=self.fill_minor_gaps,
        )
        self.stitching_finished.emit(success, result_bgr, msg)


class PanoramaDialog(QDialog):
    """
    Microsoft ICE-Style Panorama Stitcher Dialog:
    - Projection modes: Cylindrical, Spherical, Planar/Perspective.
    - Signature ICE auto-crop of wavy black borders to largest clean interior rectangle.
    - Telea inpainting of minor border gaps.
    - High-res export to same directory as source photos.
    """

    def __init__(
        self,
        photo_paths: list[Path],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.photo_paths = photo_paths
        self.output_dir = photo_paths[0].parent if photo_paths else Path.home()
        self.saved_file_path: Path | None = None
        self.current_stitched_bgr: np.ndarray | None = None
        self.raw_stitched_bgr: np.ndarray | None = None
        self._worker: PanoramaWorker | None = None

        self.setWindowTitle("Create Panorama — Windows Live Photo Gallery")
        self.setWindowIcon(make_wlpg_icon("panoramic", 24))
        self.resize(1000, 680)
        self._init_ui()
        self._load_and_stitch()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 16)
        layout.setSpacing(10)

        # Header
        lbl_title = QLabel("Microsoft Image Composite Editor (ICE) Panorama Stitcher")
        lbl_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #1e395b;")
        lbl_subtitle = QLabel(f"Stitching {len(self.photo_paths)} overlapping photos into a seamless wide-angle panorama.")
        lbl_subtitle.setStyleSheet("color: #555555; font-size: 12px;")
        layout.addWidget(lbl_title)
        layout.addWidget(lbl_subtitle)

        # Settings bar
        bar_frame = QFrame()
        bar_frame.setStyleSheet("background: #f4f6fa; border: 1px solid #d0d7e5; border-radius: 4px; padding: 4px;")
        bar_layout = QHBoxLayout(bar_frame)
        bar_layout.setContentsMargins(8, 4, 8, 4)
        bar_layout.setSpacing(12)

        bar_layout.addWidget(QLabel("Projection Mode:"))
        self.cmb_projection = QComboBox()
        self.cmb_projection.addItems([
            "Cylindrical (Standard Wide)",
            "Spherical (360 / Multi-Row)",
            "Planar / Perspective (Document / Whiteboard Scan)",
        ])
        self.cmb_projection.currentIndexChanged.connect(self._on_settings_changed)
        bar_layout.addWidget(self.cmb_projection)

        self.chk_autocrop = QCheckBox("Auto-crop wavy black edges (Signature ICE)")
        self.chk_autocrop.setChecked(True)
        self.chk_autocrop.setToolTip("Computes the largest axis-aligned interior rectangle containing zero black border pixels.")
        self.chk_autocrop.toggled.connect(self._on_settings_changed)
        bar_layout.addWidget(self.chk_autocrop)

        self.chk_inpaint = QCheckBox("Fill minor border gaps (Inpaint)")
        self.chk_inpaint.setChecked(False)
        self.chk_inpaint.setToolTip("Inpaints minor sky/grass corner gaps using Telea diffusion before cropping.")
        self.chk_inpaint.toggled.connect(self._on_settings_changed)
        bar_layout.addWidget(self.chk_inpaint)

        bar_layout.addStretch(1)

        self.btn_restitch = QPushButton("Re-Stitch")
        self.btn_restitch.setStyleSheet("background: #e1e7f0; border: 1px solid #7a96df; padding: 4px 12px; border-radius: 3px;")
        self.btn_restitch.clicked.connect(self._load_and_stitch)
        bar_layout.addWidget(self.btn_restitch)

        layout.addWidget(bar_frame)

        # Interactive Preview Canvas
        self.preview_view = QGraphicsView()
        self.preview_scene = QGraphicsScene(self)
        self.preview_view.setScene(self.preview_scene)
        self.preview_view.setStyleSheet("background: #252528; border: 1px solid #a0a0a0; border-radius: 4px;")
        self.preview_pixmap_item = QGraphicsPixmapItem()
        self.preview_scene.addItem(self.preview_pixmap_item)
        layout.addWidget(self.preview_view, 1)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setStyleSheet("QProgressBar { height: 6px; border: none; background: #e0e0e0; } QProgressBar::chunk { background: #2D72BC; }")
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet("color: #666; font-size: 11px;")

        prog_layout = QHBoxLayout()
        prog_layout.addWidget(self.progress_bar)
        prog_layout.addWidget(self.lbl_status)
        layout.addLayout(prog_layout)

        # Bottom Buttons
        btn_layout = QHBoxLayout()
        self.lbl_dims = QLabel("")
        self.lbl_dims.setStyleSheet("color: #444; font-size: 11px;")
        btn_layout.addWidget(self.lbl_dims)
        btn_layout.addStretch(1)

        self.btn_save = QPushButton("Save New Panorama...")
        self.btn_save.setEnabled(False)
        self.btn_save.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3c82e6, stop:1 #1c52ba);
                color: white; font-weight: bold; border: 1px solid #143e8c;
                border-radius: 4px; padding: 6px 18px; min-width: 110px;
            }
            QPushButton:hover { background: #4d8ff2; }
            QPushButton:disabled { background: #cccccc; color: #888888; border: 1px solid #aaaaaa; }
        """)
        self.btn_save.clicked.connect(self._save_panorama)
        btn_layout.addWidget(self.btn_save)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setStyleSheet("padding: 6px 14px; border-radius: 4px; border: 1px solid #ccc; background: #f0f0f0;")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_cancel)

        layout.addLayout(btn_layout)

    def _load_and_stitch(self) -> None:
        self.progress_bar.setVisible(True)
        self.btn_save.setEnabled(False)
        self.btn_restitch.setEnabled(False)
        self.lbl_status.setText("Loading photos & initializing stitcher...")

        # Load photos in BGR
        images_bgr = []
        for p in self.photo_paths:
            img = cv2.imread(str(p))
            if img is not None:
                images_bgr.append(img)

        if len(images_bgr) < 2:
            self.lbl_status.setText("Failed: Could not load 2+ valid images.")
            self.progress_bar.setVisible(False)
            self.btn_restitch.setEnabled(True)
            return

        mode_text = self.cmb_projection.currentText()
        proj_mode = "planar" if "Planar" in mode_text else ("spherical" if "Spherical" in mode_text else "cylindrical")

        self._worker = PanoramaWorker(
            images_bgr=images_bgr,
            projection_mode=proj_mode,
            auto_crop=self.chk_autocrop.isChecked(),
            fill_minor_gaps=self.chk_inpaint.isChecked(),
        )
        self._worker.progress_updated.connect(self.lbl_status.setText)
        self._worker.stitching_finished.connect(self._on_stitch_completed)
        self._worker.start()

    def _on_stitch_completed(self, success: bool, result_bgr: np.ndarray, msg: str) -> None:
        self.progress_bar.setVisible(False)
        self.btn_restitch.setEnabled(True)
        self.lbl_status.setText(msg)

        if not success or result_bgr is None or result_bgr.size == 0:
            QMessageBox.warning(self, "Panorama Stitching", f"Could not stitch panorama:\n{msg}")
            return

        self.current_stitched_bgr = result_bgr
        self.btn_save.setEnabled(True)

        h, w = result_bgr.shape[:2]
        self.lbl_dims.setText(f"Dimensions: {w} × {h} px ({w*h//1_000_000} MP)")

        # Render preview pixmap
        pm = _cv2_to_qpixmap(result_bgr)
        self.preview_pixmap_item.setPixmap(pm)
        self.preview_scene.setSceneRect(QRectF(pm.rect()))
        self.preview_view.fitInView(self.preview_scene.sceneRect(), Qt.KeepAspectRatio)

    def _on_settings_changed(self) -> None:
        # Re-stitch when projection or options change
        if self.current_stitched_bgr is not None:
            self._load_and_stitch()

    def _save_panorama(self) -> None:
        if self.current_stitched_bgr is None:
            return

        first_path = self.photo_paths[0] if self.photo_paths else None
        out_path = prompt_save_new_file(self, first_path, "panorama", ".jpg")
        if not out_path:
            return

        try:
            cv2.imwrite(str(out_path), self.current_stitched_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
            self.saved_file_path = out_path
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save panorama to disk:\n{e}")


# =============================================================================
# 2. Interactive Collage & 'Then & Now' Studio Dialog
# =============================================================================

class CollageDialog(QDialog):
    """
    Interactive QGraphicsView Collage & 'Then & Now' Studio:
    - 3 Layout Styles:
      1. Modern Grid / Masonry (2x1, 2x2, 3x2, Center Hero, Masonry)
      2. Classic WLPG Polaroid Table Scatter with drop shadows & handwritten caption
      3. 'Then & Now' Comparison Card with chronological sorting & badges
    - Adjustable borders, corner rounding, background themes.
    - Full print resolution export (4K UHD or 300 DPI A4).
    """

    def __init__(
        self,
        photos_data: list[dict[str, Any]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.photos_data = photos_data
        self.output_dir = Path(photos_data[0]["file_path"]).parent if photos_data and photos_data[0].get("file_path") else Path.home()
        self.saved_file_path: Path | None = None
        self.rendered_bgr: np.ndarray | None = None

        self.bg_color = QColor(245, 245, 248)
        self.bg_texture = "dark_wood"

        self.setWindowTitle("Create Collage — Windows Live Photo Gallery")
        self.setWindowIcon(make_wlpg_icon("collage", 24))
        self.resize(1060, 720)
        self._init_ui()
        self._render_preview()

    def _init_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Left Controls Panel
        ctrl_panel = QFrame()
        ctrl_panel.setFixedWidth(280)
        ctrl_panel.setStyleSheet("background: #f7f9fc; border: 1px solid #d4dae6; border-radius: 6px; padding: 6px;")
        ctrl_layout = QVBoxLayout(ctrl_panel)
        ctrl_layout.setContentsMargins(8, 8, 8, 8)
        ctrl_layout.setSpacing(10)

        lbl_header = QLabel("Collage Studio")
        lbl_header.setStyleSheet("font-size: 15px; font-weight: bold; color: #1e395b;")
        ctrl_layout.addWidget(lbl_header)

        # Style Tabs/Buttons
        ctrl_layout.addWidget(QLabel("Layout Style:"))
        self.cmb_style = QComboBox()
        self.cmb_style.addItems([
            "Modern Grid / Masonry",
            "Polaroid Table Scatter",
            "Then & Now Comparison Card",
        ])
        self.cmb_style.currentIndexChanged.connect(self._on_style_changed)
        ctrl_layout.addWidget(self.cmb_style)

        # Grid-specific controls
        self.frame_grid_opts = QFrame()
        grid_opts_layout = QVBoxLayout(self.frame_grid_opts)
        grid_opts_layout.setContentsMargins(0, 0, 0, 0)
        grid_opts_layout.setSpacing(8)

        grid_opts_layout.addWidget(QLabel("Grid Arrangement:"))
        self.cmb_grid_layout = QComboBox()
        self.cmb_grid_layout.addItems(["2x2", "2x1", "3x2", "Center Hero", "Masonry"])
        self.cmb_grid_layout.currentIndexChanged.connect(self._render_preview)
        grid_opts_layout.addWidget(self.cmb_grid_layout)

        grid_opts_layout.addWidget(QLabel("Border Thickness (0..40px):"))
        self.sld_border = QSlider(Qt.Horizontal)
        self.sld_border.setRange(0, 40)
        self.sld_border.setValue(12)
        self.sld_border.valueChanged.connect(self._render_preview)
        grid_opts_layout.addWidget(self.sld_border)

        grid_opts_layout.addWidget(QLabel("Corner Rounding (0..24px):"))
        self.sld_corner = QSlider(Qt.Horizontal)
        self.sld_corner.setRange(0, 24)
        self.sld_corner.setValue(8)
        self.sld_corner.valueChanged.connect(self._render_preview)
        grid_opts_layout.addWidget(self.sld_corner)

        self.btn_bg_color = QPushButton("Pick Background Colour")
        self.btn_bg_color.clicked.connect(self._pick_bg_color)
        grid_opts_layout.addWidget(self.btn_bg_color)
        ctrl_layout.addWidget(self.frame_grid_opts)

        # Polaroid-specific controls
        self.frame_polaroid_opts = QFrame()
        self.frame_polaroid_opts.setVisible(False)
        pol_layout = QVBoxLayout(self.frame_polaroid_opts)
        pol_layout.setContentsMargins(0, 0, 0, 0)
        pol_layout.setSpacing(8)

        pol_layout.addWidget(QLabel("Canvas Texture:"))
        self.cmb_texture = QComboBox()
        self.cmb_texture.addItems(["Mahogany Wood", "Clean Linen", "Modern Slate"])
        self.cmb_texture.currentIndexChanged.connect(self._render_preview)
        pol_layout.addWidget(self.cmb_texture)

        self.chk_captions = QCheckBox("Print Date / Caption on Border")
        self.chk_captions.setChecked(True)
        self.chk_captions.toggled.connect(self._render_preview)
        pol_layout.addWidget(self.chk_captions)
        ctrl_layout.addWidget(self.frame_polaroid_opts)

        # Export Resolution
        ctrl_layout.addSpacing(10)
        ctrl_layout.addWidget(QLabel("Export Resolution:"))
        self.cmb_resolution = QComboBox()
        self.cmb_resolution.addItems([
            "4K UHD (3840 × 2160)",
            "Full HD (1920 × 1080)",
            "A4 Print 300 DPI (3508 × 2480)",
        ])
        self.cmb_resolution.currentIndexChanged.connect(self._render_preview)
        ctrl_layout.addWidget(self.cmb_resolution)

        ctrl_layout.addStretch(1)

        # Save and Cancel
        self.btn_save = QPushButton("Save New Collage...")
        self.btn_save.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3c82e6, stop:1 #1c52ba);
                color: white; font-weight: bold; border: 1px solid #143e8c;
                border-radius: 4px; padding: 7px 18px;
            }
            QPushButton:hover { background: #4d8ff2; }
        """)
        self.btn_save.clicked.connect(self._save_collage)
        ctrl_layout.addWidget(self.btn_save)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        ctrl_layout.addWidget(btn_cancel)

        layout.addWidget(ctrl_panel)

        # Right Preview Area
        right_panel = QVBoxLayout()
        self.preview_view = QGraphicsView()
        self.preview_scene = QGraphicsScene(self)
        self.preview_view.setScene(self.preview_scene)
        self.preview_view.setStyleSheet("background: #1e1e20; border: 1px solid #999; border-radius: 6px;")
        self.preview_pixmap_item = QGraphicsPixmapItem()
        self.preview_scene.addItem(self.preview_pixmap_item)
        right_panel.addWidget(self.preview_view, 1)

        self.lbl_info = QLabel("Interactive Preview")
        self.lbl_info.setStyleSheet("color: #555; font-size: 11px;")
        right_panel.addWidget(self.lbl_info)

        layout.addLayout(right_panel, 1)

    def _on_style_changed(self, idx: int) -> None:
        self.frame_grid_opts.setVisible(idx == 0)
        self.frame_polaroid_opts.setVisible(idx == 1)
        self._render_preview()

    def _pick_bg_color(self) -> None:
        col = QColorDialog.getColor(self.bg_color, self, "Select Background Colour")
        if col.isValid():
            self.bg_color = col
            self._render_preview()

    def _get_target_resolution(self) -> tuple[int, int]:
        txt = self.cmb_resolution.currentText()
        if "4K" in txt:
            return 3840, 2160
        elif "A4" in txt:
            return 3508, 2480
        return 1920, 1080

    def _render_preview(self) -> None:
        target_size = self._get_target_resolution()
        style_idx = self.cmb_style.currentIndex()

        # Build list of numpy images
        images_bgr = []
        for d in self.photos_data:
            img = d.get("image_bgr")
            if img is None and d.get("file_path"):
                img = cv2.imread(str(d["file_path"]))
            if img is not None:
                images_bgr.append(img)

        if not images_bgr:
            return

        if style_idx == 0:
            # Modern Grid
            layout_name = self.cmb_grid_layout.currentText().lower().replace(" ", "_")
            rendered = create_grid_collage(
                images_bgr=images_bgr,
                layout=layout_name,
                border_thickness=self.sld_border.value(),
                corner_radius=self.sld_corner.value(),
                bg_color=_qcolor_to_bgr(self.bg_color),
                target_size=target_size,
            )
        elif style_idx == 1:
            # Polaroid Scatter
            tex_name = "dark_wood" if "Wood" in self.cmb_texture.currentText() else ("linen" if "Linen" in self.cmb_texture.currentText() else "slate")
            polaroid_data = []
            for i, d in enumerate(self.photos_data):
                if i < len(images_bgr):
                    polaroid_data.append({
                        "image_bgr": images_bgr[i],
                        "caption": d.get("caption") or d.get("date_taken") or d.get("filename"),
                        "date_str": str(d.get("date_taken") or ""),
                    })
            rendered = create_polaroid_scatter_collage(
                images_data=polaroid_data,
                target_size=target_size,
                bg_style=tex_name,
                print_captions=self.chk_captions.isChecked(),
            )
        else:
            # Then & Now Comparison Card
            then_now_data = []
            for i, d in enumerate(self.photos_data):
                if i < len(images_bgr):
                    then_now_data.append({
                        "image_bgr": images_bgr[i],
                        "date_taken": d.get("date_taken") or d.get("date_str") or "",
                        "caption": d.get("caption") or "",
                    })
            rendered = create_then_and_now_card(
                images_data=then_now_data,
                target_size=target_size,
            )

        self.rendered_bgr = rendered

        # Display in QGraphicsView
        pm = _cv2_to_qpixmap(rendered)
        self.preview_pixmap_item.setPixmap(pm)
        self.preview_scene.setSceneRect(QRectF(pm.rect()))
        self.preview_view.fitInView(self.preview_scene.sceneRect(), Qt.KeepAspectRatio)

        h, w = rendered.shape[:2]
        self.lbl_info.setText(f"Canvas size: {w} × {h} px • {len(images_bgr)} photos included")

    def _save_collage(self) -> None:
        if self.rendered_bgr is None:
            return

        first_path = self.photos_data[0].get("file_path") if self.photos_data else None
        if first_path is not None and not isinstance(first_path, Path):
            first_path = Path(first_path)
        out_path = prompt_save_new_file(self, first_path, "collage", ".jpg")
        if not out_path:
            return

        try:
            cv2.imwrite(str(out_path), self.rendered_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
            self.saved_file_path = out_path
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save collage:\n{e}")


# =============================================================================
# 3. Photo Fuse Dialog (Best-Face Group Shot Combiner)
# =============================================================================

class _InteractiveFaceItem(QGraphicsRectItem):
    """Interactive face box with robust click handling and hover feedback."""
    def __init__(self, rect: QRectF, index: int, on_clicked: Any) -> None:
        super().__init__(rect)
        self.index = index
        self.on_clicked = on_clicked
        self.setCursor(Qt.PointingHandCursor)
        self.setAcceptHoverEvents(True)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.on_clicked(self.index)
            event.accept()
            return
        super().mousePressEvent(event)


class PhotoFuseDialog(QDialog):
    """
    Photo Fuse — Best-Face Group Shot Combiner:
    - Automatically aligns 2..6 burst group shots to Base Frame (0) via Homography.
    - Automatically detects faces in the base frame.
    - Clicking a face box opens the bottom replacement strip of that person from all shots.
    - Seamlessly clones the chosen face into the group shot using Poisson cloning.
    """

    def __init__(
        self,
        photo_paths: list[Path],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.photo_paths = photo_paths
        self.output_dir = photo_paths[0].parent if photo_paths else Path.home()
        self.saved_file_path: Path | None = None

        self.frames_bgr: list[np.ndarray] = []
        self.aligned_frames_bgr: list[np.ndarray] = []
        self.base_fused_bgr: np.ndarray | None = None
        self.face_boxes: list[tuple[int, int, int, int]] = []
        self.selected_face_idx: int = -1
        self._first_fit: bool = True

        self.setWindowTitle("Photo Fuse — Best-Face Group Shot Combiner")
        self.setWindowIcon(make_wlpg_icon("photo_fuse", 24))
        self.resize(1080, 740)
        self._init_ui()
        self._align_and_detect()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(10)

        # Header
        lbl_title = QLabel("Photo Fuse: Combine the Best Faces from Group Bursts")
        lbl_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #1e395b;")
        lbl_sub = QLabel("Click any person's face box on the photo to see and choose their best expression from all burst shots.")
        lbl_sub.setStyleSheet("color: #555555; font-size: 12px;")
        layout.addWidget(lbl_title)
        layout.addWidget(lbl_sub)

        # Main interactive canvas
        self.view = QGraphicsView()
        self.scene = QGraphicsScene(self)
        self.view.setScene(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing, True)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.view.setStyleSheet("background: #252528; border: 1px solid #777; border-radius: 4px;")
        self.bg_item = QGraphicsPixmapItem()
        self.scene.addItem(self.bg_item)
        layout.addWidget(self.view, 1)

        # Bottom face replacement strip
        self.strip_frame = QFrame()
        self.strip_frame.setFixedHeight(140)
        self.strip_frame.setStyleSheet("background: #f0f3f8; border: 1px solid #c5cfe0; border-radius: 4px;")
        strip_layout = QVBoxLayout(self.strip_frame)
        strip_layout.setContentsMargins(8, 4, 8, 4)
        strip_layout.setSpacing(4)

        self.lbl_strip_info = QLabel("Select a face box above to view expressions from other shots:")
        self.lbl_strip_info.setStyleSheet("font-weight: bold; font-size: 11px; color: #222;")
        strip_layout.addWidget(self.lbl_strip_info)

        self.strip_scroll = QScrollArea()
        self.strip_scroll.setWidgetResizable(True)
        self.strip_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.strip_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.strip_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self.strip_content = QWidget()
        self.strip_content_layout = QHBoxLayout(self.strip_content)
        self.strip_content_layout.setContentsMargins(4, 4, 4, 4)
        self.strip_content_layout.setSpacing(12)
        self.strip_content_layout.addStretch(1)
        self.strip_scroll.setWidget(self.strip_content)
        strip_layout.addWidget(self.strip_scroll)

        layout.addWidget(self.strip_frame)

        # Bottom buttons
        btn_layout = QHBoxLayout()
        self.lbl_status = QLabel("Aligning burst frames...")
        self.lbl_status.setStyleSheet("color: #444; font-size: 11px;")
        btn_layout.addWidget(self.lbl_status)
        btn_layout.addStretch(1)

        self.btn_reset = QPushButton("Reset to Original")
        self.btn_reset.clicked.connect(self._reset_to_base)
        btn_layout.addWidget(self.btn_reset)

        self.btn_save = QPushButton("Save New Fused Photo...")
        self.btn_save.setEnabled(False)
        self.btn_save.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3c82e6, stop:1 #1c52ba);
                color: white; font-weight: bold; border: 1px solid #143e8c;
                border-radius: 4px; padding: 6px 18px; min-width: 120px;
            }
            QPushButton:hover { background: #4d8ff2; }
            QPushButton:disabled { background: #ccc; color: #888; }
        """)
        self.btn_save.clicked.connect(self._save_fused_photo)
        btn_layout.addWidget(self.btn_save)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_cancel)

        layout.addLayout(btn_layout)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._first_fit and self.scene.sceneRect().width() > 0:
            self._first_fit = False
            self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def _align_and_detect(self) -> None:
        """Loads and aligns all secondary burst photos to base frame 0."""
        for p in self.photo_paths:
            img = cv2.imread(str(p))
            if img is not None:
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                self.frames_bgr.append(img)

        if len(self.frames_bgr) < 2:
            self.lbl_status.setText("Failed: Need at least 2 photos to fuse.")
            return

        base = self.frames_bgr[0]
        self.aligned_frames_bgr.append(base.copy())
        self.base_fused_bgr = base.copy()

        for i, target in enumerate(self.frames_bgr[1:], start=2):
            self.lbl_status.setText(f"Aligning burst shot {i} of {len(self.frames_bgr)}...")
            success, aligned = align_burst_frame(base, target)
            self.aligned_frames_bgr.append(aligned)

        self.lbl_status.setText("Detecting faces...")
        self.face_boxes = detect_faces_in_image(self.base_fused_bgr)

        if not self.face_boxes:
            h, w = base.shape[:2]
            cx, cy = w // 2, h // 2
            fw, fh = int(w * 0.2), int(h * 0.25)
            self.face_boxes = [(cx - fw // 2, cy - fh // 2, fw, fh)]

        self.lbl_status.setText(f"Found {len(self.face_boxes)} faces across {len(self.frames_bgr)} shots. Ready to fuse.")
        self.btn_save.setEnabled(True)
        self._update_scene()
        if self.isVisible():
            self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def _update_scene(self) -> None:
        if self.base_fused_bgr is None:
            return

        pm = _cv2_to_qpixmap(self.base_fused_bgr)
        self.bg_item.setPixmap(pm)
        self.scene.setSceneRect(QRectF(pm.rect()))

        # Remove existing face items
        for item in list(self.scene.items()):
            if isinstance(item, _InteractiveFaceItem):
                self.scene.removeItem(item)

        # Draw interactive face rectangles
        for i, (fx, fy, fw, fh) in enumerate(self.face_boxes):
            rect_item = _InteractiveFaceItem(QRectF(fx, fy, fw, fh), i, self._select_face)
            is_sel = (i == self.selected_face_idx)
            pen = QPen(QColor(0, 180, 255) if is_sel else QColor(255, 255, 255, 200))
            pen.setWidth(3 if is_sel else 2)
            rect_item.setPen(pen)
            rect_item.setBrush(QBrush(QColor(0, 180, 255, 45) if is_sel else QColor(0, 0, 0, 0)))
            self.scene.addItem(rect_item)

    def _select_face(self, face_idx: int) -> None:
        self.selected_face_idx = face_idx
        self._update_scene()
        self._populate_face_strip(face_idx)

    def _populate_face_strip(self, face_idx: int) -> None:
        while self.strip_content_layout.count() > 1:
            child = self.strip_content_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        fx, fy, fw, fh = self.face_boxes[face_idx]
        self.lbl_strip_info.setText(f"Expressions for Person {face_idx + 1} (Choose to replace in photo):")

        for shot_idx, aligned_img in enumerate(self.aligned_frames_bgr):
            h, w = aligned_img.shape[:2]
            margin_x = int(fw * 0.2)
            margin_y = int(fh * 0.2)
            x1 = max(0, fx - margin_x)
            y1 = max(0, fy - margin_y)
            x2 = min(w, fx + fw + margin_x)
            y2 = min(h, fy + fh + margin_y)

            crop_bgr = aligned_img[y1:y2, x1:x2]
            if crop_bgr.size == 0:
                continue

            crop_thumb = cv2.resize(crop_bgr, (72, 72), interpolation=cv2.INTER_AREA)
            pm_thumb = _cv2_to_qpixmap(crop_thumb)

            btn_card = QToolButton()
            btn_card.setIcon(QIcon(pm_thumb))
            btn_card.setIconSize(pm_thumb.size())
            btn_card.setText(f"Shot {shot_idx + 1}")
            btn_card.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            btn_card.setStyleSheet("""
                QToolButton {
                    background: white; border: 1px solid #b0c0d8;
                    border-radius: 4px; padding: 4px; font-size: 10px;
                }
                QToolButton:hover {
                    border: 2px solid #0078d4; background: #eef6fc;
                }
            """)
            btn_card.clicked.connect(lambda _, s=shot_idx: self._apply_face_fuse(s))
            self.strip_content_layout.insertWidget(self.strip_content_layout.count() - 1, btn_card)

    def _apply_face_fuse(self, shot_idx: int) -> None:
        if self.selected_face_idx < 0 or self.base_fused_bgr is None:
            return

        source_aligned = self.aligned_frames_bgr[shot_idx]
        face_rect = self.face_boxes[self.selected_face_idx]

        self.lbl_status.setText(f"Blending face from Shot {shot_idx + 1} with Poisson seamless clone...")
        fused = fuse_face(self.base_fused_bgr, source_aligned, face_rect)
        self.base_fused_bgr = fused
        self._update_scene()
        self.lbl_status.setText(f"✓ Face fused successfully from Shot {shot_idx + 1}!")

    def _reset_to_base(self) -> None:
        if self.frames_bgr:
            self.base_fused_bgr = self.frames_bgr[0].copy()
            self._update_scene()
            self.lbl_status.setText("Reset to original Shot 1.")

    def _save_fused_photo(self) -> None:
        if self.base_fused_bgr is None:
            return

        first_path = self.photo_paths[0] if self.photo_paths else None
        out_path = prompt_save_new_file(self, first_path, "photoFuse", ".jpg")
        if not out_path:
            return

        try:
            cv2.imwrite(str(out_path), self.base_fused_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
            self.saved_file_path = out_path
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save photo:\n{e}")


# =============================================================================
# 4. Google Photos-Style "Colour Pop" (Subject Colour Splash) Dialog & Canvas
# =============================================================================

class ColourPopCanvasView(QGraphicsView):
    """
    High-performance interactive canvas for Colour Pop brush editing:
    - Real-time circular brush cursor following the pointer.
    - Continuous stroke interpolation between mouse moves so fast drags do not leave gaps.
    - Middle-click or Space+Drag for panning, mouse wheel for zoom.
    """
    brush_applied = Signal(int, int)
    stroke_started = Signal()
    stroke_finished = Signal()
    zoom_changed = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setMouseTracking(True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.brush_radius: int = 28
        self._is_painting: bool = False
        self._space_pressed: bool = False
        self._last_pt: QPointF | None = None

        self._cursor_item = QGraphicsEllipseItem()
        self._cursor_item.setPen(QPen(QColor(0, 160, 255, 230), 1.5, Qt.DashLine))
        self._cursor_item.setBrush(QBrush(QColor(0, 160, 255, 35)))
        self._cursor_item.setZValue(100)
        self._cursor_item.hide()

    def set_brush_radius(self, r: int) -> None:
        self.brush_radius = max(4, r)
        center = self._cursor_item.rect().center()
        rad = self.brush_radius
        self._cursor_item.setRect(QRectF(center.x() - rad, center.y() - rad, 2 * rad, 2 * rad))

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and not self._space_pressed:
            self._is_painting = True
            sp = self.mapToScene(event.pos())
            self._last_pt = sp
            self.stroke_started.emit()
            self.brush_applied.emit(int(sp.x()), int(sp.y()))
            event.accept()
            return
        elif event.button() == Qt.MiddleButton or (event.button() == Qt.LeftButton and self._space_pressed):
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            super().mousePressEvent(event)
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        sp = self.mapToScene(event.pos())
        rad = self.brush_radius
        self._cursor_item.setRect(QRectF(sp.x() - rad, sp.y() - rad, 2 * rad, 2 * rad))

        if self.scene() and self._cursor_item.scene() != self.scene():
            self.scene().addItem(self._cursor_item)
        if not self._cursor_item.isVisible() and not self._space_pressed:
            self._cursor_item.show()

        if self._is_painting and (event.buttons() & Qt.LeftButton) and not self._space_pressed:
            if self._last_pt is not None:
                p1 = self._last_pt
                p2 = sp
                dx = p2.x() - p1.x()
                dy = p2.y() - p1.y()
                dist = math.hypot(dx, dy)
                step = max(3.0, self.brush_radius * 0.35)
                steps = max(1, int(dist / step))
                for i in range(1, steps + 1):
                    ix = p1.x() + (dx * i) / steps
                    iy = p1.y() + (dy * i) / steps
                    self.brush_applied.emit(int(ix), int(iy))
            else:
                self.brush_applied.emit(int(sp.x()), int(sp.y()))
            self._last_pt = sp
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._is_painting:
            self._is_painting = False
            self._last_pt = None
            self.stroke_finished.emit()
            event.accept()
            return
        elif event.button() in (Qt.MiddleButton, Qt.LeftButton):
            if not self._space_pressed:
                self.setDragMode(QGraphicsView.NoDrag)
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Space and not self._space_pressed:
            self._space_pressed = True
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self._cursor_item.hide()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key_Space:
            self._space_pressed = False
            self.setDragMode(QGraphicsView.NoDrag)
            self._cursor_item.show()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def enterEvent(self, event) -> None:
        super().enterEvent(event)
        self.setCursor(Qt.BlankCursor)
        if self.scene() and self._cursor_item.scene() != self.scene():
            self.scene().addItem(self._cursor_item)
        self._cursor_item.show()

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        self.unsetCursor()
        self._cursor_item.hide()

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)
        self.zoom_changed.emit(self.transform().m11())
        event.accept()


class ZoomablePreviewCanvas(QGraphicsView):
    """
    Interactive zoomable/pannable canvas for creative tools previews:
    - Wheel zoom with zoom_changed signal.
    - Space+Drag and Middle-button drag for fluid panning.
    """
    zoom_changed = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._space_pressed: bool = False

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)
        self.zoom_changed.emit(self.transform().m11())
        event.accept()

    def mousePressEvent(self, event) -> None:
        if event.button() in (Qt.MiddleButton, Qt.LeftButton) and (self._space_pressed or event.button() == Qt.MiddleButton):
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            super().mousePressEvent(event)
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() in (Qt.MiddleButton, Qt.LeftButton):
            if not self._space_pressed:
                self.setDragMode(QGraphicsView.NoDrag)
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Space and not self._space_pressed:
            self._space_pressed = True
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key_Space:
            self._space_pressed = False
            self.setDragMode(QGraphicsView.NoDrag)
            event.accept()
            return
        super().keyReleaseEvent(event)


class ColourPopDialog(QDialog):
    """
    Google Photos-Style Colour Pop Studio:
    - Subject remains in full vibrant colour while background is converted to dramatic B&W.
    - Seeding via GrabCut segmentation / face detection.
    - Continuous 60 FPS proxy painting with bilateral edge-aware brush snapping.
    - Full undo/redo history, before/after compare, and live parameter readouts.
    """

    def __init__(
        self,
        image_bgr: np.ndarray,
        file_path: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        # Normalize channels to 3-channel BGR
        if len(image_bgr.shape) == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        elif image_bgr.shape[2] == 4:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)

        self.original_bgr = image_bgr.copy()
        self.file_path = file_path
        self.output_dir = file_path.parent if file_path else Path.home()
        self.saved_file_path: Path | None = None

        h, w = self.original_bgr.shape[:2]

        # Fast display proxy for instant 60 FPS painting
        max_proxy_dim = 1200
        if max(h, w) > max_proxy_dim:
            self.proxy_scale = max_proxy_dim / float(max(h, w))
            pw = max(1, int(round(w * self.proxy_scale)))
            ph = max(1, int(round(h * self.proxy_scale)))
            self.proxy_bgr = cv2.resize(self.original_bgr, (pw, ph), interpolation=cv2.INTER_AREA)
        else:
            self.proxy_scale = 1.0
            self.proxy_bgr = self.original_bgr.copy()

        # Subject mask estimation on proxy
        full_mask = estimate_subject_mask(self.original_bgr)
        self.proxy_mask = cv2.resize(full_mask, (self.proxy_bgr.shape[1], self.proxy_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Undo / Redo history
        self.undo_stack: list[np.ndarray] = [self.proxy_mask.copy()]
        self.redo_stack: list[np.ndarray] = []

        self.brush_mode = "keep_color"  # "keep_color" or "make_bw"
        self.brush_radius = 28
        self.bg_contrast = 1.15
        self.bg_brightness = 0.0
        self._first_fit = True

        self.proxy_bw: np.ndarray | None = None
        self._precompute_proxy_bw()

        self.setWindowTitle("Colour Pop Studio — Subject Colour Splash")
        self.setWindowIcon(make_wlpg_icon("color_spectrum", 24))
        self.resize(1080, 720)
        self._init_ui()
        self._refresh_canvas()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(10)

        # 1. Aero Header Bar
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #fcfdfe, stop:1 #e8eff7);
                border: 1px solid #c0d0e6;
                border-radius: 5px;
                padding: 6px 12px;
            }
        """)
        h_layout = QHBoxLayout(header_frame)
        h_layout.setContentsMargins(4, 2, 4, 2)

        lbl_icon = QLabel()
        lbl_icon.setPixmap(make_wlpg_icon("color_spectrum", 28).pixmap(28, 28))
        h_layout.addWidget(lbl_icon)

        title_vbox = QVBoxLayout()
        lbl_title = QLabel("Colour Pop Studio")
        lbl_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #1e395b;")
        lbl_sub = QLabel("Keep the main subject in vibrant colour while converting background to dramatic black & white.")
        lbl_sub.setStyleSheet("color: #555555; font-size: 11px;")
        title_vbox.addWidget(lbl_title)
        title_vbox.addWidget(lbl_sub)
        h_layout.addLayout(title_vbox)
        h_layout.addStretch(1)

        # Quick Actions in header
        self.btn_invert = QPushButton("Invert Mask")
        self.btn_invert.setToolTip("Swap colour subject and B&W background")
        self.btn_invert.setStyleSheet("padding: 4px 10px; border-radius: 3px; border: 1px solid #b0c2de; background: #ffffff;")
        self.btn_invert.clicked.connect(self._invert_mask)
        h_layout.addWidget(self.btn_invert)

        self.btn_auto_detect = QPushButton("Auto Detect")
        self.btn_auto_detect.setToolTip("Re-run automatic subject detection")
        self.btn_auto_detect.setStyleSheet("padding: 4px 10px; border-radius: 3px; border: 1px solid #b0c2de; background: #ffffff;")
        self.btn_auto_detect.clicked.connect(self._auto_detect_mask)
        h_layout.addWidget(self.btn_auto_detect)

        layout.addWidget(header_frame)

        # 2. Modern Ribbon Toolbar
        bar = QFrame()
        bar.setStyleSheet("background: #f4f6fa; border: 1px solid #d0d7e5; border-radius: 4px; padding: 4px;")
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(8, 4, 8, 4)
        bar_layout.setSpacing(10)

        # Brush mode buttons
        self.btn_group_mode = QButtonGroup(self)
        self.btn_group_mode.setExclusive(True)

        self.btn_keep = QPushButton("Keep Colour (+)")
        self.btn_keep.setCheckable(True)
        self.btn_keep.setChecked(True)
        self.btn_keep.setStyleSheet("""
            QPushButton {
                background: #ffffff; border: 1px solid #b0c0d8;
                border-radius: 3px; padding: 4px 12px; font-weight: 500;
            }
            QPushButton:checked {
                background: #cce5ff; border: 1px solid #0078d4;
                font-weight: bold; color: #004578;
            }
        """)
        self.btn_keep.clicked.connect(lambda: self._set_brush_mode("keep_color"))
        self.btn_group_mode.addButton(self.btn_keep)
        bar_layout.addWidget(self.btn_keep)

        self.btn_bw = QPushButton("Make B&W (-)")
        self.btn_bw.setCheckable(True)
        self.btn_bw.setStyleSheet("""
            QPushButton {
                background: #ffffff; border: 1px solid #b0c0d8;
                border-radius: 3px; padding: 4px 12px; font-weight: 500;
            }
            QPushButton:checked {
                background: #e2e2e6; border: 1px solid #666666;
                font-weight: bold; color: #222222;
            }
        """)
        self.btn_bw.clicked.connect(lambda: self._set_brush_mode("make_bw"))
        self.btn_group_mode.addButton(self.btn_bw)
        bar_layout.addWidget(self.btn_bw)

        # Brush size slider
        bar_layout.addSpacing(6)
        bar_layout.addWidget(QLabel("Brush:"))
        self.sld_radius = QSlider(Qt.Horizontal)
        self.sld_radius.setRange(8, 100)
        self.sld_radius.setValue(self.brush_radius)
        self.sld_radius.setFixedWidth(100)
        self.sld_radius.valueChanged.connect(self._on_radius_changed)
        bar_layout.addWidget(self.sld_radius)

        self.lbl_radius_val = QLabel(f"{self.brush_radius} px")
        self.lbl_radius_val.setStyleSheet("font-weight: bold; color: #333; min-width: 40px;")
        bar_layout.addWidget(self.lbl_radius_val)

        # B&W Contrast slider
        bar_layout.addSpacing(6)
        bar_layout.addWidget(QLabel("B&W Contrast:"))
        self.sld_contrast = QSlider(Qt.Horizontal)
        self.sld_contrast.setRange(80, 180)
        self.sld_contrast.setValue(115)
        self.sld_contrast.setFixedWidth(100)
        self.sld_contrast.valueChanged.connect(self._on_contrast_changed)
        bar_layout.addWidget(self.sld_contrast)

        self.lbl_contrast_val = QLabel("+15%")
        self.lbl_contrast_val.setStyleSheet("font-weight: bold; color: #333; min-width: 36px;")
        bar_layout.addWidget(self.lbl_contrast_val)

        # B&W Brightness slider
        bar_layout.addSpacing(6)
        bar_layout.addWidget(QLabel("Brightness:"))
        self.sld_bright = QSlider(Qt.Horizontal)
        self.sld_bright.setRange(-40, 40)
        self.sld_bright.setValue(0)
        self.sld_bright.setFixedWidth(80)
        self.sld_bright.valueChanged.connect(self._on_bright_changed)
        bar_layout.addWidget(self.sld_bright)

        self.lbl_bright_val = QLabel("0")
        self.lbl_bright_val.setStyleSheet("font-weight: bold; color: #333; min-width: 25px;")
        bar_layout.addWidget(self.lbl_bright_val)

        bar_layout.addStretch(1)

        # Undo / Redo
        self.btn_undo = QPushButton("Undo")
        self.btn_undo.setEnabled(False)
        self.btn_undo.setShortcut("Ctrl+Z")
        self.btn_undo.clicked.connect(self._undo_mask)
        bar_layout.addWidget(self.btn_undo)

        self.btn_redo = QPushButton("Redo")
        self.btn_redo.setEnabled(False)
        self.btn_redo.setShortcut("Ctrl+Y")
        self.btn_redo.clicked.connect(self._redo_mask)
        bar_layout.addWidget(self.btn_redo)

        # Hold to Compare button
        self.btn_compare = QPushButton("Hold to Compare")
        self.btn_compare.setToolTip("Press and hold to preview original photo")
        self.btn_compare.setStyleSheet("background: #f0f4fa; border: 1px solid #7a96df; font-weight: bold; padding: 4px 10px; border-radius: 3px;")
        self.btn_compare.pressed.connect(self._show_original)
        self.btn_compare.released.connect(self._restore_color_pop)
        bar_layout.addWidget(self.btn_compare)

        layout.addWidget(bar)

        # 3. Interactive Canvas
        self.view = ColourPopCanvasView(self)
        self.scene = QGraphicsScene(self)
        self.view.setScene(self.scene)
        self.view.set_brush_radius(self.brush_radius)
        self.view.brush_applied.connect(self._on_brush_applied)
        self.view.stroke_started.connect(self._on_stroke_started)
        self.view.zoom_changed.connect(self._on_zoom_changed)
        self.view.setStyleSheet("background: #252528; border: 1px solid #777; border-radius: 4px;")

        self.bg_item = QGraphicsPixmapItem()
        self.scene.addItem(self.bg_item)
        layout.addWidget(self.view, 1)

        # 4. Bottom Footer & Status
        btn_layout = QHBoxLayout()
        self.lbl_status = QLabel("✓ Paint over photo to add or remove colour • Space to Pan • Wheel to Zoom")
        self.lbl_status.setStyleSheet("color: #444; font-size: 11px;")
        btn_layout.addWidget(self.lbl_status)
        btn_layout.addStretch(1)

        # Zoom Controls
        zoom_frame = QFrame()
        zoom_frame.setStyleSheet("background: #f4f6fa; border: 1px solid #c0c8d8; border-radius: 3px; padding: 1px;")
        z_layout = QHBoxLayout(zoom_frame)
        z_layout.setContentsMargins(4, 1, 4, 1)
        z_layout.setSpacing(4)

        self.btn_zoom_out = QPushButton("−")
        self.btn_zoom_out.setFixedSize(24, 22)
        self.btn_zoom_out.setToolTip("Zoom Out (-)")
        self.btn_zoom_out.setStyleSheet("font-size: 13px; font-weight: bold; padding: 0;")
        self.btn_zoom_out.clicked.connect(lambda: self._apply_zoom(1.0 / 1.25))
        z_layout.addWidget(self.btn_zoom_out)

        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.setFixedSize(24, 22)
        self.btn_zoom_in.setToolTip("Zoom In (+)")
        self.btn_zoom_in.setStyleSheet("font-size: 13px; font-weight: bold; padding: 0;")
        self.btn_zoom_in.clicked.connect(lambda: self._apply_zoom(1.25))
        z_layout.addWidget(self.btn_zoom_in)

        self.btn_fit = QPushButton("Fit")
        self.btn_fit.setFixedHeight(22)
        self.btn_fit.setToolTip("Fit photo to canvas window")
        self.btn_fit.clicked.connect(self._fit_to_view)
        z_layout.addWidget(self.btn_fit)

        self.btn_100 = QPushButton("100%")
        self.btn_100.setFixedHeight(22)
        self.btn_100.setToolTip("View at actual 100% pixel size (Ctrl+0)")
        self.btn_100.clicked.connect(self._reset_zoom_100)
        z_layout.addWidget(self.btn_100)

        self.lbl_zoom_val = QLabel("100%")
        self.lbl_zoom_val.setStyleSheet("font-size: 11px; font-weight: bold; color: #333; min-width: 40px; padding-left: 2px;")
        z_layout.addWidget(self.lbl_zoom_val)

        btn_layout.addWidget(zoom_frame)
        btn_layout.addSpacing(12)

        # Save & Cancel
        btn_save = QPushButton("Save New Colour Pop Photo...")
        btn_save.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3c82e6, stop:1 #1c52ba);
                color: white; font-weight: bold; border: 1px solid #143e8c;
                border-radius: 4px; padding: 6px 20px; min-width: 140px;
            }
            QPushButton:hover { background: #4d8ff2; }
        """)
        btn_save.clicked.connect(self._save_photo)
        btn_layout.addWidget(btn_save)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_cancel)

        layout.addLayout(btn_layout)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._first_fit and self.scene.sceneRect().width() > 0:
            self._first_fit = False
            self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
            self._on_zoom_changed(self.view.transform().m11())

    def _on_zoom_changed(self, scale_val: float) -> None:
        self.lbl_zoom_val.setText(f"{int(round(scale_val * 100))}%")

    def _apply_zoom(self, factor: float) -> None:
        self.view.scale(factor, factor)
        self._on_zoom_changed(self.view.transform().m11())

    def _reset_zoom_100(self) -> None:
        self.view.resetTransform()
        self._on_zoom_changed(1.0)

    def _fit_to_view(self) -> None:
        if self.scene.sceneRect().isValid() and self.scene.sceneRect().width() > 0:
            self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
            self._on_zoom_changed(self.view.transform().m11())

    def _precompute_proxy_bw(self) -> None:
        gray = cv2.cvtColor(self.proxy_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_adj = np.clip((gray - 128.0) * self.bg_contrast + 128.0 + self.bg_brightness, 0.0, 255.0)
        self.proxy_bw = cv2.cvtColor(gray_adj.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    def _set_brush_mode(self, mode: str) -> None:
        self.brush_mode = mode
        self.btn_keep.setChecked(mode == "keep_color")
        self.btn_bw.setChecked(mode == "make_bw")
        self.lbl_status.setText(f"Active Brush: {'Keep Vibrant Colour (+)' if mode == 'keep_color' else 'Make Dramatic B&W (-)'}")

    def _on_radius_changed(self, val: int) -> None:
        self.brush_radius = val
        self.lbl_radius_val.setText(f"{val} px")
        self.view.set_brush_radius(val)

    def _on_contrast_changed(self, val: int) -> None:
        self.bg_contrast = val / 100.0
        delta = val - 100
        sign = "+" if delta >= 0 else ""
        self.lbl_contrast_val.setText(f"{sign}{delta}%")
        self._precompute_proxy_bw()
        self._refresh_canvas()

    def _on_bright_changed(self, val: int) -> None:
        self.bg_brightness = float(val)
        sign = "+" if val > 0 else ""
        self.lbl_bright_val.setText(f"{sign}{val}")
        self._precompute_proxy_bw()
        self._refresh_canvas()

    def _on_stroke_started(self) -> None:
        self.undo_stack.append(self.proxy_mask.copy())
        if len(self.undo_stack) > 25:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self.btn_undo.setEnabled(True)
        self.btn_redo.setEnabled(False)

    def _on_brush_applied(self, sx: int, sy: int) -> None:
        ph, pw = self.proxy_bgr.shape[:2]
        if 0 <= sx < pw and 0 <= sy < ph:
            self.proxy_mask = refine_mask_brush(
                self.proxy_mask,
                self.proxy_bgr,
                (sx, sy),
                radius=self.brush_radius,
                mode=self.brush_mode,
            )
            self._fast_refresh_proxy()

    def _fast_refresh_proxy(self) -> None:
        if self.proxy_bw is None:
            self._precompute_proxy_bw()
        m3 = self.proxy_mask[:, :, None]
        blended = (self.proxy_bgr.astype(np.float32) * m3 + self.proxy_bw.astype(np.float32) * (1.0 - m3)).astype(np.uint8)
        self.bg_item.setPixmap(_cv2_to_qpixmap(blended))

    def _refresh_canvas(self) -> None:
        self._fast_refresh_proxy()
        ph, pw = self.proxy_bgr.shape[:2]
        self.scene.setSceneRect(QRectF(0, 0, pw, ph))

    def _invert_mask(self) -> None:
        self._on_stroke_started()
        self.proxy_mask = 1.0 - self.proxy_mask
        self._fast_refresh_proxy()
        self.lbl_status.setText("✓ Inverted mask: subject and background swapped.")

    def _auto_detect_mask(self) -> None:
        self._on_stroke_started()
        full_mask = estimate_subject_mask(self.original_bgr)
        self.proxy_mask = cv2.resize(full_mask, (self.proxy_bgr.shape[1], self.proxy_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        self._fast_refresh_proxy()
        self.lbl_status.setText("✓ Re-detected subject with GrabCut.")

    def _undo_mask(self) -> None:
        if len(self.undo_stack) > 1:
            self.redo_stack.append(self.undo_stack.pop())
            self.proxy_mask = self.undo_stack[-1].copy()
            self._fast_refresh_proxy()
            self.btn_undo.setEnabled(len(self.undo_stack) > 1)
            self.btn_redo.setEnabled(True)

    def _redo_mask(self) -> None:
        if self.redo_stack:
            restored = self.redo_stack.pop()
            self.undo_stack.append(restored)
            self.proxy_mask = restored.copy()
            self._fast_refresh_proxy()
            self.btn_undo.setEnabled(True)
            self.btn_redo.setEnabled(len(self.redo_stack) > 0)

    def _show_original(self) -> None:
        self.bg_item.setPixmap(_cv2_to_qpixmap(self.proxy_bgr))

    def _restore_color_pop(self) -> None:
        self._fast_refresh_proxy()

    def _save_photo(self) -> None:
        out_path = prompt_save_new_file(self, self.file_path, "colourPop", ".jpg")
        if not out_path:
            return

        self.lbl_status.setText("Rendering high-resolution Colour Pop photo...")
        QApplication.processEvents()

        h, w = self.original_bgr.shape[:2]
        # Upscale proxy mask to full camera resolution
        if self.proxy_scale != 1.0:
            full_mask = cv2.resize(self.proxy_mask, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            full_mask = self.proxy_mask

        # Render full resolution
        full_color_pop = apply_color_pop(
            self.original_bgr,
            full_mask,
            bg_contrast=self.bg_contrast,
            bg_brightness=self.bg_brightness,
        )

        try:
            cv2.imwrite(str(out_path), full_color_pop, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
            self.saved_file_path = out_path
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save photo:\n{e}")


# Backwards compatibility aliases
ColorPopCanvasView = ColourPopCanvasView
ColorPopDialog = ColourPopDialog


# =============================================================================
# 5. Portrait Depth Bokeh & Foreground Fill Light Dialog
# =============================================================================

class PortraitBokehDialog(QDialog):
    """
    Portrait Bokeh & Fill Light Dialog:
    - Background lens blur kernel with real-time slider readout.
    - Soft-knee foreground face/subject fill light.
    - Hold to Compare and instant display proxy rendering.
    """

    def __init__(
        self,
        image_bgr: np.ndarray,
        file_path: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if len(image_bgr.shape) == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        elif image_bgr.shape[2] == 4:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)

        self.original_bgr = image_bgr.copy()
        self.file_path = file_path
        self.output_dir = file_path.parent if file_path else Path.home()
        self.saved_file_path: Path | None = None

        h, w = self.original_bgr.shape[:2]
        max_proxy_dim = 1200
        if max(h, w) > max_proxy_dim:
            self.proxy_scale = max_proxy_dim / float(max(h, w))
            pw = max(1, int(round(w * self.proxy_scale)))
            ph = max(1, int(round(h * self.proxy_scale)))
            self.proxy_bgr = cv2.resize(self.original_bgr, (pw, ph), interpolation=cv2.INTER_AREA)
        else:
            self.proxy_scale = 1.0
            self.proxy_bgr = self.original_bgr.copy()

        full_mask = estimate_subject_mask(self.original_bgr)
        self.proxy_mask = cv2.resize(full_mask, (self.proxy_bgr.shape[1], self.proxy_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        self.rendered_proxy: np.ndarray | None = None
        self._first_fit = True

        self.setWindowTitle("Portrait Depth Bokeh & Fill Light")
        self.setWindowIcon(make_wlpg_icon("noise_reduction", 24))
        self.resize(1020, 700)
        self._init_ui()
        self._render_proxy()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(10)

        # Header
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #fcfdfe, stop:1 #e8eff7);
                border: 1px solid #c0d0e6;
                border-radius: 5px;
                padding: 6px 12px;
            }
        """)
        h_layout = QHBoxLayout(header_frame)
        h_layout.setContentsMargins(4, 2, 4, 2)

        lbl_icon = QLabel()
        lbl_icon.setPixmap(make_wlpg_icon("noise_reduction", 28).pixmap(28, 28))
        h_layout.addWidget(lbl_icon)

        title_vbox = QVBoxLayout()
        lbl_title = QLabel("Portrait Depth Bokeh & Subject Fill Light")
        lbl_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #1e395b;")
        lbl_sub = QLabel("Simulate shallow depth-of-field lens bokeh behind subject with soft portrait studio fill light.")
        lbl_sub.setStyleSheet("color: #555555; font-size: 11px;")
        title_vbox.addWidget(lbl_title)
        title_vbox.addWidget(lbl_sub)
        h_layout.addLayout(title_vbox)
        h_layout.addStretch(1)

        layout.addWidget(header_frame)

        # Controls bar
        bar = QFrame()
        bar.setStyleSheet("background: #f4f6fa; border: 1px solid #d0d7e5; border-radius: 4px; padding: 4px;")
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(8, 4, 8, 4)
        bar_layout.setSpacing(14)

        bar_layout.addWidget(QLabel("Bokeh Blur:"))
        self.sld_blur = QSlider(Qt.Horizontal)
        self.sld_blur.setRange(0, 40)
        self.sld_blur.setValue(16)
        self.sld_blur.setFixedWidth(130)
        self.sld_blur.valueChanged.connect(self._on_blur_changed)
        bar_layout.addWidget(self.sld_blur)

        self.lbl_blur_val = QLabel("16 px")
        self.lbl_blur_val.setStyleSheet("font-weight: bold; color: #333; min-width: 40px;")
        bar_layout.addWidget(self.lbl_blur_val)

        bar_layout.addSpacing(10)
        bar_layout.addWidget(QLabel("Subject Fill Light:"))
        self.sld_fill = QSlider(Qt.Horizontal)
        self.sld_fill.setRange(0, 50)
        self.sld_fill.setValue(22)
        self.sld_fill.setFixedWidth(130)
        self.sld_fill.valueChanged.connect(self._on_fill_changed)
        bar_layout.addWidget(self.sld_fill)

        self.lbl_fill_val = QLabel("22%")
        self.lbl_fill_val.setStyleSheet("font-weight: bold; color: #333; min-width: 40px;")
        bar_layout.addWidget(self.lbl_fill_val)

        bar_layout.addStretch(1)

        self.btn_compare = QPushButton("Hold to Compare")
        self.btn_compare.setStyleSheet("background: #f0f4fa; border: 1px solid #7a96df; font-weight: bold; padding: 4px 10px; border-radius: 3px;")
        self.btn_compare.pressed.connect(self._show_original)
        self.btn_compare.released.connect(self._restore_bokeh)
        bar_layout.addWidget(self.btn_compare)

        layout.addWidget(bar)

        # Canvas
        self.view = ZoomablePreviewCanvas(self)
        self.scene = QGraphicsScene(self)
        self.view.setScene(self.scene)
        self.view.zoom_changed.connect(self._on_zoom_changed)
        self.view.setStyleSheet("background: #252528; border: 1px solid #777; border-radius: 4px;")
        self.bg_item = QGraphicsPixmapItem()
        self.scene.addItem(self.bg_item)
        layout.addWidget(self.view, 1)

        # Bottom buttons
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(QLabel(f"Resolution: {self.original_bgr.shape[1]} × {self.original_bgr.shape[0]} px"))
        btn_layout.addStretch(1)

        # Zoom Controls
        zoom_frame = QFrame()
        zoom_frame.setStyleSheet("background: #f4f6fa; border: 1px solid #c0c8d8; border-radius: 3px; padding: 1px;")
        z_layout = QHBoxLayout(zoom_frame)
        z_layout.setContentsMargins(4, 1, 4, 1)
        z_layout.setSpacing(4)

        self.btn_zoom_out = QPushButton("−")
        self.btn_zoom_out.setFixedSize(24, 22)
        self.btn_zoom_out.setToolTip("Zoom Out (-)")
        self.btn_zoom_out.setStyleSheet("font-size: 13px; font-weight: bold; padding: 0;")
        self.btn_zoom_out.clicked.connect(lambda: self._apply_zoom(1.0 / 1.25))
        z_layout.addWidget(self.btn_zoom_out)

        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.setFixedSize(24, 22)
        self.btn_zoom_in.setToolTip("Zoom In (+)")
        self.btn_zoom_in.setStyleSheet("font-size: 13px; font-weight: bold; padding: 0;")
        self.btn_zoom_in.clicked.connect(lambda: self._apply_zoom(1.25))
        z_layout.addWidget(self.btn_zoom_in)

        self.btn_fit = QPushButton("Fit")
        self.btn_fit.setFixedHeight(22)
        self.btn_fit.setToolTip("Fit photo to canvas window")
        self.btn_fit.clicked.connect(self._fit_to_view)
        z_layout.addWidget(self.btn_fit)

        self.btn_100 = QPushButton("100%")
        self.btn_100.setFixedHeight(22)
        self.btn_100.setToolTip("View at actual 100% pixel size (Ctrl+0)")
        self.btn_100.clicked.connect(self._reset_zoom_100)
        z_layout.addWidget(self.btn_100)

        self.lbl_zoom_val = QLabel("100%")
        self.lbl_zoom_val.setStyleSheet("font-size: 11px; font-weight: bold; color: #333; min-width: 40px; padding-left: 2px;")
        z_layout.addWidget(self.lbl_zoom_val)

        btn_layout.addWidget(zoom_frame)
        btn_layout.addSpacing(12)

        btn_save = QPushButton("Save New Portrait Bokeh Photo...")
        btn_save.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3c82e6, stop:1 #1c52ba);
                color: white; font-weight: bold; border: 1px solid #143e8c;
                border-radius: 4px; padding: 6px 20px; min-width: 130px;
            }
            QPushButton:hover { background: #4d8ff2; }
        """)
        btn_save.clicked.connect(self._save_photo)
        btn_layout.addWidget(btn_save)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_cancel)

        layout.addLayout(btn_layout)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._first_fit and self.scene.sceneRect().width() > 0:
            self._first_fit = False
            self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
            self._on_zoom_changed(self.view.transform().m11())

    def _on_zoom_changed(self, scale_val: float) -> None:
        self.lbl_zoom_val.setText(f"{int(round(scale_val * 100))}%")

    def _apply_zoom(self, factor: float) -> None:
        self.view.scale(factor, factor)
        self._on_zoom_changed(self.view.transform().m11())

    def _reset_zoom_100(self) -> None:
        self.view.resetTransform()
        self._on_zoom_changed(1.0)

    def _fit_to_view(self) -> None:
        if self.scene.sceneRect().isValid() and self.scene.sceneRect().width() > 0:
            self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
            self._on_zoom_changed(self.view.transform().m11())

    def _on_blur_changed(self, val: int) -> None:
        self.lbl_blur_val.setText(f"{val} px")
        self._render_proxy()

    def _on_fill_changed(self, val: int) -> None:
        self.lbl_fill_val.setText(f"{val}%")
        self._render_proxy()

    def _render_proxy(self) -> None:
        b_rad = max(1, int(self.sld_blur.value() * self.proxy_scale))
        fill_val = self.sld_fill.value() / 100.0
        self.rendered_proxy = apply_portrait_bokeh_and_fill_light(
            self.proxy_bgr,
            self.proxy_mask,
            bokeh_radius=b_rad,
            fill_light=fill_val,
        )
        pm = _cv2_to_qpixmap(self.rendered_proxy)
        self.bg_item.setPixmap(pm)
        ph, pw = self.proxy_bgr.shape[:2]
        self.scene.setSceneRect(QRectF(0, 0, pw, ph))

    def _show_original(self) -> None:
        self.bg_item.setPixmap(_cv2_to_qpixmap(self.proxy_bgr))

    def _restore_bokeh(self) -> None:
        if self.rendered_proxy is not None:
            self.bg_item.setPixmap(_cv2_to_qpixmap(self.rendered_proxy))

    def _save_photo(self) -> None:
        out_path = prompt_save_new_file(self, self.file_path, "portraitBokeh", ".jpg")
        if not out_path:
            return

        h, w = self.original_bgr.shape[:2]
        full_mask = cv2.resize(self.proxy_mask, (w, h), interpolation=cv2.INTER_LINEAR) if self.proxy_scale != 1.0 else self.proxy_mask
        full_bokeh = apply_portrait_bokeh_and_fill_light(
            self.original_bgr,
            full_mask,
            bokeh_radius=self.sld_blur.value(),
            fill_light=self.sld_fill.value() / 100.0,
        )

        try:
            cv2.imwrite(str(out_path), full_bokeh, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
            self.saved_file_path = out_path
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save photo:\n{e}")


# =============================================================================
# 6. 3D Cinematic Motion (Faux-3D Parallax MP4 Generator) Dialog
# =============================================================================

class CinematicMotionWorker(QThread):
    progress = Signal(int)
    finished = Signal(bool, str)

    def __init__(self, image_bgr: np.ndarray, out_path: Path, duration: float) -> None:
        super().__init__()
        self.image_bgr = image_bgr
        self.out_path = out_path
        self.duration = duration

    def run(self) -> None:
        try:
            success = render_3d_cinematic_motion_video(
                self.image_bgr,
                self.out_path,
                duration_sec=self.duration,
                fps=30,
                resolution=(1920, 1080),
                progress_callback=self.progress.emit,
            )
            self.finished.emit(success, "Video rendered successfully" if success else "Failed to open video encoder")
        except Exception as e:
            self.finished.emit(False, str(e))


class CinematicMotionDialog(QDialog):
    """
    3D Cinematic Motion Video Generator:
    - Previews animated 3D virtual camera parallax loop in real time.
    - Estimates foreground/background layers with disocclusion background inpainting.
    - Renders smooth 30fps .mp4 virtual camera dolly video.
    """

    def __init__(
        self,
        image_bgr: np.ndarray,
        file_path: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if len(image_bgr.shape) == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        elif image_bgr.shape[2] == 4:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)

        self.original_bgr = image_bgr.copy()
        self.file_path = file_path
        self.output_dir = file_path.parent if file_path else Path.home()
        self.saved_file_path: Path | None = None
        self._worker: CinematicMotionWorker | None = None

        self._preview_frames: list[QPixmap] = []
        self._current_frame_idx: int = 0
        self._anim_direction: int = 1
        self._timer: QTimer | None = None

        self.setWindowTitle("3D Cinematic Motion Video Generator")
        self.setWindowIcon(make_wlpg_icon("slideshow", 24))
        self.resize(780, 520)
        self._init_ui()
        self._build_preview_loop()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(12)

        # Header
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #fcfdfe, stop:1 #e8eff7);
                border: 1px solid #c0d0e6;
                border-radius: 5px;
                padding: 6px 12px;
            }
        """)
        h_layout = QHBoxLayout(header_frame)
        h_layout.setContentsMargins(4, 2, 4, 2)

        lbl_icon = QLabel()
        lbl_icon.setPixmap(make_wlpg_icon("slideshow", 28).pixmap(28, 28))
        h_layout.addWidget(lbl_icon)

        title_vbox = QVBoxLayout()
        lbl_title = QLabel("3D Cinematic Motion Video Generator")
        lbl_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #1e395b;")
        lbl_sub = QLabel("Renders a smooth 3D virtual camera dolly .mp4 video with parallax layer shift (Google Photos style).")
        lbl_sub.setStyleSheet("color: #555555; font-size: 11px;")
        title_vbox.addWidget(lbl_title)
        title_vbox.addWidget(lbl_sub)
        h_layout.addLayout(title_vbox)
        h_layout.addStretch(1)
        layout.addWidget(header_frame)

        # Settings
        frame_set = QFrame()
        frame_set.setStyleSheet("background: #f4f6fa; border: 1px solid #d0d7e5; border-radius: 4px; padding: 6px;")
        set_layout = QHBoxLayout(frame_set)
        set_layout.setContentsMargins(8, 6, 8, 6)
        set_layout.setSpacing(12)

        set_layout.addWidget(QLabel("Duration:"))
        self.cmb_duration = QComboBox()
        self.cmb_duration.addItems(["3 seconds (90 frames)", "5 seconds (150 frames)", "8 seconds (240 frames)"])
        self.cmb_duration.setCurrentIndex(1)
        set_layout.addWidget(self.cmb_duration)

        set_layout.addWidget(QLabel("Resolution: Full HD 1080p (1920 × 1080 @ 30fps)"))
        set_layout.addStretch(1)
        layout.addWidget(frame_set)

        # Live animated preview view
        self.lbl_thumb = QLabel()
        self.lbl_thumb.setAlignment(Qt.AlignCenter)
        self.lbl_thumb.setStyleSheet("background: #1c1c1e; border: 1px solid #666; border-radius: 4px;")
        layout.addWidget(self.lbl_thumb, 1)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("QProgressBar { height: 8px; border: none; background: #e0e0e0; } QProgressBar::chunk { background: #2D72BC; }")
        layout.addWidget(self.progress_bar)

        self.lbl_status = QLabel("Live 3D Parallax Preview (Interactive Loop)")
        self.lbl_status.setStyleSheet("color: #444; font-size: 11px;")
        layout.addWidget(self.lbl_status)

        # Bottom buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch(1)

        self.btn_render = QPushButton("Save New 3D Motion Video (.mp4)...")
        self.btn_render.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3c82e6, stop:1 #1c52ba);
                color: white; font-weight: bold; border: 1px solid #143e8c;
                border-radius: 4px; padding: 7px 22px; min-width: 160px;
            }
            QPushButton:hover { background: #4d8ff2; }
            QPushButton:disabled { background: #ccc; color: #888; }
        """)
        self.btn_render.clicked.connect(self._start_render)
        btn_layout.addWidget(self.btn_render)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self._on_cancel)
        btn_layout.addWidget(btn_cancel)

        layout.addLayout(btn_layout)

    def _build_preview_loop(self) -> None:
        """Precomputes 12 frames on proxy for real-time live preview playback."""
        small = cv2.resize(self.original_bgr, (480, 270), interpolation=cv2.INTER_AREA)
        mask = estimate_subject_mask(small)
        from src.imaging.creative_effects import inpaint_disocclusion_background, render_3d_cinematic_motion_frame
        bg_clean = inpaint_disocclusion_background(small, mask)

        self._preview_frames.clear()
        num_frames = 12
        for i in range(num_frames):
            t = i / float(num_frames - 1)
            frame = render_3d_cinematic_motion_frame(small, bg_clean, mask, t=t, canvas_size=(480, 270))
            self._preview_frames.append(_cv2_to_qpixmap(frame))

        if self._preview_frames:
            self.lbl_thumb.setPixmap(self._preview_frames[0])
            self._timer = QTimer(self)
            self._timer.timeout.connect(self._advance_preview_frame)
            self._timer.start(50)

    def _advance_preview_frame(self) -> None:
        if not self._preview_frames:
            return
        self._current_frame_idx += self._anim_direction
        if self._current_frame_idx >= len(self._preview_frames):
            self._current_frame_idx = len(self._preview_frames) - 2
            self._anim_direction = -1
        elif self._current_frame_idx < 0:
            self._current_frame_idx = 1
            self._anim_direction = 1

        self.lbl_thumb.setPixmap(self._preview_frames[self._current_frame_idx])

    def _start_render(self) -> None:
        out_path = prompt_save_new_file(
            self,
            self.file_path,
            "cinematic3D",
            ".mp4",
            "MP4 Video (*.mp4);;All Files (*.*)",
        )
        if not out_path:
            return

        if self._timer:
            self._timer.stop()
        self.btn_render.setEnabled(False)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Estimating depth layers and inpainting background disocclusion...")

        dur_text = self.cmb_duration.currentText()
        duration = 3.0 if "3" in dur_text else (8.0 if "8" in dur_text else 5.0)

        self._worker = CinematicMotionWorker(self.original_bgr, out_path, duration)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(lambda ok, msg: self._on_finished(ok, msg, out_path))
        self._worker.start()

    def _on_progress(self, val: int) -> None:
        self.progress_bar.setValue(val)
        self.lbl_status.setText(f"Encoding 3D virtual camera frames: {val}%")

    def _on_finished(self, ok: bool, msg: str, out_path: Path) -> None:
        self.btn_render.setEnabled(True)
        if ok and out_path.exists():
            self.saved_file_path = out_path
            self.lbl_status.setText(f"✓ Video saved: {out_path.name}")
            self.accept()
        else:
            QMessageBox.critical(self, "Render Error", f"Could not render 3D video:\n{msg}")
            if self._timer:
                self._timer.start(50)

    def _on_cancel(self) -> None:
        if self._timer:
            self._timer.stop()
        self.reject()

    def closeEvent(self, event) -> None:
        if self._timer:
            self._timer.stop()
        super().closeEvent(event)

