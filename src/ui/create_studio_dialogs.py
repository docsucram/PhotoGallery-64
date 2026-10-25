"""
Creation Studio Dialogs:
Authentic Windows Live Photo Gallery Create Tab Studio Dialogs:
1. PanoramaDialog — Microsoft ICE-Style Panorama Stitcher with auto-crop of wavy black borders & gap inpainting.
2. CollageDialog — Interactive QGraphicsView Collage & 'Then & Now' Studio (Grid, Polaroid Scatter, and Then & Now).
3. PhotoFuseDialog — Best-Face Group Shot Combiner with burst homography alignment and seamless Poisson face cloning.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any
import cv2
import numpy as np

from PySide6.QtCore import Qt, QThread, Signal, QRectF, QPointF
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
from src.ui.ribbon_toolbar import make_wlpg_icon


# =============================================================================
# Helper Utilities
# =============================================================================

def _cv2_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    """Converts a BGR numpy array to QPixmap."""
    if img_bgr is None or img_bgr.size == 0:
        return QPixmap()
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def _qcolor_to_bgr(color: QColor) -> tuple[int, int, int]:
    return color.blue(), color.green(), color.red()


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

        self.btn_save = QPushButton("Save Panorama")
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

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out_name = f"Panorama_{timestamp}.jpg"
        out_path = self.output_dir / out_name

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

        self.btn_bg_color = QPushButton("Pick Background Color")
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
        self.btn_save = QPushButton("Save Collage")
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
        col = QColorDialog.getColor(self.bg_color, self, "Select Background Color")
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

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out_name = f"Collage_{timestamp}.jpg"
        out_path = self.output_dir / out_name

        try:
            cv2.imwrite(str(out_path), self.rendered_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
            self.saved_file_path = out_path
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save collage:\n{e}")


# =============================================================================
# 3. Photo Fuse Dialog (Best-Face Group Shot Combiner)
# =============================================================================

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

        self.btn_save = QPushButton("Save Fused Photo")
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

    def _align_and_detect(self) -> None:
        """Loads and aligns all secondary burst photos to base frame 0."""
        # Load photos
        for p in self.photo_paths:
            img = cv2.imread(str(p))
            if img is not None:
                self.frames_bgr.append(img)

        if len(self.frames_bgr) < 2:
            self.lbl_status.setText("Failed: Need at least 2 photos to fuse.")
            return

        base = self.frames_bgr[0]
        self.aligned_frames_bgr.append(base.copy())
        self.base_fused_bgr = base.copy()

        # Align remaining photos to base frame
        for i, target in enumerate(self.frames_bgr[1:], start=2):
            self.lbl_status.setText(f"Aligning burst shot {i} of {len(self.frames_bgr)}...")
            success, aligned = align_burst_frame(base, target)
            self.aligned_frames_bgr.append(aligned)

        # Detect faces in base image
        self.lbl_status.setText("Detecting faces...")
        self.face_boxes = detect_faces_in_image(self.base_fused_bgr)

        # If no faces detected automatically, create default candidate grid
        if not self.face_boxes:
            h, w = base.shape[:2]
            cx, cy = w // 2, h // 2
            fw, fh = int(w * 0.2), int(h * 0.25)
            self.face_boxes = [(cx - fw // 2, cy - fh // 2, fw, fh)]

        self.lbl_status.setText(f"Found {len(self.face_boxes)} faces across {len(self.frames_bgr)} shots. Ready to fuse.")
        self.btn_save.setEnabled(True)
        self._update_scene()

    def _update_scene(self) -> None:
        if self.base_fused_bgr is None:
            return

        # Render base image
        pm = _cv2_to_qpixmap(self.base_fused_bgr)
        self.bg_item.setPixmap(pm)
        self.scene.setSceneRect(QRectF(pm.rect()))
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

        # Remove existing face overlay items
        for item in self.scene.items():
            if isinstance(item, QGraphicsRectItem) and item != self.bg_item:
                self.scene.removeItem(item)

        # Draw interactive face rectangles
        for i, (fx, fy, fw, fh) in enumerate(self.face_boxes):
            rect_item = QGraphicsRectItem(QRectF(fx, fy, fw, fh))
            is_sel = (i == self.selected_face_idx)
            pen = QPen(QColor(0, 180, 255) if is_sel else QColor(255, 255, 255, 180))
            pen.setWidth(3 if is_sel else 2)
            rect_item.setPen(pen)
            rect_item.setBrush(QBrush(QColor(0, 180, 255, 35) if is_sel else QColor(0, 0, 0, 0)))
            rect_item.setCursor(QCursor(Qt.PointingHandCursor))
            rect_item.setData(0, i)

            # Connect mouse press via custom event filter or mousePressEvent
            def _make_press(face_idx: int):
                return lambda event: self._select_face(face_idx)

            rect_item.mousePressEvent = _make_press(i)
            self.scene.addItem(rect_item)

    def _select_face(self, face_idx: int) -> None:
        self.selected_face_idx = face_idx
        self._update_scene()
        self._populate_face_strip(face_idx)

    def _populate_face_strip(self, face_idx: int) -> None:
        # Clear existing buttons
        while self.strip_content_layout.count() > 1:
            child = self.strip_content_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        fx, fy, fw, fh = self.face_boxes[face_idx]
        self.lbl_strip_info.setText(f"Expressions for Person {face_idx + 1} (Choose to replace in photo):")

        # Add card for each burst shot
        for shot_idx, aligned_img in enumerate(self.aligned_frames_bgr):
            h, w = aligned_img.shape[:2]
            # Extract crop with slight margin
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

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out_name = f"PhotoFuse_{timestamp}.jpg"
        out_path = self.output_dir / out_name

        try:
            cv2.imwrite(str(out_path), self.base_fused_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
            self.saved_file_path = out_path
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save photo:\n{e}")
