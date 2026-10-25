"""
OpenCV & NumPy high-fidelity imaging pipeline for PhotoGallery64.
"""
from src.imaging.opencv_engine import (
    PhotoAdjustmentRecipe,
    analyze_auto_adjust,
    detect_auto_straighten_angle,
    render_adjusted_image,
    compute_histogram_data,
)

__all__ = [
    "PhotoAdjustmentRecipe",
    "analyze_auto_adjust",
    "detect_auto_straighten_angle",
    "render_adjusted_image",
    "compute_histogram_data",
]
