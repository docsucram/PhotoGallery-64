"""
Core algorithmic and image intelligence engines for PhotoGallery64:
- FaceVectorMatcher (in-memory zero-latency vectorized cosine matcher)
- FaceEngine (InsightFace ArcFace / OpenCV Haar face detector & embedder)
"""

from src.core.face_matcher import (
    FaceMatchSuggestion,
    FaceVectorMatcher,
    cosine_similarity,
    deserialize_embedding,
    normalize_vector,
    serialize_embedding,
)
from src.core.face_engine import DetectedFace, FaceEngine

__all__ = [
    "FaceMatchSuggestion",
    "FaceVectorMatcher",
    "DetectedFace",
    "FaceEngine",
    "normalize_vector",
    "serialize_embedding",
    "deserialize_embedding",
    "cosine_similarity",
]
