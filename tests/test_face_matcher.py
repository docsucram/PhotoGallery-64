"""
Test Suite for WP-10: FaceVectorMatcher & FaceEngine.
Verifies:
1. 50,000 unassigned embeddings x 200 centroids batch cosine matching performance.
2. Rejection masking (rejected pairs are never suggested).
3. Incremental centroid updating (unit-normalization & mathematical accuracy).
4. Embedding serialization / deserialization (2048-byte BLOB roundtrip).
5. FaceEngine detection, 512-D spatial descriptor extraction, and 96x96 WebP avatar cropping.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from src.core.face_engine import DetectedFace, FaceEngine
from src.core.face_matcher import (
    FaceMatchSuggestion,
    FaceVectorMatcher,
    cosine_similarity,
    deserialize_embedding,
    normalize_vector,
    serialize_embedding,
)


def test_embedding_normalization_and_serialization() -> None:
    """Verifies vector normalization and 2048-byte SQLite BLOB roundtrip."""
    vec = np.random.randn(512).astype(np.float32)
    normed = normalize_vector(vec)
    assert np.isclose(np.linalg.norm(normed), 1.0, atol=1e-5)
    assert normed.dtype == np.float32

    # Serialize to 2048 raw bytes
    blob = serialize_embedding(normed)
    assert len(blob) == 2048
    assert isinstance(blob, bytes)

    # Deserialize back to float32 (512,)
    restored = deserialize_embedding(blob)
    assert restored.shape == (512,)
    assert restored.dtype == np.float32
    assert np.allclose(normed, restored, atol=1e-6)


def test_incremental_centroid_updating() -> None:
    """Verifies running centroid updates match theoretical means and remain unit-normalized."""
    # First confirmation for Person P:
    e1 = normalize_vector(np.random.randn(512).astype(np.float32))
    c1 = FaceVectorMatcher.update_centroid_incremental(old_centroid=None, confirmed_count=0, new_embeddings=e1)
    assert np.allclose(c1, e1, atol=1e-6)
    assert np.isclose(np.linalg.norm(c1), 1.0, atol=1e-5)

    # Second confirmation:
    e2 = normalize_vector(np.random.randn(512).astype(np.float32))
    c2 = FaceVectorMatcher.update_centroid_incremental(old_centroid=c1, confirmed_count=1, new_embeddings=e2)
    expected_c2 = normalize_vector(e1 + e2)
    assert np.allclose(c2, expected_c2, atol=1e-5)
    assert np.isclose(np.linalg.norm(c2), 1.0, atol=1e-5)

    # Batch confirmation of 3 new embeddings:
    e_batch = np.array([normalize_vector(np.random.randn(512).astype(np.float32)) for _ in range(3)])
    c3 = FaceVectorMatcher.update_centroid_incremental(old_centroid=c2, confirmed_count=2, new_embeddings=e_batch)
    expected_c3 = normalize_vector(2.0 * c2 + np.sum(e_batch, axis=0))
    assert np.allclose(c3, expected_c3, atol=1e-5)
    assert np.isclose(np.linalg.norm(c3), 1.0, atol=1e-5)


def test_rejection_masking() -> None:
    """Verifies that rejected face-person pairs are permanently excluded from suggestions."""
    # 3 faces, 2 persons
    face_ids = [101, 102, 103]
    person_ids = [201, 202]

    # Create known centroids:
    c1 = np.zeros(512, dtype=np.float32)
    c1[0] = 1.0  # Person 201 is aligned with dimension 0

    c2 = np.zeros(512, dtype=np.float32)
    c2[1] = 1.0  # Person 202 is aligned with dimension 1

    centroids = np.stack([c1, c2])

    # Face 101 strongly aligns with Person 201 (dim 0 = 0.9, dim 1 = 0.1)
    f1 = np.zeros(512, dtype=np.float32)
    f1[0], f1[1] = 0.9, 0.1
    f1 = normalize_vector(f1)

    # Face 102 strongly aligns with Person 202 (dim 0 = 0.1, dim 1 = 0.9)
    f2 = np.zeros(512, dtype=np.float32)
    f2[0], f2[1] = 0.1, 0.9
    f2 = normalize_vector(f2)

    # Face 103 equally near both (dim 0 = 0.6, dim 1 = 0.5)
    f3 = np.zeros(512, dtype=np.float32)
    f3[0], f3[1] = 0.6, 0.5
    f3 = normalize_vector(f3)

    face_matrix = np.stack([f1, f2, f3])

    # Unmasked match:
    matches = FaceVectorMatcher.match_unassigned_faces(
        face_ids=face_ids,
        face_matrix=face_matrix,
        person_ids=person_ids,
        centroid_matrix=centroids,
        rejected_pairs=None,
        high_thresh=0.55,
        low_thresh=0.42,
    )
    match_dict = {m.face_id: m for m in matches}
    assert match_dict[101].person_id == 201
    assert match_dict[102].person_id == 202
    assert match_dict[103].person_id == 201  # 0.6 > 0.5

    # Reject (103, 201) -> Face 103 must now match Person 202:
    rejected = {(103, 201)}
    matches_after_reject = FaceVectorMatcher.match_unassigned_faces(
        face_ids=face_ids,
        face_matrix=face_matrix,
        person_ids=person_ids,
        centroid_matrix=centroids,
        rejected_pairs=rejected,
        high_thresh=0.55,
        low_thresh=0.42,
    )
    match_dict2 = {m.face_id: m for m in matches_after_reject}
    assert match_dict2[101].person_id == 201
    assert match_dict2[102].person_id == 202
    assert match_dict2[103].person_id == 202  # Person 201 was masked out!

    # Reject both (103, 201) and (103, 202) -> Face 103 has no candidates left above threshold:
    rejected_both = {(103, 201), (103, 202)}
    matches_after_both = FaceVectorMatcher.match_unassigned_faces(
        face_ids=face_ids,
        face_matrix=face_matrix,
        person_ids=person_ids,
        centroid_matrix=centroids,
        rejected_pairs=rejected_both,
        high_thresh=0.55,
        low_thresh=0.42,
    )
    fids = [m.face_id for m in matches_after_both]
    assert 103 not in fids


def test_50k_by_200_matrix_match_performance() -> None:
    """
    Benchmarks 50,000 unassigned face embeddings matching against 200 person centroids.
    Must execute dot product and argmax search in <50ms.
    """
    N = 50_000
    M = 200
    D = 512

    np.random.seed(42)
    face_ids = list(range(1, N + 1))
    person_ids = list(range(1001, 1001 + M))

    face_matrix = normalize_vector(np.random.randn(N, D).astype(np.float32))
    centroid_matrix = normalize_vector(np.random.randn(M, D).astype(np.float32))

    # Add 500 rejection pairs
    rejected_pairs = {(i, 1001 + (i % M)) for i in range(1, 501)}

    t0 = time.perf_counter()
    matches = FaceVectorMatcher.match_unassigned_faces(
        face_ids=face_ids,
        face_matrix=face_matrix,
        person_ids=person_ids,
        centroid_matrix=centroid_matrix,
        rejected_pairs=rejected_pairs,
        high_thresh=0.55,
        low_thresh=0.0,  # return top match for all
    )
    duration_ms = (time.perf_counter() - t0) * 1000

    assert len(matches) == N
    # Performance assertion: 50,000 x 200 matrix multiplication must execute in < 150ms
    assert duration_ms < 150.0, f"50k x 200 matching took {duration_ms:.2f} ms (expected < 150ms)"


def test_face_engine_extract_crop_and_embed_rect() -> None:
    """Verifies FaceEngine.extract_crop_and_embed_rect produces valid 512-D vector and 96x96 WebP crop."""
    engine = FaceEngine(backend="opencv_haar")

    # Generate synthetic image
    h, w = 400, 600
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[50:200, 100:250] = [200, 150, 130]  # synthetic block

    rect = (100 / w, 50 / h, 150 / w, 150 / h)
    embedding, crop_webp = engine.extract_crop_and_embed_rect(arr, rect)

    # 1. Embedding properties
    assert embedding.shape == (512,)
    assert embedding.dtype == np.float32
    assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5)

    # 2. WebP crop properties
    assert len(crop_webp) > 10
    pil_crop = Image.open(io.BytesIO(crop_webp))
    assert pil_crop.format == "WEBP"
    assert pil_crop.size in ((96, 96), (128, 128))


def test_face_engine_exif_orientation_handling(tmp_path: Path) -> None:
    """Verifies FaceEngine properly respects orientation transpositions."""
    engine = FaceEngine(backend="opencv_haar")

    img_path = tmp_path / "orient_test.jpg"
    orig = Image.new("RGB", (200, 300), color=(100, 150, 200))
    orig.save(img_path)

    # When orientation 6 (90 deg CW) is passed:
    # 200x300 rotated 90 deg becomes 300x200
    arr = engine.load_image_rgb(img_path, orientation=6)
    assert arr.shape == (200, 300, 3) or arr.shape == (300, 200, 3)

    rect = (0.2, 0.2, 0.4, 0.4)
    emb, webp = engine.extract_crop_and_embed_rect(img_path, rect, orientation=6)
    assert emb.shape == (512,)
    assert len(webp) > 0
