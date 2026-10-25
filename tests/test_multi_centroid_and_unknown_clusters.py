"""
Verification tests for Multi-Era Sub-Centroids, Unknown Face Clustering, and Production Fallback Safeguard.
Covers:
1. FaceEngine production fallback safeguard (only allow dummy fallback when allow_fallback=True).
2. Multi-era sub-centroids matching multi-modal aging features (>0.85 similarity vs <0.60 single mean).
3. Fast greedy clustering of unknown faces into recurring clusters + noise in 'Other Unnamed Faces'.
4. BatchPeopleView inline naming for unknown person clusters.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from src.core.face_engine import FaceEngine
from src.core.face_matcher import (
    FaceMatchSuggestion,
    FaceVectorMatcher,
    cosine_similarity,
    deserialize_embedding,
    deserialize_centroids,
    normalize_vector,
    serialize_embedding,
)
from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository, ThumbnailRepository
from src.ui.batch_people_view import BatchPeopleView, _PersonGroupSectionWidget


@pytest.fixture(scope="session")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_production_fallback_safeguard():
    """
    Verifies FaceEngine does NOT fall back to OpenCV Haar / histogram descriptor in production
    when allow_fallback=False (default). Protects media_people.embedding ArcFace vector integrity.
    """
    # 1. Default production initialization:
    engine_prod = FaceEngine(backend="auto", allow_fallback=False)
    if engine_prod.backend_name != "insightface":
        assert not engine_prod.is_available
        assert engine_prod.detect_and_embed(np.zeros((100, 100, 3), dtype=np.uint8)) == []
        emb, _ = engine_prod.extract_crop_and_embed_rect(
            np.zeros((100, 100, 3), dtype=np.uint8), (0.1, 0.1, 0.5, 0.5)
        )
        assert np.allclose(emb, 0.0)

    # 2. Explicit test fallback initialization:
    engine_test = FaceEngine(backend="opencv_haar", allow_fallback=True)
    assert engine_test.is_available
    assert engine_test.backend_name == "opencv_haar"


def test_multi_era_sub_centroids_vs_single_mean():
    """
    Verifies that multi-era sub-centroids (K = 3) match both cluster modes with > 0.85 similarity
    where a single global mean vector drops below 0.60.
    """
    np.random.seed(42)
    D = 512

    # Mode 1: Infant/Toddler era
    mode1 = np.zeros(D, dtype=np.float32)
    mode1[0] = 1.0

    # Mode 2: Teen/Adult era
    mode2 = np.zeros(D, dtype=np.float32)
    mode2[1] = 1.0

    # Mode 3: Senior era
    mode3 = np.zeros(D, dtype=np.float32)
    mode3[2] = 1.0

    # 5 samples per mode with small spherical noise
    samples_mode1 = [normalize_vector(mode1 + 0.15 * normalize_vector(np.random.randn(D).astype(np.float32))) for _ in range(5)]
    samples_mode2 = [normalize_vector(mode2 + 0.15 * normalize_vector(np.random.randn(D).astype(np.float32))) for _ in range(5)]
    samples_mode3 = [normalize_vector(mode3 + 0.15 * normalize_vector(np.random.randn(D).astype(np.float32))) for _ in range(5)]
    all_samples = np.stack(samples_mode1 + samples_mode2 + samples_mode3, axis=0)  # (15, 512)

    # 1. Single Global Mean Vector
    single_mean = normalize_vector(np.mean(all_samples, axis=0, keepdims=True))  # (1, 512)

    # Test query faces from Mode 1 and Mode 2:
    query_mode1 = normalize_vector(mode1 + 0.10 * normalize_vector(np.random.randn(D).astype(np.float32)))
    query_mode2 = normalize_vector(mode2 + 0.10 * normalize_vector(np.random.randn(D).astype(np.float32)))

    sim_mean_m1 = float(np.dot(query_mode1, single_mean.flatten()))
    sim_mean_m2 = float(np.dot(query_mode2, single_mean.flatten()))

    # Single mean vector has similarity < 0.60 (~0.577)
    assert sim_mean_m1 < 0.60, f"Expected single mean similarity < 0.60, got {sim_mean_m1:.3f}"
    assert sim_mean_m2 < 0.60, f"Expected single mean similarity < 0.60, got {sim_mean_m2:.3f}"

    # 2. Multi-Era Sub-Centroids (K = 3)
    sub_centroids = FaceVectorMatcher.compute_sub_centroids(all_samples, k=3)
    assert sub_centroids.shape == (3, 512)

    # Test matching using FaceVectorMatcher
    queries = np.stack([query_mode1, query_mode2], axis=0)
    suggestions = FaceVectorMatcher.match_unassigned_faces(
        face_ids=[101, 102],
        face_matrix=queries,
        person_ids=[1],
        centroid_matrix=sub_centroids,
        high_thresh=0.70,
        low_thresh=0.40,
    )

    assert len(suggestions) == 2
    assert suggestions[0].person_id == 1
    assert suggestions[1].person_id == 1
    assert suggestions[0].score > 0.85, f"Expected score > 0.85 for mode 1, got {suggestions[0].score:.3f}"
    assert suggestions[1].score > 0.85, f"Expected score > 0.85 for mode 2, got {suggestions[1].score:.3f}"


def test_cluster_unnamed_faces_synthetic():
    """
    Verifies that cluster_unnamed_faces groups synthetic embeddings of 3 unknown individuals
    into 3 distinct clusters and leaves singleton noise in 'Other Unnamed Faces'.
    """
    np.random.seed(123)
    D = 512

    # 3 distinct individuals
    person_A_seed = np.zeros(D, dtype=np.float32)
    person_A_seed[10] = 1.0

    person_B_seed = np.zeros(D, dtype=np.float32)
    person_B_seed[20] = 1.0

    person_C_seed = np.zeros(D, dtype=np.float32)
    person_C_seed[30] = 1.0

    # 4 faces for Person A (mutual sim > 0.85)
    fids_A = [1, 2, 3, 4]
    embs_A = [normalize_vector(person_A_seed + 0.1 * normalize_vector(np.random.randn(D).astype(np.float32))) for _ in fids_A]

    # 4 faces for Person B (mutual sim > 0.85)
    fids_B = [5, 6, 7, 8]
    embs_B = [normalize_vector(person_B_seed + 0.1 * normalize_vector(np.random.randn(D).astype(np.float32))) for _ in fids_B]

    # 3 faces for Person C (mutual sim > 0.85)
    fids_C = [9, 10, 11]
    embs_C = [normalize_vector(person_C_seed + 0.1 * normalize_vector(np.random.randn(D).astype(np.float32))) for _ in fids_C]

    # 2 singleton noise faces (random directions)
    fids_noise = [98, 99]
    embs_noise = [normalize_vector(np.random.randn(D).astype(np.float32)) for _ in fids_noise]

    all_fids = fids_A + fids_B + fids_C + fids_noise
    all_embs = np.stack(embs_A + embs_B + embs_C + embs_noise, axis=0)

    clusters = FaceVectorMatcher.cluster_unnamed_faces(
        face_ids=all_fids,
        embeddings=all_embs,
        sim_threshold=0.58,
        min_cluster_size=3,
    )

    # Must produce 3 distinct unknown clusters and 1 Other Unnamed Faces
    assert "Unknown Person #1" in clusters
    assert "Unknown Person #2" in clusters
    assert "Unknown Person #3" in clusters
    assert "Other Unnamed Faces" in clusters

    # Check that fids_A, fids_B, fids_C are partitioned into the 3 unknown clusters
    cluster_sets = [
        set(clusters["Unknown Person #1"]),
        set(clusters["Unknown Person #2"]),
        set(clusters["Unknown Person #3"]),
    ]

    set_A = set(fids_A)
    set_B = set(fids_B)
    set_C = set(fids_C)

    assert set_A in cluster_sets, f"Set A not found in clusters: {cluster_sets}"
    assert set_B in cluster_sets, f"Set B not found in clusters: {cluster_sets}"
    assert set_C in cluster_sets, f"Set C not found in clusters: {cluster_sets}"

    # Check that singleton noise is in 'Other Unnamed Faces'
    noise_set = set(clusters["Other Unnamed Faces"])
    assert set(fids_noise) == noise_set


def test_batch_people_view_unknown_cluster_inline_naming(qapp, tmp_path):
    """
    Verifies that BatchPeopleView displays unknown clusters with an inline naming box
    and allows naming the cluster in one step, updating the person and centroid in SQLite.
    """
    cat_db = tmp_path / "catalog.db"
    thm_db = tmp_path / "thumbnails.db"
    engine = DualDatabaseEngine(
        catalog_path=cat_db,
        thumbnails_path=thm_db,
        schema_sql_path=Path("db/schema.sql"),
    )
    engine.initialize_schemas()

    repo = CatalogRepository(engine)
    thumb_repo = ThumbnailRepository(engine)

    # Insert test folder and media
    folder_id = repo.upsert_folder(str(tmp_path), "test_dir", None, 1000)
    mids = repo.bulk_upsert_media(
        [{"folder_id": folder_id, "filename": "test.jpg", "ext": "jpg", "size": 1000, "mtime": 1700000000}]
    )
    media_id = mids[0]

    # Insert 3 unassigned faces for an unknown person
    with engine.write_lock:
        conn = engine.catalog_conn()
        D = 512
        seed = np.zeros(D, dtype=np.float32)
        seed[50] = 1.0

        for i in range(3):
            emb = normalize_vector(seed + 0.05 * normalize_vector(np.random.randn(D).astype(np.float32)))
            conn.execute(
                """
                INSERT INTO media_people (media_id, person_id, rect_x, rect_y, rect_w, rect_h, state, embedding)
                VALUES (?, NULL, 0.1, 0.1, 0.2, 0.2, 0, ?);
                """,
                (media_id, serialize_embedding(emb)),
            )
        conn.commit()

    view = BatchPeopleView(engine)
    view.refresh_suggestions()

    # Find the section for Unknown Person #1
    sec = None
    for k, s in view._sections.items():
        if "Unknown Person" in s.person_name:
            sec = s
            break

    assert sec is not None, "Unknown Person section was not created"
    assert sec.is_unknown_cluster
    assert sec.txt_name is not None
    assert sec.btn_save_name is not None

    # Name the cluster "Grandma Alice"
    sec.txt_name.setText("Grandma Alice")
    sec._on_save_name_clicked()

    # Verify Grandma Alice exists in DB and has confirmed_count = 3
    alice_row = engine.catalog_conn().execute(
        "SELECT id, name, confirmed_count, centroid_embedding FROM people WHERE name = 'Grandma Alice';"
    ).fetchone()

    assert alice_row is not None
    assert int(alice_row["confirmed_count"]) == 3
    assert alice_row["centroid_embedding"] is not None
    assert len(bytes(alice_row["centroid_embedding"])) >= 2048

    # Verify media_people rows now have person_id = alice_row["id"] and state = 1
    mp_rows = engine.catalog_conn().execute(
        "SELECT state, person_id FROM media_people WHERE person_id = ?;", (alice_row["id"],)
    ).fetchall()
    assert len(mp_rows) == 3
    assert all(r["state"] == 1 for r in mp_rows)
    engine.close_all()
