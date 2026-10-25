"""
Unit tests for Production Face Upgrades:
- 3-stage anatomical landmark filter (confidence, min size, 5-point landmark geometry)
- Production InsightFace runtime (no Haar fallback in production)
- Negative rejection memory & mark not a face
- Purge unconfirmed suggestions (preserving confirmed state = 1)
- Recalculate signatures & fast rematch
- FaceRecognitionOptionsTab & OptionsDialog
"""

import math
import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from src.core.face_engine import DetectedFace, FaceEngine
from src.core.face_matcher import (
    FaceVectorMatcher,
    deserialize_embedding,
    normalize_vector,
    serialize_embedding,
)
from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository, ThumbnailRepository
from src.ui.options_dialog import FaceRecognitionOptionsTab, OptionsDialog


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def test_db(tmp_path):
    engine = DualDatabaseEngine(tmp_path / "catalog.db", tmp_path / "thumbs.db")
    yield engine
    engine.close_all()


def test_face_engine_production_backend():
    """Verify production engine initializes InsightFace and reports active provider."""
    engine = FaceEngine(allow_fallback=False)
    assert engine.is_available is True
    assert engine.backend_name == "insightface"
    assert engine.model_name in ("buffalo_l", "buffalo_s")
    assert engine.active_provider in ("DmlExecutionProvider", "CPUExecutionProvider")


def test_face_engine_anatomical_filter_simulation(monkeypatch):
    """
    Simulates InsightFace raw detections to verify the 3-stage anatomical filter:
    1. Confidence gate
    2. Minimum pixel size gate
    3. 5-point landmark geometry gate (eyes above mouth, inter-ocular distance)
    """
    engine = FaceEngine(min_size=40, min_score=0.65, allow_fallback=False)

    class DummyRawFace:
        def __init__(self, score, bbox, kps, emb=None):
            self.det_score = score
            self.bbox = np.array(bbox, dtype=np.float32)
            self.kps = np.array(kps, dtype=np.float32) if kps is not None else None
            self.embedding = emb if emb is not None else np.ones((512,), dtype=np.float32)

    # Valid anatomical face
    valid_face = DummyRawFace(
        score=0.85,
        bbox=[100, 100, 200, 200],  # 100x100 px
        kps=[
            [125, 130],  # left eye
            [175, 130],  # right eye
            [150, 155],  # nose
            [130, 180],  # left mouth
            [170, 180],  # right mouth
        ],
    )

    # 1. Low confidence rejection (< 0.65)
    low_conf_face = DummyRawFace(
        score=0.50,
        bbox=[100, 100, 200, 200],
        kps=[[125, 130], [175, 130], [150, 155], [130, 180], [170, 180]],
    )

    # 2. Too small rejection (< 40px)
    too_small_face = DummyRawFace(
        score=0.90,
        bbox=[100, 100, 130, 130],  # 30x30 px
        kps=[[105, 108], [125, 108], [115, 115], [108, 125], [122, 125]],
    )

    # 3. Missing landmarks
    missing_kps_face = DummyRawFace(
        score=0.90,
        bbox=[100, 100, 200, 200],
        kps=None,
    )

    # 4. Inverted face / ceiling / architecture (eyes below mouth)
    inverted_face = DummyRawFace(
        score=0.85,
        bbox=[100, 100, 200, 200],
        kps=[
            [125, 185],  # eyes at bottom!
            [175, 185],
            [150, 155],
            [130, 120],  # mouth at top
            [170, 120],
        ],
    )

    # 5. Non-face geometry / clock tower / logo (inter-ocular distance < 15% of width)
    narrow_eyes_face = DummyRawFace(
        score=0.85,
        bbox=[100, 100, 200, 200],  # width = 100px, 15% is 15px
        kps=[
            [148, 130],  # eye distance is only 4px (< 15px)
            [152, 130],
            [150, 155],
            [130, 180],
            [170, 180],
        ],
    )

    class MockApp:
        def get(self, img):
            return [valid_face, low_conf_face, too_small_face, missing_kps_face, inverted_face, narrow_eyes_face]

    engine._insight_app = MockApp()

    dummy_rgb = np.zeros((640, 640, 3), dtype=np.uint8)
    results = engine.detect_and_embed(dummy_rgb)

    # Only valid_face should pass all 3 gates!
    assert len(results) == 1
    assert results[0].score == 0.85
    assert results[0].rect_w == pytest.approx(100.0 / 640.0)


def _create_dummy_media(repo: CatalogRepository, filename: str = "test.jpg") -> int:
    fid = repo.upsert_folder("D:/Photos", "Photos", None, 1700000000)
    items = [
        {
            "folder_id": fid,
            "filename": filename,
            "ext": "jpg",
            "size": 1000,
            "mtime": 1700000000,
            "date_taken": 1700000000,
            "rating": 0,
            "flag": 0,
        }
    ]
    ids = repo.bulk_upsert_media(items)
    return ids[0]


def test_negative_rejection_memory_and_mark_not_a_face(test_db):
    """Tests add_face_rejection, get_face_rejections, and mark_not_a_face."""
    repo = CatalogRepository(test_db)
    conn = test_db.catalog_conn()

    m_id = _create_dummy_media(repo)
    p_id = repo.upsert_person("Ava")

    # Insert two face detections
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO media_people (media_id, state, rect_x, rect_y, rect_w, rect_h, suggested_person_id) "
        "VALUES (?, 0, 0.1, 0.1, 0.2, 0.2, ?);",
        (m_id, p_id),
    )
    fid1 = cur.lastrowid

    cur.execute(
        "INSERT INTO media_people (media_id, state, rect_x, rect_y, rect_w, rect_h) "
        "VALUES (?, 0, 0.5, 0.5, 0.2, 0.2);",
        (m_id,),
    )
    fid2 = cur.lastrowid
    conn.commit()

    # Reject Ava for fid1
    repo.reject_face_suggestion(fid1, p_id)
    rejections = repo.get_face_rejections()
    assert fid1 in rejections
    assert p_id in rejections[fid1]

    # Verify state = 0 and suggested_person_id is NULL
    row = conn.execute("SELECT state, suggested_person_id FROM media_people WHERE id = ?;", (fid1,)).fetchone()
    assert row["state"] == 0
    assert row["suggested_person_id"] is None

    # Mark fid2 as not a face
    repo.mark_not_a_face(fid2)
    row2 = conn.execute("SELECT state FROM media_people WHERE id = ?;", (fid2,)).fetchone()
    assert row2["state"] == -1


def test_purge_unconfirmed_suggestions_preserves_confirmed(test_db):
    """Verifies that purge_unconfirmed_suggestions deletes state != 1 and resets face_scan_version."""
    repo = CatalogRepository(test_db)
    conn = test_db.catalog_conn()

    m_id = _create_dummy_media(repo)
    repo.update_face_scan_version(m_id, version=2)

    p_id = repo.upsert_person("Alice")

    # Insert confirmed face (state = 1)
    conn.execute(
        "INSERT INTO media_people (media_id, person_id, state, rect_x, rect_y, rect_w, rect_h) "
        "VALUES (?, ?, 1, 0.1, 0.1, 0.2, 0.2);",
        (m_id, p_id),
    )
    # Insert unconfirmed suggestions (state = 0)
    conn.execute(
        "INSERT INTO media_people (media_id, state, rect_x, rect_y, rect_w, rect_h) "
        "VALUES (?, 0, 0.4, 0.4, 0.2, 0.2);",
        (m_id,),
    )
    # Insert non-face row (state = -1)
    conn.execute(
        "INSERT INTO media_people (media_id, state, rect_x, rect_y, rect_w, rect_h) "
        "VALUES (?, -1, 0.7, 0.7, 0.2, 0.2);",
        (m_id,),
    )
    conn.commit()

    # Purge
    purged = repo.purge_unconfirmed_suggestions()
    assert purged == 2

    # Check that confirmed face remains intact
    rows = conn.execute("SELECT id, state FROM media_people;").fetchall()
    assert len(rows) == 1
    assert rows[0]["state"] == 1

    # Check media face_scan_version was reset to 0 for re-scanning
    m_row = conn.execute("SELECT face_scan_version FROM media WHERE id = ?;", (m_id,)).fetchone()
    assert m_row["face_scan_version"] == 0


def test_recalculate_signatures_and_rematch(test_db):
    """Verifies fast recalculate signatures (<1s) matches unassigned faces to person centroids."""
    repo = CatalogRepository(test_db)
    conn = test_db.catalog_conn()

    m_id = _create_dummy_media(repo, "bob_confirmed.jpg")
    m2_id = _create_dummy_media(repo, "bob_unassigned.jpg")
    p_id = repo.upsert_person("Bob")

    # Create a distinct target embedding
    target_vec = np.zeros((512,), dtype=np.float32)
    target_vec[0] = 1.0  # Unit vector along dimension 0
    target_blob = serialize_embedding(target_vec)

    # Confirmed face for Bob with target_vec
    conn.execute(
        "INSERT INTO media_people (media_id, person_id, state, embedding, rect_x, rect_y, rect_w, rect_h) "
        "VALUES (?, ?, 1, ?, 0.1, 0.1, 0.2, 0.2);",
        (m_id, p_id, target_blob),
    )

    # Unassigned face on m2_id with high cosine similarity to Bob
    query_vec = np.zeros((512,), dtype=np.float32)
    query_vec[0] = 0.95
    query_vec[1] = 0.05
    query_vec = normalize_vector(query_vec)
    query_blob = serialize_embedding(query_vec)

    cur = conn.cursor()
    cur.execute(
        "INSERT INTO media_people (media_id, state, embedding, rect_x, rect_y, rect_w, rect_h) "
        "VALUES (?, 0, ?, 0.4, 0.4, 0.2, 0.2);",
        (m2_id, query_blob),
    )
    unassigned_fid = cur.lastrowid
    conn.commit()

    # Recalculate signatures & re-match
    matched_count = repo.recalculate_signatures_and_rematch(min_match_score=0.50)
    assert matched_count == 1

    # Verify unassigned face is now suggested as Bob
    row = conn.execute(
        "SELECT suggested_person_id, similarity_score FROM media_people WHERE id = ?;",
        (unassigned_fid,),
    ).fetchone()
    assert row["suggested_person_id"] == p_id
    assert row["similarity_score"] > 0.90


def test_options_dialog_and_face_tab(qapp, test_db):
    """Verifies FaceRecognitionOptionsTab and OptionsDialog initialize and save settings."""
    dlg = OptionsDialog(engine=test_db, initial_tab="face_recognition")
    assert dlg.tabs.count() >= 2
    assert dlg.face_tab is not None

    tab = dlg.face_tab
    # Change sensitivity sliders
    tab.slider_confidence.setValue(70)
    tab.slider_match.setValue(55)
    tab.slider_size.setValue(48)
    tab.save_settings()

    settings = QSettings("PhotoGallery64", "PhotoGallery64")
    assert pytest.approx(float(settings.value("face_min_score")), 0.01) == 0.70
    assert pytest.approx(float(settings.value("face_match_threshold")), 0.01) == 0.55
    assert int(settings.value("face_min_size")) == 48

    dlg.close()
