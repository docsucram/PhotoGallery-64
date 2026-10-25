"""
Test Suite for WP-11: Background Face Detection Worker (tests/test_face_worker.py).

Verifies:
1. Stage 1: Prioritized backfill of existing confirmed face tags (state = 1)
   with missing embeddings or missing 96x96 face thumbnails, plus initial centroid calculation.
2. Stage 2: Background scanning of untagged images (face_scan_version < 1)
   populating media_people (state = 0) and face_thumbnails.
3. Stage 3: Vectorized cosine-similarity matching of unassigned faces
   against confirmed person centroids using FaceVectorMatcher, updating suggested_person_id.
4. Poison-pill protection: corrupt/missing files are marked with face_scan_version = -1
   without crashing the worker.
5. Cancellation flag responsiveness: stopping the worker terminates processing cleanly.
6. Force rescan resets face_scan_version = 0 and scans all images.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from legacy_reference.imgBrowserv13 import CancellationFlag
from src.core.face_engine import DetectedFace, FaceEngine
from src.core.face_matcher import (
    FaceVectorMatcher,
    deserialize_embedding,
    normalize_vector,
    serialize_embedding,
)
from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository, ThumbnailRepository
from src.workers.face_worker import FaceDetectionWorker, FaceWorkerSignals, TARGET_FACE_SCAN_VERSION


@pytest.fixture
def test_engine(tmp_path: Path) -> DualDatabaseEngine:
    engine = DualDatabaseEngine(
        catalog_path=tmp_path / "catalog.db",
        thumbnails_path=tmp_path / "thumbnails.db",
        schema_sql_path=Path("db/schema.sql"),
    )
    engine.initialize_schemas()
    yield engine
    engine.close_all()


class MockFaceEngine(FaceEngine):
    """Deterministic MockFaceEngine for fast, reliable unit testing."""

    def __init__(self, fake_faces_per_image: int = 1) -> None:
        super().__init__(backend="opencv_haar")
        self.fake_faces_per_image = fake_faces_per_image
        # Generate two distinct canonical face vectors
        vec_a = np.zeros((512,), dtype=np.float32)
        vec_a[0:10] = 1.0
        self.vector_alice = normalize_vector(vec_a)

        vec_b = np.zeros((512,), dtype=np.float32)
        vec_b[100:110] = 1.0
        self.vector_bob = normalize_vector(vec_b)

    def detect_and_embed(
        self,
        image: str | Path | np.ndarray | Image.Image,
        orientation: int = 1,
    ) -> list[DetectedFace]:
        # If image filename contains poison, raise exception
        fname = Path(str(image)).name.lower() if isinstance(image, (str, Path)) else ""
        if "poison" in fname:
            raise RuntimeError("Corrupt image decode failure!")

        faces: list[DetectedFace] = []
        for i in range(self.fake_faces_per_image):
            # Alternate between Alice-like and Bob-like embedding
            emb = self.vector_alice if (i % 2 == 0) else self.vector_bob
            faces.append(
                DetectedFace(
                    rect_x=0.1 + 0.3 * i,
                    rect_y=0.1,
                    rect_w=0.2,
                    rect_h=0.2,
                    score=0.95,
                    embedding=emb,
                    crop_webp=b"fake_96x96_webp_avatar_crop",
                )
            )
        return faces

    def extract_crop_and_embed_rect(
        self,
        image: str | Path | np.ndarray | Image.Image,
        rect: tuple[float, float, float, float],
        orientation: int = 1,
    ) -> tuple[np.ndarray, bytes]:
        if "poison" in str(image).lower():
            raise RuntimeError("Corrupt crop extraction failure!")
        return self.vector_alice, b"backfilled_96x96_webp_avatar"


def test_face_worker_stage1_backfill_and_centroids(tmp_path: Path, test_engine: DualDatabaseEngine) -> None:
    """Stage 1 backfills missing embeddings/crops for state = 1 faces and computes centroids."""
    repo = CatalogRepository(test_engine)
    thumb_repo = ThumbnailRepository(test_engine)
    mock_fe = MockFaceEngine()

    folder_dir = tmp_path / "Family"
    folder_dir.mkdir()
    img_path = folder_dir / "Alice_and_kids.jpg"
    img = Image.new("RGB", (300, 300), color="pink")
    img.save(img_path)

    fid = repo.upsert_folder(str(folder_dir), "Family", None, 1700000000)
    mids = repo.bulk_upsert_media(
        [
            {
                "folder_id": fid,
                "filename": img_path.name,
                "ext": "jpg",
                "size": 50000,
                "mtime": 1700000000,
                "width": 300,
                "height": 300,
            }
        ]
    )
    mid = mids[0]

    # Create Alice person and manually assign confirmed face (state = 1) without embedding
    alice_pid = repo.upsert_person("Alice")
    with test_engine.write_lock:
        conn = test_engine.catalog_conn()
        cur = conn.execute(
            """
            INSERT INTO media_people (media_id, person_id, state, is_manual, rect_x, rect_y, rect_w, rect_h)
            VALUES (?, ?, 1, 1, 0.1, 0.1, 0.25, 0.25)
            RETURNING id;
            """,
            (mid, alice_pid),
        )
        mp_id = cur.fetchone()[0]
        conn.commit()

    # Pre-check: embedding and thumbnail are NULL
    detail = repo.get_media_detail(mid)
    assert len(detail["people"]) == 1
    assert thumb_repo.get_face_thumbnail(mp_id) is None

    # Run FaceDetectionWorker
    worker = FaceDetectionWorker(
        engine=test_engine,
        face_engine=mock_fe,
        batch_size=10,
        throttle_ms=0,
    )
    worker.run()

    # Verify Stage 1 backfilled embedding and thumbnail crop!
    conn = test_engine.catalog_conn()
    mp_row = conn.execute("SELECT embedding FROM media_people WHERE id = ?;", (mp_id,)).fetchone()
    assert mp_row["embedding"] is not None
    assert len(mp_row["embedding"]) == 2048  # 512 * 4 bytes float32

    thumb_blob = thumb_repo.get_face_thumbnail(mp_id)
    assert thumb_blob in (b"fake_96x96_webp_avatar_crop", b"backfilled_96x96_webp_avatar")

    # Verify Alice's centroid was calculated!
    centroids = repo.get_people_centroids()
    assert alice_pid in centroids
    alice_centroid = deserialize_embedding(centroids[alice_pid])
    assert alice_centroid.shape == (512,)
    # Similarity with Alice vector should be ~1.0
    sim = float(np.dot(alice_centroid, mock_fe.vector_alice))
    assert sim > 0.99


def test_face_worker_stage2_scan_and_stage3_match(tmp_path: Path, test_engine: DualDatabaseEngine) -> None:
    """Stage 2 scans untagged images and Stage 3 matches unassigned faces to Alice's centroid."""
    repo = CatalogRepository(test_engine)
    thumb_repo = ThumbnailRepository(test_engine)
    mock_fe = MockFaceEngine(fake_faces_per_image=1)

    folder_dir = tmp_path / "Vacation"
    folder_dir.mkdir()
    img1 = folder_dir / "IMG_101.jpg"
    img2 = folder_dir / "IMG_102.jpg"
    for p in (img1, img2):
        Image.new("RGB", (200, 200), color="white").save(p)

    fid = repo.upsert_folder(str(folder_dir), "Vacation", None, 1700000000)
    mids = repo.bulk_upsert_media(
        [
            {
                "folder_id": fid,
                "filename": img1.name,
                "ext": "jpg",
                "size": 10000,
                "mtime": 1700000000,
                "width": 200,
                "height": 200,
            },
            {
                "folder_id": fid,
                "filename": img2.name,
                "ext": "jpg",
                "size": 10000,
                "mtime": 1700000000,
                "width": 200,
                "height": 200,
            },
        ]
    )

    # Create Alice with a pre-existing centroid
    alice_pid = repo.upsert_person("Alice")
    repo.update_person_centroid(alice_pid, serialize_embedding(mock_fe.vector_alice))

    # Run FaceDetectionWorker
    worker = FaceDetectionWorker(
        engine=test_engine,
        face_engine=mock_fe,
        batch_size=10,
        throttle_ms=0,
    )

    progress_events: list[tuple[int, int]] = []
    worker.signals.batch_progressed.connect(lambda c, t: progress_events.append((c, t)))

    suggestions_updated_called = []
    worker.signals.suggestions_updated.connect(lambda: suggestions_updated_called.append(True))

    worker.run()

    # Verify Stage 2 scan: both media items should have face_scan_version = 1
    conn = test_engine.catalog_conn()
    for mid in mids:
        row = conn.execute("SELECT face_scan_version FROM media WHERE id = ?;", (mid,)).fetchone()
        assert row["face_scan_version"] == TARGET_FACE_SCAN_VERSION

    # Detections inserted into media_people with state = 0
    detections = conn.execute("SELECT id, state, suggested_person_id, similarity_score FROM media_people;").fetchall()
    assert len(detections) == 2
    for d in detections:
        assert d["state"] == 0
        # Stage 3 match should suggest Alice with high confidence (> 0.90)
        assert d["suggested_person_id"] == alice_pid
        assert float(d["similarity_score"]) > 0.90

        # Thumbnail crop stored
        crop = thumb_repo.get_face_thumbnail(int(d["id"]))
        assert crop == b"fake_96x96_webp_avatar_crop"

    assert len(progress_events) >= 2
    assert len(suggestions_updated_called) >= 1


def test_face_worker_poison_pill_protection(tmp_path: Path, test_engine: DualDatabaseEngine) -> None:
    """Poison pill or corrupt images are flagged with face_scan_version = -1 without crashing worker."""
    repo = CatalogRepository(test_engine)
    mock_fe = MockFaceEngine()

    folder_dir = tmp_path / "CorruptFolder"
    folder_dir.mkdir()
    good_img = folder_dir / "good.jpg"
    poison_img = folder_dir / "poison_file.jpg"
    missing_img_name = "deleted_on_disk.jpg"

    Image.new("RGB", (100, 100), color="green").save(good_img)
    poison_img.write_bytes(b"invalid_poison_bytes")

    fid = repo.upsert_folder(str(folder_dir), "CorruptFolder", None, 1700000000)
    mids = repo.bulk_upsert_media(
        [
            {"folder_id": fid, "filename": good_img.name, "ext": "jpg", "size": 100, "mtime": 1700000000},
            {"folder_id": fid, "filename": poison_img.name, "ext": "jpg", "size": 100, "mtime": 1700000000},
            {"folder_id": fid, "filename": missing_img_name, "ext": "jpg", "size": 100, "mtime": 1700000000},
        ]
    )

    worker = FaceDetectionWorker(engine=test_engine, face_engine=mock_fe, batch_size=10, throttle_ms=0)
    worker.run()

    conn = test_engine.catalog_conn()
    good_mid, poison_mid, missing_mid = mids[0], mids[1], mids[2]

    # Good file marked face_scan_version = TARGET_FACE_SCAN_VERSION
    assert conn.execute("SELECT face_scan_version FROM media WHERE id = ?;", (good_mid,)).fetchone()[0] == TARGET_FACE_SCAN_VERSION
    # Poison file marked face_scan_version = -1
    assert conn.execute("SELECT face_scan_version FROM media WHERE id = ?;", (poison_mid,)).fetchone()[0] == -1
    # Missing file marked face_scan_version = -1
    assert conn.execute("SELECT face_scan_version FROM media WHERE id = ?;", (missing_mid,)).fetchone()[0] == -1


def test_face_worker_cancellation(tmp_path: Path, test_engine: DualDatabaseEngine) -> None:
    """CancellationFlag halts processing immediately."""
    repo = CatalogRepository(test_engine)
    mock_fe = MockFaceEngine()

    folder_dir = tmp_path / "ManyPhotos"
    folder_dir.mkdir()
    for i in range(10):
        p = folder_dir / f"Photo_{i:02d}.jpg"
        Image.new("RGB", (50, 50), color="yellow").save(p)

    fid = repo.upsert_folder(str(folder_dir), "ManyPhotos", None, 1700000000)
    mids = repo.bulk_upsert_media(
        [
            {
                "folder_id": fid,
                "filename": f"Photo_{i:02d}.jpg",
                "ext": "jpg",
                "size": 500,
                "mtime": 1700000000,
            }
            for i in range(10)
        ]
    )

    cancel = CancellationFlag()
    cancel.set()  # Cancel prior to start

    worker = FaceDetectionWorker(
        engine=test_engine,
        face_engine=mock_fe,
        cancel_flag=cancel,
        batch_size=2,
    )
    worker.run()

    # None of the items should have been processed
    conn = test_engine.catalog_conn()
    rows = conn.execute("SELECT face_scan_version FROM media;").fetchall()
    assert all(r["face_scan_version"] == 0 for r in rows)


def test_face_worker_force_rescan(tmp_path: Path, test_engine: DualDatabaseEngine) -> None:
    """force_rescan=True resets face_scan_version to 0 and re-evaluates all images."""
    repo = CatalogRepository(test_engine)
    mock_fe = MockFaceEngine()

    folder_dir = tmp_path / "RescanFolder"
    folder_dir.mkdir()
    p = folder_dir / "RescanMe.jpg"
    Image.new("RGB", (50, 50), color="gray").save(p)

    fid = repo.upsert_folder(str(folder_dir), "RescanFolder", None, 1700000000)
    mids = repo.bulk_upsert_media(
        [{"folder_id": fid, "filename": p.name, "ext": "jpg", "size": 500, "mtime": 1700000000}]
    )
    mid = mids[0]

    # Pre-mark as face_scan_version = 1
    repo.update_face_scan_version([mid], version=1)

    worker = FaceDetectionWorker(
        engine=test_engine,
        face_engine=mock_fe,
        force_rescan=True,
        batch_size=10,
    )
    worker.run()

    conn = test_engine.catalog_conn()
    row = conn.execute("SELECT face_scan_version FROM media WHERE id = ?;", (mid,)).fetchone()
    assert row["face_scan_version"] == TARGET_FACE_SCAN_VERSION

    # Detected face was inserted
    faces = conn.execute("SELECT id FROM media_people WHERE media_id = ?;", (mid,)).fetchall()
    assert len(faces) == 1


def test_face_worker_with_default_face_engine(tmp_path: Path, test_engine: DualDatabaseEngine) -> None:
    """Verifies FaceDetectionWorker functions out-of-the-box with default FaceEngine (no mocks)."""
    repo = CatalogRepository(test_engine)

    folder_dir = tmp_path / "RealEngineFolder"
    folder_dir.mkdir()
    p = folder_dir / "landscape.jpg"
    # Plain image without faces
    Image.new("RGB", (200, 200), color="blue").save(p)

    fid = repo.upsert_folder(str(folder_dir), "RealEngineFolder", None, 1700000000)
    mids = repo.bulk_upsert_media(
        [{"folder_id": fid, "filename": p.name, "ext": "jpg", "size": 1000, "mtime": 1700000000}]
    )
    mid = mids[0]

    # Initialize worker with default FaceEngine (auto / opencv fallback)
    worker = FaceDetectionWorker(engine=test_engine, batch_size=10, throttle_ms=0)
    worker.run()

    conn = test_engine.catalog_conn()
    row = conn.execute("SELECT face_scan_version FROM media WHERE id = ?;", (mid,)).fetchone()
    assert row["face_scan_version"] == TARGET_FACE_SCAN_VERSION

