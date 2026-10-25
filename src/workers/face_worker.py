"""
Background Face Detection & Recognition Worker for PhotoGallery64.
Executes prioritized backfill of existing MP:RegionInfo face tags (Stage 1),
background face detection on untagged photos in low-priority thread pool (Stage 2),
and vectorized cosine-similarity clustering/matching against known person centroids (Stage 3).
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from src.core.face_engine import DetectedFace, FaceEngine, compute_iou
from src.core.face_matcher import (
    FaceVectorMatcher,
    cosine_similarity,
    deserialize_embedding,
    normalize_vector,
    serialize_embedding,
)
from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository, ThumbnailRepository, _iter_id_chunks

logger = logging.getLogger(__name__)

TARGET_FACE_SCAN_VERSION: int = 2


def get_box_rotation_candidates(
    rect: tuple[float, float, float, float],
) -> list[tuple[float, float, float, float]]:
    """
    Generates candidate bounding boxes under 90° CW, 90° CCW, 180° rotations
    and center-offset reversals to test against unclaimed face detections.
    """
    x, y, w, h = rect
    raw_cands = [
        (1.0 - (y + h), x, h, w),              # 90° CW
        (y, 1.0 - (x + w), h, w),              # 90° CCW
        (1.0 - (x + w), 1.0 - (y + h), w, h),  # 180°
        (x - (w / 2.0), y - (h / 2.0), w, h),  # Center coords mistakenly treated as top-left
        (x + (w / 2.0), y + (h / 2.0), w, h),  # Double shift reversal
    ]
    clamped: list[tuple[float, float, float, float]] = []
    for cx, cy, cw, ch in raw_cands:
        clamped.append(
            (
                max(0.0, min(1.0, cx)),
                max(0.0, min(1.0, cy)),
                max(0.0, min(1.0 - max(0.0, cx), cw)),
                max(0.0, min(1.0 - max(0.0, cy), ch)),
            )
        )
    return clamped


class FaceWorkerSignals(QObject):
    """Signals emitted by FaceDetectionWorker."""

    batch_progressed = Signal(int, int)  # (current_processed, total_to_process)
    suggestions_updated = Signal()       # Emitted when new face suggestions/centroids are committed
    worker_finished = Signal(int)        # (total_faces_detected)
    status_message = Signal(str)         # Human-readable progress description


class FaceDetectionWorker(QRunnable):
    """
    QRunnable worker executing background AI face detection and matching.
    Designed for single-worker execution (maxThreadCount = 1, LowPriority).

    Stages:
    - Stage 1: Prioritized backfill for existing confirmed face tags (state = 1)
               with missing embeddings or missing 96x96 face thumbnails, plus
               initial centroid calculation.
    - Stage 2: Background scanning of untagged media (face_scan_version < 1)
               populating media_people (state = 0) and face_thumbnails.
    - Stage 3: Vectorized cosine-similarity matching of unassigned faces
               against confirmed person centroids using FaceVectorMatcher.
    """

    def __init__(
        self,
        engine: DualDatabaseEngine,
        face_engine: FaceEngine | None = None,
        cancel_flag: Any | None = None,
        batch_size: int = 50,
        throttle_ms: int = 5,
        force_rescan: bool = False,
        min_match_score: float = 0.42,
        high_match_score: float = 0.55,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.face_engine = face_engine or FaceEngine()
        self.cancel_flag = cancel_flag
        self.batch_size = max(1, int(batch_size))
        self.throttle_ms = max(0, int(throttle_ms))
        self.force_rescan = force_rescan
        self.min_match_score = float(min_match_score)
        self.high_match_score = float(high_match_score)

        self.signals = FaceWorkerSignals()
        self.repo = CatalogRepository(engine)
        self.thumb_repo = ThumbnailRepository(engine)

    def _is_cancelled(self) -> bool:
        """Checks whether cancellation has been requested via the cancel flag."""
        if self.cancel_flag is None:
            return False
        if getattr(self.cancel_flag, "is_set", False):
            return True
        check_fn = getattr(self.cancel_flag, "is_cancelled", None)
        if callable(check_fn) and check_fn():
            return True
        return False

    @Slot()
    def run(self) -> None:
        """Main worker execution entrypoint."""
        total_faces_detected = 0
        try:
            if self._is_cancelled():
                return

            if self.force_rescan:
                self._reset_scan_versions()

            # Stage 1: Prioritized backfill of confirmed faces & centroid initialization
            self._run_stage1_backfill()

            if self._is_cancelled():
                return

            # Run Stage 3 matching immediately in case all photos were already scanned
            # or new confirmed people centroids were just computed in Stage 1
            self._run_stage3_match()

            if self._is_cancelled():
                return

            # Stage 2: Background scanning of untagged images & Stage 3 suggestion updates
            total_faces_detected = self._run_stage2_scan()

        except Exception as exc:
            logger.exception("FaceDetectionWorker encountered an unhandled error: %s", exc)
        finally:
            try:
                self.signals.worker_finished.emit(total_faces_detected)
            except Exception:
                pass
            # Release thread-local SQLite connections to avoid connection leaks
            self.engine.close_thread_connections()

    # -------------------------------------------------------------------------
    # Stage 1: Prioritized Backfill of Confirmed Faces
    # -------------------------------------------------------------------------
    def _run_stage1_backfill(self) -> None:
        """
        Backfills 512-D embeddings and 96x96 WebP avatar crops for existing
        confirmed face tags (e.g. imported from WLPG MP:RegionInfo or manual tags).
        Verifies confirmed regions against actual detected faces, heals misaligned
        or rotated boxes, prevents centroid poisoning from empty crops, and calculates
        initial person centroids for all people with confirmed faces.
        """
        conn = self.engine.catalog_conn()
        thumb_conn = self.engine.thumbnails_conn()

        # Query all confirmed faces that are missing embeddings or on media scanned prior to version 2
        missing_emb_rows = conn.execute(
            """
            SELECT mp.id, mp.media_id, mp.person_id, mp.rect_x, mp.rect_y, mp.rect_w, mp.rect_h,
                   m.filename, m.orientation, f.path AS folder_path
            FROM media_people mp
            JOIN media m ON m.id = mp.media_id
            JOIN folders f ON f.id = m.folder_id
            WHERE mp.state = 1
              AND m.is_missing = 0
              AND (mp.embedding IS NULL OR (m.face_scan_version >= 0 AND m.face_scan_version < 2))
            ORDER BY mp.id ASC;
            """
        ).fetchall()

        # Query existing face thumbnail IDs
        thumb_rows = thumb_conn.execute("SELECT face_id FROM face_thumbnails;").fetchall()
        existing_thumb_ids = {int(r["face_id"]) for r in thumb_rows}

        # Also find confirmed faces missing thumbnail crops
        all_confirmed = conn.execute(
            """
            SELECT mp.id, mp.media_id, mp.person_id, mp.rect_x, mp.rect_y, mp.rect_w, mp.rect_h,
                   mp.embedding, m.filename, m.orientation, f.path AS folder_path
            FROM media_people mp
            JOIN media m ON m.id = mp.media_id
            JOIN folders f ON f.id = m.folder_id
            WHERE mp.state = 1
              AND m.is_missing = 0
            ORDER BY mp.id ASC;
            """
        ).fetchall()

        # Combine items needing embedding or thumbnail grouped by media_id
        media_groups: dict[int, list[dict[str, Any]]] = {}
        for r in missing_emb_rows:
            media_groups.setdefault(int(r["media_id"]), []).append(dict(r))

        for r in all_confirmed:
            mp_id = int(r["id"])
            if mp_id not in existing_thumb_ids:
                mid = int(r["media_id"])
                if mid not in media_groups or not any(x["id"] == mp_id for x in media_groups[mid]):
                    media_groups.setdefault(mid, []).append(dict(r))

        if media_groups:
            self.signals.status_message.emit(
                f"Stage 1: Verifying and healing confirmed faces in {len(media_groups)} photos..."
            )

            for mid, m_items in media_groups.items():
                if self._is_cancelled():
                    return

                first = m_items[0]
                fpath = Path(first["folder_path"]) / str(first["filename"])
                if not fpath.exists():
                    continue

                orient = int(first["orientation"] or 1)
                try:
                    detected_faces = self.face_engine.detect_and_embed(
                        str(fpath),
                        orientation=orient,
                    )
                except Exception as exc:
                    logger.warning("Stage 1: face detection failed for %s: %s", fpath, exc)
                    from src.utils.scan_logger import log_scan_error
                    log_scan_error("FaceDetectionWorker (Stage 1)", fpath, exc)
                    self.repo.mark_face_scan_failed([mid])
                    detected_faces = []

                self._verify_and_heal_image_faces(mid, fpath, orient, detected_faces)

        # Build initial person centroids from confirmed faces
        self._recompute_people_centroids()

    def _delete_overlapping_unassigned(
        self, conn: Any, media_id: int, d_box: tuple[float, float, float, float]
    ) -> None:
        """Deletes any unassigned or suggested face detections on this media that overlap with a confirmed face."""
        unassigned_rows = conn.execute(
            "SELECT id, rect_x, rect_y, rect_w, rect_h FROM media_people WHERE media_id = ? AND state IN (0, 2);",
            (media_id,),
        ).fetchall()
        to_delete = []
        for u_row in unassigned_rows:
            u_box = (
                float(u_row["rect_x"]),
                float(u_row["rect_y"]),
                float(u_row["rect_w"]),
                float(u_row["rect_h"]),
            )
            if compute_iou(u_box, d_box) >= 0.35:
                to_delete.append(int(u_row["id"]))
        if to_delete:
            placeholders = ",".join("?" for _ in to_delete)
            with self.engine.write_lock:
                conn.execute(f"DELETE FROM media_people WHERE id IN ({placeholders});", to_delete)
                conn.commit()
            self.thumb_repo.invalidate_face_thumbnails(to_delete)

    def _verify_and_heal_image_faces(
        self,
        media_id: int,
        fpath: Path,
        orientation: int,
        detected_faces: list[DetectedFace],
    ) -> list[DetectedFace]:
        """
        Verifies confirmed (state = 1) face tags on media_id against detected faces on the upright image.
        - Case A (IoU >= 0.25): Confirms box, updates d.embedding, saves thumbnail, includes in centroid.
        - Case B (IoU < 0.15): Does NOT add non-face crop to centroid (embedding = NULL).
          Tests candidate box rotations (90° CW, 90° CCW, 180°, center-offset reversal) for IoU >= 0.25.
          If exactly 1 misplaced tag and 1 unclaimed face (or cosine similarity >= 0.40 against centroid),
          auto-snaps box to detected face, updates thumbnail, and marks metadata_dirty = 1 on media.
        Returns the list of remaining unclaimed DetectedFace objects.
        """
        conn = self.engine.catalog_conn()
        confirmed_rows = conn.execute(
            """
            SELECT mp.id, mp.person_id, mp.rect_x, mp.rect_y, mp.rect_w, mp.rect_h, mp.embedding,
                   p.name AS person_name, p.centroid_embedding
            FROM media_people mp
            LEFT JOIN people p ON p.id = mp.person_id
            WHERE mp.media_id = ? AND mp.state = 1
            ORDER BY mp.id ASC;
            """,
            (media_id,),
        ).fetchall()

        if not confirmed_rows:
            return detected_faces

        claimed_d_indices: set[int] = set()
        unmatched_rows: list[dict[str, Any]] = []

        # Pass 1: Case A (Box already aligns with a detected face: IoU >= 0.25)
        for row in confirmed_rows:
            mp_id = int(row["id"])
            rx = float(row["rect_x"])
            ry = float(row["rect_y"])
            rw = float(row["rect_w"])
            rh = float(row["rect_h"])
            r_box = (rx, ry, rw, rh)

            best_iou = -1.0
            best_d_idx = -1
            for d_idx, d in enumerate(detected_faces):
                if d_idx in claimed_d_indices:
                    continue
                d_box = (d.rect_x, d.rect_y, d.rect_w, d.rect_h)
                iou = compute_iou(r_box, d_box)
                if iou > best_iou:
                    best_iou = iou
                    best_d_idx = d_idx

            if best_iou >= 0.25 and best_d_idx >= 0:
                claimed_d_indices.add(best_d_idx)
                matched_d = detected_faces[best_d_idx]
                self._delete_overlapping_unassigned(
                    conn, media_id, (rx, ry, rw, rh)
                )
                # Centroid Quality Gate: store verified face embedding, NEVER move coordinates or rename person
                with self.engine.write_lock:
                    conn.execute(
                        "UPDATE media_people SET embedding = ? WHERE id = ?;",
                        (
                            serialize_embedding(matched_d.embedding),
                            mp_id,
                        ),
                    )
                    conn.commit()
                if matched_d.crop_webp:
                    self.thumb_repo.put_face_thumbnail(mp_id, matched_d.crop_webp)
                    pid = row["person_id"]
                    if pid:
                        with self.engine.write_lock:
                            conn.execute(
                                "UPDATE people SET thumbnail_blob = ? WHERE id = ? AND thumbnail_blob IS NULL;",
                                (matched_d.crop_webp, int(pid)),
                            )
                            conn.commit()
            else:
                unmatched_rows.append(dict(row))

        # Centroid Quality Gate for unmatched confirmed faces:
        # If FaceEngine did not confirm a valid face (IoU < 0.25), ensure embedding = NULL
        # so it does not poison person centroids. Never move rect coordinates or swap names!
        for item in unmatched_rows:
            mp_id = int(item["id"])
            with self.engine.write_lock:
                conn.execute(
                    "UPDATE media_people SET embedding = NULL WHERE id = ?;",
                    (mp_id,),
                )
                conn.commit()
            # If thumbnail is missing, generate crop from the box so the UI has an avatar
            if not self.thumb_repo.get_face_thumbnail(mp_id):
                try:
                    _, crop_bytes = self.face_engine.extract_crop_and_embed_rect(
                        str(fpath),
                        (float(item["rect_x"]), float(item["rect_y"]), float(item["rect_w"]), float(item["rect_h"])),
                        orientation=orientation,
                    )
                    if crop_bytes:
                        self.thumb_repo.put_face_thumbnail(mp_id, crop_bytes)
                except Exception:
                    pass

        # Mark image face scan as up to date
        with self.engine.write_lock:
            conn.execute(
                "UPDATE media SET face_scan_version = ? WHERE id = ?;",
                (TARGET_FACE_SCAN_VERSION, media_id),
            )
            conn.commit()

        # Return unclaimed detected faces for unassigned insertion
        return [
            d for idx, d in enumerate(detected_faces) if idx not in claimed_d_indices
        ]

    def _recompute_people_centroids(self) -> None:
        """Recomputes L2-normalized centroid embeddings for people with confirmed faces."""
        conn = self.engine.catalog_conn()
        people_rows = conn.execute(
            """
            SELECT DISTINCT person_id
            FROM media_people
            WHERE state = 1 AND person_id IS NOT NULL AND embedding IS NOT NULL;
            """
        ).fetchall()

        if not people_rows:
            return

        for p_row in people_rows:
            if self._is_cancelled():
                return
            pid = int(p_row["person_id"])
            emb_rows = conn.execute(
                "SELECT embedding FROM media_people WHERE person_id = ? AND state = 1 AND embedding IS NOT NULL;",
                (pid,),
            ).fetchall()

            if emb_rows:
                vectors = [deserialize_embedding(bytes(r["embedding"])) for r in emb_rows if r["embedding"]]
                if vectors:
                    mean_vec = np.mean(np.stack(vectors, axis=0), axis=0)
                    centroid = normalize_vector(mean_vec)
                    self.repo.update_person_centroid(pid, serialize_embedding(centroid))

        self.signals.suggestions_updated.emit()

    # -------------------------------------------------------------------------
    # Stage 2: Background Face Scanning
    # -------------------------------------------------------------------------
    def _run_stage2_scan(self) -> int:
        """
        Scans untagged media records (face_scan_version < 1), detects faces,
        inserts unassigned media_people (state = 0), writes crops to face_thumbnails,
        and runs Stage 3 matching after each batch.
        """
        conn = self.engine.catalog_conn()
        count_row = conn.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM media
            WHERE is_missing = 0 AND is_video = 0 AND face_scan_version >= 0 AND face_scan_version < ?;
            """,
            (TARGET_FACE_SCAN_VERSION,),
        ).fetchone()
        total_to_process = int(count_row["cnt"]) if count_row else 0

        if total_to_process == 0:
            return 0

        processed = 0
        total_faces_detected = 0

        self.signals.status_message.emit(f"Scanning {total_to_process} photos for faces...")
        self.signals.batch_progressed.emit(0, total_to_process)

        while not self._is_cancelled():
            batch_rows = conn.execute(
                """
                SELECT m.id, m.filename, m.orientation, f.path AS folder_path
                FROM media m
                JOIN folders f ON f.id = m.folder_id
                WHERE m.is_missing = 0 AND m.is_video = 0 AND m.face_scan_version >= 0 AND m.face_scan_version < ?
                ORDER BY m.id ASC
                LIMIT ?;
                """,
                (TARGET_FACE_SCAN_VERSION, self.batch_size),
            ).fetchall()

            if not batch_rows:
                break

            for r in batch_rows:
                if self._is_cancelled():
                    break

                mid = int(r["id"])
                fpath = Path(r["folder_path"]) / str(r["filename"])

                if not fpath.exists():
                    self.repo.mark_face_scan_failed([mid])
                    processed += 1
                    continue

                try:
                    detected_faces = self.face_engine.detect_and_embed(
                        str(fpath),
                        orientation=int(r["orientation"] or 1),
                    )
                except Exception as exc:
                    logger.warning("Failed to detect faces in poison-pill image %s: %s", fpath, exc)
                    from src.utils.scan_logger import log_scan_error
                    log_scan_error("FaceDetectionWorker (Stage 2)", fpath, exc)
                    self.repo.mark_face_scan_failed([mid])
                    processed += 1
                    continue

                # Verify & heal any confirmed faces on this media, claiming matched detected faces
                unclaimed_faces = self._verify_and_heal_image_faces(
                    mid,
                    fpath,
                    int(r["orientation"] or 1),
                    detected_faces,
                )

                if unclaimed_faces:
                    payload = [
                        {
                            "rect_x": df.rect_x,
                            "rect_y": df.rect_y,
                            "rect_w": df.rect_w,
                            "rect_h": df.rect_h,
                            "embedding": serialize_embedding(df.embedding),
                            "similarity_score": None,
                        }
                        for df in unclaimed_faces
                    ]
                    created_ids = self.repo.insert_detected_faces(mid, payload)
                    total_faces_detected += len(unclaimed_faces)

                    # Save face crop thumbnails
                    thumb_entries = [
                        (fid, df.crop_webp)
                        for fid, df in zip(created_ids, unclaimed_faces)
                        if df.crop_webp
                    ]
                    if thumb_entries:
                        self.thumb_repo.put_face_thumbnails_batch(thumb_entries)

                # Mark media face scan complete
                self.repo.update_face_scan_version([mid], version=TARGET_FACE_SCAN_VERSION)
                processed += 1
                self.signals.batch_progressed.emit(processed, total_to_process)

                if self.throttle_ms > 0:
                    time.sleep(self.throttle_ms / 1000.0)

            # Run Stage 3 matching pass after each batch
            self._run_stage3_match()

        return total_faces_detected

    # -------------------------------------------------------------------------
    # Stage 3: Vectorized Face Matching
    # -------------------------------------------------------------------------
    def _run_stage3_match(self) -> None:
        """
        Runs high-speed matrix cosine similarity between unassigned face detections
        (state = 0) and all person centroids using FaceVectorMatcher.
        Stores top candidate matches and similarity scores.
        """
        centroids = self.repo.get_people_centroids()
        if not centroids:
            return

        unassigned = self.repo.get_unassigned_face_embeddings(limit=None)
        if not unassigned:
            return

        rejected_pairs = self.repo.get_rejected_face_pairs()

        person_ids = list(centroids.keys())
        centroids_dict = {
            pid: deserialize_embedding(centroids[pid], allow_multi=True)
            for pid in person_ids
        }

        face_ids = [int(u["id"]) for u in unassigned]
        face_matrix = np.stack(
            [deserialize_embedding(bytes(u["embedding"])) for u in unassigned],
            axis=0,
        )

        suggestions = FaceVectorMatcher.match_unassigned_faces(
            face_ids=face_ids,
            face_matrix=face_matrix,
            person_ids=person_ids,
            centroid_matrix=centroids_dict,
            rejected_pairs=rejected_pairs,
            high_thresh=self.high_match_score,
            low_thresh=self.min_match_score,
            top_k=1,
            assume_normalized=True,
        )

        if suggestions:
            update_payload = [
                (s.face_id, s.person_id, s.score)
                for s in suggestions
            ]
            self.repo.update_suggested_matches(update_payload)
            self.signals.suggestions_updated.emit()

    def _reset_scan_versions(self) -> None:
        """Resets face_scan_version to 0 for all non-video media to force a complete re-scan."""
        with self.engine.write_lock:
            conn = self.engine.catalog_conn()
            conn.execute("UPDATE media SET face_scan_version = 0 WHERE is_video = 0;")
            conn.commit()
