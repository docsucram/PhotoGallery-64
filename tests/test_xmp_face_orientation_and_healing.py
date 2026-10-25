"""
Unit tests for XMP face region orientation handling and self-healing auto-snap.
Verifies:
1. mwg-rs:Regions vs MP:RegionInfo priority and orientation math.
2. Orientation transform mapping raw sensor MP:RegionInfo into display space.
3. Self-healing auto-snap rejecting empty crops and snapping misplaced boxes to detected faces.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from src.core.face_engine import DetectedFace, compute_iou
from src.core.face_matcher import (
    cosine_similarity,
    deserialize_embedding,
    normalize_vector,
    serialize_embedding,
)
from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository, ThumbnailRepository
from src.workers.face_worker import FaceDetectionWorker, get_box_rotation_candidates
from src.workers.xmp_sync_worker import (
    parse_xmp_packet_bytes,
    transform_box_raw_to_display,
)


def test_mwg_priority_and_display_space_conversion():
    """
    Test 1: Portrait image (3000x4000 display, EXIF Orientation = 6) containing both
    mwg-rs:Regions (in display space cx=0.62, cy=0.46, w=0.25, h=0.22) and a conflicting
    rotated MP:RegionInfo (x=0.35, y=0.50, w=0.22, h=0.25).
    Verifies that the parser selects mwg-rs and yields top-left (0.495, 0.35, 0.25, 0.22).
    """
    xmp_xml = """<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
        xmlns:MP="http://ns.microsoft.com/photo/1.2/"
        xmlns:MPRI="http://ns.microsoft.com/photo/1.2/t/RegionInfo#"
        xmlns:MPReg="http://ns.microsoft.com/photo/1.2/t/Region#"
        xmlns:mwg-rs="http://www.metadataworkinggroup.com/schemas/regions/"
        xmlns:stDim="http://ns.adobe.com/xap/1.0/sType/Dimensions#"
        xmlns:stArea="http://ns.adobe.com/xmp/sType/Area#">
      <MP:RegionInfo rdf:parseType="Resource">
        <MPRI:Regions>
          <rdf:Bag>
            <rdf:li rdf:parseType="Resource">
              <MPReg:PersonDisplayName>Alice Conflicted</MPReg:PersonDisplayName>
              <MPReg:Rectangle>0.350000, 0.500000, 0.220000, 0.250000</MPReg:Rectangle>
            </rdf:li>
          </rdf:Bag>
        </MPRI:Regions>
      </MP:RegionInfo>
      <mwg-rs:Regions rdf:parseType="Resource">
        <mwg-rs:AppliedToDimensions stDim:w="3000" stDim:h="4000" stDim:unit="pixel"/>
        <mwg-rs:RegionList>
          <rdf:Bag>
            <rdf:li rdf:parseType="Resource">
              <mwg-rs:Name>Alice Conflicted</mwg-rs:Name>
              <mwg-rs:Type>Face</mwg-rs:Type>
              <mwg-rs:Area stArea:x="0.620000" stArea:y="0.460000" stArea:w="0.250000" stArea:h="0.220000" stArea:unit="normalized"/>
            </rdf:li>
          </rdf:Bag>
        </mwg-rs:RegionList>
      </mwg-rs:Regions>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>""".encode("utf-8")

    parsed = parse_xmp_packet_bytes(
        xmp_xml,
        image_width=3000,
        image_height=4000,
        raw_width=4000,
        raw_height=3000,
        orientation=6,
    )

    people = parsed["people"]
    assert len(people) == 1
    person = people[0]
    assert person["name"] == "Alice Conflicted"

    rx, ry, rw, rh = person["rect"]
    # cx = 0.62, w = 0.25 -> rx = 0.62 - 0.125 = 0.495
    # cy = 0.46, h = 0.22 -> ry = 0.46 - 0.11 = 0.35
    assert pytest.approx(rx, abs=1e-4) == 0.495
    assert pytest.approx(ry, abs=1e-4) == 0.35
    assert pytest.approx(rw, abs=1e-4) == 0.25
    assert pytest.approx(rh, abs=1e-4) == 0.22


def test_raw_sensor_mp_region_orientation_transform():
    """
    Test 2: Portrait image (EXIF Orientation = 6) where only MP:RegionInfo exists in raw
    unrotated sensor coordinates. Verifies the orientation transform maps it onto the face in display space.
    """
    # Raw sensor unrotated box: (x=0.35, y=0.495, w=0.22, h=0.25)
    xmp_xml = """<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
        xmlns:MP="http://ns.microsoft.com/photo/1.2/"
        xmlns:MPRI="http://ns.microsoft.com/photo/1.2/t/RegionInfo#"
        xmlns:MPReg="http://ns.microsoft.com/photo/1.2/t/Region#">
      <MP:RegionInfo rdf:parseType="Resource">
        <MPRI:Regions>
          <rdf:Bag>
            <rdf:li rdf:parseType="Resource">
              <MPReg:PersonDisplayName>Bob WLPG</MPReg:PersonDisplayName>
              <MPReg:Rectangle>0.350000, 0.495000, 0.220000, 0.250000</MPReg:Rectangle>
            </rdf:li>
          </rdf:Bag>
        </MPRI:Regions>
      </MP:RegionInfo>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>""".encode("utf-8")

    parsed = parse_xmp_packet_bytes(
        xmp_xml,
        image_width=3000,
        image_height=4000,
        raw_width=4000,
        raw_height=3000,
        orientation=6,
    )

    people = parsed["people"]
    assert len(people) == 1
    person = people[0]
    assert person["name"] == "Bob WLPG"

    rx, ry, rw, rh = person["rect"]
    # Orientation 6: (1.0 - (y + h), x, h, w)
    # new_x = 1.0 - (0.495 + 0.25) = 1.0 - 0.745 = 0.255
    # new_y = 0.35
    # new_w = 0.25
    # new_h = 0.22
    assert pytest.approx(rx, abs=1e-4) == 0.255
    assert pytest.approx(ry, abs=1e-4) == 0.35
    assert pytest.approx(rw, abs=1e-4) == 0.25
    assert pytest.approx(rh, abs=1e-4) == 0.22


def test_self_healing_auto_snap_and_poisoning_prevention(tmp_path: Path):
    """
    Test 3: Self-healing auto-snap when an imported box has IoU == 0 with the actual face.
    Verifies that:
    1. The non-face crop is rejected from centroid learning (embedding is NULL if not healed).
    2. When a candidate rotation or 1-to-1 face is available, it snaps to the detected face.
    3. metadata_dirty = 1 is set on media.
    4. Media people row coordinates match the detected face.
    """
    catalog_path = tmp_path / "catalog.db"
    thumbs_path = tmp_path / "thumbnails.db"
    engine = DualDatabaseEngine(catalog_path, thumbs_path)
    engine.initialize_schemas()

    catalog_repo = CatalogRepository(engine)
    thumb_repo = ThumbnailRepository(engine)

    # Insert test folder and media
    folder_id = catalog_repo.upsert_folder(str(tmp_path), "test_dir", None, 1000)
    dummy_img = tmp_path / "portrait_test.jpg"
    # Create simple dummy image
    img = Image.new("RGB", (300, 400), color=(200, 200, 200))
    img.save(dummy_img, format="JPEG")

    with engine.write_lock:
        conn = engine.catalog_conn()
        cur = conn.execute(
            """
            INSERT INTO media (folder_id, filename, ext, size, mtime, width, height, orientation, is_video)
            VALUES (?, ?, '.jpg', 1000, 1000, 300, 400, 6, 0)
            RETURNING id;
            """,
            (folder_id, dummy_img.name),
        )
        media_id = int(cur.fetchone()[0])

        # Create confirmed person Charlie
        p_cur = conn.execute("INSERT INTO people (name) VALUES ('Charlie') RETURNING id;")
        person_id = int(p_cur.fetchone()[0])

        # Insert misplaced XMP face tag on kitchen sink (x=0.05, y=0.05, w=0.1, h=0.1)
        mp_cur = conn.execute(
            """
            INSERT INTO media_people (media_id, person_id, state, is_manual, rect_x, rect_y, rect_w, rect_h, embedding)
            VALUES (?, ?, 1, 1, 0.05, 0.05, 0.1, 0.1, NULL)
            RETURNING id;
            """,
            (media_id, person_id),
        )
        mp_id = int(mp_cur.fetchone()[0])
        conn.commit()

    # Create dummy detected face at actual face position (0.5, 0.4, 0.25, 0.22)
    face_emb = normalize_vector(np.random.randn(512).astype(np.float32))
    dummy_webp = b"RIFF\x00\x00\x00\x00WEBPVP8 \x00\x00\x00\x00"
    detected_face = DetectedFace(
        rect_x=0.5,
        rect_y=0.4,
        rect_w=0.25,
        rect_h=0.22,
        score=0.95,
        embedding=face_emb,
        crop_webp=dummy_webp,
    )

    worker = FaceDetectionWorker(engine=engine)
    # IoU between (0.05, 0.05, 0.1, 0.1) and (0.5, 0.4, 0.25, 0.22) is exactly 0.0
    iou = compute_iou((0.05, 0.05, 0.1, 0.1), (0.5, 0.4, 0.25, 0.22))
    assert iou == 0.0

    # Run self-healing verification
    unclaimed = worker._verify_and_heal_image_faces(
        media_id=media_id,
        fpath=dummy_img,
        orientation=6,
        detected_faces=[detected_face],
    )

    # Because there was 1 misplaced tag and 1 unclaimed detected face, it snaps to detected_face!
    assert len(unclaimed) == 0

    with engine.catalog_conn() as conn:
        row = conn.execute(
            "SELECT rect_x, rect_y, rect_w, rect_h, embedding FROM media_people WHERE id = ?;",
            (mp_id,),
        ).fetchone()
        assert row is not None
        assert pytest.approx(float(row["rect_x"]), abs=1e-4) == 0.5
        assert pytest.approx(float(row["rect_y"]), abs=1e-4) == 0.4
        assert pytest.approx(float(row["rect_w"]), abs=1e-4) == 0.25
        assert pytest.approx(float(row["rect_h"]), abs=1e-4) == 0.22

        saved_emb = deserialize_embedding(bytes(row["embedding"]))
        assert cosine_similarity(saved_emb, face_emb) >= 0.99

        media_row = conn.execute(
            "SELECT metadata_dirty FROM media WHERE id = ?;", (media_id,)
        ).fetchone()
        assert media_row["metadata_dirty"] == 1

    # Verify face thumbnail was saved
    face_thumb = thumb_repo.get_face_thumbnail(mp_id)
    assert face_thumb == dummy_webp

    engine.close_thread_connections()
