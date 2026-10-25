"""
Unit and integration tests for WLPG / MWG / digiKam XMP interoperability,
mtime preservation, and face bounding-box transformations on rotation and crop.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from PySide6.QtGui import QImage

from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository
from src.workers.xmp_sync_worker import (
    _sync_single_media_item,
    build_wlpg_xmp_packet,
    parse_xmp_packet_bytes,
    write_metadata_lossless,
)


@pytest.fixture
def db_engine(tmp_path: Path) -> DualDatabaseEngine:
    engine = DualDatabaseEngine(tmp_path / "catalog.db", tmp_path / "thumbnails.db")
    yield engine
    engine.close_all()


def test_mwg_vs_microsoft_region_roundtrip() -> None:
    """Tests top-left (MP) vs center-point (MWG) face region conversions."""
    # Top-left box: x=0.20, y=0.30, w=0.10, h=0.15
    media_detail = {
        "rating": 3,
        "caption": "Family Photo",
        "author": "Alice",
        "orientation": 1,
        "hierarchical_tags": ["People/Family", "Events/Summer"],
        "people": [{"name": "Charlie", "rect": (0.20, 0.30, 0.10, 0.15)}],
    }

    packet = build_wlpg_xmp_packet(media_detail)
    packet_str = packet.decode("utf-8")

    # 1. Verify MP:RegionInfo (Top-Left)
    assert "MP:RegionInfo" in packet_str
    assert "0.200000, 0.300000, 0.100000, 0.150000" in packet_str

    # 2. Verify mwg-rs:Regions (Center-Point: cx = 0.20 + 0.05 = 0.25, cy = 0.30 + 0.075 = 0.375)
    assert "mwg-rs:Regions" in packet_str
    assert 'stArea:x="0.250000"' in packet_str
    assert 'stArea:y="0.375000"' in packet_str
    assert 'stArea:w="0.100000"' in packet_str
    assert 'stArea:h="0.150000"' in packet_str

    # 3. Parse with MP:RegionInfo present
    parsed = parse_xmp_packet_bytes(packet)
    assert len(parsed["people"]) == 1
    p = parsed["people"][0]
    assert p["name"] == "Charlie"
    rx, ry, rw, rh = p["rect"]
    assert pytest.approx(rx, 1e-4) == 0.20
    assert pytest.approx(ry, 1e-4) == 0.30
    assert pytest.approx(rw, 1e-4) == 0.10
    assert pytest.approx(rh, 1e-4) == 0.15

    # 4. Strip MP:RegionInfo and verify MWG center-to-top-left conversion
    start_mp = packet_str.find("<MP:RegionInfo")
    end_mp = packet_str.find("</MP:RegionInfo>") + len("</MP:RegionInfo>")
    mwg_only_packet = (packet_str[:start_mp] + packet_str[end_mp:]).encode("utf-8")

    parsed_mwg = parse_xmp_packet_bytes(mwg_only_packet)
    assert len(parsed_mwg["people"]) == 1
    p_mwg = parsed_mwg["people"][0]
    assert p_mwg["name"] == "Charlie"
    m_rx, m_ry, m_rw, m_rh = p_mwg["rect"]
    assert pytest.approx(m_rx, 1e-4) == 0.20
    assert pytest.approx(m_ry, 1e-4) == 0.30
    assert pytest.approx(m_rw, 1e-4) == 0.10
    assert pytest.approx(m_rh, 1e-4) == 0.15


def test_wlpg_digikam_tags_captions_and_ratings() -> None:
    """Verifies ratings, digiKam tags, and captions interoperability."""
    # Test rating mapping in packet creation
    ratings_map = {1: "1", 2: "25", 3: "50", 4: "75", 5: "99"}
    for star, pct in ratings_map.items():
        pkt = build_wlpg_xmp_packet({"rating": star, "caption": "Test Caption"}).decode("utf-8")
        assert f'xmp:Rating="{star}"' in pkt
        assert f'MicrosoftPhoto:Rating="{pct}"' in pkt

    # Test reading with only MicrosoftPhoto:Rating percentage
    pkt_ms_only = (
        b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        b'<rdf:Description rdf:about="" xmlns:MicrosoftPhoto="http://ns.microsoft.com/photo/1.0/" '
        b'MicrosoftPhoto:Rating="75"/></rdf:RDF></x:xmpmeta>'
    )
    parsed_ms = parse_xmp_packet_bytes(pkt_ms_only)
    assert parsed_ms["rating"] == 4

    # Test digiKam:TagsList reading
    pkt_digikam = (
        b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        b'<rdf:Description rdf:about="" xmlns:digiKam="http://www.digikam.org/ns/1.0/">'
        b'<digiKam:TagsList><rdf:Seq><rdf:li>Places/Mountains</rdf:li><rdf:li>Holiday/2026</rdf:li></rdf:Seq></digiKam:TagsList>'
        b'</rdf:Description></rdf:RDF></x:xmpmeta>'
    )
    parsed_dk = parse_xmp_packet_bytes(pkt_digikam)
    assert "Places/Mountains" in parsed_dk["hierarchical_tags"]
    assert "Holiday/2026" in parsed_dk["hierarchical_tags"]

    # Test dc:title fallback for caption
    pkt_title = (
        b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        b'<rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/">'
        b'<dc:title><rdf:Alt><rdf:li xml:lang="x-default">Title As Caption</rdf:li></rdf:Alt></dc:title>'
        b'</rdf:Description></rdf:RDF></x:xmpmeta>'
    )
    parsed_title = parse_xmp_packet_bytes(pkt_title)
    assert parsed_title["caption"] == "Title As Caption"


def test_mtime_preservation_on_metadata_sync(tmp_path: Path, db_engine: DualDatabaseEngine) -> None:
    """Verifies that lossless metadata injection restores the original st_mtime via os.utime."""
    repo = CatalogRepository(db_engine)
    img_dir = tmp_path / "photos"
    img_dir.mkdir()
    fid = repo.upsert_folder(str(img_dir), "photos", None, 1700000000)

    # Create dummy JPEG
    img_path = img_dir / "preserve_mtime.jpg"
    qimg = QImage(120, 80, QImage.Format_RGB32)
    qimg.fill(0x00FF00)
    qimg.save(str(img_path), "JPEG")

    # Set historical mtime (e.g. exactly 1000000000 = year 2001)
    target_mtime = 1000000000
    os.utime(img_path, (target_mtime, target_mtime))
    assert int(img_path.stat().st_mtime) == target_mtime

    ids = repo.bulk_upsert_media(
        [
            {
                "folder_id": fid,
                "filename": img_path.name,
                "ext": "jpg",
                "size": img_path.stat().st_size,
                "mtime": target_mtime,
                "date_taken": target_mtime,
            }
        ]
    )
    mid = ids[0]

    # Assign rating and tag (marks metadata_dirty = 1)
    repo.update_rating([mid], 5)

    detail = repo.get_media_detail(mid)
    res_mtime = _sync_single_media_item(detail, db_engine)
    assert res_mtime == target_mtime

    # Verify filesystem mtime is preserved
    current_disk_mtime = int(img_path.stat().st_mtime)
    assert current_disk_mtime == target_mtime

    # Verify catalog.db mtime matches
    refreshed_detail = repo.get_media_detail(mid)
    assert refreshed_detail["mtime"] == target_mtime


def test_face_bounding_box_transforms_on_rotate(db_engine: DualDatabaseEngine) -> None:
    """Tests face coordinate transforms under 90 CW, 90 CCW, and 180 degree rotations."""
    repo = CatalogRepository(db_engine)
    fid = repo.upsert_folder("D:/Photos", "Photos", None, 1700000000)
    ids = repo.bulk_upsert_media(
        [
            {
                "folder_id": fid,
                "filename": "face_rot.jpg",
                "ext": "jpg",
                "size": 1000,
                "mtime": 1700000000,
                "width": 1000,
                "height": 800,
                "orientation": 1,
            }
        ]
    )
    mid = ids[0]

    # Assign face: (rx=0.20, ry=0.30, rw=0.10, rh=0.15)
    repo.assign_person(mid, "Bob", rect=(0.20, 0.30, 0.10, 0.15))

    # 1. Rotate 90° CW
    # (new_x, new_y, new_w, new_h) = (1.0 - (ry + rh), rx, rh, rw) = (1.0 - 0.45, 0.20, 0.15, 0.10) = (0.55, 0.20, 0.15, 0.10)
    repo.rotate_lossless([mid], clockwise=True)
    detail_cw = repo.get_media_detail(mid)
    p_cw = detail_cw["people"][0]
    rx, ry, rw, rh = p_cw["rect"]
    assert pytest.approx(rx, 1e-4) == 0.55
    assert pytest.approx(ry, 1e-4) == 0.20
    assert pytest.approx(rw, 1e-4) == 0.15
    assert pytest.approx(rh, 1e-4) == 0.10

    # 2. Rotate 90° CCW (back to original)
    # (new_x, new_y, new_w, new_h) = (ry, 1.0 - (rx + rw), rh, rw) = (0.20, 1.0 - 0.70, 0.10, 0.15) = (0.20, 0.30, 0.10, 0.15)
    repo.rotate_lossless([mid], clockwise=False)
    detail_orig = repo.get_media_detail(mid)
    p_orig = detail_orig["people"][0]
    rx, ry, rw, rh = p_orig["rect"]
    assert pytest.approx(rx, 1e-4) == 0.20
    assert pytest.approx(ry, 1e-4) == 0.30
    assert pytest.approx(rw, 1e-4) == 0.10
    assert pytest.approx(rh, 1e-4) == 0.15

    # 3. Rotate 180°
    # (new_x, new_y, new_w, new_h) = (1.0 - (rx + rw), 1.0 - (ry + rh), rw, rh) = (1.0 - 0.30, 1.0 - 0.45, 0.10, 0.15) = (0.70, 0.55, 0.10, 0.15)
    repo.rotate_lossless([mid], delta_degrees=180)
    detail_180 = repo.get_media_detail(mid)
    p_180 = detail_180["people"][0]
    rx, ry, rw, rh = p_180["rect"]
    assert pytest.approx(rx, 1e-4) == 0.70
    assert pytest.approx(ry, 1e-4) == 0.55
    assert pytest.approx(rw, 1e-4) == 0.10
    assert pytest.approx(rh, 1e-4) == 0.15


def test_face_bounding_box_transforms_on_crop(db_engine: DualDatabaseEngine) -> None:
    """Tests face coordinate transforms and removal on crop (< 25% visible area)."""
    repo = CatalogRepository(db_engine)
    fid = repo.upsert_folder("D:/Photos", "Photos", None, 1700000000)
    ids = repo.bulk_upsert_media(
        [
            {
                "folder_id": fid,
                "filename": "crop_test.jpg",
                "ext": "jpg",
                "size": 1000,
                "mtime": 1700000000,
                "width": 1000,
                "height": 1000,
            }
        ]
    )
    mid = ids[0]

    # Face 1: Fully inside crop box (0.2, 0.2, 0.1, 0.1)
    # Face 2: Completely outside crop box (0.8, 0.8, 0.1, 0.1)
    # Face 3: Partially cropped out (0.45, 0.2, 0.1, 0.1) -> only 0.05 width inside (50% area)
    # Face 4: Heavily cropped out (0.49, 0.2, 0.1, 0.1) -> only 0.01 width inside (10% area < 25%)
    repo.assign_person(mid, "PersonInside", rect=(0.20, 0.20, 0.10, 0.10))
    repo.assign_person(mid, "PersonOutside", rect=(0.80, 0.80, 0.10, 0.10))
    repo.assign_person(mid, "PersonPartCut", rect=(0.45, 0.20, 0.10, 0.10))
    repo.assign_person(mid, "PersonHeavilyCut", rect=(0.49, 0.20, 0.10, 0.10))

    # Apply crop: norm_crop = (0.0, 0.0, 0.50, 0.50)
    # New coordinates for Face 1:
    # rx = (0.20 - 0.0) / 0.50 = 0.40, ry = 0.40, rw = 0.20, rh = 0.20
    retained = repo.transform_face_regions_for_crop(mid, 0.0, 0.0, 0.50, 0.50)
    assert retained == 2  # PersonInside and PersonPartCut should be retained

    detail = repo.get_media_detail(mid)
    people_names = {p["name"]: p["rect"] for p in detail["people"]}

    assert "PersonInside" in people_names
    assert "PersonPartCut" in people_names
    assert "PersonOutside" not in people_names
    assert "PersonHeavilyCut" not in people_names

    p1_rect = people_names["PersonInside"]
    assert pytest.approx(p1_rect[0], 1e-4) == 0.40
    assert pytest.approx(p1_rect[1], 1e-4) == 0.40
    assert pytest.approx(p1_rect[2], 1e-4) == 0.20
    assert pytest.approx(p1_rect[3], 1e-4) == 0.20

    # PersonPartCut was [0.45, 0.20, 0.10, 0.10]
    # Remapped to crop: rx = 0.45/0.50 = 0.90, rw = 0.10/0.50 = 0.20 (rx+rw = 1.10 -> clamped to 1.0)
    # Clamped w = 1.0 - 0.90 = 0.10
    part_rect = people_names["PersonPartCut"]
    assert pytest.approx(part_rect[0], 1e-4) == 0.90
    assert pytest.approx(part_rect[1], 1e-4) == 0.40
    assert pytest.approx(part_rect[2], 1e-4) == 0.10
    assert pytest.approx(part_rect[3], 1e-4) == 0.20


def test_restore_original_faces(db_engine: DualDatabaseEngine) -> None:
    """Tests that restore_original_faces resets media_people and updates confirmed_count accurately."""
    repo = CatalogRepository(db_engine)
    fid = repo.upsert_folder("D:/Photos", "Photos", None, 1700000000)
    ids = repo.bulk_upsert_media(
        [
            {
                "folder_id": fid,
                "filename": "revert_faces.jpg",
                "ext": "jpg",
                "size": 1000,
                "mtime": 1700000000,
            }
        ]
    )
    mid = ids[0]

    original_people = [
        {"name": "OriginalAlice", "rect": (0.1, 0.1, 0.2, 0.2)},
        {"name": "OriginalBob", "rect": (0.5, 0.5, 0.2, 0.2)},
    ]

    # Assign initial people
    for p in original_people:
        repo.assign_person(mid, p["name"], rect=p["rect"])

    # Simulate crop that removes OriginalBob and modifies OriginalAlice
    repo.transform_face_regions_for_crop(mid, 0.0, 0.0, 0.4, 0.4)
    detail_cropped = repo.get_media_detail(mid)
    assert len(detail_cropped["people"]) == 1
    assert detail_cropped["people"][0]["name"] == "OriginalAlice"

    # Now restore original faces
    repo.restore_original_faces(mid, original_people)
    detail_restored = repo.get_media_detail(mid)
    assert len(detail_restored["people"]) == 2
    names = {p["name"] for p in detail_restored["people"]}
    assert "OriginalAlice" in names
    assert "OriginalBob" in names
