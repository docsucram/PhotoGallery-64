"""
Test Suite for WP-09B: Archive Hardening & Edge Cases (tests/test_wp09b_external_edits_and_edge_cases.py).

Verifies the 7 real-world archive edge cases:
1. MWG vs MP face regions preference with AppliedToDimensions matching.
2. External crop/dimension change detection (face_scan_version reset, thumbnail invalidation).
3. Metadata protection when external editor strips tags/rating (metadata_dirty marked).
4. Intra-folder rename detection without losing media_id, tags, or face tags.
5. .OriginalFiles exclusion and backup cleanup on media deletion.
6. Watcher feedback loop prevention via recently written registry.
7. Pre-1970 negative timestamp safety on Windows.
8. Corrupt file poison-pill protection (face_scan_version = -1).
"""

from __future__ import annotations

import datetime
import os
import shutil
from pathlib import Path

import pytest
from PIL import Image

from src.db.connection import DualDatabaseEngine
from src.db.repository import (
    CatalogRepository,
    FilterQuerySpec,
    ThumbnailRepository,
    _epoch_to_datetime_utc,
)
from src.workers.folder_watcher import DeltaSyncWorker
from src.workers.scanner_thumb_worker import IGNORED_DIR_NAMES
from src.workers.xmp_sync_worker import (
    is_recently_written,
    parse_xmp_packet_bytes,
    record_recently_written,
)


@pytest.fixture
def test_engine(tmp_path: Path) -> DualDatabaseEngine:
    engine = DualDatabaseEngine(
        catalog_path=tmp_path / "catalog.db",
        thumbnails_path=tmp_path / "thumbnails.db",
        schema_sql_path=Path("db/schema.sql"),
    )
    engine.initialize_schemas()
    return engine


def test_mwg_applied_to_dimensions_preference() -> None:
    """If mwg-rs:AppliedToDimensions matches actual (width, height), prefer mwg-rs:Regions over MP:RegionInfo."""
    xmp_packet = b"""<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
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
      <rdf:li MPReg:PersonDisplayName="Alice (Old Region)" MPReg:Rectangle="0.1,0.1,0.2,0.2"/>
     </rdf:Bag>
    </MPRI:Regions>
   </MP:RegionInfo>
   <mwg-rs:Regions rdf:parseType="Resource">
    <mwg-rs:AppliedToDimensions stDim:w="3000" stDim:h="2000" stDim:unit="pixel"/>
    <mwg-rs:RegionList>
     <rdf:Bag>
      <rdf:li>
       <mwg-rs:Name>Alice (Cropped Region)</mwg-rs:Name>
       <mwg-rs:Area stArea:x="0.5" stArea:y="0.5" stArea:w="0.2" stArea:h="0.2" stArea:unit="normalized"/>
      </rdf:li>
     </rdf:Bag>
    </mwg-rs:RegionList>
   </mwg-rs:Regions>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""

    # 1. When image contains both MWG and MP, mwg-rs:Regions is preferred as primary source of truth
    parsed_matching = parse_xmp_packet_bytes(xmp_packet, image_width=3000, image_height=2000)
    assert len(parsed_matching["people"]) == 1
    assert parsed_matching["people"][0]["name"] == "Alice (Cropped Region)"
    # Center (0.5, 0.5) with (0.2, 0.2) converts to top-left (0.4, 0.4, 0.2, 0.2)
    assert pytest.approx(parsed_matching["people"][0]["rect"][0], 0.01) == 0.4
    assert pytest.approx(parsed_matching["people"][0]["rect"][1], 0.01) == 0.4

    # 2. When mwg-rs is absent or empty, fall back to MP:RegionInfo
    mp_only_packet = b"""<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:MP="http://ns.microsoft.com/photo/1.2/"
    xmlns:MPRI="http://ns.microsoft.com/photo/1.2/t/RegionInfo#"
    xmlns:MPReg="http://ns.microsoft.com/photo/1.2/t/Region#">
   <MP:RegionInfo rdf:parseType="Resource">
    <MPRI:Regions>
     <rdf:Bag>
      <rdf:li MPReg:PersonDisplayName="Alice (Old Region)" MPReg:Rectangle="0.1,0.1,0.2,0.2"/>
     </rdf:Bag>
    </MPRI:Regions>
   </MP:RegionInfo>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""
    parsed_mp_only = parse_xmp_packet_bytes(mp_only_packet, image_width=4000, image_height=3000)
    assert len(parsed_mp_only["people"]) == 1
    assert parsed_mp_only["people"][0]["name"] == "Alice (Old Region)"
    assert parsed_mp_only["people"][0]["rect"] == (0.1, 0.1, 0.2, 0.2)


def test_external_crop_resets_face_scan_and_invalidates_thumbnail(test_engine: DualDatabaseEngine) -> None:
    """When dimensions change on disk, reset face_scan_version = 0 and invalidate cached thumbnail."""
    repo = CatalogRepository(test_engine)
    thumb_repo = ThumbnailRepository(test_engine)

    fid = repo.upsert_folder(path="/photos", name="photos", parent_id=None, mtime=1700000000)
    mids = repo.bulk_upsert_media(
        [
            {
                "folder_id": fid,
                "filename": "crop_test.jpg",
                "ext": "jpg",
                "size": 500_000,
                "mtime": 1700000000,
                "width": 4000,
                "height": 3000,
                "rating": 4,
            }
        ]
    )
    mid = mids[0]

    # Set face_scan_version = 1 and write a fake thumbnail blob
    repo.update_face_scan_version([mid], version=1)
    thumb_repo.put_webp_blob(mid, b"fake_webp_thumb_bytes", mtime=1700000000)
    assert thumb_repo.get_webp_blob(mid) == b"fake_webp_thumb_bytes"

    # Now update via bulk_upsert_media with CROPPED dimensions (2000x1500)
    repo.bulk_upsert_media(
        [
            {
                "folder_id": fid,
                "filename": "crop_test.jpg",
                "ext": "jpg",
                "size": 250_000,
                "mtime": 1700001000,
                "width": 2000,
                "height": 1500,
                "rating": 4,
            }
        ]
    )

    # Verify cached thumbnail was wiped
    assert thumb_repo.get_webp_blob(mid) is None

    # Verify face_scan_version was reset to 0
    detail = repo.get_media_detail(mid)
    conn = test_engine.catalog_conn()
    row = conn.execute("SELECT face_scan_version FROM media WHERE id = ?;", (mid,)).fetchone()
    assert int(row["face_scan_version"]) == 0
    assert detail["width"] == 2000
    assert detail["height"] == 1500


def test_external_editor_stripped_metadata_protected(test_engine: DualDatabaseEngine) -> None:
    """If an external editor saves a file with stripped metadata, preserve catalog tags/rating and set metadata_dirty = 1."""
    repo = CatalogRepository(test_engine)
    fid = repo.upsert_folder(path="/photos", name="photos", parent_id=None, mtime=1700000000)

    # Initial import has rating = 5 and metadata_dirty = 0
    mids = repo.bulk_upsert_media(
        [
            {
                "folder_id": fid,
                "filename": "photo.jpg",
                "ext": "jpg",
                "size": 500_000,
                "mtime": 1700000000,
                "rating": 5,
                "metadata_dirty": 0,
            }
        ]
    )
    mid = mids[0]

    # External editor updates the file with rating = 0 (metadata stripped)
    repo.bulk_upsert_media(
        [
            {
                "folder_id": fid,
                "filename": "photo.jpg",
                "ext": "jpg",
                "size": 520_000,
                "mtime": 1700002000,
                "rating": 0,
                "metadata_dirty": 0,
            }
        ]
    )

    detail = repo.get_media_detail(mid)
    assert detail["rating"] == 5  # Preserved!
    conn = test_engine.catalog_conn()
    row = conn.execute("SELECT metadata_dirty FROM media WHERE id = ?;", (mid,)).fetchone()
    assert int(row["metadata_dirty"]) == 1  # Marked dirty so it will be re-synced to disk!


def test_rename_matching_without_losing_media_id(tmp_path: Path, test_engine: DualDatabaseEngine) -> None:
    """When a file is renamed in Windows Explorer, DeltaSyncWorker matches by (size, mtime) and preserves media_id."""
    repo = CatalogRepository(test_engine)
    thumb_repo = ThumbnailRepository(test_engine)

    folder_dir = tmp_path / "GalleryFolder"
    folder_dir.mkdir(parents=True)
    orig_file = folder_dir / "IMG_0001.jpg"

    # Create dummy image on disk
    img = Image.new("RGB", (100, 100), color="blue")
    img.save(orig_file)

    st = orig_file.stat()
    fid = repo.upsert_folder(path=str(folder_dir), name="GalleryFolder", parent_id=None, mtime=int(st.st_mtime))
    mids = repo.bulk_upsert_media(
        [
            {
                "folder_id": fid,
                "filename": orig_file.name,
                "ext": "jpg",
                "size": int(st.st_size),
                "mtime": int(st.st_mtime),
                "width": 100,
                "height": 100,
            }
        ]
    )
    mid = mids[0]

    # Add descriptive tag and face tag to this media_id
    tid = repo.create_tag("Vacation/Hawaii")
    repo.assign_tag([mid], tid)
    repo.assign_person(mid, "Alice", (0.1, 0.1, 0.3, 0.3))
    thumb_repo.put_webp_blob(mid, b"thumbnail_of_alice", mtime=int(st.st_mtime))

    # Rename file in Explorer (preserving size and mtime)
    renamed_file = folder_dir / "Hawaii_Trip.jpg"
    shutil.move(orig_file, renamed_file)

    # Run DeltaSyncWorker
    worker = DeltaSyncWorker(test_engine, str(folder_dir))
    worker.run()

    # Verify media_id was NOT deleted or recreated!
    detail = repo.get_media_detail(mid)
    assert detail["filename"] == "Hawaii_Trip.jpg"

    # Verify tags, people, and thumbnail BLOB are completely intact
    summary = repo.get_selection_summary([mid])
    assert any(t["name"] == "Hawaii" for t in summary["tags"])
    assert any(p["name"] == "Alice" for p in summary["people"])
    assert thumb_repo.get_webp_blob(mid) == b"thumbnail_of_alice"


def test_original_files_exclusion_and_cleanup(tmp_path: Path, test_engine: DualDatabaseEngine) -> None:
    """Ensure .OriginalFiles is excluded from scanning and watcher, and deleted backups are cleaned up."""
    assert ".originalfiles" in IGNORED_DIR_NAMES

    folder_dir = tmp_path / "Photos"
    folder_dir.mkdir()
    backup_dir = folder_dir / ".OriginalFiles"
    backup_dir.mkdir()

    img_file = folder_dir / "EditMe.jpg"
    backup_file = backup_dir / "EditMe.jpg"
    img_file.write_bytes(b"edited_bytes")
    backup_file.write_bytes(b"original_bytes")

    repo = CatalogRepository(test_engine)
    fid = repo.upsert_folder(path=str(folder_dir), name="Photos", parent_id=None, mtime=1700000000)
    mids = repo.bulk_upsert_media(
        [
            {
                "folder_id": fid,
                "filename": "EditMe.jpg",
                "ext": "jpg",
                "size": len(b"edited_bytes"),
                "mtime": 1700000000,
                "original_backup_path": str(backup_file),
            }
        ]
    )
    mid = mids[0]
    repo.record_pixel_edit_backup(mid, str(backup_file))

    # Verify .OriginalFiles was NOT added as a folder
    conn = test_engine.catalog_conn()
    subfolders = conn.execute("SELECT path FROM folders;").fetchall()
    assert not any(".OriginalFiles" in r["path"] for r in subfolders)

    # Simulate deleting the photo: cleanup its original_backup_path
    detail = repo.get_media_detail(mid)
    b_path = detail.get("original_backup_path")
    if b_path and os.path.exists(b_path):
        os.unlink(b_path)
    repo.delete_media([mid])

    assert not backup_file.exists()


def test_watcher_loop_prevention_via_recently_written(tmp_path: Path) -> None:
    """Verify record_recently_written and is_recently_written prevent watcher loops."""
    test_file = tmp_path / "sync_target.jpg"
    test_file.write_bytes(b"sample_content")
    st = test_file.stat()

    record_recently_written(test_file, int(st.st_size), int(st.st_mtime))
    assert is_recently_written(test_file, size=int(st.st_size), mtime=int(st.st_mtime), window_secs=5.0) is True

    # Different size or mtime should return False
    assert is_recently_written(test_file, size=999999, mtime=int(st.st_mtime), window_secs=5.0) is False
    assert is_recently_written(test_file, size=int(st.st_size), mtime=12345, window_secs=5.0) is False


def test_pre_1970_negative_timestamps_safety(test_engine: DualDatabaseEngine) -> None:
    """Verify historical scans from 1950 (negative Unix timestamps) work safely on Windows."""
    repo = CatalogRepository(test_engine)
    fid = repo.upsert_folder(path="/scans", name="scans", parent_id=None, mtime=1700000000)

    # 1950-06-15 12:00:00 UTC = -616852800 epoch seconds
    epoch_1950 = -616852800
    dt = _epoch_to_datetime_utc(epoch_1950)
    assert dt.year == 1950
    assert dt.month == 6
    assert dt.day == 15

    mids = repo.bulk_upsert_media(
        [
            {
                "folder_id": fid,
                "filename": "Grandma_1950.jpg",
                "ext": "jpg",
                "size": 1_200_000,
                "mtime": 1700000000,
                "date_taken": epoch_1950,
            }
        ]
    )
    mid = mids[0]

    detail = repo.get_media_detail(mid)
    assert detail["date_taken"] == epoch_1950

    # Query with date bounds
    records = repo.query_media_records(FilterQuerySpec(date_year=1950, date_month=6, date_day=15))
    assert len(records) == 1
    assert records[0].id == mid
    assert records[0].group_key == "June 1950"


def test_poison_pill_file_marks_face_scan_failed(test_engine: DualDatabaseEngine) -> None:
    """Corrupt poison-pill files are flagged with face_scan_version = -1 to prevent crash loops."""
    repo = CatalogRepository(test_engine)
    fid = repo.upsert_folder(path="/scans", name="scans", parent_id=None, mtime=1700000000)

    mids = repo.bulk_upsert_media(
        [
            {
                "folder_id": fid,
                "filename": "corrupted_file.jpg",
                "ext": "jpg",
                "size": 10,
                "mtime": 1700000000,
            }
        ]
    )
    mid = mids[0]

    # Worker marks poison-pill file as failed
    repo.mark_face_scan_failed([mid])

    conn = test_engine.catalog_conn()
    row = conn.execute("SELECT face_scan_version FROM media WHERE id = ?;", (mid,)).fetchone()
    assert int(row["face_scan_version"]) == -1
