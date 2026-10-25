"""Tests for video metadata extraction (mvhd, embedded XMP, filename) and the Information pane."""

from __future__ import annotations

import calendar
import datetime
import struct
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from src.workers.scanner_thumb_worker import (
    _extract_mp4_creation_time,
    _timestamp_from_filename,
    extract_media_metadata,
)

MAC_EPOCH_OFFSET = 2082844800
EXPECTED_TS = calendar.timegm(datetime.datetime(2019, 1, 27, 13, 4, 39).timetuple())


def _box(box_type: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(payload), box_type) + payload


def _mp4_with_mvhd(unix_ts: int, version: int = 0) -> bytes:
    mac = unix_ts + MAC_EPOCH_OFFSET
    if version == 0:
        body = bytes([0, 0, 0, 0]) + struct.pack(">I", mac) + struct.pack(">I", mac) + b"\x00" * 80
    else:
        body = bytes([1, 0, 0, 0]) + struct.pack(">Q", mac) + struct.pack(">Q", mac) + b"\x00" * 80
    ftyp = _box(b"ftyp", b"isom\x00\x00\x02\x00isomiso2")
    mdat = _box(b"mdat", b"\x00" * 64)
    moov = _box(b"moov", _box(b"mvhd", body))
    return ftyp + mdat + moov


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize("version", [0, 1])
def test_mp4_mvhd_creation_time(tmp_path: Path, version: int) -> None:
    p = tmp_path / "clip.mp4"
    p.write_bytes(_mp4_with_mvhd(EXPECTED_TS, version))
    assert _extract_mp4_creation_time(p) == EXPECTED_TS


def test_mp4_without_moov_returns_none(tmp_path: Path) -> None:
    p = tmp_path / "bad.mp4"
    p.write_bytes(_box(b"ftyp", b"isom") + _box(b"mdat", b"\x00" * 16))
    assert _extract_mp4_creation_time(p) is None


def test_embedded_xmp_rating_in_mp4(tmp_path: Path) -> None:
    xmp = (
        b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF '
        b'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        b'<rdf:Description rdf:about="" '
        b'xmlns:MicrosoftPhoto="http://ns.microsoft.com/photo/1.0/" '
        b'MicrosoftPhoto:Rating="75"/></rdf:RDF></x:xmpmeta>'
    )
    p = tmp_path / "rated.mp4"
    p.write_bytes(_box(b"ftyp", b"isom") + _box(b"mdat", b"\x00" * 128) + _box(b"uuid", b"\x00" * 16 + xmp))
    meta = extract_media_metadata(p, True)
    assert meta["rating"] == 4


def test_filename_timestamp_fallback(tmp_path: Path) -> None:
    assert _timestamp_from_filename("2019-01-27 13.04.39") == EXPECTED_TS
    assert _timestamp_from_filename("VID_20190127_13-04-39") == EXPECTED_TS
    assert _timestamp_from_filename("holiday") is None

    p = tmp_path / "2019-01-27 13.04.39.mp4"
    p.write_bytes(b"not a real video")
    meta = extract_media_metadata(p, True)
    assert meta["date_taken"] == EXPECTED_TS


def test_mvhd_preferred_over_filename(tmp_path: Path) -> None:
    other = calendar.timegm(datetime.datetime(2020, 5, 5, 5, 5, 5).timetuple())
    p = tmp_path / "2019-01-27 13.04.39.mp4"
    p.write_bytes(_mp4_with_mvhd(other))
    assert extract_media_metadata(p, True)["date_taken"] == other


def _summary(is_video: bool) -> dict:
    return {
        "count": 1,
        "id": 1,
        "filename": "x.mp4" if is_video else "x.jpg",
        "date_taken": EXPECTED_TS,
        "mtime": EXPECTED_TS,
        "size": 11_861_000,
        "rating": 4,
        "flag": 0,
        "width": 1920,
        "height": 1080,
        "camera_make": "Canon",
        "camera_model": "EOS",
        "exposure_time": "1/100s",
        "f_number": 2.8,
        "focal_length": 50.0,
        "iso": 200,
        "is_video": is_video,
        "duration": 8.0,
        "fps": 30.0,
        "bitrate_kbps": 11861,
        "tags": [],
        "people": [],
        "confirmed_people": [],
        "unconfirmed_people": [],
    }


def test_info_pane_toggles_video_and_photo_rows(qapp: QApplication) -> None:
    from src.ui.right_tag_dock import TagAndCaptionDock

    dock = TagAndCaptionDock()
    dock.show()

    dock.bind_selection([1], _summary(True))
    form = dock._info_form
    assert form.isRowVisible(dock.lbl_duration)
    assert form.isRowVisible(dock.lbl_bitrate)
    assert form.isRowVisible(dock.lbl_frame_rate)
    assert not form.isRowVisible(dock.lbl_camera)
    assert not form.isRowVisible(dock.stack_author)
    assert not form.isRowVisible(dock.lbl_exposure)
    assert not form.isRowVisible(dock.lbl_aperture)
    assert not form.isRowVisible(dock.lbl_focal_length)
    assert not form.isRowVisible(dock.lbl_iso)
    assert dock.lbl_duration.text() == "00:00:08"
    assert dock.lbl_bitrate.text() == "11861kbps"
    assert dock.lbl_frame_rate.text() == "30 frames/sec"

    dock.bind_selection([2], _summary(False))
    assert not form.isRowVisible(dock.lbl_duration)
    assert not form.isRowVisible(dock.lbl_bitrate)
    assert not form.isRowVisible(dock.lbl_frame_rate)
    for w in (dock.lbl_camera, dock.stack_author, dock.lbl_exposure, dock.lbl_aperture, dock.lbl_focal_length, dock.lbl_iso):
        assert form.isRowVisible(w)
    dock.close()


def _mp4_with_tkhd_matrix(a: int, b: int, c: int, d: int) -> bytes:
    tkhd_prefix = b"\x00" * 40
    matrix = struct.pack(">9i", a, b, 0, c, d, 0, 0, 0, 0x40000000)
    tkhd_suffix = struct.pack(">II", 1280 << 16, 720 << 16)
    tkhd_box = _box(b"tkhd", tkhd_prefix + matrix + tkhd_suffix)
    trak_box = _box(b"trak", tkhd_box)
    moov_box = _box(b"moov", trak_box)
    return _box(b"ftyp", b"isom") + moov_box


def test_mp4_orientation_extraction(tmp_path: Path) -> None:
    from src.workers.scanner_thumb_worker import _extract_mp4_orientation

    p90 = tmp_path / "rot90.mp4"
    p90.write_bytes(_mp4_with_tkhd_matrix(0, 65536, -65536, 0))
    assert _extract_mp4_orientation(p90) == 6

    p180 = tmp_path / "rot180.mp4"
    p180.write_bytes(_mp4_with_tkhd_matrix(-65536, 0, 0, -65536))
    assert _extract_mp4_orientation(p180) == 3

    p270 = tmp_path / "rot270.mp4"
    p270.write_bytes(_mp4_with_tkhd_matrix(0, -65536, 65536, 0))
    assert _extract_mp4_orientation(p270) == 8

    p0 = tmp_path / "rot0.mp4"
    p0.write_bytes(_mp4_with_tkhd_matrix(65536, 0, 0, 65536))
    assert _extract_mp4_orientation(p0) == 1


def test_video_filmstrip_portrait_and_landscape(qapp: QApplication) -> None:
    from PySide6.QtGui import QColor, QImage
    from src.workers.scanner_thumb_worker import create_filmstrip_for_qimage

    portrait_frame = QImage(720, 1280, QImage.Format_RGB888)
    portrait_frame.fill(QColor("blue"))
    thumb_portrait = create_filmstrip_for_qimage(portrait_frame, target_size=256)
    assert thumb_portrait.height() > thumb_portrait.width()
    assert thumb_portrait.height() == 256

    landscape_frame = QImage(1280, 720, QImage.Format_RGB888)
    landscape_frame.fill(QColor("green"))
    thumb_landscape = create_filmstrip_for_qimage(landscape_frame, target_size=256)
    assert thumb_landscape.width() > thumb_landscape.height()
    assert thumb_landscape.width() == 256
