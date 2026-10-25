"""
High-Speed Two-Stage Folder Scanner & C++ QImageReader WebP Thumbnail Worker.
- Stage 1 (Fast Pass): os.scandir directory walk + mtime diffing (~50k files/sec).
- Stage 2 (Enrichment Pass): Batched EXIF/XMP metadata, hierarchical tag, and static-image face extraction.
- ThumbnailDecodeWorker: QImageReader(setAutoTransform=True, setScaledSize=256x256) + OpenCV smart video frame filmstrip -> WebP BLOB in thumbnails.db.
"""

from __future__ import annotations

import calendar
import datetime
import math
import os
import re
import struct
import threading
import time
from pathlib import Path
from typing import Any

import cv2
from PIL import ExifTags, Image, ImageOps
from PySide6.QtCore import (
    QBuffer,
    QByteArray,
    QIODevice,
    QLoggingCategory,
    QObject,
    QRectF,
    QRunnable,
    QSize,
    Qt,
    Signal,
    Slot,
)

QLoggingCategory.setFilterRules(
    "qt.gui.imageio.jpeg.warning=false\n"
    "qt.gui.imageio.warning=false\n"
)
from PySide6.QtGui import (
    QColor,
    QImage,
    QImageIOHandler,
    QImageReader,
    QPainter,
    QTransform,
)

from legacy_reference.imgBrowserv13 import (
    CancellationFlag,
    LruCache,
    _to_degrees_from_exif_gps,
    is_frame_too_dark,
)
from src.db.connection import DualDatabaseEngine
from src.db.repository import CatalogRepository, ThumbnailRepository
from src.workers.xmp_sync_worker import (
    _parse_iso_or_exif_datetime,
    read_xmp_from_file,
)
import src.workers.xmp_sync_worker as _xmp_sync_module
import xml.etree.ElementTree as ET

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except (ImportError, Exception):
    pass

RAW_EXT_SET = {"raf", "cr2", "cr3", "nef", "arw", "dng", "orf", "rw2"}
HEIF_EXT_SET = {"heic", "heif"}
IMAGE_EXT_SET = {"png", "jpg", "jpeg", "gif", "bmp", "ico", "tiff", "tif", "webp"} | RAW_EXT_SET | HEIF_EXT_SET
VIDEO_EXT_SET = {"mp4", "avi", "mkv", "mov", "wmv", "flv"}
ALL_MEDIA_EXT_SET = IMAGE_EXT_SET | VIDEO_EXT_SET
IGNORED_DIR_NAMES = {".originalfiles", ".git", "__pycache__", "$recycle.bin", "system volume information"}

THUMB_MAX_DIM = 256


def create_filmstrip_for_qimage(
    frame_qimage: QImage, target_size: int = THUMB_MAX_DIM
) -> QImage:
    """
    Thread-safe QImage filmstrip compositor (safe to run on background QRunnable threads,
    unlike QPixmap which requires the main GUI thread). Supports both landscape and portrait videos.
    """
    is_portrait = frame_qimage.height() > frame_qimage.width()
    if is_portrait:
        filmstrip_w = 300
        filmstrip_h = 400
        strip_width = 24
        hole_size = 14
        hole_padding = 9
    else:
        filmstrip_w = 400
        filmstrip_h = 300
        strip_width = 30
        hole_size = 18
        hole_padding = 10

    canvas = QImage(filmstrip_w, filmstrip_h, QImage.Format_ARGB32)
    canvas.fill(QColor("black"))

    available_width = filmstrip_w - (2 * strip_width)
    available_height = filmstrip_h
    scaled_frame = frame_qimage.scaled(
        available_width, available_height, Qt.KeepAspectRatio, Qt.SmoothTransformation
    )

    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.Antialiasing)

    frame_x = (filmstrip_w - scaled_frame.width()) / 2
    frame_y = (filmstrip_h - scaled_frame.height()) / 2
    painter.drawImage(int(frame_x), int(frame_y), scaled_frame)

    painter.setBrush(QColor(200, 200, 200))
    painter.setPen(Qt.NoPen)
    hole_rects = [
        QRectF(hole_x_left, current_y, hole_size, hole_size)
        for current_y in range(hole_padding, filmstrip_h - hole_size, hole_size + hole_padding)
        for hole_x_left in (
            (strip_width - hole_size) / 2,
            filmstrip_w - strip_width + (strip_width - hole_size) / 2,
        )
    ]
    painter.drawRects(hole_rects)
    painter.end()

    return canvas.scaled(
        target_size, target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
    )


def _format_rational(val: Any) -> float | None:
    try:
        if isinstance(val, tuple) and len(val) == 2:
            return float(val[0]) / float(val[1]) if val[1] != 0 else None
        return float(val)
    except Exception:
        return None


def _format_exposure_str(val: Any) -> str | None:
    r = _format_rational(val)
    if r is None or r <= 0:
        return None
    if r < 1.0:
        return f"1/{int(round(1.0 / r))}s"
    return f"{r:.2f}s"


_MAC_EPOCH_OFFSET = 2082844800  # seconds between 1904-01-01 (HFS+) and 1970-01-01 (Unix)

_FILENAME_TS_RE = re.compile(
    r"(?:VID_|IMG_)?(\d{4})[-_.]?(\d{2})[-_.]?(\d{2})[\s_-]+(\d{2})[._:-](\d{2})[._:-](\d{2})"
)


def _timestamp_from_filename(stem: str) -> int | None:
    """Parses camera-style filename timestamps (e.g. '2019-01-27 13.04.39') into a naive epoch."""
    m = _FILENAME_TS_RE.search(stem)
    if not m:
        return None
    try:
        y, mo, d, hh, mi, ss = (int(g) for g in m.groups())
        return int(calendar.timegm(datetime.datetime(y, mo, d, hh, mi, ss).timetuple()))
    except (ValueError, OverflowError):
        return None


def _extract_mp4_creation_time(file_path: Path | str) -> int | None:
    """
    Walks ISO BMFF top-level atoms (never reading 'mdat' payloads) and returns the
    'mvhd' creation_time converted to a Unix epoch, or None.
    """
    try:
        with open(file_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            file_end = f.tell()
            f.seek(0)

            def _walk(start: int, end: int, want: bytes) -> tuple[int, int] | None:
                pos = start
                while pos + 8 <= end:
                    f.seek(pos)
                    hdr = f.read(8)
                    if len(hdr) < 8:
                        return None
                    size, box_type = struct.unpack(">I4s", hdr)
                    header_len = 8
                    if size == 1:
                        ext = f.read(8)
                        if len(ext) < 8:
                            return None
                        size = struct.unpack(">Q", ext)[0]
                        header_len = 16
                    elif size == 0:
                        size = end - pos
                    if size < header_len:
                        return None
                    if box_type == want:
                        return pos + header_len, min(pos + size, end)
                    pos += size
                return None

            moov = _walk(0, file_end, b"moov")
            if moov is None:
                return None
            mvhd = _walk(moov[0], moov[1], b"mvhd")
            if mvhd is None:
                return None
            f.seek(mvhd[0])
            data = f.read(12)
            if len(data) < 8:
                return None
            version = data[0]
            if version == 1:
                if len(data) < 12:
                    return None
                creation = struct.unpack(">Q", data[4:12])[0]
            else:
                creation = struct.unpack(">I", data[4:8])[0]
            if creation > _MAC_EPOCH_OFFSET:
                return int(creation - _MAC_EPOCH_OFFSET)
    except Exception:
        return None
    return None


def _extract_mp4_orientation(file_path: Path | str) -> int:
    """
    Extracts display orientation (1, 3, 6, 8) from MP4/MOV track header matrix ('tkhd').
    Defaults to 1 (normal / 0 deg) if not found.
    """
    try:
        with open(file_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            file_end = f.tell()
            f.seek(0)

            def _walk(start: int, end: int, want: bytes) -> tuple[int, int] | None:
                pos = start
                while pos + 8 <= end:
                    f.seek(pos)
                    hdr = f.read(8)
                    if len(hdr) < 8:
                        return None
                    size, box_type = struct.unpack(">I4s", hdr)
                    header_len = 8
                    if size == 1:
                        ext = f.read(8)
                        if len(ext) < 8:
                            return None
                        size = struct.unpack(">Q", ext)[0]
                        header_len = 16
                    elif size == 0:
                        size = end - pos
                    if size < header_len:
                        return None
                    if box_type == want:
                        return pos + header_len, min(pos + size, end)
                    pos += size
                return None

            moov = _walk(0, file_end, b"moov")
            if moov is None:
                return 1
            trak_pos = moov[0]
            while trak_pos + 8 <= moov[1]:
                f.seek(trak_pos)
                hdr = f.read(8)
                if len(hdr) < 8:
                    break
                trak_size, trak_type = struct.unpack(">I4s", hdr)
                trak_hlen = 8
                if trak_size == 1:
                    ext = f.read(8)
                    if len(ext) < 8:
                        break
                    trak_size = struct.unpack(">Q", ext)[0]
                    trak_hlen = 16
                elif trak_size == 0:
                    trak_size = moov[1] - trak_pos
                if trak_type == b"trak":
                    tkhd = _walk(trak_pos + trak_hlen, min(trak_pos + trak_size, moov[1]), b"tkhd")
                    if tkhd is not None:
                        f.seek(tkhd[0])
                        ver_flags = f.read(4)
                        if len(ver_flags) == 4:
                            ver = ver_flags[0]
                            matrix_offset = 40 if ver == 0 else 52
                            f.seek(tkhd[0] + matrix_offset)
                            mdata = f.read(36)
                            if len(mdata) == 36:
                                vals = struct.unpack(">9i", mdata)
                                a, b, _, c, d = vals[0], vals[1], vals[2], vals[3], vals[4]
                                if a != 65536 or b != 0 or c != 0 or d != 65536:
                                    deg = int(round(math.degrees(math.atan2(b, a)))) % 360
                                    orient = {0: 1, 90: 6, 180: 3, 270: 8}.get(deg, 1)
                                    if orient != 1:
                                        return orient
                trak_pos += trak_size
    except Exception:
        return 1
    return 1


def _find_embedded_xmp_packet(file_path: Path | str, chunk_bytes: int = 1024 * 1024) -> bytes | None:
    """Scans the first and last 1 MB of a video container for an embedded <x:xmpmeta> packet."""
    try:
        size = os.path.getsize(file_path)
        with open(file_path, "rb") as f:
            chunks = [f.read(chunk_bytes)]
            if size > chunk_bytes:
                f.seek(max(chunk_bytes, size - chunk_bytes))
                chunks.append(f.read(chunk_bytes))
        for blob in chunks:
            s = blob.find(b"<x:xmpmeta")
            if s < 0:
                continue
            e = blob.find(b"</x:xmpmeta>", s)
            if e >= 0:
                return blob[s : e + len(b"</x:xmpmeta>")]
    except Exception:
        return None
    return None


def _ms_percent_to_stars(pct: int) -> int:
    if pct <= 0:
        return 0
    if pct <= 12:
        return 1
    if pct <= 37:
        return 2
    if pct <= 62:
        return 3
    if pct <= 87:
        return 4
    return 5


_XMP_NS = {
    "x": "adobe:ns:meta/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "xmp": "http://ns.adobe.com/xap/1.0/",
    "tiff": "http://ns.adobe.com/tiff/1.0/",
    "exif": "http://ns.adobe.com/exif/1.0/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "MicrosoftPhoto": "http://ns.microsoft.com/photo/1.0/",
    "MP": "http://ns.microsoft.com/photo/1.2/",
    "MPRI": "http://ns.microsoft.com/photo/1.2/t/RegionInfo#",
    "MPReg": "http://ns.microsoft.com/photo/1.2/t/Region#",
    "mwg-rs": "http://www.metadataworkinggroup.com/schemas/regions/",
    "stDim": "http://ns.adobe.com/xap/1.0/sType/Dimensions#",
    "stArea": "http://ns.adobe.com/xmp/sType/Area#",
    "lr": "http://ns.adobe.com/lightroom/1.0/",
    "digiKam": "http://www.digikam.org/ns/1.0/",
    "Iptc4xmpCore": "http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/",
    "photoshop": "http://ns.adobe.com/photoshop/1.0/",
}


def transform_box_raw_to_display(
    rect: tuple[float, float, float, float] | list[float],
    orientation: int,
) -> tuple[float, float, float, float]:
    """
    Transforms a normalized bounding box [x, y, w, h] from raw unrotated sensor space
    to display space according to EXIF orientation.
    """
    x, y, w, h = rect
    if orientation == 6:  # 90° CW: (x, y) -> (1 - (y + h), x), (w, h) -> (h, w)
        rx, ry, rw, rh = 1.0 - (y + h), x, h, w
    elif orientation == 8:  # 90° CCW / 270° CW: (x, y) -> (y, 1 - (x + w)), (w, h) -> (h, w)
        rx, ry, rw, rh = y, 1.0 - (x + w), h, w
    elif orientation == 3:  # 180°: (x, y) -> (1 - (x + w), 1 - (y + h)), (w, h) -> (w, h)
        rx, ry, rw, rh = 1.0 - (x + w), 1.0 - (y + h), w, h
    elif orientation == 2:  # Flip horizontal
        rx, ry, rw, rh = 1.0 - (x + w), y, w, h
    elif orientation == 4:  # Flip vertical
        rx, ry, rw, rh = x, 1.0 - (y + h), w, h
    elif orientation == 5:  # Transpose
        rx, ry, rw, rh = y, x, h, w
    elif orientation == 7:  # Transverse
        rx, ry, rw, rh = 1.0 - (y + h), 1.0 - (x + w), h, w
    else:  # Orientation 1 or default
        rx, ry, rw, rh = x, y, w, h

    rx = max(0.0, min(1.0, rx))
    ry = max(0.0, min(1.0, ry))
    rw = max(0.0, min(1.0 - rx, rw))
    rh = max(0.0, min(1.0 - ry, rh))
    return (rx, ry, rw, rh)


def parse_xmp_packet_bytes(
    xmp_bytes: bytes,
    image_width: int | None = None,
    image_height: int | None = None,
    raw_width: int | None = None,
    raw_height: int | None = None,
    orientation: int | None = None,
) -> dict[str, Any]:
    """
    Parses WLPG / Lightroom / digiKam / MWG XMP XML bytes into structured metadata fields
    with 100% deterministic face region handling matching digiKam and Windows Live Photo Gallery.
    """
    result: dict[str, Any] = {
        "rating": None,
        "orientation": None,
        "date_taken": None,
        "caption": None,
        "author": None,
        "location_name": None,
        "lat": None,
        "lon": None,
        "hierarchical_tags": [],
        "people": [],
        "applied_to_dimensions": None,
    }
    try:
        start_idx = xmp_bytes.find(b"<x:xmpmeta")
        end_idx = xmp_bytes.find(b"</x:xmpmeta>")
        if start_idx == -1 or end_idx == -1:
            return result
        xml_slice = xmp_bytes[start_idx : end_idx + len(b"</x:xmpmeta>")]
        try:
            root = ET.fromstring(xml_slice)
        except Exception:
            root = ET.fromstring(xml_slice.decode("utf-8", errors="ignore"))

        raw_ms_rating = None
        for desc in root.findall(".//rdf:Description", _XMP_NS):
            rating_attr = desc.attrib.get(f"{{{_XMP_NS['xmp']}}}Rating")
            if rating_attr is not None:
                try:
                    result["rating"] = max(0, min(5, int(rating_attr)))
                except ValueError:
                    pass

            if raw_ms_rating is None:
                ms_attr = desc.attrib.get(f"{{{_XMP_NS['MicrosoftPhoto']}}}Rating")
                if ms_attr is not None:
                    raw_ms_rating = ms_attr

            orient_attr = desc.attrib.get(f"{{{_XMP_NS['tiff']}}}Orientation")
            if orient_attr is not None:
                try:
                    result["orientation"] = int(orient_attr)
                except ValueError:
                    pass

            dt_attr = desc.attrib.get(f"{{{_XMP_NS['exif']}}}DateTimeOriginal") or desc.attrib.get(f"{{{_XMP_NS['xmp']}}}CreateDate")
            if dt_attr and result["date_taken"] is None:
                result["date_taken"] = _parse_iso_or_exif_datetime(dt_attr)

        if result["rating"] is None:
            r_el = root.find(".//xmp:Rating", _XMP_NS)
            if r_el is not None and r_el.text:
                try:
                    result["rating"] = max(0, min(5, int(r_el.text.strip())))
                except ValueError:
                    pass

        if result["rating"] is None:
            if raw_ms_rating is None:
                ms_el = root.find(".//MicrosoftPhoto:Rating", _XMP_NS)
                if ms_el is not None and ms_el.text:
                    raw_ms_rating = ms_el.text.strip()
            if raw_ms_rating is not None:
                try:
                    result["rating"] = _ms_percent_to_stars(int(raw_ms_rating))
                except ValueError:
                    pass

        if result["date_taken"] is None:
            for dt_tag in (".//xmp:CreateDate", ".//photoshop:DateCreated", ".//exif:DateTimeOriginal"):
                dt_el = root.find(dt_tag, _XMP_NS)
                if dt_el is not None and dt_el.text:
                    try:
                        result["date_taken"] = _parse_iso_or_exif_datetime(dt_el.text.strip())
                        if result["date_taken"] is not None:
                            break
                    except Exception:
                        pass

        # Caption (prefer dc:description, fallback to dc:title or photoshop:Headline)
        cap_el = root.find(".//dc:description/rdf:Alt/rdf:li", _XMP_NS)
        if cap_el is not None and cap_el.text is not None:
            result["caption"] = cap_el.text.strip()
        if not result["caption"]:
            title_el = root.find(".//dc:title/rdf:Alt/rdf:li", _XMP_NS)
            if title_el is not None and title_el.text is not None:
                result["caption"] = title_el.text.strip()
        if not result["caption"]:
            hl_el = root.find(".//photoshop:Headline", _XMP_NS)
            if hl_el is not None and hl_el.text is not None:
                result["caption"] = hl_el.text.strip()

        # Author / Creator
        author_el = root.find(".//dc:creator/rdf:Seq/rdf:li", _XMP_NS)
        if author_el is not None and author_el.text:
            result["author"] = author_el.text.strip()

        # Location name
        loc_el = root.find(".//Iptc4xmpCore:Location", _XMP_NS)
        if loc_el is not None and loc_el.text:
            result["location_name"] = loc_el.text.strip()

        # GPS lat / lon
        lat_el = root.find(".//exif:GPSLatitude", _XMP_NS)
        lon_el = root.find(".//exif:GPSLongitude", _XMP_NS)
        if lat_el is not None and lat_el.text:
            try:
                result["lat"] = float(lat_el.text.strip())
            except ValueError:
                pass
        if lon_el is not None and lon_el.text:
            try:
                result["lon"] = float(lon_el.text.strip())
            except ValueError:
                pass

        # Hierarchical and descriptive tags
        tags_found: list[str] = []
        for li in root.findall(".//MicrosoftPhoto:LastKeywordXMP/rdf:Bag/rdf:li", _XMP_NS):
            if li.text and li.text.strip() and li.text.strip() not in tags_found:
                tags_found.append(li.text.strip())
        for li in root.findall(".//lr:hierarchicalSubject/rdf:Bag/rdf:li", _XMP_NS):
            if li.text and li.text.strip() and li.text.strip() not in tags_found:
                tags_found.append(li.text.strip())
        for li in root.findall(".//digiKam:TagsList/rdf:Seq/rdf:li", _XMP_NS):
            if li.text and li.text.strip() and li.text.strip() not in tags_found:
                tags_found.append(li.text.strip())
        for li in root.findall(".//dc:subject/rdf:Bag/rdf:li", _XMP_NS):
            if li.text and li.text.strip() and li.text.strip() not in tags_found:
                tags_found.append(li.text.strip())
        result["hierarchical_tags"] = tags_found

        # ---------------------------------------------------------------------
        # Deterministic Face Region Parsing:
        # 1. Strict Per-Item <rdf:li> Pairing (never separate .findall() lists)
        # 2. Prefer mwg-rs:Regions (digiKam standard). If present, ignore MP:RegionInfo.
        # 3. Fallback to MP:RegionInfo only if mwg-rs:Regions is absent.
        # ---------------------------------------------------------------------
        def _local(t: str) -> str:
            return t.split("}")[-1] if "}" in t else t

        def _attr_or_child(el: ET.Element, name: str) -> str | None:
            for k, v in el.attrib.items():
                if _local(k).lower() == name.lower():
                    return v
            for ch in el:
                if _local(ch.tag).lower() == name.lower() and ch.text:
                    return ch.text.strip()
            return None

        # Check mwg-rs:AppliedToDimensions
        applied_w: int | None = None
        applied_h: int | None = None
        dim_el = root.find(".//mwg-rs:AppliedToDimensions", _XMP_NS)
        if dim_el is not None:
            w_str = _attr_or_child(dim_el, "w") or _attr_or_child(dim_el, "width")
            h_str = _attr_or_child(dim_el, "h") or _attr_or_child(dim_el, "height")
            if w_str and h_str:
                try:
                    applied_w = int(float(w_str))
                    applied_h = int(float(h_str))
                    result["applied_to_dimensions"] = (applied_w, applied_h)
                except ValueError:
                    pass

        mwg_people: list[dict[str, Any]] = []
        rlist_parent = root.find(".//mwg-rs:RegionList", _XMP_NS)
        if rlist_parent is not None:
            # Iterate strictly over each <rdf:li> inside mwg-rs:RegionList
            for li in rlist_parent.findall(".//rdf:li", _XMP_NS) or [c for c in rlist_parent.iter() if _local(c.tag).lower() == "li"]:
                pname = _attr_or_child(li, "name")
                if not pname:
                    name_el = li.find(".//mwg-rs:Name", _XMP_NS)
                    if name_el is not None and name_el.text:
                        pname = name_el.text.strip()
                if not pname:
                    name_el = li.find(".//Name")
                    if name_el is not None and name_el.text:
                        pname = name_el.text.strip()

                area_el = li.find(".//mwg-rs:Area", _XMP_NS)
                if area_el is None:
                    area_el = li.find(".//Area")
                if area_el is None and any(_local(k).lower() in ("x", "y", "w", "h") for k in li.attrib):
                    area_el = li

                if pname and area_el is not None:
                    cx_str = _attr_or_child(area_el, "x")
                    cy_str = _attr_or_child(area_el, "y")
                    w_str = _attr_or_child(area_el, "w") or _attr_or_child(area_el, "width")
                    h_str = _attr_or_child(area_el, "h") or _attr_or_child(area_el, "height")
                    if cx_str and cy_str and w_str and h_str:
                        try:
                            f_cx = float(cx_str)
                            f_cy = float(cy_str)
                            f_w = float(w_str)
                            f_h = float(h_str)
                            # Convert center (cx, cy, w, h) to top-left (rx, ry, rw, rh)
                            rx = max(0.0, f_cx - (f_w / 2.0))
                            ry = max(0.0, f_cy - (f_h / 2.0))
                            rw = min(1.0 - rx, f_w)
                            rh = min(1.0 - ry, f_h)
                            mwg_people.append({"name": pname, "rect": (rx, ry, rw, rh)})
                        except (ValueError, TypeError):
                            pass

        effective_orient = orientation or result.get("orientation") or 1
        disp_w = image_width
        disp_h = image_height

        if mwg_people:
            # Rule for mwg-rs:Regions (Primary Source — Used by digiKam):
            # Check AppliedToDimensions against post-exif_transpose display (disp_w, disp_h):
            # - If stDim is present and (st_w == disp_w and st_h == disp_h) (or same aspect orientation), do NOT rotate.
            # - Only rotate if st_w == raw_w and st_h == raw_h while disp_w == raw_h (orientations 5, 6, 7, 8).
            should_rotate = False
            if applied_w and applied_h:
                dim_is_portrait = (applied_h > applied_w)
                disp_is_portrait = (disp_h > disp_w) if (disp_w and disp_h) else None

                if (applied_w == disp_w and applied_h == disp_h) or (
                    disp_is_portrait is not None and dim_is_portrait == disp_is_portrait
                ):
                    should_rotate = False
                elif raw_width and raw_height and (applied_w == raw_width and applied_h == raw_height) and (
                    disp_w == raw_height or effective_orient in (5, 6, 7, 8)
                ):
                    should_rotate = True
                elif disp_is_portrait is not None and dim_is_portrait != disp_is_portrait and effective_orient in (5, 6, 7, 8):
                    should_rotate = True

            if should_rotate and effective_orient in (2, 3, 4, 5, 6, 7, 8):
                result["people"] = [
                    {"name": p["name"], "rect": transform_box_raw_to_display(p["rect"], effective_orient)}
                    for p in mwg_people
                ]
            else:
                result["people"] = mwg_people

        else:
            # Fallback to MP:RegionInfo (WLPG standard)
            mp_people: list[dict[str, Any]] = []
            for reg_node in (root.find(".//MP:RegionInfo", _XMP_NS), root.find(".//MPRI:Regions", _XMP_NS)):
                if reg_node is None:
                    continue
                for li in reg_node.findall(".//rdf:li", _XMP_NS) or [c for c in reg_node.iter() if _local(c.tag).lower() == "li"]:
                    rect_str = _attr_or_child(li, "rectangle")
                    pname = _attr_or_child(li, "persondisplayname") or _attr_or_child(li, "name")
                    if not pname:
                        name_el = li.find(".//MPReg:PersonDisplayName", _XMP_NS)
                        if name_el is not None and name_el.text:
                            pname = name_el.text.strip()
                    if not rect_str:
                        rect_el = li.find(".//MPReg:Rectangle", _XMP_NS)
                        if rect_el is not None and rect_el.text:
                            rect_str = rect_el.text.strip()

                    if pname and rect_str:
                        try:
                            parts = [float(x.strip()) for x in rect_str.split(",")]
                            if len(parts) == 4:
                                rx = max(0.0, min(1.0, parts[0]))
                                ry = max(0.0, min(1.0, parts[1]))
                                rw = max(0.0, min(1.0 - rx, parts[2]))
                                rh = max(0.0, min(1.0 - ry, parts[3]))
                                mp_people.append({"name": pname, "rect": (rx, ry, rw, rh)})
                        except (ValueError, TypeError):
                            pass
                if mp_people:
                    break

            if mp_people:
                if effective_orient in (2, 3, 4, 5, 6, 7, 8):
                    result["people"] = [
                        {"name": p["name"], "rect": transform_box_raw_to_display(p["rect"], effective_orient)}
                        for p in mp_people
                    ]
                else:
                    result["people"] = mp_people
            else:
                result["people"] = []

    except Exception:
        pass
    return result


# Ensure external modules and tests using xmp_sync_worker get this deterministic parser
_xmp_sync_module.parse_xmp_packet_bytes = parse_xmp_packet_bytes


def _parse_video_xmp_packet(packet: bytes) -> dict[str, Any]:
    """Parses an embedded XMP packet (rating, date_taken, caption, tags) from a video container."""
    try:
        parsed = parse_xmp_packet_bytes(packet)
        if isinstance(parsed, dict) and (
            parsed.get("rating") is not None
            or parsed.get("date_taken") is not None
            or parsed.get("caption")
            or parsed.get("hierarchical_tags")
        ):
            return parsed
    except Exception:
        pass

    out: dict[str, Any] = {}
    text = packet.decode("utf-8", errors="ignore")
    m = re.search(r'MicrosoftPhoto:Rating\s*=\s*["\'](\d+)["\']|<MicrosoftPhoto:Rating>\s*(\d+)\s*<', text)
    if m:
        out["rating"] = _ms_percent_to_stars(int(m.group(1) or m.group(2)))
    else:
        m = re.search(r'xmp:Rating\s*=\s*["\'](\d+)["\']|<xmp:Rating>\s*(\d+)\s*<', text)
        if m:
            out["rating"] = max(0, min(5, int(m.group(1) or m.group(2))))
    m = re.search(r'xmp:CreateDate\s*=\s*["\']([^"\']+)["\']|<xmp:CreateDate>\s*([^<\s]+)\s*<', text)
    if m:
        try:
            out["date_taken"] = _parse_iso_or_exif_datetime(m.group(1) or m.group(2))
        except Exception:
            pass
    m = re.search(r"<dc:description>.*?<rdf:li[^>]*>([^<]*)</rdf:li>", text, re.S)
    if m and m.group(1).strip():
        out["caption"] = m.group(1).strip()
    sub = re.search(r"<dc:subject>(.*?)</dc:subject>", text, re.S)
    if sub:
        tags = [t.strip() for t in re.findall(r"<rdf:li[^>]*>([^<]*)</rdf:li>", sub.group(1)) if t.strip()]
        if tags:
            out["hierarchical_tags"] = tags
    return out


def extract_media_metadata(file_path: Path, is_video: bool) -> dict[str, Any]:
    """
    Extracts EXIF, XMP, and video stream metadata from `file_path` without decoding full image pixels.
    """
    meta: dict[str, Any] = {
        "width": None,
        "height": None,
        "orientation": 1,
        "date_taken": None,
        "rating": 0,
        "caption": "",
        "camera_make": None,
        "camera_model": None,
        "author": None,
        "exposure_time": None,
        "f_number": None,
        "focal_length": None,
        "iso": None,
        "lat": None,
        "lon": None,
        "duration_ms": None,
        "video_codec": None,
        "video_fps": None,
        "hierarchical_tags": [],
        "people": [],
    }

    if not is_video:
        exif_rating_val = 0
        raw_w: int | None = None
        raw_h: int | None = None
        try:
            with Image.open(file_path) as img:
                raw_w, raw_h = img.width, img.height
                meta["width"], meta["height"] = img.width, img.height
                exif_raw = img.getexif()
                if exif_raw:
                    readable = {
                        ExifTags.TAGS.get(k, k): v for k, v in exif_raw.items()
                    }
                    orient = readable.get("Orientation")
                    if isinstance(orient, int) and 1 <= orient <= 8:
                        meta["orientation"] = orient
                        if orient in (5, 6, 7, 8):
                            meta["width"], meta["height"] = img.height, img.width

                    # First, check Exif SubIFD (where cameras store DateTimeOriginal and DateTimeDigitized)
                    sub = {}
                    try:
                        exif_ifd = exif_raw.get_ifd(ExifTags.IFD.Exif)
                        if exif_ifd:
                            sub = {
                                ExifTags.TAGS.get(k, k): v for k, v in exif_ifd.items()
                            }
                    except Exception:
                        pass

                    # DateTimeOriginal is the true Date Taken. Fall back to DateTimeDigitized, then DateTime tag 306
                    dt_str = (
                        sub.get("DateTimeOriginal")
                        or readable.get("DateTimeOriginal")
                        or sub.get("DateTimeDigitized")
                        or readable.get("DateTimeDigitized")
                        or readable.get("DateTime")
                        or sub.get("DateTime")
                    )
                    if dt_str:
                        meta["date_taken"] = _parse_iso_or_exif_datetime(str(dt_str))

                    if readable.get("Make"):
                        meta["camera_make"] = str(readable["Make"]).strip("\x00 ")
                    if readable.get("Model"):
                        meta["camera_model"] = str(readable["Model"]).strip("\x00 ")
                    if readable.get("Artist"):
                        meta["author"] = str(readable["Artist"]).strip("\x00 ")

                    # Check Exif SubIFD for ExposureTime, FNumber, FocalLength, ISO
                    if sub:
                        if "ExposureTime" in sub:
                            meta["exposure_time"] = _format_exposure_str(sub["ExposureTime"])
                        if "FNumber" in sub:
                            meta["f_number"] = _format_rational(sub["FNumber"])
                        if "FocalLength" in sub:
                            meta["focal_length"] = _format_rational(sub["FocalLength"])
                        if "ISOSpeedRatings" in sub:
                            try:
                                iso_val = sub["ISOSpeedRatings"]
                                meta["iso"] = int(iso_val[0] if isinstance(iso_val, tuple) else iso_val)
                            except Exception:
                                pass

                    # Check GPS IFD
                    try:
                        gps_ifd = exif_raw.get_ifd(ExifTags.IFD.GPSInfo)
                        if gps_ifd:
                            lat_val = gps_ifd.get(2)
                            lat_ref = gps_ifd.get(1)
                            lon_val = gps_ifd.get(4)
                            lon_ref = gps_ifd.get(3)
                            if lat_val and lat_ref and lon_val and lon_ref:
                                lat_deg = _to_degrees_from_exif_gps(lat_val)
                                lon_deg = _to_degrees_from_exif_gps(lon_val)
                                if str(lat_ref).upper().startswith("S"):
                                    lat_deg = -lat_deg
                                if str(lon_ref).upper().startswith("W"):
                                    lon_deg = -lon_deg
                                meta["lat"] = lat_deg
                                meta["lon"] = lon_deg
                    except Exception:
                        pass

                    # EXIF Rating (tag 18246 = Rating, 18249 = RatingPercent)
                    r_val = exif_raw.get(18246)
                    if r_val is not None:
                        try:
                            exif_rating_val = int(r_val)
                        except Exception:
                            pass
                    if exif_rating_val == 0:
                        pct = exif_raw.get(18249)
                        if pct is not None:
                            try:
                                pct_int = int(pct)
                                if pct_int >= 88:
                                    exif_rating_val = 5
                                elif pct_int >= 63:
                                    exif_rating_val = 4
                                elif pct_int >= 38:
                                    exif_rating_val = 3
                                elif pct_int >= 13:
                                    exif_rating_val = 2
                                elif pct_int >= 1:
                                    exif_rating_val = 1
                            except Exception:
                                pass

                    # Windows XP tags (XPKeywords=40094, XPTitle=40091, XPComment=40092, XPSubject=40095, XPAuthor=40093)
                    xp_kw = exif_raw.get(40094)
                    if xp_kw:
                        try:
                            if isinstance(xp_kw, bytes):
                                kw_str = xp_kw.decode("utf-16le", errors="ignore").rstrip("\x00")
                            else:
                                kw_str = str(xp_kw).rstrip("\x00")
                            for kw in kw_str.split(";"):
                                kw_s = kw.strip()
                                if kw_s and kw_s not in meta["hierarchical_tags"]:
                                    meta["hierarchical_tags"].append(kw_s)
                        except Exception:
                            pass

                    if not meta["author"]:
                        xp_auth = exif_raw.get(40093)
                        if xp_auth:
                            try:
                                if isinstance(xp_auth, bytes):
                                    meta["author"] = xp_auth.decode("utf-16le", errors="ignore").rstrip("\x00").strip()
                                else:
                                    meta["author"] = str(xp_auth).rstrip("\x00").strip()
                            except Exception:
                                pass

                    for xp_cap_tag in (40095, 40091, 40092):
                        if not meta["caption"]:
                            xp_cap = exif_raw.get(xp_cap_tag)
                            if xp_cap:
                                try:
                                    if isinstance(xp_cap, bytes):
                                        c_s = xp_cap.decode("utf-16le", errors="ignore").rstrip("\x00").strip()
                                    else:
                                        c_s = str(xp_cap).rstrip("\x00").strip()
                                    if c_s:
                                        meta["caption"] = c_s
                                except Exception:
                                    pass

                # IPTC metadata
                try:
                    from PIL import IptcImagePlugin

                    iptc = IptcImagePlugin.getiptcinfo(img)
                    if iptc:
                        # (2, 25) Keywords
                        iptc_kw = iptc.get((2, 25))
                        if iptc_kw:
                            if isinstance(iptc_kw, (list, tuple)):
                                for k in iptc_kw:
                                    k_s = k.decode("utf-8", errors="ignore").strip() if isinstance(k, bytes) else str(k).strip()
                                    if k_s and k_s not in meta["hierarchical_tags"]:
                                        meta["hierarchical_tags"].append(k_s)
                            else:
                                k_s = iptc_kw.decode("utf-8", errors="ignore").strip() if isinstance(iptc_kw, bytes) else str(iptc_kw).strip()
                                if k_s and k_s not in meta["hierarchical_tags"]:
                                    meta["hierarchical_tags"].append(k_s)
                        # (2, 120) Caption/Abstract
                        if not meta["caption"]:
                            iptc_cap = iptc.get((2, 120))
                            if iptc_cap:
                                c_s = iptc_cap.decode("utf-8", errors="ignore").strip() if isinstance(iptc_cap, bytes) else str(iptc_cap).strip()
                                if c_s:
                                    meta["caption"] = c_s
                except Exception:
                    pass

        except Exception:
            pass

        # Parse embedded XMP packet (tags, people regions, rating, caption, date_taken)
        xmp = read_xmp_from_file(
            file_path,
            image_width=meta.get("width"),
            image_height=meta.get("height"),
            raw_width=raw_w,
            raw_height=raw_h,
            orientation=meta.get("orientation", 1),
        )
        if xmp.get("rating") is not None and int(xmp["rating"]) > 0:
            meta["rating"] = int(xmp["rating"])
        elif exif_rating_val > 0:
            meta["rating"] = exif_rating_val

        if xmp.get("orientation") is not None:
            new_orient = int(xmp["orientation"])
            if 1 <= new_orient <= 8:
                meta["orientation"] = new_orient
        if xmp.get("date_taken") is not None:
            meta["date_taken"] = xmp["date_taken"]
        if xmp.get("caption"):
            meta["caption"] = xmp["caption"]
        if xmp.get("hierarchical_tags"):
            for t in xmp["hierarchical_tags"]:
                if t not in meta["hierarchical_tags"]:
                    meta["hierarchical_tags"].append(t)
        if xmp.get("people"):
            meta["people"] = xmp["people"]

    else:
        # Video stream metadata via OpenCV
        try:
            cap = cv2.VideoCapture(str(file_path))
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                rot_meta = float(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0.0)
                orient = {0: 1, 90: 6, 180: 3, 270: 8}.get(int(round(rot_meta)) % 360, 1)
                if orient == 1:
                    orient = _extract_mp4_orientation(file_path)
                meta["orientation"] = orient
                if w > 0 and h > 0:
                    if orient in (5, 6, 7, 8):
                        meta["width"] = h
                        meta["height"] = w
                    else:
                        meta["width"] = w
                        meta["height"] = h
                fps = float(cap.get(cv2.CAP_PROP_FPS))
                frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration_sec = (frame_count / fps) if fps > 0 and frame_count > 0 else 0.0
                if fps > 0:
                    meta["video_fps"] = round(fps, 2)
                    meta["fps"] = round(fps, 2)
                if duration_sec > 0:
                    meta["duration_ms"] = int(duration_sec * 1000)
                    meta["duration"] = duration_sec
                    try:
                        size_bytes = os.path.getsize(file_path)
                        meta["bitrate_kbps"] = int((size_bytes * 8) / (duration_sec * 1000))
                    except OSError:
                        pass
                fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
                codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ")
                if codec:
                    meta["video_codec"] = codec
                cap.release()
        except Exception:
            pass

        # 1. XMP: sidecar first, then embedded packet inside the container
        xmp: dict[str, Any] = {}
        try:
            xmp = read_xmp_from_file(file_path) or {}
        except Exception:
            xmp = {}
        if not xmp.get("rating"):
            packet = _find_embedded_xmp_packet(file_path)
            if packet:
                emb = _parse_video_xmp_packet(packet)
                for k, v in emb.items():
                    if not xmp.get(k) and v not in (None, "", []):
                        xmp[k] = v

        if xmp.get("rating") is not None and int(xmp["rating"]) > 0:
            meta["rating"] = int(xmp["rating"])
        if xmp.get("caption"):
            meta["caption"] = xmp["caption"]
        if xmp.get("hierarchical_tags"):
            for t in xmp["hierarchical_tags"]:
                if t not in meta["hierarchical_tags"]:
                    meta["hierarchical_tags"].append(t)
        if xmp.get("date_taken") is not None:
            meta["date_taken"] = xmp["date_taken"]

        # 2. mvhd creation time, then 3. filename timestamp (before the mtime fallback)
        if meta["date_taken"] is None:
            meta["date_taken"] = _extract_mp4_creation_time(file_path)
        if meta["date_taken"] is None:
            meta["date_taken"] = _timestamp_from_filename(file_path.stem)

    return meta



class FolderSyncWorkerSignals(QObject):
    """Qt Signals emitted by FolderSyncWorker."""

    progress = Signal(int, str)          # processed_count, current_folder_name
    stage1_completed = Signal(int)       # total_discovered_files
    finished = Signal(int, bool)         # upserted_or_updated_count, was_completed
    error = Signal(str)


class FolderSyncWorker(QRunnable):
    """
    Two-stage progressive folder synchronization worker:
    - Stage 1 (Fast Pass): os.scandir walk diffing mtime against catalog.db and inserting new files immediately.
    - Stage 2 (Enrichment Pass): Batched EXIF/XMP extraction for new/modified files (5,000 rows/tx).
    """

    BATCH_SIZE = 5000

    def __init__(
        self,
        engine: DualDatabaseEngine,
        root_path: Path | str,
        include_subfolders: bool = True,
        cancel_flag: CancellationFlag | None = None,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.root_path = Path(root_path).resolve()
        self.include_subfolders = include_subfolders
        self.cancel_flag = cancel_flag or CancellationFlag()
        self.signals = FolderSyncWorkerSignals()

    @Slot()
    def run(self) -> None:
        repo = CatalogRepository(self.engine)
        needs_enrichment: list[tuple[int, Path, bool, int]] = []  # (folder_id, file_path, is_video, mtime)
        total_seen = 0
        last_emit = time.perf_counter()

        try:
            if not self.root_path.exists() or not self.root_path.is_dir():
                self.signals.finished.emit(0, False)
                return

            # Preload existing folder map: path -> id
            existing_folders = {
                os.path.normcase(f["path"]): f["id"] for f in repo.get_folders()
            }

            # Walk directories using fast iterative os.scandir stack
            dir_stack: list[tuple[Path, int | None]] = [(self.root_path, None)]

            while dir_stack and not self.cancel_flag.is_set:
                curr_dir, parent_fid = dir_stack.pop(0)
                try:
                    dir_stat = curr_dir.stat()
                except OSError:
                    continue

                norm_dir_str = str(curr_dir)
                folder_id = repo.upsert_folder(
                    path=norm_dir_str,
                    name=curr_dir.name or norm_dir_str,
                    parent_id=parent_fid,
                    mtime=int(dir_stat.st_mtime),
                )
                existing_folders[os.path.normcase(norm_dir_str)] = folder_id

                # Load existing (filename_lower -> (id, mtime, width)) for this folder
                conn = self.engine.catalog_conn()
                existing_media = {
                    str(r["filename"]).lower(): (int(r["id"]), int(r["mtime"]), r["width"])
                    for r in conn.execute(
                        "SELECT id, filename, mtime, width FROM media WHERE folder_id = ?;",
                        (folder_id,),
                    ).fetchall()
                }

                fast_batch: list[dict[str, Any]] = []
                subdirs: list[tuple[Path, int | None]] = []

                try:
                    with os.scandir(curr_dir) as it:
                        for entry in it:
                            if self.cancel_flag.is_set:
                                break
                            if entry.is_dir(follow_symlinks=False):
                                if (
                                    self.include_subfolders
                                    and entry.name.lower() not in IGNORED_DIR_NAMES
                                ):
                                    subdirs.append((Path(entry.path), folder_id))
                            elif entry.is_file(follow_symlinks=False):
                                ext = Path(entry.name).suffix.lower().lstrip(".")
                                if ext not in ALL_MEDIA_EXT_SET:
                                    continue
                                try:
                                    st = entry.stat(follow_symlinks=False)
                                except OSError:
                                    continue

                                mtime_int = int(st.st_mtime)
                                size_int = int(st.st_size)
                                is_vid = 1 if ext in VIDEO_EXT_SET else 0
                                total_seen += 1

                                prev = existing_media.get(entry.name.lower())
                                if prev is not None and prev[1] == mtime_int and prev[2] is not None:
                                    # Unchanged and already enriched — skip!
                                    continue

                                if prev is None:
                                    fast_batch.append(
                                        {
                                            "folder_id": folder_id,
                                            "filename": entry.name,
                                            "ext": ext,
                                            "size": size_int,
                                            "mtime": mtime_int,
                                            "date_taken": mtime_int,
                                            "is_video": is_vid,
                                        }
                                    )
                                needs_enrichment.append(
                                    (folder_id, Path(entry.path), bool(is_vid), mtime_int)
                                )

                                now = time.perf_counter()
                                if now - last_emit > 0.25:
                                    self.signals.progress.emit(total_seen, curr_dir.name)
                                    last_emit = now
                except OSError:
                    continue

                if fast_batch:
                    for i in range(0, len(fast_batch), self.BATCH_SIZE):
                        if self.cancel_flag.is_set:
                            break
                        repo.bulk_upsert_media(fast_batch[i : i + self.BATCH_SIZE])

                if subdirs:
                    subdirs.sort(key=lambda x: x[0].name.lower())
                    dir_stack = subdirs + dir_stack

            self.signals.stage1_completed.emit(total_seen)

            # Stage 2: Deep EXIF / XMP Enrichment Pass in batches of BATCH_SIZE
            enriched_count = 0
            enrichment_rows: list[dict[str, Any]] = []
            tag_and_people_queue: list[tuple[int, str, list[str], list[dict[str, Any]]]] = []

            for folder_id, fpath, is_vid, mtime_int in needs_enrichment:
                if self.cancel_flag.is_set:
                    break
                try:
                    size_int = int(fpath.stat().st_size)
                except OSError:
                    continue

                meta = extract_media_metadata(fpath, is_vid)
                ext = fpath.suffix.lower().lstrip(".")
                enrichment_rows.append(
                    {
                        "folder_id": folder_id,
                        "filename": fpath.name,
                        "ext": ext,
                        "size": size_int,
                        "mtime": mtime_int,
                        "date_taken": meta["date_taken"] if meta["date_taken"] is not None else mtime_int,
                        "rating": meta["rating"],
                        "width": meta["width"],
                        "height": meta["height"],
                        "orientation": meta["orientation"],
                        "lat": meta["lat"],
                        "lon": meta["lon"],
                        "caption": meta["caption"],
                        "camera_make": meta["camera_make"],
                        "camera_model": meta["camera_model"],
                        "author": meta["author"],
                        "exposure_time": meta["exposure_time"],
                        "f_number": meta["f_number"],
                        "focal_length": meta["focal_length"],
                        "iso": meta["iso"],
                        "is_video": 1 if is_vid else 0,
                        "duration_ms": meta["duration_ms"],
                        "video_codec": meta["video_codec"],
                        "video_fps": meta["video_fps"],
                        "metadata_dirty": 0,
                    }
                )
                if meta["hierarchical_tags"] or meta["people"]:
                    tag_and_people_queue.append(
                        (folder_id, fpath.name, meta["hierarchical_tags"], meta["people"])
                    )

                if len(enrichment_rows) >= self.BATCH_SIZE:
                    self._flush_enrichment_batch(repo, enrichment_rows, tag_and_people_queue)
                    enriched_count += len(enrichment_rows)
                    enrichment_rows.clear()
                    tag_and_people_queue.clear()

            if enrichment_rows and not self.cancel_flag.is_set:
                self._flush_enrichment_batch(repo, enrichment_rows, tag_and_people_queue)
                enriched_count += len(enrichment_rows)

            was_completed = not self.cancel_flag.is_set
            if was_completed and enriched_count > 1000:
                try:
                    self.engine.analyze_catalog()
                except Exception:
                    pass
            self.signals.finished.emit(enriched_count, was_completed)

        except Exception as exc:
            if not self.cancel_flag.is_set:
                self.signals.error.emit(str(exc))
            self.signals.finished.emit(0, False)
        finally:
            if threading.current_thread() is not threading.main_thread():
                self.engine.close_thread_connections()

    def _flush_enrichment_batch(
        self,
        repo: CatalogRepository,
        rows: list[dict[str, Any]],
        tag_people_items: list[tuple[int, str, list[str], list[dict[str, Any]]]],
    ) -> None:
        upserted_ids = repo.bulk_upsert_media(rows)
        if not tag_people_items:
            return

        # Map (folder_id, filename) -> media_id
        key_to_id = {
            (r["folder_id"], r["filename"]): mid
            for r, mid in zip(rows, upserted_ids)
        }
        cleaned_ids: list[int] = []

        with self.engine.catalog_transaction() as _:
            for folder_id, fname, h_tags, people_list in tag_people_items:
                mid = key_to_id.get((folder_id, fname))
                if mid is None:
                    continue
                for tag_path in h_tags:
                    norm_path = str(tag_path).replace("|", "/").strip("/")
                    if norm_path.lower().startswith(("face/", "peopletags/")):
                        continue
                    tid = repo.create_tag(norm_path)
                    repo.assign_tag([mid], tid)
                for p_entry in people_list:
                    repo.assign_person(mid, p_entry["name"], rect=p_entry.get("rect", (0, 0, 0, 0)))
                cleaned_ids.append(mid)

            # Tags/people read during initial scan came from the file itself, so mark metadata_dirty = 0
            if cleaned_ids:
                repo.mark_media_clean(cleaned_ids)


class ThumbnailDecodeWorkerSignals(QObject):
    """Qt Signals emitted by ThumbnailDecodeWorker."""

    thumbnail_ready = Signal(int, QImage)   # media_id, 256x256 QImage
    error = Signal(int, str)                # media_id, error_message


class ThumbnailDecodeWorker(QRunnable):
    """
    Background worker that loads or generates a 256x256 WebP thumbnail in thumbnails.db.
    Strictly uses QImage (never QPixmap) on worker threads.
    """

    def __init__(
        self,
        engine: DualDatabaseEngine,
        media_id: int,
        file_path: str | Path,
        mtime: int,
        is_video: bool = False,
        orientation: int = 1,
        cancel_flag: CancellationFlag | None = None,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.media_id = int(media_id)
        self.file_path = Path(file_path)
        self.mtime = int(mtime)
        self.is_video = bool(is_video)
        self.orientation = int(orientation or 1)
        self.cancel_flag = cancel_flag or CancellationFlag()
        self.signals = ThumbnailDecodeWorkerSignals()

    @Slot()
    def run(self) -> None:
        if self.cancel_flag.is_set:
            return
        thumb_repo = ThumbnailRepository(self.engine)
        try:
            # 1. Check thumbnails.db for an up-to-date WebP blob
            cached_blob = thumb_repo.get_webp_blob(self.media_id, expected_mtime=self.mtime)
            if cached_blob is not None:
                qimg = QImage.fromData(cached_blob, "WEBP")
                if not qimg.isNull():
                    if not self.cancel_flag.is_set:
                        self.signals.thumbnail_ready.emit(self.media_id, qimg)
                    return

            if self.cancel_flag.is_set:
                return

            # 2. Decode from disk
            qimg = (
                self._decode_video_thumbnail()
                if self.is_video
                else self._decode_image_thumbnail()
            )
            if qimg is None or qimg.isNull():
                if not self.cancel_flag.is_set:
                    self.signals.error.emit(
                        self.media_id, f"Failed to decode thumbnail for {self.file_path}"
                    )
                return

            if self.cancel_flag.is_set:
                return

            # 3. Compress to WebP BLOB (~8-12 KB) and store in thumbnails.db
            byte_array = QByteArray()
            buffer = QBuffer(byte_array)
            buffer.open(QIODevice.WriteOnly)
            qimg.save(buffer, "WEBP", 80)
            buffer.close()
            webp_bytes = bytes(byte_array.data())

            if webp_bytes:
                thumb_repo.put_webp_blob(
                    self.media_id,
                    webp_bytes,
                    self.mtime,
                    width=qimg.width(),
                    height=qimg.height(),
                )

            if not self.cancel_flag.is_set:
                self.signals.thumbnail_ready.emit(self.media_id, qimg)

        except Exception as exc:
            if not self.cancel_flag.is_set:
                self.signals.error.emit(self.media_id, str(exc))
        finally:
            if threading.current_thread() is not threading.main_thread():
                self.engine.close_thread_connections()

    def _decode_image_thumbnail(self) -> QImage | None:
        """Uses rawpy for RAW camera formats, and QImageReader/Pillow for standard formats."""
        ext = self.file_path.suffix.lstrip(".").lower()
        if ext in RAW_EXT_SET:
            try:
                import rawpy
                with rawpy.imread(str(self.file_path)) as raw:
                    qimg = None
                    try:
                        thumb = raw.extract_thumb()
                        if thumb.format == rawpy.ThumbFormat.JPEG:
                            qimg = QImage.fromData(thumb.data, "JPEG")
                        elif thumb.format == rawpy.ThumbFormat.BITMAP:
                            h, w, ch = thumb.data.shape
                            qimg = QImage(thumb.data.data, w, h, ch * w, QImage.Format_RGB888).copy()
                    except Exception:
                        qimg = None

                    if qimg is None or qimg.isNull():
                        rgb = raw.postprocess(half_size=True, use_camera_wb=True)
                        h, w, ch = rgb.shape
                        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()

                    if qimg is not None and not qimg.isNull():
                        if qimg.width() > THUMB_MAX_DIM or qimg.height() > THUMB_MAX_DIM:
                            qimg = qimg.scaled(
                                THUMB_MAX_DIM,
                                THUMB_MAX_DIM,
                                Qt.KeepAspectRatio,
                                Qt.SmoothTransformation,
                            )
                        return qimg
            except Exception:
                pass

        reader = QImageReader(str(self.file_path))
        reader.setAutoTransform(True)
        orig_size = reader.size()
        if orig_size.isValid() and (
            orig_size.width() > THUMB_MAX_DIM or orig_size.height() > THUMB_MAX_DIM
        ):
            scaled_size = orig_size.scaled(
                QSize(THUMB_MAX_DIM, THUMB_MAX_DIM), Qt.KeepAspectRatio
            )
            reader.setScaledSize(scaled_size)

        qimg = reader.read()
        disk_trans = reader.transformation()
        del reader
        if not qimg.isNull():
            # Reconcile disk EXIF transformation with self.orientation from catalog.db
            disk_orient = 1
            if disk_trans == QImageIOHandler.Transformation.TransformationRotate90:
                disk_orient = 6
            elif disk_trans == QImageIOHandler.Transformation.TransformationRotate180:
                disk_orient = 3
            elif disk_trans == QImageIOHandler.Transformation.TransformationRotate270:
                disk_orient = 8

            orient_to_angle = {1: 0, 6: 90, 3: 180, 8: 270}
            target_orient = int(self.orientation or 1)
            delta_angle = (
                orient_to_angle.get(target_orient, 0)
                - orient_to_angle.get(disk_orient, 0)
            ) % 360
            if delta_angle != 0:
                rot_transform = QTransform().rotate(delta_angle)
                qimg = qimg.transformed(rot_transform, Qt.SmoothTransformation)

            if qimg.width() > THUMB_MAX_DIM or qimg.height() > THUMB_MAX_DIM:
                qimg = qimg.scaled(
                    THUMB_MAX_DIM,
                    THUMB_MAX_DIM,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            return qimg

        # Fallback via Pillow for CMYK or uncommon formats
        try:
            with Image.open(self.file_path) as pil_img:
                pil_img = ImageOps.exif_transpose(pil_img)
                if pil_img.mode not in ("RGB", "RGBA"):
                    pil_img = pil_img.convert("RGBA")
                pil_img.thumbnail((THUMB_MAX_DIM, THUMB_MAX_DIM), Image.Resampling.BILINEAR)
                raw_bytes = pil_img.tobytes("raw", pil_img.mode)
                fmt = (
                    QImage.Format_RGBA8888
                    if pil_img.mode == "RGBA"
                    else QImage.Format_RGB888
                )
                return QImage(
                    raw_bytes, pil_img.width, pil_img.height, fmt
                ).copy()
        except Exception:
            return None

    def _decode_video_thumbnail(self) -> QImage | None:
        """Samples a non-dark frame at 15% -> 50% -> 1000ms, respects orientation, and composites a perforated filmstrip."""
        cap = cv2.VideoCapture(str(self.file_path))
        if not cap.isOpened():
            return None
        try:
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            duration_ms = (total_frames / fps) * 1000.0 if fps > 0 else 0.0

            good_frame = None
            for ratio in (0.15, 0.50):
                if self.cancel_flag.is_set:
                    return None
                seek_ms = int(duration_ms * ratio) if duration_ms > 1000 else 0
                cap.set(cv2.CAP_PROP_POS_MSEC, seek_ms)
                ret, frame = cap.read()
                if ret and not is_frame_too_dark(frame):
                    good_frame = frame
                    break

            if good_frame is None and not self.cancel_flag.is_set:
                cap.set(cv2.CAP_PROP_POS_MSEC, 1000 if duration_ms > 1000 else 0)
                ret, frame = cap.read()
                if ret and frame is not None:
                    good_frame = frame

            if good_frame is None:
                return None

            # Reconcile frame rotation with orientation metadata if OpenCV did not auto-rotate
            target_orient = int(self.orientation or 1)
            if target_orient == 1 and not cap.get(cv2.CAP_PROP_ORIENTATION_AUTO):
                target_orient = _extract_mp4_orientation(self.file_path)

            if target_orient in (5, 6, 7, 8) and good_frame.shape[1] > good_frame.shape[0]:
                if target_orient == 6:
                    good_frame = cv2.rotate(good_frame, cv2.ROTATE_90_CLOCKWISE)
                elif target_orient == 8:
                    good_frame = cv2.rotate(good_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif target_orient == 3 and not cap.get(cv2.CAP_PROP_ORIENTATION_AUTO):
                good_frame = cv2.rotate(good_frame, cv2.ROTATE_180)

            rgb = cv2.cvtColor(good_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            frame_qimg = QImage(
                rgb.data, w, h, bytes_per_line, QImage.Format_RGB888
            ).copy()
            return create_filmstrip_for_qimage(frame_qimg, target_size=THUMB_MAX_DIM)
        finally:
            cap.release()
