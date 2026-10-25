"""
WorldMapView (QWidget) for PhotoGallery64.
Queries SQLite's R*Tree spatial index (`media_rtree`) via `CatalogRepository.query_map_clusters`
in <5ms as the map bounding box changes, clusters nearby GPS markers, and allows clicking
any pin/cluster or filtering the main thumbnail grid to the active map bounding box.
Provides both an interactive Leaflet OpenStreetMap view via `QWebEngineView` + `QWebChannel`
and a zero-dependency native QPainter interactive vector map canvas that runs in offline/headless
environments without GPU/network overhead.
"""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Sequence
from typing import Any

from PySide6.QtCore import QFile, QObject, QPointF, QRectF, QSize, Qt, Signal, Slot
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from src.db.repository import CatalogRepository

# Detect QtWebEngine and QtWebChannel availability with environment override
HAS_WEBENGINE = False
if os.environ.get("GALLERY_NO_WEBENGINE") != "1" and os.environ.get("QT_QPA_PLATFORM") != "offscreen":
    try:
        import PySide6.QtWebChannel  # noqa: F401
        from PySide6.QtWebChannel import QWebChannel
        from PySide6.QtWebEngineWidgets import QWebEngineView
        HAS_WEBENGINE = True
    except Exception:
        HAS_WEBENGINE = False


def get_qwebchannel_js() -> str:
    """Retrieves Qt's built-in qwebchannel.js script for inlining."""
    try:
        import PySide6.QtWebChannel  # noqa: F401
        f = QFile(":/qtwebchannel/qwebchannel.js")
        if f.open(QFile.ReadOnly):
            data = bytes(f.readAll()).decode("utf-8")
            f.close()
            return data
    except Exception:
        pass
    return ""


def cluster_gps_points(
    points: Sequence[dict[str, Any]], grid_degrees: float = 2.0
) -> list[dict[str, Any]]:
    """
    Fast O(N) spatial grid clustering of geotagged media items.
    Returns cluster dicts with (lat, lon, count, media_ids, location_name).
    """
    if not points:
        return []
    if "media_ids" in points[0] and "count" in points[0]:
        clusters = list(points)
        clusters.sort(key=lambda c: c["count"], reverse=True)
        return clusters
    cell_size = max(0.01, float(grid_degrees))
    buckets: dict[tuple[int, int], list[dict[str, Any]]] = {}

    for pt in points:
        lat = float(pt["lat"])
        lon = float(pt["lon"])
        gx = int(lon // cell_size)
        gy = int(lat // cell_size)
        buckets.setdefault((gx, gy), []).append(pt)

    clusters: list[dict[str, Any]] = []
    for items in buckets.values():
        cnt = len(items)
        avg_lat = sum(float(x["lat"]) for x in items) / cnt
        avg_lon = sum(float(x["lon"]) for x in items) / cnt
        mids = [int(x["id"]) for x in items]
        loc_names = [str(x.get("location_name") or "").strip() for x in items if x.get("location_name")]
        label = loc_names[0] if loc_names else f"{avg_lat:.2f}°, {avg_lon:.2f}°"
        clusters.append(
            {
                "lat": avg_lat,
                "lon": avg_lon,
                "count": cnt,
                "media_ids": mids,
                "label": label,
            }
        )
    clusters.sort(key=lambda c: c["count"], reverse=True)
    return clusters


def _clusters_to_json(clusters: Sequence[dict[str, Any]]) -> str:
    """Serializes clusters to JSON while stripping non-serializable objects like QPixmap."""
    clean = [
        {k: v for k, v in c.items() if k != "pixmap" and not isinstance(v, QPixmap)}
        for c in clusters
    ]
    return json.dumps(clean)


def generate_leaflet_html(
    clusters: Sequence[dict[str, Any]], qwebchannel_js: str | None = None
) -> str:
    """Generates a self-contained Leaflet HTML document with clustered GPS pins and QWebChannel bridge."""
    markers_json = _clusters_to_json(clusters)
    bridge_script = qwebchannel_js if qwebchannel_js is not None else get_qwebchannel_js()

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>PhotoGallery64 World Map</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html, body, #map {{ margin: 0; padding: 0; width: 100%; height: 100%; background: #D9ECFA; font-family: 'Segoe UI', -apple-system, sans-serif; }}
  .leaflet-cluster-pin {{
    background: transparent;
    border: none;
  }}
  .leaflet-popup-content-wrapper {{
    border-radius: 4px;
    box-shadow: 0 3px 10px rgba(0,0,0,0.3);
  }}
  .map-pin-badge {{
    position: absolute;
    top: -4px;
    right: -4px;
    background: #0066CC;
    color: #FFFFFF;
    border: 1.5px solid #FFFFFF;
    border-radius: 10px;
    padding: 0 5px;
    font-size: 10px;
    font-weight: bold;
    line-height: 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.3);
  }}
</style>
<script>
{bridge_script}
</script>
</head>
<body>
<div id="map"></div>
<script>
  let map = null;
  let clusterLayer = null;
  let isInternalMove = false;
  window.pendingClusters = {markers_json};

  function initMap() {{
    if (typeof L === 'undefined') {{
      document.getElementById('map').innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#1E395B;font-size:13px;text-align:center;padding:20px;">Leaflet tile service unavailable or offline.<br>Switch to the Vector Offline map above for built-in offline navigation.</div>';
      return;
    }}
    map = L.map('map', {{
      zoomControl: true,
      minZoom: 1,
      maxZoom: 18
    }}).setView([25.0, 0.0], 2);

    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap contributors'
    }}).addTo(map);

    clusterLayer = L.layerGroup().addTo(map);

    let boundsTimer = null;
    map.on('moveend', function() {{
      if (isInternalMove) {{
        isInternalMove = false;
        return;
      }}
      if (boundsTimer) clearTimeout(boundsTimer);
      boundsTimer = setTimeout(function() {{
        if (window.qtBridge) {{
          const b = map.getBounds();
          window.qtBridge.onBoundsChanged(b.getSouth(), b.getNorth(), b.getWest(), b.getEast());
        }}
      }}, 150);
    }});

    if (window.pendingClusters) {{
      updateClusters(window.pendingClusters);
      window.pendingClusters = null;
    }}
  }}

  function updateClusters(clustersData) {{
    if (!clusterLayer) {{
      window.pendingClusters = clustersData;
      return;
    }}
    clusterLayer.clearLayers();
    const clusters = (typeof clustersData === 'string') ? JSON.parse(clustersData) : clustersData;
    if (!clusters || !clusters.length) return;

    clusters.forEach(c => {{
      const count = c.count || 1;
      const hasThumb = Boolean(c.thumb_url);
      const radius = Math.min(32, Math.max(16, 15 + Math.pow(count, 0.35) * 2.5));
      const size = radius * 2;
      let innerHtml = '';
      if (hasThumb) {{
        innerHtml = `<div class="map-cluster-pin-container" style="position:relative;width:${{size}}px;height:${{size}}px;cursor:pointer;">` +
          `<div style="width:${{size}}px;height:${{size}}px;border-radius:50%;overflow:hidden;border:2.5px solid #ffffff;box-shadow:0 3px 8px rgba(0,0,0,0.45);background:#0066CC;">` +
            `<img src="${{c.thumb_url}}" style="width:100%;height:100%;object-fit:cover;display:block;" onerror="this.style.display='none'"/>` +
          `</div>` +
          (count > 1 ? `<div class="map-pin-badge">${{count}}</div>` : '') +
        `</div>`;
      }} else {{
        innerHtml = `<div style="width:${{size}}px;height:${{size}}px;line-height:${{size}}px;border-radius:50%;background:#0066CC;color:#ffffff;border:2px solid #ffffff;text-align:center;font-weight:bold;font-size:11px;box-shadow:0 2px 6px rgba(0,0,0,0.35);cursor:pointer;">${{count}}</div>`;
      }}
      const icon = L.divIcon({{
        className: 'leaflet-cluster-pin',
        html: innerHtml,
        iconSize: [size, size],
        iconAnchor: [radius, radius]
      }});
      const marker = L.marker([c.lat, c.lon], {{ icon: icon }}).addTo(clusterLayer);
      const label = c.label || `${{c.lat.toFixed(2)}}°, ${{c.lon.toFixed(2)}}°`;
      const clusterPayload = JSON.stringify({{ media_ids: c.media_ids || [], label: label, count: count }});
      const popupBtnText = (count === 1) ? 'View Photo' : 'Filter Gallery to Location';
      const popupHtml = `<div style="font-family:'Segoe UI',sans-serif;font-size:12px;color:#1E395B;min-width:140px;line-height:1.4;">` +
        `<b style="font-size:13px;color:#1E395B;">📍 ${{label}}</b><br/>` +
        `<span style="color:#5A7A9E;">${{count}} ${{count === 1 ? 'item' : 'items'}}</span><br/>` +
        `<div style="margin-top:6px;">` +
          `<button id="btn-cluster-action" style="background:#0066CC;color:white;border:none;padding:4px 10px;border-radius:3px;cursor:pointer;font-family:'Segoe UI',sans-serif;font-size:11px;font-weight:600;">${{popupBtnText}}</button>` +
        `</div>` +
      `</div>`;
      marker.bindPopup(popupHtml);
      marker.on('popupopen', function() {{
        const btn = document.getElementById('btn-cluster-action');
        if (btn) {{
          btn.onclick = function() {{
            if (window.qtBridge) {{
              window.qtBridge.onClusterClicked(clusterPayload);
            }}
          }};
        }}
      }});
      marker.on('click', () => {{
        if (window.qtBridge) {{
          window.qtBridge.onClusterClicked(clusterPayload);
        }}
      }});
    }});
  }}

  function setMapBounds(min_lat, max_lat, min_lon, max_lon) {{
    if (!map) return;
    isInternalMove = true;
    map.fitBounds([[min_lat, min_lon], [max_lat, max_lon]], {{ padding: [20, 20], maxZoom: 14 }});
  }}

  window.onMarkerClick = function(payloadJson) {{
    if (window.qtBridge) {{
      window.qtBridge.onClusterClicked(payloadJson);
    }}
  }};

  document.addEventListener('DOMContentLoaded', function() {{
    if (typeof QWebChannel !== 'undefined') {{
      new QWebChannel(qt.webChannelTransport, function(channel) {{
        window.qtBridge = channel.objects.mapBridge;
      }});
    }}
    initMap();
  }});
</script>
</body>
</html>"""


class MapBridge(QObject):
    """QWebChannel bridge exposing Qt signals and slots to JavaScript."""

    cluster_clicked = Signal(list)
    cluster_clicked_data = Signal(list, str)
    bounds_changed = Signal(float, float, float, float)

    @Slot(str)
    def onClusterClicked(self, cluster_json: str) -> None:
        try:
            data = json.loads(cluster_json)
            if isinstance(data, dict):
                mids = [int(x) for x in data.get("media_ids", [])]
                label = str(data.get("label", ""))
            elif isinstance(data, list):
                mids = [int(x) for x in data]
                label = ""
            else:
                return
            if mids:
                self.cluster_clicked.emit(mids)
                self.cluster_clicked_data.emit(mids, label)
        except Exception:
            pass

    @Slot(float, float, float, float)
    def onBoundsChanged(
        self, min_lat: float, max_lat: float, min_lon: float, max_lon: float
    ) -> None:
        self.bounds_changed.emit(
            float(min_lat), float(max_lat), float(min_lon), float(max_lon)
        )


class _InteractiveMapCanvas(QWidget):
    """
    Interactive vector world map canvas rendering latitude/longitude graticules,
    active bounding box, and clickable GPS cluster pins with zero network or GPU overhead.
    """

    cluster_clicked = Signal(list)  # list[int] of media_ids
    cluster_clicked_data = Signal(list, str)  # media_ids, label
    viewport_bbox_dragged = Signal(float, float, float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(480, 300)
        self.setMouseTracking(True)
        self.min_lat: float = -85.0
        self.max_lat: float = 85.0
        self.min_lon: float = -180.0
        self.max_lon: float = 180.0
        self.clusters: list[dict[str, Any]] = []
        self._pin_hit_rects: list[tuple[QRectF, list[int], str]] = []

    def set_bounds_and_clusters(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        clusters: list[dict[str, Any]],
    ) -> None:
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.clusters = clusters
        self.update()

    def _lat_lon_to_xy(self, lat: float, lon: float) -> QPointF:
        w = max(1.0, float(self.width()))
        h = max(1.0, float(self.height()))
        lon_span = max(0.001, self.max_lon - self.min_lon)
        lat_span = max(0.001, self.max_lat - self.min_lat)
        x = ((lon - self.min_lon) / lon_span) * w
        y = ((self.max_lat - lat) / lat_span) * h
        return QPointF(x, y)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        rect = self.rect()
        painter.fillRect(rect, QColor("#D9ECFA"))

        # Draw lat/lon graticule grid lines
        painter.setPen(QPen(QColor("#B5D3EB"), 1, Qt.DotLine))
        for lon in range(-180, 181, 30):
            if self.min_lon <= lon <= self.max_lon:
                pt = self._lat_lon_to_xy(0, lon)
                painter.drawLine(int(pt.x()), 0, int(pt.x()), rect.height())
        for lat in range(-60, 61, 30):
            if self.min_lat <= lat <= self.max_lat:
                pt = self._lat_lon_to_xy(lat, 0)
                painter.drawLine(0, int(pt.y()), rect.width(), int(pt.y()))

        # Draw Equator & Prime Meridian
        painter.setPen(QPen(QColor("#8FB8DA"), 1.2, Qt.DashLine))
        if self.min_lat <= 0 <= self.max_lat:
            eq = self._lat_lon_to_xy(0, 0)
            painter.drawLine(0, int(eq.y()), rect.width(), int(eq.y()))
        if self.min_lon <= 0 <= self.max_lon:
            pm = self._lat_lon_to_xy(0, 0)
            painter.drawLine(int(pm.x()), 0, int(pm.x()), rect.height())

        # Draw GPS Cluster Pins
        self._pin_hit_rects.clear()
        for c in self.clusters:
            pt = self._lat_lon_to_xy(float(c["lat"]), float(c["lon"]))
            cnt = int(c.get("count", 1))
            radius = min(28.0, 14.0 + (cnt ** 0.35) * 2.5)
            pin_rect = QRectF(
                pt.x() - radius, pt.y() - radius, radius * 2.0, radius * 2.0
            )
            label = str(c.get("label", ""))
            self._pin_hit_rects.append((pin_rect, c.get("media_ids", []), label))

            pix = c.get("pixmap")
            if isinstance(pix, QPixmap) and not pix.isNull():
                painter.save()
                clip_path = QPainterPath()
                clip_path.addEllipse(pin_rect)
                painter.setClipPath(clip_path)
                scaled_pix = pix.scaled(
                    int(radius * 2.0),
                    int(radius * 2.0),
                    Qt.KeepAspectRatioByExpanding,
                    Qt.SmoothTransformation,
                )
                painter.drawPixmap(
                    int(pin_rect.x()),
                    int(pin_rect.y()),
                    scaled_pix,
                )
                painter.restore()

                painter.setPen(QPen(QColor("#FFFFFF"), 2.0))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(pin_rect)

                if cnt > 1:
                    badge_radius = 8.5
                    badge_center = QPointF(pin_rect.right() - 2.0, pin_rect.top() + 2.0)
                    badge_rect = QRectF(
                        badge_center.x() - badge_radius,
                        badge_center.y() - badge_radius,
                        badge_radius * 2.0,
                        badge_radius * 2.0,
                    )
                    painter.setBrush(QBrush(QColor("#0066CC")))
                    painter.setPen(QPen(QColor("#FFFFFF"), 1.5))
                    painter.drawEllipse(badge_rect)
                    painter.setPen(QColor("#FFFFFF"))
                    painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
                    painter.drawText(badge_rect, Qt.AlignCenter, str(cnt))
            else:
                painter.setBrush(QBrush(QColor("#0066CC")))
                painter.setPen(QPen(QColor("#FFFFFF"), 2.0))
                painter.drawEllipse(pin_rect)

                painter.setPen(QColor("#FFFFFF"))
                painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
                painter.drawText(pin_rect, Qt.AlignCenter, str(cnt))

        painter.end()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            pos = event.position()
            for pin_rect, media_ids, label in self._pin_hit_rects:
                if pin_rect.contains(pos):
                    self.cluster_clicked.emit(media_ids)
                    self.cluster_clicked_data.emit(media_ids, label)
                    event.accept()
                    return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        factor = 0.8 if event.angleDelta().y() > 0 else 1.25
        c_lat = (self.min_lat + self.max_lat) / 2.0
        c_lon = (self.min_lon + self.max_lon) / 2.0
        half_lat = max(0.05, min(85.0, ((self.max_lat - self.min_lat) / 2.0) * factor))
        half_lon = max(0.05, min(180.0, ((self.max_lon - self.min_lon) / 2.0) * factor))
        self.viewport_bbox_dragged.emit(
            max(-85.0, c_lat - half_lat),
            min(85.0, c_lat + half_lat),
            max(-180.0, c_lon - half_lon),
            min(180.0, c_lon + half_lon),
        )
        event.accept()


class WorldMapView(QWidget):
    """
    Geotag World Map View backed by SQLite's R*Tree (`media_rtree`).
    Renders an interactive Leaflet map (via QWebEngineView + QWebChannel) when available,
    with an automatic fallback to high-performance vector canvas (_InteractiveMapCanvas).
    Emits `bbox_changed(min_lat, max_lat, min_lon, max_lon)` and `cluster_clicked(media_ids)`.
    """

    bbox_changed = Signal(float, float, float, float)
    cluster_clicked = Signal(list)
    cluster_clicked_data = Signal(list, str)
    filter_to_bbox_requested = Signal(float, float, float, float)
    close_requested = Signal()

    def minimumSizeHint(self) -> QSize:
        return QSize(300, 200)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("WorldMapView")

        self._min_lat: float = -85.0
        self._max_lat: float = 85.0
        self._min_lon: float = -180.0
        self._max_lon: float = 180.0
        self._current_clusters: list[dict[str, Any]] = []
        self._repo: CatalogRepository | None = None
        self._thumb_repo: Any = None
        self._web_loaded: bool = False

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        # Top Map Control Bar
        top_bar = QFrame(self)
        top_bar.setFixedHeight(40)
        top_bar.setStyleSheet("background-color: #EEF3F8; border-bottom: 1px solid #C8D2E0;")
        t_hbox = QHBoxLayout(top_bar)
        t_hbox.setContentsMargins(10, 4, 10, 4)
        t_hbox.setSpacing(6)

        lbl_title = QLabel("🌍 World Map Geotag Explorer", top_bar)
        lbl_title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        lbl_title.setStyleSheet("color: #1E395B;")
        t_hbox.addWidget(lbl_title)

        # Map Engine Toggle Button (Leaflet Web vs. Vector Offline)
        self.btn_toggle_engine = QPushButton("🗺 View: Leaflet Web", top_bar)
        self.btn_toggle_engine.setCursor(Qt.PointingHandCursor)
        self.btn_toggle_engine.clicked.connect(self._toggle_map_engine)
        if not HAS_WEBENGINE:
            self.btn_toggle_engine.setText("🗺 View: Vector Offline")
            self.btn_toggle_engine.setEnabled(False)
            self.btn_toggle_engine.setToolTip("QtWebEngine is not available in current environment.")
        t_hbox.addWidget(self.btn_toggle_engine)

        t_hbox.addSpacing(6)
        t_hbox.addWidget(QLabel("Lat:", top_bar))
        self.spin_min_lat = QDoubleSpinBox(top_bar)
        self.spin_min_lat.setRange(-90.0, 90.0)
        self.spin_min_lat.setValue(-85.0)
        t_hbox.addWidget(self.spin_min_lat)

        self.spin_max_lat = QDoubleSpinBox(top_bar)
        self.spin_max_lat.setRange(-90.0, 90.0)
        self.spin_max_lat.setValue(85.0)
        t_hbox.addWidget(self.spin_max_lat)

        t_hbox.addWidget(QLabel("Lon:", top_bar))
        self.spin_min_lon = QDoubleSpinBox(top_bar)
        self.spin_min_lon.setRange(-180.0, 180.0)
        self.spin_min_lon.setValue(-180.0)
        t_hbox.addWidget(self.spin_min_lon)

        self.spin_max_lon = QDoubleSpinBox(top_bar)
        self.spin_max_lon.setRange(-180.0, 180.0)
        self.spin_max_lon.setValue(180.0)
        t_hbox.addWidget(self.spin_max_lon)

        self.btn_apply_bbox = QPushButton("Query Map Area", top_bar)
        self.btn_apply_bbox.clicked.connect(self._on_apply_spin_bbox)
        t_hbox.addWidget(self.btn_apply_bbox)

        self.btn_filter_grid = QPushButton("Filter Gallery to Map Area", top_bar)
        self.btn_filter_grid.clicked.connect(
            lambda: self.filter_to_bbox_requested.emit(
                self._min_lat, self._max_lat, self._min_lon, self._max_lon
            )
        )
        t_hbox.addWidget(self.btn_filter_grid)

        self.btn_reset_world = QPushButton("Whole World", top_bar)
        self.btn_reset_world.clicked.connect(self._on_reset_world_clicked)
        t_hbox.addWidget(self.btn_reset_world)

        t_hbox.addStretch(1)

        self.lbl_cluster_summary = QLabel("0 geotagged items", top_bar)
        self.lbl_cluster_summary.setStyleSheet("color: #29588C; font-weight: bold;")
        t_hbox.addWidget(self.lbl_cluster_summary)

        self.btn_close_map = QPushButton("✕ Return to Grid", top_bar)
        self.btn_close_map.clicked.connect(self.close_requested.emit)
        t_hbox.addWidget(self.btn_close_map)

        vbox.addWidget(top_bar)

        # Splitter: Map Canvas/WebEngine on Left, Location Clusters List on Right
        splitter = QSplitter(Qt.Horizontal, self)

        self.map_stack = QStackedWidget(splitter)
        self.canvas = _InteractiveMapCanvas(self.map_stack)
        self.canvas.cluster_clicked.connect(self.cluster_clicked.emit)
        self.canvas.cluster_clicked_data.connect(self.cluster_clicked_data.emit)
        self.canvas.viewport_bbox_dragged.connect(
            lambda min_la, max_la, min_lo, max_lo: self.set_viewport_bbox(
                min_la, max_la, min_lo, max_lo, self._repo
            )
        )

        self.web_view: Any = None
        self.channel: Any = None
        self.bridge: Any = None

        if HAS_WEBENGINE:
            try:
                self.bridge = MapBridge(self)
                self.bridge.cluster_clicked.connect(self.cluster_clicked.emit)
                self.bridge.cluster_clicked_data.connect(self.cluster_clicked_data.emit)
                self.bridge.bounds_changed.connect(self._on_web_bounds_changed)

                self.channel = QWebChannel(self)
                self.channel.registerObject("mapBridge", self.bridge)

                self.web_view = QWebEngineView(self.map_stack)
                self.web_view.page().setWebChannel(self.channel)
                self.web_view.loadFinished.connect(self._on_web_load_finished)

                html = generate_leaflet_html(self._current_clusters)
                self.web_view.setHtml(html)

                self.map_stack.addWidget(self.web_view)
                self.map_stack.addWidget(self.canvas)
                self.map_stack.setCurrentWidget(self.web_view)
            except Exception:
                self.web_view = None
                self.map_stack.addWidget(self.canvas)
                self.map_stack.setCurrentWidget(self.canvas)
        else:
            self.map_stack.addWidget(self.canvas)
            self.map_stack.setCurrentWidget(self.canvas)

        splitter.addWidget(self.map_stack)

        self.cluster_list = QListWidget(splitter)
        self.cluster_list.setMinimumWidth(220)
        self.cluster_list.itemClicked.connect(self._on_cluster_list_clicked)
        splitter.addWidget(self.cluster_list)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

        vbox.addWidget(splitter, 1)

    def bind_repository(self, repo: CatalogRepository, thumb_repo: Any = None) -> None:
        self._repo = repo
        if thumb_repo is not None:
            self._thumb_repo = thumb_repo

    def _attach_cluster_thumbnails(self, clusters: list[dict[str, Any]]) -> None:
        """Enriches cluster dictionaries with thumbnail base64 data URIs and QPixmaps."""
        if not clusters or self._thumb_repo is None:
            return
        lead_ids = [c["media_ids"][0] for c in clusters if c.get("media_ids")]
        if not lead_ids:
            return
        try:
            blobs = self._thumb_repo.get_webp_blobs_batch(lead_ids)
            for c in clusters:
                if not c.get("media_ids"):
                    continue
                lead_id = c["media_ids"][0]
                blob = blobs.get(lead_id)
                if blob:
                    b64 = base64.b64encode(blob).decode("ascii")
                    c["thumb_url"] = f"data:image/webp;base64,{b64}"
                    pix = QPixmap()
                    if pix.loadFromData(blob, "WEBP"):
                        c["pixmap"] = pix
        except Exception:
            pass

    def _toggle_map_engine(self) -> None:
        """Toggles between the Leaflet OpenStreetMap WebEngine view and Vector Canvas view."""
        if not HAS_WEBENGINE or self.web_view is None:
            return
        if self.map_stack.currentWidget() == self.web_view:
            self.map_stack.setCurrentWidget(self.canvas)
            self.btn_toggle_engine.setText("🗺 View: Vector Offline")
            self.canvas.update()
        else:
            self.map_stack.setCurrentWidget(self.web_view)
            self.btn_toggle_engine.setText("🗺 View: Leaflet Web")
            if self._web_loaded:
                self.web_view.page().runJavaScript(
                    f"updateClusters({_clusters_to_json(self._current_clusters)});"
                )

    def _on_web_load_finished(self, ok: bool) -> None:
        self._web_loaded = ok
        if ok and self._current_clusters and self.web_view is not None:
            self.web_view.page().runJavaScript(
                f"updateClusters({_clusters_to_json(self._current_clusters)});"
            )

    def _on_web_bounds_changed(
        self, min_lat: float, max_lat: float, min_lon: float, max_lon: float
    ) -> None:
        """Called when user pans/zooms the Leaflet map."""
        self._min_lat = max(-85.0, min(85.0, float(min_lat)))
        self._max_lat = max(-85.0, min(85.0, float(max_lat)))
        self._min_lon = max(-180.0, min(180.0, float(min_lon)))
        self._max_lon = max(-180.0, min(180.0, float(max_lon)))

        self.spin_min_lat.blockSignals(True)
        self.spin_max_lat.blockSignals(True)
        self.spin_min_lon.blockSignals(True)
        self.spin_max_lon.blockSignals(True)
        self.spin_min_lat.setValue(self._min_lat)
        self.spin_max_lat.setValue(self._max_lat)
        self.spin_min_lon.setValue(self._min_lon)
        self.spin_max_lon.setValue(self._max_lon)
        self.spin_min_lat.blockSignals(False)
        self.spin_max_lat.blockSignals(False)
        self.spin_min_lon.blockSignals(False)
        self.spin_max_lon.blockSignals(False)

        lat_span = max(0.1, self._max_lat - self._min_lat)
        grid_deg = max(0.05, lat_span / 25.0)

        raw_points: list[dict[str, Any]] = []
        if self._repo is not None:
            raw_points = self._repo.query_map_clusters(
                self._min_lat, self._max_lat, self._min_lon, self._max_lon, grid_degrees=grid_deg
            )

        self._current_clusters = cluster_gps_points(raw_points, grid_degrees=grid_deg)
        self._attach_cluster_thumbnails(self._current_clusters)
        self.canvas.set_bounds_and_clusters(
            self._min_lat, self._max_lat, self._min_lon, self._max_lon, self._current_clusters
        )

        self.cluster_list.clear()
        total_items = sum(c["count"] for c in self._current_clusters)
        self.lbl_cluster_summary.setText(
            f"{total_items} geotagged items in {len(self._current_clusters)} clusters"
        )
        for c in self._current_clusters:
            item = QListWidgetItem(
                f"📍 {c['label']} ({c['count']} items)", self.cluster_list
            )
            item.setData(Qt.UserRole, (c["media_ids"], c["label"]))

        # Update markers in JS for new clusters without re-centering Leaflet viewport
        if self._web_loaded and self.web_view is not None:
            self.web_view.page().runJavaScript(
                f"updateClusters({_clusters_to_json(self._current_clusters)});"
            )

        self.bbox_changed.emit(self._min_lat, self._max_lat, self._min_lon, self._max_lon)

    def _on_apply_spin_bbox(self) -> None:
        self.set_viewport_bbox(
            self.spin_min_lat.value(),
            self.spin_max_lat.value(),
            self.spin_min_lon.value(),
            self.spin_max_lon.value(),
            self._repo,
        )
        if self._web_loaded and self.web_view is not None:
            self.web_view.page().runJavaScript(
                f"setMapBounds({self._min_lat}, {self._max_lat}, {self._min_lon}, {self._max_lon});"
            )

    def _on_reset_world_clicked(self) -> None:
        self.set_viewport_bbox(-85.0, 85.0, -180.0, 180.0, self._repo)
        if self._web_loaded and self.web_view is not None:
            self.web_view.page().runJavaScript("setMapBounds(-85.0, 85.0, -180.0, 180.0);")

    def _on_cluster_list_clicked(self, item: QListWidgetItem) -> None:
        data = item.data(Qt.UserRole)
        if isinstance(data, (tuple, list)) and len(data) >= 2 and isinstance(data[0], list):
            mids, label = data[0], str(data[1])
        elif isinstance(data, list):
            mids, label = data, ""
        else:
            return
        if mids:
            self.cluster_clicked.emit(mids)
            self.cluster_clicked_data.emit(mids, label)

    def set_viewport_bbox(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        repo: CatalogRepository | None = None,
    ) -> list[dict[str, Any]]:
        """
        Updates the active bounding box, queries `media_rtree` via `CatalogRepository.query_map_clusters`,
        clusters the results, updates the map pins (both WebEngine and vector canvas), and emits `bbox_changed`.
        """
        if repo is not None:
            self._repo = repo

        self._min_lat = min(float(min_lat), float(max_lat))
        self._max_lat = max(float(min_lat), float(max_lat))
        self._min_lon = min(float(min_lon), float(max_lon))
        self._max_lon = max(float(min_lon), float(max_lon))

        self.spin_min_lat.blockSignals(True)
        self.spin_max_lat.blockSignals(True)
        self.spin_min_lon.blockSignals(True)
        self.spin_max_lon.blockSignals(True)
        self.spin_min_lat.setValue(self._min_lat)
        self.spin_max_lat.setValue(self._max_lat)
        self.spin_min_lon.setValue(self._min_lon)
        self.spin_max_lon.setValue(self._max_lon)
        self.spin_min_lat.blockSignals(False)
        self.spin_max_lat.blockSignals(False)
        self.spin_min_lon.blockSignals(False)
        self.spin_max_lon.blockSignals(False)

        lat_span = max(0.1, self._max_lat - self._min_lat)
        grid_deg = max(0.05, lat_span / 25.0)

        raw_points: list[dict[str, Any]] = []
        if self._repo is not None:
            raw_points = self._repo.query_map_clusters(
                self._min_lat, self._max_lat, self._min_lon, self._max_lon, grid_degrees=grid_deg
            )

        self._current_clusters = cluster_gps_points(raw_points, grid_degrees=grid_deg)
        self._attach_cluster_thumbnails(self._current_clusters)

        self.canvas.set_bounds_and_clusters(
            self._min_lat,
            self._max_lat,
            self._min_lon,
            self._max_lon,
            self._current_clusters,
        )

        if self._web_loaded and self.web_view is not None:
            self.web_view.page().runJavaScript(
                f"updateClusters({_clusters_to_json(self._current_clusters)});"
            )

        self.cluster_list.clear()
        total_items = sum(c["count"] for c in self._current_clusters)
        self.lbl_cluster_summary.setText(
            f"{total_items} geotagged items in {len(self._current_clusters)} clusters"
        )
        for c in self._current_clusters:
            item = QListWidgetItem(
                f"📍 {c['label']} ({c['count']} items)", self.cluster_list
            )
            item.setData(Qt.UserRole, (c["media_ids"], c["label"]))

        self.bbox_changed.emit(
            self._min_lat, self._max_lat, self._min_lon, self._max_lon
        )
        return self._current_clusters

    @property
    def current_clusters(self) -> list[dict[str, Any]]:
        return self._current_clusters
