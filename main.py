"""
PhotoGallery64 — Modern 64-Bit Python / PySide6 Clone of Windows Live Photo Gallery.
Application entrypoint initializing High-DPI Qt rendering, DualDatabaseEngine
(db/catalog.db + db/thumbnails.db), and PhotoGalleryMainWindow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtCore import QLoggingCategory, Qt
from PySide6.QtWidgets import QApplication

from src.db.connection import DualDatabaseEngine
from src.ui.main_window import PhotoGalleryMainWindow


def create_app(argv: list[str] | None = None) -> QApplication:
    existing = QApplication.instance()
    if existing is not None and isinstance(existing, QApplication):
        return existing
    QLoggingCategory.setFilterRules(
        "qt.gui.imageio.jpeg.warning=false\n"
        "qt.gui.imageio.warning=false\n"
    )
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("PhotoGallery64")
    app.setOrganizationName("PhotoGallery64")
    app.setStyleSheet(
        "QToolTip { "
        "background-color: #FFFFFF; "
        "color: #1E395B; "
        "border: 1px solid #7DA2CE; "
        "border-radius: 2px; "
        "padding: 3px 6px; "
        "font-family: 'Segoe UI', sans-serif; "
        "font-size: 9pt; "
        "}"
    )
    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="PhotoGallery64 — Windows Live Photo Gallery 64-Bit Clone"
    )
    parser.add_argument(
        "--db-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "db",
        help="Directory for catalog.db and thumbnails.db",
    )
    parser.add_argument(
        "--scan-folder",
        type=Path,
        default=None,
        help="Optional initial folder path to scan on startup",
    )
    parser.add_argument(
        "--theme",
        choices=["wlpg_aero", "modern", "dark"],
        default=None,
        help="Initial visual stylesheet theme",
    )
    args = parser.parse_args(argv)

    app = create_app(argv)
    args.db_dir.mkdir(parents=True, exist_ok=True)

    engine = DualDatabaseEngine(
        catalog_path=args.db_dir / "catalog.db",
        thumbnails_path=args.db_dir / "thumbnails.db",
    )

    window = PhotoGalleryMainWindow(engine=engine, start_xmp_daemon=True)
    if args.theme:
        window.apply_theme(args.theme)
    window.show()

    if args.scan_folder is not None and args.scan_folder.exists():
        window.scan_folder(args.scan_folder, include_subfolders=True)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
