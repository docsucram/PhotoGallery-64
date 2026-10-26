# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller specification for PhotoGallery64 (Essentials Photo Gallery).
Builds a standalone, redistributable Windows 64-bit application package ready
for Inno Setup / installer bundling.
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

project_root = Path.cwd()

# Data files: themes, icons, database schema
datas = [
    (str(project_root / "assets"), "assets"),
    (str(project_root / "resources"), "resources"),
    (str(project_root / "db" / "schema.sql"), "db"),
]

hiddenimports = [
    "PySide6.QtSvg",
    "PySide6.QtSvgWidgets",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "sqlite3",
    "cv2",
    "numpy",
    "PIL",
    "PIL.Image",
    "PIL.ExifTags",
]

# Exclude unnecessary dev/test packages to keep binary compact
excludes = [
    "pytest",
    "unittest",
    "tkinter",
    "IPython",
    "jupyter",
]

a = Analysis(
    ["main.py"],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="EssentialsPhotoGallery",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Windowed GUI app (no black console window)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(project_root / "resources" / "icons" / "essentials_photo_gallery.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="EssentialsPhotoGallery",
)
