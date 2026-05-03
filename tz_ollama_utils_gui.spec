# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path


project_root = Path(SPECPATH)
assets_dir = project_root / "assets"
icon_path = None

if sys.platform.startswith("win"):
    icon_path = str(assets_dir / "icons" / "tz_ollama_utils_icon.ico")


a = Analysis(
    ["tz_ollama_utils_gui.py"],
    pathex=[str(project_root)],
    binaries=[],
    datas=[(str(assets_dir), "assets")],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="tz-ollama-utils",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_path,
)
