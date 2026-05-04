# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path


project_root = Path(SPECPATH)
source_dir = project_root / "src"
assets_dir = project_root / "assets"
package_assets_dir = project_root / "src" / "tz_ollama_utils" / "assets"
icon_path = None

if sys.platform.startswith("win"):
    icon_path = str(assets_dir / "icons" / "tz_ollama_utils_icon.ico")


a = Analysis(
    ["tz_ollama_utils_gui.py"],
    pathex=[str(project_root), str(source_dir)],
    binaries=[],
    datas=[
        (str(project_root / "pyproject.toml"), "."),
        (str(assets_dir), "assets"),
        (str(package_assets_dir), "tz_ollama_utils/assets"),
    ],
    hiddenimports=[
        "tz_ollama_utils",
        "tz_ollama_utils.common",
        "tz_ollama_utils.gui",
        "tz_ollama_utils.test_models",
        "tz_ollama_utils.update_models",
    ],
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
    upx=False,
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
