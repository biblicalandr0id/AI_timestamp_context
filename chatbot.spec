# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for AI Chatbot Desktop Application
Build with: pyinstaller chatbot.spec
"""

import sys
from pathlib import Path

block_cipher = None

# Collect all data files
datas = [
    ('CHATBOT.md', '.'),
    ('QUICKSTART.md', '.'),
    ('FEATURES.md', '.'),
]

# Hidden imports that PyInstaller might miss
hiddenimports = [
    'PyQt6',
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'torch',
    'transformers',
    'sentence_transformers',
    'flask',
    'flask_socketio',
    'numpy',
    'sqlite3',
    'sklearn',
    'scipy',
    'nltk',
]

# Analysis
a = Analysis(
    ['desktop_app.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'pandas'],  # Exclude heavy unused packages
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Remove duplicate files
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Platform-specific configuration
if sys.platform == 'darwin':  # macOS
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='AI_Chatbot',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon='icon.icns' if Path('icon.icns').exists() else None
    )

    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name='AI_Chatbot',
    )

    app = BUNDLE(
        coll,
        name='AI_Chatbot.app',
        icon='icon.icns' if Path('icon.icns').exists() else None,
        bundle_identifier='com.airesearch.chatbot',
        info_plist={
            'NSPrincipalClass': 'NSApplication',
            'NSHighResolutionCapable': 'True',
        },
    )

elif sys.platform == 'win32':  # Windows
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='AI_Chatbot',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,  # No console window
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon='icon.ico' if Path('icon.ico').exists() else None
    )

    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name='AI_Chatbot',
    )

else:  # Linux
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='ai-chatbot',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )

    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name='ai-chatbot',
    )
