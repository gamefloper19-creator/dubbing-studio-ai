# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for DubbingStudio.exe

Builds a standalone Windows desktop application that bundles
the complete Dubbing Studio AI pipeline with PySide6 GUI.

Usage:
    pyinstaller DubbingStudio.spec
"""

import sys
from pathlib import Path

block_cipher = None

# Collect the dubbing_studio package data
a = Analysis(
    ['desktop_app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('dubbing_studio', 'dubbing_studio'),
    ],
    hiddenimports=[
        # Core pipeline modules
        'dubbing_studio',
        'dubbing_studio.pipeline',
        'dubbing_studio.config',
        'dubbing_studio.validation',
        'dubbing_studio.emotion',
        'dubbing_studio.emotion.analyzer',
        'dubbing_studio.narration',
        'dubbing_studio.narration.engine',
        'dubbing_studio.timing',
        'dubbing_studio.timing.advanced_aligner',
        'dubbing_studio.tts',
        'dubbing_studio.tts.voice_library',
        'dubbing_studio.youtube',
        'dubbing_studio.youtube.pipeline',
        'dubbing_studio.models',
        'dubbing_studio.models.manager',
        'dubbing_studio.batch',
        'dubbing_studio.batch.processor',
        'dubbing_studio.cloning',
        'dubbing_studio.cloning.voice_cloner',
        'dubbing_studio.hardware',
        'dubbing_studio.hardware.optimizer',
        # PySide6 modules
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        # Standard library used by pipeline
        'json',
        'hashlib',
        'dataclasses',
        'concurrent.futures',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy optional dependencies not needed for desktop
        'torch',
        'torchaudio',
        'transformers',
        'gradio',
        'matplotlib',
        'scipy',
        'sklearn',
        'tensorflow',
        'keras',
        'jupyter',
        'IPython',
        'notebook',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='DubbingStudio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Windowed application (no console)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Set icon path here if available: icon='assets/icon.ico'
)
