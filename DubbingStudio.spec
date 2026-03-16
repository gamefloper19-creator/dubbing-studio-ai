# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Dubbing Studio.

Builds DubbingStudio.exe - a standalone Windows executable
that runs without requiring Python installation.

Usage:
    pyinstaller DubbingStudio.spec

Requirements:
    - PyInstaller >= 6.0
    - All project dependencies installed
    - FFmpeg must be bundled separately or installed on target system
"""

import os
import sys
from pathlib import Path

block_cipher = None

# Project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(SPEC))

# Collect all dubbing_studio package files
dubbing_studio_path = os.path.join(PROJECT_ROOT, "dubbing_studio")

a = Analysis(
    [os.path.join(PROJECT_ROOT, "main.py")],
    pathex=[PROJECT_ROOT],
    binaries=[],
    datas=[
        # Include the dubbing_studio package
        (dubbing_studio_path, "dubbing_studio"),
        # Include app.py for GUI
        (os.path.join(PROJECT_ROOT, "app.py"), "."),
    ],
    hiddenimports=[
        # Core dependencies
        "gradio",
        "psutil",
        "edge_tts",

        # Dubbing Studio modules
        "dubbing_studio",
        "dubbing_studio.config",
        "dubbing_studio.pipeline",
        "dubbing_studio.audio",
        "dubbing_studio.audio.extractor",
        "dubbing_studio.audio.cleaner",
        "dubbing_studio.audio.mixer",
        "dubbing_studio.audio.segmenter",
        "dubbing_studio.speech",
        "dubbing_studio.speech.recognizer",
        "dubbing_studio.speech.analyzer",
        "dubbing_studio.translation",
        "dubbing_studio.translation.translator",
        "dubbing_studio.tts",
        "dubbing_studio.tts.engine",
        "dubbing_studio.tts.qwen_tts",
        "dubbing_studio.tts.chatterbox_tts",
        "dubbing_studio.tts.lux_tts",
        "dubbing_studio.tts.voice_selector",
        "dubbing_studio.tts.voice_library",
        "dubbing_studio.timing",
        "dubbing_studio.timing.aligner",
        "dubbing_studio.subtitle",
        "dubbing_studio.subtitle.generator",
        "dubbing_studio.video",
        "dubbing_studio.video.renderer",
        "dubbing_studio.export",
        "dubbing_studio.export.exporter",
        "dubbing_studio.batch",
        "dubbing_studio.batch.processor",
        "dubbing_studio.hardware",
        "dubbing_studio.hardware.optimizer",
        "dubbing_studio.models",
        "dubbing_studio.models.manager",

        # Standard library modules that may be missed
        "asyncio",
        "concurrent.futures",
        "dataclasses",
        "json",
        "logging",
        "pathlib",
        "subprocess",
        "tempfile",
        "threading",
        "uuid",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy optional packages from base build
        # Users can install these separately for neural TTS
        "torch",
        "transformers",
        "TTS",
        "chatterbox",
        "tensorflow",
        "tensorboard",
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
    [],
    exclude_binaries=True,
    name="DubbingStudio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Keep console for logging output
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here if available: icon='assets/icon.ico'
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="DubbingStudio",
)
