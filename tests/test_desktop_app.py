"""Tests for the desktop application module (PySide6).

Skips all tests when PySide6 cannot be loaded (missing package or
missing native libraries like libEGL on headless CI runners).
"""

import sys

import pytest

# Skip entire module if PySide6 cannot be loaded at runtime.
# PySide6 may be *installed* but fail to import due to missing native
# libraries (e.g. libEGL.so.1) on headless CI runners.
try:
    from PySide6.QtWidgets import QApplication  # noqa: F401
except Exception:
    pytest.skip(
        "PySide6 runtime not available (missing native libraries)",
        allow_module_level=True,
    )

from desktop_app import (  # noqa: E402
    DubbingStudioApp,
    DropLineEdit,
    NARRATOR_STYLES,
    SignalLogHandler,
    SUBTITLE_FORMATS,
    VIDEO_EXTENSIONS,
    WHISPER_MODELS,
    WorkerSignals,
    main,
)


class TestConstants:
    """Test module-level constants."""

    def test_narrator_styles(self):
        assert isinstance(NARRATOR_STYLES, list)
        assert "documentary" in NARRATOR_STYLES
        assert "cinematic" in NARRATOR_STYLES

    def test_whisper_models(self):
        assert isinstance(WHISPER_MODELS, list)
        assert "base" in WHISPER_MODELS
        assert "large-v3" in WHISPER_MODELS

    def test_subtitle_formats(self):
        assert isinstance(SUBTITLE_FORMATS, list)
        assert "srt" in SUBTITLE_FORMATS
        assert "vtt" in SUBTITLE_FORMATS

    def test_video_extensions(self):
        assert ".mp4" in VIDEO_EXTENSIONS
        assert ".mkv" in VIDEO_EXTENSIONS


class TestWorkerSignals:
    """Test WorkerSignals class."""

    def test_worker_signals_exists(self):
        signals = WorkerSignals()
        assert hasattr(signals, "progress")
        assert hasattr(signals, "finished")
        assert hasattr(signals, "error")
        assert hasattr(signals, "log")


class TestDropLineEdit:
    """Test DropLineEdit class."""

    def test_drop_line_edit_exists(self):
        assert DropLineEdit is not None

    def test_drop_line_edit_accepts_drops(self):
        if not QApplication.instance():
            QApplication(sys.argv)
        widget = DropLineEdit()
        assert widget.acceptDrops()


class TestSignalLogHandler:
    """Test SignalLogHandler class."""

    def test_signal_log_handler_exists(self):
        assert SignalLogHandler is not None


class TestDesktopAppModule:
    """Test the desktop_app module can be imported."""

    def test_main_function_exists(self):
        assert callable(main)

    def test_dubbing_studio_app_class_exists(self):
        assert DubbingStudioApp is not None
