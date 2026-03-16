"""Tests for the desktop application module (PySide6)."""

import sys
from unittest.mock import patch, MagicMock

import pytest

# Skip entire module if PySide6 is not available (headless CI)
pytest.importorskip("PySide6", reason="PySide6 not available in this environment")


class TestConstants:
    """Test module-level constants."""

    def test_narrator_styles(self):
        from desktop_app import NARRATOR_STYLES
        assert isinstance(NARRATOR_STYLES, list)
        assert "documentary" in NARRATOR_STYLES
        assert "cinematic" in NARRATOR_STYLES

    def test_whisper_models(self):
        from desktop_app import WHISPER_MODELS
        assert isinstance(WHISPER_MODELS, list)
        assert "base" in WHISPER_MODELS
        assert "large-v3" in WHISPER_MODELS

    def test_subtitle_formats(self):
        from desktop_app import SUBTITLE_FORMATS
        assert isinstance(SUBTITLE_FORMATS, list)
        assert "srt" in SUBTITLE_FORMATS
        assert "vtt" in SUBTITLE_FORMATS

    def test_video_extensions(self):
        from desktop_app import VIDEO_EXTENSIONS
        assert ".mp4" in VIDEO_EXTENSIONS
        assert ".mkv" in VIDEO_EXTENSIONS


class TestWorkerSignals:
    """Test WorkerSignals class."""

    def test_worker_signals_exists(self):
        from desktop_app import WorkerSignals
        signals = WorkerSignals()
        assert hasattr(signals, "progress")
        assert hasattr(signals, "finished")
        assert hasattr(signals, "error")
        assert hasattr(signals, "log")


class TestDropLineEdit:
    """Test DropLineEdit class."""

    def test_drop_line_edit_exists(self):
        from desktop_app import DropLineEdit
        assert DropLineEdit is not None

    def test_drop_line_edit_accepts_drops(self):
        from PySide6.QtWidgets import QApplication
        if not QApplication.instance():
            app = QApplication.instance() or QApplication(sys.argv)
        from desktop_app import DropLineEdit
        widget = DropLineEdit()
        assert widget.acceptDrops()


class TestSignalLogHandler:
    """Test SignalLogHandler class."""

    def test_signal_log_handler_exists(self):
        from desktop_app import SignalLogHandler
        assert SignalLogHandler is not None


class TestDesktopAppModule:
    """Test the desktop_app module can be imported."""

    def test_main_function_exists(self):
        from desktop_app import main
        assert callable(main)

    def test_dubbing_studio_app_class_exists(self):
        from desktop_app import DubbingStudioApp
        assert DubbingStudioApp is not None
