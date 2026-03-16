"""Tests for input validation module."""

from unittest.mock import patch

import pytest

from dubbing_studio.validation import (
    SUPPORTED_AUDIO_FORMATS,
    SUPPORTED_VIDEO_FORMATS,
    ValidationError,
    validate_language,
    validate_language_pair,
    validate_video_file,
)


class TestValidateLanguage:
    def test_valid_language(self):
        assert validate_language("en") == "en"
        assert validate_language("hi") == "hi"
        assert validate_language("es") == "es"

    def test_case_insensitive(self):
        assert validate_language("EN") == "en"
        assert validate_language("Hi") == "hi"

    def test_strips_whitespace(self):
        assert validate_language("  en  ") == "en"

    def test_invalid_language(self):
        with pytest.raises(ValidationError, match="Unsupported language"):
            validate_language("xx")

    def test_empty_string(self):
        with pytest.raises(ValidationError):
            validate_language("")


class TestValidateLanguagePair:
    def test_valid_pair(self):
        src, tgt = validate_language_pair("en", "hi")
        assert src == "en"
        assert tgt == "hi"

    def test_same_language_rejected(self):
        with pytest.raises(ValidationError, match="same"):
            validate_language_pair("en", "en")

    def test_invalid_source(self):
        with pytest.raises(ValidationError, match="Unsupported"):
            validate_language_pair("xx", "en")

    def test_invalid_target(self):
        with pytest.raises(ValidationError, match="Unsupported"):
            validate_language_pair("en", "xx")


class TestValidateVideoFile:
    def test_file_not_found(self):
        with pytest.raises(ValidationError, match="not found"):
            validate_video_file("/nonexistent/file.mp4")

    def test_not_a_file(self, tmp_path):
        with pytest.raises(ValidationError, match="Not a file"):
            validate_video_file(str(tmp_path))

    def test_unsupported_format(self, tmp_path):
        bad_file = tmp_path / "test.xyz"
        bad_file.write_text("data")
        with pytest.raises(ValidationError, match="Unsupported format"):
            validate_video_file(str(bad_file))

    def test_empty_file(self, tmp_path):
        empty = tmp_path / "empty.mp4"
        empty.write_text("")
        with pytest.raises(ValidationError, match="empty"):
            validate_video_file(str(empty))

    def test_file_too_large(self, tmp_path):
        large = tmp_path / "large.mp4"
        large.write_bytes(b"x" * 100)
        with pytest.raises(ValidationError, match="too large"):
            # 100 bytes > 0.0000001 GB, so size check triggers before probe
            validate_video_file(str(large), max_size_gb=1e-12)

    @patch("dubbing_studio.validation._probe_file")
    def test_valid_file(self, mock_probe, tmp_path):
        video = tmp_path / "test.mp4"
        video.write_bytes(b"x" * 1024)
        mock_probe.return_value = {
            "duration": 60.0,
            "has_video": True,
            "has_audio": True,
            "video_codec": "h264",
            "audio_codec": "aac",
            "width": 1920,
            "height": 1080,
        }
        result = validate_video_file(str(video))
        assert result["duration"] == 60.0
        assert result["has_video"] is True
        assert result["has_audio"] is True
        assert result["format"] == ".mp4"

    @patch("dubbing_studio.validation._probe_file")
    def test_no_audio_stream(self, mock_probe, tmp_path):
        video = tmp_path / "silent.mp4"
        video.write_bytes(b"x" * 1024)
        mock_probe.return_value = {
            "duration": 60.0,
            "has_video": True,
            "has_audio": False,
        }
        with pytest.raises(ValidationError, match="no audio"):
            validate_video_file(str(video))

    @patch("dubbing_studio.validation._probe_file")
    def test_too_short(self, mock_probe, tmp_path):
        video = tmp_path / "short.mp4"
        video.write_bytes(b"x" * 1024)
        mock_probe.return_value = {
            "duration": 0.5,
            "has_video": True,
            "has_audio": True,
        }
        with pytest.raises(ValidationError, match="too short"):
            validate_video_file(str(video))

    @patch("dubbing_studio.validation._probe_file")
    def test_too_long(self, mock_probe, tmp_path):
        video = tmp_path / "long.mp4"
        video.write_bytes(b"x" * 1024)
        mock_probe.return_value = {
            "duration": 50000.0,
            "has_video": True,
            "has_audio": True,
        }
        with pytest.raises(ValidationError, match="too long"):
            validate_video_file(str(video), max_duration_hours=1.0)


class TestSupportedFormats:
    def test_video_formats_include_common(self):
        assert ".mp4" in SUPPORTED_VIDEO_FORMATS
        assert ".mkv" in SUPPORTED_VIDEO_FORMATS
        assert ".avi" in SUPPORTED_VIDEO_FORMATS
        assert ".mov" in SUPPORTED_VIDEO_FORMATS
        assert ".webm" in SUPPORTED_VIDEO_FORMATS

    def test_audio_formats_include_common(self):
        assert ".wav" in SUPPORTED_AUDIO_FORMATS
        assert ".mp3" in SUPPORTED_AUDIO_FORMATS
        assert ".flac" in SUPPORTED_AUDIO_FORMATS
