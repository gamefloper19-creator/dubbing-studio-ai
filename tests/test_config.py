"""Tests for configuration module."""

import os
from unittest.mock import patch

from dubbing_studio.config import (
    SUPPORTED_LANGUAGES,
    AppConfig,
    AudioConfig,
    BatchConfig,
    EmotionConfig,
    ExportConfig,
    MixingConfig,
    SubtitleConfig,
    TimingConfig,
    TranslationConfig,
    VoiceConfig,
    WhisperConfig,
)


class TestAudioConfig:
    def test_defaults(self):
        cfg = AudioConfig()
        assert cfg.sample_rate == 44100
        assert cfg.channels == 1
        assert cfg.normalize_loudness == -23.0
        assert cfg.min_silence_duration == 0.5
        assert cfg.silence_threshold == -40.0
        assert cfg.segment_min_duration == 5.0
        assert cfg.segment_max_duration == 15.0

    def test_custom_values(self):
        cfg = AudioConfig(sample_rate=22050, channels=2)
        assert cfg.sample_rate == 22050
        assert cfg.channels == 2


class TestWhisperConfig:
    def test_defaults(self):
        cfg = WhisperConfig()
        assert cfg.model_size == "base"
        assert cfg.device == "auto"
        assert cfg.language is None
        assert cfg.beam_size == 5
        assert cfg.vad_filter is True


class TestTranslationConfig:
    def test_defaults(self):
        cfg = TranslationConfig()
        assert cfg.provider == "gemini"
        assert cfg.model == "gemini-2.0-flash"
        assert cfg.max_retries == 3
        assert cfg.temperature == 0.7
        assert "documentary" in cfg.system_prompt.lower()


class TestVoiceConfig:
    def test_defaults(self):
        cfg = VoiceConfig()
        assert cfg.engine == "auto"
        assert cfg.speed == 1.0
        assert cfg.narrator_style == "documentary"


class TestTimingConfig:
    def test_defaults(self):
        cfg = TimingConfig()
        assert cfg.max_deviation_ms == 300.0
        assert cfg.speed_min == 0.8
        assert cfg.speed_max == 1.3


class TestMixingConfig:
    def test_defaults(self):
        cfg = MixingConfig()
        assert cfg.narration_volume == 1.0
        assert cfg.background_volume == 0.15
        assert cfg.ducking_enabled is True


class TestSubtitleConfig:
    def test_defaults(self):
        cfg = SubtitleConfig()
        assert cfg.format == "srt"
        assert cfg.embed_in_video is False
        assert cfg.font_size == 24


class TestExportConfig:
    def test_defaults(self):
        cfg = ExportConfig()
        assert cfg.video_codec == "libx264"
        assert cfg.audio_codec == "aac"
        assert cfg.audio_bitrate == "192k"


class TestBatchConfig:
    def test_defaults(self):
        cfg = BatchConfig()
        assert cfg.max_concurrent == 4
        assert cfg.retry_on_failure is True
        assert cfg.max_retries == 2


class TestEmotionConfig:
    def test_defaults(self):
        cfg = EmotionConfig()
        assert cfg.enabled is True
        assert "neutral" in cfg.emotions
        assert "dramatic" in cfg.emotions


class TestSupportedLanguages:
    def test_has_core_languages(self):
        assert "en" in SUPPORTED_LANGUAGES
        assert "hi" in SUPPORTED_LANGUAGES
        assert "es" in SUPPORTED_LANGUAGES
        assert "fr" in SUPPORTED_LANGUAGES
        assert "de" in SUPPORTED_LANGUAGES
        assert "ja" in SUPPORTED_LANGUAGES
        assert "zh" in SUPPORTED_LANGUAGES

    def test_language_count(self):
        assert len(SUPPORTED_LANGUAGES) >= 24


class TestAppConfig:
    def test_defaults(self):
        cfg = AppConfig()
        assert isinstance(cfg.audio, AudioConfig)
        assert isinstance(cfg.whisper, WhisperConfig)
        assert isinstance(cfg.translation, TranslationConfig)
        assert cfg.output_dir == "output"
        assert cfg.temp_dir == "temp"

    def test_from_env(self):
        with patch.dict(
            os.environ,
            {
                "GEMINI_API_KEY": "test-key",
                "DUBBING_OUTPUT_DIR": "/tmp/test_output",
                "WHISPER_MODEL": "medium",
            },
        ):
            cfg = AppConfig.from_env()
            assert cfg.translation.api_key == "test-key"
            assert cfg.output_dir == "/tmp/test_output"
            assert cfg.whisper.model_size == "medium"

    def test_from_env_invalid_whisper_model(self):
        with patch.dict(os.environ, {"WHISPER_MODEL": "invalid_model"}):
            cfg = AppConfig.from_env()
            assert cfg.whisper.model_size == "base"

    def test_setup_dirs(self, tmp_path):
        cfg = AppConfig(
            output_dir=str(tmp_path / "out"),
            temp_dir=str(tmp_path / "tmp"),
            cache_dir=str(tmp_path / "cache"),
        )
        cfg.cloning.profiles_dir = str(tmp_path / "profiles")
        cfg.model_management.models_dir = str(tmp_path / "models")
        cfg.setup_dirs()
        assert (tmp_path / "out").exists()
        assert (tmp_path / "tmp").exists()
        assert (tmp_path / "cache").exists()
