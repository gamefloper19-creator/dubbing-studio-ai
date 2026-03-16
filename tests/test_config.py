"""Tests for configuration module."""

import os
from unittest.mock import patch

from dubbing_studio.config import (
    AppConfig,
    AudioConfig,
    BatchConfig,
    ExportConfig,
    MixingConfig,
    SubtitleConfig,
    TimingConfig,
    TranslationConfig,
    VoiceConfig,
    WhisperConfig,
    SUPPORTED_LANGUAGES,
    VOICE_LANGUAGE_MAP,
)


class TestAudioConfig:
    def test_defaults(self):
        cfg = AudioConfig()
        assert cfg.sample_rate == 44100
        assert cfg.channels == 1
        assert cfg.normalize_loudness == -23.0
        assert cfg.noise_reduction_strength == 0.5
        assert cfg.min_silence_duration == 0.5
        assert cfg.silence_threshold == -40.0
        assert cfg.segment_min_duration == 5.0
        assert cfg.segment_max_duration == 15.0

    def test_custom_values(self):
        cfg = AudioConfig(sample_rate=48000, channels=2)
        assert cfg.sample_rate == 48000
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
        assert cfg.temperature == 0.7
        assert cfg.max_retries == 3
        assert cfg.api_key == ""
        assert "professional narrator" in cfg.system_prompt


class TestVoiceConfig:
    def test_defaults(self):
        cfg = VoiceConfig()
        assert cfg.engine == "auto"
        assert cfg.speed == 1.0
        assert cfg.pitch == 1.0
        assert cfg.narrator_gender == "auto"
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
        assert cfg.video_format == "mp4"
        assert cfg.video_codec == "libx264"
        assert cfg.video_quality == 23
        assert cfg.audio_codec == "aac"
        assert cfg.audio_only_format == "wav"


class TestBatchConfig:
    def test_defaults(self):
        cfg = BatchConfig()
        assert cfg.max_concurrent == 4
        assert cfg.retry_on_failure is True
        assert cfg.max_retries == 2
        assert cfg.auto_export is True


class TestAppConfig:
    def test_defaults(self):
        cfg = AppConfig()
        assert isinstance(cfg.audio, AudioConfig)
        assert isinstance(cfg.whisper, WhisperConfig)
        assert isinstance(cfg.translation, TranslationConfig)
        assert isinstance(cfg.voice, VoiceConfig)
        assert isinstance(cfg.timing, TimingConfig)
        assert isinstance(cfg.mixing, MixingConfig)
        assert isinstance(cfg.subtitle, SubtitleConfig)
        assert isinstance(cfg.export, ExportConfig)
        assert isinstance(cfg.batch, BatchConfig)
        assert cfg.output_dir == "output"
        assert cfg.temp_dir == "temp"
        assert cfg.cache_dir == "cache"

    def test_from_env(self):
        env = {
            "GEMINI_API_KEY": "test-key-123",
            "DUBBING_OUTPUT_DIR": "/tmp/out",
            "DUBBING_TEMP_DIR": "/tmp/tmp",
            "WHISPER_MODEL": "medium",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = AppConfig.from_env()
            assert cfg.translation.api_key == "test-key-123"
            assert cfg.output_dir == "/tmp/out"
            assert cfg.temp_dir == "/tmp/tmp"
            assert cfg.whisper.model_size == "medium"

    def test_from_env_invalid_whisper(self):
        with patch.dict(os.environ, {"WHISPER_MODEL": "invalid"}, clear=False):
            cfg = AppConfig.from_env()
            assert cfg.whisper.model_size == "base"  # stays at default

    def test_setup_dirs(self, tmp_path):
        cfg = AppConfig()
        cfg.output_dir = str(tmp_path / "output")
        cfg.temp_dir = str(tmp_path / "temp")
        cfg.cache_dir = str(tmp_path / "cache")
        cfg.setup_dirs()
        assert (tmp_path / "output").exists()
        assert (tmp_path / "temp").exists()
        assert (tmp_path / "cache").exists()


class TestSupportedLanguages:
    def test_has_expected_languages(self):
        assert "en" in SUPPORTED_LANGUAGES
        assert "hi" in SUPPORTED_LANGUAGES
        assert "es" in SUPPORTED_LANGUAGES
        assert "fr" in SUPPORTED_LANGUAGES
        assert "de" in SUPPORTED_LANGUAGES
        assert "ja" in SUPPORTED_LANGUAGES
        assert len(SUPPORTED_LANGUAGES) == 24

    def test_language_names_are_strings(self):
        for code, name in SUPPORTED_LANGUAGES.items():
            assert isinstance(code, str)
            assert isinstance(name, str)
            assert len(code) == 2


class TestVoiceLanguageMap:
    def test_covers_all_languages(self):
        for lang in SUPPORTED_LANGUAGES:
            assert lang in VOICE_LANGUAGE_MAP, f"Missing voice map for {lang}"

    def test_voice_map_structure(self):
        for lang, config in VOICE_LANGUAGE_MAP.items():
            assert "gender" in config
            assert "style" in config
            assert "engine" in config
            assert config["engine"] in ("qwen3", "chatterbox", "luxtts")
