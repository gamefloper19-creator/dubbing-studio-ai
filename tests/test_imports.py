"""Tests to verify all module imports work correctly."""

import importlib


def test_version():
    from dubbing_studio import __version__, __app_name__
    assert __version__ == "2.0.0"
    assert __app_name__ == "Dubbing Studio"


def test_import_config():
    from dubbing_studio.config import (
        AppConfig, AudioConfig, WhisperConfig, TranslationConfig,
        VoiceConfig, TimingConfig, MixingConfig, SubtitleConfig,
        ExportConfig, BatchConfig, SUPPORTED_LANGUAGES, VOICE_LANGUAGE_MAP,
    )


def test_import_audio():
    from dubbing_studio.audio import AudioExtractor, AudioCleaner, AudioSegmenter, AudioMixer


def test_import_speech():
    from dubbing_studio.speech import SpeechRecognizer, NarrationAnalyzer


def test_import_translation():
    from dubbing_studio.translation import Translator


def test_import_tts():
    from dubbing_studio.tts.engine import TTSEngine, TTSResult
    from dubbing_studio.tts.qwen_tts import QwenTTS
    from dubbing_studio.tts.chatterbox_tts import ChatterboxTTS
    from dubbing_studio.tts.lux_tts import LuxTTS
    from dubbing_studio.tts.voice_selector import VoiceSelector


def test_import_timing():
    from dubbing_studio.timing import TimingAligner


def test_import_subtitle():
    from dubbing_studio.subtitle import SubtitleGenerator


def test_import_video():
    from dubbing_studio.video import VideoRenderer


def test_import_batch():
    from dubbing_studio.batch import BatchProcessor, BatchJob, JobStatus


def test_import_hardware():
    from dubbing_studio.hardware import HardwareOptimizer


def test_import_export():
    from dubbing_studio.export import Exporter


def test_import_models():
    from dubbing_studio.models import ModelManager


def test_import_pipeline():
    from dubbing_studio.pipeline import DubbingPipeline, DubbingResult, PIPELINE_STAGES
