"""
Global configuration for Dubbing Studio.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 44100
    channels: int = 1
    normalize_loudness: float = -23.0  # LUFS
    noise_reduction_strength: float = 0.5
    min_silence_duration: float = 0.5  # seconds
    silence_threshold: float = -40.0  # dB
    segment_min_duration: float = 5.0  # seconds
    segment_max_duration: float = 15.0  # seconds


@dataclass
class WhisperConfig:
    """Whisper speech recognition configuration."""
    model_size: str = "base"  # tiny, base, medium, large-v3, auto
    device: str = "auto"  # auto, cpu, cuda
    language: Optional[str] = None  # None = auto-detect
    compute_type: str = "auto"  # auto, float16, float32, int8 (int8 not supported by openai-whisper)
    beam_size: int = 5
    vad_filter: bool = True


@dataclass
class TranslationConfig:
    """Translation engine configuration."""
    provider: str = "gemini"  # gemini
    api_key: str = ""
    model: str = "gemini-2.0-flash"
    temperature: float = 0.7
    max_retries: int = 3
    system_prompt: str = (
        "Translate this documentary narration into the target language "
        "so it sounds like a professional narrator speaking to an audience. "
        "Avoid word-for-word translation and maintain natural storytelling tone. "
        "Adapt sentence length to be suitable for spoken narration."
    )


@dataclass
class VoiceConfig:
    """Voice selection and TTS configuration."""
    engine: str = "auto"  # auto, qwen3, chatterbox, luxtts
    speed: float = 1.0
    pitch: float = 1.0
    narrator_gender: str = "auto"  # auto, male, female
    narrator_style: str = "documentary"  # documentary, cinematic, calm, storytelling


# Language-to-voice mapping for automatic voice selection
VOICE_LANGUAGE_MAP: dict[str, dict] = {
    "hi": {"gender": "male", "style": "documentary", "engine": "qwen3"},
    "es": {"gender": "male", "style": "cinematic", "engine": "chatterbox"},
    "fr": {"gender": "male", "style": "calm", "engine": "luxtts"},
    "en": {"gender": "male", "style": "storytelling", "engine": "chatterbox"},
    "de": {"gender": "male", "style": "documentary", "engine": "qwen3"},
    "ja": {"gender": "male", "style": "calm", "engine": "qwen3"},
    "ko": {"gender": "male", "style": "documentary", "engine": "qwen3"},
    "zh": {"gender": "male", "style": "documentary", "engine": "qwen3"},
    "ar": {"gender": "male", "style": "documentary", "engine": "qwen3"},
    "pt": {"gender": "male", "style": "cinematic", "engine": "chatterbox"},
    "ru": {"gender": "male", "style": "documentary", "engine": "qwen3"},
    "it": {"gender": "male", "style": "cinematic", "engine": "chatterbox"},
    "tr": {"gender": "male", "style": "documentary", "engine": "qwen3"},
    "pl": {"gender": "male", "style": "documentary", "engine": "qwen3"},
    "nl": {"gender": "male", "style": "calm", "engine": "luxtts"},
    "sv": {"gender": "male", "style": "calm", "engine": "luxtts"},
    "da": {"gender": "male", "style": "calm", "engine": "luxtts"},
    "fi": {"gender": "male", "style": "documentary", "engine": "qwen3"},
    "el": {"gender": "male", "style": "documentary", "engine": "qwen3"},
    "cs": {"gender": "male", "style": "documentary", "engine": "qwen3"},
    "ro": {"gender": "male", "style": "documentary", "engine": "qwen3"},
    "hu": {"gender": "male", "style": "documentary", "engine": "qwen3"},
    "th": {"gender": "male", "style": "calm", "engine": "qwen3"},
    "vi": {"gender": "male", "style": "documentary", "engine": "qwen3"},
}

SUPPORTED_LANGUAGES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ar": "Arabic",
    "pt": "Portuguese",
    "ru": "Russian",
    "it": "Italian",
    "tr": "Turkish",
    "pl": "Polish",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "el": "Greek",
    "cs": "Czech",
    "ro": "Romanian",
    "hu": "Hungarian",
    "th": "Thai",
    "vi": "Vietnamese",
}


@dataclass
class TimingConfig:
    """Speech timing alignment configuration."""
    max_deviation_ms: float = 300.0  # maximum allowed timing difference
    speed_min: float = 0.8
    speed_max: float = 1.3
    allow_pause_insertion: bool = True
    allow_silence_trimming: bool = True
    allow_sentence_compression: bool = True


@dataclass
class MixingConfig:
    """Audio mixing configuration."""
    narration_volume: float = 1.0  # 100%
    background_volume: float = 0.15  # 15%
    ducking_enabled: bool = True
    ducking_threshold: float = -30.0  # dB
    ducking_ratio: float = 0.1  # reduce to 10% during narration
    crossfade_duration: float = 0.1  # seconds


@dataclass
class SubtitleConfig:
    """Subtitle generation configuration."""
    format: str = "srt"  # srt, vtt, ass
    embed_in_video: bool = False
    font_size: int = 24
    font_color: str = "white"
    outline_color: str = "black"
    outline_width: int = 2
    position: str = "bottom"  # bottom, top


@dataclass
class ExportConfig:
    """Export configuration."""
    video_format: str = "mp4"
    video_codec: str = "libx264"
    video_quality: int = 23  # CRF value (lower = better)
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    audio_only_format: str = "wav"  # wav, mp3


@dataclass
class BatchConfig:
    """Batch processing configuration."""
    max_concurrent: int = 4
    retry_on_failure: bool = True
    max_retries: int = 2
    auto_export: bool = True


@dataclass
class AppConfig:
    """Main application configuration."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    mixing: MixingConfig = field(default_factory=MixingConfig)
    subtitle: SubtitleConfig = field(default_factory=SubtitleConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)

    # Directories
    output_dir: str = "output"
    temp_dir: str = "temp"
    cache_dir: str = "cache"

    def setup_dirs(self) -> None:
        """Create necessary directories."""
        for d in [self.output_dir, self.temp_dir, self.cache_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create configuration from environment variables."""
        config = cls()
        config.translation.api_key = os.environ.get("GEMINI_API_KEY", "")
        config.output_dir = os.environ.get("DUBBING_OUTPUT_DIR", "output")
        config.temp_dir = os.environ.get("DUBBING_TEMP_DIR", "temp")

        whisper_model = os.environ.get("WHISPER_MODEL", "base").strip().lower()
        if whisper_model in ("tiny", "base", "medium", "large-v3", "auto"):
            config.whisper.model_size = whisper_model

        return config
