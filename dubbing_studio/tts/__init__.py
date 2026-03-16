"""Text-to-speech engine modules."""

from dubbing_studio.tts.engine import TTSEngine, TTSResult
from dubbing_studio.tts.qwen_tts import QwenTTS
from dubbing_studio.tts.chatterbox_tts import ChatterboxTTS
from dubbing_studio.tts.lux_tts import LuxTTS
from dubbing_studio.tts.voice_selector import VoiceSelector
from dubbing_studio.tts.voice_library import VoiceLibrary

__all__ = [
    "TTSEngine", "TTSResult",
    "QwenTTS", "ChatterboxTTS", "LuxTTS",
    "VoiceSelector", "VoiceLibrary",
]
