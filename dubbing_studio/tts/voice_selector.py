"""
Automatic voice selection based on target language and narration style.
"""

import logging
from typing import Optional

from dubbing_studio.config import VOICE_LANGUAGE_MAP, VoiceConfig
from dubbing_studio.speech.analyzer import NarrationStyle
from dubbing_studio.tts.engine import TTSEngine
from dubbing_studio.tts.qwen_tts import QwenTTS
from dubbing_studio.tts.chatterbox_tts import ChatterboxTTS
from dubbing_studio.tts.lux_tts import LuxTTS

logger = logging.getLogger(__name__)


class VoiceSelector:
    """
    Automatically select the best voice model for the target language
    and narration style.
    """

    def __init__(self, config: Optional[VoiceConfig] = None):
        self.config = config or VoiceConfig()
        self._engines: dict[str, TTSEngine] = {}

    def _get_engine(self, engine_name: str) -> TTSEngine:
        """Get or create a TTS engine by name."""
        if engine_name not in self._engines:
            if engine_name == "qwen3":
                self._engines[engine_name] = QwenTTS()
            elif engine_name == "chatterbox":
                self._engines[engine_name] = ChatterboxTTS()
            elif engine_name == "luxtts":
                self._engines[engine_name] = LuxTTS()
            else:
                raise ValueError(f"Unknown TTS engine: {engine_name}")

        return self._engines[engine_name]

    def select_voice(
        self,
        target_language: str,
        narration_style: Optional[NarrationStyle] = None,
    ) -> tuple[TTSEngine, dict]:
        """
        Select the best voice for the target language and style.

        Args:
            target_language: Target language code (e.g., 'hi', 'es').
            narration_style: Optional detected narration style from original audio.

        Returns:
            Tuple of (TTSEngine instance, voice configuration dict).
        """
        # Get language-specific voice mapping
        voice_info = VOICE_LANGUAGE_MAP.get(target_language, {
            "gender": "male",
            "style": "documentary",
            "engine": "qwen3",
        })

        # Override with narration style if available
        if narration_style:
            if narration_style.gender in ("male", "female"):
                voice_info["gender"] = narration_style.gender

        # Override with user config if not auto
        if self.config.engine != "auto":
            voice_info["engine"] = self.config.engine
        if self.config.narrator_gender != "auto":
            voice_info["gender"] = self.config.narrator_gender
        if self.config.narrator_style != "documentary":
            voice_info["style"] = self.config.narrator_style

        engine_name = voice_info["engine"]

        # Try to get the requested engine, fall back to alternatives
        engine = self._get_available_engine(engine_name, target_language)

        voice_config = {
            "gender": voice_info.get("gender", "male"),
            "style": voice_info.get("style", "documentary"),
            "speed": self.config.speed,
            "pitch": self.config.pitch,
            "language": target_language,
        }

        logger.info(
            "Selected voice: engine=%s, gender=%s, style=%s, language=%s",
            engine.engine_name, voice_config["gender"],
            voice_config["style"], target_language,
        )

        return engine, voice_config

    def _get_available_engine(
        self,
        preferred_engine: str,
        language: str,
    ) -> TTSEngine:
        """Get an available engine, with fallback logic."""
        # Priority order for fallback
        engine_priority = [preferred_engine]

        # Add fallbacks based on language
        if preferred_engine != "chatterbox":
            engine_priority.append("chatterbox")
        if preferred_engine != "qwen3":
            engine_priority.append("qwen3")
        if preferred_engine != "luxtts":
            engine_priority.append("luxtts")

        for engine_name in engine_priority:
            try:
                engine = self._get_engine(engine_name)
                if engine.is_available() and engine.supports_language(language):
                    return engine
                elif engine.is_available():
                    # Engine available but doesn't officially support language
                    # Still usable as fallback
                    logger.warning(
                        "Engine %s doesn't officially support %s, using as fallback",
                        engine_name, language,
                    )
                    return engine
            except Exception as e:
                logger.warning("Engine %s not available: %s", engine_name, e)
                continue

        raise RuntimeError(
            f"No TTS engine available for language '{language}'. "
            f"Please install at least one TTS engine (edge-tts recommended)."
        )

    def unload_all(self) -> None:
        """Unload all loaded TTS engines."""
        for engine in self._engines.values():
            engine.unload()
        self._engines.clear()
        logger.info("All TTS engines unloaded")
