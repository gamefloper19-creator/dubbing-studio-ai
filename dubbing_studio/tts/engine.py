"""
Base TTS engine interface and common functionality.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TTSResult:
    """Result from TTS generation."""
    audio_path: str
    duration: float  # seconds
    sample_rate: int
    text: str


class TTSEngine(ABC):
    """Abstract base class for TTS engines."""

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Name of the TTS engine."""
        ...

    @property
    @abstractmethod
    def supported_languages(self) -> list[str]:
        """List of supported language codes."""
        ...

    @abstractmethod
    def generate_speech(
        self,
        text: str,
        output_path: str,
        language: str = "en",
        voice_id: Optional[str] = None,
        speed: float = 1.0,
        pitch: float = 1.0,
    ) -> TTSResult:
        """
        Generate speech from text.

        Args:
            text: Text to synthesize.
            output_path: Path for output audio file.
            language: Language code.
            voice_id: Optional specific voice ID.
            speed: Speech speed multiplier.
            pitch: Pitch multiplier (1.0 = no change).

        Returns:
            TTSResult with audio information.
        """
        ...

    @abstractmethod
    def list_voices(self, language: str = "") -> list[dict]:
        """
        List available voices, optionally filtered by language.

        Args:
            language: Optional language code to filter by.

        Returns:
            List of voice info dictionaries.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this TTS engine is available (installed and working)."""
        ...

    def unload(self) -> None:
        """Unload model to free resources. Override in subclasses."""
        pass

    def supports_language(self, language: str) -> bool:
        """Check if a language is supported."""
        return language in self.supported_languages
