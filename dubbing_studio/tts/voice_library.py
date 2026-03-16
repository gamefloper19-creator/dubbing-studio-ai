"""
Voice Library - Catalog of all available voices across TTS engines.

Provides a unified interface to browse, filter, and select voices
from all supported TTS engines.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from dubbing_studio.config import SUPPORTED_LANGUAGES, VOICE_LANGUAGE_MAP

logger = logging.getLogger(__name__)


@dataclass
class VoiceEntry:
    """A single voice in the library."""
    voice_id: str
    name: str
    engine: str
    gender: str  # male, female
    language_codes: list[str] = field(default_factory=list)
    style: str = "documentary"  # documentary, cinematic, calm, storytelling
    quality: str = "standard"  # standard, premium
    description: str = ""


# Edge TTS voice catalog (always available as fallback)
EDGE_TTS_VOICES: list[dict] = [
    # English
    {"voice_id": "en-US-GuyNeural", "name": "Guy (US)", "engine": "edge-tts", "gender": "male", "language_codes": ["en"], "style": "storytelling", "quality": "standard"},
    {"voice_id": "en-US-ChristopherNeural", "name": "Christopher (US)", "engine": "edge-tts", "gender": "male", "language_codes": ["en"], "style": "cinematic", "quality": "standard"},
    {"voice_id": "en-GB-RyanNeural", "name": "Ryan (UK)", "engine": "edge-tts", "gender": "male", "language_codes": ["en"], "style": "calm", "quality": "standard"},
    {"voice_id": "en-US-JennyNeural", "name": "Jenny (US)", "engine": "edge-tts", "gender": "female", "language_codes": ["en"], "style": "documentary", "quality": "standard"},
    # Hindi
    {"voice_id": "hi-IN-MadhurNeural", "name": "Madhur (India)", "engine": "edge-tts", "gender": "male", "language_codes": ["hi"], "style": "documentary", "quality": "standard"},
    {"voice_id": "hi-IN-SwaraNeural", "name": "Swara (India)", "engine": "edge-tts", "gender": "female", "language_codes": ["hi"], "style": "documentary", "quality": "standard"},
    # Spanish
    {"voice_id": "es-ES-AlvaroNeural", "name": "Alvaro (Spain)", "engine": "edge-tts", "gender": "male", "language_codes": ["es"], "style": "cinematic", "quality": "standard"},
    {"voice_id": "es-MX-JorgeNeural", "name": "Jorge (Mexico)", "engine": "edge-tts", "gender": "male", "language_codes": ["es"], "style": "documentary", "quality": "standard"},
    # French
    {"voice_id": "fr-FR-HenriNeural", "name": "Henri (France)", "engine": "edge-tts", "gender": "male", "language_codes": ["fr"], "style": "calm", "quality": "standard"},
    {"voice_id": "fr-FR-DeniseNeural", "name": "Denise (France)", "engine": "edge-tts", "gender": "female", "language_codes": ["fr"], "style": "calm", "quality": "standard"},
    # German
    {"voice_id": "de-DE-ConradNeural", "name": "Conrad (Germany)", "engine": "edge-tts", "gender": "male", "language_codes": ["de"], "style": "documentary", "quality": "standard"},
    {"voice_id": "de-DE-KillianNeural", "name": "Killian (Germany)", "engine": "edge-tts", "gender": "male", "language_codes": ["de"], "style": "calm", "quality": "standard"},
    # Japanese
    {"voice_id": "ja-JP-KeitaNeural", "name": "Keita (Japan)", "engine": "edge-tts", "gender": "male", "language_codes": ["ja"], "style": "calm", "quality": "standard"},
    {"voice_id": "ja-JP-NanamiNeural", "name": "Nanami (Japan)", "engine": "edge-tts", "gender": "female", "language_codes": ["ja"], "style": "calm", "quality": "standard"},
    # Korean
    {"voice_id": "ko-KR-InJoonNeural", "name": "InJoon (Korea)", "engine": "edge-tts", "gender": "male", "language_codes": ["ko"], "style": "documentary", "quality": "standard"},
    # Chinese
    {"voice_id": "zh-CN-YunxiNeural", "name": "Yunxi (China)", "engine": "edge-tts", "gender": "male", "language_codes": ["zh"], "style": "documentary", "quality": "standard"},
    {"voice_id": "zh-CN-XiaoxiaoNeural", "name": "Xiaoxiao (China)", "engine": "edge-tts", "gender": "female", "language_codes": ["zh"], "style": "documentary", "quality": "standard"},
    # Arabic
    {"voice_id": "ar-SA-HamedNeural", "name": "Hamed (Saudi Arabia)", "engine": "edge-tts", "gender": "male", "language_codes": ["ar"], "style": "documentary", "quality": "standard"},
    # Portuguese
    {"voice_id": "pt-BR-AntonioNeural", "name": "Antonio (Brazil)", "engine": "edge-tts", "gender": "male", "language_codes": ["pt"], "style": "cinematic", "quality": "standard"},
    # Russian
    {"voice_id": "ru-RU-DmitryNeural", "name": "Dmitry (Russia)", "engine": "edge-tts", "gender": "male", "language_codes": ["ru"], "style": "documentary", "quality": "standard"},
    # Italian
    {"voice_id": "it-IT-DiegoNeural", "name": "Diego (Italy)", "engine": "edge-tts", "gender": "male", "language_codes": ["it"], "style": "cinematic", "quality": "standard"},
    # Turkish
    {"voice_id": "tr-TR-AhmetNeural", "name": "Ahmet (Turkey)", "engine": "edge-tts", "gender": "male", "language_codes": ["tr"], "style": "documentary", "quality": "standard"},
    # Polish
    {"voice_id": "pl-PL-MarekNeural", "name": "Marek (Poland)", "engine": "edge-tts", "gender": "male", "language_codes": ["pl"], "style": "documentary", "quality": "standard"},
    # Dutch
    {"voice_id": "nl-NL-MaartenNeural", "name": "Maarten (Netherlands)", "engine": "edge-tts", "gender": "male", "language_codes": ["nl"], "style": "calm", "quality": "standard"},
    # Swedish
    {"voice_id": "sv-SE-MattiasNeural", "name": "Mattias (Sweden)", "engine": "edge-tts", "gender": "male", "language_codes": ["sv"], "style": "calm", "quality": "standard"},
    # Danish
    {"voice_id": "da-DK-JeppeNeural", "name": "Jeppe (Denmark)", "engine": "edge-tts", "gender": "male", "language_codes": ["da"], "style": "calm", "quality": "standard"},
    # Finnish
    {"voice_id": "fi-FI-HarriNeural", "name": "Harri (Finland)", "engine": "edge-tts", "gender": "male", "language_codes": ["fi"], "style": "documentary", "quality": "standard"},
    # Greek
    {"voice_id": "el-GR-NestorasNeural", "name": "Nestoras (Greece)", "engine": "edge-tts", "gender": "male", "language_codes": ["el"], "style": "documentary", "quality": "standard"},
    # Czech
    {"voice_id": "cs-CZ-AntoninNeural", "name": "Antonin (Czech Republic)", "engine": "edge-tts", "gender": "male", "language_codes": ["cs"], "style": "documentary", "quality": "standard"},
    # Romanian
    {"voice_id": "ro-RO-EmilNeural", "name": "Emil (Romania)", "engine": "edge-tts", "gender": "male", "language_codes": ["ro"], "style": "documentary", "quality": "standard"},
    # Hungarian
    {"voice_id": "hu-HU-TamasNeural", "name": "Tamas (Hungary)", "engine": "edge-tts", "gender": "male", "language_codes": ["hu"], "style": "documentary", "quality": "standard"},
    # Thai
    {"voice_id": "th-TH-NiwatNeural", "name": "Niwat (Thailand)", "engine": "edge-tts", "gender": "male", "language_codes": ["th"], "style": "calm", "quality": "standard"},
    # Vietnamese
    {"voice_id": "vi-VN-NamMinhNeural", "name": "NamMinh (Vietnam)", "engine": "edge-tts", "gender": "male", "language_codes": ["vi"], "style": "documentary", "quality": "standard"},
]

# Neural TTS engine voices (premium, require model download)
NEURAL_TTS_VOICES: list[dict] = [
    # Qwen3-TTS
    {"voice_id": "qwen3-narrator-male", "name": "Qwen3 Male Narrator", "engine": "qwen3-tts", "gender": "male", "language_codes": ["en", "zh", "ja", "ko", "hi", "ar", "de", "ru", "tr", "pl", "fi", "el", "cs", "ro", "hu", "th", "vi"], "style": "documentary", "quality": "premium"},
    {"voice_id": "qwen3-narrator-female", "name": "Qwen3 Female Narrator", "engine": "qwen3-tts", "gender": "female", "language_codes": ["en", "zh", "ja", "ko", "hi", "ar", "de", "ru", "tr", "pl", "fi", "el", "cs", "ro", "hu", "th", "vi"], "style": "documentary", "quality": "premium"},
    # Chatterbox
    {"voice_id": "chatterbox-cinematic-male", "name": "Cinematic Male", "engine": "chatterbox", "gender": "male", "language_codes": ["en", "es", "pt", "it", "fr"], "style": "cinematic", "quality": "premium"},
    {"voice_id": "chatterbox-storyteller-male", "name": "Storyteller Male", "engine": "chatterbox", "gender": "male", "language_codes": ["en"], "style": "storytelling", "quality": "premium"},
    {"voice_id": "chatterbox-narrator-female", "name": "Narrator Female", "engine": "chatterbox", "gender": "female", "language_codes": ["en", "es", "pt", "it"], "style": "documentary", "quality": "premium"},
    # LuxTTS (Coqui XTTS v2)
    {"voice_id": "lux-calm-male", "name": "Calm Male Narrator", "engine": "luxtts", "gender": "male", "language_codes": ["en", "fr", "nl", "sv", "da", "de"], "style": "calm", "quality": "premium"},
    {"voice_id": "lux-neutral-female", "name": "Neutral Female Narrator", "engine": "luxtts", "gender": "female", "language_codes": ["en", "fr", "nl", "sv", "da", "de"], "style": "calm", "quality": "premium"},
]


class VoiceLibrary:
    """
    Unified voice library across all TTS engines.

    Provides browsing, filtering, and selection of voices
    from Edge TTS, Qwen3-TTS, Chatterbox, and LuxTTS.
    """

    def __init__(self):
        self._voices: list[VoiceEntry] = []
        self._load_catalog()

    def _load_catalog(self) -> None:
        """Load the full voice catalog."""
        self._voices = []

        # Add Edge TTS voices (always available)
        for v in EDGE_TTS_VOICES:
            self._voices.append(VoiceEntry(**v))

        # Add neural TTS voices (may require model download)
        for v in NEURAL_TTS_VOICES:
            self._voices.append(VoiceEntry(**v))

    def get_all_voices(self) -> list[VoiceEntry]:
        """Get all voices in the library."""
        return list(self._voices)

    def get_voices_for_language(self, language_code: str) -> list[VoiceEntry]:
        """
        Get voices that support a specific language.

        Args:
            language_code: Language code (e.g., 'hi', 'es').

        Returns:
            List of VoiceEntry objects supporting the language.
        """
        return [
            v for v in self._voices
            if language_code in v.language_codes
        ]

    def get_voices_by_style(self, style: str) -> list[VoiceEntry]:
        """
        Get voices matching a narration style.

        Args:
            style: Narration style (documentary, cinematic, calm, storytelling).

        Returns:
            List of VoiceEntry objects matching the style.
        """
        return [
            v for v in self._voices
            if v.style == style
        ]

    def get_voices_by_gender(self, gender: str) -> list[VoiceEntry]:
        """
        Get voices matching a gender.

        Args:
            gender: Voice gender (male, female).

        Returns:
            List of VoiceEntry objects matching the gender.
        """
        return [
            v for v in self._voices
            if v.gender == gender
        ]

    def get_voices_by_engine(self, engine: str) -> list[VoiceEntry]:
        """
        Get voices from a specific TTS engine.

        Args:
            engine: Engine name (edge-tts, qwen3-tts, chatterbox, luxtts).

        Returns:
            List of VoiceEntry objects from the engine.
        """
        return [
            v for v in self._voices
            if v.engine == engine
        ]

    def search_voices(
        self,
        language: Optional[str] = None,
        gender: Optional[str] = None,
        style: Optional[str] = None,
        engine: Optional[str] = None,
        quality: Optional[str] = None,
    ) -> list[VoiceEntry]:
        """
        Search voices with multiple filters.

        Args:
            language: Filter by language code.
            gender: Filter by gender.
            style: Filter by narration style.
            engine: Filter by TTS engine.
            quality: Filter by quality level (standard, premium).

        Returns:
            List of matching VoiceEntry objects.
        """
        results = list(self._voices)

        if language:
            results = [v for v in results if language in v.language_codes]
        if gender:
            results = [v for v in results if v.gender == gender]
        if style:
            results = [v for v in results if v.style == style]
        if engine:
            results = [v for v in results if v.engine == engine]
        if quality:
            results = [v for v in results if v.quality == quality]

        return results

    def get_voice_by_id(self, voice_id: str) -> Optional[VoiceEntry]:
        """
        Get a specific voice by its ID.

        Args:
            voice_id: Unique voice identifier.

        Returns:
            VoiceEntry or None if not found.
        """
        for v in self._voices:
            if v.voice_id == voice_id:
                return v
        return None

    def get_recommended_voice(
        self,
        language: str,
        style: str = "documentary",
        gender: str = "male",
    ) -> Optional[VoiceEntry]:
        """
        Get the recommended voice for a language and style combination.

        Prefers premium (neural) voices, falls back to Edge TTS.

        Args:
            language: Language code.
            style: Narration style.
            gender: Preferred gender.

        Returns:
            Best matching VoiceEntry, or None.
        """
        # Try premium voices first
        premium = self.search_voices(
            language=language, style=style, gender=gender, quality="premium"
        )
        if premium:
            return premium[0]

        # Try premium with any style
        premium_any = self.search_voices(
            language=language, gender=gender, quality="premium"
        )
        if premium_any:
            return premium_any[0]

        # Fall back to standard (Edge TTS)
        standard = self.search_voices(
            language=language, gender=gender, quality="standard"
        )
        if standard:
            return standard[0]

        # Any voice for the language
        any_voice = self.search_voices(language=language)
        if any_voice:
            return any_voice[0]

        return None

    def get_supported_languages(self) -> dict[str, str]:
        """Get all languages that have at least one voice available."""
        available = set()
        for v in self._voices:
            available.update(v.language_codes)

        return {
            code: name
            for code, name in SUPPORTED_LANGUAGES.items()
            if code in available
        }

    def get_engine_summary(self) -> dict[str, int]:
        """Get count of voices per engine."""
        summary: dict[str, int] = {}
        for v in self._voices:
            summary[v.engine] = summary.get(v.engine, 0) + 1
        return summary
