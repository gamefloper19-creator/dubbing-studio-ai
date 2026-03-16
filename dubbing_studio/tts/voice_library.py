"""
Expanded Voice Library.

Provides a comprehensive collection of voices organized by
language, style (cinematic/documentary/neutral), and gender.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class VoiceEntry:
    """A voice entry in the library."""
    voice_id: str
    name: str
    engine: str  # qwen3, chatterbox, luxtts, edge-tts
    language: str
    gender: str  # male, female
    style: str  # cinematic, documentary, neutral, storytelling
    edge_voice: str = ""  # edge-tts voice name for fallback


# Comprehensive voice library organized by language, style, and gender
VOICE_LIBRARY: list[VoiceEntry] = [
    # ── English ──
    VoiceEntry("en_male_documentary", "David (Documentary)", "chatterbox", "en", "male", "documentary", "en-US-GuyNeural"),
    VoiceEntry("en_female_documentary", "Sarah (Documentary)", "chatterbox", "en", "female", "documentary", "en-US-JennyNeural"),
    VoiceEntry("en_male_cinematic", "James (Cinematic)", "chatterbox", "en", "male", "cinematic", "en-US-ChristopherNeural"),
    VoiceEntry("en_female_cinematic", "Emily (Cinematic)", "chatterbox", "en", "female", "cinematic", "en-US-AriaNeural"),
    VoiceEntry("en_male_neutral", "Mark (Neutral)", "qwen3", "en", "male", "neutral", "en-US-DavisNeural"),
    VoiceEntry("en_female_neutral", "Lisa (Neutral)", "qwen3", "en", "female", "neutral", "en-US-AmberNeural"),
    VoiceEntry("en_male_storytelling", "Robert (Storytelling)", "chatterbox", "en", "male", "storytelling", "en-GB-RyanNeural"),
    VoiceEntry("en_female_storytelling", "Victoria (Storytelling)", "chatterbox", "en", "female", "storytelling", "en-GB-SoniaNeural"),

    # ── Hindi ──
    VoiceEntry("hi_male_documentary", "Arun (Documentary)", "qwen3", "hi", "male", "documentary", "hi-IN-MadhurNeural"),
    VoiceEntry("hi_female_documentary", "Priya (Documentary)", "qwen3", "hi", "female", "documentary", "hi-IN-SwaraNeural"),
    VoiceEntry("hi_male_neutral", "Raj (Neutral)", "qwen3", "hi", "male", "neutral", "hi-IN-MadhurNeural"),
    VoiceEntry("hi_female_neutral", "Kavya (Neutral)", "qwen3", "hi", "female", "neutral", "hi-IN-SwaraNeural"),

    # ── Spanish ──
    VoiceEntry("es_male_documentary", "Carlos (Documentary)", "chatterbox", "es", "male", "documentary", "es-ES-AlvaroNeural"),
    VoiceEntry("es_female_documentary", "Maria (Documentary)", "chatterbox", "es", "female", "documentary", "es-ES-ElviraNeural"),
    VoiceEntry("es_male_cinematic", "Diego (Cinematic)", "chatterbox", "es", "male", "cinematic", "es-MX-JorgeNeural"),
    VoiceEntry("es_female_cinematic", "Sofia (Cinematic)", "chatterbox", "es", "female", "cinematic", "es-MX-DaliaNeural"),

    # ── French ──
    VoiceEntry("fr_male_documentary", "Pierre (Documentary)", "luxtts", "fr", "male", "documentary", "fr-FR-HenriNeural"),
    VoiceEntry("fr_female_documentary", "Claire (Documentary)", "luxtts", "fr", "female", "documentary", "fr-FR-DeniseNeural"),
    VoiceEntry("fr_male_neutral", "Jean (Neutral)", "luxtts", "fr", "male", "neutral", "fr-FR-HenriNeural"),
    VoiceEntry("fr_female_neutral", "Marie (Neutral)", "luxtts", "fr", "female", "neutral", "fr-FR-DeniseNeural"),

    # ── German ──
    VoiceEntry("de_male_documentary", "Hans (Documentary)", "luxtts", "de", "male", "documentary", "de-DE-ConradNeural"),
    VoiceEntry("de_female_documentary", "Anna (Documentary)", "luxtts", "de", "female", "documentary", "de-DE-KatjaNeural"),
    VoiceEntry("de_male_neutral", "Klaus (Neutral)", "qwen3", "de", "male", "neutral", "de-DE-ConradNeural"),
    VoiceEntry("de_female_neutral", "Heidi (Neutral)", "qwen3", "de", "female", "neutral", "de-DE-KatjaNeural"),

    # ── Japanese ──
    VoiceEntry("ja_male_documentary", "Takeshi (Documentary)", "qwen3", "ja", "male", "documentary", "ja-JP-KeitaNeural"),
    VoiceEntry("ja_female_documentary", "Yuki (Documentary)", "qwen3", "ja", "female", "documentary", "ja-JP-NanamiNeural"),
    VoiceEntry("ja_male_neutral", "Haruto (Neutral)", "qwen3", "ja", "male", "neutral", "ja-JP-KeitaNeural"),
    VoiceEntry("ja_female_neutral", "Sakura (Neutral)", "qwen3", "ja", "female", "neutral", "ja-JP-NanamiNeural"),

    # ── Korean ──
    VoiceEntry("ko_male_documentary", "Minho (Documentary)", "qwen3", "ko", "male", "documentary", "ko-KR-InJoonNeural"),
    VoiceEntry("ko_female_documentary", "Jisoo (Documentary)", "qwen3", "ko", "female", "documentary", "ko-KR-SunHiNeural"),

    # ── Chinese ──
    VoiceEntry("zh_male_documentary", "Wei (Documentary)", "qwen3", "zh", "male", "documentary", "zh-CN-YunxiNeural"),
    VoiceEntry("zh_female_documentary", "Mei (Documentary)", "qwen3", "zh", "female", "documentary", "zh-CN-XiaoxiaoNeural"),
    VoiceEntry("zh_male_cinematic", "Long (Cinematic)", "qwen3", "zh", "male", "cinematic", "zh-CN-YunjianNeural"),
    VoiceEntry("zh_female_cinematic", "Ling (Cinematic)", "qwen3", "zh", "female", "cinematic", "zh-CN-XiaohanNeural"),

    # ── Arabic ──
    VoiceEntry("ar_male_documentary", "Ahmed (Documentary)", "qwen3", "ar", "male", "documentary", "ar-SA-HamedNeural"),
    VoiceEntry("ar_female_documentary", "Fatima (Documentary)", "qwen3", "ar", "female", "documentary", "ar-SA-ZariyahNeural"),

    # ── Portuguese ──
    VoiceEntry("pt_male_documentary", "Miguel (Documentary)", "chatterbox", "pt", "male", "documentary", "pt-BR-AntonioNeural"),
    VoiceEntry("pt_female_documentary", "Ana (Documentary)", "chatterbox", "pt", "female", "documentary", "pt-BR-FranciscaNeural"),

    # ── Russian ──
    VoiceEntry("ru_male_documentary", "Ivan (Documentary)", "qwen3", "ru", "male", "documentary", "ru-RU-DmitryNeural"),
    VoiceEntry("ru_female_documentary", "Natasha (Documentary)", "qwen3", "ru", "female", "documentary", "ru-RU-SvetlanaNeural"),

    # ── Italian ──
    VoiceEntry("it_male_documentary", "Marco (Documentary)", "chatterbox", "it", "male", "documentary", "it-IT-DiegoNeural"),
    VoiceEntry("it_female_documentary", "Giulia (Documentary)", "chatterbox", "it", "female", "documentary", "it-IT-ElsaNeural"),

    # ── Turkish ──
    VoiceEntry("tr_male_documentary", "Emre (Documentary)", "qwen3", "tr", "male", "documentary", "tr-TR-AhmetNeural"),
    VoiceEntry("tr_female_documentary", "Zeynep (Documentary)", "qwen3", "tr", "female", "documentary", "tr-TR-EmelNeural"),

    # ── Dutch ──
    VoiceEntry("nl_male_documentary", "Jan (Documentary)", "luxtts", "nl", "male", "documentary", "nl-NL-MaartenNeural"),
    VoiceEntry("nl_female_documentary", "Eva (Documentary)", "luxtts", "nl", "female", "documentary", "nl-NL-ColetteNeural"),

    # ── Swedish ──
    VoiceEntry("sv_male_documentary", "Erik (Documentary)", "luxtts", "sv", "male", "documentary", "sv-SE-MattiasNeural"),
    VoiceEntry("sv_female_documentary", "Astrid (Documentary)", "luxtts", "sv", "female", "documentary", "sv-SE-SofieNeural"),

    # ── Danish ──
    VoiceEntry("da_male_documentary", "Lars (Documentary)", "luxtts", "da", "male", "documentary", "da-DK-JeppeNeural"),
    VoiceEntry("da_female_documentary", "Ida (Documentary)", "luxtts", "da", "female", "documentary", "da-DK-ChristelNeural"),

    # ── Polish ──
    VoiceEntry("pl_male_documentary", "Piotr (Documentary)", "qwen3", "pl", "male", "documentary", "pl-PL-MarekNeural"),
    VoiceEntry("pl_female_documentary", "Kasia (Documentary)", "qwen3", "pl", "female", "documentary", "pl-PL-ZofiaNeural"),

    # ── Finnish ──
    VoiceEntry("fi_male_documentary", "Matti (Documentary)", "qwen3", "fi", "male", "documentary", "fi-FI-HarriNeural"),
    VoiceEntry("fi_female_documentary", "Aino (Documentary)", "qwen3", "fi", "female", "documentary", "fi-FI-NooraNeural"),

    # ── Greek ──
    VoiceEntry("el_male_documentary", "Nikos (Documentary)", "qwen3", "el", "male", "documentary", "el-GR-NestorasNeural"),
    VoiceEntry("el_female_documentary", "Elena (Documentary)", "qwen3", "el", "female", "documentary", "el-GR-AthinaNeural"),

    # ── Czech ──
    VoiceEntry("cs_male_documentary", "Tomas (Documentary)", "qwen3", "cs", "male", "documentary", "cs-CZ-AntoninNeural"),
    VoiceEntry("cs_female_documentary", "Jana (Documentary)", "qwen3", "cs", "female", "documentary", "cs-CZ-VlastaNeural"),

    # ── Romanian ──
    VoiceEntry("ro_male_documentary", "Andrei (Documentary)", "qwen3", "ro", "male", "documentary", "ro-RO-EmilNeural"),
    VoiceEntry("ro_female_documentary", "Ioana (Documentary)", "qwen3", "ro", "female", "documentary", "ro-RO-AlinaNeural"),

    # ── Hungarian ──
    VoiceEntry("hu_male_documentary", "Gabor (Documentary)", "qwen3", "hu", "male", "documentary", "hu-HU-TamasNeural"),
    VoiceEntry("hu_female_documentary", "Zsofia (Documentary)", "qwen3", "hu", "female", "documentary", "hu-HU-NoemiNeural"),

    # ── Thai ──
    VoiceEntry("th_male_documentary", "Somchai (Documentary)", "qwen3", "th", "male", "documentary", "th-TH-NiwatNeural"),
    VoiceEntry("th_female_documentary", "Ploy (Documentary)", "qwen3", "th", "female", "documentary", "th-TH-PremwadeeNeural"),

    # ── Vietnamese ──
    VoiceEntry("vi_male_documentary", "Minh (Documentary)", "qwen3", "vi", "male", "documentary", "vi-VN-NamMinhNeural"),
    VoiceEntry("vi_female_documentary", "Linh (Documentary)", "qwen3", "vi", "female", "documentary", "vi-VN-HoaiMyNeural"),
]


class VoiceLibrary:
    """
    Expanded voice library with filtering by language, style, and gender.
    """

    def __init__(self):
        self._voices = list(VOICE_LIBRARY)

    def get_voices(
        self,
        language: Optional[str] = None,
        gender: Optional[str] = None,
        style: Optional[str] = None,
    ) -> list[VoiceEntry]:
        """
        Get voices matching the given criteria.

        Args:
            language: Language code filter (e.g., 'en', 'hi').
            gender: Gender filter ('male' or 'female').
            style: Style filter ('cinematic', 'documentary', 'neutral', 'storytelling').

        Returns:
            List of matching VoiceEntry objects.
        """
        result = self._voices

        if language:
            result = [v for v in result if v.language == language]
        if gender:
            result = [v for v in result if v.gender == gender]
        if style:
            result = [v for v in result if v.style == style]

        return result

    def get_voice_by_id(self, voice_id: str) -> Optional[VoiceEntry]:
        """Get a specific voice by ID."""
        return next((v for v in self._voices if v.voice_id == voice_id), None)

    def get_best_voice(
        self,
        language: str,
        gender: str = "male",
        style: str = "documentary",
    ) -> Optional[VoiceEntry]:
        """
        Get the best matching voice for given criteria.

        Falls back to any available voice for the language if exact match not found.
        """
        # Try exact match
        voices = self.get_voices(language=language, gender=gender, style=style)
        if voices:
            return voices[0]

        # Fall back to same language + gender
        voices = self.get_voices(language=language, gender=gender)
        if voices:
            return voices[0]

        # Fall back to same language
        voices = self.get_voices(language=language)
        if voices:
            return voices[0]

        # No match
        return None

    def get_available_languages(self) -> list[str]:
        """Get list of languages with available voices."""
        return sorted(set(v.language for v in self._voices))

    def get_available_styles(self, language: Optional[str] = None) -> list[str]:
        """Get available styles, optionally filtered by language."""
        voices = self._voices
        if language:
            voices = [v for v in voices if v.language == language]
        return sorted(set(v.style for v in voices))

    def get_available_genders(self, language: Optional[str] = None) -> list[str]:
        """Get available genders, optionally filtered by language."""
        voices = self._voices
        if language:
            voices = [v for v in voices if v.language == language]
        return sorted(set(v.gender for v in voices))

    def get_voice_count(self) -> int:
        """Get total number of voices in the library."""
        return len(self._voices)

    def get_library_summary(self) -> dict:
        """Get a summary of the voice library."""
        languages = self.get_available_languages()
        return {
            "total_voices": len(self._voices),
            "languages": len(languages),
            "styles": sorted(set(v.style for v in self._voices)),
            "engines": sorted(set(v.engine for v in self._voices)),
            "by_language": {
                lang: len(self.get_voices(language=lang))
                for lang in languages
            },
        }

    def format_for_display(
        self,
        language: Optional[str] = None,
    ) -> list[list[str]]:
        """
        Format voices for UI display as table rows.

        Returns:
            List of [name, language, gender, style, engine] rows.
        """
        voices = self.get_voices(language=language)
        return [
            [v.name, v.language, v.gender, v.style, v.engine]
            for v in voices
        ]
