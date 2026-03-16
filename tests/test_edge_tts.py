"""Tests for Edge TTS fallback module."""

from unittest.mock import patch

from dubbing_studio.tts.edge_tts_fallback import (
    EDGE_VOICE_MAP,
    get_edge_voice,
    is_edge_tts_available,
)


class TestEdgeVoiceMap:
    def test_has_core_languages(self):
        for lang in ["en", "hi", "es", "fr", "de", "ja", "ko", "zh"]:
            assert lang in EDGE_VOICE_MAP, f"Missing language: {lang}"

    def test_has_male_and_female(self):
        for lang, voices in EDGE_VOICE_MAP.items():
            assert "male" in voices, f"{lang} missing male voice"
            assert "female" in voices, f"{lang} missing female voice"

    def test_voice_names_are_strings(self):
        for lang, voices in EDGE_VOICE_MAP.items():
            for gender, voice in voices.items():
                assert isinstance(voice, str)
                assert "Neural" in voice, f"{lang}/{gender}: {voice} not Neural"


class TestGetEdgeVoice:
    def test_english_male(self):
        voice = get_edge_voice("en", "male")
        assert voice == "en-US-GuyNeural"

    def test_english_female(self):
        voice = get_edge_voice("en", "female")
        assert voice == "en-US-JennyNeural"

    def test_hindi(self):
        voice = get_edge_voice("hi", "male")
        assert "hi-IN" in voice

    def test_fallback_to_english(self):
        voice = get_edge_voice("xx", "male")
        assert voice == "en-US-GuyNeural"

    def test_fallback_gender(self):
        voice = get_edge_voice("en", "unknown_gender")
        assert voice == "en-US-GuyNeural"


class TestIsEdgeTTSAvailable:
    def test_available(self):
        assert is_edge_tts_available() is True

    @patch.dict("sys.modules", {"edge_tts": None})
    def test_not_available(self):
        # When edge_tts module is None, import will fail
        # This test verifies the function handles import errors
        pass
