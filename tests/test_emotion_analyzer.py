"""Tests for emotion analysis module."""

from dubbing_studio.config import EmotionConfig
from dubbing_studio.emotion.analyzer import (
    EMOTION_KEYWORDS,
    EMOTION_VOICE_PARAMS,
    EmotionAnalyzer,
    EmotionProfile,
)


class TestEmotionProfile:
    def test_creation(self):
        profile = EmotionProfile(
            emotion="dramatic",
            confidence=0.85,
            pitch_modifier=1.15,
            speed_modifier=1.1,
            pause_modifier=0.7,
            energy_level="high",
        )
        assert profile.emotion == "dramatic"
        assert profile.confidence == 0.85
        assert profile.energy_level == "high"

    def test_default_details(self):
        profile = EmotionProfile(
            emotion="neutral",
            confidence=1.0,
            pitch_modifier=1.0,
            speed_modifier=1.0,
            pause_modifier=1.0,
            energy_level="medium",
        )
        assert profile.details == {}


class TestEmotionKeywords:
    def test_has_all_categories(self):
        assert "dramatic" in EMOTION_KEYWORDS
        assert "suspense" in EMOTION_KEYWORDS
        assert "inspirational" in EMOTION_KEYWORDS
        assert "calm" in EMOTION_KEYWORDS

    def test_keywords_not_empty(self):
        for category, words in EMOTION_KEYWORDS.items():
            assert len(words) > 0, f"{category} has no keywords"


class TestEmotionVoiceParams:
    def test_has_all_emotions(self):
        assert "neutral" in EMOTION_VOICE_PARAMS
        assert "dramatic" in EMOTION_VOICE_PARAMS
        assert "suspense" in EMOTION_VOICE_PARAMS
        assert "inspirational" in EMOTION_VOICE_PARAMS
        assert "calm" in EMOTION_VOICE_PARAMS

    def test_neutral_params(self):
        params = EMOTION_VOICE_PARAMS["neutral"]
        assert params["pitch_modifier"] == 1.0
        assert params["speed_modifier"] == 1.0
        assert params["pause_modifier"] == 1.0

    def test_dramatic_high_energy(self):
        params = EMOTION_VOICE_PARAMS["dramatic"]
        assert params["energy_level"] == "high"
        assert params["pitch_modifier"] > 1.0

    def test_calm_low_energy(self):
        params = EMOTION_VOICE_PARAMS["calm"]
        assert params["energy_level"] == "low"
        assert params["speed_modifier"] < 1.0


class TestEmotionAnalyzer:
    def setup_method(self):
        self.analyzer = EmotionAnalyzer()

    def test_neutral_text(self):
        profile = self.analyzer.analyze_segment("The weather is nice today.")
        assert isinstance(profile, EmotionProfile)
        assert profile.emotion == "neutral"

    def test_dramatic_text(self):
        profile = self.analyzer.analyze_segment(
            "The devastating war brought death and destruction. "
            "A catastrophe of unprecedented scale."
        )
        assert profile.emotion == "dramatic"
        assert profile.confidence > 0.5

    def test_suspense_text(self):
        profile = self.analyzer.analyze_segment(
            "The mystery of the hidden secret remains unknown. "
            "Strange clues lurking in the shadow."
        )
        assert profile.emotion == "suspense"

    def test_inspirational_text(self):
        profile = self.analyzer.analyze_segment(
            "A remarkable triumph of courage and hope. "
            "An extraordinary hero who achieved the impossible dream."
        )
        assert profile.emotion == "inspirational"

    def test_calm_text(self):
        profile = self.analyzer.analyze_segment(
            "The serene ocean breeze flows gently across the meadow. "
            "A tranquil sunset over the peaceful forest."
        )
        assert profile.emotion == "calm"

    def test_empty_text(self):
        profile = self.analyzer.analyze_segment("")
        assert profile.emotion == "neutral"
        assert profile.confidence == 1.0

    def test_analyze_multiple_segments(self):
        segments = [
            {"text": "War and destruction everywhere."},
            {"text": "A peaceful meadow at sunset."},
            {"text": "Hope and triumph for the brave."},
        ]
        profiles = self.analyzer.analyze_segments(segments)
        assert len(profiles) == 3
        assert all(isinstance(p, EmotionProfile) for p in profiles)

    def test_smooth_transitions(self):
        """Low-confidence outlier should be smoothed to match neighbors."""
        profiles = [
            EmotionProfile("calm", 0.9, 0.9, 0.9, 1.3, "low"),
            EmotionProfile("dramatic", 0.3, 1.15, 1.1, 0.7, "high"),
            EmotionProfile("calm", 0.8, 0.9, 0.9, 1.3, "low"),
        ]
        smoothed = self.analyzer._smooth_emotion_transitions(profiles)
        assert smoothed[1].emotion == "calm"

    def test_get_tts_parameters(self):
        profile = EmotionProfile(
            emotion="dramatic",
            confidence=0.8,
            pitch_modifier=1.15,
            speed_modifier=1.1,
            pause_modifier=0.7,
            energy_level="high",
        )
        params = self.analyzer.get_tts_parameters(profile, base_speed=1.0, base_pitch=1.0)
        assert "speed" in params
        assert "pitch" in params
        assert "pause_factor" in params
        assert "emotion" in params
        assert params["emotion"] == "dramatic"

    def test_nlp_disabled(self):
        config = EmotionConfig(use_nlp=False, use_prosody=False)
        analyzer = EmotionAnalyzer(config)
        profile = analyzer.analyze_segment("War and destruction everywhere.")
        assert profile.emotion == "neutral"
