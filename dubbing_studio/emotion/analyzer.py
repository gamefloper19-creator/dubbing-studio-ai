"""
Emotion-aware narration analysis.

Analyzes emotional tone of each segment using NLP and prosody analysis
to modify voice style, pitch, speaking speed, and pause length.
"""

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dubbing_studio.config import EmotionConfig

logger = logging.getLogger(__name__)


# Keyword dictionaries for NLP-based emotion detection
EMOTION_KEYWORDS: dict[str, list[str]] = {
    "dramatic": [
        "war", "battle", "death", "tragedy", "catastrophe", "devastating",
        "destroyed", "collapse", "crisis", "conflict", "explosion", "terror",
        "horrifying", "shocking", "unprecedented", "revolution", "dramatic",
        "critical", "deadly", "fatal", "massive", "violent", "fierce",
    ],
    "suspense": [
        "mystery", "unknown", "hidden", "secret", "disappear", "vanish",
        "strange", "unexplained", "eerie", "haunting", "lurking", "shadow",
        "danger", "threat", "risk", "uncertain", "puzzle", "clue", "suspect",
        "investigation", "discover", "reveal", "uncover", "tension",
    ],
    "inspirational": [
        "hope", "dream", "achieve", "triumph", "victory", "overcome",
        "inspire", "courage", "brave", "hero", "remarkable", "extraordinary",
        "amazing", "wonderful", "beautiful", "transform", "miracle",
        "breakthrough", "success", "legacy", "pioneer", "innovate",
    ],
    "calm": [
        "peace", "quiet", "gentle", "serene", "tranquil", "harmony",
        "nature", "landscape", "flowing", "meadow", "sunset", "dawn",
        "forest", "ocean", "breeze", "whisper", "soothing", "stillness",
        "ancient", "timeless", "slowly", "gradually", "delicate",
    ],
}


@dataclass
class EmotionProfile:
    """Emotion analysis result for a segment."""
    emotion: str  # neutral, dramatic, suspense, inspirational, calm
    confidence: float  # 0.0 to 1.0
    pitch_modifier: float  # multiplier (1.0 = no change)
    speed_modifier: float  # multiplier (1.0 = no change)
    pause_modifier: float  # multiplier for pause length (1.0 = no change)
    energy_level: str  # low, medium, high
    details: dict = field(default_factory=dict)


# Emotion-to-voice parameter mappings
EMOTION_VOICE_PARAMS: dict[str, dict] = {
    "neutral": {
        "pitch_modifier": 1.0,
        "speed_modifier": 1.0,
        "pause_modifier": 1.0,
        "energy_level": "medium",
    },
    "dramatic": {
        "pitch_modifier": 1.15,
        "speed_modifier": 1.1,
        "pause_modifier": 0.7,
        "energy_level": "high",
    },
    "suspense": {
        "pitch_modifier": 0.95,
        "speed_modifier": 0.85,
        "pause_modifier": 1.5,
        "energy_level": "medium",
    },
    "inspirational": {
        "pitch_modifier": 1.1,
        "speed_modifier": 0.95,
        "pause_modifier": 1.2,
        "energy_level": "high",
    },
    "calm": {
        "pitch_modifier": 0.9,
        "speed_modifier": 0.9,
        "pause_modifier": 1.3,
        "energy_level": "low",
    },
}


class EmotionAnalyzer:
    """
    Analyze emotional tone of narration segments.

    Uses NLP keyword analysis and audio prosody features
    to classify emotion and generate voice modification parameters.
    """

    def __init__(self, config: Optional[EmotionConfig] = None):
        self.config = config or EmotionConfig()

    def analyze_segment(
        self,
        text: str,
        audio_path: Optional[str] = None,
    ) -> EmotionProfile:
        """
        Analyze emotion of a single segment.

        Args:
            text: Segment text content.
            audio_path: Optional path to segment audio for prosody analysis.

        Returns:
            EmotionProfile with detected emotion and voice modifiers.
        """
        scores: dict[str, float] = {e: 0.0 for e in EMOTION_KEYWORDS}

        # NLP keyword-based analysis
        if self.config.use_nlp and text:
            nlp_scores = self._analyze_nlp(text)
            for emotion, score in nlp_scores.items():
                scores[emotion] = scores.get(emotion, 0.0) + score

        # Prosody-based analysis from audio
        if self.config.use_prosody and audio_path and Path(audio_path).exists():
            prosody_scores = self._analyze_prosody(audio_path)
            for emotion, score in prosody_scores.items():
                scores[emotion] = scores.get(emotion, 0.0) + score * 0.5

        # Determine dominant emotion
        if not any(v > 0 for v in scores.values()):
            dominant_emotion = "neutral"
            confidence = 1.0
        else:
            dominant_emotion = max(scores, key=lambda k: scores[k])
            total = sum(scores.values())
            confidence = scores[dominant_emotion] / total if total > 0 else 0.5

        # Get voice parameters for the detected emotion
        params = EMOTION_VOICE_PARAMS.get(dominant_emotion, EMOTION_VOICE_PARAMS["neutral"])

        # Scale modifiers by confidence and config ranges
        pitch_mod = 1.0 + (params["pitch_modifier"] - 1.0) * confidence * self.config.pitch_range / 0.3
        speed_mod = 1.0 + (params["speed_modifier"] - 1.0) * confidence * self.config.speed_range / 0.2
        pause_mod = 1.0 + (params["pause_modifier"] - 1.0) * confidence * self.config.pause_multiplier_range / 0.5

        profile = EmotionProfile(
            emotion=dominant_emotion,
            confidence=confidence,
            pitch_modifier=round(pitch_mod, 3),
            speed_modifier=round(speed_mod, 3),
            pause_modifier=round(pause_mod, 3),
            energy_level=params["energy_level"],
            details={"scores": scores},
        )

        logger.debug(
            "Emotion detected: %s (%.1f%%) -> pitch=%.2f, speed=%.2f, pause=%.2f",
            profile.emotion, profile.confidence * 100,
            profile.pitch_modifier, profile.speed_modifier, profile.pause_modifier,
        )

        return profile

    def analyze_segments(
        self,
        segments: list[dict],
    ) -> list[EmotionProfile]:
        """
        Analyze emotion for multiple segments.

        Args:
            segments: List of dicts with 'text' and optional 'audio_path'.

        Returns:
            List of EmotionProfile for each segment.
        """
        profiles = []
        for seg in segments:
            profile = self.analyze_segment(
                text=seg.get("text", ""),
                audio_path=seg.get("audio_path"),
            )
            profiles.append(profile)

        # Smooth transitions between emotions
        profiles = self._smooth_emotion_transitions(profiles)

        return profiles

    def _analyze_nlp(self, text: str) -> dict[str, float]:
        """Analyze text using keyword matching for emotion detection."""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        word_count = len(words) if words else 1

        scores: dict[str, float] = {}

        for emotion, keywords in EMOTION_KEYWORDS.items():
            matches = sum(1 for word in words if word in keywords)
            # Normalize by word count to avoid bias toward longer segments
            scores[emotion] = matches / word_count * 10  # scale up

        return scores

    def _analyze_prosody(self, audio_path: str) -> dict[str, float]:
        """
        Analyze audio prosody features for emotion detection.

        Uses FFmpeg audio statistics as proxy for prosody analysis.
        """
        scores: dict[str, float] = {}

        try:
            # Get RMS energy and volume statistics
            cmd = [
                "ffmpeg",
                "-i", audio_path,
                "-af", "volumedetect",
                "-f", "null",
                "-",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            stderr = result.stderr

            mean_vol = -25.0
            max_vol = -10.0

            for line in stderr.split("\n"):
                if "mean_volume" in line:
                    try:
                        mean_vol = float(line.split(":")[-1].strip().replace(" dB", ""))
                    except (ValueError, IndexError):
                        pass
                elif "max_volume" in line:
                    try:
                        max_vol = float(line.split(":")[-1].strip().replace(" dB", ""))
                    except (ValueError, IndexError):
                        pass

            # Dynamic range indicates emotion intensity
            dynamic_range = max_vol - mean_vol

            # High energy + high dynamic range -> dramatic
            if mean_vol > -18 and dynamic_range > 15:
                scores["dramatic"] = 0.8
            elif mean_vol > -20:
                scores["inspirational"] = 0.5

            # Low energy -> calm
            if mean_vol < -30:
                scores["calm"] = 0.7
            elif mean_vol < -28:
                scores["suspense"] = 0.4

            # Medium dynamic range with moderate volume -> neutral
            if 10 < dynamic_range < 15 and -28 < mean_vol < -20:
                scores["neutral"] = 0.3

        except Exception as e:
            logger.debug("Prosody analysis failed: %s", e)

        return scores

    def _smooth_emotion_transitions(
        self,
        profiles: list[EmotionProfile],
    ) -> list[EmotionProfile]:
        """
        Smooth emotion transitions to avoid abrupt changes.

        Uses a simple windowed approach to prevent jarring emotion switches.
        """
        if len(profiles) <= 2:
            return profiles

        smoothed = list(profiles)

        for i in range(1, len(profiles) - 1):
            prev_emotion = profiles[i - 1].emotion
            curr_emotion = profiles[i].emotion
            next_emotion = profiles[i + 1].emotion

            # If current emotion is different from both neighbors,
            # and confidence is low, align with the dominant neighbor
            if (curr_emotion != prev_emotion and
                    curr_emotion != next_emotion and
                    profiles[i].confidence < 0.6):
                # Use the neighbor emotion with higher confidence
                if profiles[i - 1].confidence >= profiles[i + 1].confidence:
                    dominant = prev_emotion
                else:
                    dominant = next_emotion

                params = EMOTION_VOICE_PARAMS.get(dominant, EMOTION_VOICE_PARAMS["neutral"])
                smoothed[i] = EmotionProfile(
                    emotion=dominant,
                    confidence=profiles[i].confidence * 0.8,
                    pitch_modifier=params["pitch_modifier"],
                    speed_modifier=params["speed_modifier"],
                    pause_modifier=params["pause_modifier"],
                    energy_level=params["energy_level"],
                    details=profiles[i].details,
                )

        return smoothed

    def get_tts_parameters(
        self,
        emotion_profile: EmotionProfile,
        base_speed: float = 1.0,
        base_pitch: float = 1.0,
    ) -> dict:
        """
        Get TTS engine parameters adjusted for the detected emotion.

        Args:
            emotion_profile: Detected emotion profile.
            base_speed: Base speaking speed.
            base_pitch: Base pitch level.

        Returns:
            Dict with adjusted TTS parameters.
        """
        return {
            "speed": base_speed * emotion_profile.speed_modifier,
            "pitch": base_pitch * emotion_profile.pitch_modifier,
            "pause_factor": emotion_profile.pause_modifier,
            "emotion": emotion_profile.emotion,
            "energy": emotion_profile.energy_level,
        }
