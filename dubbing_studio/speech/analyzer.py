"""
Narration style analysis - detect narrator characteristics.
"""

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dubbing_studio.config import AudioConfig

logger = logging.getLogger(__name__)


@dataclass
class NarrationStyle:
    """Detected narration style characteristics."""
    gender: str  # male, female, unknown
    tone: str  # formal, casual, dramatic, calm
    pacing: str  # slow, medium, fast
    pitch_level: str  # low, medium, high
    energy: str  # low, medium, high
    speaking_rate_wpm: float  # words per minute


class NarrationAnalyzer:
    """Analyze original narration to determine voice characteristics."""

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()

    def analyze_narration(
        self,
        audio_path: str,
        transcription_text: str = "",
        total_duration: float = 0.0,
    ) -> NarrationStyle:
        """
        Analyze narration style from audio.

        Uses audio analysis for pitch/energy and text analysis for pacing.

        Args:
            audio_path: Path to audio file.
            transcription_text: Full transcription text for WPM calculation.
            total_duration: Total duration of speech in seconds.

        Returns:
            NarrationStyle with detected characteristics.
        """
        # Analyze pitch using FFmpeg
        pitch_info = self._analyze_pitch(audio_path)

        # Calculate speaking rate
        speaking_rate = self._calculate_speaking_rate(
            transcription_text, total_duration
        )

        # Determine gender from pitch
        gender = self._estimate_gender(pitch_info)

        # Determine tone from energy and pitch variation
        tone = self._estimate_tone(pitch_info)

        # Determine pacing from WPM
        pacing = self._estimate_pacing(speaking_rate)

        style = NarrationStyle(
            gender=gender,
            tone=tone,
            pacing=pacing,
            pitch_level=pitch_info.get("level", "medium"),
            energy=pitch_info.get("energy", "medium"),
            speaking_rate_wpm=speaking_rate,
        )

        logger.info(
            "Narration analysis: gender=%s, tone=%s, pacing=%s, WPM=%.0f",
            style.gender, style.tone, style.pacing, style.speaking_rate_wpm,
        )

        return style

    def _analyze_pitch(self, audio_path: str) -> dict:
        """Analyze pitch characteristics using FFmpeg."""
        # Use FFmpeg's astats filter for basic audio statistics
        cmd = [
            "ffmpeg",
            "-i", audio_path,
            "-af", "astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level",
            "-f", "null",
            "-",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        stderr = result.stderr

        # Parse RMS levels
        rms_values = []
        for line in stderr.split("\n"):
            if "RMS_level" in line:
                try:
                    val = float(line.split("=")[-1].strip())
                    if val != float("-inf"):
                        rms_values.append(val)
                except (ValueError, IndexError):
                    continue

        if rms_values:
            avg_rms = sum(rms_values) / len(rms_values)
            energy = "high" if avg_rms > -20 else "low" if avg_rms < -35 else "medium"
        else:
            avg_rms = -25.0
            energy = "medium"

        # Estimate pitch range using spectral analysis
        pitch_info = self._estimate_pitch_range(audio_path)

        return {
            "avg_rms": avg_rms,
            "energy": energy,
            "level": pitch_info.get("level", "medium"),
            "estimated_f0": pitch_info.get("f0", 150.0),
        }

    def _estimate_pitch_range(self, audio_path: str) -> dict:
        """
        Estimate fundamental frequency range.
        Uses FFmpeg's showfreqs or volumedetect as proxy.
        """
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

        # Heuristic: louder voices tend to be lower pitched in documentaries
        # This is a simplified estimation
        if mean_vol > -20:
            return {"level": "low", "f0": 120.0}
        elif mean_vol < -30:
            return {"level": "high", "f0": 220.0}
        else:
            return {"level": "medium", "f0": 160.0}

    def _estimate_gender(self, pitch_info: dict) -> str:
        """Estimate narrator gender from pitch analysis."""
        f0 = pitch_info.get("estimated_f0", 150.0)

        if f0 < 165:
            return "male"
        elif f0 > 200:
            return "female"
        else:
            return "male"  # default to male for documentary style

    def _estimate_tone(self, pitch_info: dict) -> str:
        """Estimate narration tone from audio characteristics."""
        energy = pitch_info.get("energy", "medium")

        if energy == "high":
            return "dramatic"
        elif energy == "low":
            return "calm"
        else:
            return "formal"

    def _calculate_speaking_rate(
        self,
        text: str,
        duration: float,
    ) -> float:
        """Calculate words per minute from text and duration."""
        if not text or duration <= 0:
            return 150.0  # default WPM for narration

        word_count = len(text.split())
        minutes = duration / 60.0

        if minutes <= 0:
            return 150.0

        return word_count / minutes

    def _estimate_pacing(self, wpm: float) -> str:
        """Estimate pacing from words per minute."""
        if wpm < 120:
            return "slow"
        elif wpm > 170:
            return "fast"
        else:
            return "medium"
