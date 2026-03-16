"""
Audio cleaning and noise reduction using FFmpeg filters.
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional

from dubbing_studio.config import AudioConfig

logger = logging.getLogger(__name__)


class AudioCleaner:
    """Clean and normalize audio for better speech recognition."""

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()

    def clean_audio(
        self,
        input_path: str,
        output_path: str,
    ) -> str:
        """
        Apply full audio cleaning pipeline:
        1. Normalize volume levels
        2. Remove background noise
        3. Apply high-pass filter for speech clarity

        Args:
            input_path: Path to input audio file.
            output_path: Path for cleaned audio output.

        Returns:
            Path to the cleaned audio file.
        """
        input_path = str(Path(input_path).resolve())
        output_path = str(Path(output_path).resolve())
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Build filter chain
        filters = self._build_filter_chain()

        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-af", filters,
            "-ar", str(self.config.sample_rate),
            "-ac", str(self.config.channels),
            output_path,
        ]

        logger.info("Cleaning audio: %s", input_path)
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.warning(
                "Full cleaning failed, trying basic normalization: %s",
                result.stderr,
            )
            return self._basic_normalize(input_path, output_path)

        logger.info("Audio cleaned: %s", output_path)
        return output_path

    def _build_filter_chain(self) -> str:
        """Build FFmpeg audio filter chain for cleaning."""
        filters = []

        # High-pass filter to remove low-frequency rumble
        filters.append("highpass=f=80")

        # Low-pass filter to remove high-frequency noise
        filters.append("lowpass=f=12000")

        # Noise reduction using FFmpeg's afftdn filter
        filters.append(
            f"afftdn=nf={int(self.config.noise_reduction_strength * -80)}"
        )

        # Volume normalization using loudnorm (EBU R128)
        filters.append(
            f"loudnorm=I={self.config.normalize_loudness}:TP=-1.5:LRA=11"
        )

        return ",".join(filters)

    def _basic_normalize(
        self,
        input_path: str,
        output_path: str,
    ) -> str:
        """Fallback basic volume normalization."""
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-af", f"loudnorm=I={self.config.normalize_loudness}:TP=-1.5:LRA=11",
            "-ar", str(self.config.sample_rate),
            "-ac", str(self.config.channels),
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Basic audio normalization failed: {result.stderr}"
            )

        return output_path

    def normalize_volume(
        self,
        input_path: str,
        output_path: str,
        target_lufs: Optional[float] = None,
    ) -> str:
        """
        Normalize audio volume to target LUFS level.

        Args:
            input_path: Path to input audio.
            output_path: Path for normalized audio.
            target_lufs: Target loudness in LUFS.

        Returns:
            Path to normalized audio file.
        """
        lufs = target_lufs or self.config.normalize_loudness

        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-af", f"loudnorm=I={lufs}:TP=-1.5:LRA=11",
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Volume normalization failed: {result.stderr}")

        return output_path

    def remove_noise(
        self,
        input_path: str,
        output_path: str,
        strength: Optional[float] = None,
    ) -> str:
        """
        Apply noise reduction to audio.

        Args:
            input_path: Path to input audio.
            output_path: Path for denoised audio.
            strength: Noise reduction strength (0.0 to 1.0).

        Returns:
            Path to denoised audio file.
        """
        s = strength or self.config.noise_reduction_strength
        noise_floor = int(s * -80)

        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-af", f"afftdn=nf={noise_floor}",
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Noise reduction failed: {result.stderr}")

        return output_path
