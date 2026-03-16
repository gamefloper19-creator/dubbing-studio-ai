"""
Audio mixing for combining narration with background audio.
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional

from dubbing_studio.config import MixingConfig

logger = logging.getLogger(__name__)


class AudioMixer:
    """Mix narration audio with background audio."""

    def __init__(self, config: Optional[MixingConfig] = None):
        self.config = config or MixingConfig()

    def mix_audio(
        self,
        narration_path: str,
        background_path: str,
        output_path: str,
        narration_segments: Optional[list[dict]] = None,
    ) -> str:
        """
        Mix narration audio with background audio.

        Implements automatic ducking: background audio is reduced
        when narration is playing.

        Args:
            narration_path: Path to narration audio file.
            background_path: Path to background audio file.
            output_path: Path for mixed output.
            narration_segments: Optional list of segments with start/end times
                                for precise ducking.

        Returns:
            Path to the mixed audio file.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if self.config.ducking_enabled and narration_segments:
            return self._mix_with_ducking(
                narration_path, background_path, output_path, narration_segments
            )
        else:
            return self._simple_mix(
                narration_path, background_path, output_path
            )

    def _simple_mix(
        self,
        narration_path: str,
        background_path: str,
        output_path: str,
    ) -> str:
        """Simple volume-based mixing without ducking."""
        bg_vol = self.config.background_volume
        narr_vol = self.config.narration_volume

        cmd = [
            "ffmpeg", "-y",
            "-i", narration_path,
            "-i", background_path,
            "-filter_complex", (
                f"[0:a]volume={narr_vol}[narr];"
                f"[1:a]volume={bg_vol}[bg];"
                f"[narr][bg]amix=inputs=2:duration=longest:dropout_transition=2"
            ),
            "-ac", "2",
            output_path,
        ]

        logger.info("Mixing audio (simple mode)")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Audio mixing failed: {result.stderr}")

        return output_path

    def _mix_with_ducking(
        self,
        narration_path: str,
        background_path: str,
        output_path: str,
        segments: list[dict],
    ) -> str:
        """
        Mix with automatic ducking based on narration segments.

        Background audio is reduced when narration is active.
        """
        bg_vol = self.config.background_volume
        duck_ratio = self.config.ducking_ratio
        crossfade = self.config.crossfade_duration

        # Use sidechaincompress for automatic ducking
        cmd = [
            "ffmpeg", "-y",
            "-i", narration_path,
            "-i", background_path,
            "-filter_complex", (
                f"[1:a]volume={bg_vol}[bg];"
                f"[0:a]asplit=2[narr][sc];"
                f"[bg][sc]sidechaincompress="
                f"threshold=0.01:ratio=20:attack=50:release=300"
                f"[ducked_bg];"
                f"[narr][ducked_bg]amix=inputs=2:duration=longest"
                f":dropout_transition=2"
            ),
            "-ac", "2",
            output_path,
        ]

        logger.info("Mixing audio with ducking")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.warning("Ducking mix failed, falling back to simple mix")
            return self._simple_mix(narration_path, background_path, output_path)

        return output_path

    def adjust_audio_duration(
        self,
        input_path: str,
        output_path: str,
        target_duration: float,
    ) -> str:
        """
        Adjust audio duration to match target using tempo change.

        Args:
            input_path: Path to input audio.
            output_path: Path for adjusted audio.
            target_duration: Target duration in seconds.

        Returns:
            Path to adjusted audio file.
        """
        from dubbing_studio.audio.extractor import AudioExtractor

        extractor = AudioExtractor()
        current_duration = extractor.get_audio_duration(input_path)

        if current_duration <= 0:
            raise ValueError("Input audio has zero or negative duration")

        tempo = current_duration / target_duration

        # FFmpeg atempo filter only supports 0.5 to 100.0
        tempo = max(0.5, min(100.0, tempo))

        # Chain multiple atempo filters for extreme adjustments
        atempo_filters = self._build_atempo_chain(tempo)

        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-af", atempo_filters,
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Audio duration adjustment failed: {result.stderr}")

        return output_path

    def _build_atempo_chain(self, tempo: float) -> str:
        """
        Build chain of atempo filters for FFmpeg.
        Each atempo filter supports 0.5 to 100.0 range.
        """
        filters = []

        remaining = tempo
        while remaining > 100.0:
            filters.append("atempo=100.0")
            remaining /= 100.0
        while remaining < 0.5:
            filters.append("atempo=0.5")
            remaining /= 0.5

        filters.append(f"atempo={remaining:.4f}")
        return ",".join(filters)

    def concatenate_audio(
        self,
        audio_paths: list[str],
        output_path: str,
    ) -> str:
        """
        Concatenate multiple audio files into one.

        Args:
            audio_paths: List of audio file paths in order.
            output_path: Path for concatenated output.

        Returns:
            Path to concatenated audio file.
        """
        if not audio_paths:
            raise ValueError("No audio files to concatenate")

        if len(audio_paths) == 1:
            # Just copy the single file
            cmd = ["ffmpeg", "-y", "-i", audio_paths[0], "-c", "copy", output_path]
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return output_path

        # Build concat filter
        inputs = []
        filter_parts = []

        for i, path in enumerate(audio_paths):
            inputs.extend(["-i", path])
            filter_parts.append(f"[{i}:a]")

        filter_str = (
            "".join(filter_parts)
            + f"concat=n={len(audio_paths)}:v=0:a=1[out]"
        )

        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_str,
            "-map", "[out]",
            output_path,
        ]

        logger.info("Concatenating %d audio files", len(audio_paths))
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Audio concatenation failed: {result.stderr}")

        return output_path
