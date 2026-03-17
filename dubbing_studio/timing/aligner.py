"""
Speech timing alignment engine.

Ensures generated speech matches original segment duration
within 300ms tolerance.
"""

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dubbing_studio.config import TimingConfig

logger = logging.getLogger(__name__)


@dataclass
class TimingAdjustment:
    """Result of timing alignment."""
    original_duration: float
    generated_duration: float
    adjusted_duration: float
    deviation_ms: float
    speed_factor: float
    method: str  # speed, trim, pad, compress


class TimingAligner:
    """
    Align generated speech timing to match original segment duration.

    Maximum allowed deviation: 300ms.

    Methods:
    1. Speed modification (atempo filter)
    2. Pause insertion (add silence)
    3. Silence trimming (remove trailing silence)
    4. Sentence compression (re-generate with adjusted speed)
    """

    def __init__(self, config: Optional[TimingConfig] = None):
        self.config = config or TimingConfig()

    def align_timing(
        self,
        audio_path: str,
        target_duration: float,
        output_path: str,
    ) -> TimingAdjustment:
        """
        Align audio duration to target duration.

        Args:
            audio_path: Path to generated TTS audio.
            target_duration: Target duration in seconds.
            output_path: Path for aligned audio output.

        Returns:
            TimingAdjustment with details of the adjustment.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        current_duration = self._get_duration(audio_path)
        deviation_ms = abs(current_duration - target_duration) * 1000

        logger.info(
            "Timing alignment: current=%.2fs, target=%.2fs, deviation=%.0fms",
            current_duration, target_duration, deviation_ms,
        )

        # If already within tolerance, just copy
        if deviation_ms <= self.config.max_deviation_ms:
            self._copy_audio(audio_path, output_path)
            return TimingAdjustment(
                original_duration=current_duration,
                generated_duration=current_duration,
                adjusted_duration=current_duration,
                deviation_ms=deviation_ms,
                speed_factor=1.0,
                method="none",
            )

        # Try methods in order of preference
        if current_duration > target_duration:
            # Speech is too long - need to speed up or trim
            return self._shorten_audio(
                audio_path, current_duration, target_duration, output_path
            )
        else:
            # Speech is too short - need to slow down or add pauses
            return self._lengthen_audio(
                audio_path, current_duration, target_duration, output_path
            )

    def _shorten_audio(
        self,
        audio_path: str,
        current_duration: float,
        target_duration: float,
        output_path: str,
    ) -> TimingAdjustment:
        """Shorten audio to match target duration."""
        speed_factor = current_duration / target_duration

        # Method 1: Try silence trimming first
        if self.config.allow_silence_trimming:
            trimmed_path = output_path + ".trimmed.wav"
            trimmed_duration = self._trim_silence(audio_path, trimmed_path)

            if trimmed_duration > 0:
                new_deviation = abs(trimmed_duration - target_duration) * 1000
                if new_deviation <= self.config.max_deviation_ms:
                    self._copy_audio(trimmed_path, output_path)
                    self._cleanup(trimmed_path)
                    return TimingAdjustment(
                        original_duration=current_duration,
                        generated_duration=current_duration,
                        adjusted_duration=trimmed_duration,
                        deviation_ms=new_deviation,
                        speed_factor=1.0,
                        method="trim",
                    )
                # Use trimmed as base for speed adjustment
                audio_path = trimmed_path
                current_duration = trimmed_duration
                speed_factor = current_duration / target_duration

        # Method 2: Speed adjustment
        if self.config.speed_min <= speed_factor <= self.config.speed_max:
            self._adjust_speed(audio_path, output_path, speed_factor)
            adjusted_duration = self._get_duration(output_path)
            deviation_ms = abs(adjusted_duration - target_duration) * 1000

            self._cleanup(audio_path + ".trimmed.wav")

            return TimingAdjustment(
                original_duration=current_duration,
                generated_duration=current_duration,
                adjusted_duration=adjusted_duration,
                deviation_ms=deviation_ms,
                speed_factor=speed_factor,
                method="speed",
            )

        # Method 3: Clamp speed even if outside preferred range
        clamped_speed = max(self.config.speed_min, min(self.config.speed_max, speed_factor))
        self._adjust_speed(audio_path, output_path, clamped_speed)
        adjusted_duration = self._get_duration(output_path)
        deviation_ms = abs(adjusted_duration - target_duration) * 1000

        self._cleanup(audio_path + ".trimmed.wav")

        if deviation_ms <= self.config.max_deviation_ms or not self.config.allow_sentence_compression:
            return TimingAdjustment(
                original_duration=current_duration,
                generated_duration=current_duration,
                adjusted_duration=adjusted_duration,
                deviation_ms=deviation_ms,
                speed_factor=clamped_speed,
                method="speed_clamped",
            )

        # Method 4: Sentence compression (force exact speed)
        self._adjust_speed(audio_path, output_path, speed_factor)
        adjusted_duration = self._get_duration(output_path)
        deviation_ms = abs(adjusted_duration - target_duration) * 1000

        return TimingAdjustment(
            original_duration=current_duration,
            generated_duration=current_duration,
            adjusted_duration=adjusted_duration,
            deviation_ms=deviation_ms,
            speed_factor=speed_factor,
            method="speed_forced",
        )

    def _lengthen_audio(
        self,
        audio_path: str,
        current_duration: float,
        target_duration: float,
        output_path: str,
    ) -> TimingAdjustment:
        """Lengthen audio to match target duration."""
        speed_factor = current_duration / target_duration  # < 1.0

        # Method 1: Slow down speech
        if speed_factor >= self.config.speed_min:
            self._adjust_speed(audio_path, output_path, speed_factor)
            adjusted_duration = self._get_duration(output_path)
            deviation_ms = abs(adjusted_duration - target_duration) * 1000

            return TimingAdjustment(
                original_duration=current_duration,
                generated_duration=current_duration,
                adjusted_duration=adjusted_duration,
                deviation_ms=deviation_ms,
                speed_factor=speed_factor,
                method="speed",
            )

        # Method 2: Sentence expansion (force exact speed)
        if self.config.allow_sentence_compression:
            self._adjust_speed(audio_path, output_path, speed_factor)
            adjusted_duration = self._get_duration(output_path)
            deviation_ms = abs(adjusted_duration - target_duration) * 1000

            return TimingAdjustment(
                original_duration=current_duration,
                generated_duration=current_duration,
                adjusted_duration=adjusted_duration,
                deviation_ms=deviation_ms,
                speed_factor=speed_factor,
                method="speed_forced",
            )

        # Method 3: Add pause at the end
        if self.config.allow_pause_insertion:
            pause_duration = target_duration - current_duration
            self._add_pause(audio_path, output_path, pause_duration)
            adjusted_duration = self._get_duration(output_path)
            deviation_ms = abs(adjusted_duration - target_duration) * 1000

            return TimingAdjustment(
                original_duration=current_duration,
                generated_duration=current_duration,
                adjusted_duration=adjusted_duration,
                deviation_ms=deviation_ms,
                speed_factor=1.0,
                method="pad",
            )

        # Method 4: Slow down as much as allowed and pad the rest
        clamped_speed = self.config.speed_min
        temp_path = output_path + ".temp.wav"
        self._adjust_speed(audio_path, temp_path, clamped_speed)

        slowed_duration = self._get_duration(temp_path)
        remaining_gap = target_duration - slowed_duration

        if remaining_gap > 0 and self.config.allow_pause_insertion:
            self._add_pause(temp_path, output_path, remaining_gap)
        else:
            self._copy_audio(temp_path, output_path)

        self._cleanup(temp_path)

        adjusted_duration = self._get_duration(output_path)
        deviation_ms = abs(adjusted_duration - target_duration) * 1000

        return TimingAdjustment(
            original_duration=current_duration,
            generated_duration=current_duration,
            adjusted_duration=adjusted_duration,
            deviation_ms=deviation_ms,
            speed_factor=clamped_speed,
            method="speed_pad",
        )

    def _adjust_speed(
        self,
        input_path: str,
        output_path: str,
        speed_factor: float,
    ) -> None:
        """Adjust audio speed using FFmpeg atempo filter."""
        # Build atempo chain (each filter supports 0.5-100.0)
        atempo_chain = self._build_atempo_chain(speed_factor)

        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-af", atempo_chain,
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Speed adjustment failed: {result.stderr}")

    def _build_atempo_chain(self, speed: float) -> str:
        """Build chain of atempo filters for extreme speed changes."""
        filters = []
        remaining = speed

        while remaining > 100.0:
            filters.append("atempo=100.0")
            remaining /= 100.0

        while remaining < 0.5:
            filters.append("atempo=0.5")
            remaining /= 0.5

        filters.append(f"atempo={remaining:.4f}")
        return ",".join(filters)

    def _trim_silence(self, input_path: str, output_path: str) -> float:
        """Trim trailing silence from audio."""
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-af", "silenceremove=stop_periods=-1:stop_duration=0.1:stop_threshold=-40dB",
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return 0.0

        return self._get_duration(output_path)

    def _add_pause(
        self,
        input_path: str,
        output_path: str,
        pause_duration: float,
    ) -> None:
        """Add silence at the end of audio."""
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-af", f"apad=pad_dur={pause_duration:.3f}",
            "-t", str(self._get_duration(input_path) + pause_duration),
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Pause insertion failed: {result.stderr}")

    def _get_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds."""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", audio_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return 0.0

        info = json.loads(result.stdout)
        return float(info.get("format", {}).get("duration", 0))

    def _copy_audio(self, input_path: str, output_path: str) -> None:
        """Copy audio file."""
        cmd = ["ffmpeg", "-y", "-i", input_path, "-c", "copy", output_path]
        subprocess.run(cmd, capture_output=True, text=True, check=True)

    def _cleanup(self, path: str) -> None:
        """Remove temporary file if it exists."""
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass

    def batch_align(
        self,
        segments: list[dict],
    ) -> list[TimingAdjustment]:
        """
        Align timing for multiple segments.

        Args:
            segments: List of dicts with 'audio_path', 'target_duration', 'output_path'.

        Returns:
            List of TimingAdjustment results.
        """
        results = []
        for seg in segments:
            result = self.align_timing(
                audio_path=seg["audio_path"],
                target_duration=seg["target_duration"],
                output_path=seg["output_path"],
            )
            results.append(result)

            logger.info(
                "Segment aligned: deviation=%.0fms, method=%s",
                result.deviation_ms, result.method,
            )

        return results
