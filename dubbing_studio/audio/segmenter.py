"""
Audio segmentation using silence detection.
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
class AudioSegment:
    """Represents a segment of audio."""
    segment_id: str
    start_time: float  # seconds
    end_time: float  # seconds
    duration: float  # seconds
    file_path: str = ""
    text: str = ""


class AudioSegmenter:
    """Segment audio based on silence detection."""

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()

    def detect_silence(self, audio_path: str) -> list[dict]:
        """
        Detect silence periods in audio file.

        Args:
            audio_path: Path to audio file.

        Returns:
            List of silence periods with start/end times.
        """
        cmd = [
            "ffmpeg",
            "-i", audio_path,
            "-af", (
                f"silencedetect=noise={self.config.silence_threshold}dB"
                f":d={self.config.min_silence_duration}"
            ),
            "-f", "null",
            "-",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        stderr = result.stderr

        silences = []
        silence_start = None

        for line in stderr.split("\n"):
            if "silence_start:" in line:
                try:
                    parts = line.split("silence_start:")
                    silence_start = float(parts[1].strip().split()[0])
                except (IndexError, ValueError):
                    continue
            elif "silence_end:" in line and silence_start is not None:
                try:
                    parts = line.split("silence_end:")
                    end_parts = parts[1].strip().split()
                    silence_end = float(end_parts[0])
                    silences.append({
                        "start": silence_start,
                        "end": silence_end,
                        "duration": silence_end - silence_start,
                    })
                    silence_start = None
                except (IndexError, ValueError):
                    continue

        logger.info("Detected %d silence periods in %s", len(silences), audio_path)
        return silences

    def segment_audio(
        self,
        audio_path: str,
        output_dir: str,
    ) -> list[AudioSegment]:
        """
        Segment audio into chunks based on silence detection.

        Segments are between 5-15 seconds as specified in requirements.

        Args:
            audio_path: Path to input audio file.
            output_dir: Directory for output segments.

        Returns:
            List of AudioSegment objects.
        """
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        # Get total duration
        duration = self._get_duration(audio_path)
        logger.info("Total audio duration: %.2f seconds", duration)

        # Detect silence points
        silences = self.detect_silence(audio_path)

        # Build segments from silence boundaries
        segments = self._build_segments(silences, duration)

        # Merge short segments and split long ones
        segments = self._optimize_segments(segments)

        # Extract each segment to file
        result_segments = []
        for i, seg in enumerate(segments):
            segment_id = f"{i + 1:03d}"
            output_path = str(output_dir_path / f"segment_{segment_id}.wav")

            self._extract_segment(
                audio_path,
                output_path,
                seg["start"],
                seg["end"],
            )

            result_segments.append(AudioSegment(
                segment_id=segment_id,
                start_time=seg["start"],
                end_time=seg["end"],
                duration=seg["end"] - seg["start"],
                file_path=output_path,
            ))

        logger.info("Created %d segments from %s", len(result_segments), audio_path)
        return result_segments

    def _build_segments(
        self,
        silences: list[dict],
        total_duration: float,
    ) -> list[dict]:
        """Build speech segments from silence boundaries."""
        if not silences:
            # No silence detected, return whole audio as one segment
            return [{"start": 0.0, "end": total_duration}]

        segments = []
        current_start = 0.0

        for silence in silences:
            # Create segment from current_start to silence_start
            seg_end = silence["start"]
            if seg_end - current_start > 0.1:  # skip tiny segments
                segments.append({
                    "start": current_start,
                    "end": seg_end,
                })
            current_start = silence["end"]

        # Add final segment
        if total_duration - current_start > 0.1:
            segments.append({
                "start": current_start,
                "end": total_duration,
            })

        return segments

    def _optimize_segments(self, segments: list[dict]) -> list[dict]:
        """
        Optimize segment lengths:
        - Merge segments shorter than min_duration
        - Split segments longer than max_duration
        """
        if not segments:
            return segments

        min_dur = self.config.segment_min_duration
        max_dur = self.config.segment_max_duration

        # First pass: merge short segments
        merged = []
        current = segments[0].copy()

        for seg in segments[1:]:
            current_duration = current["end"] - current["start"]
            seg_duration = seg["end"] - seg["start"]

            if current_duration < min_dur:
                # Merge with next segment
                current["end"] = seg["end"]
            elif seg_duration < min_dur:
                # Merge short next segment into current
                current["end"] = seg["end"]
            else:
                merged.append(current)
                current = seg.copy()

        merged.append(current)

        # Second pass: split long segments
        final = []
        for seg in merged:
            duration = seg["end"] - seg["start"]
            if duration > max_dur:
                # Split into roughly equal parts
                n_parts = int(duration / max_dur) + 1
                part_duration = duration / n_parts
                for i in range(n_parts):
                    final.append({
                        "start": seg["start"] + i * part_duration,
                        "end": seg["start"] + (i + 1) * part_duration,
                    })
            else:
                final.append(seg)

        return final

    def _extract_segment(
        self,
        audio_path: str,
        output_path: str,
        start: float,
        end: float,
    ) -> None:
        """Extract a segment from audio file."""
        duration = end - start
        cmd = [
            "ffmpeg", "-y",
            "-i", audio_path,
            "-ss", f"{start:.3f}",
            "-t", f"{duration:.3f}",
            "-acodec", "pcm_s16le",
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Segment extraction failed [{start:.2f}-{end:.2f}]: "
                f"{result.stderr}"
            )

    def _get_duration(self, audio_path: str) -> float:
        """Get duration of audio file."""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            audio_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFprobe failed: {result.stderr}")

        info = json.loads(result.stdout)
        return float(info["format"]["duration"])
