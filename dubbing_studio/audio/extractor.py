"""
Audio extraction from video files using FFmpeg.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

from dubbing_studio.config import AudioConfig

logger = logging.getLogger(__name__)


class AudioExtractor:
    """Extract and process audio from video files using FFmpeg."""

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self._verify_ffmpeg()

    def _verify_ffmpeg(self) -> None:
        """Verify FFmpeg is installed and accessible."""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                "FFmpeg is not installed or not accessible. "
                "Please install FFmpeg: https://ffmpeg.org/download.html"
            ) from e

    def extract_audio(
        self,
        video_path: str,
        output_path: str,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
    ) -> str:
        """
        Extract audio from a video file.

        Args:
            video_path: Path to the input video file.
            output_path: Path for the output audio file.
            sample_rate: Audio sample rate (default from config).
            channels: Number of audio channels (default from config).

        Returns:
            Path to the extracted audio file.
        """
        video_path = str(Path(video_path).resolve())
        output_path = str(Path(output_path).resolve())
        sr = sample_rate or self.config.sample_rate
        ch = channels or self.config.channels

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",  # no video
            "-acodec", "pcm_s16le",
            "-ar", str(sr),
            "-ac", str(ch),
            output_path,
        ]

        logger.info("Extracting audio from %s", video_path)
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg audio extraction failed: {result.stderr}"
            )

        logger.info("Audio extracted to %s", output_path)
        return output_path

    def extract_background_audio(
        self,
        video_path: str,
        output_path: str,
    ) -> str:
        """
        Extract background audio (attempt to isolate non-vocal audio).
        Uses a high-pass + low-pass filter to approximate background extraction.

        Args:
            video_path: Path to input video.
            output_path: Path for background audio output.

        Returns:
            Path to the background audio file.
        """
        video_path = str(Path(video_path).resolve())
        output_path = str(Path(output_path).resolve())

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Try Demucs for high-quality background extraction
        try:
            import torch  # Demucs requires PyTorch
            cmd_demucs = [
                "demucs", "--two-stems=vocals", "-o", str(Path(output_path).parent),
                video_path
            ]
            logger.info("Trying Demucs for background extraction...")
            result = subprocess.run(cmd_demucs, capture_output=True, text=True)
            if result.returncode == 0:
                # Demucs usually creates: <output_dir>/htdemucs/<filename>/no_vocals.wav
                video_name = Path(video_path).stem
                demucs_out = Path(output_path).parent / "htdemucs" / video_name / "no_vocals.wav"
                if demucs_out.exists():
                    import shutil
                    shutil.copy(str(demucs_out), output_path)
                    logger.info("Successfully extracted background with Demucs")
                    return output_path
        except Exception as e:
            logger.warning("Demucs extraction failed/unavailable: %s", e)

        # Use FFmpeg's afftdn for basic vocal reduction
        # This is a simple approach; for production, use a dedicated vocal remover
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-af", (
                "pan=stereo|c0=c0-c1|c1=c1-c0,"
                "lowpass=f=8000,highpass=f=100"
            ),
            "-ar", str(self.config.sample_rate),
            "-ac", "2",
            output_path,
        ]

        logger.info("Extracting background audio from %s", video_path)
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            # Fallback: just extract full audio at lower volume
            logger.warning(
                "Background extraction failed, using full audio: %s",
                result.stderr,
            )
            return self._extract_attenuated_audio(video_path, output_path)

        return output_path

    def _extract_attenuated_audio(
        self,
        video_path: str,
        output_path: str,
    ) -> str:
        """Extract audio with reduced volume as fallback background."""
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-af", "volume=0.15",
            "-ar", str(self.config.sample_rate),
            "-ac", str(self.config.channels),
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to extract attenuated audio: {result.stderr}"
            )
        return output_path

    def get_audio_duration(self, audio_path: str) -> float:
        """
        Get the duration of an audio file in seconds.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Duration in seconds.
        """
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            audio_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"FFprobe failed: {result.stderr}"
            )

        info = json.loads(result.stdout)
        return float(info["format"]["duration"])

    def get_video_info(self, video_path: str) -> dict:
        """
        Get video file information.

        Args:
            video_path: Path to the video file.

        Returns:
            Dictionary with video metadata.
        """
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFprobe failed: {result.stderr}")

        return json.loads(result.stdout)
