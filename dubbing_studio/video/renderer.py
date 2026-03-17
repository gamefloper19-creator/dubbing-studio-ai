"""
Final video rendering with dubbed audio and optional subtitles.
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional

from dubbing_studio.config import ExportConfig, SubtitleConfig

logger = logging.getLogger(__name__)


class VideoRenderer:
    """Render final video with dubbed audio and subtitles."""

    def __init__(
        self,
        export_config: Optional[ExportConfig] = None,
        subtitle_config: Optional[SubtitleConfig] = None,
    ):
        self.export_config = export_config or ExportConfig()
        self.subtitle_config = subtitle_config or SubtitleConfig()

    def render_video(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        subtitle_path: Optional[str] = None,
        embed_subtitles: Optional[bool] = None,
    ) -> str:
        """
        Render final video with dubbed audio.

        Args:
            video_path: Original video file path.
            audio_path: Dubbed audio file path.
            output_path: Output video file path.
            subtitle_path: Optional subtitle file to embed.
            embed_subtitles: Whether to burn subtitles into video.

        Returns:
            Path to rendered video.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        should_embed = embed_subtitles if embed_subtitles is not None else self.subtitle_config.embed_in_video

        if should_embed and subtitle_path:
            return self._render_with_subtitles(
                video_path, audio_path, output_path, subtitle_path
            )
        else:
            return self._render_without_subtitles(
                video_path, audio_path, output_path
            )

    def _render_without_subtitles(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
    ) -> str:
        """Render video with dubbed audio, no subtitles."""
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", self.export_config.video_codec,
            "-crf", str(self.export_config.video_quality),
            "-c:a", self.export_config.audio_codec,
            "-b:a", self.export_config.audio_bitrate,
            "-map", "0:v:0",  # video from first input
            "-map", "1:a:0",  # audio from second input
            "-shortest",
            output_path,
        ]

        logger.info("Rendering video: %s", output_path)
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Video rendering failed: {result.stderr}")

        logger.info("Video rendered successfully: %s", output_path)
        return output_path

    def _render_with_subtitles(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        subtitle_path: str,
    ) -> str:
        """Render video with dubbed audio and burned-in subtitles."""
        sub_ext = Path(subtitle_path).suffix.lower()

        if sub_ext == ".ass":
            # ASS subtitles with styling
            safe_path = subtitle_path.replace("\\", "/").replace(":", "\\:")
            subtitle_filter = f"ass='{safe_path}'"
        else:
            # SRT/VTT subtitles
            safe_path = subtitle_path.replace("\\", "/").replace(":", "\\:")
            subtitle_filter = (
                f"subtitles='{safe_path}'"
                f":force_style='FontSize={self.subtitle_config.font_size},"
                f"PrimaryColour=&H00FFFFFF,"
                f"OutlineColour=&H00000000,"
                f"Outline={self.subtitle_config.outline_width}'"
            )

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", self.export_config.video_codec,
            "-crf", str(self.export_config.video_quality),
            "-vf", subtitle_filter,
            "-c:a", self.export_config.audio_codec,
            "-b:a", self.export_config.audio_bitrate,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            output_path,
        ]

        logger.info("Rendering video with subtitles: %s", output_path)
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.warning(
                "Subtitle rendering failed, rendering without subtitles: %s",
                result.stderr,
            )
            return self._render_without_subtitles(
                video_path, audio_path, output_path
            )

        logger.info("Video with subtitles rendered: %s", output_path)
        return output_path

    def render_audio_only(
        self,
        audio_path: str,
        output_path: str,
        format: Optional[str] = None,
    ) -> str:
        """
        Export audio in specified format.

        Args:
            audio_path: Input audio file.
            output_path: Output audio file path.
            format: Audio format (wav, mp3). Uses config default if None.

        Returns:
            Path to exported audio.
        """
        fmt = format or self.export_config.audio_only_format
        output_path = str(Path(output_path).with_suffix(f".{fmt}"))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if fmt == "wav":
            cmd = [
                "ffmpeg", "-y",
                "-i", audio_path,
                "-acodec", "pcm_s16le",
                output_path,
            ]
        elif fmt == "mp3":
            cmd = [
                "ffmpeg", "-y",
                "-i", audio_path,
                "-acodec", "libmp3lame",
                "-b:a", self.export_config.audio_bitrate,
                output_path,
            ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-i", audio_path,
                output_path,
            ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Audio export failed: {result.stderr}")

        logger.info("Audio exported: %s", output_path)
        return output_path

    def add_subtitle_stream(
        self,
        video_path: str,
        subtitle_path: str,
        output_path: str,
    ) -> str:
        """
        Add subtitle as a separate stream (not burned in).

        Args:
            video_path: Video file with dubbed audio.
            subtitle_path: Subtitle file.
            output_path: Output video path.

        Returns:
            Path to video with subtitle stream.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        sub_ext = Path(subtitle_path).suffix.lower()
        sub_codec = "mov_text" if sub_ext in (".srt", ".vtt") else "ass"

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", subtitle_path,
            "-c:v", "copy",
            "-c:a", "copy",
            "-c:s", sub_codec,
            "-map", "0:v",
            "-map", "0:a",
            "-map", "1:0",
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning("Subtitle stream addition failed: %s", result.stderr)
            return video_path

        return output_path
