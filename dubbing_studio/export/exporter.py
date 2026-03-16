"""
Export manager for final output in various formats.
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

from dubbing_studio.config import ExportConfig
from dubbing_studio.video.renderer import VideoRenderer

logger = logging.getLogger(__name__)


class Exporter:
    """
    Export dubbed content in various formats.

    Supported:
    - MP4 (video with dubbed audio)
    - WAV (audio only, uncompressed)
    - MP3 (audio only, compressed)
    - Subtitles (SRT, VTT, ASS)
    """

    def __init__(self, config: Optional[ExportConfig] = None):
        self.config = config or ExportConfig()
        self.renderer = VideoRenderer(export_config=self.config)

    def export_video(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        subtitle_path: Optional[str] = None,
        embed_subtitles: bool = False,
    ) -> str:
        """
        Export dubbed video in MP4 format.

        Args:
            video_path: Original video path.
            audio_path: Dubbed audio path.
            output_path: Output video path.
            subtitle_path: Optional subtitle file.
            embed_subtitles: Whether to burn subtitles in.

        Returns:
            Path to exported video.
        """
        output_path = str(Path(output_path).with_suffix(".mp4"))

        result = self.renderer.render_video(
            video_path=video_path,
            audio_path=audio_path,
            output_path=output_path,
            subtitle_path=subtitle_path,
            embed_subtitles=embed_subtitles,
        )

        logger.info("Exported video: %s", result)
        return result

    def export_audio(
        self,
        audio_path: str,
        output_path: str,
        format: Optional[str] = None,
    ) -> str:
        """
        Export audio in WAV or MP3 format.

        Args:
            audio_path: Dubbed audio path.
            output_path: Output audio path.
            format: Audio format (wav, mp3).

        Returns:
            Path to exported audio.
        """
        fmt = format or self.config.audio_only_format

        result = self.renderer.render_audio_only(
            audio_path=audio_path,
            output_path=output_path,
            format=fmt,
        )

        logger.info("Exported audio (%s): %s", fmt, result)
        return result

    def export_all(
        self,
        video_path: str,
        audio_path: str,
        output_dir: str,
        base_name: str,
        subtitle_path: Optional[str] = None,
        embed_subtitles: bool = False,
        formats: Optional[list[str]] = None,
    ) -> dict[str, str]:
        """
        Export in all requested formats.

        Args:
            video_path: Original video.
            audio_path: Dubbed audio.
            output_dir: Output directory.
            base_name: Base filename.
            subtitle_path: Optional subtitle file.
            embed_subtitles: Whether to burn subtitles.
            formats: List of formats to export (default: mp4, wav, mp3).

        Returns:
            Dict mapping format to output path.
        """
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        if formats is None:
            formats = ["mp4", "wav", "mp3"]

        results = {}

        for fmt in formats:
            output_path = str(output_dir_path / f"{base_name}.{fmt}")

            if fmt == "mp4":
                results["mp4"] = self.export_video(
                    video_path, audio_path, output_path,
                    subtitle_path, embed_subtitles,
                )
            elif fmt in ("wav", "mp3"):
                results[fmt] = self.export_audio(
                    audio_path, output_path, fmt,
                )

        # Copy subtitle files if they exist
        if subtitle_path:
            sub_ext = Path(subtitle_path).suffix
            sub_output = str(output_dir_path / f"{base_name}{sub_ext}")
            shutil.copy2(subtitle_path, sub_output)
            results[f"subtitle_{sub_ext[1:]}"] = sub_output

        return results
