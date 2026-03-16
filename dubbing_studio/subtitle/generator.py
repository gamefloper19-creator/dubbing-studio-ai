"""
Subtitle generation in SRT, VTT, and ASS formats.
"""

import logging
from pathlib import Path
from typing import Optional

from dubbing_studio.config import SubtitleConfig
from dubbing_studio.translation.translator import TranslatedSegment

logger = logging.getLogger(__name__)


class SubtitleGenerator:
    """Generate subtitle files from transcription/translation segments."""

    def __init__(self, config: Optional[SubtitleConfig] = None):
        self.config = config or SubtitleConfig()

    def generate(
        self,
        segments: list[TranslatedSegment],
        output_path: str,
        format: Optional[str] = None,
    ) -> str:
        """
        Generate subtitle file.

        Args:
            segments: List of translated segments with timing.
            output_path: Output file path (extension will be adjusted).
            format: Subtitle format (srt, vtt, ass). Uses config default if None.

        Returns:
            Path to generated subtitle file.
        """
        fmt = format or self.config.format

        # Ensure correct extension
        output_path = str(Path(output_path).with_suffix(f".{fmt}"))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if fmt == "srt":
            return self._generate_srt(segments, output_path)
        elif fmt == "vtt":
            return self._generate_vtt(segments, output_path)
        elif fmt == "ass":
            return self._generate_ass(segments, output_path)
        else:
            raise ValueError(f"Unsupported subtitle format: {fmt}")

    def generate_all_formats(
        self,
        segments: list[TranslatedSegment],
        output_dir: str,
        base_name: str = "subtitles",
    ) -> dict[str, str]:
        """
        Generate subtitles in all supported formats.

        Args:
            segments: Translated segments.
            output_dir: Output directory.
            base_name: Base filename without extension.

        Returns:
            Dict mapping format to file path.
        """
        paths = {}
        for fmt in ["srt", "vtt", "ass"]:
            output_path = str(Path(output_dir) / f"{base_name}.{fmt}")
            paths[fmt] = self.generate(segments, output_path, fmt)

        return paths

    def _generate_srt(
        self,
        segments: list[TranslatedSegment],
        output_path: str,
    ) -> str:
        """Generate SRT subtitle file."""
        lines = []

        for i, seg in enumerate(segments):
            start = self._format_time_srt(seg.start_time)
            end = self._format_time_srt(seg.end_time)
            text = seg.translated_text

            lines.append(f"{i + 1}")
            lines.append(f"{start} --> {end}")
            lines.append(text)
            lines.append("")

        content = "\n".join(lines)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info("Generated SRT subtitles: %s", output_path)
        return output_path

    def _generate_vtt(
        self,
        segments: list[TranslatedSegment],
        output_path: str,
    ) -> str:
        """Generate WebVTT subtitle file."""
        lines = ["WEBVTT", ""]

        for i, seg in enumerate(segments):
            start = self._format_time_vtt(seg.start_time)
            end = self._format_time_vtt(seg.end_time)
            text = seg.translated_text

            lines.append(f"{i + 1}")
            lines.append(f"{start} --> {end}")
            lines.append(text)
            lines.append("")

        content = "\n".join(lines)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info("Generated VTT subtitles: %s", output_path)
        return output_path

    def _generate_ass(
        self,
        segments: list[TranslatedSegment],
        output_path: str,
    ) -> str:
        """Generate ASS (Advanced SubStation Alpha) subtitle file."""
        header = self._ass_header()
        events = []

        for seg in segments:
            start = self._format_time_ass(seg.start_time)
            end = self._format_time_ass(seg.end_time)
            text = seg.translated_text.replace("\n", "\\N")

            events.append(
                f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}"
            )

        content = header + "\n".join(events) + "\n"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info("Generated ASS subtitles: %s", output_path)
        return output_path

    def _ass_header(self) -> str:
        """Generate ASS file header."""
        font_size = self.config.font_size
        primary_color = self._color_to_ass(self.config.font_color)
        outline_color = self._color_to_ass(self.config.outline_color)
        outline_width = self.config.outline_width

        # ASS alignment: 2 = bottom center, 8 = top center
        alignment = 2 if self.config.position == "bottom" else 8

        return (
            "[Script Info]\n"
            "Title: Dubbing Studio Subtitles\n"
            "ScriptType: v4.00+\n"
            "WrapStyle: 0\n"
            "PlayResX: 1920\n"
            "PlayResY: 1080\n"
            "\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding\n"
            f"Style: Default,Arial,{font_size},{primary_color},&H000000FF,"
            f"{outline_color},&H80000000,-1,0,0,0,100,100,0,0,1,"
            f"{outline_width},0,{alignment},20,20,40,1\n"
            "\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
            "Effect, Text\n"
        )

    def _format_time_srt(self, seconds: float) -> str:
        """Format time for SRT: HH:MM:SS,mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _format_time_vtt(self, seconds: float) -> str:
        """Format time for VTT: HH:MM:SS.mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    def _format_time_ass(self, seconds: float) -> str:
        """Format time for ASS: H:MM:SS.cc"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centis = int((seconds % 1) * 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"

    def _color_to_ass(self, color: str) -> str:
        """Convert color name to ASS format (&HAABBGGRR)."""
        color_map = {
            "white": "&H00FFFFFF",
            "black": "&H00000000",
            "red": "&H000000FF",
            "yellow": "&H0000FFFF",
            "green": "&H0000FF00",
            "blue": "&H00FF0000",
            "cyan": "&H00FFFF00",
        }
        return color_map.get(color.lower(), "&H00FFFFFF")
