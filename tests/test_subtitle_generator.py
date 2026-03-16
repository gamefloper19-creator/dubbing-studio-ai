"""Tests for subtitle generation module."""

from dubbing_studio.config import SubtitleConfig
from dubbing_studio.subtitle.generator import SubtitleGenerator
from dubbing_studio.translation.translator import TranslatedSegment


def _make_segment(seg_id: str, start: float, end: float, text: str) -> TranslatedSegment:
    return TranslatedSegment(
        segment_id=seg_id,
        start_time=start,
        end_time=end,
        original_text=text,
        translated_text=text,
        source_language="en",
        target_language="hi",
    )


class TestSubtitleGenerator:
    def setup_method(self):
        self.gen = SubtitleGenerator()

    def test_generate_srt(self, tmp_path):
        segments = [
            _make_segment("s1", 0.0, 3.5, "Hello world."),
            _make_segment("s2", 4.0, 7.2, "This is a test."),
        ]
        output = self.gen.generate(segments, str(tmp_path / "sub"), format="srt")
        assert output.endswith(".srt")
        with open(output) as f:
            content = f.read()
        assert "1\n" in content
        assert "00:00:00,000 --> 00:00:03,500" in content
        assert "Hello world." in content
        assert "2\n" in content

    def test_generate_vtt(self, tmp_path):
        segments = [_make_segment("s1", 1.5, 4.0, "VTT test.")]
        output = self.gen.generate(segments, str(tmp_path / "sub"), format="vtt")
        assert output.endswith(".vtt")
        with open(output) as f:
            content = f.read()
        assert content.startswith("WEBVTT")
        assert "00:00:01.500 --> 00:00:04.000" in content

    def test_generate_ass(self, tmp_path):
        segments = [_make_segment("s1", 0.0, 5.0, "ASS test.")]
        output = self.gen.generate(segments, str(tmp_path / "sub"), format="ass")
        assert output.endswith(".ass")
        with open(output) as f:
            content = f.read()
        assert "[Script Info]" in content
        assert "[Events]" in content
        assert "ASS test." in content

    def test_generate_all_formats(self, tmp_path):
        segments = [_make_segment("s1", 0.0, 3.0, "All formats.")]
        paths = self.gen.generate_all_formats(segments, str(tmp_path), "test")
        assert "srt" in paths
        assert "vtt" in paths
        assert "ass" in paths

    def test_unsupported_format(self, tmp_path):
        import pytest

        with pytest.raises(ValueError, match="Unsupported"):
            self.gen.generate([], str(tmp_path / "sub"), format="xyz")

    def test_srt_time_format(self):
        result = self.gen._format_time_srt(3661.123)
        assert result == "01:01:01,123"

    def test_vtt_time_format(self):
        result = self.gen._format_time_vtt(3661.123)
        assert result == "01:01:01.123"

    def test_ass_time_format(self):
        result = self.gen._format_time_ass(3661.5)
        assert result == "1:01:01.50"

    def test_color_to_ass(self):
        assert self.gen._color_to_ass("white") == "&H00FFFFFF"
        assert self.gen._color_to_ass("black") == "&H00000000"
        assert self.gen._color_to_ass("unknown") == "&H00FFFFFF"

    def test_custom_config(self, tmp_path):
        config = SubtitleConfig(format="vtt", font_size=32, position="top")
        gen = SubtitleGenerator(config)
        segments = [_make_segment("s1", 0.0, 3.0, "Custom.")]
        output = gen.generate(segments, str(tmp_path / "sub"))
        assert output.endswith(".vtt")
