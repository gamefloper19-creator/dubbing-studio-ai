"""Tests for subtitle generation module."""

from dubbing_studio.subtitle.generator import SubtitleGenerator
from dubbing_studio.config import SubtitleConfig
from dubbing_studio.translation.translator import TranslatedSegment


def _make_segments():
    """Create test translated segments."""
    return [
        TranslatedSegment(
            segment_id="seg_001",
            original_text="Hello world",
            translated_text="Hola mundo",
            source_language="en",
            target_language="es",
            start_time=0.0,
            end_time=3.5,
        ),
        TranslatedSegment(
            segment_id="seg_002",
            original_text="This is a test",
            translated_text="Esta es una prueba",
            source_language="en",
            target_language="es",
            start_time=4.0,
            end_time=7.2,
        ),
        TranslatedSegment(
            segment_id="seg_003",
            original_text="Goodbye",
            translated_text="Adiós",
            source_language="en",
            target_language="es",
            start_time=8.0,
            end_time=10.0,
        ),
    ]


class TestSubtitleGenerator:
    def test_generate_srt(self, tmp_path):
        gen = SubtitleGenerator()
        segments = _make_segments()
        output = str(tmp_path / "test.srt")
        result = gen.generate(segments, output, "srt")
        assert result.endswith(".srt")

        with open(result) as f:
            content = f.read()
        assert "1" in content
        assert "Hola mundo" in content
        assert "Esta es una prueba" in content
        assert "-->" in content
        # SRT uses comma for milliseconds
        assert "," in content

    def test_generate_vtt(self, tmp_path):
        gen = SubtitleGenerator()
        segments = _make_segments()
        output = str(tmp_path / "test.vtt")
        result = gen.generate(segments, output, "vtt")
        assert result.endswith(".vtt")

        with open(result) as f:
            content = f.read()
        assert "WEBVTT" in content
        assert "Hola mundo" in content

    def test_generate_ass(self, tmp_path):
        gen = SubtitleGenerator()
        segments = _make_segments()
        output = str(tmp_path / "test.ass")
        result = gen.generate(segments, output, "ass")
        assert result.endswith(".ass")

        with open(result) as f:
            content = f.read()
        assert "[Script Info]" in content
        assert "Hola mundo" in content

    def test_generate_all_formats(self, tmp_path):
        gen = SubtitleGenerator()
        segments = _make_segments()
        output_dir = str(tmp_path / "subs")
        results = gen.generate_all_formats(segments, output_dir, "test")
        assert "srt" in results
        assert "vtt" in results
        assert "ass" in results

    def test_empty_segments(self, tmp_path):
        gen = SubtitleGenerator()
        output = str(tmp_path / "empty.srt")
        result = gen.generate([], output, "srt")
        with open(result) as f:
            content = f.read()
        # Should produce valid (possibly empty) SRT
        assert isinstance(content, str)

    def test_custom_config(self, tmp_path):
        cfg = SubtitleConfig(font_size=32, font_color="yellow")
        gen = SubtitleGenerator(cfg)
        segments = _make_segments()
        output = str(tmp_path / "custom.ass")
        result = gen.generate(segments, output, "ass")
        with open(result) as f:
            content = f.read()
        assert "32" in content  # font size should appear
