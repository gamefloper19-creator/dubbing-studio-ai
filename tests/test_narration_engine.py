"""Tests for cinematic narration engine."""

from dubbing_studio.config import CinematicNarrationConfig
from dubbing_studio.narration.engine import CinematicNarrationEngine, NarrationBlock
from dubbing_studio.translation.translator import TranslatedSegment


def _make_segment(
    seg_id: str,
    start: float,
    end: float,
    text: str,
    original: str = "",
) -> TranslatedSegment:
    return TranslatedSegment(
        segment_id=seg_id,
        start_time=start,
        end_time=end,
        original_text=original or text,
        translated_text=text,
        source_language="en",
        target_language="hi",
    )


class TestCinematicNarrationEngine:
    def setup_method(self):
        self.engine = CinematicNarrationEngine()

    def test_empty_segments(self):
        result = self.engine.optimize_narration([])
        assert result == []

    def test_single_segment_unchanged(self):
        seg = _make_segment("s1", 0.0, 10.0, "Hello world.")
        result = self.engine.optimize_narration([seg])
        assert len(result) == 1
        assert result[0].translated_text == "Hello world."

    def test_merges_short_segments(self):
        segments = [
            _make_segment("s1", 0.0, 1.0, "Short one."),
            _make_segment("s2", 1.1, 2.0, "Short two."),
            _make_segment("s3", 2.1, 3.0, "Short three."),
        ]
        result = self.engine.optimize_narration(segments)
        assert len(result) < len(segments)

    def test_preserves_long_segments(self):
        segments = [
            _make_segment("s1", 0.0, 10.0, "This is a long segment about history."),
            _make_segment("s2", 15.0, 25.0, "Another long segment about science."),
        ]
        result = self.engine.optimize_narration(segments)
        assert len(result) == 2

    def test_smooth_transitions_adds_period(self):
        segments = [
            _make_segment("s1", 0.0, 5.0, "First sentence"),
            _make_segment("s2", 5.5, 10.0, "Second sentence"),
        ]
        result = self.engine.optimize_narration(segments)
        for seg in result:
            assert seg.translated_text.endswith(".")

    def test_removes_redundant_connectors(self):
        engine = CinematicNarrationEngine()
        result = engine._remove_redundant_connectors(
            "And then, the story continued with great detail."
        )
        assert not result.startswith("And then")
        assert result[0].isupper()

    def test_optimize_pacing_fast_segment(self):
        config = CinematicNarrationConfig(target_wpm=145.0)
        engine = CinematicNarrationEngine(config)
        # ~600 WPM (way too fast)
        seg = _make_segment("s1", 0.0, 1.0, " ".join(["word"] * 10))
        result = engine._optimize_pacing([seg])
        assert result[0].end_time >= seg.end_time

    def test_analyze_narration_blocks(self):
        segments = [
            _make_segment("s1", 0.0, 5.0, "Block one content."),
            _make_segment("s2", 6.0, 11.0, "Block two content."),
        ]
        blocks = self.engine.analyze_narration_blocks(segments)
        assert len(blocks) == 2
        assert all(isinstance(b, NarrationBlock) for b in blocks)
        assert blocks[0].word_count == 3
        assert blocks[0].pause_after > 0

    def test_disabled_features(self):
        config = CinematicNarrationConfig(
            merge_short_segments=False,
            smooth_transitions=False,
            optimize_pacing=False,
        )
        engine = CinematicNarrationEngine(config)
        segments = [
            _make_segment("s1", 0.0, 1.0, "Short"),
            _make_segment("s2", 1.1, 2.0, "Segments"),
        ]
        result = engine.optimize_narration(segments)
        assert len(result) == 2
