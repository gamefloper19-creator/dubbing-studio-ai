"""Tests for the main dubbing pipeline orchestrator."""

from unittest.mock import patch, MagicMock

from dubbing_studio.pipeline import DubbingPipeline, DubbingResult, PIPELINE_STAGES


class TestPipelineStages:
    def test_stage_count(self):
        assert len(PIPELINE_STAGES) == 12

    def test_stage_names(self):
        assert "Audio Extraction" in PIPELINE_STAGES
        assert "Audio Cleaning" in PIPELINE_STAGES
        assert "Segmentation" in PIPELINE_STAGES
        assert "Speech Recognition" in PIPELINE_STAGES
        assert "Narration Analysis" in PIPELINE_STAGES
        assert "Translation" in PIPELINE_STAGES
        assert "Voice Selection" in PIPELINE_STAGES
        assert "Speech Generation" in PIPELINE_STAGES
        assert "Timing Alignment" in PIPELINE_STAGES
        assert "Background Mixing" in PIPELINE_STAGES
        assert "Subtitle Generation" in PIPELINE_STAGES
        assert "Video Rendering" in PIPELINE_STAGES


class TestDubbingResult:
    def test_defaults(self):
        result = DubbingResult(
            video_path="/in/video.mp4",
            output_video_path="/out/video.mp4",
            output_audio_path="/out/audio.wav",
        )
        assert result.video_path == "/in/video.mp4"
        assert result.subtitle_paths == {}
        assert result.source_language == ""
        assert result.target_language == ""
        assert result.total_segments == 0
        assert result.total_duration == 0.0
        assert result.processing_time == 0.0
        assert result.narration_style is None


class TestDubbingPipeline:
    def test_initialization(self, tmp_path):
        from dubbing_studio.config import AppConfig
        cfg = AppConfig()
        cfg.output_dir = str(tmp_path / "output")
        cfg.temp_dir = str(tmp_path / "temp")
        cfg.cache_dir = str(tmp_path / "cache")
        pipeline = DubbingPipeline(cfg)

        assert pipeline.extractor is not None
        assert pipeline.cleaner is not None
        assert pipeline.segmenter is not None
        assert pipeline.recognizer is not None
        assert pipeline.analyzer is not None
        assert pipeline.translator is not None
        assert pipeline.voice_selector is not None
        assert pipeline.aligner is not None
        assert pipeline.mixer is not None
        assert pipeline.subtitle_gen is not None
        assert pipeline.renderer is not None
        assert pipeline.exporter is not None

    def test_cleanup_temp(self, tmp_path):
        from dubbing_studio.config import AppConfig
        cfg = AppConfig()
        cfg.output_dir = str(tmp_path / "output")
        cfg.temp_dir = str(tmp_path / "temp")
        cfg.cache_dir = str(tmp_path / "cache")
        pipeline = DubbingPipeline(cfg)

        work_dir = tmp_path / "work"
        work_dir.mkdir()
        (work_dir / "test.txt").write_text("temp data")

        pipeline._cleanup_temp(work_dir)
        assert not work_dir.exists()

    def test_cleanup_temp_nonexistent(self, tmp_path):
        from dubbing_studio.config import AppConfig
        cfg = AppConfig()
        cfg.output_dir = str(tmp_path / "output")
        cfg.temp_dir = str(tmp_path / "temp")
        cfg.cache_dir = str(tmp_path / "cache")
        pipeline = DubbingPipeline(cfg)

        # Should not raise
        pipeline._cleanup_temp(tmp_path / "nonexistent")
