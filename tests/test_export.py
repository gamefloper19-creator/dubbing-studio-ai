"""Tests for export module."""

from unittest.mock import patch, MagicMock

from dubbing_studio.config import ExportConfig
from dubbing_studio.export.exporter import Exporter


class TestExporter:
    def test_initialization_default(self):
        exp = Exporter()
        assert exp.config is not None
        assert exp.renderer is not None

    def test_initialization_custom(self):
        cfg = ExportConfig(video_codec="libx265", audio_bitrate="320k")
        exp = Exporter(cfg)
        assert exp.config.video_codec == "libx265"
        assert exp.config.audio_bitrate == "320k"

    def test_export_video_adds_mp4_suffix(self):
        exp = Exporter()
        with patch.object(exp.renderer, "render_video", return_value="/out/test.mp4"):
            result = exp.export_video(
                video_path="/in/video.mp4",
                audio_path="/in/audio.wav",
                output_path="/out/test",
            )
            assert result == "/out/test.mp4"

    def test_export_audio_wav(self):
        exp = Exporter()
        with patch.object(exp.renderer, "render_audio_only", return_value="/out/test.wav"):
            result = exp.export_audio(
                audio_path="/in/audio.wav",
                output_path="/out/test.wav",
                format="wav",
            )
            assert result == "/out/test.wav"

    def test_export_audio_mp3(self):
        exp = Exporter()
        with patch.object(exp.renderer, "render_audio_only", return_value="/out/test.mp3"):
            result = exp.export_audio(
                audio_path="/in/audio.wav",
                output_path="/out/test.mp3",
                format="mp3",
            )
            assert result == "/out/test.mp3"

    def test_export_audio_default_format(self):
        cfg = ExportConfig(audio_only_format="mp3")
        exp = Exporter(cfg)
        with patch.object(exp.renderer, "render_audio_only", return_value="/out/test.mp3") as mock:
            exp.export_audio("/in/audio.wav", "/out/test.mp3")
            mock.assert_called_once_with(
                audio_path="/in/audio.wav",
                output_path="/out/test.mp3",
                format="mp3",
            )

    def test_export_all(self, tmp_path):
        exp = Exporter()
        with patch.object(exp.renderer, "render_video", return_value=str(tmp_path / "out.mp4")):
            with patch.object(exp.renderer, "render_audio_only", side_effect=[
                str(tmp_path / "out.wav"),
                str(tmp_path / "out.mp3"),
            ]):
                results = exp.export_all(
                    video_path="/in/video.mp4",
                    audio_path="/in/audio.wav",
                    output_dir=str(tmp_path),
                    base_name="test_output",
                    formats=["mp4", "wav", "mp3"],
                )
                assert "mp4" in results
                assert "wav" in results
                assert "mp3" in results

    def test_export_all_with_subtitles(self, tmp_path):
        exp = Exporter()
        sub_file = tmp_path / "subs.srt"
        sub_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nTest\n")

        with patch.object(exp.renderer, "render_video", return_value=str(tmp_path / "out.mp4")):
            results = exp.export_all(
                video_path="/in/video.mp4",
                audio_path="/in/audio.wav",
                output_dir=str(tmp_path / "output"),
                base_name="test",
                subtitle_path=str(sub_file),
                formats=["mp4"],
            )
            assert "mp4" in results
            assert "subtitle_srt" in results

    def test_export_all_default_formats(self, tmp_path):
        exp = Exporter()
        with patch.object(exp.renderer, "render_video", return_value=str(tmp_path / "out.mp4")):
            with patch.object(exp.renderer, "render_audio_only", side_effect=[
                str(tmp_path / "out.wav"),
                str(tmp_path / "out.mp3"),
            ]):
                results = exp.export_all(
                    video_path="/in/video.mp4",
                    audio_path="/in/audio.wav",
                    output_dir=str(tmp_path),
                    base_name="test",
                )
                # Default formats: mp4, wav, mp3
                assert len(results) == 3
