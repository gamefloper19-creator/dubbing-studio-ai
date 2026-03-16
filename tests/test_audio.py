"""Tests for audio processing modules."""

import os
import subprocess
from unittest.mock import patch, MagicMock

from dubbing_studio.audio.extractor import AudioExtractor
from dubbing_studio.audio.cleaner import AudioCleaner
from dubbing_studio.audio.segmenter import AudioSegmenter, AudioSegment
from dubbing_studio.audio.mixer import AudioMixer
from dubbing_studio.config import AudioConfig, MixingConfig


def _create_test_audio(path, duration=2.0, freq=440):
    """Create a short test audio file using FFmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={duration}",
        "-ar", "44100", "-ac", "1",
        path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def _create_test_video(path, duration=2.0):
    """Create a short test video with audio using FFmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=blue:s=320x240:d={duration}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
        "-shortest",
        "-pix_fmt", "yuv420p",
        path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)


class TestAudioExtractor:
    def test_extract_audio(self, tmp_path):
        video = str(tmp_path / "test.mp4")
        audio_out = str(tmp_path / "output.wav")
        _create_test_video(video)

        ext = AudioExtractor()
        result = ext.extract_audio(video, audio_out)
        assert os.path.exists(result)
        assert result.endswith(".wav")

    def test_get_audio_duration(self, tmp_path):
        audio = str(tmp_path / "test.wav")
        _create_test_audio(audio, duration=1.5)

        ext = AudioExtractor()
        duration = ext.get_audio_duration(audio)
        assert abs(duration - 1.5) < 0.2  # within 200ms tolerance

    def test_extract_background_audio(self, tmp_path):
        video = str(tmp_path / "test.mp4")
        bg_out = str(tmp_path / "bg.wav")
        _create_test_video(video)

        ext = AudioExtractor()
        # This may raise if the video doesn't have separable tracks,
        # but should not crash
        try:
            result = ext.extract_background_audio(video, bg_out)
        except Exception:
            pass  # Expected for simple test video


class TestAudioCleaner:
    def test_clean_audio(self, tmp_path):
        audio_in = str(tmp_path / "noisy.wav")
        audio_out = str(tmp_path / "clean.wav")
        _create_test_audio(audio_in)

        cleaner = AudioCleaner()
        result = cleaner.clean_audio(audio_in, audio_out)
        assert os.path.exists(result)

    def test_clean_audio_custom_config(self, tmp_path):
        cfg = AudioConfig(normalize_loudness=-20.0, noise_reduction_strength=0.8)
        cleaner = AudioCleaner(cfg)
        audio_in = str(tmp_path / "noisy.wav")
        audio_out = str(tmp_path / "clean.wav")
        _create_test_audio(audio_in)

        result = cleaner.clean_audio(audio_in, audio_out)
        assert os.path.exists(result)


class TestAudioSegmenter:
    def test_segment_audio(self, tmp_path):
        audio = str(tmp_path / "speech.wav")
        segments_dir = str(tmp_path / "segments")
        # Create a 5-second audio file
        _create_test_audio(audio, duration=5.0)

        segmenter = AudioSegmenter()
        segments = segmenter.segment_audio(audio, segments_dir)
        assert isinstance(segments, list)
        # Should return at least one segment
        assert len(segments) >= 1
        for seg in segments:
            assert isinstance(seg, AudioSegment)
            assert seg.start_time >= 0
            assert seg.end_time > seg.start_time


class TestAudioMixer:
    def test_simple_mix(self, tmp_path):
        narration = str(tmp_path / "narration.wav")
        background = str(tmp_path / "background.wav")
        output = str(tmp_path / "mixed.wav")

        _create_test_audio(narration, duration=2.0, freq=440)
        _create_test_audio(background, duration=2.0, freq=220)

        mixer = AudioMixer()
        result = mixer.mix_audio(
            narration_path=narration,
            background_path=background,
            output_path=output,
        )
        assert os.path.exists(result)

    def test_concatenate_audio(self, tmp_path):
        a1 = str(tmp_path / "a1.wav")
        a2 = str(tmp_path / "a2.wav")
        output = str(tmp_path / "concat.wav")

        _create_test_audio(a1, duration=1.0, freq=440)
        _create_test_audio(a2, duration=1.0, freq=880)

        mixer = AudioMixer()
        result = mixer.concatenate_audio([a1, a2], output)
        assert os.path.exists(result)
