"""Tests for TTS helper functions (async runner, WAV conversion)."""

import asyncio
import os
from unittest.mock import patch, MagicMock

from dubbing_studio.tts.qwen_tts import _run_async, _convert_to_wav


class TestRunAsync:
    def test_run_outside_event_loop(self):
        """_run_async works when no event loop is running."""
        async def simple_coro():
            return 42

        result = _run_async(simple_coro())
        assert result == 42

    def test_run_inside_event_loop(self):
        """_run_async works when called from within an existing event loop."""
        async def inner():
            return "hello"

        async def outer():
            return _run_async(inner())

        # This simulates the Gradio context where an event loop is already running
        result = asyncio.run(outer())
        assert result == "hello"


class TestConvertToWav:
    def test_convert_creates_output(self, tmp_path):
        """Test WAV conversion with a real FFmpeg call."""
        # Create a small test audio file using FFmpeg
        input_file = str(tmp_path / "test.mp3")
        output_file = str(tmp_path / "test.wav")

        import subprocess
        # Generate a short sine wave as MP3
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=0.5",
            "-acodec", "libmp3lame",
            input_file,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        assert os.path.exists(input_file)

        _convert_to_wav(input_file, output_file)

        assert os.path.exists(output_file)
        # Original MP3 should be cleaned up
        assert not os.path.exists(input_file)

    def test_convert_cleanup_on_success(self, tmp_path):
        """Input file is removed after successful conversion."""
        input_file = str(tmp_path / "test.mp3")
        output_file = str(tmp_path / "test.wav")

        import subprocess
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=0.2",
            "-acodec", "libmp3lame",
            input_file,
        ]
        subprocess.run(cmd, capture_output=True, check=True)

        _convert_to_wav(input_file, output_file)
        assert not os.path.exists(input_file)

    def test_convert_fallback_on_failure(self, tmp_path):
        """On FFmpeg failure, falls back to copying the file."""
        input_file = str(tmp_path / "test.mp3")
        output_file = str(tmp_path / "test.wav")

        # Write dummy data
        with open(input_file, "w") as f:
            f.write("not real audio")

        # Patch subprocess to simulate FFmpeg failure
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stderr = "error"
            mock_run.return_value = mock_result

            _convert_to_wav(input_file, output_file)

        # Should still produce an output (copied from input)
        assert os.path.exists(output_file)
