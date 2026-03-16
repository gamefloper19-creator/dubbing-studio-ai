"""
Input validation for video files, languages, and pipeline parameters.

Validates file formats, sizes, and configuration before pipeline execution
to provide clear error messages and prevent wasted processing time.
"""

import logging
import subprocess
from pathlib import Path

from dubbing_studio.config import SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)

# Supported video container formats
SUPPORTED_VIDEO_FORMATS: set[str] = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".webm",
    ".flv",
    ".wmv",
    ".m4v",
    ".ts",
    ".mts",
}

# Supported audio-only formats (for audio-only dubbing)
SUPPORTED_AUDIO_FORMATS: set[str] = {
    ".wav",
    ".mp3",
    ".flac",
    ".aac",
    ".ogg",
    ".m4a",
    ".wma",
}

# Default limits
DEFAULT_MAX_FILE_SIZE_GB: float = 10.0
DEFAULT_MAX_DURATION_HOURS: float = 4.0
DEFAULT_MIN_DURATION_SECONDS: float = 1.0


class ValidationError(Exception):
    """Raised when input validation fails."""


def validate_video_file(
    path: str,
    max_size_gb: float = DEFAULT_MAX_FILE_SIZE_GB,
    max_duration_hours: float = DEFAULT_MAX_DURATION_HOURS,
    min_duration_seconds: float = DEFAULT_MIN_DURATION_SECONDS,
) -> dict:
    """
    Validate a video file for dubbing pipeline compatibility.

    Checks:
    - File exists and is readable
    - File extension is a supported format
    - File size is within limits
    - File contains valid video/audio streams
    - Duration is within acceptable range

    Args:
        path: Path to the video file.
        max_size_gb: Maximum allowed file size in gigabytes.
        max_duration_hours: Maximum allowed duration in hours.
        min_duration_seconds: Minimum required duration in seconds.

    Returns:
        Dict with file metadata (format, duration, size, streams).

    Raises:
        ValidationError: If validation fails.
    """
    file_path = Path(path)

    # Check existence
    if not file_path.exists():
        raise ValidationError(f"File not found: {path}")

    if not file_path.is_file():
        raise ValidationError(f"Not a file: {path}")

    # Check format
    suffix = file_path.suffix.lower()
    all_supported = SUPPORTED_VIDEO_FORMATS | SUPPORTED_AUDIO_FORMATS
    if suffix not in all_supported:
        raise ValidationError(
            f"Unsupported format '{suffix}'. "
            f"Supported video: {sorted(SUPPORTED_VIDEO_FORMATS)}, "
            f"audio: {sorted(SUPPORTED_AUDIO_FORMATS)}"
        )

    # Check file size
    file_size_bytes = file_path.stat().st_size
    file_size_gb = file_size_bytes / (1024**3)
    if file_size_gb > max_size_gb:
        raise ValidationError(f"File too large: {file_size_gb:.2f} GB " f"(max: {max_size_gb} GB)")

    if file_size_bytes == 0:
        raise ValidationError("File is empty (0 bytes)")

    # Probe file with ffprobe for stream info
    probe_info = _probe_file(path)

    # Check duration
    duration = probe_info.get("duration", 0.0)
    if duration < min_duration_seconds:
        raise ValidationError(f"File too short: {duration:.1f}s " f"(min: {min_duration_seconds}s)")

    max_duration_seconds = max_duration_hours * 3600
    if duration > max_duration_seconds:
        raise ValidationError(
            f"File too long: {duration / 3600:.1f} hours " f"(max: {max_duration_hours} hours)"
        )

    # Check for audio stream
    if not probe_info.get("has_audio"):
        raise ValidationError("File has no audio stream — nothing to dub")

    logger.info(
        "Validated: %s (%.1fs, %.2f GB, %s)",
        file_path.name,
        duration,
        file_size_gb,
        suffix,
    )

    return {
        "path": str(file_path.resolve()),
        "format": suffix,
        "size_bytes": file_size_bytes,
        "size_gb": round(file_size_gb, 3),
        "duration": duration,
        "has_video": probe_info.get("has_video", False),
        "has_audio": probe_info.get("has_audio", False),
        "video_codec": probe_info.get("video_codec", ""),
        "audio_codec": probe_info.get("audio_codec", ""),
        "width": probe_info.get("width", 0),
        "height": probe_info.get("height", 0),
    }


def validate_language(language_code: str) -> str:
    """
    Validate and normalize a language code.

    Args:
        language_code: ISO 639-1 language code (e.g., 'hi', 'es').

    Returns:
        Normalized language code.

    Raises:
        ValidationError: If language is not supported.
    """
    code = language_code.strip().lower()

    if code not in SUPPORTED_LANGUAGES:
        supported = ", ".join(f"{k} ({v})" for k, v in sorted(SUPPORTED_LANGUAGES.items()))
        raise ValidationError(
            f"Unsupported language: '{language_code}'. " f"Supported: {supported}"
        )

    return code


def validate_language_pair(source: str, target: str) -> tuple[str, str]:
    """
    Validate source and target language pair.

    Args:
        source: Source language code.
        target: Target language code.

    Returns:
        Tuple of (source, target) normalized codes.

    Raises:
        ValidationError: If languages are invalid or identical.
    """
    src = validate_language(source)
    tgt = validate_language(target)

    if src == tgt:
        raise ValidationError(
            f"Source and target languages are the same: '{src}'. "
            "Translation requires different languages."
        )

    return src, tgt


def _probe_file(path: str) -> dict:
    """
    Probe a media file using ffprobe to extract stream information.

    Args:
        path: Path to media file.

    Returns:
        Dict with duration, stream info, codecs, and dimensions.
    """
    info: dict = {
        "duration": 0.0,
        "has_video": False,
        "has_audio": False,
        "video_codec": "",
        "audio_codec": "",
        "width": 0,
        "height": 0,
    }

    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            raise ValidationError(f"Cannot read file (ffprobe failed): {result.stderr.strip()}")

        import json

        data = json.loads(result.stdout)

        # Duration
        fmt = data.get("format", {})
        info["duration"] = float(fmt.get("duration", 0))

        # Streams
        for stream in data.get("streams", []):
            codec_type = stream.get("codec_type", "")
            if codec_type == "video":
                info["has_video"] = True
                info["video_codec"] = stream.get("codec_name", "")
                info["width"] = int(stream.get("width", 0))
                info["height"] = int(stream.get("height", 0))
            elif codec_type == "audio":
                info["has_audio"] = True
                info["audio_codec"] = stream.get("codec_name", "")

    except subprocess.TimeoutExpired as e:
        raise ValidationError("File probe timed out — file may be corrupted") from e
    except (ValueError, KeyError) as e:
        raise ValidationError(f"Failed to parse file metadata: {e}") from e

    return info
