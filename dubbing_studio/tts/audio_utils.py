"""
Audio utilities for TTS post-processing.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def apply_pitch_shift(path: str, factor: float) -> None:
    """
    Apply pitch shift while preserving duration using FFmpeg.

    Args:
        path: Audio file path to modify in place.
        factor: Pitch multiplier (1.0 = no change).
    """
    if factor == 1.0:
        return

    sample_rate = _probe_sample_rate(path) or 24000
    tempo = 1.0 / max(factor, 0.01)
    atempo_chain = _build_atempo_chain(tempo)

    src = Path(path)
    tmp = src.with_suffix(src.suffix + ".pitch.wav")

    filter_chain = (
        f"asetrate={int(sample_rate * factor)},"
        f"{atempo_chain},"
        f"aresample={sample_rate}"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-af", filter_chain,
        str(tmp),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("Pitch shift failed, keeping original audio: %s", result.stderr)
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return

    tmp.replace(src)


def _probe_sample_rate(path: str) -> int | None:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None

    try:
        data = json.loads(result.stdout)
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "audio":
                return int(stream.get("sample_rate", 0)) or None
    except (ValueError, TypeError):
        return None

    return None


def _build_atempo_chain(tempo: float) -> str:
    """Build chain of atempo filters for extreme tempo changes."""
    filters = []
    remaining = tempo

    while remaining > 100.0:
        filters.append("atempo=100.0")
        remaining /= 100.0

    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5

    filters.append(f"atempo={remaining:.4f}")
    return ",".join(filters)
