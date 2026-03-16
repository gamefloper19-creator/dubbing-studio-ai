"""
Edge TTS fallback - shared helper for all TTS engines.

Microsoft Edge TTS serves as the universal fallback when native
TTS models (Qwen3, Chatterbox, LuxTTS) are unavailable. This module
centralizes the Edge TTS logic to avoid code duplication.
"""

import asyncio
import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Comprehensive voice map covering all supported languages.
# Each language maps to a narrator-style male voice by default.
EDGE_VOICE_MAP: dict[str, dict[str, str]] = {
    # Asian & Middle Eastern
    "en": {"male": "en-US-GuyNeural", "female": "en-US-JennyNeural"},
    "hi": {"male": "hi-IN-MadhurNeural", "female": "hi-IN-SwaraNeural"},
    "zh": {"male": "zh-CN-YunxiNeural", "female": "zh-CN-XiaoxiaoNeural"},
    "ja": {"male": "ja-JP-KeitaNeural", "female": "ja-JP-NanamiNeural"},
    "ko": {"male": "ko-KR-InJoonNeural", "female": "ko-KR-SunHiNeural"},
    "ar": {"male": "ar-SA-HamedNeural", "female": "ar-SA-ZariyahNeural"},
    "th": {"male": "th-TH-NiwatNeural", "female": "th-TH-PremwadeeNeural"},
    "vi": {"male": "vi-VN-NamMinhNeural", "female": "vi-VN-HoaiMyNeural"},
    "id": {"male": "id-ID-ArdiNeural", "female": "id-ID-GadisNeural"},
    "bn": {"male": "bn-IN-BashkarNeural", "female": "bn-IN-TanishaaNeural"},
    "ta": {"male": "ta-IN-ValluvarNeural", "female": "ta-IN-PallaviNeural"},
    "te": {"male": "te-IN-MohanNeural", "female": "te-IN-ShrutiNeural"},
    "kn": {"male": "kn-IN-GaganNeural", "female": "kn-IN-SapnaNeural"},
    "mr": {"male": "mr-IN-ManoharNeural", "female": "mr-IN-AarohiNeural"},
    "gu": {"male": "gu-IN-NiranjanNeural", "female": "gu-IN-DhwaniNeural"},
    "ur": {"male": "ur-PK-AsadNeural", "female": "ur-PK-UzmaNeural"},
    "he": {"male": "he-IL-AvriNeural", "female": "he-IL-HilaNeural"},
    # European
    "de": {"male": "de-DE-ConradNeural", "female": "de-DE-KatjaNeural"},
    "fr": {"male": "fr-FR-HenriNeural", "female": "fr-FR-DeniseNeural"},
    "es": {"male": "es-ES-AlvaroNeural", "female": "es-ES-ElviraNeural"},
    "pt": {"male": "pt-BR-AntonioNeural", "female": "pt-BR-FranciscaNeural"},
    "it": {"male": "it-IT-DiegoNeural", "female": "it-IT-ElsaNeural"},
    "ru": {"male": "ru-RU-DmitryNeural", "female": "ru-RU-SvetlanaNeural"},
    "tr": {"male": "tr-TR-AhmetNeural", "female": "tr-TR-EmelNeural"},
    "pl": {"male": "pl-PL-MarekNeural", "female": "pl-PL-ZofiaNeural"},
    "nl": {"male": "nl-NL-MaartenNeural", "female": "nl-NL-ColetteNeural"},
    "sv": {"male": "sv-SE-MattiasNeural", "female": "sv-SE-SofieNeural"},
    "da": {"male": "da-DK-JeppeNeural", "female": "da-DK-ChristelNeural"},
    "fi": {"male": "fi-FI-HarriNeural", "female": "fi-FI-NooraNeural"},
    "el": {"male": "el-GR-NestorasNeural", "female": "el-GR-AthinaNeural"},
    "cs": {"male": "cs-CZ-AntoninNeural", "female": "cs-CZ-VlastaNeural"},
    "ro": {"male": "ro-RO-EmilNeural", "female": "ro-RO-AlinaNeural"},
    "hu": {"male": "hu-HU-TamasNeural", "female": "hu-HU-NoemiNeural"},
    "uk": {"male": "uk-UA-OstapNeural", "female": "uk-UA-PolinaNeural"},
}


def get_edge_voice(language: str, gender: str = "male") -> str:
    """Get the appropriate Edge TTS voice for a language and gender.

    Args:
        language: ISO 639-1 language code.
        gender: 'male' or 'female'.

    Returns:
        Edge TTS voice identifier string.
    """
    voices = EDGE_VOICE_MAP.get(language, EDGE_VOICE_MAP["en"])
    return voices.get(gender, voices.get("male", "en-US-GuyNeural"))


def _run_async(coro):
    """Run an async coroutine safely, handling existing event loops.

    This is critical for Gradio compatibility - Gradio runs its own
    asyncio event loop, so asyncio.run() will raise RuntimeError.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside an existing event loop (e.g., Gradio).
        # Create a new thread to run the coroutine.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result(timeout=120)
    else:
        return asyncio.run(coro)


def generate_edge_tts(
    text: str,
    output_path: str,
    language: str = "en",
    gender: str = "male",
    speed: float = 1.0,
) -> str:
    """Generate speech using Microsoft Edge TTS.

    Args:
        text: Text to synthesize.
        output_path: Path for output audio file.
        language: Language code.
        gender: Voice gender ('male' or 'female').
        speed: Speech speed multiplier (1.0 = normal).

    Returns:
        Path to generated audio file.

    Raises:
        ImportError: If edge-tts is not installed.
        RuntimeError: If TTS generation fails.
    """
    try:
        import edge_tts
    except ImportError:
        raise ImportError(
            "edge-tts is required as fallback. Install with: pip install edge-tts"
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    voice = get_edge_voice(language, gender)
    rate_str = f"+{int((speed - 1) * 100)}%" if speed >= 1 else f"{int((speed - 1) * 100)}%"

    async def _generate():
        communicate = edge_tts.Communicate(text, voice, rate=rate_str)
        await communicate.save(output_path)

    logger.info("Edge TTS: voice=%s, language=%s, speed=%s", voice, language, rate_str)

    try:
        _run_async(_generate())
    except Exception as e:
        raise RuntimeError(f"Edge TTS generation failed: {e}") from e

    return output_path


def get_audio_duration(path: str) -> float:
    """Get audio file duration using ffprobe.

    Args:
        path: Path to audio file.

    Returns:
        Duration in seconds, or 0.0 on failure.
    """
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        info = json.loads(result.stdout)
        return float(info.get("format", {}).get("duration", 0))
    return 0.0


def is_edge_tts_available() -> bool:
    """Check if edge-tts package is installed."""
    try:
        import edge_tts
        return True
    except ImportError:
        return False
