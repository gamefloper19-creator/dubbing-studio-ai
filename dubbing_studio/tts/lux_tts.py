"""
LuxTTS engine integration.

LuxTTS provides calm, neutral voice synthesis
suitable for European language narration.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

from dubbing_studio.tts.engine import TTSEngine, TTSResult

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run an async coroutine, handling the case where an event loop is already running."""
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


class LuxTTS(TTSEngine):
    """
    LuxTTS engine.

    Best for: French, Dutch, Swedish, Danish -
    calm, neutral narration styles.
    """

    def __init__(self):
        self._model = None

    @property
    def engine_name(self) -> str:
        return "luxtts"

    @property
    def supported_languages(self) -> list[str]:
        return ["en", "fr", "nl", "sv", "da", "de"]

    def generate_speech(
        self,
        text: str,
        output_path: str,
        language: str = "en",
        voice_id: Optional[str] = None,
        speed: float = 1.0,
    ) -> TTSResult:
        """Generate speech using LuxTTS."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Try native model
        if self._model is not None or self._try_load():
            return self._generate_with_model(text, output_path, language, speed)

        # Fallback to edge-tts
        return self._generate_with_fallback(text, output_path, language, speed)

    def _try_load(self) -> bool:
        """Try loading the model."""
        try:
            self._load_model()
            return True
        except Exception as e:
            logger.warning("LuxTTS model not available: %s", e)
            return False

    def _load_model(self) -> None:
        """Load LuxTTS model."""
        try:
            # Try to import LuxTTS or compatible TTS library
            from TTS.api import TTS
            import torch
        except ImportError:
            raise ImportError(
                "Coqui TTS is not installed. "
                "Install with: pip install TTS"
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading LuxTTS (Coqui TTS) on %s", device)

        # Use XTTS-v2 as the backend for LuxTTS
        self._model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        logger.info("LuxTTS model loaded successfully")

    def _generate_with_model(
        self,
        text: str,
        output_path: str,
        language: str,
        speed: float,
    ) -> TTSResult:
        """Generate using native model."""
        self._model.tts_to_file(
            text=text,
            language=language,
            file_path=output_path,
            speed=speed,
        )

        duration = self._get_audio_duration(output_path)

        return TTSResult(
            audio_path=output_path,
            duration=duration,
            sample_rate=24000,
            text=text,
        )

    def _generate_with_fallback(
        self,
        text: str,
        output_path: str,
        language: str,
        speed: float,
    ) -> TTSResult:
        """Fallback to edge-tts for calm, neutral voices."""
        try:
            import edge_tts
            import asyncio
        except ImportError:
            raise ImportError(
                "edge-tts is required as fallback. Install with: pip install edge-tts"
            )

        voice_map = {
            "en": "en-GB-RyanNeural",
            "fr": "fr-FR-HenriNeural",
            "nl": "nl-NL-MaartenNeural",
            "sv": "sv-SE-MattiasNeural",
            "da": "da-DK-JeppeNeural",
            "de": "de-DE-KillianNeural",
        }

        voice = voice_map.get(language, "en-GB-RyanNeural")
        rate_str = f"+{int((speed - 1) * 100)}%" if speed >= 1 else f"{int((speed - 1) * 100)}%"

        async def _generate():
            communicate = edge_tts.Communicate(text, voice, rate=rate_str)
            await communicate.save(output_path)

        _run_async(_generate())

        duration = self._get_audio_duration(output_path)

        return TTSResult(
            audio_path=output_path,
            duration=duration,
            sample_rate=24000,
            text=text,
        )

    def _get_audio_duration(self, path: str) -> float:
        """Get audio file duration."""
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

    def list_voices(self, language: str = "") -> list[dict]:
        """List available voices."""
        voices = [
            {"id": "lux-calm-male", "name": "Calm Male Narrator", "gender": "male", "style": "calm", "languages": ["en", "fr", "nl", "sv", "da", "de"]},
            {"id": "lux-neutral-female", "name": "Neutral Female Narrator", "gender": "female", "style": "neutral", "languages": ["en", "fr", "nl", "sv", "da", "de"]},
        ]
        if language:
            voices = [v for v in voices if language in v["languages"]]
        return voices

    def is_available(self) -> bool:
        """Check if LuxTTS or fallback is available."""
        try:
            import edge_tts
            return True
        except ImportError:
            pass
        try:
            from TTS.api import TTS
            return True
        except ImportError:
            return False

    def unload(self) -> None:
        """Unload model."""
        if self._model is not None:
            del self._model
            self._model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
