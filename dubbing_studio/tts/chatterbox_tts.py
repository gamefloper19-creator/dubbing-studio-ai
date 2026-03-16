"""
Chatterbox TTS engine integration.

Chatterbox provides high-quality, expressive voice synthesis
particularly suited for cinematic and storytelling narration.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

from dubbing_studio.tts.engine import TTSEngine, TTSResult

from dubbing_studio.tts.qwen_tts import _run_async, _convert_to_wav

logger = logging.getLogger(__name__)


class ChatterboxTTS(TTSEngine):
    """
    Chatterbox TTS engine.

    Best for: English, Spanish, Portuguese, Italian -
    cinematic and storytelling narration styles.
    """

    def __init__(self):
        self._model = None

    @property
    def engine_name(self) -> str:
        return "chatterbox"

    @property
    def supported_languages(self) -> list[str]:
        return ["en", "es", "pt", "it", "fr"]

    def _load_model(self) -> None:
        """Load Chatterbox TTS model."""
        try:
            from chatterbox.tts import ChatterboxTTS as CBModel
            import torch
        except ImportError:
            raise ImportError(
                "Chatterbox TTS is not installed. "
                "Install with: pip install chatterbox-tts"
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading Chatterbox TTS on %s", device)

        self._model = CBModel.from_pretrained(device=device)
        logger.info("Chatterbox TTS loaded successfully")

    def generate_speech(
        self,
        text: str,
        output_path: str,
        language: str = "en",
        voice_id: Optional[str] = None,
        speed: float = 1.0,
    ) -> TTSResult:
        """Generate speech using Chatterbox TTS."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Try native model first
        if self._model is not None or self._try_load():
            return self._generate_with_model(text, output_path, language, speed)

        # Fallback to edge-tts
        return self._generate_with_fallback(text, output_path, language, speed)

    def _try_load(self) -> bool:
        """Try to load model, return False if unavailable."""
        try:
            self._load_model()
            return True
        except Exception as e:
            logger.warning("Chatterbox model not available: %s", e)
            return False

    def _generate_with_model(
        self,
        text: str,
        output_path: str,
        language: str,
        speed: float,
    ) -> TTSResult:
        """Generate using native Chatterbox model."""
        import torch
        import torchaudio

        wav = self._model.generate(text)
        sample_rate = 24000

        # Apply speed adjustment if needed
        if speed != 1.0:
            effects = [["tempo", str(speed)]]
            wav, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
                wav.unsqueeze(0), sample_rate, effects
            )
            wav = wav.squeeze(0)

        torchaudio.save(output_path, wav.unsqueeze(0), sample_rate)

        duration = wav.shape[0] / sample_rate

        return TTSResult(
            audio_path=output_path,
            duration=duration,
            sample_rate=sample_rate,
            text=text,
        )

    def _generate_with_fallback(
        self,
        text: str,
        output_path: str,
        language: str,
        speed: float,
    ) -> TTSResult:
        """Fallback to edge-tts."""
        try:
            import edge_tts
            import asyncio
        except ImportError:
            raise ImportError(
                "edge-tts is required as fallback. Install with: pip install edge-tts"
            )

        voice_map = {
            "en": "en-US-ChristopherNeural",
            "es": "es-ES-AlvaroNeural",
            "pt": "pt-BR-AntonioNeural",
            "it": "it-IT-DiegoNeural",
            "fr": "fr-FR-HenriNeural",
        }

        voice = voice_map.get(language, "en-US-ChristopherNeural")
        rate_str = f"+{int((speed - 1) * 100)}%" if speed >= 1 else f"{int((speed - 1) * 100)}%"

        # edge-tts outputs MP3; save to temp then convert to WAV
        temp_mp3 = output_path + ".edgetts.mp3"

        async def _generate():
            communicate = edge_tts.Communicate(text, voice, rate=rate_str)
            await communicate.save(temp_mp3)

        _run_async(_generate())

        # Convert MP3 to WAV for pipeline compatibility
        _convert_to_wav(temp_mp3, output_path)

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
            {"id": "chatterbox-cinematic-male", "name": "Cinematic Male", "gender": "male", "style": "cinematic", "languages": ["en", "es", "pt", "it"]},
            {"id": "chatterbox-storyteller-male", "name": "Storyteller Male", "gender": "male", "style": "storytelling", "languages": ["en"]},
            {"id": "chatterbox-narrator-female", "name": "Narrator Female", "gender": "female", "style": "documentary", "languages": ["en", "es", "pt", "it"]},
        ]
        if language:
            voices = [v for v in voices if language in v["languages"]]
        return voices

    def is_available(self) -> bool:
        """Check if Chatterbox is available."""
        try:
            import edge_tts
            return True
        except ImportError:
            pass
        try:
            from chatterbox.tts import ChatterboxTTS
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
