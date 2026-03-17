"""
Qwen3-TTS engine integration.

Qwen3-TTS is a neural TTS model from Alibaba that supports
multiple languages with high-quality voice synthesis.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

from dubbing_studio.tts.engine import TTSEngine, TTSResult
from dubbing_studio.tts.audio_utils import apply_pitch_shift

logger = logging.getLogger(__name__)


class QwenTTS(TTSEngine):
    """
    Qwen3-TTS engine.

    Supports multiple languages with deep, natural-sounding voices.
    Best for: Hindi, German, Japanese, Korean, Chinese, Arabic, Russian,
    and other Asian/Middle Eastern languages.
    """

    def __init__(self):
        self._model = None
        self._processor = None

    @property
    def engine_name(self) -> str:
        return "qwen3-tts"

    @property
    def supported_languages(self) -> list[str]:
        return [
            "en", "zh", "ja", "ko", "hi", "ar", "de", "ru",
            "tr", "pl", "fi", "el", "cs", "ro", "hu", "th", "vi",
        ]

    def _load_model(self) -> None:
        """Load Qwen3-TTS model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError(
                "Transformers and torch are required for Qwen3-TTS. "
                "Install with: pip install transformers torch"
            )

        model_name = "Qwen/Qwen2.5-TTS"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading Qwen3-TTS model on %s", device)

        try:
            self._processor = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
            )
            logger.info("Qwen3-TTS model loaded successfully")
        except Exception as e:
            logger.warning("Failed to load Qwen3-TTS model: %s", e)
            self._model = None
            self._processor = None
            raise

    def generate_speech(
        self,
        text: str,
        output_path: str,
        language: str = "en",
        voice_id: Optional[str] = None,
        speed: float = 1.0,
        pitch: float = 1.0,
    ) -> TTSResult:
        """Generate speech using Qwen3-TTS."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Try using the model directly
        if self._model is not None or self._try_load():
            result = self._generate_with_model(text, output_path, language, speed)
        else:
            # Fallback to edge-tts for the same languages
            result = self._generate_with_fallback(text, output_path, language, speed, voice_id)

        if pitch != 1.0:
            apply_pitch_shift(result.audio_path, pitch)
            result.duration = self._get_audio_duration(result.audio_path)

        return result

    def _try_load(self) -> bool:
        """Try to load the model, return False if not available."""
        try:
            self._load_model()
            return True
        except Exception:
            return False

    def _generate_with_model(
        self,
        text: str,
        output_path: str,
        language: str,
        speed: float,
    ) -> TTSResult:
        """Generate speech using the loaded model."""
        import torch
        import numpy as np

        inputs = self._processor(text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=4096)

        audio_data = outputs[0].cpu().numpy()

        # Save to WAV
        import wave
        sample_rate = 24000

        with wave.open(output_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            audio_int16 = (audio_data * 32767).astype(np.int16)
            wf.writeframes(audio_int16.tobytes())

        duration = len(audio_data) / sample_rate

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
        voice_id: Optional[str] = None,
    ) -> TTSResult:
        """
        Fallback TTS using edge-tts (Microsoft Edge TTS).
        """
        try:
            import edge_tts
            import asyncio
        except ImportError:
            raise ImportError(
                "edge-tts is required as fallback. Install with: pip install edge-tts"
            )

        voice_map = {
            "en": "en-US-GuyNeural",
            "hi": "hi-IN-MadhurNeural",
            "zh": "zh-CN-YunxiNeural",
            "ja": "ja-JP-KeitaNeural",
            "ko": "ko-KR-InJoonNeural",
            "ar": "ar-SA-HamedNeural",
            "de": "de-DE-ConradNeural",
            "ru": "ru-RU-DmitryNeural",
            "tr": "tr-TR-AhmetNeural",
            "pl": "pl-PL-MarekNeural",
            "fi": "fi-FI-HarriNeural",
            "el": "el-GR-NestorasNeural",
            "cs": "cs-CZ-AntoninNeural",
            "ro": "ro-RO-EmilNeural",
            "hu": "hu-HU-TamasNeural",
            "th": "th-TH-NiwatNeural",
            "vi": "vi-VN-NamMinhNeural",
        }

        voice = voice_id or voice_map.get(language, "en-US-GuyNeural")
        rate_str = f"+{int((speed - 1) * 100)}%" if speed >= 1 else f"{int((speed - 1) * 100)}%"

        async def _generate():
            communicate = edge_tts.Communicate(text, voice, rate=rate_str)
            await communicate.save(output_path)

        asyncio.run(_generate())

        # Get duration
        duration = self._get_audio_duration(output_path)

        return TTSResult(
            audio_path=output_path,
            duration=duration,
            sample_rate=24000,
            text=text,
        )

    def _get_audio_duration(self, path: str) -> float:
        """Get audio file duration using ffprobe."""
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
            {"id": "qwen3-narrator-male", "name": "Qwen3 Male Narrator", "gender": "male", "languages": self.supported_languages},
            {"id": "qwen3-narrator-female", "name": "Qwen3 Female Narrator", "gender": "female", "languages": self.supported_languages},
        ]
        if language:
            voices = [v for v in voices if language in v["languages"]]
        return voices

    def is_available(self) -> bool:
        """Check if Qwen3-TTS dependencies are available."""
        try:
            import edge_tts
            return True
        except ImportError:
            pass
        try:
            from transformers import AutoModelForCausalLM
            return True
        except ImportError:
            return False

    def unload(self) -> None:
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
