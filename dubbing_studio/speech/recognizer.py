"""
Speech recognition using OpenAI Whisper.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from dubbing_studio.audio.segmenter import AudioSegment
from dubbing_studio.config import WhisperConfig
from dubbing_studio.hardware.optimizer import HardwareOptimizer

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionSegment:
    """A transcribed segment with timing information."""
    segment_id: str
    start_time: float
    end_time: float
    text: str
    language: str
    confidence: float = 0.0


@dataclass
class TranscriptionResult:
    """Complete transcription result."""
    segments: list[TranscriptionSegment]
    detected_language: str
    full_text: str


class SpeechRecognizer:
    """Speech recognition using OpenAI Whisper (local)."""

    def __init__(self, config: Optional[WhisperConfig] = None):
        self.config = config or WhisperConfig()
        self._model = None
        self._optimizer = HardwareOptimizer()

    def _load_model(self) -> None:
        """Load Whisper model with hardware-appropriate settings."""
        try:
            import whisper
        except ImportError:
            raise ImportError(
                "OpenAI Whisper is not installed. "
                "Install it with: pip install openai-whisper"
            )

        device = self.config.device
        if device == "auto":
            device = "cuda" if self._optimizer.has_gpu() else "cpu"

        model_size = self.config.model_size

        # Auto-select model based on hardware
        if model_size == "auto":
            if self._optimizer.has_gpu():
                gpu_mem = self._optimizer.get_gpu_memory()
                if gpu_mem and gpu_mem > 8000:
                    model_size = "large-v3"
                elif gpu_mem and gpu_mem > 4000:
                    model_size = "medium"
                else:
                    model_size = "base"
            else:
                model_size = "base"

        logger.info(
            "Loading Whisper model '%s' on device '%s'",
            model_size, device,
        )
        self._model = whisper.load_model(model_size, device=device)
        logger.info("Whisper model loaded successfully")

    def transcribe_audio(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to audio file.
            language: Optional language code (None = auto-detect).

        Returns:
            TranscriptionResult with segments and detected language.
        """
        if self._model is None:
            self._load_model()

        options = {
            "beam_size": self.config.beam_size,
            "word_timestamps": True,
            "verbose": False,
        }

        if language or self.config.language:
            options["language"] = language or self.config.language

        logger.info("Transcribing audio: %s", audio_path)
        result = self._model.transcribe(audio_path, **options)

        segments = []
        for i, seg in enumerate(result.get("segments", [])):
            segments.append(TranscriptionSegment(
                segment_id=f"{i + 1:03d}",
                start_time=seg["start"],
                end_time=seg["end"],
                text=seg["text"].strip(),
                language=result.get("language", "unknown"),
                confidence=seg.get("avg_logprob", 0.0),
            ))

        detected_lang = result.get("language", "unknown")
        full_text = result.get("text", "").strip()

        logger.info(
            "Transcription complete: %d segments, language=%s",
            len(segments), detected_lang,
        )

        return TranscriptionResult(
            segments=segments,
            detected_language=detected_lang,
            full_text=full_text,
        )

    def transcribe_segments(
        self,
        audio_segments: list[AudioSegment],
        language: Optional[str] = None,
    ) -> list[TranscriptionSegment]:
        """
        Transcribe a list of audio segments.

        Args:
            audio_segments: List of AudioSegment objects with file paths.
            language: Optional language code.

        Returns:
            List of TranscriptionSegment objects.
        """
        if self._model is None:
            self._load_model()

        all_transcriptions = []

        for seg in audio_segments:
            if not seg.file_path:
                logger.warning("Segment %s has no file path, skipping", seg.segment_id)
                continue

            result = self.transcribe_audio(seg.file_path, language)

            for tseg in result.segments:
                # Adjust timing to absolute position
                all_transcriptions.append(TranscriptionSegment(
                    segment_id=seg.segment_id,
                    start_time=seg.start_time + tseg.start_time,
                    end_time=seg.start_time + tseg.end_time,
                    text=tseg.text,
                    language=tseg.language,
                    confidence=tseg.confidence,
                ))

        return all_transcriptions

    def unload_model(self) -> None:
        """Unload the Whisper model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            logger.info("Whisper model unloaded")

            # Try to free GPU memory
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
