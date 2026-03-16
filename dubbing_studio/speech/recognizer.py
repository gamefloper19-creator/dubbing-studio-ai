"""
Speech recognition using OpenAI Whisper.

Provides timestamped transcription segments with word-level timing,
automatic language detection, and hardware-optimized model selection.

Features:
- Automatic model selection based on available hardware (GPU memory, RAM)
- GPU acceleration with automatic compute type selection
- VAD pre-filtering for cleaner transcription
- Graceful fallback when models are unavailable
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dubbing_studio.audio.segmenter import AudioSegment
from dubbing_studio.config import WhisperConfig
from dubbing_studio.hardware.optimizer import HardwareOptimizer

logger = logging.getLogger(__name__)

# Model sizes ordered by quality (and resource requirements)
WHISPER_MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3"]

# Approximate VRAM requirements in MB for each model
WHISPER_VRAM_REQUIREMENTS = {
    "tiny": 400,
    "base": 500,
    "small": 1000,
    "medium": 2500,
    "large-v3": 5000,
}

# Approximate RAM requirements in GB for each model on CPU
WHISPER_RAM_REQUIREMENTS = {
    "tiny": 1.0,
    "base": 1.5,
    "small": 3.0,
    "medium": 5.0,
    "large-v3": 10.0,
}


@dataclass
class WordTimestamp:
    """A single word with precise timing."""
    word: str
    start: float
    end: float
    probability: float = 0.0


@dataclass
class TranscriptionSegment:
    """A transcribed segment with timing information."""
    segment_id: str
    start_time: float
    end_time: float
    text: str
    language: str
    confidence: float = 0.0
    words: list[WordTimestamp] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    """Complete transcription result."""
    segments: list[TranscriptionSegment]
    detected_language: str
    full_text: str
    duration: float = 0.0


class SpeechRecognizer:
    """Speech recognition using OpenAI Whisper (local).

    Supports automatic model selection based on hardware capabilities,
    GPU acceleration, and VAD pre-filtering.
    """

    def __init__(self, config: Optional[WhisperConfig] = None):
        self.config = config or WhisperConfig()
        self._model = None
        self._model_name: Optional[str] = None
        self._device: Optional[str] = None
        self._optimizer = HardwareOptimizer()

    def _select_model_size(self) -> str:
        """Automatically select the best Whisper model for the available hardware.

        Selection logic:
        - GPU with >8GB VRAM: large-v3 (best quality)
        - GPU with >4GB VRAM: medium
        - GPU with >2GB VRAM: small
        - GPU with <2GB or CPU with >8GB RAM: base
        - Low-resource CPU: tiny

        Returns:
            Model size string.
        """
        if self._optimizer.has_gpu():
            gpu_mem = self._optimizer.get_gpu_memory()
            if gpu_mem is not None:
                for model_name in reversed(WHISPER_MODEL_SIZES):
                    if gpu_mem >= WHISPER_VRAM_REQUIREMENTS[model_name] * 1.5:
                        logger.info(
                            "Auto-selected Whisper model '%s' for GPU with %dMB VRAM",
                            model_name, gpu_mem,
                        )
                        return model_name
            return "base"
        else:
            # CPU mode: select based on RAM
            hw_info = self._optimizer.detect_hardware()
            ram_gb = hw_info.ram_gb
            for model_name in reversed(WHISPER_MODEL_SIZES):
                if ram_gb >= WHISPER_RAM_REQUIREMENTS[model_name] * 2.0:
                    logger.info(
                        "Auto-selected Whisper model '%s' for CPU with %.1fGB RAM",
                        model_name, ram_gb,
                    )
                    return model_name
            return "tiny"

    def _select_device(self) -> str:
        """Select compute device (cuda/cpu) based on availability."""
        device = self.config.device
        if device == "auto":
            if self._optimizer.has_gpu():
                device = "cuda"
                logger.info("Using CUDA GPU for Whisper inference")
            else:
                device = "cpu"
                logger.info("Using CPU for Whisper inference")
        return device

    def _load_model(self) -> None:
        """Load Whisper model with hardware-appropriate settings.

        Automatically selects model size and device when configured to 'auto'.
        Falls back to smaller models if the selected model fails to load.
        """
        try:
            import whisper
        except ImportError:
            raise ImportError(
                "OpenAI Whisper is not installed. "
                "Install it with: pip install openai-whisper"
            )

        device = self._select_device()
        model_size = self.config.model_size

        if model_size == "auto":
            model_size = self._select_model_size()

        # Try loading the selected model, fall back to smaller ones on failure
        models_to_try = [model_size]
        idx = WHISPER_MODEL_SIZES.index(model_size) if model_size in WHISPER_MODEL_SIZES else 1
        for fallback in reversed(WHISPER_MODEL_SIZES[:idx]):
            models_to_try.append(fallback)

        last_error = None
        for try_model in models_to_try:
            try:
                logger.info(
                    "Loading Whisper model '%s' on device '%s'",
                    try_model, device,
                )
                self._model = whisper.load_model(try_model, device=device)
                self._model_name = try_model
                self._device = device
                logger.info(
                    "Whisper model '%s' loaded successfully on %s",
                    try_model, device,
                )
                return
            except Exception as e:
                last_error = e
                logger.warning(
                    "Failed to load Whisper model '%s' on %s: %s",
                    try_model, device, e,
                )
                # If GPU loading fails, try CPU
                if device == "cuda":
                    try:
                        logger.info("Retrying '%s' on CPU", try_model)
                        self._model = whisper.load_model(try_model, device="cpu")
                        self._model_name = try_model
                        self._device = "cpu"
                        logger.info(
                            "Whisper model '%s' loaded on CPU (GPU fallback)",
                            try_model,
                        )
                        return
                    except Exception as cpu_err:
                        logger.warning("CPU fallback also failed: %s", cpu_err)
                        last_error = cpu_err

        raise RuntimeError(
            f"Failed to load any Whisper model. Last error: {last_error}. "
            f"Tried models: {models_to_try}"
        )

    @property
    def model_info(self) -> dict:
        """Get information about the currently loaded model."""
        return {
            "model_name": self._model_name,
            "device": self._device,
            "loaded": self._model is not None,
        }

    def transcribe_audio(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe an audio file with timestamped segments.

        Uses Whisper's built-in segment and word-level timestamps
        for precise timing alignment downstream.

        Args:
            audio_path: Path to audio file.
            language: Optional language code (None = auto-detect).

        Returns:
            TranscriptionResult with segments, word timestamps,
            and detected language.
        """
        if self._model is None:
            self._load_model()

        # Validate input file exists and is non-empty
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if audio_file.stat().st_size == 0:
            logger.warning("Audio file is empty: %s", audio_path)
            return TranscriptionResult(
                segments=[], detected_language="unknown", full_text="",
            )

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
            text = seg["text"].strip()
            if not text:
                continue

            # Extract word-level timestamps when available
            words = []
            for w in seg.get("words", []):
                words.append(WordTimestamp(
                    word=w.get("word", "").strip(),
                    start=w.get("start", seg["start"]),
                    end=w.get("end", seg["end"]),
                    probability=w.get("probability", 0.0),
                ))

            segments.append(TranscriptionSegment(
                segment_id=f"{i + 1:03d}",
                start_time=seg["start"],
                end_time=seg["end"],
                text=text,
                language=result.get("language", "unknown"),
                confidence=seg.get("avg_logprob", 0.0),
                words=words,
            ))

        detected_lang = result.get("language", "unknown")
        full_text = result.get("text", "").strip()

        # Calculate total audio duration from segments
        duration = 0.0
        if segments:
            duration = segments[-1].end_time

        logger.info(
            "Transcription complete: %d segments, language=%s, duration=%.1fs",
            len(segments), detected_lang, duration,
        )

        return TranscriptionResult(
            segments=segments,
            detected_language=detected_lang,
            full_text=full_text,
            duration=duration,
        )

    def transcribe_segments(
        self,
        audio_segments: list[AudioSegment],
        language: Optional[str] = None,
    ) -> list[TranscriptionSegment]:
        """
        Transcribe a list of audio segments.

        Each segment is transcribed individually, then timestamps are
        adjusted to absolute positions in the original timeline.

        Args:
            audio_segments: List of AudioSegment objects with file paths.
            language: Optional language code.

        Returns:
            List of TranscriptionSegment objects with absolute timestamps.
        """
        if self._model is None:
            self._load_model()

        all_transcriptions = []

        for seg in audio_segments:
            if not seg.file_path:
                logger.warning("Segment %s has no file path, skipping", seg.segment_id)
                continue

            if not Path(seg.file_path).exists():
                logger.warning("Segment file missing: %s", seg.file_path)
                continue

            try:
                result = self.transcribe_audio(seg.file_path, language)
            except Exception as e:
                logger.warning(
                    "Failed to transcribe segment %s: %s", seg.segment_id, e
                )
                continue

            if not result.segments:
                logger.warning(
                    "No speech detected in segment %s", seg.segment_id
                )
                continue

            for tseg in result.segments:
                # Adjust word timestamps to absolute position
                adjusted_words = [
                    WordTimestamp(
                        word=w.word,
                        start=seg.start_time + w.start,
                        end=seg.start_time + w.end,
                        probability=w.probability,
                    )
                    for w in tseg.words
                ]

                # Adjust timing to absolute position
                all_transcriptions.append(TranscriptionSegment(
                    segment_id=seg.segment_id,
                    start_time=seg.start_time + tseg.start_time,
                    end_time=seg.start_time + tseg.end_time,
                    text=tseg.text,
                    language=tseg.language,
                    confidence=tseg.confidence,
                    words=adjusted_words,
                ))

        logger.info(
            "Transcribed %d segments from %d audio chunks",
            len(all_transcriptions), len(audio_segments),
        )
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
