"""Speech recognition and analysis modules."""

from dubbing_studio.speech.recognizer import (
    SpeechRecognizer,
    TranscriptionResult,
    TranscriptionSegment,
    WordTimestamp,
)
from dubbing_studio.speech.analyzer import NarrationAnalyzer, NarrationStyle

__all__ = [
    "SpeechRecognizer",
    "TranscriptionResult",
    "TranscriptionSegment",
    "WordTimestamp",
    "NarrationAnalyzer",
    "NarrationStyle",
]
