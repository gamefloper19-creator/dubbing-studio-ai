"""
Dubbing Studio - Professional AI Documentary Dubbing Platform

An automated pipeline for converting videos into another language
with natural narration and accurate timing.

Pipeline stages:
1. Audio extraction from video
2. Audio cleaning and normalization
3. Silence-based segmentation
4. Whisper speech recognition with word-level timestamps
5. Narration style analysis
6. Semantic translation via Google Gemini
7. Multi-engine TTS with automatic voice selection
8. Speech timing alignment (<300ms tolerance)
9. Background audio preservation with ducking
10. Subtitle generation (SRT, VTT, ASS)
11. Final video rendering via FFmpeg
"""

__version__ = "2.0.0"
__app_name__ = "Dubbing Studio"
