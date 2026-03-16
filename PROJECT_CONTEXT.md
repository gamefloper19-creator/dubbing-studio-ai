# Project Context

## Overview

**Dubbing Studio v2.0** is a professional AI-powered documentary dubbing platform. It automatically converts video content from one language to another with natural narration and accurate timing. The platform is designed for documentary and voiceover-style content where lip-sync is not required.

## Architecture

The project follows a modular pipeline architecture with each stage handled by a dedicated module:

```
dubbing_studio/
├── audio/              # Audio processing modules
│   ├── extractor.py    # Extract audio tracks from video files
│   ├── cleaner.py      # Clean and enhance extracted audio
│   ├── segmenter.py    # Split audio into speech segments
│   └── mixer.py        # Mix dubbed audio with background audio
├── speech/             # Speech processing
│   ├── recognizer.py   # Whisper-based speech-to-text
│   └── analyzer.py     # Narration style analysis (tone, pacing, gender)
├── translation/        # Language translation
│   └── translator.py   # Google Gemini-based semantic translation
├── tts/                # Text-to-speech engines
│   ├── engine.py       # Base TTS engine interface
│   ├── qwen_tts.py     # Qwen3-TTS (Asian/Middle Eastern languages)
│   ├── chatterbox_tts.py # Chatterbox (cinematic English, Romance languages)
│   ├── lux_tts.py      # LuxTTS/Coqui (calm European narration)
│   └── voice_selector.py # Automatic engine selection per language
├── timing/             # Timing alignment
│   └── aligner.py      # Align generated speech to original timing (<300ms)
├── subtitle/           # Subtitle generation
│   └── generator.py    # Generate SRT, VTT, ASS subtitle files
├── video/              # Video output
│   └── renderer.py     # Render final video with dubbed audio
├── batch/              # Batch processing
│   └── processor.py    # Process up to 25 videos concurrently
├── hardware/           # Hardware optimization
│   └── optimizer.py    # Auto-detect GPU/CPU and select optimal models
├── export/             # Export handling
│   └── exporter.py     # Multi-format export (MP4, WAV, MP3)
├── config.py           # Centralized configuration management
└── pipeline.py         # Main pipeline orchestrator
```

## Entry Points

| File | Purpose |
|------|---------|
| `app.py` | Gradio web interface (default at `http://localhost:7860`) |
| `main.py` | CLI entry point with subcommands: `gui`, `dub`, `batch` |

## Pipeline Flow

```
Upload Video
  → Audio Extraction (FFmpeg)
  → Audio Cleaning (noise reduction, normalization)
  → Segmentation (split into speech segments)
  → Speech Recognition (OpenAI Whisper)
  → Narration Analysis (detect tone, pacing, gender)
  → Translation (Google Gemini - semantic, not word-for-word)
  → Voice Selection (auto-select best TTS engine per language)
  → Speech Generation (Qwen3 / Chatterbox / LuxTTS / Edge TTS fallback)
  → Timing Alignment (match original timing within 300ms)
  → Background Mixing (automatic ducking of original ambience)
  → Subtitle Generation (SRT / VTT / ASS)
  → Video Rendering (final output with dubbed audio)
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| Web UI | Gradio 4.x |
| Speech Recognition | OpenAI Whisper |
| Translation | Google Gemini API |
| TTS (Asian/ME) | Qwen3-TTS (Transformers) |
| TTS (Cinematic) | Chatterbox TTS |
| TTS (European) | LuxTTS (Coqui TTS) |
| TTS (Fallback) | Microsoft Edge TTS |
| Audio/Video | FFmpeg (system dependency) |
| Hardware Detection | psutil |
| Language | Python 3.10+ |

## Configuration

Configuration is managed via `dubbing_studio/config.py` with environment variable overrides:

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key for translation | (required) |
| `WHISPER_MODEL` | Whisper model size | `base` |
| `DUBBING_OUTPUT_DIR` | Output directory | `output` |
| `DUBBING_TEMP_DIR` | Temp directory | `temp` |

## Supported Languages

24 languages including: Arabic, Bengali, Chinese, Dutch, English, French, German, Greek, Gujarati, Hebrew, Hindi, Indonesian, Italian, Japanese, Kannada, Korean, Marathi, Polish, Portuguese, Russian, Spanish, Swedish, Tamil, Telugu, Thai, Turkish, Ukrainian, Urdu, Vietnamese.

## Key Design Decisions

1. **Semantic Translation**: Uses Gemini for natural storytelling translation rather than literal word-for-word translation, preserving documentary narrative quality.
2. **Multi-Engine TTS**: Different TTS engines are selected per language based on quality - no single engine handles all languages well.
3. **Edge TTS Fallback**: Microsoft Edge TTS serves as a universal fallback when native models are unavailable, ensuring the platform always works.
4. **No Lip-Sync**: Designed specifically for documentary/voiceover content where visual lip synchronization is not needed.
5. **Modular Pipeline**: Each processing stage is isolated, making it easy to swap or upgrade individual components.
