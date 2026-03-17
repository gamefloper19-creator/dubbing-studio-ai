# Dubbing Studio — Build & Development Guide

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the web app (opens at http://localhost:7860)
python app.py

# 3. CLI usage
python main.py dub video.mp4 -l hi -s documentary -k YOUR_GEMINI_KEY
```

## Building the Windows Executable

```bash
# Produces dist/DubbingStudio/DubbingStudio.exe
python build.py
```

### Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.10+ | [python.org](https://python.org) |
| FFmpeg | Any | [ffmpeg.org](https://ffmpeg.org/download.html) — must be on PATH |
| PyInstaller | 6.0+ | `pip install pyinstaller` |

### Build Output

```
dist/
└── DubbingStudio/
    ├── DubbingStudio.exe   ← main executable (~31 MB)
    └── _internal/          ← bundled runtime files
```

> **Note**: FFmpeg must also be installed on the target Windows machine.
> The executable starts a local Gradio web server and opens your browser.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Required for translation (Google Gemini) |
| `WHISPER_MODEL` | Whisper model size: tiny/base/small/medium/large |
| `DUBBING_OUTPUT_DIR` | Output directory (default: ./output) |
| `DUBBING_TEMP_DIR` | Temp directory (default: ./temp) |

## Architecture

```
dubbing_studio/
├── audio/        # Extraction, cleaning, segmentation, mixing
├── speech/       # Whisper recognition, narration analysis
├── translation/  # Gemini semantic translation
├── tts/          # Voice engines (Qwen3, Chatterbox, LuxTTS, Edge TTS)
├── timing/       # Speech timing alignment (<300ms deviation)
├── subtitle/     # SRT, VTT, ASS generation
├── video/        # Final video rendering
├── batch/        # Batch queue processing
├── hardware/     # GPU/CPU detection and optimization
├── models/       # Model manager (auto-download)
├── export/       # Multi-format export
├── config.py     # Configuration management
└── pipeline.py   # Main pipeline orchestrator
```

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## Pipeline

```
Upload Video
  → Audio Extraction
  → Audio Cleaning
  → Segmentation
  → Speech Recognition (Whisper)
  → Narration Analysis
  → Translation (Gemini)
  → Voice Selection
  → Speech Generation (TTS)
  → Timing Alignment
  → Background Mixing
  → Subtitle Generation
  → Final Video Rendering
```

## Verified Build Environment

- ✅ FFmpeg 8.0
- ✅ Python 3.x (all 35 modules import cleanly)
- ✅ All `requirements.txt` dependencies installed
- ✅ `DubbingStudio.exe` builds successfully (31.1 MB)
