# Dubbing Studio v2.0

**Professional AI Documentary Dubbing Platform**

Automatically convert videos into another language with natural narration and accurate timing. Designed for documentary and voiceover-style content — no lip-sync required.

## Pipeline

```
Upload Video → Audio Extraction → Audio Cleaning → Segmentation →
Speech Recognition → Narration Analysis → Translation →
Voice Selection → Speech Generation → Timing Alignment →
Background Mixing → Subtitle Generation → Video Rendering
```

## Features

- **Smart Translation** — Uses Google Gemini for natural, storytelling-quality translation (not word-for-word)
- **Multiple TTS Engines** — Qwen3-TTS, Chatterbox, LuxTTS with automatic voice selection per language
- **Timing Alignment** — Generated speech matches original timing within 300ms
- **Background Audio Mixing** — Automatic ducking keeps original ambience
- **Batch Processing** — Process up to 25 videos simultaneously
- **24 Languages** — Full support for major world languages
- **Subtitle Generation** — SRT, VTT, and ASS formats
- **Hardware Optimization** — Auto-detects GPU/CPU and selects optimal models
- **Export Options** — MP4 video, WAV audio, MP3 audio, embedded subtitles

## Quick Start

### Prerequisites

- Python 3.10+
- FFmpeg (system install)

### Installation

```bash
# Install core dependencies
pip install -r requirements.txt

# For GPU acceleration (optional)
pip install torch torchaudio

# For native TTS engines (optional)
pip install chatterbox-tts    # Chatterbox TTS
pip install TTS               # LuxTTS (Coqui TTS)
```

### Launch GUI

```bash
python app.py
# Opens at http://localhost:7860
```

### CLI Usage

```bash
# Single video
python main.py dub video.mp4 -l hi -s documentary -k YOUR_GEMINI_KEY

# Batch processing
python main.py batch ./videos/ -l es -c 4 -k YOUR_GEMINI_KEY
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Google Gemini API key for translation |
| `WHISPER_MODEL` | Whisper model size (tiny/base/medium/large-v3) |
| `DUBBING_OUTPUT_DIR` | Output directory (default: output) |
| `DUBBING_TEMP_DIR` | Temp directory (default: temp) |

## Voice Engine Selection

| Language | Engine | Style |
|----------|--------|-------|
| Hindi | Qwen3-TTS | Deep documentary narrator |
| Spanish | Chatterbox | Cinematic narrator |
| French | LuxTTS | Calm neutral narrator |
| English | Chatterbox | Professional storytelling |

All engines fall back to Microsoft Edge TTS when native models are unavailable.

## Architecture

```
dubbing_studio/
├── audio/          # Extraction, cleaning, segmentation, mixing
├── speech/         # Whisper recognition, narration analysis
├── translation/    # Gemini semantic translation
├── tts/            # Voice engines (Qwen3, Chatterbox, LuxTTS)
├── timing/         # Speech timing alignment (<300ms deviation)
├── subtitle/       # SRT, VTT, ASS generation
├── video/          # Final video rendering
├── batch/          # Batch queue processing
├── hardware/       # GPU/CPU detection and optimization
├── export/         # Multi-format export
├── config.py       # Configuration management
└── pipeline.py     # Main pipeline orchestrator
```

## License

MIT
