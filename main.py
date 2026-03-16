#!/usr/bin/env python3
"""
Dubbing Studio - Main Entry Point

Usage:
    python main.py                  # Launch Gradio GUI
    python main.py desktop          # Launch desktop (PySide6) GUI
    python main.py --cli VIDEO      # CLI mode for single video
    python main.py --batch DIR      # CLI mode for batch processing
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dubbing_studio import __app_name__, __version__
from dubbing_studio.config import AppConfig, SUPPORTED_LANGUAGES


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def cli_single(args: argparse.Namespace) -> None:
    """Process a single video via CLI."""
    from dubbing_studio.pipeline import DubbingPipeline

    config = AppConfig.from_env()

    if args.api_key:
        config.translation.api_key = args.api_key
    if args.whisper_model:
        config.whisper.model_size = args.whisper_model
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.embed_subtitles:
        config.subtitle.embed_in_video = True
    if args.subtitle_format:
        config.subtitle.format = args.subtitle_format
    if args.narrator_style:
        config.voice.narrator_style = args.narrator_style

    config.setup_dirs()
    pipeline = DubbingPipeline(config)

    def progress_cb(stage: str, progress: float) -> None:
        bar_len = 30
        filled = int(bar_len * progress)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r  [{bar}] {progress*100:5.1f}% | {stage}", end="", flush=True)

    print(f"\n{__app_name__} v{__version__}")
    print(f"{'═' * 50}")
    print(f"Input:    {args.video}")
    print(f"Language: {SUPPORTED_LANGUAGES.get(args.language, args.language)}")
    print(f"Style:    {args.narrator_style or 'documentary'}")
    print(f"{'═' * 50}\n")

    result = pipeline.process_video(
        video_path=args.video,
        target_language=args.language,
        narrator_style=args.narrator_style or "documentary",
        progress_callback=progress_cb,
        output_dir=args.output_dir,
    )

    print(f"\n\n{'═' * 50}")
    print(f"Dubbing Complete!")
    print(f"  Source:     {result.source_language}")
    print(f"  Target:     {result.target_language}")
    print(f"  Segments:   {result.total_segments}")
    print(f"  Duration:   {result.total_duration:.1f}s")
    print(f"  Time:       {result.processing_time:.1f}s")
    print(f"  Video:      {result.output_video_path}")
    print(f"  Audio:      {result.output_audio_path}")
    for fmt, path in result.subtitle_paths.items():
        print(f"  Subtitles:  {path}")
    print(f"{'═' * 50}\n")


def cli_batch(args: argparse.Namespace) -> None:
    """Process multiple videos via CLI."""
    from dubbing_studio.batch.processor import BatchProcessor
    from dubbing_studio.pipeline import DubbingPipeline

    config = AppConfig.from_env()

    if args.api_key:
        config.translation.api_key = args.api_key
    if args.whisper_model:
        config.whisper.model_size = args.whisper_model
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.max_concurrent:
        config.batch.max_concurrent = args.max_concurrent

    config.setup_dirs()
    pipeline = DubbingPipeline(config)
    batch = BatchProcessor(config.batch)

    # Find video files
    video_dir = Path(args.batch_dir)
    video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}
    video_files = sorted([
        str(f) for f in video_dir.iterdir()
        if f.suffix.lower() in video_extensions
    ])

    if not video_files:
        print(f"No video files found in {args.batch_dir}")
        sys.exit(1)

    print(f"\n{__app_name__} v{__version__} - Batch Mode")
    print(f"{'═' * 50}")
    print(f"Directory: {args.batch_dir}")
    print(f"Videos:    {len(video_files)}")
    print(f"Language:  {SUPPORTED_LANGUAGES.get(args.language, args.language)}")
    print(f"Concurrent: {config.batch.max_concurrent}")
    print(f"{'═' * 50}\n")

    batch.add_videos(video_files, args.language, args.narrator_style or "documentary")

    def on_progress(prog):
        print(
            f"\r  Progress: {prog.completed_jobs}/{prog.total_jobs} completed, "
            f"{prog.failed_jobs} failed, "
            f"{prog.active_jobs} active",
            end="", flush=True,
        )

    jobs = batch.process_all(
        dubbing_function=pipeline.process_video_for_batch,
        progress_callback=on_progress,
    )

    print(f"\n\n{'═' * 50}")
    print("Batch Complete!")
    for job in jobs:
        status = "DONE" if job.status.value == "completed" else "FAIL"
        name = Path(job.video_path).name
        if job.output_path:
            print(f"  [{status}] {name} -> {job.output_path}")
        else:
            print(f"  [{status}] {name}: {job.error_message[:80]}")
    print(f"{'═' * 50}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description=f"{__app_name__} v{__version__} - Professional AI Documentary Dubbing Platform",
    )

    parser.add_argument(
        "--version", action="version",
        version=f"{__app_name__} v{__version__}",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command")

    # GUI command (default)
    gui_parser = subparsers.add_parser("gui", help="Launch GUI interface")
    gui_parser.add_argument(
        "--port", type=int, default=7860,
        help="Server port (default: 7860)",
    )
    gui_parser.add_argument(
        "--share", action="store_true",
        help="Create a public shareable link",
    )

    # CLI single video
    cli_parser = subparsers.add_parser("dub", help="Dub a single video")
    cli_parser.add_argument("video", help="Path to video file")
    cli_parser.add_argument(
        "-l", "--language", required=True,
        help=f"Target language code ({', '.join(SUPPORTED_LANGUAGES.keys())})",
    )
    cli_parser.add_argument(
        "-s", "--narrator-style",
        choices=["documentary", "cinematic", "calm", "storytelling"],
        default="documentary",
        help="Narrator style",
    )
    cli_parser.add_argument(
        "-o", "--output-dir", default="output",
        help="Output directory",
    )
    cli_parser.add_argument(
        "-k", "--api-key",
        help="Gemini API key (or set GEMINI_API_KEY env var)",
    )
    cli_parser.add_argument(
        "-w", "--whisper-model",
        choices=["tiny", "base", "medium", "large-v3"],
        default="base",
        help="Whisper model size",
    )
    cli_parser.add_argument(
        "--embed-subtitles", action="store_true",
        help="Burn subtitles into video",
    )
    cli_parser.add_argument(
        "--subtitle-format",
        choices=["srt", "vtt", "ass"],
        default="srt",
        help="Subtitle file format",
    )

    # Desktop GUI
    subparsers.add_parser("desktop", help="Launch standalone desktop GUI (PySide6)")

    # CLI batch
    batch_parser = subparsers.add_parser("batch", help="Batch process videos")
    batch_parser.add_argument("batch_dir", help="Directory containing videos")
    batch_parser.add_argument(
        "-l", "--language", required=True,
        help="Target language code",
    )
    batch_parser.add_argument(
        "-s", "--narrator-style",
        choices=["documentary", "cinematic", "calm", "storytelling"],
        default="documentary",
    )
    batch_parser.add_argument(
        "-o", "--output-dir", default="output",
    )
    batch_parser.add_argument(
        "-k", "--api-key",
        help="Gemini API key",
    )
    batch_parser.add_argument(
        "-w", "--whisper-model",
        choices=["tiny", "base", "medium", "large-v3"],
        default="base",
    )
    batch_parser.add_argument(
        "-c", "--max-concurrent",
        type=int, default=4,
        help="Maximum concurrent processing jobs",
    )

    args = parser.parse_args()
    setup_logging(getattr(args, "verbose", False))

    if args.command == "dub":
        cli_single(args)
    elif args.command == "batch":
        cli_batch(args)
    elif args.command == "desktop":
        from desktop_app import main as desktop_main
        desktop_main()
    else:
        # Default: launch GUI
        from app import create_ui
        app = create_ui()
        port = getattr(args, "port", 7860)
        share = getattr(args, "share", False)
        app.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=share,
            show_error=True,
        )


if __name__ == "__main__":
    main()
