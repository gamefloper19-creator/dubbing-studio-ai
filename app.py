"""
Dubbing Studio - Gradio Web Interface

Professional AI Documentary Dubbing Platform GUI.
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

import gradio as gr

from dubbing_studio import __app_name__, __version__
from dubbing_studio.batch.processor import BatchProcessor, JobStatus
from dubbing_studio.config import AppConfig, SUPPORTED_LANGUAGES
from dubbing_studio.hardware.optimizer import HardwareOptimizer
from dubbing_studio.pipeline import DubbingPipeline, PIPELINE_STAGES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Global state ──
_config: Optional[AppConfig] = None
_pipeline: Optional[DubbingPipeline] = None
_batch_processor: Optional[BatchProcessor] = None
_hardware_info = None


def get_config() -> AppConfig:
    """Get or create app configuration."""
    global _config
    if _config is None:
        _config = AppConfig.from_env()
        _config.setup_dirs()
    return _config


def get_pipeline() -> DubbingPipeline:
    """Get or create the dubbing pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = DubbingPipeline(get_config())
    return _pipeline


def get_hardware_info():
    """Get hardware information."""
    global _hardware_info
    if _hardware_info is None:
        optimizer = HardwareOptimizer()
        _hardware_info = optimizer.detect_hardware()
    return _hardware_info


# ── Language options ──
LANGUAGE_CHOICES = [(name, code) for code, name in sorted(SUPPORTED_LANGUAGES.items(), key=lambda x: x[1])]
NARRATOR_STYLES = ["documentary", "cinematic", "calm", "storytelling"]
WHISPER_MODELS = ["auto", "tiny", "base", "small", "medium", "large-v3"]
SUBTITLE_FORMATS = ["srt", "vtt", "ass"]


# ── Processing functions ──

def process_single_video(
    video_file,
    target_language: str,
    narrator_style: str,
    whisper_model: str,
    embed_subtitles: bool,
    subtitle_format: str,
    bg_volume: float,
    gemini_api_key: str,
    progress=gr.Progress(track_tqdm=True),
):
    """Process a single video through the full dubbing pipeline.

    Pipeline stages:
    video -> audio extraction -> segmentation -> speech recognition ->
    translation -> TTS narration -> timing alignment -> background mixing ->
    final video render.
    """
    if video_file is None:
        return None, None, None, "Please upload a video file."

    if not gemini_api_key and not os.environ.get("GEMINI_API_KEY"):
        return None, None, None, "Please provide a Gemini API key for translation."

    # Update config
    config = get_config()
    if gemini_api_key:
        config.translation.api_key = gemini_api_key
    config.whisper.model_size = whisper_model
    config.subtitle.embed_in_video = embed_subtitles
    config.subtitle.format = subtitle_format
    config.mixing.background_volume = bg_volume / 100.0
    config.voice.narrator_style = narrator_style

    # Reinitialize pipeline with updated config
    global _pipeline
    _pipeline = DubbingPipeline(config)

    pipeline = get_pipeline()

    status_log = []
    start_time = time.time()

    def on_progress(stage: str, prog: float):
        pct = int(prog * 100)
        elapsed = time.time() - start_time
        status_log.append(f"[{pct:3d}%] {stage} ({elapsed:.1f}s)")
        progress(prog, desc=stage)

    try:
        video_path = video_file if isinstance(video_file, str) else video_file.name

        result = pipeline.process_video(
            video_path=video_path,
            target_language=target_language,
            narrator_style=narrator_style,
            progress_callback=on_progress,
        )

        # Build status summary with pipeline stage details
        total_elapsed = time.time() - start_time
        summary = (
            f"Dubbing Complete!\n"
            f"─────────────────────────────\n"
            f"Source Language: {result.source_language}\n"
            f"Target Language: {result.target_language}\n"
            f"Segments: {result.total_segments}\n"
            f"Duration: {result.total_duration:.1f}s\n"
            f"Processing Time: {total_elapsed:.1f}s\n"
            f"Whisper Model: {whisper_model}\n"
        )
        if result.narration_style:
            summary += (
                f"Narrator: {result.narration_style.gender}, "
                f"{result.narration_style.tone}, "
                f"{result.narration_style.pacing} pace\n"
            )
        summary += f"\n─── Pipeline Log ───\n" + "\n".join(status_log[-15:])

        # Return outputs
        subtitle_file = None
        for fmt, path in result.subtitle_paths.items():
            if Path(path).exists():
                subtitle_file = path
                break

        return (
            result.output_video_path,
            result.output_audio_path,
            subtitle_file,
            summary,
        )

    except Exception as e:
        error_msg = f"Error: {str(e)}\n\nLog:\n" + "\n".join(status_log[-10:])
        logger.error("Processing failed: %s", e, exc_info=True)
        return None, None, None, error_msg


def process_batch_videos(
    video_files,
    target_language: str,
    narrator_style: str,
    whisper_model: str,
    embed_subtitles: bool,
    subtitle_format: str,
    bg_volume: float,
    gemini_api_key: str,
    max_concurrent: int,
    progress=gr.Progress(track_tqdm=True),
):
    """Process multiple videos in batch with per-video progress tracking.

    Supports up to 25 videos with queue management and ETA estimation.
    """
    if not video_files:
        return "No videos uploaded."

    if len(video_files) > 25:
        return "Maximum 25 videos allowed per batch. Please reduce your selection."

    if not gemini_api_key and not os.environ.get("GEMINI_API_KEY"):
        return "Please provide a Gemini API key for translation."

    # Update config
    config = get_config()
    if gemini_api_key:
        config.translation.api_key = gemini_api_key
    config.whisper.model_size = whisper_model
    config.subtitle.embed_in_video = embed_subtitles
    config.subtitle.format = subtitle_format
    config.mixing.background_volume = bg_volume / 100.0
    config.voice.narrator_style = narrator_style
    config.batch.max_concurrent = min(max_concurrent, 25)

    global _pipeline, _batch_processor
    _pipeline = DubbingPipeline(config)
    pipeline = get_pipeline()

    # Setup batch processor
    batch = BatchProcessor(config.batch)
    _batch_processor = batch

    video_paths = []
    for vf in video_files:
        path = vf if isinstance(vf, str) else vf.name
        video_paths.append(path)

    try:
        batch.add_videos(video_paths, target_language, narrator_style)
    except (ValueError, FileNotFoundError) as e:
        return f"Error adding videos: {e}"

    batch_start = time.time()
    status_lines = [
        f"Batch Processing: {len(video_paths)} videos",
        f"Target Language: {SUPPORTED_LANGUAGES.get(target_language, target_language)}",
        f"Whisper Model: {whisper_model}",
        f"Max Concurrent: {config.batch.max_concurrent}",
        "─" * 50,
    ]

    def on_batch_progress(batch_progress):
        elapsed = time.time() - batch_start
        progress(
            batch_progress.overall_progress,
            desc=(
                f"Batch: {batch_progress.completed_jobs}/{batch_progress.total_jobs} done "
                f"({elapsed:.0f}s elapsed)"
            ),
        )

    try:
        jobs = batch.process_all(
            dubbing_function=pipeline.process_video_for_batch,
            progress_callback=on_batch_progress,
        )

        total_elapsed = time.time() - batch_start

        # Build per-video results summary
        status_lines.append("Per-Video Results:")
        for i, job in enumerate(jobs, 1):
            video_name = Path(job.video_path).name
            if job.status == JobStatus.COMPLETED:
                elapsed = job.end_time - job.start_time
                status_lines.append(
                    f"  {i:2d}. [DONE] {video_name} ({elapsed:.1f}s)"
                )
                if job.output_path:
                    status_lines.append(f"      Output: {job.output_path}")
            else:
                status_lines.append(
                    f"  {i:2d}. [FAIL] {video_name}"
                )
                if job.error_message:
                    status_lines.append(f"      Error: {job.error_message[:120]}")

        batch_info = batch.get_progress()
        status_lines.append("─" * 50)
        status_lines.append(
            f"Summary: {batch_info.completed_jobs} completed, "
            f"{batch_info.failed_jobs} failed, "
            f"Total time: {total_elapsed:.1f}s"
        )

        return "\n".join(status_lines)

    except Exception as e:
        logger.error("Batch processing failed: %s", e, exc_info=True)
        return f"Batch processing failed: {str(e)}"


def get_system_info():
    """Get system hardware information for display."""
    try:
        info = get_hardware_info()
        lines = [
            f"Platform: {info.platform}",
            f"GPU: {'Yes' if info.has_gpu else 'No'} ({info.gpu_name})",
        ]
        if info.has_gpu:
            lines.append(f"GPU Memory: {info.gpu_memory_mb} MB")
        lines.extend([
            f"CPU Cores: {info.cpu_count}",
            f"RAM: {info.ram_gb:.1f} GB",
            f"Recommended Whisper: {info.recommended_whisper_model}",
            f"Recommended Batch Size: {info.recommended_batch_size}",
        ])
        return "\n".join(lines)
    except Exception as e:
        return f"Hardware detection error: {e}"


# ── Build Gradio Interface ──

def create_ui() -> gr.Blocks:
    """Create the Gradio web interface."""

    with gr.Blocks(
        title=f"{__app_name__} v{__version__}",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        ),
        css="""
        .main-header {
            text-align: center;
            margin-bottom: 20px;
        }
        .stage-info {
            font-family: monospace;
            font-size: 13px;
            white-space: pre-wrap;
        }
        .batch-status {
            font-family: monospace;
            font-size: 12px;
            white-space: pre-wrap;
        }
        .pipeline-info {
            background: #f0f4f8;
            padding: 10px;
            border-radius: 8px;
            margin-top: 8px;
        }
        """,
    ) as app:

        gr.Markdown(
            f"""
            # {__app_name__} v{__version__}
            ### Professional AI Documentary Dubbing Platform
            *Transform videos into any language with natural narration and accurate timing*
            """,
            elem_classes=["main-header"],
        )

        with gr.Tabs():

            # ── Tab 1: Single Video ──
            with gr.Tab("Single Video Dubbing", id="single"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Upload & Settings")

                        video_input = gr.Video(
                            label="Upload Video",
                            sources=["upload"],
                        )

                        target_lang = gr.Dropdown(
                            choices=LANGUAGE_CHOICES,
                            value="hi",
                            label="Target Language",
                            info="Select the language to dub into",
                        )

                        narrator_style_input = gr.Dropdown(
                            choices=NARRATOR_STYLES,
                            value="documentary",
                            label="Narrator Style",
                            info="Choose narration style",
                        )

                        with gr.Accordion("Advanced Settings", open=False):
                            whisper_model_input = gr.Dropdown(
                                choices=WHISPER_MODELS,
                                value="auto",
                                label="Whisper Model",
                                info="'auto' selects the best model for your hardware",
                            )

                            embed_subs = gr.Checkbox(
                                value=False,
                                label="Embed Subtitles in Video",
                            )

                            sub_format = gr.Dropdown(
                                choices=SUBTITLE_FORMATS,
                                value="srt",
                                label="Subtitle Format",
                            )

                            bg_volume = gr.Slider(
                                minimum=0,
                                maximum=50,
                                value=15,
                                step=1,
                                label="Background Audio Volume (%)",
                            )

                            gemini_key = gr.Textbox(
                                label="Gemini API Key",
                                type="password",
                                placeholder="Enter your Google Gemini API key",
                                info="Required for translation. Can also be set via GEMINI_API_KEY env var.",
                            )

                        process_btn = gr.Button(
                            "Start Dubbing",
                            variant="primary",
                            size="lg",
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### Output")

                        status_output = gr.Textbox(
                            label="Status",
                            lines=10,
                            interactive=False,
                            elem_classes=["stage-info"],
                        )

                        video_output = gr.Video(
                            label="Dubbed Video",
                        )

                        audio_output = gr.Audio(
                            label="Dubbed Audio",
                        )

                        subtitle_output = gr.File(
                            label="Subtitles",
                        )

                process_btn.click(
                    fn=process_single_video,
                    inputs=[
                        video_input,
                        target_lang,
                        narrator_style_input,
                        whisper_model_input,
                        embed_subs,
                        sub_format,
                        bg_volume,
                        gemini_key,
                    ],
                    outputs=[
                        video_output,
                        audio_output,
                        subtitle_output,
                        status_output,
                    ],
                )

            # ── Tab 2: Batch Processing ──
            with gr.Tab("Batch Dubbing", id="batch"):
                gr.Markdown(
                    "Upload up to **25 videos** for batch dubbing. "
                    "Videos are processed concurrently with per-video progress tracking."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Batch Upload")

                        batch_videos = gr.File(
                            label="Upload Videos (drag & drop up to 25)",
                            file_count="multiple",
                            file_types=["video"],
                        )

                        batch_target_lang = gr.Dropdown(
                            choices=LANGUAGE_CHOICES,
                            value="hi",
                            label="Target Language",
                        )

                        batch_narrator_style = gr.Dropdown(
                            choices=NARRATOR_STYLES,
                            value="documentary",
                            label="Narrator Style",
                        )

                        max_concurrent = gr.Slider(
                            minimum=1,
                            maximum=25,
                            value=4,
                            step=1,
                            label="Max Concurrent Jobs",
                            info="Number of videos to process simultaneously (up to 25)",
                        )

                        with gr.Accordion("Advanced Settings", open=False):
                            batch_whisper = gr.Dropdown(
                                choices=WHISPER_MODELS,
                                value="auto",
                                label="Whisper Model",
                                info="'auto' selects based on hardware",
                            )

                            batch_embed_subs = gr.Checkbox(
                                value=False,
                                label="Embed Subtitles",
                            )

                            batch_sub_format = gr.Dropdown(
                                choices=SUBTITLE_FORMATS,
                                value="srt",
                                label="Subtitle Format",
                            )

                            batch_bg_volume = gr.Slider(
                                minimum=0,
                                maximum=50,
                                value=15,
                                step=1,
                                label="Background Volume (%)",
                            )

                            batch_gemini_key = gr.Textbox(
                                label="Gemini API Key",
                                type="password",
                                placeholder="Gemini API key",
                            )

                        batch_btn = gr.Button(
                            "Start Batch Dubbing",
                            variant="primary",
                            size="lg",
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### Batch Results")

                        batch_status = gr.Textbox(
                            label="Batch Status & Per-Video Progress",
                            lines=25,
                            interactive=False,
                            elem_classes=["batch-status"],
                        )

                batch_btn.click(
                    fn=process_batch_videos,
                    inputs=[
                        batch_videos,
                        batch_target_lang,
                        batch_narrator_style,
                        batch_whisper,
                        batch_embed_subs,
                        batch_sub_format,
                        batch_bg_volume,
                        batch_gemini_key,
                        max_concurrent,
                    ],
                    outputs=[batch_status],
                )

            # ── Tab 3: System Info ──
            with gr.Tab("System Info", id="system"):
                gr.Markdown("### Hardware & System Information")

                sys_info = gr.Textbox(
                    label="System Details",
                    lines=10,
                    interactive=False,
                    value=get_system_info(),
                )

                refresh_btn = gr.Button("Refresh", size="sm")
                refresh_btn.click(fn=get_system_info, outputs=[sys_info])

                gr.Markdown(
                    """
                    ### Full Dubbing Pipeline
                    The pipeline processes each video through these stages:
                    """
                    + "\n".join(f"{i+1}. **{stage}**" for i, stage in enumerate(PIPELINE_STAGES))
                    + """

                    ### Supported Languages (24)
                    """
                    + ", ".join(f"{name} ({code})" for code, name in sorted(SUPPORTED_LANGUAGES.items(), key=lambda x: x[1]))
                    + """

                    ### Voice Engines (with automatic fallback)
                    - **Qwen3-TTS** — Best for Asian and Middle Eastern languages
                    - **Chatterbox** — Best for cinematic English, Spanish, Portuguese, Italian
                    - **LuxTTS** — Best for calm European narration (French, Dutch, Swedish, Danish)
                    - **Edge TTS** — Universal fallback for all languages

                    ### Key Features
                    - **Auto Whisper Model Selection**: Automatically picks the best model for your GPU/CPU
                    - **Timing Alignment**: Generated narration matches original duration (< 300ms deviation)
                    - **Background Audio Ducking**: Background audio is smoothly reduced during narration
                    - **Batch Processing**: Process up to 25 videos concurrently with progress tracking
                    """
                )

        gr.Markdown(
            f"""
            ---
            *{__app_name__} v{__version__} — Professional AI Documentary Dubbing Platform*
            """,
        )

    return app


def main():
    """Launch the Dubbing Studio application."""
    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
