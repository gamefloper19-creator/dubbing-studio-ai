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
from dubbing_studio.models.manager import ModelManager
from dubbing_studio.pipeline import DubbingPipeline, PIPELINE_STAGES
from dubbing_studio.tts.voice_library import VoiceLibrary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Real-time log capture ──
_log_buffer: list[str] = []
_MAX_LOG_LINES = 200


class GUILogHandler(logging.Handler):
    """Capture log messages for the GUI log panel."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            _log_buffer.append(msg)
            if len(_log_buffer) > _MAX_LOG_LINES:
                _log_buffer.pop(0)
        except Exception:
            pass


# Attach the GUI log handler to the root logger
_gui_handler = GUILogHandler()
_gui_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
)
logging.getLogger().addHandler(_gui_handler)

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
WHISPER_MODELS = ["tiny", "base", "medium", "large-v3"]
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
    """Process a single video through the dubbing pipeline."""
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

    def on_progress(stage: str, prog: float):
        pct = int(prog * 100)
        status_log.append(f"[{pct:3d}%] {stage}")
        progress(prog, desc=stage)

    try:
        video_path = video_file if isinstance(video_file, str) else video_file.name

        result = pipeline.process_video(
            video_path=video_path,
            target_language=target_language,
            narrator_style=narrator_style,
            progress_callback=on_progress,
        )

        # Build status summary
        summary = (
            f"Dubbing Complete!\n"
            f"─────────────────────────────\n"
            f"Source Language: {result.source_language}\n"
            f"Target Language: {result.target_language}\n"
            f"Segments: {result.total_segments}\n"
            f"Duration: {result.total_duration:.1f}s\n"
            f"Processing Time: {result.processing_time:.1f}s\n"
        )
        if result.narration_style:
            summary += (
                f"Narrator: {result.narration_style.gender}, "
                f"{result.narration_style.tone}, "
                f"{result.narration_style.pacing} pace\n"
            )

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
    """Process multiple videos in batch."""
    if not video_files:
        return "No videos uploaded."

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
    config.batch.max_concurrent = max_concurrent

    global _pipeline
    _pipeline = DubbingPipeline(config)
    pipeline = get_pipeline()

    # Setup batch processor
    batch = BatchProcessor(config.batch)

    video_paths = []
    for vf in video_files:
        path = vf if isinstance(vf, str) else vf.name
        video_paths.append(path)

    batch.add_videos(video_paths, target_language, narrator_style)

    status_lines = [f"Batch Processing: {len(video_paths)} videos"]
    status_lines.append(f"Target Language: {SUPPORTED_LANGUAGES.get(target_language, target_language)}")
    status_lines.append(f"Max Concurrent: {max_concurrent}")
    status_lines.append("─" * 40)

    def on_batch_progress(batch_progress):
        progress(
            batch_progress.overall_progress,
            desc=f"Batch: {batch_progress.completed_jobs}/{batch_progress.total_jobs} done",
        )

    try:
        jobs = batch.process_all(
            dubbing_function=pipeline.process_video_for_batch,
            progress_callback=on_batch_progress,
        )

        # Build results summary
        for job in jobs:
            status = "DONE" if job.status == JobStatus.COMPLETED else "FAILED"
            video_name = Path(job.video_path).name
            if job.status == JobStatus.COMPLETED:
                elapsed = job.end_time - job.start_time
                status_lines.append(f"  [{status}] {video_name} ({elapsed:.1f}s) -> {job.output_path}")
            else:
                status_lines.append(f"  [{status}] {video_name}: {job.error_message[:100]}")

        batch_info = batch.get_progress()
        status_lines.append("─" * 40)
        status_lines.append(
            f"Results: {batch_info.completed_jobs} completed, "
            f"{batch_info.failed_jobs} failed"
        )

        return "\n".join(status_lines)

    except Exception as e:
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


def get_model_status():
    """Get model installation status for display."""
    try:
        manager = ModelManager()
        results = manager.preload_essential_models()
        lines = ["Model Availability:"]
        lines.append("-" * 40)
        for name, available in results.items():
            status = "Installed" if available else "Not Installed"
            lines.append(f"  {name}: {status}")
        lines.append("")
        lines.append(f"Model cache: {manager.cache_dir}")
        lines.append(f"Cache size: {manager.get_cache_size_mb():.1f} MB")
        return "\n".join(lines)
    except Exception as e:
        return f"Model status check error: {e}"


def get_voice_library_info():
    """Get voice library information for display."""
    try:
        library = VoiceLibrary()
        voices = library.get_all_voices()
        summary = library.get_engine_summary()

        lines = [f"Total voices: {len(voices)}"]
        lines.append("-" * 40)
        lines.append("Voices per engine:")
        for engine, count in sorted(summary.items()):
            lines.append(f"  {engine}: {count}")
        lines.append("")
        lines.append("Languages with voice support:")
        supported = library.get_supported_languages()
        for code, name in sorted(supported.items(), key=lambda x: x[1]):
            lang_voices = library.get_voices_for_language(code)
            lines.append(f"  {name} ({code}): {len(lang_voices)} voices")

        return "\n".join(lines)
    except Exception as e:
        return f"Voice library error: {e}"


def get_real_time_logs():
    """Get the current real-time log buffer contents."""
    if not _log_buffer:
        return "No log messages yet. Process a video to see real-time logs."
    return "\n".join(_log_buffer[-100:])


def clear_logs():
    """Clear the log buffer."""
    _log_buffer.clear()
    return "Logs cleared."


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
        }
        .log-panel {
            font-family: monospace;
            font-size: 12px;
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
                                value="base",
                                label="Whisper Model",
                                info="Larger models are more accurate but slower",
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
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Batch Upload")

                        batch_videos = gr.File(
                            label="Upload Videos (up to 25)",
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
                            info="Number of videos to process simultaneously",
                        )

                        with gr.Accordion("Advanced Settings", open=False):
                            batch_whisper = gr.Dropdown(
                                choices=WHISPER_MODELS,
                                value="base",
                                label="Whisper Model",
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
                            label="Batch Status",
                            lines=20,
                            interactive=False,
                            elem_classes=["stage-info"],
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

            # ── Tab 3: Real-Time Logs ──
            with gr.Tab("Logs", id="logs"):
                gr.Markdown("### Real-Time Processing Logs")

                log_output = gr.Textbox(
                    label="Log Output",
                    lines=25,
                    interactive=False,
                    elem_classes=["log-panel"],
                    value="No log messages yet. Process a video to see real-time logs.",
                )

                with gr.Row():
                    refresh_logs_btn = gr.Button("Refresh Logs", size="sm")
                    clear_logs_btn = gr.Button("Clear Logs", size="sm", variant="secondary")

                refresh_logs_btn.click(fn=get_real_time_logs, outputs=[log_output])
                clear_logs_btn.click(fn=clear_logs, outputs=[log_output])

            # ── Tab 4: Models & Voices ──
            with gr.Tab("Models & Voices", id="models"), gr.Row():
                with gr.Column():
                    gr.Markdown("### Model Manager")
                    gr.Markdown(
                        "Models are downloaded automatically when first needed. "
                        "Check status below to see which models are available."
                    )

                    model_status = gr.Textbox(
                        label="Model Status",
                        lines=12,
                        interactive=False,
                        value=get_model_status(),
                    )
                    model_refresh_btn = gr.Button("Refresh Status", size="sm")
                    model_refresh_btn.click(fn=get_model_status, outputs=[model_status])

                with gr.Column():
                    gr.Markdown("### Voice Library")
                    voice_info = gr.Textbox(
                        label="Available Voices",
                        lines=12,
                        interactive=False,
                        value=get_voice_library_info(),
                    )
                    voice_refresh_btn = gr.Button("Refresh", size="sm")
                    voice_refresh_btn.click(fn=get_voice_library_info, outputs=[voice_info])

            # ── Tab 5: System Info ──
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
                    ### Pipeline Stages
                    """
                    + "\n".join(f"{i+1}. {stage}" for i, stage in enumerate(PIPELINE_STAGES))
                    + """

                    ### Supported Languages
                    """
                    + ", ".join(f"{name} ({code})" for code, name in sorted(SUPPORTED_LANGUAGES.items(), key=lambda x: x[1]))
                    + """

                    ### Voice Engines
                    - **Qwen3-TTS** — Best for Asian and Middle Eastern languages
                    - **Chatterbox** — Best for cinematic English, Spanish, Portuguese, Italian
                    - **LuxTTS** — Best for calm European narration (French, Dutch, Swedish, Danish)

                    All engines fall back to Microsoft Edge TTS when native models are unavailable.
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
    logger.info("Starting %s v%s ...", __app_name__, __version__)

    # Pre-check models
    manager = ModelManager()
    manager.preload_essential_models()

    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
