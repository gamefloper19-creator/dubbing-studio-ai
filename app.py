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
from dubbing_studio.cloning.voice_cloner import VoiceCloner, VoiceProfileManager
from dubbing_studio.config import AppConfig, SUPPORTED_LANGUAGES
from dubbing_studio.hardware.optimizer import HardwareOptimizer
from dubbing_studio.models.manager import ModelManager
from dubbing_studio.pipeline import DubbingPipeline, PIPELINE_STAGES
from dubbing_studio.tts.voice_library import VoiceLibrary
from dubbing_studio.youtube.pipeline import YouTubeDubbingPipeline

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
_voice_library: Optional[VoiceLibrary] = None
_model_manager: Optional[ModelManager] = None
_voice_cloner: Optional[VoiceCloner] = None
_youtube_pipeline: Optional[YouTubeDubbingPipeline] = None


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


def get_voice_library() -> VoiceLibrary:
    """Get or create the voice library."""
    global _voice_library
    if _voice_library is None:
        _voice_library = VoiceLibrary()
    return _voice_library


def get_model_manager() -> ModelManager:
    """Get or create the model manager."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager(get_config().model_management)
    return _model_manager


def get_voice_cloner() -> VoiceCloner:
    """Get or create the voice cloner."""
    global _voice_cloner
    if _voice_cloner is None:
        _voice_cloner = VoiceCloner(get_config().cloning)
    return _voice_cloner


def get_youtube_pipeline() -> YouTubeDubbingPipeline:
    """Get or create the YouTube dubbing pipeline."""
    global _youtube_pipeline
    if _youtube_pipeline is None:
        _youtube_pipeline = YouTubeDubbingPipeline(get_config().youtube)
    return _youtube_pipeline


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


# ── Voice Library functions ──

def get_voice_library_data(language_filter: str = ""):
    """Get voice library data for UI display."""
    library = get_voice_library()
    lang = language_filter if language_filter else None
    rows = library.format_for_display(language=lang)
    summary = library.get_library_summary()
    info = (
        f"Total voices: {summary['total_voices']}\n"
        f"Languages: {summary['languages']}\n"
        f"Styles: {', '.join(summary['styles'])}\n"
        f"Engines: {', '.join(summary['engines'])}"
    )
    return rows, info


# ── Voice Cloning functions ──

def create_voice_profile(
    audio_file,
    profile_name: str,
    gender: str,
    language: str,
    description: str,
):
    """Create a new voice profile from an audio sample."""
    if audio_file is None:
        return "Please upload a voice sample audio file."
    if not profile_name:
        return "Please provide a name for the voice profile."

    try:
        cloner = get_voice_cloner()
        audio_path = audio_file if isinstance(audio_file, str) else audio_file.name

        profile = cloner.create_voice_profile(
            sample_path=audio_path,
            name=profile_name,
            gender=gender,
            language=language,
            description=description,
        )

        return (
            f"Voice profile created!\n"
            f"ID: {profile.profile_id}\n"
            f"Name: {profile.name}\n"
            f"Duration: {profile.duration:.1f}s\n"
            f"Gender: {profile.gender}\n"
            f"Language: {profile.language}"
        )
    except Exception as e:
        return f"Error creating voice profile: {e}"


def list_voice_profiles():
    """List all saved voice profiles."""
    try:
        cloner = get_voice_cloner()
        profiles = cloner.profile_manager.list_profiles()
        if not profiles:
            return "No voice profiles saved yet."

        lines = ["Saved Voice Profiles:", "─" * 40]
        for p in profiles:
            lines.append(
                f"  [{p.profile_id}] {p.name} "
                f"({p.gender}, {p.language}, {p.duration:.1f}s)"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing profiles: {e}"


def delete_voice_profile(profile_id: str):
    """Delete a voice profile."""
    if not profile_id:
        return "Please enter a profile ID."
    try:
        cloner = get_voice_cloner()
        success = cloner.profile_manager.delete_profile(profile_id)
        if success:
            return f"Profile {profile_id} deleted."
        return f"Profile {profile_id} not found."
    except Exception as e:
        return f"Error deleting profile: {e}"


# ── YouTube functions ──

def process_youtube_video(
    url: str,
    target_language: str,
    narrator_style: str,
    whisper_model: str,
    gemini_api_key: str,
    progress=gr.Progress(track_tqdm=True),
):
    """Download and dub a YouTube video."""
    if not url:
        return None, "Please enter a YouTube URL."

    if not gemini_api_key and not os.environ.get("GEMINI_API_KEY"):
        return None, "Please provide a Gemini API key."

    config = get_config()
    if gemini_api_key:
        config.translation.api_key = gemini_api_key
    config.whisper.model_size = whisper_model
    config.voice.narrator_style = narrator_style

    global _pipeline
    _pipeline = DubbingPipeline(config)
    pipeline = get_pipeline()
    yt = get_youtube_pipeline()

    status_lines = []

    try:
        # Download
        progress(0.1, desc="Downloading video...")
        status_lines.append("Downloading video...")
        download_dir = str(Path(config.temp_dir) / "youtube")
        video_path, metadata = yt.download_video(url, download_dir)
        status_lines.append(f"Downloaded: {metadata.title}")

        # Process
        progress(0.2, desc="Processing dubbing pipeline...")
        status_lines.append("Processing through dubbing pipeline...")

        def on_progress(stage: str, prog: float):
            adjusted = 0.2 + prog * 0.7
            progress(adjusted, desc=stage)

        result = pipeline.process_video(
            video_path=video_path,
            target_language=target_language,
            narrator_style=narrator_style,
            progress_callback=on_progress,
        )

        progress(1.0, desc="Complete!")

        status_lines.extend([
            "─" * 40,
            "YouTube Dubbing Complete!",
            f"Title: {metadata.title}",
            f"Source: {result.source_language}",
            f"Target: {result.target_language}",
            f"Segments: {result.total_segments}",
            f"Processing Time: {result.processing_time:.1f}s",
            f"Output: {result.output_video_path}",
        ])

        return result.output_video_path, "\n".join(status_lines)

    except Exception as e:
        status_lines.append(f"Error: {e}")
        return None, "\n".join(status_lines)


# ── Model Management functions ──

def scan_models():
    """Scan for installed models."""
    try:
        mgr = get_model_manager()
        models = mgr.scan_models()
        lines = ["Model Status:", "─" * 50]
        for m in models:
            status = "INSTALLED" if m.installed else "MISSING"
            req = " (required)" if m.required else ""
            size = f" [{m.size_mb}MB]" if m.size_mb > 0 else ""
            lines.append(f"  [{status}] {m.name}{req}{size}")
            lines.append(f"           {m.description}")

        cache_size = mgr.get_cache_size()
        lines.extend(["─" * 50, f"Cache size: {cache_size:.1f} MB"])
        return "\n".join(lines)
    except Exception as e:
        return f"Error scanning models: {e}"


def download_missing_models(progress=gr.Progress(track_tqdm=True)):
    """Download all missing required models."""
    try:
        mgr = get_model_manager()
        missing = mgr.get_missing_required()

        if not missing:
            return "All required models are installed!"

        lines = [f"Downloading {len(missing)} missing models..."]

        for i, model in enumerate(missing):
            progress((i + 1) / len(missing), desc=f"Installing {model.name}...")
            success = mgr.download_model(model.name)
            status = "OK" if success else "FAILED"
            lines.append(f"  [{status}] {model.name}")

        return "\n".join(lines)
    except Exception as e:
        return f"Error downloading models: {e}"


# ── Build Gradio Interface ──

def create_ui() -> gr.Blocks:
    """Create the Gradio web interface."""

    with gr.Blocks(
        title=f"{__app_name__} v{__version__}",
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

            # ── Tab 3: Voice Library ──
            with gr.Tab("Voice Library", id="voices"):
                gr.Markdown("### Voice Library")
                gr.Markdown("Browse available narrator voices by language, style, and gender.")

                with gr.Row():
                    voice_lang_filter = gr.Dropdown(
                        choices=[("All Languages", "")] + LANGUAGE_CHOICES,
                        value="",
                        label="Filter by Language",
                    )
                    voice_refresh_btn = gr.Button("Refresh", size="sm")

                voice_table = gr.Dataframe(
                    headers=["Name", "Language", "Gender", "Style", "Engine"],
                    label="Available Voices",
                    interactive=False,
                )

                voice_info = gr.Textbox(
                    label="Library Summary",
                    lines=4,
                    interactive=False,
                )

                def update_voice_table(lang):
                    rows, info = get_voice_library_data(lang)
                    return rows, info

                voice_refresh_btn.click(
                    fn=update_voice_table,
                    inputs=[voice_lang_filter],
                    outputs=[voice_table, voice_info],
                )
                voice_lang_filter.change(
                    fn=update_voice_table,
                    inputs=[voice_lang_filter],
                    outputs=[voice_table, voice_info],
                )

            # ── Tab 4: Voice Cloning ──
            with gr.Tab("Voice Cloning", id="cloning"):
                gr.Markdown("### Voice Cloning Manager")
                gr.Markdown(
                    "Create custom narrator voices from audio samples (10-60 seconds)."
                )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Create New Profile")
                        clone_audio = gr.Audio(
                            label="Voice Sample (10-60s)",
                            type="filepath",
                        )
                        clone_name = gr.Textbox(
                            label="Profile Name",
                            placeholder="e.g., David Attenborough Style",
                        )
                        clone_gender = gr.Dropdown(
                            choices=["male", "female", "unknown"],
                            value="male",
                            label="Gender",
                        )
                        clone_language = gr.Dropdown(
                            choices=LANGUAGE_CHOICES,
                            value="en",
                            label="Primary Language",
                        )
                        clone_desc = gr.Textbox(
                            label="Description (optional)",
                            placeholder="Describe the voice characteristics",
                        )
                        clone_btn = gr.Button(
                            "Create Voice Profile",
                            variant="primary",
                        )

                    with gr.Column():
                        gr.Markdown("#### Saved Profiles")
                        profiles_display = gr.Textbox(
                            label="Voice Profiles",
                            lines=10,
                            interactive=False,
                        )
                        refresh_profiles_btn = gr.Button("Refresh Profiles", size="sm")

                        delete_profile_id = gr.Textbox(
                            label="Profile ID to Delete",
                            placeholder="Enter profile ID",
                        )
                        delete_profile_btn = gr.Button("Delete Profile", variant="stop", size="sm")

                clone_status = gr.Textbox(
                    label="Status",
                    lines=6,
                    interactive=False,
                )

                clone_btn.click(
                    fn=create_voice_profile,
                    inputs=[clone_audio, clone_name, clone_gender, clone_language, clone_desc],
                    outputs=[clone_status],
                )
                refresh_profiles_btn.click(
                    fn=list_voice_profiles,
                    outputs=[profiles_display],
                )
                delete_profile_btn.click(
                    fn=delete_voice_profile,
                    inputs=[delete_profile_id],
                    outputs=[clone_status],
                )

            # ── Tab 5: YouTube Dubbing ──
            with gr.Tab("YouTube Dubbing", id="youtube"):
                gr.Markdown("### YouTube Video Dubbing")
                gr.Markdown(
                    "Paste a YouTube URL to automatically download, dub, and export."
                )

                with gr.Row():
                    with gr.Column():
                        yt_url = gr.Textbox(
                            label="YouTube URL",
                            placeholder="https://www.youtube.com/watch?v=...",
                        )
                        yt_target_lang = gr.Dropdown(
                            choices=LANGUAGE_CHOICES,
                            value="hi",
                            label="Target Language",
                        )
                        yt_narrator_style = gr.Dropdown(
                            choices=NARRATOR_STYLES,
                            value="documentary",
                            label="Narrator Style",
                        )
                        yt_whisper = gr.Dropdown(
                            choices=WHISPER_MODELS,
                            value="base",
                            label="Whisper Model",
                        )
                        yt_gemini_key = gr.Textbox(
                            label="Gemini API Key",
                            type="password",
                            placeholder="Gemini API key (or set env var)",
                        )
                        yt_btn = gr.Button(
                            "Start YouTube Dubbing",
                            variant="primary",
                            size="lg",
                        )

                    with gr.Column():
                        yt_video_output = gr.Video(label="Dubbed Video")
                        yt_status = gr.Textbox(
                            label="Status",
                            lines=12,
                            interactive=False,
                            elem_classes=["stage-info"],
                        )

                yt_btn.click(
                    fn=process_youtube_video,
                    inputs=[yt_url, yt_target_lang, yt_narrator_style, yt_whisper, yt_gemini_key],
                    outputs=[yt_video_output, yt_status],
                )

            # ── Tab 6: Model Manager ──
            with gr.Tab("Model Manager", id="models"):
                gr.Markdown("### Model Download Manager")
                gr.Markdown(
                    "Manage AI models used by the dubbing pipeline. "
                    "Missing models are downloaded automatically when needed."
                )

                model_status = gr.Textbox(
                    label="Model Status",
                    lines=15,
                    interactive=False,
                )

                with gr.Row():
                    scan_btn = gr.Button("Scan Models", size="sm")
                    download_btn = gr.Button(
                        "Download Missing Models",
                        variant="primary",
                        size="sm",
                    )

                download_status = gr.Textbox(
                    label="Download Status",
                    lines=5,
                    interactive=False,
                )

                scan_btn.click(fn=scan_models, outputs=[model_status])
                download_btn.click(
                    fn=download_missing_models,
                    outputs=[download_status],
                )

            # ── Tab 7: GPU & System Info ──
            with gr.Tab("System Info", id="system"):
                gr.Markdown("### Hardware & GPU Status Monitor")

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

                    ### New Advanced Features
                    - **Emotion-Aware Narration** — Detects emotional tone and adjusts voice style
                    - **Voice Cloning** — Create custom narrator voices from audio samples
                    - **Multi-Speaker Dubbing** — Detects and assigns unique voices per speaker
                    - **Cinematic Narration** — Merges segments for professional documentary flow
                    - **Advanced Timing** — Scene-aware synchronization with micro-timing correction
                    - **YouTube Pipeline** — Download, dub, and export YouTube videos
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
        """,
    )


if __name__ == "__main__":
    main()
