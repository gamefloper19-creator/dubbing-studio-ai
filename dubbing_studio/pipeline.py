"""
Main dubbing pipeline orchestrator.

Coordinates all stages of the dubbing process:
1. Audio Extraction
2. Audio Cleaning
3. Silence Detection & Segmentation
4. Speech Recognition
5. Speaker Analysis
6. Semantic Translation
7. Narration Style Detection
8. Automatic Voice Selection
9. Neural Voice Generation
10. Speech Timing Alignment
11. Background Audio Mixing
12. Subtitle Generation
13. Video Rendering
"""

import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from dubbing_studio.audio.cleaner import AudioCleaner
from dubbing_studio.audio.extractor import AudioExtractor
from dubbing_studio.audio.mixer import AudioMixer
from dubbing_studio.audio.segmenter import AudioSegment, AudioSegmenter
from dubbing_studio.config import AppConfig
from dubbing_studio.emotion.analyzer import EmotionAnalyzer, EmotionProfile
from dubbing_studio.export.exporter import Exporter
from dubbing_studio.narration.engine import CinematicNarrationEngine
from dubbing_studio.speech.analyzer import NarrationAnalyzer, NarrationStyle
from dubbing_studio.speech.recognizer import SpeechRecognizer, TranscriptionSegment
from dubbing_studio.subtitle.generator import SubtitleGenerator
from dubbing_studio.timing.aligner import TimingAligner
from dubbing_studio.timing.advanced_aligner import AdvancedTimingAligner
from dubbing_studio.translation.translator import TranslatedSegment, Translator
from dubbing_studio.tts.voice_selector import VoiceSelector
from dubbing_studio.video.renderer import VideoRenderer

logger = logging.getLogger(__name__)


PIPELINE_STAGES = [
    "Audio Extraction",
    "Audio Cleaning",
    "Segmentation",
    "Speech Recognition",
    "Narration Analysis",
    "Emotion Analysis",
    "Translation",
    "Cinematic Optimization",
    "Voice Selection",
    "Speech Generation",
    "Timing Alignment",
    "Background Mixing",
    "Subtitle Generation",
    "Video Rendering",
]


@dataclass
class DubbingResult:
    """Result of a complete dubbing pipeline run."""
    video_path: str
    output_video_path: str
    output_audio_path: str
    subtitle_paths: dict[str, str] = field(default_factory=dict)
    source_language: str = ""
    target_language: str = ""
    total_segments: int = 0
    total_duration: float = 0.0
    processing_time: float = 0.0
    narration_style: Optional[NarrationStyle] = None
    emotion_profiles: list = field(default_factory=list)
    cinematic_optimized: bool = False


class DubbingPipeline:
    """
    Main orchestrator for the dubbing pipeline.

    Coordinates all processing stages from video input to final output.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig.from_env()
        self.config.setup_dirs()

        # Initialize core components (existing pipeline)
        self.extractor = AudioExtractor(self.config.audio)
        self.cleaner = AudioCleaner(self.config.audio)
        self.segmenter = AudioSegmenter(self.config.audio)
        self.recognizer = SpeechRecognizer(self.config.whisper)
        self.analyzer = NarrationAnalyzer(self.config.audio)
        self.translator = Translator(self.config.translation)
        self.voice_selector = VoiceSelector(self.config.voice)
        self.aligner = TimingAligner(self.config.timing)
        self.mixer = AudioMixer(self.config.mixing)
        self.subtitle_gen = SubtitleGenerator(self.config.subtitle)
        self.renderer = VideoRenderer(self.config.export, self.config.subtitle)
        self.exporter = Exporter(self.config.export)

        # Initialize advanced components (new capabilities)
        self.emotion_analyzer = EmotionAnalyzer(self.config.emotion)
        self.cinematic_engine = CinematicNarrationEngine(self.config.cinematic)
        self.advanced_aligner = AdvancedTimingAligner(self.config.timing)

    def process_video(
        self,
        video_path: str,
        target_language: str,
        narrator_style: str = "documentary",
        progress_callback: Optional[Callable[[str, float], None]] = None,
        output_dir: Optional[str] = None,
    ) -> DubbingResult:
        """
        Process a single video through the full dubbing pipeline.

        Args:
            video_path: Path to the input video file.
            target_language: Target language code (e.g., 'hi', 'es', 'fr').
            narrator_style: Style of narration.
            progress_callback: Optional callback(stage_name, progress_float).
            output_dir: Optional output directory override.

        Returns:
            DubbingResult with all output paths and metadata.
        """
        start_time = time.time()
        video_path = str(Path(video_path).resolve())
        video_name = Path(video_path).stem

        # Setup working directories
        out_dir = Path(output_dir or self.config.output_dir)
        work_dir = Path(self.config.temp_dir) / f"{video_name}_{int(time.time())}"
        out_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)

        def report_progress(stage: str, progress: float) -> None:
            logger.info("Stage: %s (%.0f%%)", stage, progress * 100)
            if progress_callback:
                progress_callback(stage, progress)

        try:
            # ── Pre-flight Validation ──
            from dubbing_studio.validation import validate_video_file, validate_language
            validate_video_file(video_path)
            validate_language(target_language)

            # ── Stage 1: Audio Extraction ──
            report_progress("Audio Extraction", 0.0)
            raw_audio = str(work_dir / "raw_audio.wav")
            self.extractor.extract_audio(video_path, raw_audio)

            bg_audio = str(work_dir / "background_audio.wav")
            try:
                self.extractor.extract_background_audio(video_path, bg_audio)
            except Exception as e:
                logger.warning("Background audio extraction failed: %s", e)
                bg_audio = None

            report_progress("Audio Extraction", 1.0 / len(PIPELINE_STAGES))

            # ── Stage 2: Audio Cleaning ──
            report_progress("Audio Cleaning", 1.0 / len(PIPELINE_STAGES))
            clean_audio = str(work_dir / "clean_audio.wav")
            self.cleaner.clean_audio(raw_audio, clean_audio)
            report_progress("Audio Cleaning", 2.0 / len(PIPELINE_STAGES))

            # ── Stage 3: Segmentation ──
            report_progress("Segmentation", 2.0 / len(PIPELINE_STAGES))
            segments_dir = str(work_dir / "segments")
            audio_segments = self.segmenter.segment_audio(clean_audio, segments_dir)
            report_progress("Segmentation", 3.0 / len(PIPELINE_STAGES))

            # ── Stage 4: Speech Recognition ──
            report_progress("Speech Recognition", 3.0 / len(PIPELINE_STAGES))
            transcription_segments = self.recognizer.transcribe_segments(audio_segments)
            detected_language = (
                transcription_segments[0].language
                if transcription_segments else "en"
            )
            full_text = " ".join(s.text for s in transcription_segments)
            report_progress("Speech Recognition", 4.0 / len(PIPELINE_STAGES))

            # ── Stage 5: Narration Analysis ──
            report_progress("Narration Analysis", 4.0 / len(PIPELINE_STAGES))
            total_duration = self.extractor.get_audio_duration(raw_audio)
            narration_style_info = self.analyzer.analyze_narration(
                clean_audio, full_text, total_duration
            )
            report_progress("Narration Analysis", 5.0 / len(PIPELINE_STAGES))

            # ── Stage 6: Emotion Analysis (NEW) ──
            report_progress("Emotion Analysis", 5.0 / len(PIPELINE_STAGES))
            emotion_profiles: list[EmotionProfile] = []
            if self.config.emotion.enabled:
                emotion_segments = [
                    {"text": seg.text}
                    for seg in transcription_segments
                ]
                emotion_profiles = self.emotion_analyzer.analyze_segments(
                    emotion_segments
                )
                logger.info(
                    "Emotion analysis: %d profiles generated",
                    len(emotion_profiles),
                )
            report_progress("Emotion Analysis", 6.0 / len(PIPELINE_STAGES))

            # ── Stage 7: Translation ──
            report_progress("Translation", 6.0 / len(PIPELINE_STAGES))
            translated_segments = self.translator.translate_segments(
                transcription_segments, target_language
            )
            report_progress("Translation", 7.0 / len(PIPELINE_STAGES))

            # ── Stage 8: Cinematic Narration Optimization (NEW) ──
            report_progress("Cinematic Optimization", 7.0 / len(PIPELINE_STAGES))
            cinematic_optimized = False
            if self.config.cinematic.enabled:
                translated_segments = self.cinematic_engine.optimize_narration(
                    translated_segments
                )
                cinematic_optimized = True
                logger.info(
                    "Cinematic optimization applied: %d segments",
                    len(translated_segments),
                )
            report_progress("Cinematic Optimization", 8.0 / len(PIPELINE_STAGES))

            # ── Stage 9: Voice Selection ──
            report_progress("Voice Selection", 8.0 / len(PIPELINE_STAGES))
            tts_engine, voice_config = self.voice_selector.select_voice(
                target_language, narration_style_info
            )
            report_progress("Voice Selection", 9.0 / len(PIPELINE_STAGES))

            # ── Stage 10: Speech Generation ──
            report_progress("Speech Generation", 9.0 / len(PIPELINE_STAGES))
            tts_dir = str(work_dir / "tts")
            Path(tts_dir).mkdir(parents=True, exist_ok=True)

            tts_audio_paths = []
            for i, seg in enumerate(translated_segments):
                tts_path = str(Path(tts_dir) / f"tts_{seg.segment_id}.wav")

                # Apply emotion-aware speed/pitch if available
                tts_speed = voice_config.get("speed", 1.0)
                if emotion_profiles and i < len(emotion_profiles):
                    ep = emotion_profiles[i]
                    tts_params = self.emotion_analyzer.get_tts_parameters(
                        ep, base_speed=tts_speed
                    )
                    tts_speed = tts_params["speed"]

                tts_result = tts_engine.generate_speech(
                    text=seg.translated_text,
                    output_path=tts_path,
                    language=target_language,
                    speed=tts_speed,
                )
                tts_audio_paths.append({
                    "audio_path": tts_result.audio_path,
                    "segment": seg,
                    "duration": tts_result.duration,
                })

                sub_progress = 9.0 + (i + 1) / len(translated_segments)
                report_progress("Speech Generation", sub_progress / len(PIPELINE_STAGES))

            report_progress("Speech Generation", 10.0 / len(PIPELINE_STAGES))

            # ── Stage 11: Timing Alignment (with advanced enhancements) ──
            report_progress("Timing Alignment", 10.0 / len(PIPELINE_STAGES))
            aligned_dir = str(work_dir / "aligned")
            Path(aligned_dir).mkdir(parents=True, exist_ok=True)

            # Detect scene boundaries for advanced timing
            try:
                self.advanced_aligner.detect_scene_boundaries(video_path)
            except Exception as e:
                logger.debug("Scene boundary detection skipped: %s", e)

            aligned_paths = []
            for item in tts_audio_paths:
                seg = item["segment"]
                target_dur = seg.end_time - seg.start_time
                aligned_path = str(Path(aligned_dir) / f"aligned_{seg.segment_id}.wav")

                # Use advanced aligner with scene awareness
                try:
                    adv_result = self.advanced_aligner.align_with_scene_awareness(
                        audio_path=item["audio_path"],
                        target_duration=target_dur,
                        output_path=aligned_path,
                        segment_start=seg.start_time,
                        segment_end=seg.end_time,
                    )
                    logger.debug(
                        "Advanced alignment: deviation=%.0fms, scene=%s, micro=%d",
                        adv_result.adjustment.deviation_ms,
                        adv_result.scene_aligned,
                        adv_result.micro_corrections,
                    )
                except Exception as e:
                    # Fallback to standard aligner
                    logger.debug("Advanced aligner fallback: %s", e)
                    self.aligner.align_timing(
                        audio_path=item["audio_path"],
                        target_duration=target_dur,
                        output_path=aligned_path,
                    )

                aligned_paths.append({
                    "audio_path": aligned_path,
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                })

            report_progress("Timing Alignment", 11.0 / len(PIPELINE_STAGES))

            # ── Stage 12: Build Full Narration Track ──
            report_progress("Background Mixing", 11.0 / len(PIPELINE_STAGES))

            # Build full narration audio with proper timing
            narration_audio = str(work_dir / "narration.wav")
            self._build_narration_track(
                aligned_paths, narration_audio, total_duration
            )

            # Mix with background
            if bg_audio and Path(bg_audio).exists():
                mixed_audio = str(work_dir / "mixed_audio.wav")
                self.mixer.mix_audio(
                    narration_path=narration_audio,
                    background_path=bg_audio,
                    output_path=mixed_audio,
                    narration_segments=aligned_paths,
                )
            else:
                mixed_audio = narration_audio

            report_progress("Background Mixing", 12.0 / len(PIPELINE_STAGES))

            # ── Stage 13: Subtitle Generation ──
            report_progress("Subtitle Generation", 12.0 / len(PIPELINE_STAGES))
            subtitles_dir = str(out_dir / f"{video_name}_subtitles")
            subtitle_paths = self.subtitle_gen.generate_all_formats(
                translated_segments,
                subtitles_dir,
                f"{video_name}_{target_language}",
            )
            report_progress("Subtitle Generation", 13.0 / len(PIPELINE_STAGES))

            # ── Stage 14: Video Rendering ──
            report_progress("Video Rendering", 13.0 / len(PIPELINE_STAGES))

            output_video = str(out_dir / f"{video_name}_{target_language}_dubbed.mp4")
            subtitle_for_embed = subtitle_paths.get(
                self.config.subtitle.format, None
            )

            self.renderer.render_video(
                video_path=video_path,
                audio_path=mixed_audio,
                output_path=output_video,
                subtitle_path=subtitle_for_embed,
                embed_subtitles=self.config.subtitle.embed_in_video,
            )

            # Export audio-only versions
            output_audio_wav = str(out_dir / f"{video_name}_{target_language}_dubbed.wav")
            self.exporter.export_audio(mixed_audio, output_audio_wav, "wav")

            output_audio_mp3 = str(out_dir / f"{video_name}_{target_language}_dubbed.mp3")
            self.exporter.export_audio(mixed_audio, output_audio_mp3, "mp3")

            report_progress("Video Rendering", 1.0)

            # Cleanup temp files
            self._cleanup_temp(work_dir)

            # Unload models to free memory
            self.recognizer.unload_model()
            self.voice_selector.unload_all()

            processing_time = time.time() - start_time

            result = DubbingResult(
                video_path=video_path,
                output_video_path=output_video,
                output_audio_path=output_audio_wav,
                subtitle_paths=subtitle_paths,
                source_language=detected_language,
                target_language=target_language,
                total_segments=len(translated_segments),
                total_duration=total_duration,
                processing_time=processing_time,
                narration_style=narration_style_info,
                emotion_profiles=emotion_profiles,
                cinematic_optimized=cinematic_optimized,
            )

            logger.info(
                "Dubbing complete: %s -> %s in %.1fs (%d segments)",
                detected_language, target_language,
                processing_time, len(translated_segments),
            )

            return result

        except Exception as e:
            logger.error("Dubbing pipeline failed: %s", str(e), exc_info=True)
            self._cleanup_temp(work_dir)
            raise

    def _build_narration_track(
        self,
        aligned_paths: list[dict],
        output_path: str,
        total_duration: float,
    ) -> str:
        """
        Build a complete narration track with proper timing.

        Places each aligned audio segment at its correct position
        in the timeline with silence between segments.
        """
        import subprocess

        if not aligned_paths:
            # Generate silence for the full duration
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", f"anullsrc=r=44100:cl=mono:d={total_duration}",
                "-t", str(total_duration),
                output_path,
            ]
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return output_path

        # Build a complex filter to place segments at correct times
        inputs = []
        filter_parts = []

        # Start with silence base track
        inputs.extend([
            "-f", "lavfi",
            "-i", f"anullsrc=r=44100:cl=mono:d={total_duration}",
        ])

        for i, seg in enumerate(aligned_paths):
            inputs.extend(["-i", seg["audio_path"]])

        # Build adelay filter to position each segment
        mix_parts = ["[0:a]"]  # base silence track

        for i, seg in enumerate(aligned_paths):
            delay_ms = int(seg["start_time"] * 1000)
            filter_parts.append(
                f"[{i + 1}:a]adelay={delay_ms}|{delay_ms},"
                f"apad=whole_dur={total_duration}[s{i}]"
            )
            mix_parts.append(f"[s{i}]")

        # Mix all tracks
        n_inputs = len(aligned_paths) + 1
        filter_parts.append(
            "".join(mix_parts)
            + f"amix=inputs={n_inputs}:duration=longest"
            + f":dropout_transition=0[out]"
        )

        filter_complex = ";".join(filter_parts)

        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-t", str(total_duration),
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(
                "Complex narration build failed, using simple concatenation: %s",
                result.stderr[:500],
            )
            # Fallback: simple concatenation
            audio_paths = [seg["audio_path"] for seg in aligned_paths]
            self.mixer.concatenate_audio(audio_paths, output_path)

        return output_path

    def _cleanup_temp(self, work_dir: Path) -> None:
        """Remove temporary working directory."""
        try:
            if work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)
        except Exception as e:
            logger.warning("Failed to cleanup temp dir: %s", e)

    def process_video_for_batch(
        self,
        video_path: str,
        target_language: str,
        narrator_style: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> str:
        """
        Wrapper for batch processing - returns output path string.

        Args:
            video_path: Path to video.
            target_language: Target language code.
            narrator_style: Narration style.
            progress_callback: Progress callback.

        Returns:
            Output video path.
        """
        result = self.process_video(
            video_path=video_path,
            target_language=target_language,
            narrator_style=narrator_style,
            progress_callback=progress_callback,
        )
        return result.output_video_path
