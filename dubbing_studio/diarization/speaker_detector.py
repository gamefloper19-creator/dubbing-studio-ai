"""
Multi-speaker diarization system.

Detects different speakers in original audio and assigns
unique voice models to each speaker for consistent dubbing.
"""

import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dubbing_studio.config import DiarizationConfig

logger = logging.getLogger(__name__)


@dataclass
class SpeakerSegment:
    """A segment attributed to a specific speaker."""
    segment_id: str
    speaker_id: str
    start_time: float
    end_time: float
    duration: float
    confidence: float = 0.0
    text: str = ""


@dataclass
class SpeakerProfile:
    """Profile for a detected speaker."""
    speaker_id: str
    total_duration: float  # total speaking time
    segment_count: int
    assigned_voice: str = ""  # assigned TTS voice ID
    assigned_gender: str = "unknown"
    metadata: dict = field(default_factory=dict)


@dataclass
class DiarizationResult:
    """Complete diarization result."""
    speakers: list[SpeakerProfile]
    segments: list[SpeakerSegment]
    num_speakers: int


class SpeakerDetector:
    """
    Detect and track multiple speakers in audio.

    Uses energy-based segmentation and spectral clustering
    as a lightweight alternative to neural diarization models.
    When pyannote.audio is available, uses it for better accuracy.
    """

    def __init__(self, config: Optional[DiarizationConfig] = None):
        self.config = config or DiarizationConfig()
        self._pipeline = None

    def diarize(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
    ) -> DiarizationResult:
        """
        Perform speaker diarization on audio file.

        Args:
            audio_path: Path to audio file.
            num_speakers: Optional expected number of speakers.

        Returns:
            DiarizationResult with detected speakers and segments.
        """
        audio_path = str(Path(audio_path).resolve())

        # Try pyannote.audio first for best results
        try:
            return self._diarize_with_pyannote(audio_path, num_speakers)
        except Exception as e:
            logger.info("pyannote.audio not available (%s), using FFmpeg-based diarization", e)

        # Fallback to FFmpeg-based energy analysis
        return self._diarize_with_energy(audio_path, num_speakers)

    def _diarize_with_pyannote(
        self,
        audio_path: str,
        num_speakers: Optional[int],
    ) -> DiarizationResult:
        """Diarize using pyannote.audio pipeline."""
        try:
            from pyannote.audio import Pipeline
        except ImportError:
            raise ImportError("pyannote.audio is not installed")

        if self._pipeline is None:
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1"
            )

        kwargs = {}
        if num_speakers:
            kwargs["num_speakers"] = num_speakers
        elif self.config.min_speakers == self.config.max_speakers:
            kwargs["num_speakers"] = self.config.min_speakers
        else:
            kwargs["min_speakers"] = self.config.min_speakers
            kwargs["max_speakers"] = self.config.max_speakers

        diarization = self._pipeline(audio_path, **kwargs)

        # Convert to our format
        segments = []
        speaker_durations: dict[str, float] = {}
        speaker_counts: dict[str, int] = {}

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            seg = SpeakerSegment(
                segment_id=f"{len(segments)+1:03d}",
                speaker_id=speaker,
                start_time=turn.start,
                end_time=turn.end,
                duration=turn.end - turn.start,
            )
            segments.append(seg)

            speaker_durations[speaker] = speaker_durations.get(speaker, 0) + seg.duration
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1

        speakers = []
        for spk_id in sorted(speaker_durations.keys()):
            speakers.append(SpeakerProfile(
                speaker_id=spk_id,
                total_duration=speaker_durations[spk_id],
                segment_count=speaker_counts[spk_id],
            ))

        logger.info(
            "Diarization complete (pyannote): %d speakers, %d segments",
            len(speakers), len(segments),
        )

        return DiarizationResult(
            speakers=speakers,
            segments=segments,
            num_speakers=len(speakers),
        )

    def _diarize_with_energy(
        self,
        audio_path: str,
        num_speakers: Optional[int],
    ) -> DiarizationResult:
        """
        Lightweight diarization using FFmpeg energy analysis.

        Segments audio by silence detection and clusters segments
        based on spectral characteristics (volume, pitch proxy).
        """
        # Detect silence to find speech segments
        silence_segments = self._detect_silence(audio_path)
        total_duration = self._get_duration(audio_path)

        # Build speech segments from silence boundaries
        speech_segments = self._build_speech_segments(silence_segments, total_duration)

        if not speech_segments:
            # Single speaker, full audio
            speakers = [SpeakerProfile(
                speaker_id="SPEAKER_00",
                total_duration=total_duration,
                segment_count=1,
            )]
            segments = [SpeakerSegment(
                segment_id="001",
                speaker_id="SPEAKER_00",
                start_time=0.0,
                end_time=total_duration,
                duration=total_duration,
            )]
            return DiarizationResult(speakers=speakers, segments=segments, num_speakers=1)

        # Analyze each segment's audio characteristics
        segment_features = []
        for seg in speech_segments:
            features = self._analyze_segment_features(
                audio_path, seg["start"], seg["end"]
            )
            segment_features.append(features)

        # Cluster segments into speakers based on features
        target_speakers = num_speakers or self._estimate_num_speakers(segment_features)
        target_speakers = max(1, min(target_speakers, self.config.max_speakers))

        speaker_assignments = self._cluster_speakers(
            segment_features, target_speakers
        )

        # Build results
        segments = []
        speaker_durations: dict[str, float] = {}
        speaker_counts: dict[str, int] = {}

        for i, seg in enumerate(speech_segments):
            speaker_id = f"SPEAKER_{speaker_assignments[i]:02d}"
            duration = seg["end"] - seg["start"]

            segments.append(SpeakerSegment(
                segment_id=f"{i+1:03d}",
                speaker_id=speaker_id,
                start_time=seg["start"],
                end_time=seg["end"],
                duration=duration,
            ))

            speaker_durations[speaker_id] = speaker_durations.get(speaker_id, 0) + duration
            speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1

        speakers = []
        for spk_id in sorted(speaker_durations.keys()):
            speakers.append(SpeakerProfile(
                speaker_id=spk_id,
                total_duration=speaker_durations[spk_id],
                segment_count=speaker_counts[spk_id],
            ))

        logger.info(
            "Diarization complete (energy-based): %d speakers, %d segments",
            len(speakers), len(segments),
        )

        return DiarizationResult(
            speakers=speakers,
            segments=segments,
            num_speakers=len(speakers),
        )

    def assign_voices(
        self,
        diarization_result: DiarizationResult,
        available_voices: list[dict],
    ) -> dict[str, dict]:
        """
        Assign unique voice models to each detected speaker.

        Args:
            diarization_result: Diarization results.
            available_voices: List of available voice configurations.

        Returns:
            Dict mapping speaker_id to voice configuration.
        """
        voice_assignments: dict[str, dict] = {}

        # Sort speakers by total duration (primary speaker first)
        sorted_speakers = sorted(
            diarization_result.speakers,
            key=lambda s: s.total_duration,
            reverse=True,
        )

        used_voices: set[str] = set()

        for i, speaker in enumerate(sorted_speakers):
            # Pick a voice that hasn't been used yet
            voice = None
            for v in available_voices:
                voice_id = v.get("id", f"voice_{i}")
                if voice_id not in used_voices:
                    voice = v
                    used_voices.add(voice_id)
                    break

            if voice is None and available_voices:
                # Cycle through voices if we run out
                voice = available_voices[i % len(available_voices)]

            if voice:
                voice_assignments[speaker.speaker_id] = voice
                speaker.assigned_voice = voice.get("id", "")
                speaker.assigned_gender = voice.get("gender", "unknown")

        logger.info(
            "Assigned %d unique voices to %d speakers",
            len(voice_assignments), len(sorted_speakers),
        )

        return voice_assignments

    def _detect_silence(self, audio_path: str) -> list[dict]:
        """Detect silence periods in audio."""
        cmd = [
            "ffmpeg",
            "-i", audio_path,
            "-af", "silencedetect=noise=-35dB:d=0.8",
            "-f", "null", "-",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        stderr = result.stderr

        silences = []
        silence_start = None

        for line in stderr.split("\n"):
            if "silence_start:" in line:
                try:
                    parts = line.split("silence_start:")
                    silence_start = float(parts[1].strip().split()[0])
                except (IndexError, ValueError):
                    continue
            elif "silence_end:" in line and silence_start is not None:
                try:
                    parts = line.split("silence_end:")
                    silence_end = float(parts[1].strip().split()[0])
                    silences.append({"start": silence_start, "end": silence_end})
                    silence_start = None
                except (IndexError, ValueError):
                    continue

        return silences

    def _build_speech_segments(
        self,
        silences: list[dict],
        total_duration: float,
    ) -> list[dict]:
        """Build speech segments from silence boundaries."""
        if not silences:
            return [{"start": 0.0, "end": total_duration}]

        segments = []
        current_start = 0.0

        for silence in silences:
            if silence["start"] - current_start > self.config.min_segment_duration:
                segments.append({"start": current_start, "end": silence["start"]})
            current_start = silence["end"]

        if total_duration - current_start > self.config.min_segment_duration:
            segments.append({"start": current_start, "end": total_duration})

        return segments

    def _analyze_segment_features(
        self,
        audio_path: str,
        start: float,
        end: float,
    ) -> dict:
        """Extract audio features for a segment."""
        duration = end - start

        cmd = [
            "ffmpeg",
            "-i", audio_path,
            "-ss", f"{start:.3f}",
            "-t", f"{duration:.3f}",
            "-af", "volumedetect",
            "-f", "null", "-",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        stderr = result.stderr

        mean_vol = -25.0
        max_vol = -10.0

        for line in stderr.split("\n"):
            if "mean_volume" in line:
                try:
                    mean_vol = float(line.split(":")[-1].strip().replace(" dB", ""))
                except (ValueError, IndexError):
                    pass
            elif "max_volume" in line:
                try:
                    max_vol = float(line.split(":")[-1].strip().replace(" dB", ""))
                except (ValueError, IndexError):
                    pass

        return {
            "mean_vol": mean_vol,
            "max_vol": max_vol,
            "dynamic_range": max_vol - mean_vol,
            "start": start,
            "end": end,
        }

    def _estimate_num_speakers(self, features: list[dict]) -> int:
        """Estimate number of speakers from feature variance."""
        if len(features) < 2:
            return 1

        volumes = [f["mean_vol"] for f in features]
        vol_range = max(volumes) - min(volumes)

        # Heuristic: large volume variance suggests multiple speakers
        if vol_range > 12:
            return min(3, self.config.max_speakers)
        elif vol_range > 6:
            return min(2, self.config.max_speakers)
        return 1

    def _cluster_speakers(
        self,
        features: list[dict],
        num_speakers: int,
    ) -> list[int]:
        """
        Cluster segments into speakers using simple k-means-like approach.

        Returns list of speaker indices for each segment.
        """
        if num_speakers <= 1:
            return [0] * len(features)

        # Extract volume features for clustering
        volumes = [f["mean_vol"] for f in features]

        if not volumes:
            return [0] * len(features)

        # Simple equal-interval clustering based on volume
        vol_min = min(volumes)
        vol_max = max(volumes)
        vol_range = vol_max - vol_min

        if vol_range < 1.0:
            return [0] * len(features)

        assignments = []
        for vol in volumes:
            normalized = (vol - vol_min) / vol_range
            cluster = min(int(normalized * num_speakers), num_speakers - 1)
            assignments.append(cluster)

        return assignments

    def _get_duration(self, audio_path: str) -> float:
        """Get audio file duration."""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", audio_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            info = json.loads(result.stdout)
            return float(info.get("format", {}).get("duration", 0))
        return 0.0

    def unload(self) -> None:
        """Unload diarization pipeline."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
