"""
Voice cloning system.

Allows users to provide a short voice sample (10-60 seconds)
to create a narrator voice model for full narration.
"""

import json
import logging
import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from dubbing_studio.config import VoiceCloningConfig
from dubbing_studio.tts.engine import TTSEngine, TTSResult

logger = logging.getLogger(__name__)


@dataclass
class VoiceProfile:
    """A stored voice profile from a cloned voice sample."""
    profile_id: str
    name: str
    sample_path: str
    embedding_path: str
    duration: float  # duration of original sample
    created_at: str = ""
    gender: str = "unknown"
    language: str = "en"
    description: str = ""
    metadata: dict = field(default_factory=dict)


class VoiceProfileManager:
    """Manage locally stored voice profiles."""

    def __init__(self, config: Optional[VoiceCloningConfig] = None):
        self.config = config or VoiceCloningConfig()
        self.profiles_dir = Path(self.config.profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    def list_profiles(self) -> list[VoiceProfile]:
        """List all saved voice profiles."""
        profiles = []
        index_file = self.profiles_dir / "profiles.json"

        if index_file.exists():
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for entry in data.get("profiles", []):
                    profiles.append(VoiceProfile(**entry))
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning("Failed to load profiles index: %s", e)

        return profiles

    def save_profile(self, profile: VoiceProfile) -> None:
        """Save a voice profile to the index."""
        profiles = self.list_profiles()

        # Remove existing profile with same ID
        profiles = [p for p in profiles if p.profile_id != profile.profile_id]
        profiles.append(profile)

        self._write_index(profiles)
        logger.info("Saved voice profile: %s (%s)", profile.name, profile.profile_id)

    def delete_profile(self, profile_id: str) -> bool:
        """Delete a voice profile."""
        profiles = self.list_profiles()
        profile = next((p for p in profiles if p.profile_id == profile_id), None)

        if not profile:
            return False

        # Remove files
        for path in [profile.sample_path, profile.embedding_path]:
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                pass

        # Remove profile directory
        profile_dir = self.profiles_dir / profile_id
        if profile_dir.exists():
            shutil.rmtree(profile_dir, ignore_errors=True)

        # Update index
        profiles = [p for p in profiles if p.profile_id != profile_id]
        self._write_index(profiles)

        logger.info("Deleted voice profile: %s", profile_id)
        return True

    def get_profile(self, profile_id: str) -> Optional[VoiceProfile]:
        """Get a specific voice profile by ID."""
        profiles = self.list_profiles()
        return next((p for p in profiles if p.profile_id == profile_id), None)

    def _write_index(self, profiles: list[VoiceProfile]) -> None:
        """Write the profiles index file."""
        index_file = self.profiles_dir / "profiles.json"
        data = {"profiles": [asdict(p) for p in profiles]}

        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


class VoiceCloner:
    """
    Clone voices from audio samples.

    Extracts speaker embedding from a short voice sample
    and generates synthetic speech matching the original voice.
    """

    def __init__(self, config: Optional[VoiceCloningConfig] = None):
        self.config = config or VoiceCloningConfig()
        self.profile_manager = VoiceProfileManager(config)
        self._tts_model = None

    def create_voice_profile(
        self,
        sample_path: str,
        name: str,
        gender: str = "unknown",
        language: str = "en",
        description: str = "",
    ) -> VoiceProfile:
        """
        Create a voice profile from an audio sample.

        Args:
            sample_path: Path to voice sample audio (10-60 seconds).
            name: Display name for the voice profile.
            gender: Gender of the voice (male/female/unknown).
            language: Primary language of the voice.
            description: Optional description.

        Returns:
            VoiceProfile with stored embedding.

        Raises:
            ValueError: If sample duration is out of range.
        """
        sample_path = str(Path(sample_path).resolve())

        # Validate sample duration
        duration = self._get_audio_duration(sample_path)
        if duration < self.config.min_sample_duration:
            raise ValueError(
                f"Voice sample too short ({duration:.1f}s). "
                f"Minimum is {self.config.min_sample_duration}s."
            )
        if duration > self.config.max_sample_duration:
            raise ValueError(
                f"Voice sample too long ({duration:.1f}s). "
                f"Maximum is {self.config.max_sample_duration}s."
            )

        # Generate profile ID
        profile_id = str(uuid.uuid4())[:8]
        profile_dir = Path(self.config.profiles_dir) / profile_id
        profile_dir.mkdir(parents=True, exist_ok=True)

        # Preprocess and copy sample
        processed_sample = str(profile_dir / "sample.wav")
        self._preprocess_sample(sample_path, processed_sample)

        # Extract speaker embedding
        embedding_path = str(profile_dir / "embedding.npy")
        self._extract_embedding(processed_sample, embedding_path)

        import datetime
        profile = VoiceProfile(
            profile_id=profile_id,
            name=name,
            sample_path=processed_sample,
            embedding_path=embedding_path,
            duration=duration,
            created_at=datetime.datetime.now().isoformat(),
            gender=gender,
            language=language,
            description=description,
        )

        self.profile_manager.save_profile(profile)

        logger.info(
            "Created voice profile: %s (duration=%.1fs, id=%s)",
            name, duration, profile_id,
        )

        return profile

    def generate_speech_with_clone(
        self,
        text: str,
        profile_id: str,
        output_path: str,
        language: str = "en",
        speed: float = 1.0,
    ) -> TTSResult:
        """
        Generate speech using a cloned voice profile.

        Args:
            text: Text to synthesize.
            profile_id: Voice profile ID to use.
            output_path: Output audio path.
            language: Language code.
            speed: Speech speed multiplier.

        Returns:
            TTSResult with generated audio info.
        """
        profile = self.profile_manager.get_profile(profile_id)
        if not profile:
            raise ValueError(f"Voice profile not found: {profile_id}")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Try using XTTS-v2 for voice cloning (supports reference audio)
        try:
            return self._generate_with_xtts(
                text, profile, output_path, language, speed
            )
        except Exception as e:
            logger.warning("XTTS voice cloning failed: %s, using edge-tts fallback", e)

        # Fallback to edge-tts (without cloning, best available voice)
        return self._generate_with_fallback(text, output_path, language, speed)

    def _generate_with_xtts(
        self,
        text: str,
        profile: VoiceProfile,
        output_path: str,
        language: str,
        speed: float,
    ) -> TTSResult:
        """Generate speech using XTTS-v2 with voice cloning."""
        try:
            from TTS.api import TTS
            import torch
        except ImportError:
            raise ImportError(
                "Coqui TTS is required for voice cloning. "
                "Install with: pip install TTS"
            )

        if self._tts_model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

        self._tts_model.tts_to_file(
            text=text,
            speaker_wav=profile.sample_path,
            language=language,
            file_path=output_path,
            speed=speed,
        )

        duration = self._get_audio_duration(output_path)

        return TTSResult(
            audio_path=output_path,
            duration=duration,
            sample_rate=24000,
            text=text,
        )

    def _generate_with_fallback(
        self,
        text: str,
        output_path: str,
        language: str,
        speed: float,
    ) -> TTSResult:
        """Fallback to edge-tts without voice cloning."""
        try:
            import edge_tts
            import asyncio
        except ImportError:
            raise ImportError("edge-tts is required. Install with: pip install edge-tts")

        voice_map = {
            "en": "en-US-ChristopherNeural",
            "hi": "hi-IN-MadhurNeural",
            "es": "es-ES-AlvaroNeural",
            "fr": "fr-FR-HenriNeural",
            "de": "de-DE-ConradNeural",
        }

        voice = voice_map.get(language, "en-US-ChristopherNeural")
        rate_str = f"+{int((speed - 1) * 100)}%" if speed >= 1 else f"{int((speed - 1) * 100)}%"

        async def _generate():
            communicate = edge_tts.Communicate(text, voice, rate=rate_str)
            await communicate.save(output_path)

        asyncio.run(_generate())

        duration = self._get_audio_duration(output_path)

        return TTSResult(
            audio_path=output_path,
            duration=duration,
            sample_rate=24000,
            text=text,
        )

    def _preprocess_sample(self, input_path: str, output_path: str) -> None:
        """Preprocess voice sample for embedding extraction."""
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ar", "16000",
            "-ac", "1",
            "-af", "highpass=f=80,lowpass=f=8000,loudnorm=I=-23:TP=-1.5:LRA=11",
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Sample preprocessing failed: {result.stderr}")

    def _extract_embedding(self, audio_path: str, output_path: str) -> None:
        """
        Extract speaker embedding from audio.

        Tries resemblyzer first, falls back to saving raw audio features.
        """
        try:
            from resemblyzer import VoiceEncoder, preprocess_wav
            import numpy as np

            encoder = VoiceEncoder()
            wav = preprocess_wav(Path(audio_path))
            embedding = encoder.embed_utterance(wav)
            np.save(output_path, embedding)

            logger.info("Speaker embedding extracted using resemblyzer")
            return

        except ImportError:
            logger.info("resemblyzer not available, using basic feature extraction")

        # Fallback: save basic audio statistics as embedding proxy
        import struct

        cmd = [
            "ffmpeg",
            "-i", audio_path,
            "-af", "volumedetect",
            "-f", "null", "-",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        stderr = result.stderr

        features = {"mean_vol": -25.0, "max_vol": -10.0}
        for line in stderr.split("\n"):
            if "mean_volume" in line:
                try:
                    features["mean_vol"] = float(line.split(":")[-1].strip().replace(" dB", ""))
                except (ValueError, IndexError):
                    pass
            elif "max_volume" in line:
                try:
                    features["max_vol"] = float(line.split(":")[-1].strip().replace(" dB", ""))
                except (ValueError, IndexError):
                    pass

        # Save as JSON alongside the .npy path
        json_path = output_path.replace(".npy", ".json")
        with open(json_path, "w") as f:
            json.dump(features, f)

        # Create a placeholder numpy file
        try:
            import numpy as np
            np.save(output_path, [features["mean_vol"], features["max_vol"]])
        except ImportError:
            Path(output_path).touch()

        logger.info("Basic audio features extracted as embedding proxy")

    def _get_audio_duration(self, path: str) -> float:
        """Get audio file duration."""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            info = json.loads(result.stdout)
            return float(info.get("format", {}).get("duration", 0))
        return 0.0

    def unload(self) -> None:
        """Unload TTS model."""
        if self._tts_model is not None:
            del self._tts_model
            self._tts_model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
