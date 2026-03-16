"""
Model Manager - Automatic model downloading and caching.

Handles downloading, caching, and lifecycle management
of all AI models used in the dubbing pipeline.
"""

import hashlib
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default cache directory
DEFAULT_CACHE_DIR = os.path.join(
    os.environ.get("DUBBING_CACHE_DIR", ""),
    "models",
) if os.environ.get("DUBBING_CACHE_DIR") else os.path.join(
    str(Path.home()), ".dubbing_studio", "models"
)


@dataclass
class ModelInfo:
    """Information about a downloadable model."""
    name: str
    description: str
    size_mb: int
    model_type: str  # whisper, tts, translation
    required: bool = False
    installed: bool = False
    path: str = ""


# Registry of known models
MODEL_REGISTRY: dict[str, dict] = {
    "whisper-tiny": {
        "name": "Whisper Tiny",
        "description": "Fastest speech recognition, lower accuracy",
        "size_mb": 75,
        "model_type": "whisper",
        "loader": "whisper",
        "model_id": "tiny",
    },
    "whisper-base": {
        "name": "Whisper Base",
        "description": "Good balance of speed and accuracy",
        "size_mb": 145,
        "model_type": "whisper",
        "loader": "whisper",
        "model_id": "base",
    },
    "whisper-medium": {
        "name": "Whisper Medium",
        "description": "High accuracy, moderate speed",
        "size_mb": 1500,
        "model_type": "whisper",
        "loader": "whisper",
        "model_id": "medium",
    },
    "whisper-large-v3": {
        "name": "Whisper Large V3",
        "description": "Highest accuracy, requires GPU",
        "size_mb": 3100,
        "model_type": "whisper",
        "loader": "whisper",
        "model_id": "large-v3",
    },
    "edge-tts": {
        "name": "Microsoft Edge TTS",
        "description": "Cloud-based TTS, no download needed",
        "size_mb": 0,
        "model_type": "tts",
        "loader": "edge_tts",
        "model_id": "edge-tts",
    },
    "qwen-tts": {
        "name": "Qwen3-TTS",
        "description": "Neural TTS for Asian and Middle Eastern languages",
        "size_mb": 4000,
        "model_type": "tts",
        "loader": "transformers",
        "model_id": "Qwen/Qwen2.5-TTS",
    },
    "chatterbox-tts": {
        "name": "Chatterbox TTS",
        "description": "Expressive TTS for cinematic narration",
        "size_mb": 2000,
        "model_type": "tts",
        "loader": "chatterbox",
        "model_id": "chatterbox",
    },
    "coqui-xtts-v2": {
        "name": "Coqui XTTS v2",
        "description": "Multilingual TTS for European languages",
        "size_mb": 3000,
        "model_type": "tts",
        "loader": "coqui",
        "model_id": "tts_models/multilingual/multi-dataset/xtts_v2",
    },
}


class ModelManager:
    """
    Manage AI model downloads, caching, and lifecycle.

    Features:
    - Automatic model downloading on first use
    - Disk-based caching
    - Model status checking
    - Disk space management
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._loaded_models: dict[str, object] = {}

    def get_model_status(self) -> list[ModelInfo]:
        """
        Get status of all known models.

        Returns:
            List of ModelInfo with installation status.
        """
        models = []
        for key, info in MODEL_REGISTRY.items():
            installed = self._is_model_installed(key, info)
            model_path = self._get_model_path(key, info)

            models.append(ModelInfo(
                name=info["name"],
                description=info["description"],
                size_mb=info["size_mb"],
                model_type=info["model_type"],
                required=info.get("required", False),
                installed=installed,
                path=str(model_path) if installed else "",
            ))

        return models

    def _is_model_installed(self, key: str, info: dict) -> bool:
        """Check if a model is installed/available."""
        loader = info.get("loader", "")

        if loader == "whisper":
            return self._check_whisper_model(info["model_id"])
        elif loader == "edge_tts":
            return self._check_package("edge_tts")
        elif loader == "transformers":
            return self._check_package("transformers")
        elif loader == "chatterbox":
            return self._check_package("chatterbox")
        elif loader == "coqui":
            return self._check_package("TTS")

        return False

    def _check_package(self, package_name: str) -> bool:
        """Check if a Python package is installed."""
        try:
            __import__(package_name)
            return True
        except ImportError:
            return False

    def _check_whisper_model(self, model_id: str) -> bool:
        """Check if a Whisper model is cached locally."""
        try:
            import whisper
            model_dir = os.path.join(
                os.path.expanduser("~"), ".cache", "whisper"
            )
            # Whisper stores models as .pt files
            model_file = os.path.join(model_dir, f"{model_id}.pt")
            return os.path.exists(model_file)
        except ImportError:
            return False

    def _get_model_path(self, key: str, info: dict) -> Path:
        """Get the expected path for a model."""
        return self.cache_dir / key

    def ensure_whisper_model(self, model_size: str) -> str:
        """
        Ensure a Whisper model is downloaded and available.

        Downloads automatically if not present.

        Args:
            model_size: Whisper model size (tiny, base, medium, large-v3).

        Returns:
            Model name string.
        """
        try:
            import whisper
        except ImportError:
            raise ImportError(
                "OpenAI Whisper is not installed. "
                "Install with: pip install openai-whisper"
            )

        logger.info("Ensuring Whisper model '%s' is available...", model_size)

        # whisper.load_model will automatically download if needed
        # We just verify we can import and the model name is valid
        valid_models = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
        if model_size not in valid_models:
            logger.warning(
                "Unknown Whisper model '%s', defaulting to 'base'",
                model_size,
            )
            model_size = "base"

        logger.info("Whisper model '%s' ready (will download on first use)", model_size)
        return model_size

    def ensure_edge_tts(self) -> bool:
        """
        Ensure edge-tts is available.

        Returns:
            True if edge-tts is available.
        """
        try:
            import edge_tts
            logger.info("edge-tts is available")
            return True
        except ImportError:
            logger.warning(
                "edge-tts is not installed. "
                "Install with: pip install edge-tts"
            )
            return False

    def get_cache_size_mb(self) -> float:
        """Get total size of cached models in MB."""
        total = 0
        for path in self.cache_dir.rglob("*"):
            if path.is_file():
                total += path.stat().st_size

        return total / (1024 * 1024)

    def clear_cache(self) -> None:
        """Clear all cached models."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Model cache cleared")

    def get_recommended_models(self, has_gpu: bool, gpu_memory_mb: int = 0) -> list[str]:
        """
        Get recommended models based on hardware.

        Args:
            has_gpu: Whether a GPU is available.
            gpu_memory_mb: GPU memory in MB.

        Returns:
            List of recommended model keys.
        """
        recommended = ["edge-tts"]  # Always recommend edge-tts as fallback

        if has_gpu and gpu_memory_mb > 8000:
            recommended.extend(["whisper-large-v3", "qwen-tts", "chatterbox-tts"])
        elif has_gpu and gpu_memory_mb > 4000:
            recommended.extend(["whisper-medium", "chatterbox-tts"])
        elif has_gpu:
            recommended.extend(["whisper-base"])
        else:
            recommended.extend(["whisper-base"])

        return recommended

    def preload_essential_models(self) -> dict[str, bool]:
        """
        Check and report on essential model availability.

        Returns:
            Dict mapping model name to availability status.
        """
        results = {}

        # Check Whisper
        try:
            import whisper
            results["whisper"] = True
            logger.info("Whisper: Available")
        except ImportError:
            results["whisper"] = False
            logger.warning("Whisper: Not installed")

        # Check edge-tts
        try:
            import edge_tts
            results["edge-tts"] = True
            logger.info("edge-tts: Available")
        except ImportError:
            results["edge-tts"] = False
            logger.warning("edge-tts: Not installed")

        # Check Google Generative AI
        try:
            import google.generativeai
            results["gemini"] = True
            logger.info("Google Gemini: Available")
        except ImportError:
            results["gemini"] = False
            logger.warning("Google Gemini: Not installed")

        # Check optional engines
        for name, import_path in [
            ("transformers", "transformers"),
            ("chatterbox", "chatterbox"),
            ("coqui-tts", "TTS"),
        ]:
            try:
                __import__(import_path)
                results[name] = True
                logger.info("%s: Available", name)
            except ImportError:
                results[name] = False
                logger.info("%s: Not installed (optional)", name)

        return results
