"""
Model Manager - Download, cache, and manage AI models.

Provides centralized management for all AI models used by the dubbing pipeline:
- Whisper (speech recognition)
- TTS engines (Qwen3-TTS, Chatterbox, LuxTTS)
- Translation models
"""

import logging
import os
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Status of a managed model."""
    NOT_INSTALLED = "not_installed"
    DOWNLOADING = "downloading"
    INSTALLED = "installed"
    LOADED = "loaded"
    ERROR = "error"


@dataclass
class ModelInfo:
    """Information about a managed model."""
    model_id: str
    name: str
    category: str  # whisper, tts, translation
    description: str
    size_mb: int  # approximate download size in MB
    status: ModelStatus = ModelStatus.NOT_INSTALLED
    install_command: str = ""
    required: bool = False
    error_message: str = ""


# Registry of all models used by the dubbing pipeline
MODEL_REGISTRY: dict[str, ModelInfo] = {
    # Whisper models
    "whisper-tiny": ModelInfo(
        model_id="whisper-tiny",
        name="Whisper Tiny",
        category="whisper",
        description="Fastest, least accurate speech recognition (39M params)",
        size_mb=75,
        install_command="pip install openai-whisper",
        required=True,
    ),
    "whisper-base": ModelInfo(
        model_id="whisper-base",
        name="Whisper Base",
        category="whisper",
        description="Good balance of speed and accuracy (74M params)",
        size_mb=140,
        install_command="pip install openai-whisper",
        required=True,
    ),
    "whisper-medium": ModelInfo(
        model_id="whisper-medium",
        name="Whisper Medium",
        category="whisper",
        description="High accuracy, moderate speed (769M params)",
        size_mb=1500,
        install_command="pip install openai-whisper",
    ),
    "whisper-large-v3": ModelInfo(
        model_id="whisper-large-v3",
        name="Whisper Large v3",
        category="whisper",
        description="Highest accuracy, GPU recommended (1550M params)",
        size_mb=3000,
        install_command="pip install openai-whisper",
    ),
    # TTS engines
    "edge-tts": ModelInfo(
        model_id="edge-tts",
        name="Microsoft Edge TTS",
        category="tts",
        description="Cloud-based TTS, universal fallback for all engines (no local model)",
        size_mb=0,
        install_command="pip install edge-tts",
        required=True,
    ),
    "qwen3-tts": ModelInfo(
        model_id="qwen3-tts",
        name="Qwen3-TTS",
        category="tts",
        description="Neural TTS for Asian/Middle Eastern languages (requires GPU)",
        size_mb=4000,
        install_command="pip install transformers torch",
    ),
    "chatterbox-tts": ModelInfo(
        model_id="chatterbox-tts",
        name="Chatterbox TTS",
        category="tts",
        description="Expressive TTS for cinematic narration (requires GPU)",
        size_mb=2000,
        install_command="pip install chatterbox-tts",
    ),
    "luxtts": ModelInfo(
        model_id="luxtts",
        name="LuxTTS (Coqui XTTS-v2)",
        category="tts",
        description="Calm, neutral narration for European languages",
        size_mb=3000,
        install_command="pip install TTS",
    ),
    # Translation
    "gemini-translation": ModelInfo(
        model_id="gemini-translation",
        name="Google Gemini Translation",
        category="translation",
        description="Cloud-based semantic translation via Gemini API (requires API key)",
        size_mb=0,
        install_command="pip install google-generativeai",
        required=True,
    ),
}


class ModelManager:
    """
    Centralized model management for the dubbing pipeline.

    Handles:
    - Checking which models are installed
    - Reporting model status
    - Providing install instructions
    - Managing cache directories
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or os.environ.get(
            "DUBBING_CACHE_DIR", "cache"
        ))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._models: dict[str, ModelInfo] = {}
        self._refresh_registry()

    def _refresh_registry(self) -> None:
        """Refresh model registry with current status."""
        for model_id, info in MODEL_REGISTRY.items():
            model = ModelInfo(
                model_id=info.model_id,
                name=info.name,
                category=info.category,
                description=info.description,
                size_mb=info.size_mb,
                install_command=info.install_command,
                required=info.required,
            )
            model.status = self._check_model_status(model_id)
            self._models[model_id] = model

    def _check_model_status(self, model_id: str) -> ModelStatus:
        """Check if a model is installed and available."""
        try:
            if model_id.startswith("whisper-"):
                import whisper
                return ModelStatus.INSTALLED
            elif model_id == "edge-tts":
                import edge_tts
                return ModelStatus.INSTALLED
            elif model_id == "qwen3-tts":
                from transformers import AutoModelForCausalLM
                return ModelStatus.INSTALLED
            elif model_id == "chatterbox-tts":
                from chatterbox.tts import ChatterboxTTS
                return ModelStatus.INSTALLED
            elif model_id == "luxtts":
                from TTS.api import TTS
                return ModelStatus.INSTALLED
            elif model_id == "gemini-translation":
                import google.generativeai
                return ModelStatus.INSTALLED
        except ImportError:
            return ModelStatus.NOT_INSTALLED
        except Exception as e:
            logger.warning("Error checking model %s: %s", model_id, e)
            return ModelStatus.ERROR

        return ModelStatus.NOT_INSTALLED

    def get_all_models(self) -> list[ModelInfo]:
        """Get information about all registered models."""
        return list(self._models.values())

    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        return self._models.get(model_id)

    def get_models_by_category(self, category: str) -> list[ModelInfo]:
        """Get models filtered by category."""
        return [m for m in self._models.values() if m.category == category]

    def get_installed_models(self) -> list[ModelInfo]:
        """Get all installed models."""
        return [
            m for m in self._models.values()
            if m.status in (ModelStatus.INSTALLED, ModelStatus.LOADED)
        ]

    def get_missing_required(self) -> list[ModelInfo]:
        """Get required models that are not installed."""
        return [
            m for m in self._models.values()
            if m.required and m.status == ModelStatus.NOT_INSTALLED
        ]

    def get_status_report(self) -> str:
        """Generate a human-readable status report of all models."""
        lines = ["Model Manager Status Report", "=" * 40]

        for category in ["whisper", "tts", "translation"]:
            lines.append(f"\n{category.upper()} Models:")
            lines.append("-" * 30)

            for model in self.get_models_by_category(category):
                status_icon = {
                    ModelStatus.INSTALLED: "[OK]",
                    ModelStatus.LOADED: "[LOADED]",
                    ModelStatus.NOT_INSTALLED: "[NOT INSTALLED]",
                    ModelStatus.DOWNLOADING: "[DOWNLOADING]",
                    ModelStatus.ERROR: "[ERROR]",
                }.get(model.status, "[?]")

                required_tag = " (REQUIRED)" if model.required else ""
                lines.append(
                    f"  {status_icon} {model.name}{required_tag}"
                )
                if model.status == ModelStatus.NOT_INSTALLED:
                    lines.append(
                        f"         Install: {model.install_command}"
                    )
                if model.size_mb > 0:
                    lines.append(
                        f"         Size: ~{model.size_mb} MB"
                    )

        # Summary
        installed = len(self.get_installed_models())
        total = len(self._models)
        missing_required = self.get_missing_required()

        lines.append(f"\n{'=' * 40}")
        lines.append(f"Installed: {installed}/{total}")
        if missing_required:
            lines.append(
                f"MISSING REQUIRED: {', '.join(m.name for m in missing_required)}"
            )
        else:
            lines.append("All required models are installed.")

        return "\n".join(lines)

    def get_cache_size_mb(self) -> float:
        """Get total cache directory size in MB."""
        total = 0
        for path in self.cache_dir.rglob("*"):
            if path.is_file():
                total += path.stat().st_size
        return total / (1024 * 1024)

    def clear_cache(self) -> None:
        """Clear the model cache directory."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Model cache cleared")

    def refresh(self) -> None:
        """Refresh model status (re-check what's installed)."""
        self._refresh_registry()
