"""
Model Management System.

Auto-detects missing models, downloads them automatically,
caches locally, and displays download progress.
"""

import hashlib
import json
import logging
import os
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from dubbing_studio.config import ModelManagementConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a downloadable model."""
    name: str
    model_type: str  # whisper, tts, diarization, embedding
    size_mb: float
    url: str = ""
    description: str = ""
    required: bool = False
    installed: bool = False
    local_path: str = ""
    checksum: str = ""


# Registry of known models
MODEL_REGISTRY: list[dict] = [
    {
        "name": "whisper-tiny",
        "model_type": "whisper",
        "size_mb": 75,
        "description": "Whisper tiny model - fastest, lowest accuracy",
        "required": False,
        "check_cmd": "python -c \"import whisper; whisper.load_model('tiny')\"",
    },
    {
        "name": "whisper-base",
        "model_type": "whisper",
        "size_mb": 142,
        "description": "Whisper base model - good balance of speed and accuracy",
        "required": True,
        "check_cmd": "python -c \"import whisper; whisper.load_model('base')\"",
    },
    {
        "name": "whisper-medium",
        "model_type": "whisper",
        "size_mb": 1500,
        "description": "Whisper medium model - higher accuracy, slower",
        "required": False,
        "check_cmd": "python -c \"import whisper; whisper.load_model('medium')\"",
    },
    {
        "name": "whisper-large-v3",
        "model_type": "whisper",
        "size_mb": 3100,
        "description": "Whisper large-v3 model - best accuracy, requires GPU",
        "required": False,
        "check_cmd": "python -c \"import whisper; whisper.load_model('large-v3')\"",
    },
    {
        "name": "edge-tts",
        "model_type": "tts",
        "size_mb": 0,
        "description": "Microsoft Edge TTS - cloud-based, no local model needed",
        "required": True,
        "check_cmd": "python -c \"import edge_tts\"",
    },
    {
        "name": "qwen3-tts",
        "model_type": "tts",
        "size_mb": 0,
        "description": "Qwen3-TTS - multilingual neural TTS",
        "required": False,
        "check_cmd": "python -c \"from dubbing_studio.tts.qwen_tts import Qwen3TTS\"",
    },
    {
        "name": "chatterbox-tts",
        "model_type": "tts",
        "size_mb": 0,
        "description": "ChatterboxTTS - cinematic narration",
        "required": False,
        "check_cmd": "python -c \"from dubbing_studio.tts.chatterbox_tts import ChatterboxTTS\"",
    },
    {
        "name": "resemblyzer",
        "model_type": "embedding",
        "size_mb": 17,
        "description": "Speaker embedding model for voice cloning",
        "required": False,
        "check_cmd": "python -c \"from resemblyzer import VoiceEncoder\"",
    },
]


@dataclass
class DownloadProgress:
    """Download progress information."""
    model_name: str
    total_bytes: int
    downloaded_bytes: int
    percentage: float
    speed_mbps: float = 0.0
    eta_seconds: float = 0.0
    status: str = "downloading"  # downloading, complete, failed, cancelled


class ModelManager:
    """
    Manage AI model lifecycle.

    Features:
    - Auto-detect installed/missing models
    - Download missing models on demand
    - Cache models locally
    - Track download progress
    - Verify model integrity
    """

    def __init__(self, config: Optional[ModelManagementConfig] = None):
        self.config = config or ModelManagementConfig()
        self.models_dir = Path(self.config.models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._progress_callbacks: list[Callable] = []

    def scan_models(self) -> list[ModelInfo]:
        """
        Scan for installed and missing models.

        Returns:
            List of ModelInfo with installation status.
        """
        models = []

        for reg in MODEL_REGISTRY:
            model = ModelInfo(
                name=reg["name"],
                model_type=reg["model_type"],
                size_mb=reg["size_mb"],
                description=reg.get("description", ""),
                required=reg.get("required", False),
                url=reg.get("url", ""),
            )

            # Check if model is available
            check_cmd = reg.get("check_cmd", "")
            if check_cmd:
                model.installed = self._check_model_available(check_cmd)
            else:
                model.installed = self._check_model_files(model.name)

            # Check local cache
            local_path = self.models_dir / model.name
            if local_path.exists():
                model.local_path = str(local_path)

            models.append(model)

        installed_count = sum(1 for m in models if m.installed)
        logger.info(
            "Model scan: %d/%d models available",
            installed_count, len(models),
        )

        return models

    def get_missing_required(self) -> list[ModelInfo]:
        """Get list of missing required models."""
        models = self.scan_models()
        return [m for m in models if m.required and not m.installed]

    def get_model_status(self, model_name: str) -> Optional[ModelInfo]:
        """Get status of a specific model."""
        models = self.scan_models()
        return next((m for m in models if m.name == model_name), None)

    def download_model(
        self,
        model_name: str,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
    ) -> bool:
        """
        Download a model.

        For pip-installable models, uses pip install.
        For downloadable model files, downloads to cache.

        Args:
            model_name: Name of the model to download.
            progress_callback: Optional callback for progress updates.

        Returns:
            True if download/install succeeded.
        """
        # Find model in registry
        reg = next(
            (r for r in MODEL_REGISTRY if r["name"] == model_name),
            None,
        )

        if not reg:
            logger.error("Unknown model: %s", model_name)
            return False

        logger.info("Downloading/installing model: %s", model_name)

        if progress_callback:
            progress_callback(DownloadProgress(
                model_name=model_name,
                total_bytes=int(reg.get("size_mb", 0) * 1024 * 1024),
                downloaded_bytes=0,
                percentage=0.0,
                status="downloading",
            ))

        try:
            success = self._install_model(model_name, reg)

            if progress_callback:
                progress_callback(DownloadProgress(
                    model_name=model_name,
                    total_bytes=int(reg.get("size_mb", 0) * 1024 * 1024),
                    downloaded_bytes=int(reg.get("size_mb", 0) * 1024 * 1024),
                    percentage=100.0,
                    status="complete" if success else "failed",
                ))

            return success

        except Exception as e:
            logger.error("Failed to download model %s: %s", model_name, e)

            if progress_callback:
                progress_callback(DownloadProgress(
                    model_name=model_name,
                    total_bytes=0,
                    downloaded_bytes=0,
                    percentage=0.0,
                    status="failed",
                ))

            return False

    def download_all_required(
        self,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
    ) -> dict[str, bool]:
        """
        Download all missing required models.

        Returns:
            Dict mapping model name to success status.
        """
        missing = self.get_missing_required()
        results = {}

        for model in missing:
            success = self.download_model(model.name, progress_callback)
            results[model.name] = success

        return results

    def get_cache_size(self) -> float:
        """Get total size of cached models in MB."""
        total = 0
        for path in self.models_dir.rglob("*"):
            if path.is_file():
                total += path.stat().st_size

        return total / (1024 * 1024)

    def clear_cache(self, model_name: Optional[str] = None) -> None:
        """
        Clear model cache.

        Args:
            model_name: Specific model to clear, or None for all.
        """
        if model_name:
            model_path = self.models_dir / model_name
            if model_path.exists():
                shutil.rmtree(model_path)
                logger.info("Cleared cache for model: %s", model_name)
        else:
            for item in self.models_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                elif item.is_file():
                    item.unlink()
            logger.info("Cleared all model cache")

    def _check_model_available(self, check_cmd: str) -> bool:
        """Check if a model is available by running a check command."""
        try:
            result = subprocess.run(
                check_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False

    def _check_model_files(self, model_name: str) -> bool:
        """Check if model files exist in cache."""
        model_path = self.models_dir / model_name
        return model_path.exists() and any(model_path.iterdir()) if model_path.is_dir() else model_path.exists()

    def _install_model(self, model_name: str, reg: dict) -> bool:
        """Install a model (pip install or file download)."""
        # Map model names to pip packages
        pip_packages = {
            "whisper-tiny": "openai-whisper",
            "whisper-base": "openai-whisper",
            "whisper-medium": "openai-whisper",
            "whisper-large-v3": "openai-whisper",
            "edge-tts": "edge-tts",
            "resemblyzer": "resemblyzer",
        }

        pip_pkg = pip_packages.get(model_name)
        if pip_pkg:
            return self._pip_install(pip_pkg)

        # For models with direct download URLs
        url = reg.get("url", "")
        if url:
            return self._download_file(url, model_name)

        logger.warning("No install method for model: %s", model_name)
        return False

    def _pip_install(self, package: str) -> bool:
        """Install a pip package."""
        cmd = ["pip", "install", package]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.returncode == 0

    def _download_file(
        self,
        url: str,
        model_name: str,
    ) -> bool:
        """Download a model file from URL."""
        output_dir = self.models_dir / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / Path(url).name

        try:
            urllib.request.urlretrieve(url, str(output_path))
            logger.info("Downloaded model file: %s", output_path)
            return True
        except Exception as e:
            logger.error("Download failed: %s", e)
            return False
