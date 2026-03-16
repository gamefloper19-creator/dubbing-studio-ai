"""
Hardware detection and optimization for model loading.
"""

import logging
import os
import platform
import subprocess
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class HardwareInfo:
    """System hardware information."""
    has_gpu: bool
    gpu_name: str
    gpu_memory_mb: int
    cpu_count: int
    ram_gb: float
    platform: str
    recommended_whisper_model: str
    recommended_batch_size: int


class HardwareOptimizer:
    """Detect hardware capabilities and optimize model loading."""

    _cached_info: Optional[HardwareInfo] = None

    def detect_hardware(self) -> HardwareInfo:
        """
        Detect system hardware and return recommendations.

        Returns:
            HardwareInfo with capabilities and recommendations.
        """
        if HardwareOptimizer._cached_info is not None:
            return HardwareOptimizer._cached_info

        gpu_available = self.has_gpu()
        gpu_name = self._get_gpu_name() if gpu_available else "None"
        gpu_memory = self.get_gpu_memory() or 0
        cpu_count = os.cpu_count() or 4
        ram_gb = self._get_ram_gb()

        # Recommend Whisper model based on hardware
        if gpu_available and gpu_memory > 8000:
            whisper_model = "large-v3"
            batch_size = 8
        elif gpu_available and gpu_memory > 4000:
            whisper_model = "medium"
            batch_size = 4
        elif gpu_available:
            whisper_model = "base"
            batch_size = 2
        elif ram_gb > 16:
            whisper_model = "medium"
            batch_size = 2
        elif ram_gb > 8:
            whisper_model = "base"
            batch_size = 2
        else:
            whisper_model = "tiny"
            batch_size = 1

        info = HardwareInfo(
            has_gpu=gpu_available,
            gpu_name=gpu_name,
            gpu_memory_mb=gpu_memory,
            cpu_count=cpu_count,
            ram_gb=ram_gb,
            platform=platform.system(),
            recommended_whisper_model=whisper_model,
            recommended_batch_size=batch_size,
        )

        HardwareOptimizer._cached_info = info

        logger.info(
            "Hardware detected: GPU=%s (%s, %dMB), CPU=%d cores, RAM=%.1fGB",
            gpu_available, gpu_name, gpu_memory, cpu_count, ram_gb,
        )
        logger.info(
            "Recommendations: Whisper=%s, Batch=%d",
            whisper_model, batch_size,
        )

        return info

    def has_gpu(self) -> bool:
        """Check if CUDA GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            pass

        # Fallback: check nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def get_gpu_memory(self) -> Optional[int]:
        """Get GPU memory in MB."""
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                return props.total_mem // (1024 * 1024)
        except ImportError:
            pass

        # Fallback: nvidia-smi
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return int(result.stdout.strip().split("\n")[0])
        except (FileNotFoundError, ValueError):
            pass

        return None

    def _get_gpu_name(self) -> str:
        """Get GPU name."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0)
        except ImportError:
            pass

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return result.stdout.strip().split("\n")[0]
        except FileNotFoundError:
            pass

        return "Unknown"

    def _get_ram_gb(self) -> float:
        """Get total system RAM in GB."""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024 ** 3)
        except ImportError:
            pass

        # Fallback for Linux
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        return kb / (1024 ** 2)
        except (FileNotFoundError, ValueError):
            pass

        return 8.0  # default assumption

    def get_optimal_device(self) -> str:
        """Get optimal device string for model loading."""
        if self.has_gpu():
            return "cuda"
        return "cpu"

    def get_optimal_dtype(self) -> str:
        """Get optimal data type for model loading."""
        if self.has_gpu():
            gpu_mem = self.get_gpu_memory()
            if gpu_mem and gpu_mem > 8000:
                return "float16"
            return "float16"
        return "float32"

    def optimize_batch_config(self, total_items: int) -> dict:
        """
        Get optimal batch processing configuration.

        Args:
            total_items: Total number of items to process.

        Returns:
            Dict with batch processing parameters.
        """
        info = self.detect_hardware()

        max_concurrent = info.recommended_batch_size
        if total_items <= max_concurrent:
            max_concurrent = total_items

        return {
            "max_concurrent": max_concurrent,
            "device": self.get_optimal_device(),
            "dtype": self.get_optimal_dtype(),
            "whisper_model": info.recommended_whisper_model,
        }
