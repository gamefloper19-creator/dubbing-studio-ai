"""Tests for hardware optimization module."""

from unittest.mock import patch, MagicMock

from dubbing_studio.hardware.optimizer import HardwareInfo, HardwareOptimizer


class TestHardwareInfo:
    def test_creation(self):
        info = HardwareInfo(
            has_gpu=True,
            gpu_name="NVIDIA RTX 4090",
            gpu_memory_mb=24000,
            cpu_count=16,
            ram_gb=32.0,
            platform="Linux",
            recommended_whisper_model="large-v3",
            recommended_batch_size=8,
        )
        assert info.has_gpu is True
        assert info.gpu_name == "NVIDIA RTX 4090"
        assert info.gpu_memory_mb == 24000
        assert info.recommended_whisper_model == "large-v3"


class TestHardwareOptimizer:
    def setup_method(self):
        # Clear cached info between tests
        HardwareOptimizer._cached_info = None

    def test_has_gpu_no_torch_no_nvidia(self):
        opt = HardwareOptimizer()
        with patch.dict("sys.modules", {"torch": None}):
            with patch("subprocess.run", side_effect=FileNotFoundError):
                result = opt.has_gpu()
                assert result is False

    def test_get_optimal_device_no_gpu(self):
        opt = HardwareOptimizer()
        with patch.object(opt, "has_gpu", return_value=False):
            assert opt.get_optimal_device() == "cpu"

    def test_get_optimal_device_with_gpu(self):
        opt = HardwareOptimizer()
        with patch.object(opt, "has_gpu", return_value=True):
            assert opt.get_optimal_device() == "cuda"

    def test_get_optimal_dtype_cpu(self):
        opt = HardwareOptimizer()
        with patch.object(opt, "has_gpu", return_value=False):
            assert opt.get_optimal_dtype() == "float32"

    def test_get_optimal_dtype_gpu(self):
        opt = HardwareOptimizer()
        with patch.object(opt, "has_gpu", return_value=True):
            with patch.object(opt, "get_gpu_memory", return_value=16000):
                assert opt.get_optimal_dtype() == "float16"

    def test_detect_hardware_cpu_only(self):
        opt = HardwareOptimizer()
        with patch.object(opt, "has_gpu", return_value=False):
            with patch.object(opt, "_get_ram_gb", return_value=12.0):
                info = opt.detect_hardware()
                assert info.has_gpu is False
                assert info.recommended_whisper_model == "base"

    def test_detect_hardware_high_ram_cpu(self):
        opt = HardwareOptimizer()
        with patch.object(opt, "has_gpu", return_value=False):
            with patch.object(opt, "_get_ram_gb", return_value=32.0):
                info = opt.detect_hardware()
                assert info.recommended_whisper_model == "medium"

    def test_detect_hardware_low_ram_cpu(self):
        opt = HardwareOptimizer()
        with patch.object(opt, "has_gpu", return_value=False):
            with patch.object(opt, "_get_ram_gb", return_value=4.0):
                info = opt.detect_hardware()
                assert info.recommended_whisper_model == "tiny"

    def test_detect_hardware_caching(self):
        opt = HardwareOptimizer()
        with patch.object(opt, "has_gpu", return_value=False):
            with patch.object(opt, "_get_ram_gb", return_value=8.0):
                info1 = opt.detect_hardware()
                info2 = opt.detect_hardware()
                assert info1 is info2  # same object returned

    def test_optimize_batch_config(self):
        opt = HardwareOptimizer()
        with patch.object(opt, "has_gpu", return_value=False):
            with patch.object(opt, "_get_ram_gb", return_value=16.0):
                with patch.object(opt, "get_gpu_memory", return_value=None):
                    config = opt.optimize_batch_config(10)
                    assert "max_concurrent" in config
                    assert "device" in config
                    assert "dtype" in config
                    assert "whisper_model" in config
                    assert config["device"] == "cpu"

    def test_get_ram_gb_fallback(self):
        opt = HardwareOptimizer()
        with patch("builtins.open", side_effect=FileNotFoundError):
            with patch.dict("sys.modules", {"psutil": None}):
                ram = opt._get_ram_gb()
                # Should return either actual value or fallback
                assert ram > 0
