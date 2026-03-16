"""Tests for model manager module."""

from dubbing_studio.models.manager import (
    ModelInfo,
    ModelManager,
    ModelStatus,
    MODEL_REGISTRY,
)


class TestModelStatus:
    def test_all_statuses(self):
        assert ModelStatus.NOT_INSTALLED.value == "not_installed"
        assert ModelStatus.DOWNLOADING.value == "downloading"
        assert ModelStatus.INSTALLED.value == "installed"
        assert ModelStatus.LOADED.value == "loaded"
        assert ModelStatus.ERROR.value == "error"


class TestModelInfo:
    def test_creation(self):
        info = ModelInfo(
            model_id="test-model",
            name="Test Model",
            category="tts",
            description="A test model",
            size_mb=100,
        )
        assert info.model_id == "test-model"
        assert info.status == ModelStatus.NOT_INSTALLED
        assert info.required is False


class TestModelRegistry:
    def test_has_whisper_models(self):
        whisper_models = [
            k for k in MODEL_REGISTRY if k.startswith("whisper-")
        ]
        assert len(whisper_models) >= 3

    def test_has_tts_models(self):
        tts_models = [
            k for k, v in MODEL_REGISTRY.items() if v.category == "tts"
        ]
        assert len(tts_models) >= 3

    def test_has_translation_model(self):
        translation_models = [
            k for k, v in MODEL_REGISTRY.items()
            if v.category == "translation"
        ]
        assert len(translation_models) >= 1

    def test_required_models_exist(self):
        required = [
            k for k, v in MODEL_REGISTRY.items() if v.required
        ]
        assert len(required) >= 2


class TestModelManager:
    def test_initialization(self, tmp_path):
        mm = ModelManager(cache_dir=str(tmp_path / "cache"))
        assert (tmp_path / "cache").exists()

    def test_get_all_models(self, tmp_path):
        mm = ModelManager(cache_dir=str(tmp_path / "cache"))
        models = mm.get_all_models()
        assert len(models) == len(MODEL_REGISTRY)

    def test_get_model(self, tmp_path):
        mm = ModelManager(cache_dir=str(tmp_path / "cache"))
        model = mm.get_model("edge-tts")
        assert model is not None
        assert model.name == "Microsoft Edge TTS"

    def test_get_model_nonexistent(self, tmp_path):
        mm = ModelManager(cache_dir=str(tmp_path / "cache"))
        assert mm.get_model("nonexistent") is None

    def test_get_models_by_category(self, tmp_path):
        mm = ModelManager(cache_dir=str(tmp_path / "cache"))
        whisper = mm.get_models_by_category("whisper")
        assert len(whisper) >= 3
        for m in whisper:
            assert m.category == "whisper"

    def test_get_installed_models(self, tmp_path):
        mm = ModelManager(cache_dir=str(tmp_path / "cache"))
        installed = mm.get_installed_models()
        # edge-tts and google-generativeai should be installed
        installed_ids = [m.model_id for m in installed]
        assert "edge-tts" in installed_ids

    def test_get_status_report(self, tmp_path):
        mm = ModelManager(cache_dir=str(tmp_path / "cache"))
        report = mm.get_status_report()
        assert "Model Manager Status Report" in report
        assert "WHISPER" in report
        assert "TTS" in report
        assert "TRANSLATION" in report

    def test_get_cache_size(self, tmp_path):
        mm = ModelManager(cache_dir=str(tmp_path / "cache"))
        size = mm.get_cache_size_mb()
        assert size >= 0.0

    def test_clear_cache(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "test_file").write_text("data")
        mm = ModelManager(cache_dir=str(cache_dir))
        mm.clear_cache()
        assert cache_dir.exists()
        assert not (cache_dir / "test_file").exists()

    def test_refresh(self, tmp_path):
        mm = ModelManager(cache_dir=str(tmp_path / "cache"))
        mm.refresh()  # should not raise
        models = mm.get_all_models()
        assert len(models) > 0
