from dubbing_studio.config import AppConfig


def test_appconfig_from_env(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "abc123")
    monkeypatch.setenv("WHISPER_MODEL", "auto")
    monkeypatch.setenv("DUBBING_OUTPUT_DIR", "out_dir")
    monkeypatch.setenv("DUBBING_TEMP_DIR", "tmp_dir")

    config = AppConfig.from_env()

    assert config.translation.api_key == "abc123"
    assert config.whisper.model_size == "auto"
    assert config.output_dir == "out_dir"
    assert config.temp_dir == "tmp_dir"
