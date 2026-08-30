import pytest

from traffic_knowledge.settings import Settings


def test_settings_create_runtime_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAFFIC_KNOWLEDGE_DATA_DIR", str(tmp_path))

    settings = Settings.from_env()
    settings.ensure_directories()

    assert settings.database_path == tmp_path / "metadata.sqlite3"
    assert settings.chroma_path == tmp_path / "chroma"
    assert settings.chroma_path.is_dir()


def test_settings_use_safe_network_defaults(monkeypatch):
    monkeypatch.delenv("TRAFFIC_FORECAST_BASE_URL", raising=False)

    settings = Settings.from_env()

    assert settings.forecast_base_url == "http://127.0.0.1:18000"


def test_settings_default_to_evidence_only_answers(monkeypatch):
    monkeypatch.delenv("TRAFFIC_ANSWER_MODE", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    settings = Settings.from_env()

    assert settings.answer_mode == "evidence"
    assert settings.deepseek_api_key is None
    assert settings.deepseek_model == "deepseek-v4-flash"


def test_deepseek_mode_requires_api_key(monkeypatch):
    monkeypatch.setenv("TRAFFIC_ANSWER_MODE", "deepseek")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        Settings.from_env()


def test_deepseek_settings_are_loaded(monkeypatch):
    monkeypatch.setenv("TRAFFIC_ANSWER_MODE", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

    settings = Settings.from_env()

    assert settings.deepseek_api_key == "test-key"
    assert settings.deepseek_timeout_seconds == 20
    assert settings.deepseek_temperature == pytest.approx(0.2)
    assert settings.deepseek_max_output_tokens == 800
