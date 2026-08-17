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
