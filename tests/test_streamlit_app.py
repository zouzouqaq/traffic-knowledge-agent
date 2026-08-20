import importlib.util
import json
from pathlib import Path

import httpx
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_app_module():
    path = PROJECT_ROOT / "streamlit_app.py"
    spec = importlib.util.spec_from_file_location("traffic_streamlit_app", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_module_imports_without_starting_streamlit_and_exposes_four_views():
    module = _load_app_module()

    assert module.VIEW_NAMES == (
        "文档管理",
        "知识问答",
        "预测分析",
        "模型基准",
    )
    assert callable(module.render_document_management)
    assert callable(module.render_cited_qa)
    assert callable(module.render_forecast_analysis)
    assert callable(module.render_benchmark)


def test_api_url_is_configurable(monkeypatch):
    monkeypatch.setenv("TRAFFIC_KNOWLEDGE_API_URL", "http://127.0.0.1:19000/")

    module = _load_app_module()

    assert module.configured_api_url() == "http://127.0.0.1:19000"


def test_module_does_not_import_agent_or_repository_internals():
    source = (PROJECT_ROOT / "streamlit_app.py").read_text(encoding="utf-8")

    assert "traffic_knowledge" not in source
    assert "sqlite" not in source.lower()
    assert "chromadb" not in source.lower()


def test_http_client_calls_only_document_chat_and_benchmark_endpoints():
    module = _load_app_module()
    requests = []

    def handler(request):
        requests.append((request.method, request.url.path))
        if request.url.path == "/documents" and request.method == "POST":
            return httpx.Response(201, json={"document_id": "d1", "duplicate": False})
        if request.url.path == "/documents" and request.method == "GET":
            return httpx.Response(200, json={"documents": []})
        if request.url.path == "/chat":
            payload = json.loads(request.read())
            return httpx.Response(
                200,
                json={
                    "answer": payload["question"],
                    "citations": [],
                    "tool_calls": [],
                    "partial": False,
                    "errors": [],
                },
            )
        if request.url.path == "/benchmarks/latest":
            return httpx.Response(200, json={"dataset": "PEMS04", "models": []})
        raise AssertionError(request.url)

    client = module.TrafficApiClient(
        "http://127.0.0.1:18100",
        transport=httpx.MockTransport(handler),
    )

    client.upload_document("guide.md", b"content", "text/markdown")
    client.list_documents()
    client.chat("What is MAE?")
    client.run_forecast("Predict traffic", "gru", [[[[1.0]]]])
    client.latest_benchmark()

    assert requests == [
        ("POST", "/documents"),
        ("GET", "/documents"),
        ("POST", "/chat"),
        ("POST", "/chat"),
        ("GET", "/benchmarks/latest"),
    ]


def test_http_client_raises_typed_error_from_api_envelope():
    module = _load_app_module()

    def handler(request):
        return httpx.Response(
            503,
            json={
                "error": {
                    "code": "FORECAST_UNAVAILABLE",
                    "message": "service down",
                    "details": {},
                }
            },
        )

    client = module.TrafficApiClient(
        "http://127.0.0.1:18100",
        transport=httpx.MockTransport(handler),
    )

    try:
        client.latest_benchmark()
    except module.ApiClientError as error:
        assert error.code == "FORECAST_UNAVAILABLE"
        assert error.status_code == 503
    else:
        raise AssertionError("ApiClientError was not raised")


def _run_app(source: str) -> AppTest:
    app = AppTest.from_string(source)
    app.run(timeout=10)
    assert not app.exception
    return app


def test_document_renderer_lists_documents_without_widget_errors():
    app = _run_app(
        """
from streamlit_app import render_document_management
class Client:
    def list_documents(self):
        return {"documents": [{"document_id": "d1", "filename": "guide.md", "status": "indexed"}]}
    def delete_document(self, document_id):
        return None
render_document_management(Client())
"""
    )

    assert "guide.md" in [item.value for item in app.markdown]
    assert any(button.label == "删除" for button in app.button)


def test_cited_qa_renderer_shows_source_excerpt():
    app = AppTest.from_string(
        """
from streamlit_app import render_cited_qa
class Client:
    def chat(self, question):
        return {
            "answer": "MAE measures error [S1].",
            "citations": [{
                "label": "S1", "filename": "guide.md", "location": "section 1",
                "excerpt": "MAE is an error metric.", "chunk_id": "c1"
            }],
            "partial": False, "errors": []
        }
render_cited_qa(Client())
"""
    ).run(timeout=10)
    app.text_area[0].set_value("What is MAE?")
    app.button[0].click().run(timeout=10)

    assert not app.exception
    assert any("MAE measures error" in item.value for item in app.markdown)
    assert app.expander[0].label.startswith("[S1] guide.md")


def test_forecast_renderer_reports_invalid_json_without_calling_api():
    app = AppTest.from_string(
        """
from streamlit_app import render_forecast_analysis
class Client:
    def run_forecast(self, question, model, inputs):
        raise AssertionError("must not be called")
render_forecast_analysis(Client())
"""
    ).run(timeout=10)
    app.text_area[0].set_value("{")
    app.button[0].click().run(timeout=10)

    assert not app.exception
    assert "INPUT_JSON_INVALID" in app.error[0].value


def test_partial_agent_result_displays_actionable_error_message():
    app = _run_app(
        """
from streamlit_app import _render_agent_result
_render_agent_result({
    "answer": "Knowledge is available.", "citations": [], "partial": True,
    "errors": [{
        "tool": "run_traffic_forecast", "code": "FORECAST_UNAVAILABLE",
        "message": "service down"
    }]
})
"""
    )

    visible_text = [item.value for item in app.caption]
    assert any("service down" in value for value in visible_text)


def test_benchmark_renderer_shows_retrieval_and_model_comparisons():
    app = _run_app(
        """
from streamlit_app import render_benchmark
class Client:
    def latest_benchmark(self):
        return {
            "dataset": "PEMS04", "split": "test", "created_at": "2026-08-20T00:00:00+08:00",
            "horizon": {"steps": 12, "interval_minutes": 5}, "environment": {"device": "cpu"},
            "retrieval": [{"name": "hybrid", "hit_at_1": 0.8, "hit_at_3": 0.9, "mrr": 0.85}],
            "models": [{"name": "gru", "mae": 27.1, "rmse": 41.0, "mape": 23.0}]
        }
render_benchmark(Client())
"""
    )

    assert len(app.dataframe) == 2
    headings = [item.value for item in app.subheader]
    assert "检索效果" in headings
    assert "预测效果" in headings
