"""Streamlit demonstration UI backed exclusively by the FastAPI service."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
import streamlit as st

VIEW_NAMES = (
    "文档管理",
    "知识问答",
    "预测分析",
    "模型基准",
)


class ApiClientError(RuntimeError):
    def __init__(self, code: str, message: str, status_code: int) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(f"{code}: {message}")


class TrafficApiClient:
    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 60,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    def _request(self, method: str, path: str, **kwargs):
        try:
            with httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = client.request(method, path, **kwargs)
        except httpx.HTTPError as error:
            raise ApiClientError("API_UNAVAILABLE", str(error), 503) from error
        if response.is_error:
            try:
                error = response.json()["error"]
                code = str(error["code"])
                message = str(error["message"])
            except (KeyError, TypeError, ValueError):
                code = "API_REQUEST_FAILED"
                message = f"HTTP {response.status_code}"
            raise ApiClientError(code, message, response.status_code)
        if response.status_code == 204:
            return None
        return response.json()

    def health(self):
        return self._request("GET", "/health")

    def upload_document(self, filename: str, content: bytes, media_type: str):
        return self._request(
            "POST",
            "/documents",
            files={"file": (filename, content, media_type)},
        )

    def list_documents(self):
        return self._request("GET", "/documents")

    def delete_document(self, document_id: str):
        return self._request("DELETE", f"/documents/{document_id}")

    def chat(self, question: str):
        return self._request("POST", "/chat", json={"question": question})

    def run_forecast(self, question: str, model: str, inputs: list):
        return self._request(
            "POST",
            "/chat",
            json={
                "question": question,
                "forecast_model": model,
                "forecast_inputs": inputs,
            },
        )

    def latest_benchmark(self):
        return self._request("GET", "/benchmarks/latest")


def configured_api_url() -> str:
    return os.getenv(
        "TRAFFIC_KNOWLEDGE_API_URL", "http://127.0.0.1:18100"
    ).rstrip("/")


def _show_error(error: ApiClientError) -> None:
    st.error(f"{error.code}: {error.message}")


def _render_agent_result(result: dict[str, Any]) -> None:
    st.markdown(result.get("answer", ""))
    if result.get("partial"):
        st.warning("部分工具未能完成")
    for error in result.get("errors", []):
        st.caption(
            f'{error.get("tool", "tool")}: {error.get("code", "ERROR")} - '
            f'{error.get("message", "no details")}'
        )
    generation = result.get("generation", {})
    mode = generation.get("answer_mode", "evidence")
    model = generation.get("answer_model")
    caption = f"回答模式: {model or mode}"
    if generation.get("llm_fallback"):
        code = generation.get("llm_error_code", "LLM_ERROR")
        caption += f" · 已回退 ({code})"
    elif mode == "deepseek":
        duration = float(generation.get("duration_ms", 0))
        prompt_tokens = int(generation.get("prompt_tokens", 0))
        completion_tokens = int(generation.get("completion_tokens", 0))
        caption += (
            f" · {duration:.0f} ms"
            f" · tokens {prompt_tokens}+{completion_tokens}"
        )
    st.caption(caption)
    citations = result.get("citations", [])
    if citations:
        st.subheader("来源")
        for citation in citations:
            label = citation.get("label", "S")
            filename = citation.get("filename", "unknown")
            location = citation.get("location", "")
            with st.expander(f"[{label}] {filename} · {location}"):
                st.write(citation.get("excerpt", ""))
                st.caption(citation.get("chunk_id", ""))


def render_document_management(client: TrafficApiClient) -> None:
    st.header("文档管理")
    uploaded = st.file_uploader(
        "上传知识文档",
        type=["md", "pdf", "docx"],
        accept_multiple_files=False,
    )
    if uploaded is not None and st.button("加入知识库", type="primary"):
        try:
            result = client.upload_document(
                uploaded.name,
                uploaded.getvalue(),
                uploaded.type or "application/octet-stream",
            )
            if result.get("duplicate"):
                st.info("该文档已存在")
            else:
                st.success(f'已索引 {result.get("chunk_count", 0)} 个知识片段')
        except ApiClientError as error:
            _show_error(error)

    try:
        documents = client.list_documents().get("documents", [])
    except ApiClientError as error:
        _show_error(error)
        return
    if not documents:
        st.info("知识库暂无文档")
        return
    for document in documents:
        columns = st.columns([5, 2, 1])
        columns[0].write(document.get("filename", "unknown"))
        columns[1].caption(document.get("status", "unknown"))
        if columns[2].button(
            "删除",
            key=f'delete-{document.get("document_id")}',
            type="secondary",
        ):
            try:
                client.delete_document(document["document_id"])
                st.rerun()
            except ApiClientError as error:
                _show_error(error)


def render_cited_qa(client: TrafficApiClient) -> None:
    st.header("知识问答")
    question = st.text_area("问题", placeholder="例如: MAE 指标表示什么?")
    if st.button("查询", type="primary", disabled=not question.strip()):
        try:
            _render_agent_result(client.chat(question.strip()))
        except ApiClientError as error:
            _show_error(error)


def render_forecast_analysis(client: TrafficApiClient) -> None:
    st.header("预测分析")
    first, second = st.columns([2, 1])
    question = first.text_input("分析目标", value="预测未来一小时交通流")
    model = second.selectbox("模型", ("gru",))
    default_payload = json.dumps([[[[0.0]]]], ensure_ascii=False)
    payload_text = st.text_area("四维输入数据", value=default_payload, height=140)
    if st.button("运行预测", type="primary"):
        try:
            inputs = json.loads(payload_text)
        except json.JSONDecodeError as error:
            st.error(f"INPUT_JSON_INVALID: {error.msg}")
            return
        try:
            _render_agent_result(client.run_forecast(question, model, inputs))
        except ApiClientError as error:
            _show_error(error)


def render_benchmark(client: TrafficApiClient) -> None:
    st.header("模型基准")
    try:
        benchmark = client.latest_benchmark()
    except ApiClientError as error:
        _show_error(error)
        return
    metadata = st.columns(3)
    metadata[0].metric("数据集", benchmark.get("dataset", "-"))
    metadata[1].metric("划分", benchmark.get("split", "-"))
    horizon = benchmark.get("horizon", {})
    metadata[2].metric(
        "预测范围",
        f'{horizon.get("steps", "-")} x {horizon.get("interval_minutes", "-")} min',
    )
    retrieval_rows = [
        {
            "方法": method.get("name"),
            "Hit@1": method.get("hit_at_1"),
            "Hit@3": method.get("hit_at_3"),
            "MRR": method.get("mrr"),
        }
        for method in benchmark.get("retrieval", [])
    ]
    st.subheader("检索效果")
    if retrieval_rows:
        st.dataframe(retrieval_rows, use_container_width=True, hide_index=True)
    else:
        st.info("尚无检索评估结果")

    model_rows = [
        {
            "模型": model.get("name"),
            "MAE": model.get("mae"),
            "RMSE": model.get("rmse"),
            "MAPE (%)": model.get("mape"),
        }
        for model in benchmark.get("models", [])
    ]
    st.subheader("预测效果")
    st.dataframe(model_rows, use_container_width=True, hide_index=True)
    environment = benchmark.get("environment", {})
    st.caption(
        f'生成时间 {benchmark.get("created_at", "-")} · '
        f'运行环境 {environment.get("device", "unknown")}'
    )


def _apply_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f6f7f8; color: #172126; }
        [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #dfe4e7; }
        [data-testid="stHeader"] { background: transparent; }
        .block-container { max-width: 1120px; padding-top: 2.2rem; }
        h1, h2, h3 { letter-spacing: 0; color: #172126; }
        div[data-testid="stMetric"] { border-left: 3px solid #157a6e; padding-left: 0.8rem; }
        .stButton button[kind="primary"] { background: #157a6e; border-color: #157a6e; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_health_status(status: str) -> None:
    if status == "ok":
        st.success("API 在线")
    else:
        st.warning("部分服务不可用")


def main() -> None:
    st.set_page_config(
        page_title="城市交通知识与预测 Agent",
        page_icon=":material/traffic:",
        layout="wide",
    )
    _apply_theme()
    client = TrafficApiClient(configured_api_url())
    st.title("城市交通知识与预测 Agent")
    with st.sidebar:
        st.caption(configured_api_url())
        try:
            health = client.health()
            status = health.get("status", "unknown")
            _render_health_status(status)
        except ApiClientError:
            st.error("API 离线")
        view = st.radio("视图", VIEW_NAMES)
    renderers = {
        "文档管理": render_document_management,
        "知识问答": render_cited_qa,
        "预测分析": render_forecast_analysis,
        "模型基准": render_benchmark,
    }
    renderers[view](client)


if __name__ == "__main__":
    main()
