# DeepSeek Grounded Answers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional DeepSeek-based grounded answer generation while preserving deterministic routing, trusted tool outputs, citations, and evidence-only fallback.

**Architecture:** Existing LangGraph nodes continue to classify intent and execute the same three bounded tools. A new `GroundedAnswerGenerator` receives a structured context after tool execution; DeepSeek may rewrite that evidence into natural Chinese, while a deterministic generator remains the default and the fallback for every external-model failure. The API exposes generation metadata without exposing prompts or secrets.

**Tech Stack:** Python 3.11, FastAPI, LangGraph, httpx, Pydantic, Streamlit, pytest, Ruff, DeepSeek OpenAI-compatible Chat Completions API

## Global Constraints

- Default runtime mode remains `evidence`; external API calls require `TRAFFIC_ANSWER_MODE=deepseek`.
- Default DeepSeek model is `deepseek-v4-flash`, base URL `https://api.deepseek.com`, timeout 20 seconds, temperature 0.2, and maximum output 800 tokens.
- Send `{"thinking": {"type": "disabled"}}`; DeepSeek must not select or call tools.
- Never log, serialize, return, commit, or paste `DEEPSEEK_API_KEY`.
- The three existing tools, deterministic intent classifier, and maximum tool-call count remain unchanged.
- Invalid citations, empty responses, authentication errors, rate limits, timeouts, network failures, and 5xx responses fall back to evidence-only answers.
- `AgentResponse.partial` continues to represent tool failure only; LLM fallback is reported through generation metadata.
- Use CPU only; no GPU is required.
- Work only in `/8t/usr/zhouh2024/projects/traffic-knowledge-agent/.worktrees/mvp` through tmux session `b`.

---

## File Map

- Modify `src/traffic_knowledge/settings.py`: validate answer mode and DeepSeek settings.
- Create `src/traffic_knowledge/integrations/deepseek.py`: isolate HTTP protocol, retries, response parsing, and typed errors.
- Create `src/traffic_knowledge/application/grounded_answers.py`: build evidence prompts, validate citations, and perform deterministic fallback.
- Modify `src/traffic_knowledge/application/question_answering.py`: retain all retrieved citation evidence for final generation.
- Modify `src/traffic_knowledge/domain/agent.py`: add non-secret generation metadata to responses.
- Modify `src/traffic_knowledge/application/agent_graph.py`: delegate final composition to a generator without changing tools or routing.
- Modify `src/traffic_knowledge/api/dependencies.py`: construct evidence-only or DeepSeek generator from settings.
- Modify `streamlit_app.py`: show answer mode, model, fallback, latency, and token usage in a compact caption.
- Create `src/traffic_knowledge/evaluation/deepseek_runner.py`: calculate live-generation success, citation, latency, token, and cost-input statistics.
- Create `scripts/evaluate_deepseek_answers.py`: run the first 10 knowledge questions and write reproducible JSON.
- Modify `.env.example` and `README.md`: document secure configuration and startup.

---

### Task 1: Runtime Configuration

**Files:**
- Modify: `src/traffic_knowledge/settings.py`
- Modify: `.env.example`
- Test: `tests/test_settings.py`

**Interfaces:**
- Produces: `Settings.answer_mode`, `deepseek_api_key`, `deepseek_base_url`, `deepseek_model`, `deepseek_timeout_seconds`, `deepseek_temperature`, and `deepseek_max_output_tokens`.
- Consumes: environment variables listed in the design specification.

- [ ] **Step 1: Write failing configuration tests**

```python
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
```

- [ ] **Step 2: Run tests and confirm the missing fields fail**

Run: `.venv/bin/python -m pytest tests/test_settings.py -v`

Expected: failures showing `Settings` has no `answer_mode` and deepseek mode is not validated.

- [ ] **Step 3: Add validated fields to `Settings`**

```python
answer_mode = os.getenv("TRAFFIC_ANSWER_MODE", "evidence").strip().lower()
if answer_mode not in {"evidence", "deepseek"}:
    raise ValueError("TRAFFIC_ANSWER_MODE must be evidence or deepseek")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "").strip() or None
if answer_mode == "deepseek" and deepseek_api_key is None:
    raise ValueError("DEEPSEEK_API_KEY is required in deepseek mode")
deepseek_base_url = os.getenv(
    "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
).strip().rstrip("/")
if not deepseek_base_url:
    raise ValueError("DEEPSEEK_BASE_URL must not be empty")
```

Add these constructor values:

```python
answer_mode=answer_mode,
deepseek_api_key=deepseek_api_key,
deepseek_base_url=deepseek_base_url,
deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash").strip(),
deepseek_timeout_seconds=_positive_float("DEEPSEEK_TIMEOUT_SECONDS", "20"),
deepseek_temperature=_nonnegative_float("DEEPSEEK_TEMPERATURE", "0.2"),
deepseek_max_output_tokens=_positive_int("DEEPSEEK_MAX_OUTPUT_TOKENS", "800"),
```

- [ ] **Step 4: Add empty-key examples to `.env.example` and rerun tests**

```dotenv
TRAFFIC_ANSWER_MODE=evidence
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_TIMEOUT_SECONDS=20
DEEPSEEK_TEMPERATURE=0.2
DEEPSEEK_MAX_OUTPUT_TOKENS=800
```

Run: `.venv/bin/python -m pytest tests/test_settings.py -v && .venv/bin/ruff check src/traffic_knowledge/settings.py tests/test_settings.py`

Expected: all settings tests pass and Ruff reports `All checks passed!`.

- [ ] **Step 5: Commit**

```bash
git add .env.example src/traffic_knowledge/settings.py tests/test_settings.py
git commit -m "feat: configure optional DeepSeek answers"
```

---

### Task 2: Typed DeepSeek HTTP Client

**Files:**
- Create: `src/traffic_knowledge/integrations/deepseek.py`
- Create: `tests/test_deepseek_client.py`

**Interfaces:**
- Produces: `DeepSeekClient.generate(system_prompt: str, user_prompt: str) -> DeepSeekGeneration`.
- Produces: `DeepSeekGeneration(content, model, prompt_tokens, completion_tokens, duration_ms)`.
- Produces: `DeepSeekClientError(code: str, message: str, status_code: int | None)`.

- [ ] **Step 1: Write failing success and request-contract tests**

```python
def _valid_response(content="基于证据的答案 [S1]。"):
    return {
        "model": "deepseek-v4-flash",
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 18},
    }


def _client(transport):
    return DeepSeekClient(
        api_key="test-key",
        base_url="https://api.deepseek.com",
        model="deepseek-v4-flash",
        timeout_seconds=20,
        temperature=0.2,
        max_output_tokens=800,
        transport=transport,
    )


def test_client_disables_thinking_and_parses_usage():
    captured = []

    def handler(request):
        captured.append(json.loads(request.read()))
        return httpx.Response(
            200,
            json={
                "model": "deepseek-v4-flash",
                "choices": [{"message": {"content": "基于证据的答案 [S1]。"}}],
                "usage": {"prompt_tokens": 120, "completion_tokens": 18},
            },
        )

    client = _client(httpx.MockTransport(handler))
    result = client.generate("system", "evidence")

    assert captured[0]["thinking"] == {"type": "disabled"}
    assert captured[0]["temperature"] == pytest.approx(0.2)
    assert captured[0]["max_tokens"] == 800
    assert "test-key" not in json.dumps(captured[0])
    assert result.content == "基于证据的答案 [S1]。"
    assert result.prompt_tokens == 120
    assert result.completion_tokens == 18
```

- [ ] **Step 2: Write failing error mapping and retry tests**

```python
@pytest.mark.parametrize(
    ("status", "code"),
    [(401, "LLM_AUTH_FAILED"), (429, "LLM_RATE_LIMITED"), (503, "LLM_UNAVAILABLE")],
)
def test_client_maps_http_errors(status, code):
    client = _client(httpx.MockTransport(lambda request: httpx.Response(status)))

    with pytest.raises(DeepSeekClientError) as captured:
        client.generate("system", "evidence")

    assert captured.value.code == code


def test_client_retries_one_server_error_then_succeeds():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503)
        return httpx.Response(200, json=_valid_response())

    result = _client(httpx.MockTransport(handler)).generate("system", "evidence")

    assert calls == 2
    assert result.content
```

- [ ] **Step 3: Run tests and confirm the module is absent**

Run: `.venv/bin/python -m pytest tests/test_deepseek_client.py -v`

Expected: collection fails with `ModuleNotFoundError: traffic_knowledge.integrations.deepseek`.

- [ ] **Step 4: Implement the client and typed response**

```python
@dataclass(frozen=True)
class DeepSeekGeneration:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    duration_ms: float


class DeepSeekClientError(RuntimeError):
    def __init__(self, code: str, message: str, status_code: int | None = None):
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(f"{code}: {message}")


class DeepSeekClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float,
        temperature: float,
        max_output_tokens: int,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.transport = transport

    def generate(self, system_prompt: str, user_prompt: str) -> DeepSeekGeneration:
        started = time.perf_counter_ns()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "thinking": {"type": "disabled"},
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
            "stream": False,
        }
        response = self._post_with_one_retry(payload)
        try:
            content = response["choices"][0]["message"]["content"].strip()
            usage = response.get("usage", {})
        except (KeyError, IndexError, TypeError, AttributeError) as error:
            raise DeepSeekClientError(
                "LLM_INVALID_RESPONSE", "DeepSeek response schema is invalid"
            ) from error
        if not content:
            raise DeepSeekClientError("LLM_INVALID_RESPONSE", "DeepSeek answer is empty")
        return DeepSeekGeneration(
            content=content,
            model=str(response.get("model", self.model)),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            duration_ms=round((time.perf_counter_ns() - started) / 1_000_000, 3),
        )

    def _post_with_one_retry(self, payload: dict) -> dict:
        for attempt in range(2):
            try:
                with httpx.Client(
                    base_url=self.base_url,
                    timeout=self.timeout_seconds,
                    transport=self.transport,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                ) as client:
                    response = client.post("/chat/completions", json=payload)
            except httpx.TimeoutException as error:
                if attempt == 0:
                    continue
                raise DeepSeekClientError("LLM_TIMEOUT", "DeepSeek request timed out") from error
            except httpx.HTTPError as error:
                if attempt == 0:
                    continue
                raise DeepSeekClientError(
                    "LLM_UNAVAILABLE", "DeepSeek network request failed"
                ) from error

            retryable = response.status_code == 429 or response.status_code >= 500
            if retryable and attempt == 0:
                continue
            if response.status_code in {401, 403}:
                raise DeepSeekClientError(
                    "LLM_AUTH_FAILED", "DeepSeek authentication failed", response.status_code
                )
            if response.status_code == 429:
                raise DeepSeekClientError(
                    "LLM_RATE_LIMITED", "DeepSeek rate limit reached", response.status_code
                )
            if response.status_code >= 500:
                raise DeepSeekClientError(
                    "LLM_UNAVAILABLE", "DeepSeek service unavailable", response.status_code
                )
            if response.is_error:
                raise DeepSeekClientError(
                    "LLM_INVALID_RESPONSE", "DeepSeek rejected the request", response.status_code
                )
            try:
                return response.json()
            except ValueError as error:
                raise DeepSeekClientError(
                    "LLM_INVALID_RESPONSE", "DeepSeek returned invalid JSON"
                ) from error
        raise DeepSeekClientError("LLM_UNAVAILABLE", "DeepSeek request failed")
```

The internal request must use `Authorization: Bearer <key>`, catch `httpx.TimeoutException` as `LLM_TIMEOUT`, catch other `httpx.HTTPError` as `LLM_UNAVAILABLE`, retry one time only for transport errors, 429, and 5xx, and never include the key in exception messages.

- [ ] **Step 5: Run focused and static checks**

Run: `.venv/bin/python -m pytest tests/test_deepseek_client.py -v && .venv/bin/ruff check src/traffic_knowledge/integrations/deepseek.py tests/test_deepseek_client.py`

Expected: all DeepSeek client tests pass and Ruff is clean.

- [ ] **Step 6: Commit**

```bash
git add src/traffic_knowledge/integrations/deepseek.py tests/test_deepseek_client.py
git commit -m "feat: add typed DeepSeek client"
```

---

### Task 3: Grounded Answer Generators and Citation Fallback

**Files:**
- Create: `src/traffic_knowledge/application/grounded_answers.py`
- Modify: `src/traffic_knowledge/application/question_answering.py`
- Modify: `src/traffic_knowledge/domain/agent.py`
- Create: `tests/test_grounded_answers.py`
- Modify: `tests/test_question_answering.py`

**Interfaces:**
- Produces: `GroundedAnswerContext(question, knowledge, metrics, forecast, errors)`.
- Produces: `GroundedAnswerResult(answer, citations, generation)`.
- Produces: `AnswerGenerationMetadata` for both deterministic and DeepSeek results.
- Produces: `GroundedAnswerGenerator.generate(context) -> GroundedAnswerResult` protocol.
- Produces: `EvidenceOnlyAnswerGenerator.generate(context)` and `ResilientDeepSeekAnswerGenerator.generate(context)`.
- Consumes: `DeepSeekClient.generate()` from Task 2.

- [ ] **Step 1: Preserve all retrieved evidence separately from cited output**

Add a defaulted field to `AnswerResult`:

```python
@dataclass(frozen=True)
class AnswerResult:
    answer: str
    citations: tuple[Citation, ...]
    insufficient_evidence: bool
    elapsed_ms: float
    evidence: tuple[Citation, ...] = ()
```

When retrieval succeeds, return `evidence=citations`; insufficient responses return `evidence=()`.

Test:

```python
def test_answer_result_preserves_all_retrieved_evidence():
    model = RecordingChatModel(response="Only the first source is cited [S1].")
    service = QuestionAnsweringService(
        retriever=StaticRetriever((_hit("first", "First."), _hit("second", "Second."))),
        chat_model=model,
    )

    result = service.answer("Question")

    assert [item.label for item in result.citations] == ["S1"]
    assert [item.label for item in result.evidence] == ["S1", "S2"]
```

- [ ] **Step 2: Write failing generator tests**

```python
def test_deepseek_generator_uses_all_evidence_and_keeps_only_used_citations():
    client = RecordingDeepSeekClient("综合两个证据得到结论 [S1][S2]。")
    generator = ResilientDeepSeekAnswerGenerator(
        client=client,
        fallback=EvidenceOnlyAnswerGenerator(),
    )

    result = generator.generate(_knowledge_context())

    assert [citation.label for citation in result.citations] == ["S1", "S2"]
    assert result.generation.answer_mode == "deepseek"
    assert result.generation.llm_fallback is False
    assert "<evidence label=\"S2\"" in client.user_prompt


def test_invalid_deepseek_citation_falls_back_without_marking_tool_partial():
    client = RecordingDeepSeekClient("不存在的证据 [S99]。")
    result = ResilientDeepSeekAnswerGenerator(
        client=client,
        fallback=EvidenceOnlyAnswerGenerator(),
    ).generate(_knowledge_context())

    assert result.generation.answer_mode == "evidence"
    assert result.generation.llm_fallback is True
    assert result.generation.llm_error_code == "LLM_CITATION_INVALID"
    assert "[S99]" not in result.answer
```

- [ ] **Step 3: Implement deterministic and resilient generators**

```python
@dataclass(frozen=True)
class AnswerGenerationMetadata:
    answer_mode: str = "evidence"
    answer_model: str | None = None
    llm_fallback: bool = False
    llm_error_code: str | None = None
    duration_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass(frozen=True)
class GroundedAnswerContext:
    question: str
    knowledge: AnswerResult | None = None
    metrics: MetricsSnapshot | None = None
    forecast: ForecastResult | None = None
    errors: tuple[AgentError, ...] = ()


@dataclass(frozen=True)
class GroundedAnswerResult:
    answer: str
    citations: tuple[Citation, ...]
    generation: AnswerGenerationMetadata


class GroundedAnswerGenerator(Protocol):
    def generate(self, context: GroundedAnswerContext) -> GroundedAnswerResult: ...


class EvidenceOnlyAnswerGenerator:
    def generate(self, context: GroundedAnswerContext) -> GroundedAnswerResult:
        answer, citations = compose_evidence_only(context)
        return GroundedAnswerResult(
            answer=answer,
            citations=citations,
            generation=AnswerGenerationMetadata(answer_mode="evidence"),
        )


class ResilientDeepSeekAnswerGenerator:
    def generate(self, context: GroundedAnswerContext) -> GroundedAnswerResult:
        try:
            generated = self.client.generate(SYSTEM_PROMPT, build_prompt(context))
            citations = validate_and_select_citations(
                generated.content,
                context.knowledge.evidence if context.knowledge else (),
            )
            return GroundedAnswerResult(
                answer=generated.content,
                citations=citations,
                generation=AnswerGenerationMetadata(
                    answer_mode="deepseek",
                    answer_model=generated.model,
                    duration_ms=generated.duration_ms,
                    prompt_tokens=generated.prompt_tokens,
                    completion_tokens=generated.completion_tokens,
                ),
            )
        except (DeepSeekClientError, CitationValidationError) as error:
            fallback = self.fallback.generate(context)
            code = getattr(error, "code", "LLM_CITATION_INVALID")
            return replace(
                fallback,
                generation=replace(
                    fallback.generation,
                    llm_fallback=True,
                    llm_error_code=code,
                ),
            )
```

`build_prompt` must HTML-escape untrusted excerpts, serialize metric and forecast numbers exactly as provided, include tool errors, and contain no filesystem path or API key.

- [ ] **Step 4: Run focused tests**

Run: `.venv/bin/python -m pytest tests/test_question_answering.py tests/test_grounded_answers.py -v`

Expected: all evidence retention, prompt isolation, citation, successful generation, and fallback tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/traffic_knowledge/application/grounded_answers.py src/traffic_knowledge/application/question_answering.py src/traffic_knowledge/domain/agent.py tests/test_grounded_answers.py tests/test_question_answering.py
git commit -m "feat: generate grounded answers with fallback"
```

---

### Task 4: LangGraph Integration and Response Metadata

**Files:**
- Modify: `src/traffic_knowledge/domain/agent.py`
- Modify: `src/traffic_knowledge/application/agent_graph.py`
- Modify: `tests/test_agent_graph.py`
- Modify: `tests/test_api.py`

**Interfaces:**
- Produces: `AnswerGenerationMetadata` nested under `AgentResponse.generation`.
- Consumes: one `answer_generator` injected through `AgentDependencies`.

- [ ] **Step 1: Add failing graph tests without changing routing expectations**

```python
class RecordingAnswerGenerator:
    def __init__(self):
        self.contexts = []

    def generate(self, context):
        self.contexts.append(context)
        return GroundedAnswerResult(
            answer="统一生成的回答 [S1]。",
            citations=context.knowledge.citations if context.knowledge else (),
            generation=AnswerGenerationMetadata(
                answer_mode="deepseek",
                answer_model="deepseek-v4-flash",
                prompt_tokens=100,
                completion_tokens=20,
            ),
        )


def test_final_generator_receives_all_successful_tool_results():
    generator = RecordingAnswerGenerator()
    graph, _ = _graph("combined", answer_generator=generator)

    response = graph.invoke(_combined_request())["response"]

    assert generator.contexts[0].knowledge is not None
    assert generator.contexts[0].metrics is not None
    assert generator.contexts[0].forecast is not None
    assert response.generation.answer_mode == "deepseek"
    assert [call.name for call in response.tool_calls] == list(TOOL_NAMES)
```

- [ ] **Step 2: Add generation metadata to `AgentResponse` with a safe default**

```python
@dataclass(frozen=True)
class AgentResponse:
    answer: str
    citations: tuple[Citation, ...]
    tool_calls: tuple[ToolCallRecord, ...]
    partial: bool
    errors: tuple[AgentError, ...]
    generation: AnswerGenerationMetadata = field(
        default_factory=AnswerGenerationMetadata
    )
```

- [ ] **Step 3: Replace inline composition with the injected generator**

```python
context = GroundedAnswerContext(
    question=state["question"],
    knowledge=state.get("knowledge_result"),
    metrics=state.get("metrics_result"),
    forecast=state.get("forecast_result"),
    errors=tuple(state.get("errors", [])),
)
generated = dependencies.answer_generator.generate(context)
return {
    "response": AgentResponse(
        answer=generated.answer,
        citations=generated.citations,
        tool_calls=tuple(state.get("tool_calls", [])),
        partial=bool(context.errors),
        errors=context.errors,
        generation=generated.generation,
    )
}
```

- [ ] **Step 4: Assert API serialization includes safe metadata**

```python
assert response.json()["generation"] == {
    "answer_mode": "evidence",
    "answer_model": None,
    "llm_fallback": False,
    "llm_error_code": None,
    "duration_ms": 0.0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
}
assert "api_key" not in response.text.lower()
```

- [ ] **Step 5: Run graph and API tests**

Run: `.venv/bin/python -m pytest tests/test_agent_graph.py tests/test_api.py -v`

Expected: routing, tool limit, partial responses, citation serialization, and generation metadata all pass.

- [ ] **Step 6: Commit**

```bash
git add src/traffic_knowledge/domain/agent.py src/traffic_knowledge/application/agent_graph.py tests/test_agent_graph.py tests/test_api.py
git commit -m "feat: compose agent responses through answer generators"
```

---

### Task 5: Dependency Wiring and Streamlit Status

**Files:**
- Modify: `src/traffic_knowledge/api/dependencies.py`
- Modify: `streamlit_app.py`
- Modify: `tests/test_api.py`
- Modify: `tests/test_streamlit_app.py`

**Interfaces:**
- Consumes: Task 1 settings, Task 2 client, and Task 3 generators.
- Produces: evidence-only default and DeepSeek opt-in dependency graphs.

- [ ] **Step 1: Write failing dependency construction tests**

Extract a focused factory with signature `build_answer_generator(settings: Settings, transport: httpx.BaseTransport | None = None) -> GroundedAnswerGenerator`.

Test:

```python
def test_answer_generator_defaults_to_evidence_only(monkeypatch):
    settings = _settings(answer_mode="evidence", deepseek_api_key=None)

    generator = build_answer_generator(settings)

    assert isinstance(generator, EvidenceOnlyAnswerGenerator)


def test_deepseek_generator_wraps_evidence_fallback():
    settings = _settings(answer_mode="deepseek", deepseek_api_key="test-key")

    generator = build_answer_generator(
        settings,
        transport=httpx.MockTransport(lambda request: httpx.Response(503)),
    )

    assert isinstance(generator, ResilientDeepSeekAnswerGenerator)
```

- [ ] **Step 2: Implement factory and inject it into `AgentDependencies`**

```python
def build_answer_generator(settings: Settings, transport=None):
    fallback = EvidenceOnlyAnswerGenerator()
    if settings.answer_mode == "evidence":
        return fallback
    client = DeepSeekClient(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=settings.deepseek_model,
        timeout_seconds=settings.deepseek_timeout_seconds,
        temperature=settings.deepseek_temperature,
        max_output_tokens=settings.deepseek_max_output_tokens,
        transport=transport,
    )
    return ResilientDeepSeekAnswerGenerator(client=client, fallback=fallback)
```

- [ ] **Step 3: Add a compact generation caption to Streamlit**

```python
generation = result.get("generation", {})
mode = generation.get("answer_mode", "evidence")
model = generation.get("answer_model")
caption = f"回答模式: {model or mode}"
if generation.get("llm_fallback"):
    caption += f' · 已回退 ({generation.get("llm_error_code", "LLM_ERROR")})'
elif mode == "deepseek":
    caption += (
        f' · {generation.get("duration_ms", 0):.0f} ms'
        f' · tokens {generation.get("prompt_tokens", 0)}+'
        f'{generation.get("completion_tokens", 0)}'
    )
st.caption(caption)
```

- [ ] **Step 4: Test normal and fallback captions**

```python
def test_agent_result_displays_deepseek_generation_metadata():
    app = _run_app('''
from streamlit_app import _render_agent_result
_render_agent_result({
    "answer": "回答 [S1]。", "citations": [], "partial": False, "errors": [],
    "generation": {"answer_mode": "deepseek", "answer_model": "deepseek-v4-flash",
                   "llm_fallback": False, "duration_ms": 321,
                   "prompt_tokens": 120, "completion_tokens": 18}
})
''')
    assert any("deepseek-v4-flash" in item.value for item in app.caption)
    assert any("tokens 120+18" in item.value for item in app.caption)
```

- [ ] **Step 5: Run dependency, API, and UI tests**

Run: `.venv/bin/python -m pytest tests/test_api.py tests/test_streamlit_app.py -v`

Expected: all tests pass; no Streamlit internal object is rendered.

- [ ] **Step 6: Commit**

```bash
git add src/traffic_knowledge/api/dependencies.py streamlit_app.py tests/test_api.py tests/test_streamlit_app.py
git commit -m "feat: expose grounded answer generation status"
```

---

### Task 6: Reproducible DeepSeek Evaluation

**Files:**
- Create: `src/traffic_knowledge/evaluation/deepseek_runner.py`
- Create: `scripts/evaluate_deepseek_answers.py`
- Create: `tests/test_deepseek_evaluation.py`
- Modify: `.gitignore` only if `artifacts/` is no longer already ignored; otherwise leave it unchanged.

**Interfaces:**
- Produces: `evaluate_deepseek_answers(api_client, questions) -> dict`.
- Produces: ignored artifact `artifacts/deepseek_answer_metrics.json`.
- Consumes: first 10 `expected_tool == "knowledge"` rows from `evaluation/questions.jsonl`.
- Uses a conservative peak-price snapshot for `deepseek-v4-flash`: uncached input USD 0.44 per million tokens and output USD 1.32 per million tokens, with the official pricing URL stored in the artifact.

- [ ] **Step 1: Write failing aggregate-metrics test**

```python
def test_evaluation_aggregates_success_citations_latency_and_tokens():
    responses = [
        {
            "answer": "答案 [S1]。",
            "citations": [{"label": "S1"}],
            "generation": {
                "answer_mode": "deepseek",
                "llm_fallback": False,
                "duration_ms": 100,
                "prompt_tokens": 80,
                "completion_tokens": 20,
            },
        },
        {
            "answer": "模板答案 [S1]。",
            "citations": [{"label": "S1"}],
            "generation": {
                "answer_mode": "evidence",
                "llm_fallback": True,
                "duration_ms": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            },
        },
    ]

    metrics = summarize_responses(responses)

    assert metrics["request_count"] == 2
    assert metrics["deepseek_success_rate"] == pytest.approx(0.5)
    assert metrics["citation_presence_rate"] == pytest.approx(1.0)
    assert metrics["fallback_rate"] == pytest.approx(0.5)
    assert metrics["total_prompt_tokens"] == 80
    assert metrics["total_completion_tokens"] == 20
```

- [ ] **Step 2: Implement evaluation without storing generated answer text**

```python
def summarize_responses(responses: list[dict]) -> dict:
    count = len(responses)
    if count == 0:
        raise ValueError("responses must not be empty")
    successful = [
        item for item in responses
        if item["generation"]["answer_mode"] == "deepseek"
        and not item["generation"]["llm_fallback"]
    ]
    latencies = sorted(item["generation"]["duration_ms"] for item in successful)
    return {
        "request_count": count,
        "deepseek_success_rate": len(successful) / count,
        "citation_presence_rate": sum(bool(item["citations"]) for item in responses) / count,
        "fallback_rate": sum(item["generation"]["llm_fallback"] for item in responses) / count,
        "p50_generation_ms": percentile(latencies, 50) if latencies else None,
        "p95_generation_ms": percentile(latencies, 95) if latencies else None,
        "total_prompt_tokens": sum(item["generation"]["prompt_tokens"] for item in responses),
        "total_completion_tokens": sum(item["generation"]["completion_tokens"] for item in responses),
        "estimated_peak_cost_usd": (
            sum(item["generation"]["prompt_tokens"] for item in responses) / 1_000_000 * 0.44
            + sum(item["generation"]["completion_tokens"] for item in responses) / 1_000_000 * 1.32
        ),
    }
```

The artifact must also include UTC creation time, Git commit, `git_dirty`, SHA-256 of the questions file, model, base URL host without credentials, the 10 question IDs, price snapshot timestamp `2026-08-29`, rates, and `https://api-docs.deepseek.com/quick_start/pricing`. It must label the cost as a conservative peak estimate and must not include answers, prompts, excerpts, request headers, or the API key.

- [ ] **Step 3: Add CLI arguments and strict JSON output**

```text
--api-url http://127.0.0.1:18100
--questions-path evaluation/questions.jsonl
--question-count 10
--output-path artifacts/deepseek_answer_metrics.json
```

Run the test CLI against an in-process fake HTTP server and assert exit code 0, exactly 10 `/chat` requests, valid JSON, no `sk-` substring, and all provenance fields present.

- [ ] **Step 4: Run evaluation tests and commit**

Run: `.venv/bin/python -m pytest tests/test_deepseek_evaluation.py -v && .venv/bin/ruff check src/traffic_knowledge/evaluation/deepseek_runner.py scripts/evaluate_deepseek_answers.py tests/test_deepseek_evaluation.py`

Expected: all evaluation tests pass and Ruff is clean.

```bash
git add src/traffic_knowledge/evaluation/deepseek_runner.py scripts/evaluate_deepseek_answers.py tests/test_deepseek_evaluation.py
git commit -m "feat: evaluate DeepSeek grounded answers"
```

---

### Task 7: Documentation, Full Regression, and Live API Acceptance

**Files:**
- Modify: `README.md`
- Modify: `docs/mvp-acceptance.md`
- Runtime-only: `.env` and `artifacts/deepseek_answer_metrics.json` remain ignored.

**Interfaces:**
- Consumes: completed implementation from Tasks 1-6 and a newly generated private API key entered directly on the server.
- Produces: documented startup, real HTTP evidence, and a reproducible non-secret evaluation artifact.

- [ ] **Step 1: Document secure key entry without terminal echo**

Add this exact operational pattern to README, while instructing the user to type the value only at the terminal prompt:

```bash
cd /8t/usr/zhouh2024/projects/traffic-knowledge-agent/.worktrees/mvp
umask 077
read -rsp "DeepSeek API Key: " DEEPSEEK_API_KEY
printf '\n'
temporary_env=$(mktemp)
if [ -f .env ]; then
  grep -vE '^(DEEPSEEK_API_KEY|TRAFFIC_ANSWER_MODE|DEEPSEEK_MODEL)=' .env > "$temporary_env"
fi
printf 'DEEPSEEK_API_KEY=%s\n' "$DEEPSEEK_API_KEY" >> "$temporary_env"
printf 'TRAFFIC_ANSWER_MODE=deepseek\n' >> "$temporary_env"
printf 'DEEPSEEK_MODEL=deepseek-v4-flash\n' >> "$temporary_env"
install -m 600 "$temporary_env" .env
rm -f "$temporary_env"
unset DEEPSEEK_API_KEY
set -a
source .env
set +a
```

Do not use shell history commands that embed the key and do not print the variable.

- [ ] **Step 2: Run the complete offline regression before using the real key**

Run:

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
git diff --check
```

Expected: at least the existing 168 tests plus all new tests pass, Ruff is clean, and `git diff --check` prints nothing.

- [ ] **Step 3: Restart only the project-owned API and UI windows in tmux `b`**

Keep services bound to loopback. Do not run `systemctl`, restart Docker, or stop unrelated processes. Confirm:

```bash
curl -fsS http://127.0.0.1:18100/health
curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:18501
```

Expected: API returns `ok` or the existing forecast-only `degraded` state; UI returns `200`.

- [ ] **Step 4: Execute one successful answer and three failure drills**

Successful request:

```bash
curl -sS http://127.0.0.1:18100/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"当前 PEMS04 数据包含多少个交通节点?"}' \
  | .venv/bin/python -m json.tool
```

Expected: `generation.answer_mode` is `deepseek`, model is `deepseek-v4-flash`, citations are non-empty, and token counts are positive.

Repeat with an invalid key in a temporary process, an unreachable temporary base URL, and a 1-millisecond temporary timeout. Each response must contain an evidence answer, `llm_fallback=true`, the matching safe error code, and no secret value. Restore the real environment immediately after each isolated process ends.

- [ ] **Step 5: Run the 10-question live evaluation**

```bash
.venv/bin/python scripts/evaluate_deepseek_answers.py \
  --api-url http://127.0.0.1:18100 \
  --questions-path evaluation/questions.jsonl \
  --question-count 10 \
  --output-path artifacts/deepseek_answer_metrics.json
.venv/bin/python -m json.tool artifacts/deepseek_answer_metrics.json >/dev/null
```

Expected: 10 requests, valid JSON, citations present, DeepSeek success and fallback rates reported, latency and token totals reported, and no answer text or secret stored.

- [ ] **Step 6: Rerun the existing 50-question benchmark**

Use the repository's existing benchmark command from `README.md`. Expected: knowledge/metrics route accuracy remains 100%, tool-call success does not regress, and generated-answer latency is reported separately from the evidence-only baseline.

- [ ] **Step 7: Update acceptance evidence and commit**

Record measured values from the real artifacts without claiming unsupported quality improvements. Then run:

```bash
git add README.md docs/mvp-acceptance.md
git commit -m "docs: record DeepSeek acceptance results"
git push origin HEAD:main
git push origin feature/mvp
git status --short
```

Expected: both remote branches point to the final acceptance commit and the worktree is clean.

---

## Final Review Checklist

- [ ] Existing intent routes and `TOOL_NAMES` are byte-for-byte unchanged.
- [ ] External model calls occur only in `TRAFFIC_ANSWER_MODE=deepseek`.
- [ ] Thinking mode is disabled and no tool schema is sent to DeepSeek.
- [ ] All model-supplied citations are checked against the current evidence whitelist.
- [ ] Every modeled failure returns a deterministic answer and safe metadata.
- [ ] `partial` remains tied to tool failures rather than LLM fallback.
- [ ] API and Streamlit reveal no prompt, evidence body, path, header, or API key.
- [ ] Real artifacts contain provenance, latency, token counts, and no generated answer text.
- [ ] Resume remains unchanged until real DeepSeek acceptance metrics exist.
