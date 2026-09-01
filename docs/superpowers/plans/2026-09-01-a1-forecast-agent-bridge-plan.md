# A1 Forecast Agent Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Connect the knowledge Agent to the existing GRU and Historical Average prediction service through explicit, reproducible input windows.

**Architecture:** `traffic-ops-agent` owns model inference and exposes a stable JSON contract for both models. `traffic-knowledge-agent` forwards `forecast_model`, `forecast_inputs`, and target time features through its bounded forecast tool; DeepSeek only explains trusted results.

**Tech Stack:** Python 3.11, FastAPI, NumPy, PyTorch, httpx, pytest, Ruff.

## Global Constraints

- Work only in the two user-owned project directories through tmux session `b`.
- Do not use GPU, `systemctl`, Docker restart, or files outside the project directories.
- Do not change deterministic intent routing or the three-tool boundary.
- Forecast inputs are explicitly supplied by the caller; no automatic real-time or Kunming data claim.
- Every API result reports model, dataset, shapes, prediction summary and inference time.
- All failures return stable safe error codes and never expose credentials.

### Task 1: Historical Average Forecast API

**Files:** `traffic-ops-agent/ai-service/src/traffic_ai/api/app.py`, `tests/test_historical_average_api.py`

- [ ] Write tests for successful shape/summary response and invalid time-feature input.
- [ ] Run the focused tests and confirm the endpoint is absent.
- [ ] Add a loaded Historical Average service to `create_app` and implement `POST /v1/forecast/historical-average`.
- [ ] Run focused tests, all AI-service tests and Ruff.
- [ ] Commit the API contract.

### Task 2: Unified GRU response metadata

**Files:** `traffic-ops-agent/ai-service/src/traffic_ai/api/app.py`, `tests/test_gru_api.py`

- [ ] Add a regression assertion for `dataset` and `inference_time_ms` in the GRU response.
- [ ] Implement the smallest response extension without changing prediction values.
- [ ] Run API tests and Ruff, then commit.

### Task 3: Knowledge Agent forecast payload bridge

**Files:** `traffic-knowledge-agent/src/traffic_knowledge/api/app.py`, `src/traffic_knowledge/integrations/forecast_client.py`, tests for API/client/graph.

- [ ] Add a failing test proving target time features and selected model are forwarded.
- [ ] Extend `ChatRequest` and `ForecastClient.forecast` with explicit model/input/time-feature fields.
- [ ] Preserve `partial=true` and `FORECAST_UNAVAILABLE` for offline services.
- [ ] Run focused and full regressions, then commit.

### Task 4: End-to-end A1 acceptance

- [ ] Start the GRU service in its existing tmux window using CPU only.
- [ ] Send one explicit-window GRU request through the Agent API.
- [ ] Add/fit Historical Average service data without changing other users' files.
- [ ] Verify success, output shape, inference time, and DeepSeek/evidence explanation.
- [ ] Record commands and measured values in `docs/mvp-acceptance.md` and push both branches.
