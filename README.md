# Traffic Knowledge Agent

A traffic-domain knowledge retrieval and analysis Agent. This repository is
separate from `traffic-ops-agent`: it owns document ingestion, retrieval,
citations and tool orchestration, while the forecasting repository owns model
training and inference.

## Current milestone

Task 1 establishes a tested Python package and safe runtime configuration.
Document ingestion, retrieval and Agent behavior are implemented in later
milestones described in `docs/superpowers/plans/`.

## Requirements

- Python 3.11
- Linux or another environment supported by the listed Python packages
- No Docker or GPU is required for the project foundation

## Development setup

```bash
conda activate traffic-agent
python -m pip install -e ".[dev]"
pytest -v
ruff check .
```

Copy `.env.example` to `.env` only when local overrides are needed. The `.env`
file, runtime data, indexes, artifacts and model files are ignored by Git.

## Optional DeepSeek answers

The default `TRAFFIC_ANSWER_MODE=evidence` is fully offline and deterministic.
To let DeepSeek compose the final answer from retrieved citations and tool
results, create `.env`, set `TRAFFIC_ANSWER_MODE=deepseek`, and enter the API
key directly in your terminal as `DEEPSEEK_API_KEY`. Never paste a key into
chat, source code, shell history, screenshots or Git-tracked files.

Enter the key without terminal echo and write it only to the ignored `.env`:

```bash
cd /8t/usr/zhouh2024/projects/traffic-knowledge-agent/.worktrees/mvp
umask 077
read -rsp "DeepSeek API Key: " DEEPSEEK_API_KEY
printf '\n'
temporary_env=$(mktemp)
if [ -f .env ]; then
  grep -vE '^(DEEPSEEK_API_KEY|TRAFFIC_ANSWER_MODE|DEEPSEEK_MODEL)=' \
    .env > "$temporary_env"
fi
printf 'DEEPSEEK_API_KEY=%s\n' "$DEEPSEEK_API_KEY" >> "$temporary_env"
printf 'TRAFFIC_ANSWER_MODE=deepseek\n' >> "$temporary_env"
printf 'DEEPSEEK_MODEL=deepseek-v4-flash\n' >> "$temporary_env"
install -m 600 "$temporary_env" .env
rm -f "$temporary_env"
unset DEEPSEEK_API_KEY
```

DeepSeek does not choose tools and cannot add evidence. The existing Agent
still performs intent routing and calls only the bounded knowledge, metrics and
forecast tools. Invalid citations, authentication errors, rate limits, timeouts
and service failures automatically fall back to the deterministic evidence
answer; the UI reports the answer mode and safe fallback code.

With the API and DeepSeek mode already running, evaluate a bounded question
sample without storing question text, answer text, prompts or credentials:

```bash
python scripts/evaluate_deepseek_answers.py \
  --api-url http://127.0.0.1:18100 \
  --questions-path evaluation/questions.jsonl \
  --question-count 10 \
  --output-path artifacts/deepseek_answer_evaluation.json
python -m json.tool artifacts/deepseek_answer_evaluation.json
```

The report records success, fallback and citation-presence rates, latency
p50/p95, token totals, a conservative price estimate, Git state and the
question-file fingerprint. Pricing is a time-stamped estimate rather than a
billing statement.

## Runtime storage

By default, runtime files are stored under `./data`:

```text
data/
|-- metadata.sqlite3
`-- chroma/
```

Set `TRAFFIC_KNOWLEDGE_DATA_DIR` to use another project-owned location. Do not
point it at another user's directory on a shared server.

## Safety

Development services bind to `127.0.0.1`. The project does not restart Docker,
run `systemctl`, execute uploaded document instructions, or access unrelated
server files.

## MVP Reproduction

The following commands reproduce the current local MVP. They use the checked-in
five-document corpus and the local `BAAI/bge-small-zh-v1.5` model directory.

```bash
python scripts/ingest_documents.py evaluation/corpus/01_pems04_dataset.md \
  --database-path data/mvp/metadata.sqlite3
python scripts/evaluate_retrieval.py \
  --questions evaluation/questions.jsonl \
  --data-dir data/evaluation \
  --embedding-model /path/to/bge-small-zh-v1.5 \
  --output artifacts/retrieval_metrics.json
python scripts/run_benchmark.py \
  --questions evaluation/questions.jsonl \
  --corpus-dir evaluation/corpus \
  --output artifacts/agent_benchmark.json \
  --metrics-path tests/fixtures/metrics_snapshot.json \
  --embedding-model /path/to/bge-small-zh-v1.5 \
  --index-dir artifacts/benchmark_index \
  --top-k 5 --warmup-runs 5 --measured-runs 30 \
  --throughput-concurrency 4 --expected-question-count 50
python -m json.tool artifacts/retrieval_metrics.json
python -m json.tool artifacts/agent_benchmark.json
```

`run_benchmark.py` measures deterministic citation/tool metrics, serial
latency p50/p95, a separate fixed-concurrency throughput run, RSS sampled while
requests are running, and the persisted Chroma index size. It also records Git,
question/corpus/model/metrics fingerprints and dependency versions. Ragas or an
independent LLM judge is reported as `not_run` when no evaluator is configured.

The benchmark questions are a checked-in regression set co-designed with this
corpus; they demonstrate reproducibility, not generalization to unseen users.
PEMS04 is an academic benchmark and must not be presented as real-time Kunming
traffic. A Kunming deployment requires an authorized local historical data
source and a separately validated model.

## Evidence Map

| Resume claim | Evidence | Verification |
| --- | --- | --- |
| Hybrid retrieval combines BGE and BM25 | `artifacts/retrieval_metrics.json` | `scripts/evaluate_retrieval.py` |
| 50-question routing and citation evaluation | `artifacts/agent_benchmark.json` | `scripts/run_benchmark.py` |
| Reproducible environment and input provenance | benchmark `environment`, `input_fingerprints` | `python -m json.tool` |
| PEMS04 model metrics | `tests/fixtures/metrics_snapshot.json` | `scripts/run_benchmark.py` |
| Optional grounded DeepSeek answers and fallback | `artifacts/deepseek_answer_metrics.json` | `scripts/evaluate_deepseek_answers.py` |
