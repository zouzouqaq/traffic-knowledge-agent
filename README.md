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
