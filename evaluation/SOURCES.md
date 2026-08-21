# Evaluation Corpus Sources

The checked-in corpus is a concise, Chinese-language fact sheet. It is not a copy of
the referenced papers. Each statement must remain traceable to one of the sources
below or to a project-generated artifact.

## Project-owned evidence

- PEMS04 array shapes and dtypes:
  `/8t/usr/zhouh2024/zouz/Drive/data/PEMS04/samples_12_12_5.npz`
  SHA256: `43da74dbc85021a4785015a79b74f35f46eb512be4f13045537e6f44a5d1c980`.
  Verify with `numpy.load`; the dataset itself is not tracked by this repository.
- Historical Average metrics:
  `/8t/usr/zhouh2024/projects/traffic-ops-agent/ai-service/artifacts/historical_average_metrics.json`
  SHA256: `6e9074447e51a1de9cf337d4983a8333ebe0ff45557a3063d10b9d1c2ca9b56c`.
- GRU metrics:
  `/8t/usr/zhouh2024/projects/traffic-ops-agent/ai-service/artifacts/gru_best_metrics.json`
  SHA256: `1db91a759eb6e0a52269214ee4c63c8ae9c129d6d0711bbb09085da9e7fb4a09`.
- Data validation, scaling, ingestion, retrieval, citation and agent behavior:
  the implementation and automated tests in `traffic-ops-agent` and this repository.
  The inspected `traffic-ops-agent` commit is
  `e12ad4f2f856f172a95bfe1a4d1490319e153625` with a clean worktree.

The absolute paths identify the evidence used for this run. Reproduction on another
machine must provide the same files by content, not assume the same path.

## Primary publications

- STGCN: Bing Yu, Haoteng Yin and Zhanxing Zhu, “Spatio-Temporal Graph
  Convolutional Networks: A Deep Learning Framework for Traffic Forecasting,”
  IJCAI 2018. https://www.ijcai.org/proceedings/2018/0505
- DCRNN: Yaguang Li et al., “Diffusion Convolutional Recurrent Neural Network:
  Data-Driven Traffic Forecasting,” ICLR 2018.
  https://openreview.net/forum?id=SJiHXGWAZ
- Graph WaveNet: Zonghan Wu et al., “Graph WaveNet for Deep Spatial-Temporal
  Graph Modeling,” IJCAI 2019. https://arxiv.org/abs/1906.00121

## Curation rules

- The 50 relevance judgments in `questions.jsonl` were mapped to actual chunk IDs
  after the five corpus files were ingested with 1000 maximum characters and 100
  overlap characters.
- Each source section intentionally produces one chunk under that configuration.
- No model score is included unless it exists in a project-generated JSON artifact.
- PEMS04 results must not be described as real-time Kunming traffic performance.
