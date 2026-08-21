# Retrieval Evaluation Scope

This directory contains a deterministic regression benchmark for the first MVP.

- `corpus/` contains 50 isolated, reviewable traffic-knowledge sections.
- `questions.jsonl` contains 50 manually written questions in five balanced categories.
- Each relevance judgment references one or more chunk IDs recreated by the production
  Markdown loader and chunker with `max_characters=1000` and
  `overlap_characters=100`.
- `SOURCES.md` records the primary publications and project-generated evidence used
  to check corpus statements.

The questions intentionally cover the same terminology and facts as the fixed corpus.
The benchmark measures regression quality on this declared knowledge base; it is not
an unbiased estimate of performance on unseen user questions. Results must be reported
as fixed-set retrieval metrics and must retain the generated corpus/question hashes.
