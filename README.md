# FinDisputeEval

**Research tooling for source-grounded financial-dispute intake evaluation.**

This portfolio demonstrates a traceable workflow from complaint-source analysis and seed curation to synthetic dialogue generation, evaluation experiments, and documented limitations. It focuses on English-language dispute intake and case clarification, not refund decisions, fraud liability determinations, legal advice, or access to bank accounts.

## Start here

- [Methods: cleaning, selection, and rationale](docs/portfolio/METHODS.md)
- [Stage results and research limitations](docs/portfolio/RESULTS.md)
- [Reproducibility and notebook guide](docs/portfolio/REPRODUCIBILITY.md)
- [Data and publication policy](DATA_POLICY.md)

## What has been demonstrated

| Workstream | Evidence from local research runs | Status |
|---|---|---|
| Source analysis | Six source-EDA workstreams; CFPB v05.1 analyzes 205,589 unique complaint IDs | Executed locally |
| Seed curation | 2,000 proportional-core + 358 coverage + 646 enrichment candidates | Structural checks completed; privacy clearance incomplete |
| Generation engineering | NeMo Data Designer orchestrating Nemotron Super; latest 20-case exploration | Executed locally; 19/20 format-conforming, not a quality pass rate |
| Evaluation tooling | Deterministic validation, blind semantic-judge runners, failure audits, review workbooks, linguistic comparisons | Implemented; no judge approved for automatic release |
| Next research question | Can a fixed-fact clarification task produce more useful case handoffs? | Proposed five-case experiment; not executed |

This is **not a finished benchmark or a released conversation dataset**. Aggregate results are stage observations, not independent validation of model quality or privacy safety.

## Engineering and research contributions

- Separate population and enrichment frames rather than conflating coverage with prevalence.
- Preserve raw, canonical, matching, and lexical text views for different purposes.
- Apply deterministic stratified sampling and reviewed duplicate-group constraints.
- Track source hashes, configuration snapshots, and isolated run artifacts.
- Validate redaction-offset alignment before applying full-text findings to excerpts.
- Separate execution success, formatting, factual fidelity, safety, and task usefulness.
- Document negative results: fluent conversations can repeat generic referrals or change source facts; an unvalidated LLM judge is not a substitute for human evidence.

## Repository map

```text
configs/                 Source metadata, environment requirements, judge rubrics
docs/portfolio/          Public methods, stage results, and reproducibility notes
notebooks/               Output-free source copies, grouped by pipeline stage
scripts/                 Preparation, generation, validation, and analysis tools
src/findisputeeval/      Reusable EDA and seed-curation modules
tests/                   Offline fixtures and tests; some need private run artifacts
DATA_POLICY.md           Publication boundaries
WORK_PROGRESS.md         Public status marker, not the internal session log
```

The workflow is **source data → curation → NeMo Data Designer → language model → review and analysis**. Data Designer is a framework, not an alternative to Nemotron. Qwen and DeepSeek experiments in this repository concern semantic judging, not a controlled comparison of generation models.

## Running the code

Use a dedicated Python environment and the requirements file appropriate to the notebook or tool. See [reproducibility](docs/portfolio/REPRODUCIBILITY.md). The repository intentionally excludes complaint narratives, curated seeds, annotations, API responses, model caches, and executed notebook output. Full historical runs therefore cannot be reproduced from a clone alone.

Notebook code may mount Google Drive, install dependencies, or invoke paid APIs. Read the configuration and source cells before execution. Never use **Run All** merely to preview the portfolio.

## Current limitations

- Consumer narratives are allegations, not verified case facts or agent transcripts.
- The latest 20-case sample is coverage-oriented, not population-representative or family-held-out.
- Human review remains incompletely finalized; prior AI-assisted discussion is not independent blind annotation.
- Historical judge metrics used changing development oracles and settings.
- Formal privacy clearance, a reliable automatic release judge, and the formal pilot remain incomplete.

## Publication and licensing

This repository publishes selected source code and methods documentation only. Dataset availability elsewhere does not authorize redistribution here. No new open-source license is assigned by this preparation; code licensing requires a separate owner decision. Third-party dependencies and datasets retain their own terms.
