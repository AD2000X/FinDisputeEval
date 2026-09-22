# Dataset sources and coverage

## CFPB source coverage

The CFPB v05.1 analysis underlying Seed v05.2 uses three frozen source extracts
with different purposes. Dates below refer to complaint dates (`Date received`),
not download dates.

https://www.consumerfinance.gov/data-research/consumer-complaints/search/?dateRange=All&date_received_max=2026-09-22&date_received_min=2011-12-01&page=1&searchField=all&size=25&sort=created_date_desc

| Source | Requested date range | Observed date range | Extract rows before cross-query deduplication |
|---|---|---|---:|
| Main population (`inscope_recent`) | 2025-01-01 to 2026-07-02 | 2025-01-01 to 2026-06-12 | 197,489 |
| Zelle enrichment (`zelle_fulltext`) | 2017-06-01 to 2026-07-02 | 2017-12-07 to 2026-06-10 | 18,013 |
| Prepaid enrichment (`prepaid_alltime`) | 2011-12-01 to 2026-07-02 | 2015-03-23 to 2026-06-10 | 11,269 |

The three extracts contain 226,771 rows before merging overlapping Complaint IDs.
After merging, the analysis contains:

- 197,489 population records.
- 8,100 enrichment-only records absent from the population frame.
- 205,589 unique records in total.

The full collection should not be described as containing only complaints from
2025 onward. Historical enrichment records are kept separate from population-level
prevalence estimates. The 3,004 selected Seed v05.2 records are a downstream subset,
not the complete source collection.

## Snapshot and reproducibility

The source snapshot directory is tagged `seed_build_v03_cache_2026-07-04`.
This directory date is not proof that every extract was downloaded on that day.
The requested end date, 2026-07-02, is also not the latest observed complaint date.

The date ranges and row counts above are recorded in the private EDA v05.1
manifest at
`outputs/data_pipeline/cfpb_seed_source_eda/eda_v051/run_20260713T145423Z/manifest.json`.
That manifest and the underlying payloads are not distributed in the public
portfolio.

These dates describe the frozen research inputs, not the current coverage of the
live CFPB database. A date-only search URL does not reproduce the product/search
filters or the frozen snapshot. See [methods and selection rationale](../docs/portfolio/METHODS.md)
for the population/enrichment design and downstream cleaning and selection.

## Local dataset zones

The following directories describe the private working layout. The public
portfolio includes this documentation, not the data payloads.

- `external/`: immutable third-party payloads and dated source snapshots.
- `interim/`: extracted or normalized data that can be rebuilt from external sources.
- `curated/`: project-owned seed pools, human annotations, and promoted inputs.
- `benchmarks/`: frozen evaluation cases and split manifests. Training data must not be written here.
- `knowledge/`: regulations, policy documents, and other retrieval sources.
- `samples/`: small fixtures for tests and examples, subject to publication review.
- `cache/`: disposable model and dataset caches.

Large payloads and CFPB narratives are excluded from Git. Every promoted dataset
must have a manifest containing source IDs, hashes, row counts, schema version,
and creation time. Formal privacy clearance and benchmark release remain incomplete.
