# Methods: data cleaning and seed curation

This document describes the executed CFPB v05.1 / Seed v05.2 method, not an optimality claim for its thresholds. It contains aggregate information only; private source data and run manifests are not distributed.

## Source design

Three frozen source extracts contributed 226,771 rows: `inscope_recent` (197,489), `zelle_fulltext` (18,013), and `prepaid_alltime` (11,269). Merging on Complaint ID produced 205,589 unique records. The 21,182-row difference is cross-query duplication, not a count of invalid complaints.

The analysis separates a 197,489-record population frame from 8,100 enrichment-only IDs absent from that frame. Source conflicts were recorded for 4,617 IDs; the retained source priority is recent in-scope, Zelle, then prepaid. This deterministic precedence does not establish which account of events is true.

## Cleaning is not rewriting

| Operation | Method | Rationale and limitation |
|---|---|---|
| Source validation | Schema, file hashes, ID and date checks | Detect wrong inputs; hashes do not establish factual truth |
| Byte-literal handling | Typed parsing and UTF-8 decoding; 534 successful unwraps, 78 typed failures retained | Correct serialization noise without silently guessing malformed text |
| Text views | Raw, canonical, matching, and lexical representations | Preserve source language while using different views for matching and measurement |
| Canonical text | NFC, whitespace/newline normalization, standardized existing redaction/date/amount markers | Reduce formatting noise; not comprehensive de-identification |
| Duplicate candidates | Exact hashes, template-family signatures, MinHash candidates followed by review | Avoid overrepresenting repeated wording; similar topics are not necessarily duplicate templates |
| Quality flags | Short/long text, capitalization, unusual formatting, bounded redaction proportions | Identify review and stress candidates, not automatic quality labels |
| Language | Candidate language-ID status; English high/low confidence admitted by seed rules | English scope, with classifier uncertainty explicitly retained |
| Privacy | Regex and NER candidate detection plus separate governance checks | Detection and masking do not constitute formal privacy clearance |

The EDA redaction warning threshold (>0.15) is different from the seed selection threshold (>0.70). Text below 100 characters is flagged but retained under the seed rules; text above 5,000 characters is directed to stress-only handling. These are project-specific operational choices.

## Selection order and accounting

After eligibility and reservation rules, 202,526 candidates entered sequential caps:

| Cap | Before | After |
|---|---:|---:|
| One per exact-text group | 202,526 | 171,532 |
| One per template family | 171,532 | 167,561 |
| One per reviewed fuzzy group | 167,561 | 167,187 |

The capped population subset contains 159,286 candidates. Core sampling is proportional to this eligible, capped pool, not a simple random sample of all original complaints.

| Subset | Rows | Selection purpose |
|---|---:|---|
| Population core | 2,000 | Product × Issue × month stratification; largest-remainder allocation |
| Coverage supplement | 358 | One per eligible stratum not covered by the core; not prevalence-eligible |
| Enrichment | 646 | 300 historical prepaid + 300 historical Zelle + 45 outside-population Zelle + 1 overlap |
| Stress, separate | 200 | Reason-stratified challenging cases |
| Excluded, separate | 9 | Explicit exclusion decisions, not all unselected records |

Within strata, stable hash-based selection uses seed `20260713`. Coverage does not shrink the core. Enrichment targeted 651 rows but accepted a documented five-row shortage after eligibility/caps in the outside-population Zelle frame. The combined 3,004-row collection is not a prevalence sample.

## Generation preprocessing

The release provides prefix excerpts up to 1,400 characters. A later preparer verifies that each excerpt is a prefix of full seed text and that reused NER offsets align with the corresponding finding text; mismatches fail closed.

A minimum of 12 post-redaction non-placeholder tokens removes 26 of 3,004 candidates. This threshold was based on the observed lower tail of that distribution; it does not prove sufficient semantic detail. Excluding eligible members of the old development sample leaves 2,959 candidates for the latest exploratory sample.

The latest 20 cases comprise 8 core, 4 coverage, and 8 enrichment records, with no exact seed-ID overlap with the old development set. Template-family separation was not established, so this is not a held-out benchmark.

## Libraries and reproducibility

The implementation uses pandas/NumPy and Parquet tooling, fastText language identification, spaCy/Presidio privacy or linguistic candidates, and MinHash-based duplicate candidate discovery. Library-backed features remain measurements or candidates, not human gold labels. See the versioned environment files and reusable modules in `src/findisputeeval/`.
