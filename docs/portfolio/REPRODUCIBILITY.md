# Reproducibility and notebook guide

## What a public clone supports

- Reading and inspecting the source implementation.
- Running self-contained tests with synthetic fixtures after installing their dependencies.
- Reviewing output-free notebook source and configuration.

It does **not** contain private complaint data, seed bundles, annotation records, or run artifacts. Scripts that require these inputs should fail rather than recreate historical evidence silently. Some regression tests need those private files; a full test-suite pass is not claimed for a data-free clone.

## Environment

Use Python 3.11 in a dedicated environment. Select a requirements file from `configs/environments/` for the workflow you intend to inspect. Different tasks have different optional dependencies; there is intentionally no claim that one minimal install reproduces every historical environment.

A small data-free check uses the seed sampling tests:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install pandas numpy
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_cfpb_seed_common.py
```

This checks selected sampling/governance logic with synthetic inputs. It does not validate a release or call a model.

## Notebook entry points

Use the stage directories and version names to distinguish current and historical work:

1. `notebooks/10_source_eda/cfpb/canonical/FinDisputeEval_CFPB_SeedSource_EDA_colab_v051.ipynb`
2. `notebooks/20_seed_curation/cfpb_dispute/canonical/FinDisputeEval_CFPB_SeedPool_Build_colab_v052.ipynb`
3. `notebooks/30_generation/nemo_data_designer/seeded_dialogue/canonical/FinDisputeEval_CFPB_v052_nemotron_super_exploration_20_v02_colab.ipynb`
4. `notebooks/30_generation/nemo_data_designer/seeded_dialogue/canonical/FinDisputeEval_CFPB_v052_generation_linguistic_comparison_v01_colab.ipynb`

Older versions and judge attempts are retained as source history, not recommended production defaults. A `canonical` directory label does not imply privacy clearance or that every experiment succeeded.

All distributed notebook outputs, execution counters, attachments, and runtime metadata have been removed from **copies**. Embedded compressed Python modules remain part of source code; the export scanner also checks their decoded text for known credential patterns.

## Private inputs and credentials

Authorized users must obtain source data under its applicable terms and separately satisfy the project's data-governance gates. Filenames and hashes document provenance, but no public download authorization is implied.

Use runtime environment variables or interactive secret prompts. Do not commit API keys, authentication output, credentials, or Google Drive contents. Running generation can incur provider charges; inspection and the example unit test do not require an API key.

The `WORK_PROGRESS.md` in this public copy is a minimal project-root marker and status note. It is not the internal session log used during research.
