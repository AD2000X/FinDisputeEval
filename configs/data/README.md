# Data configuration

This directory contains machine-readable source metadata, snapshot locks, and deterministic split definitions.

The human-facing source catalog is `docs/data_catalog/FinDisputeEval_Dataset_Reference_v2.xlsx`. Operational status belongs in the machine registry and run manifests, not in folder names or notebook versions.

Dataset roles such as complaint language, intent classification, dialogue structure, emotion, summarization, and policy grounding are metadata. A source is stored once under its stable dataset ID even when it serves multiple roles.

