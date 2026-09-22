# Data and publication policy

The public portfolio includes selected source code, cleaned notebook source, configurations, aggregate findings, and methods documentation.

See [CFPB source dates and dataset coverage](dataset/README.md) for the frozen
research inputs' requested and observed complaint-date ranges. This is metadata
documentation, not a distribution of the source records or private run manifests.

It excludes:

- Raw or curated complaint narratives and seed payloads.
- Generated conversations and summaries, raw API responses, and failed-response bodies.
- Human/LLM case-level judgments, reviewer identities, and annotation workbooks.
- Executed notebook output, attachments, widget state, and execution metadata.
- Private run manifests/reports, local environments, caches, logs, credentials, and editor/agent settings.
- External reference PDFs, screenshots, and dataset example files whose redistribution has not been reviewed.

Formal privacy clearance and benchmark release gates remain incomplete. Structural checks and masked text are not evidence of full de-identification. Public access to a source dataset does not remove licensing, privacy, or project-governance obligations.

The export applies an allowlist, notebook cleanup, and a bounded credential-pattern scan. These controls reduce accidental disclosure; they are not a complete security/privacy audit. Review the final diff before publishing. Use GitHub secret-scanning protections where available. If a credential is ever exposed, revoke/rotate it; simply deleting a file does not remove it from Git history.

Code licensing is a separate owner decision. No MIT, Apache, or other open-source license has been assigned automatically.
