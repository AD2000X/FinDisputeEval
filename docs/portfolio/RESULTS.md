# Stage results and limitations

Snapshot: 2026-09-21. Aggregate findings are summarized from local research artifacts, which are excluded from the public repository. This is not a dataset release or an independently verified leaderboard.

## Latest generation exploration

NeMo Data Designer 0.7.0 orchestrated `nvidia/nemotron-3-super-120b-a12b` with temperature 1.0, top-p 0.95, non-thinking mode, and a maximum output budget of 4,096 tokens. Twenty dialogues were produced and retained. Nineteen met message-count/alternation checks. Four triggered deterministic review flags; those flags are not four confirmed quality failures. No case was automatically promoted to a final dataset.

The workflow uses sampled 4/6/8 messages, not 4/6/8 user-assistant pairs. Seed selection is deterministic; model output and all conditioning samples are not guaranteed bitwise reproducible.

## Human review snapshot

| Dimension | Pass | Fail | Uncertain |
|---|---:|---:|---:|
| Factual fidelity | 17 | 3 | 0 |
| Unsupported content | 16 | 3 | 1 |
| Label consistency | 20 | 0 | 0 |
| Multi-turn coherence | 19 | 1 | 0 |
| Safety | 20 | 0 | 0 |
| Dialogue quality | 20 | 0 | 0 |
| Task usefulness | 19 | 0 | 1 |

The seven dimensions contain 131 pass, 7 fail, and 2 uncertain entries. These are not interchangeable scale items or 140 independent cases. Do not convert them into an overall quality pass rate.

Final decisions remain 19 uncertain and 1 unusable. Reviewer identity/time and completion status have not been finalized. Three case-evidence notes followed prior AI-assisted discussion, so these ratings are not independent blind annotation or a multi-rater reliability study.

## Main finding

The dialogues are generally fluent, but some repeat generic referrals after the customer has already tried the same route. Source comparisons also reveal chronology, participant-role, and scenario changes. Fluency, factual fidelity, and useful information gain must be evaluated separately.

The source provides consumer narratives and categorical company-response fields, not genuine agent-side transcripts. This limits conversational-realism validation, but does not make all clarification questions unsupported. An assistant can summarize known facts and elicit missing information without claiming account access or inventing policy.

## Other completed analyses

- Five-view linguistic comparison: full seed, source excerpt, post-redaction grounding, generated user, generated assistant.
- 100 case-view observations from 20 cases, not 100 independent samples.
- 480 paired feature differences; role-separated phrase and question-response inspection.
- Historical blind semantic-judge calibration with Qwen and DeepSeek. These experiments used a separate development set and LLM-created reference judgments, not human gold. Different oracle hashes and configurations limit direct comparison. No judge was approved for automatic release.

## Boundaries and next step

The evidence does not establish a systemic impossibility of seeded generation, a controlled cross-model failure, or a completed benchmark. Formal privacy clearance remains incomplete. The planned next experiment is five fixed-fact clarification tasks with human-controlled customer disclosure and three primary criteria: factual fidelity, interaction progress, and useful handoff. It has not yet been executed.
