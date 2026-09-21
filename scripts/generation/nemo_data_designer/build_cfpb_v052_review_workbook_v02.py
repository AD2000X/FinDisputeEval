"""Build the rubric-carrying v02 review workbook from review_reference.json.

The v01 workbook already had dropdowns, colour coding, and a completeness
validator. v02 adds the scoring rubric and worked anchors as sheets, and hides
the deterministic-flag column so the reviewer scores blind to the automated
rules before seeing them.

Source rows are never modified. Reviewer entries from an existing workbook can
be carried over with --merge-from.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

REFERENCE_COLUMNS = ["seed_id", "product", "sub_product", "issue", "sub_issue",
                     "grounding_evidence", "generated_dialogue", "generated_summary",
                     "generated_privacy_notes", "deterministic_flags_json"]
DIMENSIONS = ["factual_fidelity", "unsupported_content", "dialogue_quality", "task_usefulness"]
EDIT_COLUMNS = DIMENSIONS + ["review_decision", "problem_evidence", "revision_suggestion",
                             "reviewer_kind", "reviewer_id", "reviewed_utc", "review_status"]
COLUMNS = REFERENCE_COLUMNS + EDIT_COLUMNS
CHOICES = {**{key: ["pass", "fail", "uncertain"] for key in DIMENSIONS},
           "review_decision": ["usable", "needs_revision", "unusable", "uncertain"],
           "reviewer_kind": ["human", "llm"], "review_status": ["pending", "reviewed"]}
BLIND_COLUMN = "deterministic_flags_json"

# Preserved verbatim from the v01 workbook generator. Operational mechanics
# only; the scoring rubric lives on the Rubric sheet.
GUIDE_LINES_V01 = [
    '探索性 smoke；不是 oracle、人工 gold、privacy clearance 或 benchmark。',
    '灰色欄位是來源證據，不可修改；黃色欄位由 reviewer 填寫。',
    '先讀 grounding，再看 dialogue。分類是 metadata，不是絕對真相。',
    'factual_fidelity: 是否保持授權狀態、爭議類型、事件與付款狀態？',
    'unsupported_content: pass = 沒有發現重大虛構政策、權利、承諾或細節。',
    'dialogue_quality: 是否自然、有實質互動，而非重複客服模板？',
    'task_usefulness: 是否適合作為金融爭議 intake 的探索性合成對話？',
    'usable=原樣可用於探索；needs_revision=可修；unusable=需重做；uncertain=需討論。',
    'usable 只表示本次品質閱讀判斷；不會解除 smoke-only 或 privacy gate。',
    '所有四面向都填 pass/fail/uncertain；非 usable 請填問題片段/說明。',
    'reviewer_kind 必須標 human 或 llm；AI 初評不能冒充 human。',
    '填 reviewer_id 與 UTC ISO 時間，例如 2026-09-19T12:00:00Z，再設 reviewed。',
    'deterministic_flags 僅是既有規則的提醒；空白問題集合不代表語意合格。',
    '長文字可能超出 Excel 列高；請用 review_readable.html 看完整內容。',
    '儲存並同步 Excel 後，重跑最後的摘要 cell；它不會呼叫模型或改寫 Excel。',
]

RUBRIC = [
    ("HOW TO SCORE", ""),
    ("Order", "Read grounding_evidence first, before any generated text. Then score in this order: "
              "factual_fidelity, unsupported_content, dialogue_quality, task_usefulness."),
    ("Do not revise", "Do not change an earlier dimension after reading a later one."),
    ("Why order matters", "Fluent text reads as faithful text. Reading the dialogue before the grounding "
                          "creates a halo effect."),
    ("Hidden column", "deterministic_flags_json is hidden on purpose. Score all four dimensions first, "
                      "then unhide it and record any disagreement in problem_evidence. Three of the four "
                      "flagged rows carry a rule already shown to fire on benign phrasing."),
    ("", ""),
    ("CORE DISTINCTION", "The two evidence dimensions separate by direction. Do not fold one into the other."),
    ("factual_fidelity", "CONTRADICTION. The grounding establishes a fact and the dialogue states the "
                         "opposite, or changes it materially."),
    ("unsupported_content", "ADDITION. The grounding is silent and the dialogue asserts something anyway."),
    ("Two questions", "1. Did the dialogue get a grounded fact wrong? That is factual_fidelity. "
                      "2. Does the dialogue assert something with no basis? That is unsupported_content. "
                      "A row can fail both."),
    ("", ""),
    ("factual_fidelity: FAIL", "The dialogue contradicts something the grounding establishes."),
    ("  Authorization state", "The consumer authorized or initiated the payment but the dialogue calls it "
                              "unauthorized or unapproved, or the reverse."),
    ("  Dispute type", "Not-as-described, non-delivery, unauthorized transaction, fee dispute, and billing "
                       "error are different claims. Swapping them is a fail."),
    ("  Status", "paid or unpaid, resolved or unresolved, pending or completed, open or closed, accurate or "
                 "inaccurate reporting."),
    ("  Actor or chronology", "Who did what to whom, or in what order, changed materially."),
    ("factual_fidelity: NOT a fail", "Adding a detail the grounding is silent about, which is "
                                     "unsupported_content instead. Paraphrase, generalization, or omission. "
                                     "A label that disagrees with the grounding, because labels are metadata."),
    ("", ""),
    ("unsupported_content: FAIL", "pass means no material fabrication was found."),
    ("  Legal or rights claims", "You have the right to, they must, or any statute or regulatory conclusion."),
    ("  Product or policy", "Fee rules, eligibility, category restrictions, network or issuer rules, "
                            "internal processing cycles."),
    ("  Procedural claims", "A named department, a specific investigation or escalation path, this is the "
                            "usual process, a promised status report, a goodwill adjustment, a set timeframe."),
    ("  Promises", "Outcome or deadline promises made by the assistant about what will happen."),
    ("  Invented specifics", "An actor, amount, date, document, or channel absent from the grounding."),
    ("unsupported_content: NOT a fail", "Generic advice to contact the institution through its official "
                                        "channel. Suggesting the user ask a third party when something will "
                                        "happen. Asking clarifying questions or advising record retention."),
    ("", ""),
    ("dialogue_quality: FAIL", "Template repetition, near-identical openers or closers, or turns whose only "
                               "content is an instruction to contact customer service. No real exchange "
                               "across successive turns. Roles not alternating or malformed turns. No "
                               "actionable next step at the end."),
    ("dialogue_quality: NOT a fail", "Tone, grammar, or brevity alone."),
    ("", ""),
    ("task_usefulness: FAIL", "Grounding too thin to support any scenario. Not recognisably a financial "
                              "dispute intake. So generic it would teach or measure nothing. Would mislead "
                              "about dispute handling, for example by coaching a dispute basis that does not "
                              "match the facts."),
    ("", ""),
    ("INSUFFICIENT GROUNDING", "When the grounding does not establish a discernible actor, action, and "
                               "problem after redaction, stop and apply this rule instead of scoring normally."),
    ("  factual_fidelity", "uncertain. Nothing is established, so fidelity is unverifiable. Do not mark pass, "
                           "which would wrongly imply it was checked."),
    ("  unsupported_content", "fail, if the dialogue invents a full scenario."),
    ("  task_usefulness", "fail."),
    ("  review_decision", "unusable."),
    ("  problem_evidence", "Quote the grounding and write insufficient grounding."),
    ("", ""),
    ("DECISION MAPPING", "The workbook enforces that usable requires all four dimensions to pass."),
    ("  usable", "All four dimensions pass."),
    ("  needs_revision", "Exactly one dimension fails and the fix is local, such as deleting one unsupported "
                         "assertion, leaving the scenario intact."),
    ("  unusable", "factual_fidelity fails, or two or more dimensions fail, or the insufficient-grounding "
                   "rule applies. A fidelity failure is not editable because the premise is wrong."),
    ("  uncertain", "Needs discussion. Explain why."),
    ("  Principle", "The split rests on whether the scenario premise survives. Unsupported assertions are "
                    "usually removable. Contradictions are not."),
    ("", ""),
    ("USING UNCERTAIN", "uncertain requires a written reason in problem_evidence. Use it when the grounding "
                        "is genuinely ambiguous, not when the call is merely hard. If more than about three "
                        "of twenty rows end up uncertain, revise the rubric rather than the data."),
    ("COMPLETENESS", "A row counts as reviewed only when all four dimensions, review_decision, reviewer_kind, "
                     "reviewer_id, and reviewed_utc are filled and review_status is reviewed. reviewer_kind "
                     "must be human or llm; an AI pre-pass must never be recorded as human."),
    ("SCOPE", "usable is a quality reading only. It does not lift smoke-only, privacy, or benchmark gates."),
]

ANCHORS = [
    ("Case", "What happened", "Scoring"),
    ("Authorized scam payment",
     "Consumer knowingly sent a payment to a scammer; assistant advised saying the transaction was not authorized",
     "factual_fidelity fail, authorization state"),
    ("Purchase not as described",
     "Seller did not deliver as promised; the user turn opens with a charge the consumer did not approve",
     "factual_fidelity fail, dispute type and authorization"),
    ("Resolved negative balance",
     "Grounding states the balance was paid and resolved; dialogue recasts it as inaccurate reporting and calls "
     "withdrawal the usual process",
     "factual_fidelity fail on status, and unsupported_content fail on procedure"),
    ("Rights assertion",
     "Assistant states the consumer has the right to have the issuer correct an entry",
     "unsupported_content fail, legal or rights claim"),
    ("Fee justification",
     "Assistant invents category restrictions and non-domestic merchant rules to explain a foreign transaction fee",
     "unsupported_content fail, product policy"),
    ("Sparse grounding",
     "82 characters of grounding; a full debt-settlement scenario is generated",
     "Insufficient grounding rule"),
    ("Benign timing suggestion",
     "Assistant suggests the user ask the support team when the deposit should be completed",
     "All dimensions pass; this is not a promise"),
]


def merged_entries(path: Path) -> dict[str, dict[str, str]]:
    """Carry reviewer entries forward from an existing workbook, keyed by seed_id."""

    workbook = load_workbook(path)
    sheet = workbook["Review"]
    header = [cell.value for cell in sheet[1]]
    index = {name: position for position, name in enumerate(header)}
    if "seed_id" not in index:
        raise ValueError(f"No seed_id column in {path}")
    entries: dict[str, dict[str, str]] = {}
    for row in range(2, sheet.max_row + 1):
        seed_id = sheet.cell(row, index["seed_id"] + 1).value
        if not seed_id:
            continue
        values = {}
        for name in EDIT_COLUMNS:
            if name in index:
                value = sheet.cell(row, index[name] + 1).value
                values[name] = "" if value is None else str(value)
        entries[str(seed_id)] = values
    return entries


def style_sheet(sheet, records: list[dict[str, str]]) -> None:
    for position, name in enumerate(COLUMNS, start=1):
        letter = get_column_letter(position)
        sheet.column_dimensions[letter].width = (
            70 if name in {"grounding_evidence", "generated_dialogue"} else
            42 if name in {"generated_summary", "problem_evidence", "revision_suggestion"} else 24)
        for row in sheet.iter_rows(min_col=position, max_col=position):
            cell = row[0]
            if isinstance(cell.value, str):
                if len(cell.value) > 32767:
                    raise ValueError("Excel cell limit exceeded; refusing silent truncation")
                cell.data_type = "s"  # Untrusted text is never an Excel formula.
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            cell.fill = PatternFill("solid", fgColor="FFF2CC" if name in EDIT_COLUMNS else "E7E6E6")
        sheet.cell(1, position).font = Font(bold=True, color="FFFFFF")
        sheet.cell(1, position).fill = PatternFill("solid", fgColor="1F4E78")
        if name in CHOICES:
            validation = DataValidation(
                type="list", formula1='"' + ",".join(CHOICES[name]) + '"', allow_blank=True)
            validation.showErrorMessage = True
            validation.errorStyle = "stop"
            sheet.add_data_validation(validation)
            validation.add(f"{letter}2:{letter}{len(records) + 1}")
        if name == BLIND_COLUMN:
            sheet.column_dimensions[letter].hidden = True
    sheet.freeze_panes = "F2"
    sheet.auto_filter.ref = sheet.dimensions
    for row in range(2, len(records) + 2):
        sheet.row_dimensions[row].height = 240


def add_text_sheet(workbook, title: str, rows, widths: list[int]) -> None:
    sheet = workbook.create_sheet(title)
    for row in rows:
        sheet.append(list(row))
    for position, width in enumerate(widths, start=1):
        sheet.column_dimensions[get_column_letter(position)].width = width
    for row in sheet.iter_rows():
        for cell in row:
            if isinstance(cell.value, str):
                cell.data_type = "s"
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        if row[0].value and not str(row[0].value).startswith(" ") and str(row[0].value).isupper():
            row[0].font = Font(bold=True)
    sheet.cell(1, 1).font = Font(bold=True)


def build(reference: Path, output: Path, merge_from: Path | None) -> None:
    payload = json.loads(reference.read_text(encoding="utf-8"))
    records = payload["rows"]
    lineage = payload["lineage"]
    carried = merged_entries(merge_from) if merge_from else {}

    for record in records:
        existing = carried.get(str(record["seed_id"]), {})
        for name in EDIT_COLUMNS:
            if existing.get(name, "").strip():
                record[name] = existing[name]

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Review"
    sheet.append(COLUMNS)
    for record in records:
        sheet.append([record.get(name, "") for name in COLUMNS])
    style_sheet(sheet, records)

    add_text_sheet(workbook, "Rubric", RUBRIC, [30, 112])
    add_text_sheet(workbook, "Anchors", ANCHORS, [28, 72, 52])

    guide = workbook.create_sheet("Guide")
    for line in GUIDE_LINES_V01:
        guide.append([line])
    guide.column_dimensions["A"].width = 115
    for row in guide:
        row[0].alignment = Alignment(wrap_text=True, vertical="top")

    meta = workbook.create_sheet("_lineage")
    meta.append([json.dumps(lineage, sort_keys=True)])
    meta.sheet_state = "hidden"

    output.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output)
    print(json.dumps({
        "output": str(output),
        "rows": len(records),
        "carried_reviewer_entries": sum(1 for record in records if str(record["seed_id"]) in carried),
        "blind_column_hidden": BLIND_COLUMN,
        "sheets": workbook.sheetnames,
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--merge-from", type=Path, default=None,
                        help="Existing workbook whose reviewer entries should be carried forward.")
    args = parser.parse_args()
    build(args.reference, args.output, args.merge_from)


if __name__ == "__main__":
    main()
