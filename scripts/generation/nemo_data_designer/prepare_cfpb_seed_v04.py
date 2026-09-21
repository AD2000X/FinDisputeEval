import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SEED_POOL = (
    PROJECT_ROOT
    / "dataset"
    / "curated"
    / "seed_pools"
    / "cfpb_dispute"
    / "seed_v04"
    / "cfpb_seed_pool.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "generation"
    / "benchmark_v01"
    / "nemo_data_designer"
    / "seeded_dialogue"
    / "recipe_v01"
    / "prepared_inputs"
)


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def compact_seed(row: dict, max_chars: int) -> dict:
    ling = row.get("linguistic", {})
    narrative = row.get("narrative_norm", "")
    return {
        "seed_id": row["seed_id"],
        "complaint_id": row["complaint_id"],
        "card_type": row["card_type"],
        "claim_type": row["claim_type"],
        "route_hint": row["route_hint"],
        "zelle_mention": bool(row.get("zelle_mention", False)),
        "anchor_confidence": row.get("anchor_confidence", ""),
        "quality_score": float(row.get("quality_score", 0.0)),
        "source_query": row.get("source_query", ""),
        "date_received": row.get("date_received", ""),
        "company": row.get("company", ""),
        "n_tokens": int(ling.get("n_tokens", 0)),
        "n_sents": int(ling.get("n_sents", 0)),
        "mask_density": float(ling.get("mask_density", 0.0)),
        "vader_compound": float(ling.get("vader_compound", 0.0)),
        "seed_narrative_excerpt": narrative[:max_chars],
        "label_clean_policy": "relabel_required_false_v4",
    }


def stratified_sample(rows, n: int, seed: int):
    if n <= 0 or n >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    by_cell = defaultdict(list)
    for row in rows:
        by_cell[(row["card_type"], row["claim_type"])].append(row)

    cells = sorted(by_cell)
    base = n // len(cells)
    rem = n % len(cells)
    selected = []
    leftovers = []
    for idx, cell in enumerate(cells):
        bucket = sorted(by_cell[cell], key=lambda r: (-r["quality_score"], r["seed_id"]))
        take = min(len(bucket), base + (1 if idx < rem else 0))
        selected.extend(bucket[:take])
        leftovers.extend(bucket[take:])

    if len(selected) < n:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: n - len(selected)])
    rng.shuffle(selected)
    return selected[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-pool", type=Path, default=DEFAULT_SEED_POOL)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--sample-size", type=int, default=100)
    ap.add_argument("--max-excerpt-chars", type=int, default=1400)
    ap.add_argument("--random-seed", type=int, default=20260708)
    args = ap.parse_args()

    rows = [r for r in load_jsonl(args.seed_pool) if not bool(r.get("relabel_required", False))]
    compact = [compact_seed(r, args.max_excerpt_chars) for r in rows]
    sample = stratified_sample(compact, args.sample_size, args.random_seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    full_path = args.out_dir / "nemo_seed_relabel_false.jsonl"
    sample_path = args.out_dir / f"nemo_seed_relabel_false_stratified_{len(sample)}.jsonl"
    manifest_path = args.out_dir / "nemo_seed_manifest.json"

    write_jsonl(full_path, compact)
    write_jsonl(sample_path, sample)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_seed_pool": str(args.seed_pool),
        "policy": "Use only relabel_required=False formal v4 seeds for first NeMo Data Designer trial.",
        "full_rows": len(compact),
        "sample_rows": len(sample),
        "sample_size_requested": args.sample_size,
        "max_excerpt_chars": args.max_excerpt_chars,
        "full_cell_distribution": dict(sorted(Counter(f"{r['card_type']}|{r['claim_type']}" for r in compact).items())),
        "sample_cell_distribution": dict(sorted(Counter(f"{r['card_type']}|{r['claim_type']}" for r in sample).items())),
        "outputs": {
            "full_seed_jsonl": str(full_path),
            "sample_seed_jsonl": str(sample_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "full_rows": len(compact),
        "sample_rows": len(sample),
        "full_seed_jsonl": str(full_path),
        "sample_seed_jsonl": str(sample_path),
        "manifest": str(manifest_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
