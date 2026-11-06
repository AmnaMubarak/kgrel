"""Build the annotation overlap CSVs.

From the pilot per-triple CSV, picks a deterministic sample of ~30 triples per
model (stratified across: matched vs unmatched, in-schema vs open-world, and
the four candidate error types) and writes two CSVs:

    data/annotation/<run_id>_annotator_A.csv
    data/annotation/<run_id>_annotator_B.csv

Both files contain the SAME triples in the SAME order, with an empty
`human_label` column for each annotator to fill in independently. The two
files are later fed to `12_kappa.py` for Cohen's kappa.

Usage:
    python thesis/implementation/scripts/11_overlap_csv.py --run-id pilot --n-per-model 30
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve()
IMPL = HERE.parents[1]

HUMAN_COLS = ["human_label", "human_comment", "new_type_candidate", "synonym_of"]
ALLOWED_LABELS = "correct | T1 | T2 | T3 | T5"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="pilot")
    ap.add_argument("--n-per-model", type=int, default=30)
    args = ap.parse_args()

    cfg = yaml.safe_load((IMPL / "configs" / "config.yaml").read_text(encoding="utf-8"))
    seed = cfg["seed"]
    results_dir = IMPL / cfg["paths"]["results"]
    src_csv = results_dir / f"{args.run_id}_per_triple.csv"
    if not src_csv.exists():
        print(f"[overlap] no per-triple CSV at {src_csv}; run 10_pilot.py first")
        return 2

    rows = list(csv.DictReader(src_csv.open("r", encoding="utf-8")))
    if not rows:
        print("[overlap] per-triple CSV is empty")
        return 2

    rng = random.Random(seed + 3)

    # stratify: (model, matched?, openworld?) buckets
    buckets = {}
    for r in rows:
        key = (r["model"], r["matched"], r["openworld_held_out"])
        buckets.setdefault(key, []).append(r)
    for b in buckets.values():
        rng.shuffle(b)

    # per model: pull round-robin across its buckets up to n-per-model
    models = sorted({r["model"] for r in rows})
    chosen = []
    for model in models:
        keys = [k for k in buckets if k[0] == model]
        i = 0; picked = 0
        while picked < args.n_per_model and any(buckets[k] for k in keys):
            k = keys[i % len(keys)]
            if buckets[k]:
                chosen.append(buckets[k].pop(0))
                picked += 1
            i += 1

    # write two identical CSVs, one per annotator
    out_dir = IMPL / "data" / "annotation"
    out_dir.mkdir(parents=True, exist_ok=True)

    annotator_names = ["A_amna", "B_labmate"]
    base_cols = ["model", "sent_id", "ontology", "openworld_held_out",
                 "sentence", "subject", "relation", "object",
                 "matched", "support", "novelty", "format_ok", "gate_action"]
    out_cols = base_cols + HUMAN_COLS

    for name in annotator_names:
        path = out_dir / f"{args.run_id}_annotator_{name}.csv"
        with path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=out_cols)
            w.writeheader()
            for r in chosen:
                row = {c: r.get(c, "") for c in base_cols}
                row.update({c: "" for c in HUMAN_COLS})
                w.writerow(row)
        print(f"[overlap] wrote {path}  ({len(chosen)} rows)")

    # README for the two annotators
    readme = out_dir / f"{args.run_id}_HOW_TO_ANNOTATE.md"
    readme.write_text(
        "# How to annotate the pilot overlap set\n\n"
        f"You have two files in this folder, one per annotator. **Do NOT read each "
        f"other's file until you are both done** — that would defeat the kappa check.\n\n"
        "Fill in only these columns:\n\n"
        f"| Column | Allowed values |\n"
        f"|---|---|\n"
        f"| `human_label` | `{ALLOWED_LABELS}` |\n"
        f"| `human_comment` | free text (short), especially when you're unsure |\n"
        f"| `new_type_candidate` | `1` if the model's relation looks like a reasonable new ontology relation, else leave blank |\n"
        f"| `synonym_of` | if the model's relation is a synonym of an ontology relation, write the ontology relation name |\n\n"
        "Read `thesis/implementation/annotation/guidelines.md` BEFORE you start.\n\n"
        "When both files are filled, run:\n\n"
        f"```powershell\npython thesis/implementation/scripts/12_kappa.py --run-id {args.run_id}\n```\n",
        encoding="utf-8",
    )
    print(f"[overlap] wrote {readme}")
    print("\n[overlap] done — hand both CSVs to the annotators")
    return 0


if __name__ == "__main__":
    sys.exit(main())
