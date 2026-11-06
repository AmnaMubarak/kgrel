"""Read two annotated CSVs and compute Cohen's kappa + a confusion table.

Gate: kappa >= 0.6 on content types (T1, T2, T3, correct). T5 (format errors)
are pipeline-detected and excluded from the human-agreement calculation.

Usage:
    python thesis/implementation/scripts/12_kappa.py --run-id pilot
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve()
IMPL = HERE.parents[1]
sys.path.insert(0, str(IMPL / "src"))

from kgrel.metrics import cohens_kappa  # noqa: E402

CONTENT_LABELS = ["correct", "T1", "T2", "T3"]
EXCLUDED = {"T5", "", None}


def load_annotated(path: Path):
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    if not rows:
        raise SystemExit(f"[kappa] {path} is empty")
    return rows


def key(row):
    return (row["model"], row["sent_id"],
            row["subject"], row["relation"], row["object"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="pilot")
    ap.add_argument("--a", default=None, help="annotator A CSV (optional override)")
    ap.add_argument("--b", default=None, help="annotator B CSV (optional override)")
    args = ap.parse_args()

    ann_dir = IMPL / "data" / "annotation"
    a_path = Path(args.a) if args.a else ann_dir / f"{args.run_id}_annotator_A_amna.csv"
    b_path = Path(args.b) if args.b else ann_dir / f"{args.run_id}_annotator_B_labmate.csv"
    if not a_path.exists():
        print(f"[kappa] missing {a_path}"); return 2
    if not b_path.exists():
        print(f"[kappa] missing {b_path}"); return 2

    rows_a = {key(r): r for r in load_annotated(a_path)}
    rows_b = {key(r): r for r in load_annotated(b_path)}
    common = sorted(rows_a.keys() & rows_b.keys())
    if not common:
        print("[kappa] no overlapping (model, sent_id, triple) rows between the two files")
        return 2

    pairs_all = []
    pairs_content = []
    skipped = 0
    for k in common:
        la = (rows_a[k].get("human_label") or "").strip()
        lb = (rows_b[k].get("human_label") or "").strip()
        if la in EXCLUDED or lb in EXCLUDED:
            skipped += 1
            continue
        pairs_all.append((la, lb))
        if la in CONTENT_LABELS and lb in CONTENT_LABELS:
            pairs_content.append((la, lb))

    print(f"[kappa] rows A={len(rows_a)}  B={len(rows_b)}  "
          f"common={len(common)}  usable={len(pairs_all)}  skipped={skipped}")

    if not pairs_content:
        print("[kappa] no content-label pairs to score"); return 1

    a, b = zip(*pairs_content)
    k_val = cohens_kappa(list(a), list(b))
    print(f"\n[kappa] Cohen kappa on content labels ({', '.join(CONTENT_LABELS)}): "
          f"**{k_val:.3f}**  "
          f"{'PASS (>= 0.60)' if k_val >= 0.60 else 'FAIL (< 0.60) — fix the guidelines, not the data'}")

    # per-label breakdown (agreement rate per label as marginal sanity check)
    ca, cb = Counter(a), Counter(b)
    print("\nlabel   annotator A   annotator B   agree")
    agree_per = Counter()
    seen_per = Counter()
    for la, lb in pairs_content:
        seen_per[la] += 1
        if la == lb:
            agree_per[la] += 1
    for lab in CONTENT_LABELS:
        print(f"  {lab:<5}  {ca[lab]:>11}  {cb[lab]:>12}  "
              f"{(agree_per[lab] / seen_per[lab] if seen_per[lab] else 0):.2f}")

    # confusion matrix
    cats = CONTENT_LABELS
    cm = {r: {c: 0 for c in cats} for r in cats}
    for la, lb in pairs_content:
        cm[la][lb] += 1
    print("\nconfusion (rows = annotator A, cols = annotator B):")
    print("        " + "  ".join(f"{c:>8}" for c in cats))
    for r in cats:
        print(f"  {r:<5} " + "  ".join(f"{cm[r][c]:>8}" for c in cats))

    # write a small summary file next to the annotation CSVs
    out = ann_dir / f"{args.run_id}_kappa.md"
    out.write_text(
        f"# Cohen kappa — {args.run_id}\n\n"
        f"- common rows: **{len(common)}**  usable: **{len(pairs_all)}**\n"
        f"- content-label pairs (T1/T2/T3/correct): **{len(pairs_content)}**\n"
        f"- Cohen kappa: **{k_val:.3f}**  "
        f"({'PASS' if k_val >= 0.60 else 'FAIL'} against 0.60 target)\n",
        encoding="utf-8",
    )
    print(f"\n[kappa] wrote {out}")
    return 0 if k_val >= 0.60 else 1


if __name__ == "__main__":
    sys.exit(main())
