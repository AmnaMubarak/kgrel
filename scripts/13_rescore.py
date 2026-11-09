"""Offline rescore: recompute downstream artifacts (per-triple CSV,
summary JSON, markdown report) from cached per-model prediction files in
data/runs/<model>/<run_id>.jsonl, using the CURRENT schema/signal/gate
settings. No API calls.

Use this when you change:
  - the ontology layer (synth -> real, or vice versa)
  - the gate thresholds in configs/config.yaml
  - a signal implementation
...and want to see the delta WITHOUT burning API quota on a fresh run.

Usage:
    python scripts/13_rescore.py --run-id pilot
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List

import yaml

HERE = Path(__file__).resolve()
IMPL = HERE.parents[1]
sys.path.insert(0, str(IMPL / "src"))

from kgrel.dataset import load_or_synthesize_ontologies
from kgrel.gate import GateInput, GateThresholds, decide
from kgrel.match import match_sentence
from kgrel.metrics import aggregate_prf1
from kgrel.schema import GoldItem, Prediction, Triple
from kgrel.signals import ontology_novelty, source_support


def load_sample(path: Path) -> List[GoldItem]:
    items: List[GoldItem] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            items.append(GoldItem(
                sent_id=str(row["sent_id"]),
                ontology_id=str(row["ontology_id"]),
                sentence=str(row["sentence"]),
                triples=[Triple.from_dict(t) for t in row.get("triples", [])],
                stratum=str(row.get("stratum", row["ontology_id"])),
                openworld_held_out=bool(row.get("openworld_held_out", False)),
                held_out_relations=row.get("held_out_relations") or [],
            ))
    return items


def load_predictions(path: Path) -> List[Prediction]:
    preds: List[Prediction] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            preds.append(Prediction(
                sent_id=str(row["sent_id"]),
                model=str(row.get("model") or row.get("model_name", "")),
                run_id=str(row.get("run_id", "")),
                raw_output=str(row.get("raw_output") or row.get("raw", "")),
                triples=[Triple.from_dict(t) for t in row.get("triples", [])],
                parse_ok=bool(row.get("parse_ok", False)),
                latency_s=float(row.get("latency_s", 0.0)),
            ))
    return preds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="pilot")
    ap.add_argument("--models", default=None,
                    help="comma list; default = all under data/runs/ that have <run_id>.jsonl")
    args = ap.parse_args()

    cfg = yaml.safe_load((IMPL / "configs" / "config.yaml").read_text(encoding="utf-8"))
    sample = load_sample(IMPL / cfg["paths"]["sample"] / "sample.jsonl")
    by_sent = {s.sent_id: s for s in sample}

    ontologies = load_or_synthesize_ontologies(
        str(IMPL / cfg["benchmark"]["ontologies_glob"]), fallback_items=sample)

    runs_dir = IMPL / cfg["paths"]["runs"]
    results_dir = IMPL / cfg["paths"]["results"]
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.models:
        model_names = [m.strip() for m in args.models.split(",")]
    else:
        model_names = sorted(
            p.parent.name for p in runs_dir.glob(f"*/{args.run_id}.jsonl"))
    if not model_names:
        print(f"[rescore] no cached runs under {runs_dir} for run_id={args.run_id}")
        return 2
    print(f"[rescore] rescoring models: {model_names}  (run_id={args.run_id})")

    th = GateThresholds(
        support_min=cfg["gate"]["support_min"],
        stability_min=cfg["gate"]["stability_min"],
        propose_schema_min_support=cfg["gate"]["propose_schema_min_support"],
    )
    fuzzy = cfg["canonical"]["fuzzy_threshold"]

    per_triple_rows = []
    per_model_summary = {}

    for model in model_names:
        pred_path = runs_dir / model / f"{args.run_id}.jsonl"
        preds = load_predictions(pred_path)
        print(f"[rescore] {model}: loaded {len(preds)} cached predictions")
        matches = []
        n_format_fail = 0
        for pred in preds:
            if not pred.parse_ok:
                n_format_fail += 1
            item = by_sent.get(pred.sent_id)
            if item is None:
                continue
            ont = ontologies.get(item.ontology_id)
            if ont is None:
                continue
            m = match_sentence(item.sent_id, pred.triples, item.triples,
                               fuzzy_threshold=fuzzy)
            matches.append(m)
            matched_preds = {id(p) for p, _, _ in m.matched_pairs}
            for p in pred.triples:
                support = source_support(p, item.sentence)
                novelty = ontology_novelty(p, ont)
                gi = GateInput(triple=p, support=support, instability=0.0,
                               novelty=novelty, format_ok=pred.parse_ok)
                action = decide(gi, th)
                per_triple_rows.append({
                    "model": model,
                    "sent_id": pred.sent_id,
                    "ontology": item.ontology_id,
                    "openworld_held_out": item.openworld_held_out,
                    "sentence": item.sentence,
                    "subject": p.subject, "relation": p.relation, "object": p.object,
                    "matched": id(p) in matched_preds,
                    "support": round(support, 3),
                    "novelty": novelty,
                    "format_ok": pred.parse_ok,
                    "gate_action": action,
                })

        n_triples = sum(len(p.triples) for p in preds)
        pp, rr, ff = aggregate_prf1(matches, mode="canonical")
        per_model_summary[model] = {
            "n_sentences": len(preds),
            "n_triples": n_triples,
            "parser_recovery": round(1 - n_format_fail / max(1, len(preds)), 3),
            "n_format_fail": n_format_fail,
            "prf1_canonical": {"P": round(pp, 3), "R": round(rr, 3), "F1": round(ff, 3)},
        }

    csv_path = results_dir / f"{args.run_id}_per_triple.csv"
    if per_triple_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(per_triple_rows[0].keys()))
            w.writeheader(); w.writerows(per_triple_rows)
        print(f"[rescore] wrote {csv_path}  ({len(per_triple_rows)} triples)")

    sum_path = results_dir / f"{args.run_id}_summary.json"
    sum_path.write_text(json.dumps(per_model_summary, indent=2), encoding="utf-8")
    print(f"[rescore] wrote {sum_path}")

    # Minimal rescored report (overwrites the original; signal change summary)
    lines = [f"# Pilot report (rescored with current config)\n",
             f"Models: {', '.join(sorted(per_model_summary.keys()))}\n\n"]
    for model, s in per_model_summary.items():
        lines.append(
            f"## `{model}`\n"
            f"- triples emitted: **{s['n_triples']}** "
            f"(avg {s['n_triples']/max(1, s['n_sentences']):.2f} per sentence)\n"
            f"- parser recovery: **{s['parser_recovery']:.1%}**  "
            f"format failures: {s['n_format_fail']}\n"
            f"- P / R / F1 (canonical): {s['prf1_canonical']['P']:.2f} / "
            f"{s['prf1_canonical']['R']:.2f} / {s['prf1_canonical']['F1']:.2f}\n"
        )
    rep_path = results_dir / f"{args.run_id}_report.md"
    rep_path.write_text("".join(lines), encoding="utf-8")
    print(f"[rescore] wrote {rep_path}")
    print(f"\n[rescore] done. Signal changes were applied without re-calling the API.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
