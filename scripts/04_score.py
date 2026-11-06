"""Score predictions against gold, compute per-type labels, risk signals, gate
decisions, and metrics. Writes a per-triple CSV plus a JSON summary.

Usage:
    python thesis/implementation/scripts/04_score.py --run-id core
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import yaml

HERE = Path(__file__).resolve()
IMPL = HERE.parents[1]
sys.path.insert(0, str(IMPL / "src"))

from kgrel.dataset import load_ontologies                      # noqa: E402
from kgrel.gate import GateInput, GateThresholds, decide        # noqa: E402
from kgrel.match import match_sentence                          # noqa: E402
from kgrel.metrics import (                                     # noqa: E402
    aggregate_prf1, bootstrap_prf1, gated_counts, per_type_rates,
)
from kgrel.schema import GoldItem, Triple                       # noqa: E402
from kgrel.signals import ontology_novelty, source_support      # noqa: E402


ERROR_TYPES = ["T1", "T2", "T3", "T4", "T5"]


def load_sample(path: Path) -> Dict[str, GoldItem]:
    out: Dict[str, GoldItem] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["sent_id"]] = GoldItem(
            sent_id=row["sent_id"],
            ontology_id=row["ontology_id"],
            sentence=row["sentence"],
            triples=[Triple(**t) for t in row.get("triples", [])],
            stratum=row.get("stratum"),
            openworld_held_out=bool(row.get("openworld_held_out", False)),
            held_out_relations=row.get("held_out_relations") or [],
        )
    return out


def load_run(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["triples"] = [Triple(**t) for t in row.get("triples", [])]
        yield row


def label_rule_based(triple: Triple, matched: bool, novelty: str, format_ok: bool,
                     support: float) -> str:
    """Fast rule-based labeler. Humans will relabel on the annotation tier.
    Returned label is one of T1..T5 (or 'correct').
    """
    if not format_ok:
        return "T5"
    if matched:
        return "correct"
    if support < 0.25:
        return "T1"            # unsupported hallucination
    if novelty in ("new_type", "junk"):
        return "T2"            # relational distortion / ontology violation
    return "T3"                # entity error (fallback for unmatched content)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="core")
    args = ap.parse_args()

    cfg = yaml.safe_load((IMPL / "configs" / "config.yaml").read_text(encoding="utf-8"))
    th = GateThresholds(
        support_min=cfg["gate"]["support_min"],
        stability_min=cfg["gate"]["stability_min"],
        propose_schema_min_support=cfg["gate"]["propose_schema_min_support"],
    )
    fuzzy = cfg["canonical"]["fuzzy_threshold"]

    sample = load_sample(IMPL / cfg["paths"]["sample"] / "sample.jsonl")
    ontologies = load_ontologies(str(IMPL / cfg["benchmark"]["ontologies_glob"]))
    runs_dir = IMPL / cfg["paths"]["runs"]
    results_dir = IMPL / cfg["paths"]["results"]
    results_dir.mkdir(parents=True, exist_ok=True)

    rows_out: List[dict] = []
    per_model_matches: Dict[str, list] = {}

    for model_dir in sorted(runs_dir.glob("*")):
        run_path = model_dir / f"{args.run_id}.jsonl"
        if not run_path.exists():
            continue
        model_name = model_dir.name
        matches = []
        for pred in load_run(run_path):
            gold = sample.get(pred["sent_id"])
            if gold is None:
                continue
            ont = ontologies.get(gold.ontology_id)
            m = match_sentence(pred["sent_id"], pred["triples"], gold.triples,
                               fuzzy_threshold=fuzzy)
            matches.append(m)

            matched_preds = {id(p) for p, _, _ in m.matched_pairs}
            for p in pred["triples"]:
                is_match = id(p) in matched_preds
                support = source_support(p, gold.sentence)
                novelty = ontology_novelty(p, ont) if ont is not None else "junk"
                label = label_rule_based(p, is_match, novelty,
                                         pred.get("parse_ok", True), support)
                gi = GateInput(
                    triple=p, support=support, instability=0.0,  # pilot: single sample
                    novelty=novelty, format_ok=pred.get("parse_ok", True),
                )
                action = decide(gi, th)
                rows_out.append({
                    "model": model_name,
                    "sent_id": pred["sent_id"],
                    "ontology": gold.ontology_id,
                    "openworld_held_out": gold.openworld_held_out,
                    "subject": p.subject, "relation": p.relation, "object": p.object,
                    "matched": is_match,
                    "support": round(support, 3),
                    "novelty": novelty,
                    "format_ok": pred.get("parse_ok", True),
                    "rule_label": label,
                    "gate_action": action,
                })
        per_model_matches[model_name] = matches

    # per-triple CSV
    csv_path = results_dir / f"{args.run_id}_per_triple.csv"
    if rows_out:
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
            writer.writeheader(); writer.writerows(rows_out)
        print(f"[score] wrote {csv_path}  ({len(rows_out)} triples)")

    # summary JSON
    summary = {}
    for model, matches in per_model_matches.items():
        strict = bootstrap_prf1(matches, mode="strict")
        canon  = bootstrap_prf1(matches, mode="canonical")
        labels_for_model = [r["rule_label"] for r in rows_out if r["model"] == model]
        decisions = [r["gate_action"] for r in rows_out if r["model"] == model]
        correct   = [r["matched"]     for r in rows_out if r["model"] == model]
        summary[model] = {
            "n_sentences": len(matches),
            "n_pred_triples": sum(1 for r in rows_out if r["model"] == model),
            "prf1_strict":    strict,
            "prf1_canonical": canon,
            "per_type_rule":  per_type_rates(labels_for_model,
                                              ERROR_TYPES + ["correct"]),
            "gate":           gated_counts(decisions, correct),
        }
    sum_path = results_dir / f"{args.run_id}_summary.json"
    sum_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"[score] wrote {sum_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
