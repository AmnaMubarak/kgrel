"""Phase 2 pilot: 40 sentences x 2 small models (default: Llama-3.1-8B + Gemma-2-9B).

Deliberately small. Checks four things before we scale to all 300 sentences:

    1. parser_recovery       >= 0.95         (format failures are rare)
    2. taxonomy_coverage     "other" <= 10%  (T1-T5 cover what we see)
    3. gate_decisions reasonable            (we're not abstaining on everything)
    4. extraction latency    sane           (no silent 30s stalls)

Outputs:
    data/runs/<model>/pilot.jsonl          (predictions)
    data/results/pilot_per_triple.csv      (per-triple scoring)
    data/results/pilot_summary.json        (headline stats)
    data/results/pilot_report.md           (short human-readable gate report)

Usage:
    python thesis/implementation/scripts/10_pilot.py
    python thesis/implementation/scripts/10_pilot.py --n 40 --models llama-3.1-8b,gemma2-9b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

HERE = Path(__file__).resolve()
IMPL = HERE.parents[1]
sys.path.insert(0, str(IMPL / "src"))

# Load .env so GROQ_API_KEY is picked up automatically.
try:
    from dotenv import load_dotenv
    load_dotenv(IMPL / ".env")
except ImportError:
    pass

from kgrel.dataset import load_or_synthesize_ontologies        # noqa: E402
from kgrel.extract import ModelSpec, extract_one, make_client  # noqa: E402
from kgrel.gate import GateInput, GateThresholds, decide        # noqa: E402
from kgrel.match import match_sentence                          # noqa: E402
from kgrel.metrics import aggregate_prf1                        # noqa: E402
from kgrel.schema import GoldItem, Triple, dumps_dc             # noqa: E402
from kgrel.signals import ontology_novelty, source_support      # noqa: E402

# The pilot uses the two smallest configured models for a fast sanity run.
# (Full N=300 run on all three models is scripts/03_extract.py.)
DEFAULT_PILOT_MODELS = ["qwen3-27b", "gpt-oss-20b"]
DEFAULT_PILOT_N = 40


def load_sample(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        yield GoldItem(
            sent_id=row["sent_id"], ontology_id=row["ontology_id"],
            sentence=row["sentence"],
            triples=[Triple(**t) for t in row.get("triples", [])],
            stratum=row.get("stratum"),
            openworld_held_out=bool(row.get("openworld_held_out", False)),
            held_out_relations=row.get("held_out_relations") or [],
        )


def pilot_pick(sample, n_total):
    """Stratified pick of n_total sentences, half held-out, half not, across ontologies."""
    by_group = {("held_out", s.ontology_id): [] for s in sample}
    by_group.update({("in_schema", s.ontology_id): [] for s in sample})
    for s in sample:
        key = ("held_out" if s.openworld_held_out else "in_schema", s.ontology_id)
        by_group[key].append(s)

    picked = []
    i = 0
    keys = list(by_group.keys())
    while len(picked) < n_total and any(by_group[k] for k in keys):
        k = keys[i % len(keys)]
        if by_group[k]:
            picked.append(by_group[k].pop(0))
        i += 1
    return picked[:n_total]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=DEFAULT_PILOT_N)
    ap.add_argument("--models", default=",".join(DEFAULT_PILOT_MODELS))
    ap.add_argument("--run-id", default="pilot")
    args = ap.parse_args()

    cfg = yaml.safe_load((IMPL / "configs" / "config.yaml").read_text(encoding="utf-8"))
    wanted = {m.strip() for m in args.models.split(",")}
    model_cfgs = [m for m in cfg["models"] if m["name"] in wanted]
    if not model_cfgs:
        print(f"[pilot] no models matched {args.models}")
        return 2

    sample_path = IMPL / cfg["paths"]["sample"] / "sample.jsonl"
    full = list(load_sample(sample_path))
    pilot = pilot_pick(full, args.n)
    print(f"[pilot] sample={len(pilot)}  "
          f"(open-world={sum(1 for s in pilot if s.openworld_held_out)})  "
          f"models={[m['name'] for m in model_cfgs]}")

    ontologies = load_or_synthesize_ontologies(
        str(IMPL / cfg["benchmark"]["ontologies_glob"]),
        fallback_items=full,
    )
    if not ontologies:
        print("[pilot] FATAL: no ontologies available (disk empty AND frozen "
              "sample has no gold relations). Nothing to run.")
        return 3
    missing_oids = sorted({s.ontology_id for s in pilot
                           if s.ontology_id not in ontologies})
    if missing_oids:
        print(f"[pilot] WARNING: {len(missing_oids)} pilot sentences reference "
              f"ontology ids not in the schema: {missing_oids[:5]}...")

    runs_dir = IMPL / cfg["paths"]["runs"]
    results_dir = IMPL / cfg["paths"]["results"]
    runs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    th = GateThresholds(
        support_min=cfg["gate"]["support_min"],
        stability_min=cfg["gate"]["stability_min"],
        propose_schema_min_support=cfg["gate"]["propose_schema_min_support"],
    )
    fuzzy = cfg["canonical"]["fuzzy_threshold"]

    import csv
    per_triple_rows = []
    per_model_summary = {}
    per_model_runs = {}

    for mc in model_cfgs:
        spec = ModelSpec(
            name=mc["name"], backend=mc["backend"],
            api_model_id=mc.get("api_model_id"),
            temperature=mc.get("temperature", 0.0),
            top_p=mc.get("top_p", 1.0),
            max_tokens=mc.get("max_tokens", 512),
        )
        client = make_client(spec)
        out_dir = runs_dir / spec.name; out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.run_id}.jsonl"
        preds = []
        matches = []
        n_format_fail = 0
        n_skipped = 0
        total_latency = 0.0

        with out_path.open("w", encoding="utf-8") as fh:
            for item in tqdm(pilot, desc=spec.name):
                ont = ontologies.get(item.ontology_id)
                if ont is None:
                    n_skipped += 1
                    continue
                pred = extract_one(client, item, ont, run_id=args.run_id)
                preds.append(pred)
                total_latency += pred.latency_s
                if not pred.parse_ok:
                    n_format_fail += 1
                fh.write(json.dumps(dumps_dc(pred), ensure_ascii=False) + "\n")

                # inline scoring for the per-triple CSV + gate
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
                    is_match = id(p) in matched_preds
                    per_triple_rows.append({
                        "model": spec.name,
                        "sent_id": pred.sent_id,
                        "ontology": item.ontology_id,
                        "openworld_held_out": item.openworld_held_out,
                        "sentence": item.sentence,
                        "subject": p.subject, "relation": p.relation, "object": p.object,
                        "matched": is_match,
                        "support": round(support, 3),
                        "novelty": novelty,
                        "format_ok": pred.parse_ok,
                        "gate_action": action,
                    })

        per_model_runs[spec.name] = preds
        n_triples = sum(len(p.triples) for p in preds)
        p, r, f = aggregate_prf1(matches, mode="canonical")
        per_model_summary[spec.name] = {
            "n_sentences": len(preds),
            "n_triples": n_triples,
            "n_skipped_missing_ontology": n_skipped,
            "parser_recovery": round(1 - n_format_fail / max(1, len(preds)), 3),
            "n_format_fail": n_format_fail,
            "avg_latency_s": round(total_latency / max(1, len(preds)), 2),
            "prf1_canonical": {"P": round(p, 3), "R": round(r, 3), "F1": round(f, 3)},
        }
        if n_skipped:
            print(f"[{spec.name}] WARNING: {n_skipped}/{len(pilot)} sentences "
                  "were skipped because their ontology_id was missing.")
        if len(preds) == 0:
            print(f"[{spec.name}] FATAL: 0 sentences processed. The API was "
                  "never called. Check ontology loading and GROQ_API_KEY.")

    # per-triple CSV (one row per emitted triple, across all pilot models)
    csv_path = results_dir / f"{args.run_id}_per_triple.csv"
    if per_triple_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(per_triple_rows[0].keys()))
            w.writeheader(); w.writerows(per_triple_rows)
        print(f"[pilot] wrote {csv_path}  ({len(per_triple_rows)} triples)")

    # summary.json
    sum_path = results_dir / f"{args.run_id}_summary.json"
    sum_path.write_text(json.dumps(per_model_summary, indent=2), encoding="utf-8")
    print(f"[pilot] wrote {sum_path}")

    # markdown report
    report_lines = ["# Pilot report\n",
                    f"**n sentences = {len(pilot)}**  "
                    f"(open-world = {sum(1 for s in pilot if s.openworld_held_out)})\n"]
    pass_recovery = True
    for model, s in per_model_summary.items():
        recovery = s["parser_recovery"]
        if recovery < 0.95:
            pass_recovery = False
        report_lines.append(
            f"## `{model}`\n"
            f"- triples emitted: **{s['n_triples']}**  "
            f"(avg {s['n_triples'] / max(1, s['n_sentences']):.2f} per sentence)\n"
            f"- parser recovery: **{recovery:.1%}** "
            f"{'OK' if recovery >= 0.95 else 'BELOW 95% GATE'}\n"
            f"- format failures: {s['n_format_fail']}\n"
            f"- P / R / F1 (canonical): "
            f"{s['prf1_canonical']['P']:.2f} / "
            f"{s['prf1_canonical']['R']:.2f} / "
            f"{s['prf1_canonical']['F1']:.2f}\n"
            f"- avg latency: {s['avg_latency_s']}s\n"
        )
    report_lines.append("## Phase 2 gate\n")
    report_lines.append(
        f"- parser recovery >= 95% on every model: **"
        f"{'PASS' if pass_recovery else 'FAIL — fix the parser or prompt before scaling'}**\n"
        "- taxonomy coverage: inspect `pilot_per_triple.csv`, open 30 random unmatched rows per model, "
        "count how many don't fit any of T1-T5. Fail if > 10%.\n"
        "- Cohen kappa (>= 0.6): run `scripts/12_kappa.py` after two annotators label the overlap CSV.\n"
    )
    rep_path = results_dir / f"{args.run_id}_report.md"
    rep_path.write_text("".join(report_lines), encoding="utf-8")
    print(f"[pilot] wrote {rep_path}")
    print("\n[pilot] done — next: scripts/11_overlap_csv.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
