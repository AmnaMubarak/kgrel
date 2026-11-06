"""End-to-end smoke test with synthetic fixtures. No network, no API keys.

Builds a 2-ontology, 5-sentence fake benchmark; runs the sampler, two fake
model clients, the matcher, risk signals, the gate, and the metrics; and
prints a tiny summary. Used to prove the pipeline is wired correctly before
scaling.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

HERE = Path(__file__).resolve()
IMPL = HERE.parents[1]
sys.path.insert(0, str(IMPL / "src"))

from kgrel.canonical import norm                              # noqa: E402
from kgrel.extract import ModelSpec                            # noqa: E402
from kgrel.gate import GateInput, GateThresholds, decide       # noqa: E402
from kgrel.match import match_sentence                         # noqa: E402
from kgrel.metrics import aggregate_prf1, bootstrap_prf1, gated_counts  # noqa: E402
from kgrel.sample import assign_openworld_holdout, stratified_sample    # noqa: E402
from kgrel.schema import GoldItem, Ontology, Triple            # noqa: E402
from kgrel.signals import ontology_novelty, paraphrase_instability, source_support  # noqa: E402


def fixtures() -> List[GoldItem]:
    o1 = "ont_book"
    o2 = "ont_city"
    return [
        GoldItem(sent_id="s1", ontology_id=o1, stratum=o1,
                 sentence="The novel Dune was written by Frank Herbert in 1965.",
                 triples=[Triple("Dune", "author", "Frank Herbert"),
                          Triple("Dune", "publication year", "1965")]),
        GoldItem(sent_id="s2", ontology_id=o1, stratum=o1,
                 sentence="Herbert also wrote Dune Messiah.",
                 triples=[Triple("Dune Messiah", "author", "Frank Herbert")]),
        GoldItem(sent_id="s3", ontology_id=o2, stratum=o2,
                 sentence="Karachi is the largest city of Pakistan.",
                 triples=[Triple("Karachi", "country", "Pakistan")]),
        GoldItem(sent_id="s4", ontology_id=o2, stratum=o2,
                 sentence="Lahore is located in Punjab, Pakistan.",
                 triples=[Triple("Lahore", "country", "Pakistan"),
                          Triple("Lahore", "located in", "Punjab")]),
        GoldItem(sent_id="s5", ontology_id=o2, stratum=o2,
                 sentence="Islamabad is the capital of Pakistan.",
                 triples=[Triple("Islamabad", "capital of", "Pakistan")]),
    ]


ONTOS = {
    "ont_book": Ontology(id="ont_book", label="Book",
                         entity_types=["Book", "Person", "Year"],
                         relation_types=["author", "publication year", "publisher"]),
    "ont_city": Ontology(id="ont_city", label="City",
                         entity_types=["City", "Country", "Region"],
                         relation_types=["country", "located in", "population"]),
}


def fake_model_outputs(items: List[GoldItem], noise: str) -> dict:
    """Return {sent_id: [Triple]} that mimics a model's output."""
    out = {}
    for it in items:
        trs = list(it.triples)
        if noise == "clean":
            pass
        elif noise == "noisy":
            # drop the last gold triple; add one unsupported fact; mangle one object
            if trs:
                trs = trs[:-1]
            trs.append(Triple("Elon Musk", "author", "Dune"))                 # T1
            if it.sent_id == "s3":
                trs.append(Triple("Karachi", "capital of", "Pakistan"))       # T2-ish
        out[it.sent_id] = trs
    return out


def main() -> int:
    print("[smoke] building fixtures")
    items = fixtures()
    sample = stratified_sample(items, n_total=5, seed=42)
    sample = assign_openworld_holdout(sample, frac=0.4, seed=42)

    print(f"[smoke] sample={len(sample)} "
          f"open-world={sum(1 for s in sample if s.openworld_held_out)}")

    runs = {
        "model_clean":  fake_model_outputs(sample, "clean"),
        "model_noisy":  fake_model_outputs(sample, "noisy"),
    }

    th = GateThresholds()
    for model, preds_by_id in runs.items():
        print(f"\n--- {model} ---")
        matches = []
        rows = []
        for it in sample:
            preds = preds_by_id[it.sent_id]
            m = match_sentence(it.sent_id, preds, it.triples, fuzzy_threshold=90)
            matches.append(m)
            matched_preds = {id(p) for p, _, _ in m.matched_pairs}
            # fake "variants" so instability is non-trivial
            variants = [preds, preds[:-1] if len(preds) > 1 else preds, preds]
            for p in preds:
                support = source_support(p, it.sentence)
                novelty = ontology_novelty(p, ONTOS[it.ontology_id])
                instab = paraphrase_instability(p, variants)
                gi = GateInput(triple=p, support=support, instability=instab,
                               novelty=novelty, format_ok=True)
                action = decide(gi, th)
                rows.append((p, id(p) in matched_preds, round(support, 2),
                             round(instab, 2), novelty, action))

        p, r, f = aggregate_prf1(matches, mode="canonical")
        boot = bootstrap_prf1(matches, mode="canonical", n_boot=200)
        print(f"P/R/F1 (canonical) = {p:.2f}/{r:.2f}/{f:.2f}")
        print(f"  F1 95% CI = [{boot['f1'][1]:.2f}, {boot['f1'][2]:.2f}]")

        decisions = [row[5] for row in rows]
        correct   = [row[1] for row in rows]
        gc = gated_counts(decisions, correct)
        print("  gate:", gc)
        for pt, match, sup, ins, nov, act in rows:
            print(f"    {act:>23}  match={str(match):5}  "
                  f"supp={sup:.2f} inst={ins:.2f} nov={nov:9}  "
                  f"({pt.subject} | {pt.relation} | {pt.object})")

    print("\n[smoke] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
