"""Phase 4 — paraphrase sub-study for the `paraphrase_instability` signal.

Three phases run back-to-back in one command:

    A  Generate k paraphrases per sentence for the frozen paraphrase_subset
       (50 sentences). Uses one small model (default: gpt-oss-20b) with
       `llm_paraphrase()`; falls back to a deterministic rewriter if the LLM
       returns fewer than k variants. Writes
       data/sample/paraphrase_variants.jsonl  (frozen, resumable).

    B  Extract triples from each paraphrase variant with every configured
       extraction model. Writes
       data/runs/<model>/<run_id>_para_v<j>.jsonl  for j in 0..k-1.
       Reuses cached files if they exist (so a crashed run resumes cheaply).

    C  Score instability per original triple: for each triple produced from
       the ORIGINAL sentence (read from data/runs/<model>/core.jsonl),
       instability = fraction of paraphrase variants where the triple does
       NOT reappear under canonical fuzzy matching (threshold from
       config.yaml). Writes
       data/results/<run_id>_paraphrase_per_triple.csv
       data/results/<run_id>_paraphrase_summary.json

Call budget: 50 generation calls + 50 x k x M extraction calls. With the
default k = 3 paraphrases per sentence and M = 3 models that is
    50 + 50*3*3 = 500 API calls, ~35-45 min wall-clock.

Usage:
    python scripts/14_paraphrase.py --run-id core
    python scripts/14_paraphrase.py --run-id core --generator-model gpt-oss-20b
    python scripts/14_paraphrase.py --run-id core --k 3 --resume
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import yaml
from tqdm import tqdm

HERE = Path(__file__).resolve()
IMPL = HERE.parents[1]
sys.path.insert(0, str(IMPL / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(IMPL / ".env")
except ImportError:
    pass

from kgrel.canonical import fuzzy_match                         # noqa: E402
from kgrel.extract import ModelSpec, extract_one, make_client   # noqa: E402
from kgrel.paraphrase import llm_paraphrase, mock_paraphrase    # noqa: E402
from kgrel.schema import GoldItem, Prediction, Triple, dumps_dc # noqa: E402


def load_paraphrase_subset(path: Path) -> List[GoldItem]:
    items: List[GoldItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        items.append(GoldItem(
            sent_id=row["sent_id"],
            ontology_id=row["ontology_id"],
            sentence=row["sentence"],
            triples=[Triple(**t) for t in row.get("triples", [])],
            stratum=row.get("stratum"),
            openworld_held_out=bool(row.get("openworld_held_out", False)),
            held_out_relations=row.get("held_out_relations") or [],
        ))
    return items


def load_ontologies(cfg) -> Dict[str, object]:
    from kgrel.dataset import load_or_synthesize_ontologies
    sample = load_paraphrase_subset(IMPL / cfg["paths"]["sample"] / "sample.jsonl")
    return load_or_synthesize_ontologies(
        str(IMPL / cfg["benchmark"]["ontologies_glob"]), fallback_items=sample)


def phase_a_generate(items: List[GoldItem], k: int, generator_spec: ModelSpec,
                     out_path: Path, resume: bool) -> List[dict]:
    """Return list of dicts: {sent_id, original, variants: [str, ...]}."""
    if resume and out_path.exists():
        print(f"[para][A] reusing {out_path.name} (--resume)")
        return [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines()
                if l.strip()]

    print(f"[para][A] generating {k} paraphrases per sentence with {generator_spec.name}")
    client = make_client(generator_spec)
    variants: List[dict] = []
    for item in tqdm(items, desc="paraphrase-gen"):
        got = []
        try:
            got = llm_paraphrase(client, item.sentence, k=k)
        except Exception as e:
            print(f"  [{item.sent_id}] llm_paraphrase raised: {e}")
        # Pad with deterministic rewrites if the LLM returned fewer than k.
        if len(got) < k:
            fallback = mock_paraphrase(item.sentence, k=k)
            got = got + [v for v in fallback if v not in got]
            got = got[:k]
        variants.append({
            "sent_id": item.sent_id, "ontology_id": item.ontology_id,
            "original": item.sentence, "variants": got,
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(v, ensure_ascii=False) for v in variants)
                        + "\n", encoding="utf-8")
    print(f"[para][A] wrote {out_path}  ({len(variants)} sentences)")
    return variants


def phase_b_extract(variants: List[dict], model_cfgs: List[dict],
                    ontologies: dict, run_id: str, runs_dir: Path,
                    resume: bool) -> None:
    """For each (model, variant), call extract_one and save predictions."""
    k = max(len(v["variants"]) for v in variants)
    for mc in model_cfgs:
        spec = ModelSpec(
            name=mc["name"], backend=mc["backend"],
            api_model_id=mc.get("api_model_id"),
            temperature=mc.get("temperature", 0.0),
            top_p=mc.get("top_p", 1.0),
            max_tokens=mc.get("max_tokens", 2048),
        )
        client = make_client(spec)
        out_dir = runs_dir / spec.name; out_dir.mkdir(parents=True, exist_ok=True)

        for j in range(k):
            out_path = out_dir / f"{run_id}_para_v{j}.jsonl"
            if resume and out_path.exists():
                print(f"[para][B] {spec.name} v{j}: reusing {out_path.name} (--resume)")
                continue
            print(f"[para][B] {spec.name}  variant {j+1}/{k}  -> {out_path.name}")
            with out_path.open("w", encoding="utf-8") as fh:
                for v in tqdm(variants, desc=f"{spec.name}-v{j}"):
                    ont = ontologies.get(v["ontology_id"])
                    if ont is None or j >= len(v["variants"]):
                        continue
                    # synthesize a GoldItem-like object so extract_one can
                    # render the prompt using the paraphrase as "sentence"
                    item = GoldItem(
                        sent_id=f"{v['sent_id']}#v{j}",
                        ontology_id=v["ontology_id"],
                        sentence=v["variants"][j],
                        triples=[], stratum=v["ontology_id"],
                        openworld_held_out=False, held_out_relations=[],
                    )
                    pred = extract_one(client, item, ont, run_id=run_id)
                    # restore the real sent_id (without the #vj suffix)
                    pred.sent_id = v["sent_id"]
                    rec = dumps_dc(pred)
                    rec["paraphrase_idx"] = j
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _triples_in(target: Triple, bag: List[Triple], fuzzy: int = 90) -> bool:
    for t in bag:
        if (fuzzy_match(t.subject, target.subject, fuzzy)
                and fuzzy_match(t.relation, target.relation, fuzzy)
                and fuzzy_match(t.object, target.object, fuzzy)):
            return True
    return False


def phase_c_score(variants: List[dict], model_cfgs: List[dict], run_id: str,
                  runs_dir: Path, results_dir: Path, fuzzy: int) -> None:
    import csv

    para_sent_ids = {v["sent_id"] for v in variants}
    k = max(len(v["variants"]) for v in variants)

    rows_out = []
    summary = {}
    for mc in model_cfgs:
        model = mc["name"]
        # 1) Load original-sentence predictions from core.jsonl, restricted to
        #    the 50 paraphrase sent_ids.
        orig_path = runs_dir / model / "core.jsonl"
        if not orig_path.exists():
            print(f"[para][C] {model}: NO core.jsonl; skipping")
            continue
        orig_triples_by_sent: Dict[str, List[Triple]] = {}
        for line in orig_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["sent_id"] not in para_sent_ids:
                continue
            orig_triples_by_sent[row["sent_id"]] = [
                Triple(**t) for t in row.get("triples", [])
            ]

        # 2) Load paraphrase variant predictions, grouped by sent_id -> [[t...], ...]
        para_triples_by_sent: Dict[str, List[List[Triple]]] = {
            sid: [[] for _ in range(k)] for sid in para_sent_ids
        }
        for j in range(k):
            vpath = runs_dir / model / f"{run_id}_para_v{j}.jsonl"
            if not vpath.exists():
                continue
            for line in vpath.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                para_triples_by_sent.setdefault(row["sent_id"],
                                                [[] for _ in range(k)])
                para_triples_by_sent[row["sent_id"]][j] = [
                    Triple(**t) for t in row.get("triples", [])
                ]

        # 3) For each original triple, compute instability
        per_model = {"n_triples": 0, "mean_instability": 0.0,
                     "triples_always_reproduced": 0,
                     "triples_never_reproduced": 0}
        for sid, origs in orig_triples_by_sent.items():
            variants_for_sid = para_triples_by_sent.get(sid, [])
            for t in origs:
                presence = [_triples_in(t, bag, fuzzy) for bag in variants_for_sid]
                if not presence:
                    continue
                inst = 1.0 - (sum(presence) / len(presence))
                rows_out.append({
                    "model": model, "sent_id": sid,
                    "subject": t.subject, "relation": t.relation, "object": t.object,
                    "in_p0": presence[0] if len(presence) > 0 else False,
                    "in_p1": presence[1] if len(presence) > 1 else False,
                    "in_p2": presence[2] if len(presence) > 2 else False,
                    "paraphrase_instability": round(inst, 3),
                })
                per_model["n_triples"] += 1
                per_model["mean_instability"] += inst
                if inst == 0.0:
                    per_model["triples_always_reproduced"] += 1
                elif inst == 1.0:
                    per_model["triples_never_reproduced"] += 1
        if per_model["n_triples"]:
            per_model["mean_instability"] = round(
                per_model["mean_instability"] / per_model["n_triples"], 3)
        summary[model] = per_model

    # Write artifacts
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{run_id}_paraphrase_per_triple.csv"
    if rows_out:
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
            w.writeheader(); w.writerows(rows_out)
        print(f"[para][C] wrote {csv_path}  ({len(rows_out)} triples)")

    sum_path = results_dir / f"{run_id}_paraphrase_summary.json"
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[para][C] wrote {sum_path}")
    for m, s in summary.items():
        print(f"  {m:16s}  n={s['n_triples']:3d}  "
              f"mean_instability={s['mean_instability']:.3f}  "
              f"always_reproduced={s['triples_always_reproduced']}  "
              f"never_reproduced={s['triples_never_reproduced']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="core")
    ap.add_argument("--k", type=int, default=3,
                    help="paraphrases per sentence (default 3)")
    ap.add_argument("--generator-model", default="gpt-oss-20b",
                    help="name of the model used to generate paraphrases")
    ap.add_argument("--resume", action="store_true",
                    help="skip phases whose outputs already exist")
    args = ap.parse_args()

    cfg = yaml.safe_load((IMPL / "configs" / "config.yaml").read_text(encoding="utf-8"))
    model_cfgs = cfg["models"]
    generator_cfg = next((m for m in model_cfgs if m["name"] == args.generator_model),
                         None)
    if generator_cfg is None:
        print(f"[para] FATAL: --generator-model {args.generator_model} "
              f"not in config.yaml models; options: "
              f"{[m['name'] for m in model_cfgs]}")
        return 2
    generator_spec = ModelSpec(
        name=generator_cfg["name"], backend=generator_cfg["backend"],
        api_model_id=generator_cfg.get("api_model_id"),
        temperature=generator_cfg.get("temperature", 0.0),
        top_p=generator_cfg.get("top_p", 1.0),
        max_tokens=generator_cfg.get("max_tokens", 2048),
    )

    sample_dir = IMPL / cfg["paths"]["sample"]
    runs_dir = IMPL / cfg["paths"]["runs"]
    results_dir = IMPL / cfg["paths"]["results"]

    subset = load_paraphrase_subset(sample_dir / "paraphrase_subset.jsonl")
    print(f"[para] {len(subset)} sentences in paraphrase_subset.jsonl")

    ontologies = load_ontologies(cfg)

    # Phase A — generate paraphrases
    variants_path = sample_dir / "paraphrase_variants.jsonl"
    variants = phase_a_generate(subset, args.k, generator_spec,
                                variants_path, resume=args.resume)

    # Phase B — extract triples from each (model x variant)
    phase_b_extract(variants, model_cfgs, ontologies, args.run_id,
                    runs_dir, resume=args.resume)

    # Phase C — score instability per original triple
    fuzzy = cfg["canonical"]["fuzzy_threshold"]
    phase_c_score(variants, model_cfgs, args.run_id, runs_dir, results_dir, fuzzy)

    print("[para] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
