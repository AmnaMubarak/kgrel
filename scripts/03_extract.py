"""Run the configured models on the frozen sample.

Output: data/runs/<model>/<run_id>.jsonl   (one Prediction per line)

Usage:
    python thesis/implementation/scripts/03_extract.py --models all --run-id core
    python thesis/implementation/scripts/03_extract.py --models mock --run-id dryrun
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

from kgrel.dataset import load_or_synthesize_ontologies  # noqa: E402
from kgrel.extract import ModelSpec, extract_one, make_client  # noqa: E402
from kgrel.schema import GoldItem, Triple, dumps_dc  # noqa: E402


def load_sample(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        yield GoldItem(
            sent_id=row["sent_id"],
            ontology_id=row["ontology_id"],
            sentence=row["sentence"],
            triples=[Triple(**t) for t in row.get("triples", [])],
            stratum=row.get("stratum"),
            openworld_held_out=bool(row.get("openworld_held_out", False)),
            held_out_relations=row.get("held_out_relations") or [],
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all",
                    help="comma-separated model names, or 'all'")
    ap.add_argument("--run-id", default="core")
    ap.add_argument("--limit", type=int, default=None,
                    help="only run this many sentences (debug)")
    ap.add_argument("--mock", action="store_true",
                    help="force backend=mock for every model (no API key needed)")
    args = ap.parse_args()

    cfg = yaml.safe_load((IMPL / "configs" / "config.yaml").read_text(encoding="utf-8"))
    model_cfgs = cfg["models"]
    if args.models != "all":
        wanted = {m.strip() for m in args.models.split(",")}
        model_cfgs = [m for m in model_cfgs if m["name"] in wanted]
    if not model_cfgs:
        print(f"[extract] no models matched {args.models}")
        return 2

    sample = list(load_sample(IMPL / cfg["paths"]["sample"] / "sample.jsonl"))
    if args.limit:
        sample = sample[: args.limit]
    print(f"[extract] sample={len(sample)} models={[m['name'] for m in model_cfgs]}")

    ontologies = load_or_synthesize_ontologies(
        str(IMPL / cfg["benchmark"]["ontologies_glob"]),
        fallback_items=sample,
    )
    if not ontologies:
        print("[extract] FATAL: no ontologies available. Nothing to run.")
        return 3

    runs_dir = IMPL / cfg["paths"]["runs"]
    runs_dir.mkdir(parents=True, exist_ok=True)

    for mc in model_cfgs:
        spec = ModelSpec(
            name=mc["name"], backend="mock" if args.mock else mc["backend"],
            api_model_id=mc.get("api_model_id"),
            temperature=mc.get("temperature", 0.0),
            top_p=mc.get("top_p", 1.0),
            max_tokens=mc.get("max_tokens", 512),
        )
        client = make_client(spec)
        out_dir = runs_dir / spec.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.run_id}.jsonl"
        print(f"[extract] -> {out_path}")

        n_done = 0
        n_skipped = 0
        with out_path.open("w", encoding="utf-8") as fh:
            for item in tqdm(sample, desc=spec.name):
                ont = ontologies.get(item.ontology_id)
                if ont is None:
                    n_skipped += 1
                    continue
                pred = extract_one(client, item, ont, run_id=args.run_id)
                fh.write(json.dumps(dumps_dc(pred), ensure_ascii=False) + "\n")
                n_done += 1
        print(f"[extract]   {spec.name}: {n_done} done, {n_skipped} skipped")

    print("[extract] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
