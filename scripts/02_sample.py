"""Build the frozen N=300 stratified sample and the open-world hold-out slice.

Output: data/sample/sample.jsonl  (one GoldItem per line)
        data/sample/paraphrase_subset.jsonl  (n=50 subset for RQ2)

Both files are reproducible from the config seed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve()
IMPL = HERE.parents[1]
sys.path.insert(0, str(IMPL / "src"))

from kgrel.dataset import load_gold_items, load_ontologies  # noqa: E402
from kgrel.sample import (                                   # noqa: E402
    assign_openworld_holdout, paraphrase_subset, stratified_sample,
)
from kgrel.schema import dumps_dc                            # noqa: E402


def main() -> int:
    cfg = yaml.safe_load((IMPL / "configs" / "config.yaml").read_text(encoding="utf-8"))

    ontologies = load_ontologies(str(IMPL / cfg["benchmark"]["ontologies_glob"]))
    print(f"[sample] loaded {len(ontologies)} ontologies")

    items = list(load_gold_items(
        str(IMPL / cfg["benchmark"]["tests_glob"]),
        str(IMPL / cfg["benchmark"]["ground_glob"]),
    ))
    print(f"[sample] loaded {len(items)} test sentences with gold")

    picked = stratified_sample(items, cfg["sample"]["n_total"], seed=cfg["seed"])
    picked = assign_openworld_holdout(
        picked, cfg["sample"]["openworld_holdout_frac"], seed=cfg["seed"],
    )
    n_hold = sum(1 for p in picked if p.openworld_held_out)
    print(f"[sample] stratified sample: {len(picked)} ({n_hold} open-world hold-out)")

    para = paraphrase_subset(picked, cfg["sample"]["paraphrase_subset_n"], seed=cfg["seed"])
    print(f"[sample] paraphrase subset: {len(para)}")

    out_dir = IMPL / cfg["paths"]["sample"]
    out_dir.mkdir(parents=True, exist_ok=True)

    def dump(path: Path, rows):
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(dumps_dc(r), ensure_ascii=False) + "\n")

    dump(out_dir / "sample.jsonl", picked)
    dump(out_dir / "paraphrase_subset.jsonl", para)
    print(f"[sample] wrote {out_dir / 'sample.jsonl'}")
    print(f"[sample] wrote {out_dir / 'paraphrase_subset.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
