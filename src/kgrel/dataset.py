"""Text2KGBench Wikidata-TekGen loader.

Expected on-disk layout under `data/raw/text2kgbench/`:

    benchmark/
      ontologies/
        ont_<id>_<label>.json            # has id, label, concepts, relations
      testing/
        <ontology_id>_test.jsonl         # each line: {"id","sent_id","sentence"}
      ground_truth/
        <ontology_id>_test.jsonl         # each line: {"id","sent_id","triples":[{sub,rel,obj}, ...]}

This loader is tolerant: it only loads what exists and skips the rest.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Dict, Iterator, List, Optional

from .schema import GoldItem, Ontology, Triple


def _label_of(x: Any) -> str:
    """Text2KGBench ontology items are {'qid': 'Q...', 'label': '...'}."""
    if isinstance(x, dict):
        return str(x.get("label") or x.get("name") or "").strip()
    return str(x).strip()


def load_ontologies(ontologies_glob: str) -> Dict[str, Ontology]:
    out: Dict[str, Ontology] = {}
    for path in sorted(glob.glob(ontologies_glob)):
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        oid = str(raw.get("id") or raw.get("ontology_id")
                  or os.path.splitext(os.path.basename(path))[0])
        label = str(raw.get("title") or raw.get("label") or oid)
        ont = Ontology(
            id=oid,
            label=label,
            description=str(raw.get("description", "")),
            entity_types=[_label_of(x) for x in
                          (raw.get("concepts") or raw.get("entity_types") or [])
                          if _label_of(x)],
            relation_types=[_label_of(x) for x in
                            (raw.get("relations") or raw.get("relation_types") or [])
                            if _label_of(x)],
        )
        out[ont.id] = ont
    return out


def _index_ground_truth(ground_glob: str) -> Dict[str, Dict[str, List[Triple]]]:
    """Returns {ontology_id: {sent_id: [Triple, ...]}}."""
    idx: Dict[str, Dict[str, List[Triple]]] = {}
    for path in sorted(glob.glob(ground_glob)):
        ont_id = _ont_id_from_filename(os.path.basename(path))
        bucket = idx.setdefault(ont_id, {})
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                sent_id = str(row.get("sent_id") or row.get("id"))
                triples = [Triple.from_dict(t) for t in (row.get("triples") or [])]
                bucket[sent_id] = triples
    return idx


def _ont_id_from_filename(fname: str) -> str:
    """Normalise '{...}_test.jsonl', '{...}_ground_truth.jsonl', '{N}_{name}_ontology.json'
    to a single ontology id like 'ont_4_book'.
    """
    base = os.path.splitext(fname)[0]
    for suffix in ("_ground_truth", "_test", "_ontology"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    # Ontology files are named '4_book_ontology.json' (no 'ont_' prefix),
    # but tests / gt use 'ont_4_book_test.jsonl'. Normalise to 'ont_<id>'.
    if not base.startswith("ont_"):
        base = "ont_" + base
    return base


def load_gold_items(tests_glob: str, ground_glob: str) -> Iterator[GoldItem]:
    """Yield GoldItem for every sentence in the test set that has gold triples."""
    gt_index = _index_ground_truth(ground_glob)
    for path in sorted(glob.glob(tests_glob)):
        ont_id = _ont_id_from_filename(os.path.basename(path))
        gt = gt_index.get(ont_id, {})
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                sent_id = str(row.get("sent_id") or row.get("id"))
                sentence = str(row.get("sent") or row.get("sentence") or "")
                triples = gt.get(sent_id, [])
                if not triples:
                    continue   # skip sentences with no gold (sampler needs gold)
                yield GoldItem(
                    sent_id=sent_id,
                    ontology_id=ont_id,
                    sentence=sentence,
                    triples=triples,
                    stratum=ont_id,
                )


def load_all(ontologies_glob: str, tests_glob: str, ground_glob: str
             ) -> Dict[str, object]:
    return {
        "ontologies": load_ontologies(ontologies_glob),
        "items": list(load_gold_items(tests_glob, ground_glob)),
    }
