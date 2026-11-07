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
from typing import Any, Dict, Iterable, Iterator, List, Optional

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


def synthesize_ontologies_from_sample(items: Iterable[GoldItem]
                                      ) -> Dict[str, Ontology]:
    """Build minimal Ontology objects by aggregating gold relations per
    ontology_id in the frozen sample.

    Used as a fallback when the raw Text2KGBench ontology JSONs are not on
    disk (e.g. a fresh clone of the public repo that only ships the frozen
    sample). This honours the README promise that experiments reproduce
    without the 1.4 GB benchmark download.

    Notes:
      * Only `relation_types` is populated. The `ontology_novelty` signal is
        the only consumer; entity_types are not needed by current scoring.
      * Per-sentence open-world hold-out is applied at the prompt/sample
        layer, not here. This synthesizer always returns the UNION of seen
        relations for each ontology, mirroring how `load_ontologies` would
        behave on real benchmark files.
    """
    buckets: Dict[str, set] = {}
    for item in items:
        bucket = buckets.setdefault(item.ontology_id, set())
        for t in item.triples:
            rel = (t.relation or "").strip()
            if rel:
                bucket.add(rel)
    return {
        oid: Ontology(
            id=oid,
            label=oid,
            description="(synthesized from frozen sample gold triples)",
            entity_types=[],
            relation_types=sorted(rels),
        )
        for oid, rels in buckets.items()
    }


def load_or_synthesize_ontologies(ontologies_glob: str,
                                  fallback_items: Iterable[GoldItem]
                                  ) -> Dict[str, Ontology]:
    """Try `load_ontologies(glob)`; if empty, synthesize from the sample.

    Returns a (possibly empty) dict and prints one line telling the caller
    which path was taken so nothing is silent.
    """
    ontologies = load_ontologies(ontologies_glob)
    if ontologies:
        print(f"[dataset] loaded {len(ontologies)} ontologies from disk "
              f"({ontologies_glob})")
        return ontologies
    # Fallback: synthesize from the frozen sample's gold triples.
    items_list = list(fallback_items)
    ontologies = synthesize_ontologies_from_sample(items_list)
    print(f"[dataset] no raw ontology files matched {ontologies_glob} — "
          f"synthesized {len(ontologies)} ontologies from the frozen sample "
          f"({sum(len(o.relation_types) for o in ontologies.values())} "
          f"relations total). Run scripts/01_fetch_data.py to use the "
          f"authoritative Text2KGBench schema instead.")
    return ontologies
