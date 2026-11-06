"""Stratified sampler for N=300 across the 10 Wikidata-TekGen ontologies,
with an open-world hold-out slice where gold relation types are hidden
from the prompt.
"""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Tuple

from .schema import GoldItem


def stratified_sample(items: Iterable[GoldItem], n_total: int, seed: int = 20261022
                      ) -> List[GoldItem]:
    """Round-robin across strata (ontology id), skipping items with no gold triples."""
    rng = random.Random(seed)
    pools: Dict[str, List[GoldItem]] = {}
    for it in items:
        if not it.triples:
            continue
        pools.setdefault(it.stratum or it.ontology_id, []).append(it)

    for pool in pools.values():
        rng.shuffle(pool)

    strata = sorted(pools.keys())
    picked: List[GoldItem] = []
    idx = {s: 0 for s in strata}

    while len(picked) < n_total and any(idx[s] < len(pools[s]) for s in strata):
        for s in strata:
            if len(picked) >= n_total:
                break
            if idx[s] < len(pools[s]):
                picked.append(pools[s][idx[s]])
                idx[s] += 1

    return picked


def assign_openworld_holdout(sample: List[GoldItem], frac: float, seed: int = 20261022
                             ) -> List[GoldItem]:
    """Mark `frac` of the sample as open-world: hide those sentences' gold
    relation types from the extraction prompt.

    We hold out at the (sentence) level and record which relations are hidden
    so the gate / metrics can tell "new type seen for the first time" apart
    from "model invented a type".
    """
    rng = random.Random(seed + 1)
    n_hold = max(1, int(round(len(sample) * frac)))
    indices = list(range(len(sample)))
    rng.shuffle(indices)
    hold = set(indices[:n_hold])

    for i, item in enumerate(sample):
        if i in hold:
            item.openworld_held_out = True
            item.held_out_relations = sorted({t.relation for t in item.triples if t.relation})
    return sample


def paraphrase_subset(sample: List[GoldItem], n: int, seed: int = 20261022
                      ) -> List[GoldItem]:
    """Pick n sentences for the RQ2 paraphrase sub-study, stratified by ontology."""
    rng = random.Random(seed + 2)
    by_ont: Dict[str, List[GoldItem]] = {}
    for it in sample:
        by_ont.setdefault(it.ontology_id, []).append(it)
    for pool in by_ont.values():
        rng.shuffle(pool)

    out: List[GoldItem] = []
    strata = sorted(by_ont.keys())
    idx = {s: 0 for s in strata}
    while len(out) < n and any(idx[s] < len(by_ont[s]) for s in strata):
        for s in strata:
            if len(out) >= n:
                break
            if idx[s] < len(by_ont[s]):
                out.append(by_ont[s][idx[s]])
                idx[s] += 1
    return out
