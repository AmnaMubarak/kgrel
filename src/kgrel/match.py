"""Match predicted triples against gold triples.

Returns per-sentence matches so the metrics module can compute P/R/F1 and the
error-labeler can type the misses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .canonical import fuzzy_match, norm
from .schema import Triple


@dataclass
class MatchResult:
    sent_id: str
    tp_strict: int
    tp_canonical: int
    fp: int
    fn: int
    matched_pairs: List[Tuple[Triple, Triple, str]]      # (pred, gold, mode)
    unmatched_preds: List[Triple]
    unmatched_gold: List[Triple]


def _triple_eq_strict(a: Triple, b: Triple) -> bool:
    return a.subject == b.subject and a.relation == b.relation and a.object == b.object


def _triple_eq_canonical(a: Triple, b: Triple, fuzzy_threshold: int = 90) -> bool:
    return (
        fuzzy_match(a.subject, b.subject, fuzzy_threshold)
        and fuzzy_match(a.relation, b.relation, fuzzy_threshold)
        and fuzzy_match(a.object, b.object, fuzzy_threshold)
    )


def match_sentence(sent_id: str, preds: List[Triple], gold: List[Triple],
                   fuzzy_threshold: int = 90) -> MatchResult:
    """Greedy matching: strict first, then canonical, each gold at most once."""
    matched_pairs: List[Tuple[Triple, Triple, str]] = []
    used_gold = set()
    used_pred = set()

    # 1) strict
    for pi, p in enumerate(preds):
        for gi, g in enumerate(gold):
            if gi in used_gold:
                continue
            if _triple_eq_strict(p, g):
                matched_pairs.append((p, g, "strict"))
                used_pred.add(pi)
                used_gold.add(gi)
                break

    # 2) canonical / fuzzy on what remains
    for pi, p in enumerate(preds):
        if pi in used_pred:
            continue
        for gi, g in enumerate(gold):
            if gi in used_gold:
                continue
            if _triple_eq_canonical(p, g, fuzzy_threshold):
                matched_pairs.append((p, g, "canonical"))
                used_pred.add(pi)
                used_gold.add(gi)
                break

    unmatched_preds = [p for i, p in enumerate(preds) if i not in used_pred]
    unmatched_gold = [g for i, g in enumerate(gold) if i not in used_gold]

    tp_strict = sum(1 for _, _, m in matched_pairs if m == "strict")
    tp_canonical = len(matched_pairs)

    return MatchResult(
        sent_id=sent_id,
        tp_strict=tp_strict,
        tp_canonical=tp_canonical,
        fp=len(unmatched_preds),
        fn=len(unmatched_gold),
        matched_pairs=matched_pairs,
        unmatched_preds=unmatched_preds,
        unmatched_gold=unmatched_gold,
    )
