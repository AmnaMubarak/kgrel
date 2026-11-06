"""Three risk signals for RQ2.

- source_support(triple, sentence): is the triple grounded in the passage?
- paraphrase_instability(triple, variants_preds): how often does the triple
  disappear across paraphrase variants of the same source sentence?
- ontology_novelty(triple, ontology): known relation, synonym of a known
  relation, candidate new type, or junk?
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

from .canonical import fuzzy_match, match_relation, norm
from .schema import Ontology, Triple


# -------- Source support (string-overlap baseline) --------
# A deeper NLI-based implementation can replace this later. For the pilot we
# use token-overlap with the source sentence as a cheap, reproducible floor.

def source_support(triple: Triple, sentence: str) -> float:
    s_tokens = set(norm(sentence).split())
    if not s_tokens:
        return 0.0
    t_tokens = set(norm(triple.subject).split()) | set(norm(triple.object).split())
    if not t_tokens:
        return 0.0
    return len(t_tokens & s_tokens) / len(t_tokens)


# -------- Paraphrase instability --------

def _triples_contain(triples: Sequence[Triple], target: Triple, fuzzy: int = 90) -> bool:
    for t in triples:
        if (fuzzy_match(t.subject, target.subject, fuzzy)
                and fuzzy_match(t.relation, target.relation, fuzzy)
                and fuzzy_match(t.object, target.object, fuzzy)):
            return True
    return False


def paraphrase_instability(triple: Triple,
                           variants_triples: Iterable[Sequence[Triple]]) -> float:
    """Fraction of paraphrase variants where the triple is NOT produced.
    1.0 = always disappears (very unstable). 0.0 = reappears every time.
    """
    variants = list(variants_triples)
    if not variants:
        return 0.0
    missing = sum(0 if _triples_contain(v, triple) else 1 for v in variants)
    return missing / len(variants)


# -------- Ontology novelty --------

def ontology_novelty(triple: Triple, ontology: Ontology,
                     synonym_fuzzy: int = 92) -> str:
    """Return one of:
        "in_schema"  — relation matches an ontology relation exactly
        "synonym"    — relation matches an ontology relation under fuzzy threshold
        "new_type"   — relation looks plausible but is not in the ontology
        "junk"       — relation is empty / nonsense
    """
    r = (triple.relation or "").strip()
    if not r:
        return "junk"
    if r in ontology.relation_types:
        return "in_schema"
    if match_relation(r, ontology.relation_types, threshold=synonym_fuzzy) is not None:
        return "synonym"
    # Heuristic junk check: at least two chars and not pure punctuation.
    if len(norm(r)) < 2:
        return "junk"
    return "new_type"
