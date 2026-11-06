"""Entity + relation canonicalisation.

Strict matching (exact string) is kept so we can report the strict-vs-canonical
delta, as Wadhwa et al. (2023) recommend. Canonical matching lowercases,
strips punctuation, drops articles, and uses rapidfuzz for near-ties.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

try:
    from rapidfuzz import fuzz
except ImportError:  # fallback so dry-run works without the dep
    class _FallbackFuzz:
        @staticmethod
        def WRatio(a: str, b: str) -> int:
            return 100 if a == b else 0
    fuzz = _FallbackFuzz()  # type: ignore


_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_WS_RE = re.compile(r"\s+")
_ARTICLES = {"the", "a", "an"}


def norm(text: str, drop_articles: bool = True) -> str:
    s = text.strip().lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    if drop_articles:
        s = " ".join(tok for tok in s.split() if tok not in _ARTICLES)
    return s


def fuzzy_match(a: str, b: str, threshold: int = 90) -> bool:
    if norm(a) == norm(b):
        return True
    return fuzz.WRatio(norm(a), norm(b)) >= threshold


def match_relation(pred_rel: str, allowed: Iterable[str], threshold: int = 92
                   ) -> Optional[str]:
    """Return the ontology relation that best matches a predicted relation,
    else None. 'Best' = exact normalized match or WRatio >= threshold.
    """
    p = norm(pred_rel)
    best = None
    best_score = -1
    for cand in allowed:
        c = norm(cand)
        if c == p:
            return cand
        score = fuzz.WRatio(p, c)
        if score > best_score:
            best_score = score
            best = cand
    return best if best_score >= threshold else None
