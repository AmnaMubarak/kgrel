"""Metrics: P/R/F1 with bootstrap CIs, per-type error rates, Cohen's kappa,
AUROC and precision@k for risk signals, gated vs ungated graph quality.
"""

from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .match import MatchResult


# -------- Precision / recall / F1 --------

def prf1_from_counts(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f


def aggregate_prf1(matches: Iterable[MatchResult], mode: str = "canonical"
                   ) -> Tuple[float, float, float]:
    tp_key = "tp_canonical" if mode == "canonical" else "tp_strict"
    tp = fp = fn = 0
    for m in matches:
        tp += getattr(m, tp_key)
        fp += m.fp - (m.tp_canonical - getattr(m, tp_key) if mode == "strict" else 0)
        fn += m.fn
    # fp above subtracts the canonical-only matches when mode="strict",
    # so strict and canonical use consistent denominators.
    fp = max(fp, 0)
    return prf1_from_counts(tp, fp, fn)


def bootstrap_prf1(matches: Sequence[MatchResult], mode: str = "canonical",
                   n_boot: int = 1000, alpha: float = 0.05,
                   seed: int = 20261022
                   ) -> Dict[str, Tuple[float, float, float]]:
    """Returns {'precision': (point, lo, hi), 'recall': ..., 'f1': ...}."""
    rng = random.Random(seed)
    N = len(matches)
    if N == 0:
        return {k: (0.0, 0.0, 0.0) for k in ("precision", "recall", "f1")}

    point = aggregate_prf1(matches, mode)
    ps, rs, fs = [], [], []
    for _ in range(n_boot):
        sample = [matches[rng.randrange(N)] for _ in range(N)]
        p, r, f = aggregate_prf1(sample, mode)
        ps.append(p); rs.append(r); fs.append(f)

    def ci(xs: List[float]) -> Tuple[float, float]:
        xs = sorted(xs)
        lo = xs[int(math.floor(alpha / 2 * len(xs)))]
        hi = xs[int(math.ceil((1 - alpha / 2) * len(xs))) - 1]
        return lo, hi

    plo, phi = ci(ps); rlo, rhi = ci(rs); flo, fhi = ci(fs)
    return {
        "precision": (point[0], plo, phi),
        "recall":    (point[1], rlo, rhi),
        "f1":        (point[2], flo, fhi),
    }


# -------- Per-type error rates --------

def per_type_rates(labels: Iterable[str], universe: Sequence[str]) -> Dict[str, float]:
    labels = list(labels)
    n = len(labels) or 1
    counts = {t: 0 for t in universe}
    for lab in labels:
        if lab in counts:
            counts[lab] += 1
    return {t: counts[t] / n for t in universe}


# -------- Cohen's kappa (two annotators, categorical) --------

def cohens_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    assert len(a) == len(b) and len(a) > 0
    cats = sorted(set(a) | set(b))
    idx = {c: i for i, c in enumerate(cats)}
    N = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / N
    na = [0] * len(cats); nb = [0] * len(cats)
    for x in a: na[idx[x]] += 1
    for y in b: nb[idx[y]] += 1
    pe = sum((na[i] * nb[i]) for i in range(len(cats))) / (N * N)
    if 1 - pe == 0:
        return 1.0
    return (po - pe) / (1 - pe)


# -------- Precision@k and AUROC for risk signals --------

def precision_at_k(scores: Sequence[float], labels: Sequence[int], k: int) -> float:
    """`labels[i] == 1` means "unsafe triple" (human: do not store)."""
    assert len(scores) == len(labels)
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    top = order[:k]
    if not top:
        return 0.0
    return sum(labels[i] for i in top) / len(top)


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Mann-Whitney / Wilcoxon style, O(n log n). labels in {0,1}."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    ranks = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg], key=lambda x: x[0])
    rank_sum = 0.0
    i = 0
    while i < len(ranks):
        j = i
        while j + 1 < len(ranks) and ranks[j + 1][0] == ranks[i][0]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            if ranks[k][1] == 1:
                rank_sum += avg_rank
        i = j + 1
    u = rank_sum - len(pos) * (len(pos) + 1) / 2
    return u / (len(pos) * len(neg))


# -------- Gated vs ungated graph quality --------

def gated_counts(decisions: Sequence[str], correct: Sequence[bool]
                 ) -> Dict[str, int]:
    """`correct[i]` is whether the triple is a true match (not an error).
    Decisions in {"insert","abstain","propose_schema_update"}."""
    out = {"insert_correct": 0, "insert_wrong": 0,
           "abstain_correct": 0, "abstain_wrong": 0,
           "proposed": 0}
    for d, c in zip(decisions, correct):
        if d == "insert":
            out["insert_correct" if c else "insert_wrong"] += 1
        elif d == "abstain":
            out["abstain_wrong" if c else "abstain_correct"] += 1
        elif d == "propose_schema_update":
            out["proposed"] += 1
    return out
