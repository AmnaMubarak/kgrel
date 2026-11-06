"""Paraphrase generation + meaning-preservation filter.

For the thesis we need paraphrases of the SOURCE sentence (not the prompt), with
a filter that keeps only paraphrases that preserve meaning. We expose two paths:

- LLM-based generator (same API client we use for extraction, with a dedicated
  system prompt);
- A placeholder NLI check using sentence-transformers (optional dep).

The pilot smoke test uses a trivial deterministic paraphraser so no network is
required.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ParaphraseSet:
    sent_id: str
    original: str
    variants: List[str]


# --- deterministic / mock paraphraser (used by the smoke test) --------------

def mock_paraphrase(sentence: str, k: int = 3) -> List[str]:
    """Three cheap, deterministic rewrites. NOT meaning-preserving in general;
    they exist only so the end-to-end pipeline runs without a network call."""
    base = sentence.strip().rstrip(".")
    v1 = f"In fact, {base.lower()}."
    v2 = f"It is reported that {base.lower()}."
    v3 = f"{base}, according to the source."
    return [v1, v2, v3][:k]


# --- LLM paraphraser --------------------------------------------------------

PARA_SYSTEM = (
    "Rewrite the user's sentence while preserving its meaning exactly. "
    "Change word order and phrasing; keep every named entity, number, and date. "
    "Return ONLY a JSON object: {\"paraphrases\": [\"...\", \"...\", \"...\"]}."
)


def llm_paraphrase(client, sentence: str, k: int = 3) -> List[str]:
    """Ask a chat client for k meaning-preserving paraphrases."""
    msgs = [
        {"role": "system", "content": PARA_SYSTEM},
        {"role": "user", "content": f"Sentence: {sentence}\nReturn {k} paraphrases."},
    ]
    raw = client.call(msgs)
    try:
        data = json.loads(raw)
        out = [str(x).strip() for x in (data.get("paraphrases") or []) if str(x).strip()]
        return out[:k]
    except Exception:
        return []


# --- NLI meaning-preservation filter ---------------------------------------

def nli_filter(original: str, variants: List[str], threshold: float = 0.6
               ) -> List[str]:
    """Keep variants whose entailment-against-original score passes `threshold`.

    Requires `sentence-transformers`. If missing, returns `variants` unchanged
    with a warning printed once.
    """
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
    except ImportError:
        print("[paraphrase] sentence-transformers not installed; skipping NLI filter.")
        return variants

    model = CrossEncoder("cross-encoder/nli-deberta-v3-base")
    kept: List[str] = []
    for v in variants:
        # The CrossEncoder returns logits [contradiction, entailment, neutral]
        scores = model.predict([(original, v), (v, original)])
        ent_a = float(scores[0][1])
        ent_b = float(scores[1][1])
        if min(ent_a, ent_b) >= threshold:
            kept.append(v)
    return kept
