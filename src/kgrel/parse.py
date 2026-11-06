"""Tolerant JSON parser for model outputs.

Models drift: markdown fences, trailing commas, prose preamble, single quotes.
We try several recovery steps before giving up and marking a format failure.
"""

from __future__ import annotations

import json
import re
from typing import List, Tuple

from .schema import Triple


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _try_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def parse_triples(raw: str) -> Tuple[List[Triple], bool]:
    """Return (triples, parse_ok). parse_ok is False on format failure (T5)."""
    if raw is None:
        return [], False
    text = raw.strip()
    if not text:
        return [], False

    # 1) direct
    data = _try_json(text)

    # 2) ```json ... ```
    if data is None:
        m = _FENCE_RE.search(text)
        if m:
            data = _try_json(m.group(1).strip())

    # 3) largest {...} substring
    if data is None:
        m = _OBJECT_RE.search(text)
        if m:
            data = _try_json(m.group(0))

    # 4) replace single quotes as a last resort
    if data is None and "'" in text and '"' not in text:
        data = _try_json(text.replace("'", '"'))

    if data is None:
        return [], False

    triples_raw = []
    if isinstance(data, dict):
        if isinstance(data.get("triples"), list):
            triples_raw = data["triples"]
        elif all(k in data for k in ("subject", "relation", "object")):
            triples_raw = [data]
    elif isinstance(data, list):
        triples_raw = data

    triples: List[Triple] = []
    for t in triples_raw:
        if not isinstance(t, dict):
            continue
        s = str(t.get("subject") or t.get("head") or "").strip()
        r = str(t.get("relation") or t.get("predicate") or "").strip()
        o = str(t.get("object") or t.get("tail") or "").strip()
        if s and r and o:
            triples.append(Triple(s, r, o))

    return triples, True
