"""Insert / abstain / propose-schema-update gate (RQ3).

The gate is intentionally transparent: a short rule block. It consumes the
risk signals from `signals.py` plus a format-OK flag, and emits one of:

    insert
    abstain
    propose_schema_update

The thresholds live in `configs/config.yaml` so they can be tuned after the
pilot without touching code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .schema import Triple


@dataclass
class GateThresholds:
    support_min: float = 0.5
    stability_min: float = 0.5          # 1 - instability
    propose_schema_min_support: float = 0.75


@dataclass
class GateInput:
    triple: Triple
    support: float
    instability: float
    novelty: str                        # "in_schema" | "synonym" | "new_type" | "junk"
    format_ok: bool = True


def decide(inp: GateInput, th: Optional[GateThresholds] = None) -> str:
    th = th or GateThresholds()

    # Format failures and junk never enter.
    if not inp.format_ok or inp.novelty == "junk":
        return "abstain"

    stability = 1.0 - inp.instability
    supported = inp.support >= th.support_min
    stable = stability >= th.stability_min

    # Unsupported or unstable triples abstain regardless of novelty.
    if not supported or not stable:
        return "abstain"

    if inp.novelty in ("in_schema", "synonym"):
        return "insert"

    if inp.novelty == "new_type":
        if inp.support >= th.propose_schema_min_support:
            return "propose_schema_update"
        return "abstain"

    return "abstain"
