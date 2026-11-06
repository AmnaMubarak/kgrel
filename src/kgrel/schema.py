"""Core dataclasses: Triple, Ontology, GoldItem, Prediction, LabeledTriple.

Pure stdlib. Pydantic is used elsewhere, not here, so we keep this file 3.8 safe
and easy to import from scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any


# -------- Core atoms --------

@dataclass(frozen=True)
class Triple:
    subject: str
    relation: str
    object: str

    def as_tuple(self) -> tuple:
        return (self.subject, self.relation, self.object)

    def to_dict(self) -> Dict[str, str]:
        return {"subject": self.subject, "relation": self.relation, "object": self.object}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Triple":
        # Text2KGBench uses sub/rel/obj; our internal API uses subject/relation/object.
        return cls(
            subject=str(d.get("subject") or d.get("sub") or d.get("head") or "").strip(),
            relation=str(d.get("relation") or d.get("rel") or d.get("predicate") or "").strip(),
            object=str(d.get("object") or d.get("obj") or d.get("tail") or "").strip(),
        )


@dataclass
class Ontology:
    """A Text2KGBench-style ontology: id, label, entity types, relation types."""
    id: str
    label: str
    entity_types: List[str] = field(default_factory=list)
    relation_types: List[str] = field(default_factory=list)
    description: str = ""

    def relations_set(self) -> set:
        return {r.lower() for r in self.relation_types}


# -------- Gold / predictions --------

@dataclass
class GoldItem:
    """One benchmark sentence with its gold triples."""
    sent_id: str
    ontology_id: str
    sentence: str
    triples: List[Triple]
    # fields filled by sampler
    stratum: Optional[str] = None            # ontology id by default
    openworld_held_out: bool = False         # True = hide gold relation types from the prompt
    held_out_relations: List[str] = field(default_factory=list)


@dataclass
class Prediction:
    sent_id: str
    model: str
    run_id: str                              # extraction run (e.g. "paraphrase_v1_s3")
    triples: List[Triple]
    raw_output: str = ""
    parse_ok: bool = True
    latency_s: float = 0.0


@dataclass
class LabeledTriple:
    """A model-emitted triple after matching + human / rule labeling."""
    sent_id: str
    model: str
    run_id: str
    triple: Triple
    matched_gold: Optional[Triple] = None    # strict or canonical match
    match_mode: Optional[str] = None         # "strict" | "canonical" | None
    error_type: Optional[str] = None         # T1..T5 or None if correct
    source_support: Optional[float] = None   # risk signal in [0,1]
    stability: Optional[float] = None        # risk signal in [0,1]
    novelty: Optional[str] = None            # "in_schema" | "synonym" | "new_type" | "junk"
    gate_action: Optional[str] = None        # "insert" | "abstain" | "propose_schema_update"


def dumps_dc(obj: Any) -> Any:
    """asdict for dataclasses, passthrough otherwise. Lists supported."""
    if isinstance(obj, list):
        return [dumps_dc(o) for o in obj]
    try:
        return asdict(obj)
    except TypeError:
        return obj
