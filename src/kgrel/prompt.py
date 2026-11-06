"""Fixed Text2KGBench-style extraction prompt.

Design choices (locked for the thesis):

- Same system + user template for every model: fairness anchor.
- JSON-only output.
- Open-world sentences get a shortened ontology: the gold relation types for
  that sentence are hidden. The model may propose a new relation; the gate
  decides whether to accept, merge, or refuse it.
"""

from __future__ import annotations

from typing import List, Optional

from .schema import GoldItem, Ontology


SYSTEM = (
    "You extract knowledge-graph triples from a short passage. "
    "Return ONLY a JSON object with a single key `triples` whose value is a "
    "list of objects, each with keys `subject`, `relation`, `object`. "
    "Prefer relations that appear in the given ontology. If no triple is "
    "supported by the passage, return `{\"triples\": []}`."
)


def _format_relations(relations: List[str], max_show: int = 60) -> str:
    rels = relations[:max_show]
    extra = "" if len(relations) <= max_show else f"  (+{len(relations) - max_show} more)"
    return ", ".join(rels) + extra


def render(item: GoldItem, ontology: Ontology, hide_held_out: bool = True) -> str:
    """Return the user-side prompt text for one sentence."""
    rels = list(ontology.relation_types)
    if hide_held_out and item.openworld_held_out and item.held_out_relations:
        held = set(r.lower() for r in item.held_out_relations)
        rels = [r for r in rels if r.lower() not in held]

    entity_types = ", ".join(ontology.entity_types[:40]) or "(none provided)"
    relation_types = _format_relations(rels) or "(none; the ontology may need to grow)"

    return (
        f"Ontology: {ontology.label}\n"
        f"Entity types: {entity_types}\n"
        f"Relation types: {relation_types}\n\n"
        f"Passage:\n{item.sentence}\n\n"
        "Return a JSON object with the key `triples`."
    )


def render_messages(item: GoldItem, ontology: Ontology, hide_held_out: bool = True
                    ) -> List[dict]:
    """Chat-format wrapper."""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": render(item, ontology, hide_held_out=hide_held_out)},
    ]
