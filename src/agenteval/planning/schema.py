"""Requirement graph — the condition, turned into something checkable.

Conformance cannot be judged from the prompt text directly. "By Marina Bay the
white Merlion continuously spouts water while three towers light up and tourists
raise their phones" is one sentence and roughly nine separate claims, each true
or false independently, each needing different evidence: one is a count, one is
a continuity-over-time property, one is a co-occurrence.

Asking a VLM "does this video match the prompt?" collapses all nine into a
single impression, and impressions are where the ceiling comes from -- a video
that gets eight right and drops one subject still reads as "mostly matching".
Decomposing first makes the failure countable and, more usefully, nameable.

The compilation is **output-blind**: it looks only at the condition, never at a
candidate video. That is what makes it fair across models -- every candidate is
measured against the identical list -- and it is also the anti-gaming property,
since a policy cannot dodge a requirement it did not get to choose.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

RequirementKind = Literal[
    "entity",      # a thing must be present
    "attribute",   # a thing must have a property (colour, material, state)
    "count",       # how many
    "relation",    # spatial relation between things
    "action",      # something must happen
    "order",       # actions in a stated sequence, or concurrent
    "camera",      # camera behaviour
    "style",       # visual style / medium
    "text",        # rendered text
]

#: How the requirement is verified, which decides what evidence a skill gathers.
VerifyMode = Literal[
    "present_any",     # visible in at least one frame
    "present_most",    # visible through most of the clip
    "continuous",      # holds continuously, not just once
    "co_occur",        # several things visible together in one frame
    "sequence",        # ordering between events
    "trajectory",      # camera or object path over time
]


@dataclass
class Requirement:
    rid: str
    kind: RequirementKind
    text: str                       # what must be true, in words
    verify: VerifyMode
    aspect: str                     # which reported aspect it scores
    subject: str = ""               # entity it is about
    value: Any = None               # count, attribute value, relation, ...
    hard: bool = True               # a hard miss is disqualifying for its aspect
    weight: float = 1.0
    note: str = ""                  # what NOT to count as a failure

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OrderEdge:
    before: str                     # rid
    after: str                      # rid
    relation: Literal["before", "concurrent"] = "before"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RequirementGraph:
    condition_id: str
    condition_text: str
    requirements: list[Requirement] = field(default_factory=list)
    order: list[OrderEdge] = field(default_factory=list)
    camera: dict[str, Any] = field(default_factory=dict)
    invariants: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def by_aspect(self, aspect: str) -> list[Requirement]:
        return [r for r in self.requirements if r.aspect == aspect]

    def by_kind(self, *kinds: str) -> list[Requirement]:
        return [r for r in self.requirements if r.kind in kinds]

    def to_json(self) -> dict[str, Any]:
        return {"condition_id": self.condition_id,
                "condition_text": self.condition_text,
                "requirements": [r.to_json() for r in self.requirements],
                "order": [e.to_json() for e in self.order],
                "camera": self.camera, "invariants": self.invariants,
                "meta": self.meta}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "RequirementGraph":
        return cls(
            condition_id=d.get("condition_id", ""),
            condition_text=d.get("condition_text", ""),
            requirements=[Requirement(**r) for r in d.get("requirements", [])],
            order=[OrderEdge(**e) for e in d.get("order", [])],
            camera=d.get("camera", {}), invariants=d.get("invariants", []),
            meta=d.get("meta", {}),
        )

    def save(self, path: str | Path) -> Path:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_json(), ensure_ascii=False, indent=1),
                     encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "RequirementGraph":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


#: requirement kind -> reported aspect
KIND_TO_ASPECT: dict[str, str] = {
    "entity": "entity_presence",
    "attribute": "attribute_binding",
    "count": "object_count",
    "relation": "spatial_relation",
    "action": "action_execution",
    "order": "action_order",
    "camera": "camera_control",
    "style": "style_match",
    "text": "text_rendering",
}


def validate(g: RequirementGraph) -> list[str]:
    """Structural problems worth failing a compilation over.

    A malformed graph is worse than none: it silently changes what every model
    is measured against, and unlike a bad verdict it does so identically for all
    of them, so nothing looks anomalous.
    """
    errs: list[str] = []
    if not g.requirements:
        errs.append("no requirements extracted")
    seen: set[str] = set()
    for r in g.requirements:
        if r.rid in seen:
            errs.append(f"duplicate rid {r.rid}")
        seen.add(r.rid)
        if r.aspect not in set(KIND_TO_ASPECT.values()):
            errs.append(f"{r.rid}: unknown aspect {r.aspect!r}")
        if not r.text.strip():
            errs.append(f"{r.rid}: empty text")
    for e in g.order:
        for rid in (e.before, e.after):
            if rid not in seen:
                errs.append(f"order edge references unknown rid {rid}")
    if len(g.requirements) > 25:
        errs.append(f"{len(g.requirements)} requirements is implausibly many; "
                    "the compiler is probably splitting one claim into fragments")
    return errs
