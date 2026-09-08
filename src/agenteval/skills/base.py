"""Skill — one evaluation dimension, with its own way of looking.

A skill owns four things, and the fact that they differ per dimension is the
whole reason the harness is agentic rather than a pipeline:

  system prompt   what this dimension means, what counts as a defect here
  action menu     what this skill is allowed to look at next
  seed evidence   what it starts from before asking the model anything
  verdict schema  what a conclusion for this dimension has to contain

A skill never returns a bare number. It returns findings that point at
evidence, and the score is derived from those findings afterwards, so every
deducted point can be traced to something a human can re-examine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

from enum import Enum

from agenteval.engine.actions import Action
from agenteval.engine.evidence import Evidence, EvidenceBus
from agenteval.media.clip import VideoHandle


class Presentation(str, Enum):
    """How a skill wants its visual evidence encoded.

    Not cosmetic. Compositing several moments into one image makes *comparative*
    judgements easy (is this region different from that one) because the model
    no longer has to hold a visual memory across images. It makes *rate*
    judgements impossible, because a grid discards timing: a stutter and a slow
    passage look identical laid out spatially. Motion, rhythm and continuity
    therefore need the sequence kept, and physical plausibility often needs the
    clip itself. Each skill declares what its question actually requires.
    """

    COMPOSITE = "composite"   # one labelled image; comparative judgements
    ORDERED = "ordered"       # separate images in temporal order; rate/continuity
    VIDEO = "video"           # native video input, where the endpoint accepts it


@dataclass
class Finding:
    """One defect or requirement violation, localized and evidence-backed."""

    kind: str                                   # defect type or requirement id
    severity: str                               # minor | major | critical
    t_span: tuple[int, int] | None = None
    bbox: tuple[float, float, float, float] | None = None
    confidence: float = 0.5
    rationale: str = ""
    evidence: list[str] = field(default_factory=list)
    #: Overrides the defect-key lookup. Conformance findings are unmet
    #: requirements rather than taxonomy defects, so they name their aspect.
    aspect: str | None = None
    retracted: bool = False
    retraction_reason: str = ""

    @property
    def counts(self) -> bool:
        return not self.retracted

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "severity": self.severity,
            "t_span": list(self.t_span) if self.t_span else None,
            "bbox": [round(v, 4) for v in self.bbox] if self.bbox else None,
            "confidence": round(self.confidence, 3), "rationale": self.rationale,
            "evidence": self.evidence, "aspect": self.aspect,
            "retracted": self.retracted,
            "retraction_reason": self.retraction_reason,
        }


@dataclass
class SkillVerdict:
    skill: str
    findings: list[Finding] = field(default_factory=list)
    summary: str = ""
    rounds: int = 0
    vlm_calls: int = 0
    tool_calls: int = 0
    budget_exhausted: bool = False
    error: str | None = None

    @property
    def live(self) -> list[Finding]:
        return [f for f in self.findings if f.counts]

    def to_json(self) -> dict[str, Any]:
        return {
            "skill": self.skill, "summary": self.summary,
            "findings": [f.to_json() for f in self.findings],
            "n_findings": len(self.live), "n_retracted": len(self.findings) - len(self.live),
            "rounds": self.rounds, "vlm_calls": self.vlm_calls,
            "tool_calls": self.tool_calls,
            "budget_exhausted": self.budget_exhausted, "error": self.error,
        }


@dataclass
class SkillContext:
    """Everything a skill gets to work with for one clip."""

    video: VideoHandle
    bus: EvidenceBus
    condition: dict[str, Any] = field(default_factory=dict)   # prompt, camera, ...
    hints: dict[str, Any] = field(default_factory=dict)       # from the router


VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["summary", "findings"],
    "properties": {
        "summary": {"type": "string"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["kind", "severity", "confidence", "rationale", "evidence"],
                "properties": {
                    "kind": {"type": "string"},
                    "severity": {"enum": ["minor", "major", "critical"]},
                    "t_span": {"type": "array", "items": {"type": "integer"}},
                    "bbox": {"type": "array", "items": {"type": "number"}},
                    "confidence": {"type": "number"},
                    "rationale": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
}

# Shared discipline appended to every skill prompt. Kept in one place because
# these rules are what stop the two failure modes that matter: inventing defects
# when asked to find some, and reciting tool numbers instead of looking.
#
# The confidence anchors exist because the first live run returned 0.5 on every
# single finding across three models and eight skills. An unanchored 0..1 float
# invites the midpoint, and a constant confidence makes every downstream use of
# it -- escalation, weighting, triage -- silently inert. Discrete anchors tied
# to what was actually seen give the number something to attach to.
JUDGE_RULES = """\
## 通用规则
1. 只依据给你的证据作答。看不清就说看不清,禁止脑补。
2. 工具给出的数值是**线索**,不是判决。它们告诉你往哪里看,不替你下结论。
3. 若画面所见与工具数值冲突,**以画面为准**,并在 rationale 里写明冲突。
4. rationale 必须描述你在图像里**看到了什么**,不能只复述数值。
5. 每条 finding 必须给出它依据的 evidence id(形如 E01)。给不出就不要报这条。
6. 没有发现问题是完全正常的结论。不要为了交差而编造 finding。
"""


class Skill(ABC):
    name: str
    dimension: str
    #: Aspect keys this skill actually examines. Required, because "no finding"
    #: only means "clean" for aspects something looked at. Without this an
    #: unexamined aspect silently scores 10, which is the same error as
    #: awarding a landscape full marks for subject fidelity.
    covers: tuple[str, ...] = ()
    max_rounds: int = 4
    presentation: Presentation = Presentation.COMPOSITE
    max_images: int = 12

    #: Set by the orchestrator to the per-clip assembled prompt. When present it
    #: replaces the static one, so a skill never has to know it is being routed.
    _dynamic_prompt: str | None = None

    @property
    def prompt(self) -> str:
        return self._dynamic_prompt or self.system_prompt

    @property
    @abstractmethod
    def system_prompt(self) -> str: ...

    @abstractmethod
    def actions(self, ctx: SkillContext) -> list[Action]:
        """The menu for this skill, with `run` bound to this context."""

    def seed(self, ctx: SkillContext) -> list[Evidence]:
        """Evidence gathered before the model is asked anything. Cheap signals
        belong here: they cost nothing and they aim the first question."""
        return []

    def applies(self, ctx: SkillContext) -> bool:
        """Router hook. A skill that cannot apply should say so rather than
        return a meaningless full score."""
        return True

    def render_state(self, ctx: SkillContext, evidence: Sequence[Evidence],
                     history: Sequence[str]) -> str:
        blocks = [e.to_prompt_block() for e in evidence if e.admissible]
        import json as _json
        parts = []
        cond = ctx.condition.get("prompt") or ctx.condition.get("prompt_zh")
        if cond:
            parts.append(f"## 生成条件\n{cond}")
        parts.append(f"## 视频\n{ctx.video.total} 帧, {ctx.video.fps:.1f} fps, "
                     f"{ctx.video.duration_s:.1f}s")
        parts.append("## 已有证据(数值)\n```json\n"
                     + _json.dumps(blocks, ensure_ascii=False, indent=1) + "\n```")
        if history:
            parts.append("## 已经做过的动作\n" + "\n".join(f"- {h}" for h in history))
        return "\n\n".join(parts)
