"""Turning findings into scores, deterministically.

The judge is never asked for a number. Absolute Likert ratings from a VLM are
poorly calibrated and, worse, not comparable across conditions: an 8 on a busy
crowd scene and an 8 on a static macro shot mean nothing together. So scores are
computed from objects that *can* be compared -- confirmed, localized,
severity-rated findings -- by a rule that is fixed and inspectable.

Three consequences, all intended:

* every deducted point resolves to a ``(frame span, box, type)`` a human can
  re-examine, so a disputed score is a factual question rather than an argument
  about taste;
* re-running with the same findings yields the same score, whatever mood the
  judge was in;
* the score is hard to inflate, because raising it means removing findings that
  survived falsification, not talking the judge round.

Severity is weighted by *coverage* as well as by grade. A single-frame blemish
in a corner and a defect running the whole clip across half the frame are not
the same event, and a rule that ignores extent lets either dominate the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from agenteval.skills.base import Finding, SkillVerdict

SEVERITY_WEIGHT: dict[str, float] = {"minor": 0.10, "major": 0.35, "critical": 0.85}

#: A finding this bad caps its dimension however clean everything else is. A
#: video with a critical structural failure is not a good video that happens to
#: have one problem, and a plain weighted mean would let volume of good outweigh
#: it.
SEVERITY_CAP: dict[str, float] = {"critical": 3.0, "major": 6.5}

#: Integrity dimensions start clean and lose points; conformance dimensions are
#: satisfaction ratios and start empty. They must not be aggregated the same way.
INTEGRITY = {"temporal_integrity", "motion_quality", "human_integrity",
             "physical_integrity", "appearance_integrity"}
CONFORMANCE = {"semantic_conformance", "action_conformance",
               "camera_conformance", "style_conformance"}

DEFAULT_WEIGHTS: dict[str, float] = {
    "semantic_conformance": 1.3, "action_conformance": 1.2,
    "camera_conformance": 0.8, "style_conformance": 0.6,
    "temporal_integrity": 1.0, "motion_quality": 1.2,
    "human_integrity": 1.2, "physical_integrity": 1.0,
    "appearance_integrity": 0.8,
}


@dataclass
class DimensionScore:
    dimension: str
    score: float                      # 0..10
    n_findings: int = 0
    n_retracted: int = 0
    capped_by: str | None = None
    applicable: bool = True
    reason: str = ""
    findings: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {"dimension": self.dimension, "score": round(self.score, 2),
                "n_findings": self.n_findings, "n_retracted": self.n_retracted,
                "capped_by": self.capped_by, "applicable": self.applicable,
                "reason": self.reason, "findings": self.findings}


@dataclass
class VideoScore:
    overall: float
    dimensions: dict[str, DimensionScore]
    n_findings: int = 0
    n_retracted: int = 0
    retraction_rate: float = 0.0
    caps_fired: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "overall": round(self.overall, 2),
            "dimensions": {k: v.to_json() for k, v in self.dimensions.items()},
            "n_findings": self.n_findings, "n_retracted": self.n_retracted,
            "retraction_rate": round(self.retraction_rate, 3),
            "caps_fired": self.caps_fired,
        }


def coverage(f: Finding, total_frames: int) -> float:
    """Fraction of the clip-volume a finding occupies, in [0.05, 1].

    Floored rather than allowed to reach zero: a defect localized to one frame
    is still a real defect, and multiplying its weight by ~0 would make precise
    localization *reduce* the penalty -- rewarding vagueness.
    """
    t = 1.0
    if f.t_span and total_frames > 0:
        t = max(0.0, min(1.0, (f.t_span[1] - f.t_span[0]) / total_frames))
    a = 1.0
    if f.bbox:
        a = max(0.0, min(1.0, float(f.bbox[2]) * float(f.bbox[3])))
        a = a ** 0.5          # area under-weights wide thin defects; soften it
    return max(0.05, min(1.0, (0.5 + 0.5 * t) * (0.4 + 0.6 * a)))


def score_integrity(dimension: str, findings: Sequence[Finding],
                    total_frames: int) -> DimensionScore:
    live = [f for f in findings if f.counts]
    penalty = 0.0
    worst_cap, cap_by = None, None
    for f in live:
        w = SEVERITY_WEIGHT.get(f.severity, 0.10)
        penalty += w * coverage(f, total_frames) * max(0.3, f.confidence)
        cap = SEVERITY_CAP.get(f.severity)
        if cap is not None and (worst_cap is None or cap < worst_cap):
            worst_cap, cap_by = cap, f.kind
    score = 10.0 * max(0.0, 1.0 - min(1.0, penalty))
    if worst_cap is not None and score > worst_cap:
        score = worst_cap
    else:
        cap_by = None
    return DimensionScore(
        dimension=dimension, score=score, n_findings=len(live),
        n_retracted=len(findings) - len(live), capped_by=cap_by,
        findings=[f.to_json() for f in findings],
    )


def score_conformance(dimension: str, findings: Sequence[Finding],
                      n_requirements: int, total_frames: int) -> DimensionScore:
    """Requirement satisfaction. Findings here are unmet requirements, so the
    score is the satisfied fraction rather than a deduction from clean."""
    live = [f for f in findings if f.counts]
    if n_requirements <= 0:
        return DimensionScore(dimension, 10.0, applicable=False,
                              reason="no requirements extracted from the condition")
    lost = 0.0
    for f in live:
        lost += {"critical": 1.0, "major": 0.6, "minor": 0.25}.get(f.severity, 0.25)
    score = 10.0 * max(0.0, 1.0 - min(1.0, lost / n_requirements))
    return DimensionScore(dimension, score, n_findings=len(live),
                          n_retracted=len(findings) - len(live),
                          findings=[f.to_json() for f in findings])


def synthesize(verdicts: Iterable[SkillVerdict], *, total_frames: int,
               disabled: dict[str, str] | None = None,
               n_requirements: dict[str, int] | None = None,
               weights: dict[str, float] | None = None,
               alpha: float = 0.5) -> VideoScore:
    """Combine per-skill verdicts into dimension scores and one overall.

    The overall blends the weighted mean with the worst dimension
    (``alpha`` toward the minimum). A pure mean lets six clean dimensions bury
    one catastrophic failure, which is exactly the case a viewer would call the
    video broken; taking the raw minimum instead throws away everything else.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    reqs = n_requirements or {}
    dims: dict[str, DimensionScore] = {}
    total_f = total_r = 0

    for v in verdicts:
        dim = v.skill
        if v.error:
            dims[dim] = DimensionScore(dim, 0.0, applicable=False,
                                       reason=f"skill error: {v.error}")
            continue
        if dim in CONFORMANCE:
            ds = score_conformance(dim, v.findings, reqs.get(dim, 0), total_frames)
        else:
            ds = score_integrity(dim, v.findings, total_frames)
        dims[dim] = ds
        total_f += ds.n_findings
        total_r += ds.n_retracted

    for dim, why in (disabled or {}).items():
        dims.setdefault(dim, DimensionScore(dim, 0.0, applicable=False, reason=why))

    scored = [(d, w.get(name, 1.0)) for name, d in dims.items() if d.applicable]
    if not scored:
        return VideoScore(0.0, dims, total_f, total_r, 0.0)
    mean = sum(d.score * wt for d, wt in scored) / sum(wt for _, wt in scored)
    worst = min(d.score for d, _ in scored)
    overall = (1 - alpha) * mean + alpha * min(mean, worst)
    rate = total_r / max(1, total_f + total_r)
    caps = [f"{d.dimension}:{d.capped_by}" for d, _ in scored if d.capped_by]
    return VideoScore(overall, dims, total_f, total_r, rate, caps)
