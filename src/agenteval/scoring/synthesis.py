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

from agenteval.scoring.aspects import (ASPECTS, BY_GROUP, DEFECT_TO_ASPECT,
                                       GROUP_LABEL_ZH, NOT_SCORED, Aspect)
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
    dimensions: dict[str, DimensionScore]        # per skill, the working detail
    aspects: dict[str, "AspectScore"] = field(default_factory=dict)
    groups: dict[str, float | None] = field(default_factory=dict)
    not_scored: dict[str, str] = field(default_factory=lambda: dict(NOT_SCORED))
    n_findings: int = 0
    n_retracted: int = 0
    retraction_rate: float = 0.0
    caps_fired: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "overall": round(self.overall, 2),
            "aspects": {k: v.to_json() for k, v in self.aspects.items()},
            "groups": {k: (None if v is None else round(v, 2))
                       for k, v in self.groups.items()},
            "not_scored": self.not_scored,
            "skill_detail": {k: v.to_json() for k, v in self.dimensions.items()},
            "n_findings": self.n_findings, "n_retracted": self.n_retracted,
            "retraction_rate": round(self.retraction_rate, 3),
            "caps_fired": self.caps_fired,
        }

    def table(self) -> str:
        rows: list[str] = []
        for g, items in BY_GROUP.items():
            gv = self.groups.get(g)
            head = f"{GROUP_LABEL_ZH.get(g, g)}" + (
                f"  [{gv:.2f}]" if gv is not None else "  [n/a]")
            rows.append(f"\n  {head}")
            for it in items:
                a = self.aspects.get(it.key)
                if a is None:
                    continue
                if not a.judgeable:
                    rows.append(f"    {a.label:10s} {'':>6}   —  {a.reason}")
                else:
                    note = ""
                    if a.n_findings:
                        note = f"{a.n_findings} 处 ({a.worst_severity})"
                        if a.actionable:
                            note += f"  → {a.actionable}"
                    rows.append(f"    {a.label:10s} {a.score:6.2f}   {note}")
        rows.append(f"\n  {'总分':10s} {self.overall:6.2f}"
                    f"   (仅计入可判项)")
        return "\n".join(rows)


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
               measurements: dict[str, Any] | None = None,
               skill_covers: dict[str, tuple[str, ...]] | None = None,
               alpha: float = 0.5) -> VideoScore:
    """Combine per-skill verdicts into dimension scores and one overall.

    The overall blends the weighted mean with the worst dimension
    (``alpha`` toward the minimum). A pure mean lets six clean dimensions bury
    one catastrophic failure, which is exactly the case a viewer would call the
    video broken; taking the raw minimum instead throws away everything else.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    reqs = n_requirements or {}
    verdicts_list = list(verdicts)
    verdicts = verdicts_list
    skill_covers = skill_covers or {}
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

    examined: set[str] = set()
    for v in verdicts_list:
        if v.error is None:
            examined.update(skill_covers.get(v.skill, ()))
    asp = score_aspects(list(verdicts_list), total_frames=total_frames,
                        measurements=measurements or {}, examined=examined)
    judged = [a for a in asp.values() if a.judgeable and a.score is not None]
    if judged:
        # Weight by judgeability: a "low" aspect still reports, but must not
        # move the headline as much as one we can actually measure.
        w_by = {"high": 1.0, "medium": 0.7, "low": 0.4}
        ws = [w_by.get(a.judgeability, 0.7) for a in judged]
        mean_a = sum(a.score * w for a, w in zip(judged, ws)) / sum(ws)
        worst_a = min(a.score for a in judged)
        overall = (1 - alpha) * mean_a + alpha * min(mean_a, worst_a)
    return VideoScore(overall, dims, asp, group_rollup(asp), dict(NOT_SCORED),
                      total_f, total_r, rate, caps)


@dataclass
class AspectScore:
    key: str
    label: str
    group: str
    score: float | None                 # None when not judgeable
    judgeable: bool = True
    reason: str = ""                    # why not, when not
    judgeability: str = "medium"
    n_findings: int = 0
    n_retracted: int = 0
    worst_severity: str | None = None
    actionable: str = ""
    findings: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {"key": self.key, "label": self.label, "group": self.group,
                "score": None if self.score is None else round(self.score, 2),
                "judgeable": self.judgeable, "reason": self.reason,
                "judgeability": self.judgeability,
                "n_findings": self.n_findings, "n_retracted": self.n_retracted,
                "worst_severity": self.worst_severity,
                "actionable": self.actionable, "findings": self.findings}


def collect_measurements(bus_snapshot: dict[str, Any],
                         routing: dict[str, Any] | None = None) -> dict[str, Any]:
    """Flatten what was measured on this clip, for the aspect gates.

    Gates need facts, not opinions: how big the largest face was, whether hands
    were found, how much of the clip the sweep actually covered. Those live
    scattered across tool results, so they are lifted into one flat dict here
    rather than each gate knowing the bus layout.
    """
    ev: dict[str, Any] = {}
    if routing:
        ev.update(routing.get("measurements") or {})
        ev.update({k: v for k, v in (routing.get("hints") or {}).items()
                   if isinstance(v, (int, float, bool, str))})
        cues = (routing.get("hints") or {}).get("cues") or {}
        ev["has_text"] = bool(cues.get("text"))
    for _eid, e in (bus_snapshot.get("evidence") or {}).items():
        val = e.get("value") or {}
        for k in ("n_hands_max", "n_frames_with_face", "max_face_frac",
                  "n_persons_max", "perceptual_coverage", "frame_coverage",
                  "survival_rate", "mean_conf"):
            if k in val and isinstance(val[k], (int, float)):
                if k == "mean_conf" and e.get("tool") == "pose_detect":
                    ev["pose_mean_conf"] = val[k]
                elif k == "survival_rate":
                    ev["track_survival"] = max(ev.get("track_survival", 0.0), val[k])
                else:
                    ev[k] = max(ev.get(k, 0), val[k]) if isinstance(val[k], (int, float)) else val[k]
        # largest hand, derived from landmark spans
        for fr in (val.get("per_frame") or []) if e.get("tool") == "hands_detect" else []:
            for hd in fr.get("hands", []):
                xs = [k[0] for k in hd.get("kpts", [])]
                if xs:
                    ev["max_hand_frac"] = max(ev.get("max_hand_frac", 0.0),
                                              max(xs) - min(xs))
    return ev


def score_aspects(verdicts: Sequence[SkillVerdict], *, total_frames: int,
                  measurements: dict[str, Any],
                  examined: set[str] | None = None) -> dict[str, AspectScore]:
    """Score each aspect from the findings assigned to it, subject to its gate.

    Findings route by *defect type*, not by which skill produced them: a hand
    defect belongs to hand structure whether the sweep or the human skill found
    it. That keeps the report stable as skills are added or split.
    """
    by_aspect: dict[str, list[Finding]] = {}
    for v in verdicts:
        for f in v.findings:
            key = DEFECT_TO_ASPECT.get(f.kind)
            if key:
                by_aspect.setdefault(key, []).append(f)

    out: dict[str, AspectScore] = {}
    for a in ASPECTS:
        fs = by_aspect.get(a.key, [])
        live = [f for f in fs if f.counts]
        # "No finding" is only evidence of cleanliness if something looked.
        if examined is not None and a.key not in examined:
            out[a.key] = AspectScore(
                a.key, a.label_zh, a.group, None, judgeable=False,
                reason="未检查(没有 skill 覆盖该项)",
                judgeability=a.judgeability, actionable=a.actionable)
            continue
        ok, why = a.check_gate(measurements)
        if not ok:
            out[a.key] = AspectScore(
                a.key, a.label_zh, a.group, None, judgeable=False, reason=why,
                judgeability=a.judgeability, actionable=a.actionable,
                n_retracted=len(fs) - len(live))
            continue
        penalty = 0.0
        worst = None
        order = {"critical": 0, "major": 1, "minor": 2}
        for f in live:
            penalty += (SEVERITY_WEIGHT.get(f.severity, 0.1)
                        * coverage(f, total_frames) * max(0.3, f.confidence))
            if worst is None or order.get(f.severity, 3) < order.get(worst, 3):
                worst = f.severity
        score = 10.0 * max(0.0, 1.0 - min(1.0, penalty))
        cap = SEVERITY_CAP.get(worst) if worst else None
        if cap is not None:
            score = min(score, cap)
        out[a.key] = AspectScore(
            a.key, a.label_zh, a.group, score, judgeability=a.judgeability,
            n_findings=len(live), n_retracted=len(fs) - len(live),
            worst_severity=worst, actionable=a.actionable if live else "",
            findings=[f.to_json() for f in fs])
    return out


def group_rollup(aspects: dict[str, AspectScore]) -> dict[str, float | None]:
    """Coarse per-group numbers, for leaderboards only.

    Worst-of rather than mean, and ``None`` when no aspect in the group could be
    judged. Provided because a ranking needs few numbers, but the aspect table
    is the primary output: a group score cannot tell you whether to go fix hands
    or fix identity.
    """
    out: dict[str, float | None] = {}
    for g, items in BY_GROUP.items():
        vals = [aspects[i.key].score for i in items
                if i.key in aspects and aspects[i.key].judgeable
                and aspects[i.key].score is not None]
        out[g] = min(vals) if vals else None
    return out
def rollup(skill_scores: dict[str, DimensionScore],
           disabled: dict[str, str]) -> dict[str, ReportDimension]:
    """Fold per-skill results into the six reported dimensions.

    Several skills can feed one dimension. They are combined by taking the
    *worst* contributing score rather than the mean: the dimensions are already
    coarse, and averaging inside one would let a clean sub-aspect mask a broken
    one, which is the failure the six-dimension split exists to prevent.
    """
    buckets: dict[str, list[tuple[str, DimensionScore]]] = {}
    for skill, ds in skill_scores.items():
        key = dimension_of(skill=skill)
        if key:
            buckets.setdefault(key, []).append((skill, ds))

    out: dict[str, ReportDimension] = {}
    for key in REPORT_DIMENSIONS:
        entries = buckets.get(key, [])
        usable = [(s, d) for s, d in entries if d.applicable]
        if not usable:
            why = "; ".join(d.reason or disabled.get(s, "not run")
                            for s, d in entries) or "no skill covers this dimension yet"
            out[key] = ReportDimension(key, 0.0, applicable=False, reason=why,
                                       contributing_skills=[s for s, _ in entries])
            continue
        worst = min(usable, key=lambda sd: sd[1].score)
        tops: list[dict[str, Any]] = []
        for _s, d in usable:
            tops.extend(f for f in d.findings if not f.get("retracted"))
        tops.sort(key=lambda f: {"critical": 0, "major": 1}.get(f.get("severity"), 2))
        out[key] = ReportDimension(
            key=key, score=worst[1].score, applicable=True,
            n_findings=sum(d.n_findings for _, d in usable),
            n_retracted=sum(d.n_retracted for _, d in usable),
            capped_by=worst[1].capped_by,
            contributing_skills=[s for s, _ in usable],
            top_findings=tops[:3],
        )
    return out
