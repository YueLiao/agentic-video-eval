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

from agenteval.rubrics.taxonomy import (EXTENT, GRADE_SCORE, SALIENCE,
                                        normalize_extent, normalize_grade,
                                        normalize_salience)
from agenteval.scoring.aspects import (ASPECTS, BY_GROUP, DEFECT_TO_ASPECT,
                                       GROUP_LABEL_ZH, NOT_SCORED, Aspect)
from agenteval.skills.base import Finding, SkillVerdict

#: How much an additional finding of this grade erodes the score below the
#: leading one's anchor.
SEVERITY_WEIGHT: dict[str, float] = {
    "trace": 0.04, "minor": 0.12, "major": 0.30, "severe": 0.60,
}

#: The ceiling a finding of this grade imposes: however clean everything else
#: is, one critical structural failure means the aspect is not above 3.0.
#:
#: It is a **ceiling, not a clamp**. Applying it as `min(score, cap)` was a bug
#: that collapsed the entire scale: across 420 scored aspects only three values
#: ever appeared -- 10.00, 6.50, 3.00 -- because one major finding pinned the
#: result to exactly 6.5 and further findings changed nothing. All the work in
#: the penalty term (severity x coverage x confidence, accumulating over
#: findings) was discarded at the last step, so a video with one small
#: blemish and one with six large ones scored identically.
#:
#: Now the ceiling sets where the aspect *starts* once that grade is present,
#: and the accumulated penalty continues to push it down from there.
SEVERITY_CEILING: dict[str, float] = dict(GRADE_SCORE)

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


#: Retraction rates outside this band mean the falsification pass is not
#: filtering. At the top, every accusation is being explained away and the
#: report is a silent pass; at the bottom, nothing is being challenged and the
#: pass is decorative. Either way the scores are not trustworthy, and a run that
#: looks clean is more dangerous than one that crashes.
RETRACTION_OK = (0.05, 0.75)


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
    warnings: list[str] = field(default_factory=list)

    @property
    def trustworthy(self) -> bool:
        return not self.warnings

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
            "warnings": self.warnings,
            "trustworthy": self.trustworthy,
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
        for w in self.warnings:
            rows.append(f"\n  ⚠ 本次结果不可信: {w}")
        return "\n".join(rows)


def impact(f: Finding) -> float:
    """How much this finding actually matters, from the two judged axes.

    Replaces deriving extent from ``t_span``/``bbox``. Those numbers looked
    precise and were not: across sampling phases the judge reported the same
    defect as @0-5, @1-9 and @1-10, so a formula reading them was mostly reading
    sampling phase. "A flash" versus "most of the clip" is the same information
    stated at a resolution the judge can actually hold.
    """
    return (EXTENT.get(normalize_extent(f.extent), 0.6)
            * SALIENCE.get(normalize_salience(f.salience), 0.75))


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


def _score_from_findings(live: Sequence[Finding],
                         total_frames: int) -> tuple[float, str | None]:
    """Accumulated penalty, then lowered to the worst grade's ceiling.

    Both terms matter and each fixes the other's failure. The penalty alone
    lets many small findings out-weigh one disqualifying failure; the ceiling
    alone throws away every distinction between one problem and six. So the
    ceiling establishes the starting point for that grade and the remaining
    findings keep pushing down from it -- which is what gives the scale values
    other than the three it had.
    """
    if not live:
        return 10.0, None
    order = {"severe": 0, "major": 1, "minor": 2, "trace": 3}
    # Index rather than the object: identity comparison would skip every finding
    # that happens to be the same object, and more importantly it makes the
    # "exclude the leading finding" rule depend on object identity rather than
    # on position, which is not a property real data guarantees.
    lead_i = min(range(len(live)),
                 key=lambda i: order.get(normalize_grade(live[i].severity), 4))
    worst = live[lead_i]
    ceiling = SEVERITY_CEILING.get(normalize_grade(worst.severity), 6.0)

    # How far the leading finding actually pulls the score down to its ceiling.
    # Confidence and extent belong here rather than only in the residual: a
    # marginal, briefly-visible major is not the same event as one that persists
    # across the clip, and collapsing both onto the ceiling was what left the
    # scale with three distinct values.
    lead_strength = (max(0.3, min(1.0, worst.confidence))
                     * (0.4 + 0.6 * impact(worst)))
    score = 10.0 - (10.0 - ceiling) * lead_strength

    # Everything after the leading finding accumulates from there, so quantity
    # still separates one blemish from six.
    residual = 0.0
    for i, f in enumerate(live):
        if i == lead_i:
            continue
        residual += (SEVERITY_WEIGHT.get(normalize_grade(f.severity), 0.12)
                     * impact(f) * max(0.3, f.confidence))
    score *= max(0.0, 1.0 - min(1.0, residual))
    cap_by = worst.kind if ceiling < 10.0 else None
    return round(score, 3), cap_by


def score_integrity(dimension: str, findings: Sequence[Finding],
                    total_frames: int) -> DimensionScore:
    live = [f for f in findings if f.counts]
    score, cap_by = _score_from_findings(live, total_frames)
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
        overall = overall_from_aspects(judged, alpha=alpha)
    # A clean-looking report from a broken pipeline is the worst output this
    # system can produce, so the conditions that make one are stated loudly.
    warnings: list[str] = []
    raised = total_f + total_r
    if raised >= 4 and rate > RETRACTION_OK[1]:
        warnings.append(
            f"撤回率 {rate:.0%}({total_r}/{raised}):反证环节在否定一切而非筛选,"
            "满分很可能是静默失败而不是真的没问题")
    elif raised >= 4 and rate < RETRACTION_OK[0]:
        warnings.append(
            f"撤回率 {rate:.0%}:几乎没有指控被挑战,反证环节形同虚设")
    n_judgeable = sum(1 for a in asp.values() if a.judgeable)
    if n_judgeable and total_f == 0 and raised == 0:
        warnings.append("没有任何 skill 提出过 finding:请确认判官确实在检查,"
                        "而不是每轮都直接 conclude")
    return VideoScore(overall, dims, asp, group_rollup(asp), dict(NOT_SCORED),
                      total_f, total_r, rate, caps, warnings)


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
            key = f.aspect or DEFECT_TO_ASPECT.get(f.kind)
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
        order = {"severe": 0, "major": 1, "minor": 2, "trace": 3}
        worst = (normalize_grade(min(live, key=lambda f: order.get(
            normalize_grade(f.severity), 4)).severity) if live else None)
        score, _ = _score_from_findings(live, total_frames)
        out[a.key] = AspectScore(
            a.key, a.label_zh, a.group, score, judgeability=a.judgeability,
            n_findings=len(live), n_retracted=len(fs) - len(live),
            worst_severity=worst, actionable=a.actionable if live else "",
            findings=[f.to_json() for f in fs])
    return out


#: How many of the worst aspects the headline is built from. A clip is judged
#: by what is wrong with it, not by the count of things that happen to be fine.
DEFECT_FOCUS_K = 5


def overall_from_aspects(judged: Sequence[AspectScore], *, alpha: float = 0.5,
                         k: int = DEFECT_FOCUS_K) -> float:
    """Headline score, built from the worst aspects rather than all of them.

    Averaging every judged aspect compresses everything into the top of the
    scale and destroys the differences that matter. Measured on three clips:
    83-92% of aspects scored a clean 10, which anchored the mean at 9.4-9.7, so
    a model with twice as many defects as another (four versus two) finished
    0.16 below it. Adding aspects made this worse, not better -- each clean one
    is another vote for "fine".

    The `min(mean, worst)` term was inert for the same reason: with the scale
    collapsed, all three clips had an identical worst aspect of 6.50, so the
    term contributed the same value to each and separated nothing.

    So the headline is the mean of the *k worst* judged aspects, blended toward
    the single worst. Clean aspects still matter -- they are what a video needs
    to have in order for its worst to be its only problem -- but they no longer
    outvote the defects by sheer count.
    """
    if not judged:
        return 0.0
    w_by = {"high": 1.0, "medium": 0.7, "low": 0.4}
    ranked = sorted(judged, key=lambda a: a.score or 10.0)
    focus = ranked[:max(1, min(k, len(ranked)))]
    ws = [w_by.get(a.judgeability, 0.7) for a in focus]
    mean_focus = sum((a.score or 10.0) * w for a, w in zip(focus, ws)) / sum(ws)
    worst = ranked[0].score or 10.0
    return (1 - alpha) * mean_focus + alpha * worst


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
        tops.sort(key=lambda f: {"severe": 0, "major": 1, "minor": 2}.get(
            normalize_grade(f.get("severity")), 3))
        out[key] = ReportDimension(
            key=key, score=worst[1].score, applicable=True,
            n_findings=sum(d.n_findings for _, d in usable),
            n_retracted=sum(d.n_retracted for _, d in usable),
            capped_by=worst[1].capped_by,
            contributing_skills=[s for s, _ in usable],
            top_findings=tops[:3],
        )
    return out
