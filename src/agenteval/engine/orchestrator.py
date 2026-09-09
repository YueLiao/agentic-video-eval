"""End-to-end evaluation of one clip.

Routes, runs the selected skills against a shared evidence bus, falsifies every
finding, and synthesizes the scores. The whole trajectory is recorded, because
the output of this system is meant to be arguable: a reader should be able to
follow any deducted point back to the probe that produced it and the counter
evidence it survived.

Two ordering choices matter.

Skills share one bus, so evidence computed for one is free for the next --
motion curves cost the same whether one skill or three want them. That makes
enabling an extra dimension much cheaper than its budget suggests.

Falsification runs after *all* skills have reported, not inside each one. A
finding is easier to overturn once its neighbours are known, and a judge that
has just argued for a defect is the worst possible reviewer of it.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from agenteval.consensus.vote import consensus
from agenteval.engine.evidence import EvidenceBus
from agenteval.engine.loop import LoopBudget, Step, falsify, run_skill
from agenteval.llm.client import VLMClient
from agenteval.media.clip import VideoHandle
from agenteval.planning.schema import RequirementGraph
from agenteval.prompting.builder import (PromptBuilder, Slot, briefing_from,
                                         scope_fragment)
from agenteval.router.router import RouteDecision, route
from agenteval.rubrics.taxonomy import rubric_for
from agenteval.scoring.synthesis import VideoScore, synthesize
from agenteval.skills.base import Skill, SkillContext, SkillVerdict


@dataclass
class EvalResult:
    video: str
    condition: dict[str, Any]
    score: VideoScore
    verdicts: list[SkillVerdict] = field(default_factory=list)
    routing: dict[str, Any] = field(default_factory=dict)
    steps: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    prompt_manifests: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    consensus: dict[str, Any] = field(default_factory=dict)
    bus: dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0
    vlm_calls: int = 0
    tool_calls: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "video": self.video, "condition": self.condition,
            "score": self.score.to_json(),
            "verdicts": [v.to_json() for v in self.verdicts],
            "routing": self.routing, "steps": self.steps,
            "consensus": self.consensus,
            "prompt_manifests": self.prompt_manifests, "bus": self.bus,
            "elapsed_s": round(self.elapsed_s, 2),
            "vlm_calls": self.vlm_calls, "tool_calls": self.tool_calls,
        }


def _dynamic_prompt(skill: Skill, decision: RouteDecision,
                    active: Sequence[str], bus: EvidenceBus,
                    condition: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Assemble this skill's system prompt for this specific clip."""
    pb = PromptBuilder()
    pb.add(Slot.BASE, skill.system_prompt, source=f"skill:{skill.name}")
    # The defect taxonomy for this skill's own dimension. Without it the judge
    # invents its own labels and its own severity scale, and findings stop being
    # countable or comparable across clips.
    pb.add(Slot.RUBRIC, rubric_for(skill.name), key=f"rubric:{skill.name}",
           source=f"taxonomy:{skill.name}")
    pb.add(Slot.SCOPE, scope_fragment(skill.name, active), source="router:active_skills")
    cond = condition.get("prompt") or condition.get("prompt_zh") or ""
    if cond:
        pb.add(Slot.CONDITION, cond, source="condition")
    pb.extend(decision.fragments)
    brief = briefing_from([e.to_prompt_block() for e in bus.admissible()])
    pb.add(Slot.BRIEFING, brief, source="evidence_bus")
    rel = [e.result.reliability for e in bus.admissible()]
    from agenteval.prompting.builder import TOOL_TRUST_HIGH, TOOL_TRUST_LOW
    trust = TOOL_TRUST_HIGH if (rel and sum(rel) / len(rel) >= 0.6) else TOOL_TRUST_LOW
    pb.add(Slot.TOOL_TRUST, trust, source="measured_reliability")
    return pb.build(), pb.manifest()


def evaluate(video_path: str | Path, condition: dict[str, Any],
             skills: dict[str, Callable[[], Skill]], vlm: VLMClient,
             *, out_dir: str | Path, budget: LoopBudget | None = None,
             do_falsify: bool = True,
             decision: RouteDecision | None = None,
             graph: "RequirementGraph | None" = None,
             phases: Sequence[float] = (0.0,),
             consensus_fraction: float = 0.5) -> EvalResult:
    """Evaluate one clip, optionally across several sampling phases.

    With more than one phase the whole skill pass is repeated with uniform
    sampling shifted, and only findings that recur across a majority of phases
    are kept. That is not redundancy: measured on three phases of the same clips,
    shifting which frames get sampled changed every aspect carrying signal, with
    noise exceeding between-model signal on all of them, and the overall ranking
    took three different orders in three runs. A defect that appears at one phase
    and not the others was a property of the sampling, not of the video.
    """
    if len(phases) > 1:
        return _evaluate_multiphase(
            video_path, condition, skills, vlm, out_dir=out_dir, budget=budget,
            do_falsify=do_falsify, decision=decision, graph=graph,
            phases=phases, consensus_fraction=consensus_fraction)
    return _evaluate_once(video_path, condition, skills, vlm, out_dir=out_dir,
                          budget=budget, do_falsify=do_falsify,
                          decision=decision, graph=graph)


def _evaluate_once(video_path: str | Path, condition: dict[str, Any],
                   skills: dict[str, Callable[[], Skill]], vlm: VLMClient,
                   *, out_dir: str | Path, budget: LoopBudget | None = None,
                   do_falsify: bool = True,
                   decision: RouteDecision | None = None,
                   graph: "RequirementGraph | None" = None) -> EvalResult:
    t0 = time.perf_counter()
    video = VideoHandle(video_path)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    d = decision or route(video, condition, available=list(skills), graph=graph)
    bus = EvidenceBus(video.hash)
    base = budget or LoopBudget()

    verdicts: list[SkillVerdict] = []
    covers_map: dict[str, tuple[str, ...]] = {}
    steps: dict[str, list[dict[str, Any]]] = {}
    manifests: dict[str, list[dict[str, Any]]] = {}
    contexts: dict[str, SkillContext] = {}

    for name in d.skills:
        skill = skills[name]()
        ctx = SkillContext(video=video, bus=bus, condition=condition, hints=d.hints)
        contexts[name] = ctx
        if not skill.applies(ctx):
            d.disabled[name] = "skill reported not applicable for this clip"
            continue
        scale = d.budget_scale.get(name, 1.0)
        b = LoopBudget(max_rounds=max(1, int(skill.max_rounds * scale)),
                       max_vlm_calls=max(2, int(base.max_vlm_calls * scale)),
                       max_tool_calls=max(3, int(base.max_tool_calls * scale)),
                       max_wall_s=base.max_wall_s * scale)
        # The prompt is assembled per skill *and* per clip, after seeding, so
        # the briefing reflects what was actually measured on this video.
        prompt, manifest = _dynamic_prompt(skill, d, d.skills, bus, condition)
        manifests[name] = manifest
        object.__setattr__(skill, "_dynamic_prompt", prompt)
        covers_map[skill.name] = tuple(skill.covers)
        v, st = run_skill(skill, ctx, vlm, b)
        verdicts.append(v)
        steps[name] = [s.to_json() for s in st]

    if do_falsify:
        for v in verdicts:
            if v.findings:
                falsify(v, contexts[v.skill], vlm)

    from agenteval.scoring.synthesis import collect_measurements
    meas = collect_measurements(bus.snapshot(), d.to_json())
    # Semantic aspects gate on the *condition* having asked for that kind of
    # thing: a condition with no text requirement cannot fail text rendering,
    # and scoring it would be meaningless rather than merely uninformative.
    if graph is not None:
        for r in graph.requirements:
            meas[f"n_req_{r.kind}"] = meas.get(f"n_req_{r.kind}", 0) + 1
        meas["n_req_order"] = len(graph.order)
    score = synthesize(verdicts, total_frames=video.total, disabled=d.disabled,
                       measurements=meas, skill_covers=covers_map)
    res = EvalResult(
        video=str(video.path), condition=condition, score=score,
        verdicts=verdicts, routing=d.to_json(), steps=steps,
        prompt_manifests=manifests, bus=bus.snapshot(),
        elapsed_s=time.perf_counter() - t0,
        vlm_calls=sum(v.vlm_calls for v in verdicts),
        tool_calls=bus.calls,
    )
    (out / "result.json").write_text(
        json.dumps(res.to_json(), ensure_ascii=False, indent=1), encoding="utf-8")
    return res


def _evaluate_multiphase(video_path, condition, skills, vlm, *, out_dir,
                         budget, do_falsify, decision, graph,
                         phases: Sequence[float],
                         consensus_fraction: float) -> EvalResult:
    """Run every phase, then keep only findings that recur across them.

    Cost is evidence collection, not judgement quality: each phase is a full
    pass, so this multiplies VLM calls by the number of phases. Worth it because
    the alternative measured worse than noise -- more clips cannot fix a score
    that changes when you shift the sampling by three frames.
    """
    from agenteval.media.clip import set_sample_phase
    from agenteval.scoring.synthesis import collect_measurements, synthesize
    from agenteval.skills.base import SkillVerdict

    out = Path(out_dir)
    per_phase: list[EvalResult] = []
    try:
        for i, ph in enumerate(phases):
            set_sample_phase(ph)
            per_phase.append(_evaluate_once(
                video_path, condition, skills, vlm, out_dir=out / f"phase{i}",
                budget=budget, do_falsify=do_falsify, decision=decision,
                graph=graph))
    finally:
        set_sample_phase(0.0)

    base = per_phase[0]
    by_skill: dict[str, list[list]] = {}
    for r in per_phase:
        seen = {v.skill for v in r.verdicts}
        for v in r.verdicts:
            by_skill.setdefault(v.skill, []).append(
                [f for f in v.findings if f.counts])
        for name in by_skill:
            if name not in seen:            # skill did not run this phase
                by_skill[name].append([])

    merged: list[SkillVerdict] = []
    report: dict[str, Any] = {"phases": list(phases), "per_skill": {}}
    for name, phase_findings in by_skill.items():
        kept, rep = consensus(phase_findings, min_fraction=consensus_fraction)
        report["per_skill"][name] = rep
        src = next((v for r in per_phase for v in r.verdicts if v.skill == name), None)
        mv = SkillVerdict(skill=name, findings=kept,
                          summary=(src.summary if src else ""),
                          rounds=sum(v.rounds for r in per_phase
                                     for v in r.verdicts if v.skill == name),
                          vlm_calls=sum(v.vlm_calls for r in per_phase
                                        for v in r.verdicts if v.skill == name),
                          tool_calls=sum(v.tool_calls for r in per_phase
                                         for v in r.verdicts if v.skill == name))
        merged.append(mv)

    tot = sum(r["n_clusters"] for r in report["per_skill"].values())
    dropped = sum(r["n_dropped"] for r in report["per_skill"].values())
    report["n_clusters"] = tot
    report["n_dropped"] = dropped
    report["discard_rate"] = round(dropped / max(1, tot), 3)

    video = VideoHandle(video_path)
    meas = collect_measurements(base.bus, base.routing)
    covers = {name: tuple(skills[name]().covers) for name in base.routing["skills"]
              if name in skills}
    score = synthesize(merged, total_frames=video.total,
                       disabled=base.routing.get("disabled") or {},
                       measurements=meas, skill_covers=covers)
    res = EvalResult(
        video=str(video.path), condition=condition, score=score,
        verdicts=merged, routing=base.routing,
        steps={f"phase{i}": r.steps for i, r in enumerate(per_phase)},  # type: ignore[misc]
        prompt_manifests=base.prompt_manifests, bus=base.bus,
        consensus=report,
        elapsed_s=sum(r.elapsed_s for r in per_phase),
        vlm_calls=sum(r.vlm_calls for r in per_phase),
        tool_calls=sum(r.tool_calls for r in per_phase),
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(
        json.dumps(res.to_json(), ensure_ascii=False, indent=1), encoding="utf-8")
    return res
