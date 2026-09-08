"""The bounded agent loop.

One skill, one clip. The model repeatedly picks its next probe from that
skill's menu, the harness executes it, and the resulting evidence goes back
into the next prompt. It ends when the model concludes, when the budget runs
out, or when it stops learning anything.

Three termination rules, each fixing a specific way these loops go wrong:

``conclude``          the model says it has enough. Cheapest and most common.
``budget``            hard ceiling on rounds / VLM calls / tool calls, so a
                      confused model costs a bounded amount instead of looping.
``no new information``an action that returns evidence already in the bus does
                      not count as progress. Without this, a model that likes
                      one action will burn the whole budget re-running it.

Then a **falsification pass**: every finding is re-presented with counter
evidence and must be defended or retracted. A judge asked to find problems will
find problems; on the integrity side there is no gold answer to catch that, so
this is the only defence. The retraction rate is recorded, not hidden — it
diagnoses the judge and the prompt at the same time.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from agenteval.engine.actions import (CONCLUDE, DECISION_SCHEMA, Action,
                                      decision_instructions)
from agenteval.engine.evidence import Evidence
from agenteval.llm.client import ImageRef, VLMClient
from agenteval.skills.base import (JUDGE_RULES, VERDICT_SCHEMA, Finding,
                                   Presentation, Skill, SkillContext,
                                   SkillVerdict)


@dataclass
class LoopBudget:
    max_rounds: int = 4
    max_vlm_calls: int = 8
    max_tool_calls: int = 20
    max_wall_s: float = 180.0


@dataclass
class Step:
    round: int
    thought: str
    action: str
    args: dict[str, Any]
    evidence_id: str | None
    new_information: bool
    elapsed_ms: float

    def to_json(self) -> dict[str, Any]:
        return self.__dict__


FALSIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["verdict", "reason"],
    "properties": {
        "verdict": {"enum": ["uphold", "retract"]},
        "reason": {"type": "string"},
        "severity": {"enum": ["minor", "major", "critical"]},
    },
}

FALSIFY_SYSTEM = """\
你在做**反证审查**。下面有一条别人提出的缺陷指控,以及相关证据——
包括可疑区域本身、它在相邻时刻的样子、以及一块作为对照的正常区域。

你的任务不是复核这条指控对不对,而是**尽力反驳它**:
- 这个现象能否由正常的运动模糊、正常的遮挡、正常的光照变化解释?
- 对照区域是否也有同样的现象?若有,那这就是这段视频的正常表现,不是缺陷。
- 相邻时刻是否也这样?若是,那它不是一个"事件"。
- 证据是否根本不足以看清?看不清就是看不清,不能算缺陷成立。

只有当你**反驳不掉**时才 uphold。存疑一律 retract。
只输出 JSON: {"verdict": "uphold"|"retract", "reason": "...", "severity": "minor"|"major"|"critical"}
"""


def _images_for(evidence: Sequence[Evidence], cap: int = 12,
                presentation: Presentation = Presentation.COMPOSITE) -> list[ImageRef]:
    """Collect images under the skill's presentation policy.

    Under ORDERED the newest evidence is kept and its internal frame order is
    preserved, because for a rate judgement the recent, densely-sampled window
    is the evidence and truncating its tail destroys exactly the interval being
    judged. Under COMPOSITE, earlier evidence is kept first: those are the
    single-image comparisons the skill asked for, and each is self-contained.
    """
    usable = [e for e in evidence if e.admissible]
    if presentation is Presentation.ORDERED:
        out: list[ImageRef] = []
        for e in reversed(usable):
            imgs = e.images()
            if len(out) + len(imgs) > cap:
                out = imgs[:cap - len(out)] + out if len(out) < cap else out
                break
            out = imgs + out
        return out[:cap]
    out = []
    for e in usable:
        for im in e.images():
            out.append(im)
            if len(out) >= cap:
                return out
    return out


def run_skill(skill: Skill, ctx: SkillContext, vlm: VLMClient,
              budget: LoopBudget | None = None) -> tuple[SkillVerdict, list[Step]]:
    b = budget or LoopBudget(max_rounds=skill.max_rounds)
    t_start = time.perf_counter()
    steps: list[Step] = []
    vlm_calls = tool_calls = 0

    if not skill.applies(ctx):
        return SkillVerdict(skill=skill.name, summary="skill not applicable"), steps

    evidence: list[Evidence] = list(skill.seed(ctx))
    tool_calls += len(evidence)
    actions = {a.name: a for a in skill.actions(ctx)}
    actions[CONCLUDE.name] = CONCLUDE
    menu = list(actions.values())
    history: list[str] = []
    exhausted = False

    for rnd in range(1, b.max_rounds + 1):
        if (vlm_calls >= b.max_vlm_calls or tool_calls >= b.max_tool_calls
                or time.perf_counter() - t_start > b.max_wall_s):
            exhausted = True
            break
        t0 = time.perf_counter()
        user = (skill.render_state(ctx, evidence, history) + "\n\n"
                + decision_instructions(menu))
        resp = vlm.ask(system=skill.prompt + "\n" + JUDGE_RULES, user=user,
                       images=_images_for(evidence, skill.max_images, skill.presentation),
                       schema=DECISION_SCHEMA,
                       tag=f"{skill.name}/decide/{rnd}")
        vlm_calls += 1
        if not resp.ok:
            steps.append(Step(rnd, "", "<error>", {}, None, False,
                              (time.perf_counter() - t0) * 1000))
            break

        d = resp.parsed or {}
        name = str(d.get("action", "")).strip()
        args = d.get("args") or {}
        thought = str(d.get("thought", ""))[:500]
        if name == CONCLUDE.name or name not in actions:
            steps.append(Step(rnd, thought, name or "<missing>", args, None, False,
                              (time.perf_counter() - t0) * 1000))
            break

        act = actions[name]
        before = len(ctx.bus.all())
        try:
            ev = act.run(**args) if act.run else None
        except Exception as e:  # noqa: BLE001 - a bad arg must not kill the run
            history.append(f"{name}({args}) -> 失败: {type(e).__name__}: {e}")
            steps.append(Step(rnd, thought, name, args, None, False,
                              (time.perf_counter() - t0) * 1000))
            continue
        tool_calls += 1
        new = ev is not None and len(ctx.bus.all()) > before
        if ev is not None and ev.eid not in {x.eid for x in evidence}:
            evidence.append(ev)
        history.append(f"{name}({json.dumps(args, ensure_ascii=False)}) -> "
                       f"{ev.eid if ev else 'none'}{'' if new else ' (无新信息)'}")
        steps.append(Step(rnd, thought, name, args, ev.eid if ev else None, new,
                          (time.perf_counter() - t0) * 1000))
        if not new:
            break                      # repeating itself: stop paying for it

    # ---- verdict ---------------------------------------------------------
    user = (skill.render_state(ctx, evidence, history)
            + "\n\n## 现在给出结论\n"
            + "列出你确认的问题(findings)。每条必须带 evidence id、严重度、"
              "以及尽可能精确的 t_span(帧区间)和 bbox(归一化 x,y,w,h)。\n"
              "没有问题就返回空的 findings 数组。\n"
              "只输出 JSON: {\"summary\": \"...\", \"findings\": [...]}")
    vres = vlm.ask(system=skill.prompt + "\n" + JUDGE_RULES, user=user,
                   images=_images_for(evidence, skill.max_images, skill.presentation),
                   schema=VERDICT_SCHEMA,
                   tag=f"{skill.name}/verdict")
    vlm_calls += 1

    verdict = SkillVerdict(skill=skill.name, rounds=len(steps),
                           vlm_calls=vlm_calls, tool_calls=tool_calls,
                           budget_exhausted=exhausted)
    if not vres.ok:
        verdict.error = vres.error or "verdict unparseable"
        return verdict, steps

    payload = vres.parsed or {}
    verdict.summary = str(payload.get("summary", ""))[:1000]
    for f in payload.get("findings", []) or []:
        try:
            verdict.findings.append(Finding(
                kind=str(f.get("kind", "unknown")),
                severity=str(f.get("severity", "minor")),
                t_span=tuple(f["t_span"][:2]) if f.get("t_span") else None,
                bbox=tuple(f["bbox"][:4]) if f.get("bbox") else None,
                confidence=float(f.get("confidence", 0.5)),
                rationale=str(f.get("rationale", ""))[:800],
                evidence=[str(x) for x in (f.get("evidence") or [])],
                aspect=(str(f["aspect"]) if f.get("aspect") else None),
            ))
        except (TypeError, ValueError, KeyError):
            continue
    return verdict, steps


def falsify(verdict: SkillVerdict, ctx: SkillContext, vlm: VLMClient,
            *, counter_evidence: Any = None) -> SkillVerdict:
    """Re-present each finding with counter evidence; retract what cannot be
    defended. Records why, so the retraction rate is auditable."""
    for f in verdict.findings:
        ev = ctx.bus.admissible(f.evidence)
        images = _images_for(ev)
        if counter_evidence is not None:
            extra = counter_evidence(f)
            if extra:
                ev = ev + extra
                images = _images_for(ev)
        body = {
            "指控": f.kind, "严重度": f.severity, "理由": f.rationale,
            "时间": list(f.t_span) if f.t_span else None,
            "区域": [round(v, 3) for v in f.bbox] if f.bbox else None,
        }
        blocks = [e.to_prompt_block() for e in ev]
        user = ("## 待审查的指控\n```json\n"
                + json.dumps(body, ensure_ascii=False, indent=1) + "\n```\n\n"
                "## 证据\n```json\n"
                + json.dumps(blocks, ensure_ascii=False, indent=1) + "\n```\n\n"
                "请尽力反驳这条指控。")
        r = vlm.ask(system=FALSIFY_SYSTEM, user=user, images=images,
                    schema=FALSIFY_SCHEMA, tag=f"{verdict.skill}/falsify")
        verdict.vlm_calls += 1
        if not r.ok:
            continue
        p = r.parsed or {}
        if str(p.get("verdict")) == "retract":
            f.retracted = True
            f.retraction_reason = str(p.get("reason", ""))[:400]
        elif p.get("severity"):
            f.severity = str(p["severity"])
    return verdict
