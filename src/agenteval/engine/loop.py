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
from agenteval.rubrics.taxonomy import normalize_grade
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


#: Restated immediately before the output block. The same four anchors live in
#: JUDGE_RULES, but the first live run returned exactly 0.5 on all 14 findings
#: with them there: as rule 7 of nine, 2500 characters up a prompt that also
#: carries a rubric, a scope note and an evidence briefing, they were simply not
#: read. An instruction has to sit where the model is when it writes the field.
CONFIDENCE_CONTRACT = """\
**confidence 只能是 0.9 / 0.7 / 0.4 / 0.2 四个值之一,禁止填 0.5:**
- `0.9` 在放大证据里直接看清了,能说出是哪一帧、哪个部位、什么形态
- `0.7` 看得到异常,但受分辨率/遮挡/运动影响,细节不完全确定
- `0.4` 只是可疑,现有证据不足以确认
- `0.2` 基本看不清,只是因为工具数值异常才提出
"""

FALSIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["verdict", "reason"],
    "properties": {
        "verdict": {"enum": ["uphold", "retract", "insufficient"]},
        "alternative": {"type": "string"},
        "evidence_for_alternative": {"type": "string"},
        "reason": {"type": "string"},
        "severity": {"enum": ["trace", "minor", "major", "severe"]},
    },
}

FALSIFY_SYSTEM = """\
你在做**反证审查**:判断一条缺陷指控是否站得住。

## 你的任务不是"能不能想出别的解释"

对任何视觉现象,总能构造出一个听起来合理的无罪解释——运动模糊、遮挡、材质、光照。
**仅仅提出一个替代解释不构成撤回理由。**
你必须进一步**验证这个替代解释是否与证据一致**:

- 说是**运动模糊**导致的?那么在运动慢的帧、或运动停止后的帧上,该部位应当是清晰、
  结构正确的。**去证据里找那样的帧。找不到,或那些帧上问题依然存在,则解释不成立。**
- 说是**遮挡/材质覆盖**(泥浆、水、衣物)导致的?那么被覆盖的轮廓应当仍然连贯,
  且覆盖物本身应当有一致的外观。**如果结构在覆盖下仍然违反解剖(手指数量变了、
  出现违反关节的突起),覆盖解释不成立。**
- 说是**正常物理现象**?那么它应当在时间上连续且符合该材料的行为。
  凭空出现或消失、或形态突变,物理解释不成立。
- 说是**透视/角度**造成的?那么变化应当随视角平滑变化,不应在相邻帧间跳变。

## 判定

- **uphold**:指控描述的现象确实存在,且你**验证过的**替代解释都与证据矛盾。
- **retract**:你提出了替代解释,**并且在证据中找到了支持它的具体依据**
  (指出是哪一帧、哪个区域)。
- **insufficient**:证据不足以判断——画面看不清、缺少必要的对照帧。
  这**不等于** retract:它意味着这条指控既没被证实也没被推翻。

严重度可以下调而不必整条撤回:如果现象存在但比指控说的轻微,用 uphold + 更低的 severity。

只输出 JSON:
{"verdict":"uphold"|"retract"|"insufficient",
 "alternative":"<你考虑的替代解释,没有则空>",
 "evidence_for_alternative":"<支持该替代解释的具体证据:哪一帧、哪个区域。没有则空>",
 "reason":"...",
 "severity":"minor"|"major"|"critical"}
"""


#: Judges return confidence as words as often as numbers. Coercing with a bare
#: float() raises, and the enclosing except then drops the entire finding --
#: losing a real defect because of a formatting choice. Parse leniently instead.
_CONF_WORDS: dict[str, float] = {
    "high": 0.9, "very high": 0.95, "certain": 0.95, "confident": 0.9,
    "medium": 0.6, "moderate": 0.6, "mid": 0.5,
    "low": 0.3, "very low": 0.2, "uncertain": 0.3, "unsure": 0.3,
    "高": 0.9, "较高": 0.8, "中": 0.6, "中等": 0.6, "低": 0.3, "较低": 0.35,
}


def parse_confidence(v: Any, default: float = 0.5) -> float:
    if isinstance(v, bool):
        return default
    if isinstance(v, (int, float)):
        return max(0.0, min(1.0, float(v)))
    if isinstance(v, str):
        s = v.strip().lower().rstrip("%")
        try:
            f = float(s)
            return max(0.0, min(1.0, f / 100 if f > 1 else f))
        except ValueError:
            pass
        # longest key first: "very low" must win over the "low" inside it
        for k in sorted(_CONF_WORDS, key=len, reverse=True):
            if k in s:
                return _CONF_WORDS[k]
    return default


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
              "没有问题就返回空的 findings 数组。\n\n"
            + CONFIDENCE_CONTRACT
            + "\n只输出 JSON:\n"
              '{"summary": "...", "findings": [{"kind": "...", '
              '"severity": "trace|minor|major|severe", '
              '"extent": "flash|brief|recurring|throughout", '
              '"salience": "peripheral|secondary|primary", '
              '"confidence": 0.9|0.7|0.4|0.2, '
              '"t_span": [起,止], "bbox": [x,y,w,h], "rationale": "...", '
              '"evidence": ["E01"]}]}')
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
                extent=str(f.get("extent", "brief")),
                salience=str(f.get("salience", "secondary")),
                t_span=tuple(f["t_span"][:2]) if f.get("t_span") else None,
                bbox=tuple(f["bbox"][:4]) if f.get("bbox") else None,
                confidence=parse_confidence(f.get("confidence")),
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
        v = str(p.get("verdict", ""))
        alt = str(p.get("alternative", "")).strip()
        ev = str(p.get("evidence_for_alternative", "")).strip()
        if v == "retract":
            # A retraction has to name evidence for its alternative. Without
            # that requirement the pass degenerates into explaining everything
            # away, since a plausible-sounding alternative always exists.
            if alt and not ev:
                f.retraction_reason = ("反证提出了替代解释但未给出支持它的证据,"
                                       f"不予撤回:{alt[:160]}")
            else:
                f.retracted = True
                f.retraction_reason = (str(p.get("reason", ""))[:300]
                                       + (f"  [依据:{ev[:120]}]" if ev else ""))
        elif v == "insufficient":
            # Neither confirmed nor overturned: keep it, but weakened, so it
            # cannot dominate a score on evidence nobody could read.
            f.confidence = min(f.confidence, 0.3)
            # A finding nobody could verify should not also carry the top grade.
            f.severity = {"severe": "major", "major": "minor"}.get(
                normalize_grade(f.severity), normalize_grade(f.severity))
            f.retraction_reason = f"证据不足,降权保留:{str(p.get('reason',''))[:200]}"
        elif p.get("severity"):
            f.severity = str(p["severity"])
    return verdict
