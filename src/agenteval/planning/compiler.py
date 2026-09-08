"""Compiling a condition into a requirement graph.

One text-only LLM call per condition, run offline and frozen. The economics
matter: the graph is reused across every model, seed and re-run being compared,
so a strong model can be afforded here even when the per-video judge must be
cheap and local. Compile 150 conditions once, evaluate a thousand videos against
them forever.

Freezing is not only about cost. If the graph were regenerated per video, two
models would be measured against subtly different lists and their scores would
not be comparable -- and the difference would be invisible, since both runs look
identical from outside.

The compiler is told to under-produce rather than over-produce. Splitting one
claim into fragments inflates the denominator and makes every video look better
than it is; the validator rejects implausibly long graphs for the same reason.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from agenteval.llm.client import VLMClient
from agenteval.planning.schema import (KIND_TO_ASPECT, OrderEdge, Requirement,
                                       RequirementGraph, validate)

SYSTEM = """\
你是视频生成评测的**条件编译器**。输入是一条视频生成条件(prompt),
输出是一份**可逐条核对**的要求清单。

你**看不到任何视频**,只根据条件本身工作。你的输出会被冻结,用来评测所有候选模型,
所以它必须只反映条件要求了什么,不能包含任何对生成结果的猜测。

## 拆解规则

把条件拆成**彼此独立、各自可判真假**的要求。每条要求属于以下之一:

- `entity`   某个主体/物体必须出现
- `attribute` 某主体必须具备某属性(颜色、材质、状态、穿着)
- `count`    数量要求
- `relation` 主体之间的空间关系
- `action`   某个动作必须发生
- `order`    动作之间的先后或同时关系
- `camera`   运镜要求
- `style`    画面风格/媒介
- `text`     画面中需要出现的文字

每条要求还要指明**怎么验证**(verify):
- `present_any`  至少某一帧可见即可
- `present_most` 大部分时间都应可见
- `continuous`   必须持续存在/持续发生(条件里说"持续""一直""不停"时用)
- `co_occur`     若干对象必须**同时**出现在同一帧
- `sequence`     事件之间的先后关系
- `trajectory`   镜头或物体在时间上的路径

## 重要约束

1. **宁少勿多。** 把一句话拆成过多碎片会虚增分母,让所有视频看起来都更好。
   通常一条 15-40 字的中文条件产出 4-10 条要求。超过 15 条几乎一定是拆碎了。
2. **只写条件明确要求的东西。** 不要补充条件没说的常识性期望。
   条件没提天气,就不要产出"天气应当合理"。
3. **`hard` 字段**:主体缺失、数量错误这类"没做到就是没做到"的设为 true;
   细微属性、次要背景设为 false。
4. **`note` 字段**:写明**什么不算失败**。例如"主体短暂被遮挡不算缺失"、
   "颜色受光照影响的偏差不算错误"。这一条直接决定假阳性的多少,请认真写。
5. 条件里若有运镜描述,必须产出一条 `camera` 要求,并在 `camera` 字段里
   给出 `{"type": "static|push_in|pull_out|pan_left|pan_right|tilt|orbit|follow",
   "speed": "slow|normal|fast"}`。

## 输出格式

只输出 JSON,不要任何多余文字:
{
 "requirements": [
   {"rid":"r1","kind":"entity","text":"画面中出现白色鱼尾狮雕像","verify":"present_most",
    "subject":"鱼尾狮雕像","value":null,"hard":true,"weight":1.0,
    "note":"被游客短暂遮挡不算缺失"},
   ...
 ],
 "order": [{"before":"r3","after":"r4","relation":"concurrent"}],
 "camera": {"type":"pull_out","speed":"slow"},
 "invariants": ["human.anatomy","gravity.freefall"]
}

`invariants` 列出条件隐含的、无论条件说没说都应当成立的物理/解剖约束,
从这个集合里选:human.anatomy, human.identity, gravity.freefall,
rigid.no_deform, contact.no_interpenetration, object.permanence, fluid.continuity
"""


def _fallback_graph(condition_id: str, text: str, camera: str | None) -> RequirementGraph:
    """A minimal graph when no LLM is available.

    Deliberately thin: one presence requirement plus the camera move if the
    router's keywords found one. It keeps the pipeline runnable offline, and its
    thinness is recorded in meta so a run built on it is never mistaken for a
    compiled one.
    """
    reqs = [Requirement(rid="r1", kind="entity",
                        text=f"画面内容与条件描述一致:{text[:60]}",
                        verify="present_most", aspect="entity_presence",
                        hard=True, note="次要细节缺失不算失败")]
    cam: dict[str, Any] = {}
    if camera:
        reqs.append(Requirement(rid="r2", kind="camera",
                                text=f"运镜为 {camera}", verify="trajectory",
                                aspect="camera_control", value=camera, hard=True,
                                note="轻微抖动不算运镜错误"))
        cam = {"type": camera, "speed": "normal"}
    return RequirementGraph(condition_id, text, reqs, [], cam, [],
                            {"compiler": "fallback", "degraded": True})


#: Camera cues, for the deterministic backstop below. Kept minimal and literal:
#: this is a safety net, not a parser.
_CAMERA_CUES: dict[str, tuple[str, ...]] = {
    "static": ("camera fixed", "fixed camera", "static camera", "locked off",
               "camera is static", "固定机位", "镜头固定", "机位固定"),
    "push_in": ("push in", "pushes in", "dolly in", "zoom in", "推近", "推进"),
    "pull_out": ("pull out", "pulls out", "dolly out", "zoom out", "拉远", "拉出"),
    "pan": ("pan left", "pan right", "pans ", "panning", "摇镜", "横移"),
    "orbit": ("orbit", "arc around", "环绕"),
    "follow": ("follows", "tracking shot", "跟拍", "跟随"),
}


def _backstop(g: RequirementGraph, text: str,
              camera_hint: str | None) -> RequirementGraph:
    """Add a camera requirement the compiler dropped.

    Observed on the first live run: a condition ending "Workshop light, camera
    fixed." produced a style requirement and no camera one. The instruction sits
    at the end of a sentence alongside an ambient description, so it reads as
    scene-setting rather than a directive -- and an explicit rule in the system
    prompt did not prevent the omission.

    That is the general lesson: a constraint that must always hold belongs in
    code, not in prompt text. An LLM asked to always do something will
    occasionally not, and a silently missing requirement is invisible -- the
    aspect simply reports "condition specified no camera move" and everyone
    believes it.
    """
    if any(r.kind == "camera" for r in g.requirements):
        return g
    low = (text or "").lower()
    found = camera_hint
    if not found:
        for kind, cues in _CAMERA_CUES.items():
            if any(c in low for c in cues):
                found = kind
                break
    if not found:
        return g
    rid = f"r{len(g.requirements) + 1}"
    g.requirements.append(Requirement(
        rid=rid, kind="camera", text=f"运镜为 {found}", verify="trajectory",
        aspect=KIND_TO_ASPECT["camera"], value=found, hard=True,
        note="轻微抖动不算运镜错误;主体运动不等于镜头运动"))
    if not g.camera:
        g.camera = {"type": found, "speed": "normal"}
    g.meta["camera_backstop"] = f"compiler omitted a camera requirement; added {found}"
    return g


def compile_condition(condition_id: str, text: str, llm: VLMClient | None = None,
                      *, camera_hint: str | None = None,
                      max_requirements: int = 15) -> RequirementGraph:
    if llm is None:
        return _fallback_graph(condition_id, text, camera_hint)

    r = llm.ask(system=SYSTEM, user=f"条件:\n{text}\n\n请输出 JSON。",
                schema={"type": "object"}, tag=f"compile/{condition_id}")
    if not r.ok:
        g = _fallback_graph(condition_id, text, camera_hint)
        g.meta["error"] = r.error or "compile failed"
        return g

    p = r.parsed or {}
    reqs: list[Requirement] = []
    for i, item in enumerate(p.get("requirements") or []):
        kind = str(item.get("kind", "entity"))
        aspect = KIND_TO_ASPECT.get(kind)
        if aspect is None:
            continue
        try:
            reqs.append(Requirement(
                rid=str(item.get("rid") or f"r{i+1}"), kind=kind,  # type: ignore[arg-type]
                text=str(item.get("text", "")).strip(),
                verify=str(item.get("verify", "present_any")),  # type: ignore[arg-type]
                aspect=aspect, subject=str(item.get("subject", "")),
                value=item.get("value"), hard=bool(item.get("hard", True)),
                weight=float(item.get("weight", 1.0)),
                note=str(item.get("note", "")),
            ))
        except (TypeError, ValueError):
            continue
    reqs = reqs[:max_requirements]
    rids = {r_.rid for r_ in reqs}
    order = [OrderEdge(str(e.get("before")), str(e.get("after")),
                       str(e.get("relation", "before")))  # type: ignore[arg-type]
             for e in (p.get("order") or [])
             if str(e.get("before")) in rids and str(e.get("after")) in rids]

    g = _backstop(RequirementGraph(
        condition_id=condition_id, condition_text=text, requirements=reqs,
        order=order, camera=p.get("camera") or {},
        invariants=[str(v) for v in (p.get("invariants") or [])],
        meta={"compiler": llm.model, "n_raw": len(p.get("requirements") or [])}),
        text, camera_hint)
    errs = validate(g)
    if errs:
        g.meta["validation_errors"] = errs
    return g


def compile_batch(conditions: Sequence[tuple[str, str]], llm: VLMClient | None,
                  out_dir: str | Path, *, overwrite: bool = False) -> dict[str, Any]:
    """Compile many conditions to ``<out_dir>/<condition_id>.json``.

    Skips existing files by default: graphs are meant to be frozen, and silently
    recompiling one mid-experiment would change what a model is measured against
    without anything looking different.
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    stats = {"compiled": 0, "skipped": 0, "degraded": 0, "invalid": 0,
             "n_requirements": []}
    for cid, text in conditions:
        p = out / f"{cid}.json"
        if p.exists() and not overwrite:
            stats["skipped"] += 1
            continue
        g = compile_condition(cid, text, llm)
        g.save(p)
        stats["compiled"] += 1
        if g.meta.get("degraded"):
            stats["degraded"] += 1
        if g.meta.get("validation_errors"):
            stats["invalid"] += 1
        stats["n_requirements"].append(len(g.requirements))
    ns = stats.pop("n_requirements")
    stats["mean_requirements"] = round(sum(ns) / len(ns), 2) if ns else 0.0
    return stats
