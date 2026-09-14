"""Route a defect class to the layer that can actually find it.

The coverage bench drew a hard line. Numbers own frame-level temporal defects --
freeze and frame-drop at 100% detection and 97-100% localisation against a 5%
false-positive rate, luma pulses at 94% -- and cover spatially local ones at
essentially nothing: every signal peak sits on the noise floor, and locus
extraction tops out at 31% recall at top-3 while an untouched clip still yields
ten nominations.

So the nominate-then-adjudicate chain is capped by its nomination step for
exactly the defects it was built for, which is why it measured 52.8% against
65.9% for handing the model the whole clip. The two classes need opposite
pipelines:

    temporal   numbers locate it exactly, the model only says whether the
               stillness was legitimate -- a static camera on a still scene and
               a stalled subject are the same measurement and different verdicts
    spatial    no nomination at all; the model watches the clip and says where,
               and only then is that place magnified to confirm

The model is never asked for a number. Every confidence it has been asked for in
this project came back a constant -- margin 'clear' on 797 of 800 calls,
integrity 10 on 96.8% of loci, confidence 0.5 on 14 of 14 -- so the score is
computed from the structure of what came back, not stated by the judge.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from agenteval.llm.client import ImageRef, VideoRef, VLMClient
from agenteval.media.clip import VideoHandle

# Measured detection at a 5% false-positive rate, from runs/detmat.
TEMPORAL_COVERAGE = {"frame_repeat": 1.00, "frame_drop": 1.00, "luma_pulse": 0.94}
SPATIAL_UNCOVERED = ("local_blur", "region_shuffle", "patch_jump",
                     "affine_warp", "patch_swap")


@dataclass
class Finding:
    """One confirmed problem, with where it is and what confirmed it."""

    kind: str                       # temporal | spatial
    aspect: str
    t_span: tuple[float, float]     # normalised
    bbox: tuple[float, float, float, float] | None
    severity: str                   # 轻微 | 明显 | 严重
    found_by: str                   # which layer nominated it
    confirmed_by: str               # which call confirmed it
    note: str = ""

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.kind, "aspect": self.aspect,
                "t_span": [round(v, 3) for v in self.t_span],
                "bbox": [round(v, 3) for v in self.bbox] if self.bbox else None,
                "severity": self.severity, "found_by": self.found_by,
                "confirmed_by": self.confirmed_by, "note": self.note[:160]}


@dataclass
class ClipReport:
    video: str
    findings: list[Finding] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    score: float = 10.0

    def to_json(self) -> dict[str, Any]:
        return {"video": self.video, "score": round(self.score, 2),
                "findings": [f.to_json() for f in self.findings],
                "checked": self.checked, "unresolved": self.unresolved}


SEVERITY_COST = {"轻微": 0.6, "明显": 1.6, "严重": 3.0}


def score_from(findings: Sequence[Finding]) -> float:
    """Accumulate, do not take the worst.

    One severe failure and three mild ones are different clips, and worst-of
    calls them equal. Measured the hard way: a severity cap implemented as
    min(score, cap) discarded the accumulated penalties and collapsed 420
    scores onto three distinct values.
    """
    return max(0.0, 10.0 - sum(SEVERITY_COST.get(f.severity, 1.0)
                               for f in findings))


# ---- temporal: numbers locate, the model adjudicates ---------------------

STILL_SYSTEM = """\
你在核实一段 **AI 生成视频**里的一处「画面几乎没有变化」的时段。

这个时段是**测量出来的**,不是猜的——该段相邻帧之间的差异远低于全片水平。
测量能告诉你「这里没变」,**不能**告诉你这算不算问题。只有两种可能:

  · **正常**:静止镜头拍静止的东西;人物本来就停顿;镜头处于稳定的静止段落。
  · **卡顿**:此刻画面里**本来应该有动作正在进行**,却停住了,之后又突然接上。

判据只有一条:**看这段前后的画面,此刻本该发生什么?**
前后如果是连贯的动作而中间停住,就是卡顿;如果本来就没什么在动,就是正常。
"""

STILL_PROMPT = """下面依次是:该时段**之前**的连续帧、**之内**的连续帧、**之后**的连续帧。

1. `before_after`:前后两段里,有什么东西正在运动?(说具体的物体和动作)
2. `during`:中间这段里,那个东西在做什么?
3. `verdict`:`正常`(本来就没在动/静止镜头) | `卡顿`(该动却停住了) | `看不清`
4. `severity`:仅当 verdict=卡顿 时给,`轻微`|`明显`|`严重`

输出 JSON:{"before_after":"...","during":"...","verdict":"正常"|"卡顿"|"看不清",
 "severity":"轻微"|"明显"|"严重"|null}"""


def temporal_pass(video: VideoHandle, vlm: VLMClient, out_dir: Path, *,
                  calibration: dict | None = None, z: float = 3.0,
                  max_events: int = 3) -> tuple[list[Finding], list[str]]:
    """Find still spans by measurement, then ask whether each one is legitimate."""
    import cv2

    from agenteval.tools.renders import _label, _resize_side, _tile, _write

    idx = list(range(min(video.total, 96)))
    gray = video.read_gray(idx, max_side=192)
    if len(gray) < 12:
        return [], ["temporal:解码不足"]
    e = np.asarray([float(cv2.absdiff(a, b).mean())
                    for a, b in zip(gray, gray[1:])])
    med = float(np.median(e)) or 1e-6
    ratio = e / med
    # A still span is a run below a fraction of the clip's own typical change.
    quiet = ratio < 0.25
    spans, i = [], 0
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j + 1 < len(quiet) and quiet[j + 1]:
                j += 1
            if j - i + 1 >= 3:
                spans.append((i, j + 1, float(ratio[i:j + 1].mean())))
            i = j + 1
        else:
            i += 1
    spans.sort(key=lambda s: s[2])
    spans = spans[:max_events]
    if not spans:
        return [], []

    findings, unresolved = [], []
    n = len(e)
    for k, (a, b, r) in enumerate(spans):
        pre = list(range(max(0, a - 6), a))
        mid = list(range(a, min(n, b)))
        post = list(range(b, min(n, b + 6)))
        imgs = []
        for name, span in (("之前", pre), ("之内", mid), ("之后", post)):
            if not span:
                continue
            pick = span[:: max(1, len(span) // 4)][:4]
            tiles = [_label(_resize_side(f, 360), f"{name} f{t}")
                     for t, f in zip(pick, video.read(pick))]
            if tiles:
                p = _write(out_dir / "still", f"still_{video.path.stem}_{k}_{name}",
                           _tile(tiles, len(tiles)))
                imgs.append(ImageRef(path=p, caption=f"{name}"))
        if not imgs:
            continue
        resp = vlm.ask(system=STILL_SYSTEM, user=STILL_PROMPT, images=imgs,
                       schema={"type": "object"},
                       tag=f"routed/still/{video.path.stem}/{k}")
        p_ = resp.parsed or {}
        v = str(p_.get("verdict", "看不清"))
        if v == "卡顿":
            findings.append(Finding(
                kind="temporal", aspect="卡顿", t_span=(a / n, b / n), bbox=None,
                severity=str(p_.get("severity") or "明显"),
                found_by=f"measurement(ratio={r:.2f})", confirmed_by="vlm/still",
                note=str(p_.get("during") or "")))
        elif v == "看不清":
            unresolved.append(f"卡顿@{a/n:.2f}-{b/n:.2f}")
    return findings, unresolved


# ---- spatial: the model looks, then the place it names is magnified ------

WATCH_SYSTEM = """\
你在看一段 **AI 生成视频**,任务是**描述你实际看到的画面异常**。

只说你能指出来的:是什么东西、大概在第几秒、画面的哪个位置。
**不要**评价整体好坏,不要给分数,不要用「质量不高」这类笼统说法。
如果整段都没有能具体指出的异常,就返回空列表——**空列表是合法且常见的答案**。
"""

# The claim's type decides what evidence can settle it. The audit found every
# spatial claim unconfirmable, and not because the model was wrong: a
# penetration claim was answered with a side view that cannot show ordering, a
# rigidity claim with two frames of an object that was turning, a trajectory
# claim with a twelfth of a second at the apex of a toss. The evidence has to
# match the claim.
CLAIM_TYPES = {
    "穿模": "一个物体穿过另一个物体,而不是绕过、碰撞或被遮挡",
    "刚体形变": "本该保持形状的物体(车、桌、工具、角、骨架)在运动中弯曲或扭曲",
    "物理轨迹": "运动不符合物理(该下落的悬停、该减速的加速、凭空改变方向)",
    "结构崩坏": "肢体/手指/五官的数量或连接关系改变,物体融合或糊烂",
    "凭空出现消失": "东西突然出现或消失,没有进出画面或被遮挡的过程",
}

WATCH_PROMPT = """请列出你在这段视频里**能具体指认**的画面异常,每条给:

- `type`:必须是下面五类之一
%s
- `what`:具体是什么(哪个物体、发生了什么)
- `t`:大约在整段的第几成(0~1)
- `where`:画面的哪个区域,用 [x, y, w, h] 表示,**都用 0~1 的比例**,原点在左上

最多列 3 条,按严重程度排。没有就返回 {"items": []}。

输出 JSON:{"items":[{"type":"...","what":"...","t":0~1,"where":[x,y,w,h]}]}""" % (
    "\n".join(f"  · `{k}` —— {v}" for k, v in CLAIM_TYPES.items()))

# Per type: how much time the evidence should span, and which tool gathers it.
EVIDENCE_PLAN = {
    "穿模": ("depth", 0.10),
    "刚体形变": ("rigidity", 0.08),
    "物理轨迹": ("trajectory", 1.00),
    "结构崩坏": ("zoom", 0.05),
    "凭空出现消失": ("span", 0.20),
}

VERIFY_SYSTEM_BASE = """\
你在核实一条关于 **AI 生成视频**的具体指控。

看不清就选 `看不清`。**宁可说看不清,也不要把正常现象说成缺陷。**
下面这些都会让画面变得奇怪,但都**不是**缺陷:快速运动的模糊、被遮挡后重现、
物体转向镜头造成的透视缩短、光影与反射变化、水花火焰树叶等本就细碎的纹理。
"""

VERIFY_RULE = {
    "穿模": "判据:看深度图。两物重叠处若前后关系**明确反转**(本该在后的跑到前面,"
            "或一个物体被另一个从中间切成前后两段)才算穿模;若近者始终在前、"
            "只是轮廓重叠,那是**遮挡,不是缺陷**。深度图有噪声,边界模糊不算。",
    "刚体形变": "判据:刚体上任意两点的距离应当恒定。已给出跟踪测量。"
                "**注意排除透视缩短**——物体转向镜头时投影长度本来就会变,"
                "那不是形变。只有当物体朝向基本不变而形状仍在变时才算。",
    "物理轨迹": "判据:看整段的运动曲线和轨迹。自由落体应当**匀加速**(残影间距越来越大),"
                "推力消失后应当减速。**注意**:抛物线顶点附近本来就几乎不动,"
                "短时间内位置不变**不是**缺陷。",
    "结构崩坏": "判据只有一条:同一个物体在相邻两格之间,**拓扑变了没有**?"
                "该有几根手指还是几根,该连着的还连着。",
    "凭空出现消失": "判据:看给出的较长时段。物体若从画面边缘进入、或从遮挡物后出现,"
                    "那是**正常的**;只有在画面中央、无遮挡处突然出现或消失才算。",
}

VERIFY_PROMPT = """待核实的指控(类型:%s):%s

%s

1. `observed`:证据里具体呈现了什么?(先描述,不下结论)
2. `verdict`:`确认` | `排除` | `看不清`
3. `severity`:仅当 verdict=确认 时给,`轻微`|`明显`|`严重`

输出 JSON:{"observed":"...","verdict":"确认"|"排除"|"看不清",
 "severity":"轻微"|"明显"|"严重"|null}"""


def _norm_box(b, w: int, h: int):
    """The model gives pixels as often as fractions; normalise defensively."""
    if not b or len(b) != 4:
        return None
    try:
        v = [float(x) for x in b]
    except (TypeError, ValueError):
        return None
    if max(v) > 1.5:
        v = [v[0] / w, v[1] / h, v[2] / w, v[3] / h]
    x, y, bw, bh = [min(max(t, 0.0), 1.0) for t in v]
    if bw < 0.03 or bh < 0.03:
        return None
    return (x, y, min(bw, 1 - x), min(bh, 1 - y))


def gather_evidence(video: VideoHandle, kind: str, t: float, box, out_dir: Path):
    """The evidence the claim type needs, not the evidence that is cheapest."""
    from agenteval.signals.suspicion import SuspicionLocus
    from agenteval.tools.depth import depth_pair
    from agenteval.tools.locus_view import locus_strip
    from agenteval.tools.renders import filmstrip, motion_trail

    plan, frac = EVIDENCE_PLAN.get(kind, ("zoom", 0.05))
    half = max(2, int(video.total * frac / 2))
    centre = int(min(max(t, 0.0), 1.0) * (video.total - 1))
    t0, t1 = max(0, centre - half), min(video.total, centre + half + 1)

    if plan == "depth":
        return depth_pair(video, out_dir, t_span=(t0, t1), bbox=box, n=3)
    if plan == "trajectory":
        return motion_trail(video, out_dir, n=8, tag="traj")
    if plan == "span":
        return filmstrip(video, out_dir, t0=t0, t1=t1, n=8, cols=4,
                         side=340, tag="span")
    if plan == "rigidity":
        from agenteval.tools.physics import rigidity_check, track_points
        tr = track_points(video, bbox=box, t_span=(t0, t1))
        strip = locus_strip(video, SuspicionLocus(
            locus_id="R", t_span=(t0, t1), bbox=box or (0, 0, 1, 1),
            score=0.0, signals={}, n_cells=0), out_dir, n=4, tag="rig")
        rc = rigidity_check(tr) if not tr.value.get("error") else None
        if rc is not None and not rc.value.get("error"):
            strip.value["rigidity"] = rc.value
            strip.hint += ("\n\n跟踪测量:" + json.dumps(
                {k: v for k, v in rc.value.items() if isinstance(v, (int, float))},
                ensure_ascii=False) + "\n" + (rc.hint or ""))
        return strip
    return locus_strip(video, SuspicionLocus(
        locus_id="Z", t_span=(t0, t1), bbox=box or (0, 0, 1, 1),
        score=0.0, signals={}, n_cells=0), out_dir, n=4, tag="zoom")


def spatial_pass(video: VideoHandle, vlm: VLMClient, out_dir: Path, *,
                 max_items: int = 3) -> tuple[list[Finding], list[str]]:
    """No nomination step: the model watches, says where, and names the type.

    The type is what routes the evidence. Without it every claim got the same
    four magnified frames, and an audit of seven claims on real clips confirmed
    none of them -- not because the claims were wrong but because four frames
    cannot settle a penetration, a rigidity or a trajectory question.
    """
    resp = vlm.ask_multimodal(system=WATCH_SYSTEM, user=WATCH_PROMPT,
                              parts=[VideoRef(video.path)],
                              schema={"type": "object"},
                              tag=f"routed/watch/{video.path.stem}")
    items = (resp.parsed or {}).get("items") or []
    if not isinstance(items, list):
        return [], ["spatial:回复格式异常"]

    findings, unresolved = [], []
    probe = video.read([0])
    if not probe:
        return [], ["spatial:解码失败"]
    h, w = probe[0].shape[:2]
    for k, it in enumerate(items[:max_items]):
        if not isinstance(it, dict):
            continue
        what = str(it.get("what") or "")[:80]
        kind = str(it.get("type") or "").strip()
        if kind not in CLAIM_TYPES:
            kind = "结构崩坏"
        try:
            t = float(it.get("t"))
        except (TypeError, ValueError):
            unresolved.append(f"{what}(未给时刻)")
            continue
        box = _norm_box(it.get("where"), w, h) or (0.0, 0.0, 1.0, 1.0)
        ev = gather_evidence(video, kind, t, box, out_dir / "ev")
        if not ev.images:
            unresolved.append(f"{what}(证据无法生成)")
            continue
        v = vlm.ask(
            system=VERIFY_SYSTEM_BASE,
            user=VERIFY_PROMPT % (kind, what, VERIFY_RULE.get(kind, ""))
            + "\n\n" + (ev.hint or ""),
            images=[ImageRef(path=p, caption="证据") for p in ev.images],
            schema={"type": "object"},
            tag=f"routed/verify/{kind}/{video.path.stem}/{k}")
        p_ = v.parsed or {}
        verdict = str(p_.get("verdict", "看不清"))
        if verdict == "确认":
            findings.append(Finding(
                kind="spatial", aspect=f"{kind}:{what}", t_span=(t, t),
                bbox=box, severity=str(p_.get("severity") or "明显"),
                found_by="vlm/watch", confirmed_by=f"vlm/verify[{kind}]",
                note=str(p_.get("observed") or "")))
        elif verdict == "看不清":
            unresolved.append(f"{kind}:{what}")
    return findings, unresolved


def evaluate_clip(path: str | Path, vlm: VLMClient, out_dir: Path, *,
                  calibration: dict | None = None) -> ClipReport:
    """Both passes, then a score computed from what they returned."""
    v = VideoHandle(path)
    rep = ClipReport(video=Path(path).stem)
    tf, tu = temporal_pass(v, vlm, out_dir, calibration=calibration)
    sf, su = spatial_pass(v, vlm, out_dir)
    rep.findings = tf + sf
    # Silence is not cleanliness: record what was examined, so an aspect that
    # was never checked cannot be read as an aspect that passed.
    rep.checked = ["卡顿(测量定位)", "局部结构(整段观看)"]
    rep.unresolved = tu + su
    rep.score = score_from(rep.findings)
    return rep
