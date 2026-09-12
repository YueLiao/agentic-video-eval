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

WATCH_PROMPT = """请列出你在这段视频里**能具体指认**的画面异常,每条给:

- `what`:是什么(例如:手指数量变了 / 物体穿过另一个物体 / 刚性物体在扭曲 /
  东西凭空出现或消失 / 纹理糊成一团且不恢复)
- `t`:大约在整段的第几成(0~1)
- `where`:画面的哪个区域,用 [x, y, w, h] 表示,**都用 0~1 的比例**,原点在左上

最多列 3 条,按严重程度排。没有就返回 {"items": []}。

输出 JSON:{"items":[{"what":"...","t":0~1,"where":[x,y,w,h]}]}"""

VERIFY_SYSTEM = """\
你在核实一条关于 **AI 生成视频**的具体指控,证据是该处放大后的连续帧。

判据只有一条:**同一个物体在相邻两格之间,拓扑变了没有?**
该有几根手指还是几根,该连着的还连着,刚体还是那个形状。

下面这些都会让画面变得奇怪,但**都不是崩坏**:快速运动的模糊、被遮挡后重现、
肢体或物体转向镜头造成的透视缩短、光影与反射变化、水花火焰树叶等本就细碎的纹理。

看不清就选 `看不清`。**宁可说看不清,也不要把运动模糊说成崩坏。**
"""

VERIFY_PROMPT = """待核实的指控:%s

1. `observed`:这几格之间,那个物体具体发生了什么变化?(先描述,不下结论)
2. `verdict`:`确认`(拓扑确实变了) | `排除`(可由运动/遮挡/转向/光影解释) | `看不清`
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


def spatial_pass(video: VideoHandle, vlm: VLMClient, out_dir: Path, *,
                 max_items: int = 3) -> tuple[list[Finding], list[str]]:
    """No nomination step: the model watches the clip and says where to look."""
    from agenteval.tools.locus_view import locus_strip
    from agenteval.signals.suspicion import SuspicionLocus

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
        try:
            t = float(it.get("t"))
        except (TypeError, ValueError):
            unresolved.append(f"{what}(未给时刻)")
            continue
        box = _norm_box(it.get("where"), w, h) or (0.0, 0.0, 1.0, 1.0)
        centre = int(min(max(t, 0.0), 1.0) * (video.total - 1))
        loc = SuspicionLocus(
            locus_id=f"W{k:02d}",
            t_span=(max(0, centre - 3), min(video.total, centre + 4)),
            bbox=box, score=0.0, signals={}, n_cells=0)
        strip = locus_strip(video, loc, out_dir / "watch", n=4, tag="w")
        if not strip.images:
            unresolved.append(f"{what}(无法渲染)")
            continue
        v = vlm.ask(system=VERIFY_SYSTEM, user=VERIFY_PROMPT % what,
                    images=[ImageRef(path=p, caption="放大处") for p in strip.images],
                    schema={"type": "object"},
                    tag=f"routed/verify/{video.path.stem}/{k}")
        p_ = v.parsed or {}
        verdict = str(p_.get("verdict", "看不清"))
        if verdict == "确认":
            findings.append(Finding(
                kind="spatial", aspect=what, t_span=(t, t),
                bbox=box, severity=str(p_.get("severity") or "明显"),
                found_by="vlm/watch", confirmed_by="vlm/verify",
                note=str(p_.get("observed") or "")))
        elif verdict == "看不清":
            unresolved.append(what)
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
