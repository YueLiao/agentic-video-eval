"""Routing — deciding what to run, and what to tell it.

Two jobs, and the second matters as much as the first.

*Which skills.* Running every dimension on every clip wastes most of the budget:
human_integrity on a landscape has nothing to judge, and returning a full score
for it would be worse than useless, because it silently rewards a clip for
containing none of the hard content. So skills are enabled from cheap evidence
and disabled otherwise, and a disabled skill reports "not applicable" rather
than a number.

*What each skill is told.* The router is where DO_NOT_PENALIZE gets built, and
that fragment is the main brake on false positives. Most bad VLM video verdicts
are correct observations filed under the wrong heading -- real motion blur on a
fast pan, real grain in a dark scene, real bokeh from a shallow lens. None of
those can be ruled out by a prompt written in advance; they can only be ruled
out by measuring this clip and saying so explicitly.

The router is deliberately not an LLM. It is cheap, reproducible, auditable, and
its decisions are recorded with the reason attached, so a surprising evaluation
can be traced to the routing decision that shaped it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from agenteval.media.clip import VideoHandle, uniform_indices
from agenteval.prompting.builder import (CONTENT_PRIORS, DO_NOT_PENALIZE,
                                         Fragment, Slot)

# Condition cues. Chinese and English, since conditions arrive in both.
CUES: dict[str, tuple[str, ...]] = {
    "human": ("人", "男", "女", "孩子", "老人", "person", "man", "woman", "child",
              "people", "girl", "boy", "dancer", "player", "他", "她"),
    "hands": ("手", "手指", "握", "拿", "抓", "弹奏", "hand", "finger", "grip",
              "holding", "typing", "playing"),
    "face_closeup": ("特写", "面部", "脸", "close-up", "closeup", "portrait", "face"),
    "text": ("文字", "字", "招牌", "标语", "text", "sign", "letters", "written",
             "logo", "caption"),
    "water": ("水", "海", "河", "雨", "波", "喷", "water", "sea", "ocean", "river",
              "rain", "wave", "splash", "fountain"),
    "crowd": ("人群", "多人", "一群", "crowd", "group of", "several people"),
    "stylized": ("动画", "卡通", "二次元", "水墨", "油画", "CG", "anime", "cartoon",
                 "painting", "illustration", "3d render", "stylized"),
    "falling": ("掉", "落", "坠", "抛", "扔", "fall", "drop", "throw", "toss",
                "tumble"),
    "multi_shot": ("切换", "转场", "剪辑", "cut to", "transition", "montage"),
}

CAMERA_CUES: dict[str, tuple[str, ...]] = {
    "static": ("固定", "静止", "不动", "static", "locked", "fixed"),
    "push_in": ("推近", "拉近", "推进", "push in", "zoom in", "dolly in"),
    "pull_out": ("拉远", "后退", "拉出", "pull out", "zoom out", "dolly out"),
    "pan": ("摇", "横移", "平移", "pan", "truck"),
    "orbit": ("环绕", "旋转", "orbit", "arc"),
    "follow": ("跟随", "跟拍", "follow", "tracking shot"),
}


@dataclass
class RouteDecision:
    skills: list[str]
    disabled: dict[str, str] = field(default_factory=dict)     # skill -> why
    hints: dict[str, Any] = field(default_factory=dict)
    fragments: list[Fragment] = field(default_factory=list)
    budget_scale: dict[str, float] = field(default_factory=dict)
    measurements: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"skills": self.skills, "disabled": self.disabled,
                "hints": self.hints, "budget_scale": self.budget_scale,
                "measurements": self.measurements,
                "fragments": [{"slot": f.slot.name, "source": f.source,
                               "key": f.key} for f in self.fragments]}


def _match(text: str, phrase: str) -> bool:
    """Substring for CJK, loose stem match for latin so that "pushes in" hits the
    "push in" cue. Keyword matching is a stopgap: the real source of condition
    structure is the Stage-A compiler, and this exists so routing still works
    before a condition has been compiled."""
    p = phrase.lower()
    if not p.isascii():
        return p in text
    stem = r"\w*\s+".join(re.escape(w) for w in p.split())
    return re.search(rf"\b{stem}\w*", text) is not None


def parse_condition(text: str) -> dict[str, Any]:
    t = (text or "").lower()
    cues = {k: any(_match(t, c) for c in words) for k, words in CUES.items()}
    camera = next((k for k, words in CAMERA_CUES.items()
                   if any(_match(t, c) for c in words)), None)
    return {"cues": cues, "camera": camera}


def probe(video: VideoHandle, *, n: int = 6) -> dict[str, Any]:
    """A few cheap measurements. No VLM, no heavy model beyond one detector pass."""
    import cv2

    idx = uniform_indices(video.total, n)
    gray = video.read_gray(idx, max_side=224)
    bright = float(np.mean(gray)) if gray.size else 0.0

    g2 = video.read_gray(uniform_indices(video.total, min(24, video.total)),
                         max_side=192)
    mags = []
    for a, b in zip(g2, g2[1:]):
        fl = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 2, 11, 2, 5, 1.1, 0)
        mags.append(float(np.sqrt(fl[..., 0] ** 2 + fl[..., 1] ** 2).mean()))
    motion = float(np.mean(mags)) if mags else 0.0

    sharp = float(np.mean([cv2.Laplacian(g, cv2.CV_32F).var() for g in gray])) \
        if gray.size else 0.0

    from agenteval.tools import detectors as D
    face = D.face_detect(video, idx)
    pose = D.pose_detect(video, idx)
    max_face = float(face.value.get("max_face_frac", 0.0))
    return {
        "brightness": round(bright, 1),
        "motion_level": round(motion, 4),
        "sharpness": round(sharp, 1),
        "n_frames_with_face": int(face.value.get("n_frames_with_face", 0)),
        "max_face_frac": round(max_face, 4),
        "n_persons_max": int(pose.value.get("n_persons_max", 0)),
        "duration_s": round(video.duration_s, 2),
        "total_frames": video.total,
    }


# Thresholds are conservative on purpose: routing errs toward enabling a skill
# and toward suppressing a penalty, because a missed dimension costs one wasted
# budget slot while a false positive corrupts the score.
MOTION_FAST = 0.9
MOTION_STATIC = 0.06
DARK = 55.0
SMALL_FACE = 0.08


#: Which requirement kinds each conformance skill answers for.
CONFORMANCE_KINDS: dict[str, tuple[str, ...]] = {
    "semantic_conformance": ("entity", "attribute", "count", "relation",
                             "style", "text"),
    "action_conformance": ("action", "order"),
    "camera_conformance": ("camera",),
}


def route(video: VideoHandle, condition: dict[str, Any],
          *, available: list[str] | None = None,
          graph: Any = None) -> RouteDecision:
    text = condition.get("prompt") or condition.get("prompt_zh") or ""
    parsed = parse_condition(text)
    cues = parsed["cues"]
    m = probe(video)

    has_human = bool(cues["human"] or cues["crowd"]
                     or m["n_persons_max"] > 0 or m["n_frames_with_face"] > 0)
    fast = m["motion_level"] > MOTION_FAST
    static = m["motion_level"] < MOTION_STATIC
    dark = m["brightness"] < DARK
    small_face = 0 < m["max_face_frac"] < SMALL_FACE

    d = RouteDecision(skills=[], measurements=m,
                      hints={"has_human": has_human, "fast_motion": fast,
                             "static": static, "camera": parsed["camera"],
                             "cues": cues})

    # ---- which skills ---------------------------------------------------
    # static_integrity always runs: frame-observable defects are content-
    # independent, and it is the only skill covering visual_quality.
    always = ["temporal_integrity", "motion_quality", "static_integrity"]
    d.skills.extend(always)
    if has_human:
        d.skills.append("human_integrity")
    else:
        d.disabled["human_integrity"] = "no person detected in condition or frames"
    if cues["falling"] or cues["water"] or m["motion_level"] > 0.2:
        d.skills.append("physical_integrity")
    else:
        d.disabled["physical_integrity"] = "near-static scene with no物理事件线索"
    # Conformance routing comes from the compiled graph when there is one.
    # Keyword cues are a stopgap for running before compilation, and they miss
    # real requirements -- "pushes in" against a "push in" cue, an action stated
    # without any listed verb. The graph is the authority on what was asked for.
    if graph is not None:
        kinds = {r.kind for r in getattr(graph, "requirements", [])}
        if getattr(graph, "order", None):
            kinds.add("order")
        for skill, wanted in CONFORMANCE_KINDS.items():
            if kinds & set(wanted):
                if skill not in d.skills:
                    d.skills.append(skill)
            else:
                d.disabled[skill] = f"条件中没有 {'/'.join(wanted)} 类要求"
        d.hints["n_requirements"] = len(getattr(graph, "requirements", []))
    else:
        if parsed["camera"]:
            d.skills.append("camera_conformance")
        else:
            d.disabled["camera_conformance"] = "condition specifies no camera move"
        d.disabled["action_conformance"] = "no compiled requirement graph"
    if available:
        keep = [s for s in d.skills if s in available]
        for s in d.skills:
            if s not in available:
                d.disabled[s] = "skill not implemented yet"
        d.skills = keep

    # ---- budget ---------------------------------------------------------
    for s in d.skills:
        d.budget_scale[s] = 1.0
    if has_human and m["max_face_frac"] > SMALL_FACE:
        d.budget_scale["human_integrity"] = 1.4      # visible faces reward zooming
    if fast:
        d.budget_scale["motion_quality"] = 1.3
    if static:
        d.budget_scale["motion_quality"] = 0.6

    # ---- prompt fragments ------------------------------------------------
    for cue, on in cues.items():
        if on and cue in CONTENT_PRIORS:
            d.fragments.append(Fragment(Slot.PRIORS, CONTENT_PRIORS[cue],
                                        key=cue, source=f"condition:{cue}"))
    if has_human and not cues["human"]:
        d.fragments.append(Fragment(Slot.PRIORS, CONTENT_PRIORS["human"],
                                    key="human", source="probe:person_detected"))
    if fast:
        d.fragments.append(Fragment(Slot.PRIORS, CONTENT_PRIORS["fast_motion"],
                                    key="fast_motion", source="probe:motion_level"))
        d.fragments.append(Fragment(Slot.DO_NOT_PENALIZE,
                                    DO_NOT_PENALIZE["fast_motion"],
                                    key="fast_motion",
                                    source=f"probe:motion_level={m['motion_level']}"))
    if static and not cues["falling"]:
        d.fragments.append(Fragment(Slot.DO_NOT_PENALIZE,
                                    DO_NOT_PENALIZE["static_scene"],
                                    key="static_scene",
                                    source=f"probe:motion_level={m['motion_level']}"))
        d.fragments.append(Fragment(Slot.PRIORS, CONTENT_PRIORS["static_camera"],
                                    key="static_camera", source="probe:motion_level"))
    if dark:
        d.fragments.append(Fragment(Slot.DO_NOT_PENALIZE, DO_NOT_PENALIZE["low_light"],
                                    key="low_light",
                                    source=f"probe:brightness={m['brightness']}"))
        d.fragments.append(Fragment(Slot.PRIORS, CONTENT_PRIORS["low_light"],
                                    key="low_light", source="probe:brightness"))
    if small_face:
        d.fragments.append(Fragment(
            Slot.DO_NOT_PENALIZE, DO_NOT_PENALIZE["small_face"], key="small_face",
            source=f"probe:max_face_frac={m['max_face_frac']}"))
    if cues["stylized"]:
        d.fragments.append(Fragment(Slot.DO_NOT_PENALIZE, DO_NOT_PENALIZE["stylized"],
                                    key="stylized", source="condition:stylized"))
    if cues["multi_shot"]:
        d.fragments.append(Fragment(Slot.DO_NOT_PENALIZE,
                                    DO_NOT_PENALIZE["intentional_cut"],
                                    key="intentional_cut", source="condition:multi_shot"))
    return d
