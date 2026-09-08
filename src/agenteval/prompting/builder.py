"""Dynamic system-prompt construction.

A skill's prompt cannot be a fixed string. The same "judge temporal integrity"
instruction produces false positives on a fast handheld shot (motion blur read
as softness) and false negatives on a locked-off macro shot (subtle boiling
dismissed as noise). What the judge should be told depends on the condition,
on what the detectors measured, and on which other skills are running.

So a prompt is *assembled* per input from typed fragments:

    BASE            what this dimension is
    RUBRIC          the defect taxonomy for it: definitions, what each defect is
                    confusably NOT, and per-defect severity anchors
    SCOPE           what it must NOT judge, derived from the active skill set,
                    so two skills cannot both charge for the same defect
    CONDITION       the specific entities / actions / camera asked for
    PRIORS          failure modes likely for *this* content (hands, water, text...)
    BRIEFING        what the signals and detectors actually found, in words
    DO_NOT_PENALIZE measured facts that would otherwise be misread as defects
    ANCHORS         severity calibration examples matched to the content
    TOOL_TRUST      how far to trust the numbers, set by measured reliability
    RULES           the invariant judging discipline
    OUTPUT          the response contract

``DO_NOT_PENALIZE`` is the fragment that earns its keep. Most false positives in
VLM video judging are correct observations misfiled as defects: real motion blur
under fast motion, real grain in a dark scene, real bokeh from a shallow lens.
Those can only be ruled out by measuring the clip first, which is why this must
be built after seeing the input rather than written once by hand.

Fragments are ordered, deduplicated, and rendered with stable headings so the
whole prompt stays diffable across runs — a prompt that silently changes shape
makes two evaluations incomparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Iterable, Sequence


class Slot(IntEnum):
    BASE = 10
    RUBRIC = 15
    SCOPE = 20
    CONDITION = 30
    PRIORS = 40
    BRIEFING = 50
    DO_NOT_PENALIZE = 60
    ANCHORS = 70
    TOOL_TRUST = 80
    RULES = 90
    OUTPUT = 100


HEADINGS: dict[Slot, str] = {
    Slot.BASE: "",
    Slot.RUBRIC: "## 缺陷清单与严重度标尺",
    Slot.SCOPE: "## 本维度的边界",
    Slot.CONDITION: "## 这条视频的生成条件",
    Slot.PRIORS: "## 针对本条内容的重点关注",
    Slot.BRIEFING: "## 信号与检测器的测量结果",
    Slot.DO_NOT_PENALIZE: "## 以下现象**不要**当作缺陷扣分",
    Slot.ANCHORS: "## 严重度标尺(参考样例)",
    Slot.TOOL_TRUST: "## 如何对待工具数值",
    Slot.RULES: "## 判定纪律",
    Slot.OUTPUT: "## 输出",
}


@dataclass(frozen=True)
class Fragment:
    slot: Slot
    text: str
    key: str = ""          # dedupe key; empty means "always keep"
    source: str = ""       # what produced it, for auditing the prompt


@dataclass
class PromptBuilder:
    fragments: list[Fragment] = field(default_factory=list)

    def add(self, slot: Slot, text: str, *, key: str = "", source: str = "") -> "PromptBuilder":
        if text and text.strip():
            self.fragments.append(Fragment(slot, text.strip(), key, source))
        return self

    def extend(self, frags: Iterable[Fragment]) -> "PromptBuilder":
        for f in frags:
            if f.text and f.text.strip():
                self.fragments.append(f)
        return self

    def build(self) -> str:
        seen: set[tuple[int, str]] = set()
        by_slot: dict[Slot, list[str]] = {}
        for f in sorted(self.fragments, key=lambda x: x.slot):
            k = (int(f.slot), f.key or f.text)
            if k in seen:
                continue
            seen.add(k)
            by_slot.setdefault(f.slot, []).append(f.text)
        out: list[str] = []
        for slot in sorted(by_slot):
            head = HEADINGS.get(slot, "")
            body = "\n".join(by_slot[slot])
            out.append(f"{head}\n{body}" if head else body)
        return "\n\n".join(out)

    def manifest(self) -> list[dict[str, Any]]:
        """What went into this prompt and why. Logged with every run so a
        surprising verdict can be traced back to the fragment that caused it."""
        return [{"slot": f.slot.name, "source": f.source, "key": f.key,
                 "chars": len(f.text)} for f in sorted(self.fragments,
                                                       key=lambda x: x.slot)]


# ---- fragment libraries -------------------------------------------------

#: Content cue -> extra guidance. Cues come from the condition text and from
#: cheap detector passes, so this fires on what is actually in the clip.
CONTENT_PRIORS: dict[str, str] = {
    "human": "画面里有人。人体是生成模型最脆弱的部分,优先检查:手指数量与形态、"
             "面部五官在时间上是否稳定、四肢长度是否恒定、关节朝向是否可能。",
    "hands": "画面里有清晰可见的手。手指粘连、多指少指、指节反向是高发缺陷,"
             "必须放大到手部再判断,整帧缩略图一定看不出来。",
    "face_closeup": "存在人脸特写。重点看:五官比例在帧间是否漂移、牙齿与眼睛的细节、"
                    "以及是否有过度平滑导致的塑料感。另外真人会眨眼,"
                    "整段完全不眨眼是生成视频的常见破绽。",
    "text": "条件要求画面中出现文字。生成模型的文字渲染极不可靠,"
            "检查字形是否成立、是否在帧间变化。",
    "water": "画面含水/流体。检查流动方向是否一致、是否有凭空出现或消失的水体、"
             "水面反射是否跟随主体。",
    "crowd": "画面中有多个人物。除了单体质量,还要检查人数在帧间是否稳定,"
             "以及远景人物是否退化成模糊的团块。",
    "fast_motion": "画面存在大幅快速运动。重点检查运动是否连贯、有无跳变,"
                   "以及主体在高速运动中是否发生形变。",
    "static_camera": "镜头基本固定。这种情况下背景应当稳定,"
                     "任何背景漂移、纹理蠕动都是明确缺陷。",
    "low_light": "画面整体偏暗。注意区分真实的暗部噪点与生成瑕疵。",
}

#: Measured fact -> what must not be charged as a defect. This is the
#: false-positive brake, and it only works when built from measurements.
DO_NOT_PENALIZE: dict[str, str] = {
    "fast_motion": "本段视频实测运动幅度很大。**运动模糊是物理上正确的表现**,"
                   "不要把运动模糊当成清晰度缺陷。只有在低运动区域出现的模糊才算缺陷。",
    "shallow_dof": "画面存在明显的景深虚化。**背景虚化是摄影意图**,不是生成缺陷。",
    "low_light": "画面偏暗。**暗部噪点与低对比是正常的**,不要据此扣画质分。",
    "intentional_cut": "条件本身要求了镜头切换。**镜头切换处的画面突变是预期的**,"
                       "不要当作时间跳变缺陷。",
    "stylized": "条件要求了非写实风格(动画/CG/绘画等)。"
                "**不要用照片写实的标准去要求它**,也不要把风格化的简化当成细节缺失。",
    "small_face": "画面中的人脸很小。**在没有放大裁剪的情况下不得对人脸细节下结论**,"
                  "看不清就报'证据不足',不要猜。",
    "static_scene": "条件本身描述的就是近乎静止的场景。"
                    "**画面变化少不等于卡顿**,不要据此报运动缺陷。",
}

TOOL_TRUST_HIGH = (
    "下面给出的工具数值来自较可靠的测量,可以作为**去哪里看**的依据。"
    "但它们仍然只是线索:最终判断必须来自你在图像里看到的内容。"
)
TOOL_TRUST_LOW = (
    "下面的工具数值来自可靠性较低的估计(检测器在本段视频上表现不稳)。"
    "**只能**用它们决定看哪里,**不得**用它们本身作为缺陷成立的证据。"
)

RULES = """\
1. 只依据给你的证据作答。看不清就说看不清,禁止脑补。
2. 若画面所见与工具数值冲突,**以画面为准**,并在 rationale 里写明冲突。
3. rationale 必须描述你在图像里**看到了什么**,不能只复述数值。
4. 每条 finding 必须给出它依据的 evidence id(形如 E01)。给不出就不要报这条。
5. 没有发现问题是完全正常的结论。不要为了交差而编造 finding。
6. 严重度按实际影响判定:minor=仔细看才注意到;major=正常观看即可察觉;
   critical=一眼就毁掉整段观感,或使视频无法用于其预期用途。
"""


def scope_fragment(this_skill: str, active_skills: Sequence[str]) -> str:
    """Tell a skill what belongs to its siblings.

    Without this, every skill charges for every visible flaw and the dimension
    scores become copies of one another — the exact failure that makes a
    multi-dimensional report worthless.
    """
    others = [s for s in active_skills if s != this_skill]
    if not others:
        return ""
    return ("本次评测同时在跑这些维度:" + "、".join(others) + "。\n"
            "属于它们职责范围的问题,你**不要**重复报告,也不要计入你的严重度判断。"
            "如果你注意到了但它不属于你,忽略即可。")


def briefing_from(evidence_blocks: Sequence[dict[str, Any]]) -> str:
    """Render measurements as prose. A judge reads a sentence far better than a
    JSON blob, and prose forces us to state what the number *means*."""
    if not evidence_blocks:
        return ""
    lines = []
    for b in evidence_blocks:
        hint = b.get("hint", "")
        eid, tool = b.get("eid", "?"), b.get("tool", "?")
        lines.append(f"- [{eid}] {tool}: {hint}" if hint else f"- [{eid}] {tool}")
    return "\n".join(lines)
