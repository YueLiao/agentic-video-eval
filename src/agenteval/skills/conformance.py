"""Conformance — did the video do what the condition asked?

Scored against a frozen requirement graph rather than an overall impression.
"Does this match the prompt?" collapses a dozen independent claims into one
feeling, and a video that satisfies eleven and drops a subject still reads as
"mostly matching" -- which is where conformance ceilings come from. Walking the
requirements one at a time makes the miss countable and nameable.

Findings here are *unmet requirements*, not taxonomy defects, so each names its
aspect directly and carries the requirement id.

Split three ways because the evidence genuinely differs. Presence, attributes
and counts are decided from stills laid out together. Actions and their ordering
need the sequence preserved. Camera is largely decidable without the model at
all, from a homography estimate, with the model only confirming.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from agenteval.engine.actions import Action
from agenteval.engine.evidence import Evidence
from agenteval.planning.schema import Requirement, RequirementGraph
from agenteval.skills.base import Presentation, Skill, SkillContext
from agenteval.tools import camera as C
from agenteval.tools import renders as R
from agenteval.tools import scan

_SHARED_RULES = """\
## 判定规则

1. 逐条核对下面的要求清单,**只判断条件要求的东西**,不要评判条件没提的内容。
2. 每条要求都附了 `note`,写明**什么不算失败**。报告未满足之前先排除这些情况。
3. 只有确实未满足才报 finding。`kind` 必须写要求的 rid(如 "r3"),
   `aspect` 必须写该要求的 aspect 字段值。
4. 证据不足以判断某条要求时,**报 confidence 低于 0.4 并在 rationale 里说明**,
   不要猜。看不清而误报"未满足",比漏报更糟。
5. 全部满足时返回空的 findings 数组。
"""


def _render_requirements(reqs: Sequence[Requirement]) -> str:
    if not reqs:
        return "(本维度在该条件下没有要求)"
    lines = []
    for r in reqs:
        hard = "硬性" if r.hard else "次要"
        lines.append(f"- [{r.rid}] ({hard}, 验证方式={r.verify}, aspect={r.aspect}) "
                     f"{r.text}" + (f"\n      不算失败:{r.note}" if r.note else ""))
    return "\n".join(lines)


class _ConformanceBase(Skill):
    kinds: tuple[str, ...] = ()

    def __init__(self, out_dir: str | Path, graph: RequirementGraph) -> None:
        self.out_dir = Path(out_dir)
        self.graph = graph

    @property
    def requirements(self) -> list[Requirement]:
        return self.graph.by_kind(*self.kinds)

    def applies(self, ctx: SkillContext) -> bool:
        return bool(self.requirements)

    @property
    def system_prompt(self) -> str:
        return (self.BASE + "\n\n## 需要核对的要求\n"
                + _render_requirements(self.requirements) + "\n" + _SHARED_RULES)


class SemanticConformance(_ConformanceBase):
    """Entities, attributes, counts, spatial relations, style, rendered text."""

    name = "semantic_conformance"
    dimension = "semantic"
    kinds = ("entity", "attribute", "count", "relation", "style", "text")
    covers = ("entity_presence", "attribute_binding", "object_count",
              "spatial_relation", "style_match", "text_rendering")
    presentation = Presentation.COMPOSITE
    max_images = 10
    max_rounds = 4

    BASE = """\
你在核对一段 **AI 生成视频**是否满足条件中的**语义要求**:
主体是否出现、属性是否正确、数量是否对、空间关系是否成立、风格是否匹配、文字是否正确。

注意验证方式的差别:
- `present_any` 至少某一帧出现即可
- `present_most` 大部分时间应当可见,偶尔被遮挡不算失败
- `co_occur` 若干对象必须**同时出现在同一帧**——分别出现过不算满足
- `continuous` 必须持续存在

你**不要**判断:画质好坏、有没有瑕疵、动作自不自然。那些是别的维度的事。"""

    def seed(self, ctx: SkillContext) -> list[Evidence]:
        plan = scan.adaptive_sweep_plan(ctx.video, budget_calls=2, batch=9)
        b = (plan.get("batches") or [[0]])[0]
        return [ctx.bus.get_or_run(
            "contact_sheet", "1.0",
            lambda: scan.contact_sheet(ctx.video, self.out_dir, b, tag="conf"),
            batch_index="conf0")]

    def actions(self, ctx: SkillContext) -> list[Action]:
        def wide(n: int = 8) -> Evidence:
            return ctx.bus.get_or_run(
                "overview_frames", "1.0",
                lambda: R.ordered_frames(ctx.video, self.out_dir, t0=0,
                                         t1=ctx.video.total, n=int(n),
                                         tag="conf_wide"), n=int(n))

        def zoom(bbox: list, t: int) -> Evidence:
            bb = tuple(float(v) for v in bbox[:4])
            a = max(0, int(t) - 2)
            return ctx.bus.get_or_run(
                "zoom_confirm", "1.0",
                lambda: R.zoom_grid(ctx.video, self.out_dir, bbox=bb,
                                    t_span=(a, a + 6), n=4, cols=2,
                                    tag=f"cz_{t}"),
                bbox=[round(v, 3) for v in bb], t=int(t))

        def tile(t: int) -> Evidence:
            return ctx.bus.get_or_run(
                "tile_frame", "1.0",
                lambda: scan.tile_frame(ctx.video, self.out_dir, t=int(t),
                                        grid=(2, 2)), t=int(t), grid=2)

        return [
            Action("wide", "取覆盖全片的若干整帧,用于核对主体是否出现、场景是否符合",
                   {"n": "int: 取几帧,默认 8"}, run=wide),
            Action("zoom", "放大指定帧的指定区域,用于确认属性、数量、文字",
                   {"bbox": "list[float]: 归一化 x,y,w,h", "t": "int: 帧号"},
                   run=zoom),
            Action("tile", "把某一帧切成 2x2 放大,用于在不确定位置时找小目标",
                   {"t": "int: 帧号"}, run=tile),
        ]


class ActionConformance(_ConformanceBase):
    """Actions happening, and happening in the stated order."""

    name = "action_conformance"
    dimension = "semantic"
    kinds = ("action", "order")
    covers = ("action_execution", "action_order")
    presentation = Presentation.ORDERED
    max_images = 12
    max_rounds = 4

    BASE = """\
你在核对一段 **AI 生成视频**是否执行了条件要求的**动作**,以及动作之间的**时序关系**。

关键区分:
- `present_any`:动作发生过即可
- `continuous`:条件说了"持续""一直""不停",那么动作必须**贯穿全片**,
  中途停止就是未满足
- 时序关系 `before`:前一个动作必须先于后一个完成
- 时序关系 `concurrent`:两个动作必须**同时**发生,先后发生不算满足

给你的是**按时间顺序**的帧,每帧带帧号与时间戳。请按顺序观察,
判断动作是否真的发生、以及发生的时间关系。

你**不要**判断动作自不自然、流不流畅(那是运动质量维度),
只判断**动作有没有按要求发生**。"""

    def seed(self, ctx: SkillContext) -> list[Evidence]:
        return [ctx.bus.get_or_run(
            "ordered_frames", "1.0",
            lambda: R.ordered_frames(ctx.video, self.out_dir, t0=0,
                                     t1=ctx.video.total, n=10, tag="act_all"),
            t0=0, t1=ctx.video.total, n=10)]

    def actions(self, ctx: SkillContext) -> list[Action]:
        def window(t0: int, t1: int, n: int = 8) -> Evidence:
            return ctx.bus.get_or_run(
                "ordered_frames", "1.0",
                lambda: R.ordered_frames(ctx.video, self.out_dir, t0=int(t0),
                                         t1=int(t1), n=int(n), tag=f"act_{t0}"),
                t0=int(t0), t1=int(t1), n=int(n))

        def zoom_action(bbox: list, t0: int, t1: int) -> Evidence:
            bb = tuple(float(v) for v in bbox[:4])
            return ctx.bus.get_or_run(
                "zoom_action", "1.0",
                lambda: R.filmstrip(ctx.video, self.out_dir, t0=int(t0),
                                    t1=int(t1), bbox=bb, n=6, cols=6,
                                    tag=f"actz_{t0}"),
                bbox=[round(v, 3) for v in bb], t0=int(t0), t1=int(t1))

        return [
            Action("window", "取某个时间窗内的顺序帧,用于确认动作是否在该时段发生",
                   {"t0": "int: 起始帧", "t1": "int: 结束帧",
                    "n": "int: 取几帧,默认 8"}, run=window),
            Action("zoom_action", "把某个区域按时间横向排开,用于看清局部动作",
                   {"bbox": "list[float]: 归一化 x,y,w,h",
                    "t0": "int: 起始帧", "t1": "int: 结束帧"}, run=zoom_action),
        ]


class CameraConformance(_ConformanceBase):
    """Camera behaviour against the instruction.

    Seeded with a measured trajectory rather than a question, because global
    image motion is directly estimable and VLMs judging camera motion from
    sampled frames confuse it with subject motion constantly. The model's job is
    to confirm or overturn the estimate, not to produce one.
    """

    name = "camera_conformance"
    dimension = "semantic"
    kinds = ("camera",)
    covers = ("camera_control",)
    presentation = Presentation.ORDERED
    max_images = 10
    max_rounds = 3

    BASE = """\
你在核对一段 **AI 生成视频**的**运镜**是否符合条件要求。

系统已经用相邻帧的相似变换估计出了相机运动(类型、缩放率、累计平移、抖动指数)。
**这是一个待你确认的假设,不是结论。**

它的主要失效模式是:画面中有大面积移动的主体时,特征匹配会被主体带走,
把"主体在动"误读成"镜头在摇"。inlier_ratio 低于 0.5 时该估计不可信。

所以请对照顺序帧确认:**背景**是否真的在按估计的方式移动。
判断依据是背景的行为,不是主体的行为。

如果条件要求固定机位,那么明显的抖动或漂移就是未满足;
如果条件要求推近/拉远/摇,那么方向必须正确,幅度必须可察觉。"""

    def seed(self, ctx: SkillContext) -> list[Evidence]:
        cam = ctx.bus.get_or_run(
            "camera_motion", "1.0",
            lambda: C.camera_motion(ctx.video, max_frames=None, stride=2))
        seq = ctx.bus.get_or_run(
            "ordered_frames", "1.0",
            lambda: R.ordered_frames(ctx.video, self.out_dir, t0=0,
                                     t1=ctx.video.total, n=8, tag="cam_all"),
            t0=0, t1=ctx.video.total, n=8)
        return [cam, seq]

    def actions(self, ctx: SkillContext) -> list[Action]:
        def window(t0: int, t1: int, n: int = 8) -> Evidence:
            return ctx.bus.get_or_run(
                "ordered_frames", "1.0",
                lambda: R.ordered_frames(ctx.video, self.out_dir, t0=int(t0),
                                         t1=int(t1), n=int(n), tag=f"cam_{t0}"),
                t0=int(t0), t1=int(t1), n=int(n))

        def trail(t0: int, t1: int) -> Evidence:
            return ctx.bus.get_or_run(
                "motion_trail", "1.0",
                lambda: R.motion_trail(ctx.video, self.out_dir, t0=int(t0),
                                       t1=int(t1)), t0=int(t0), t1=int(t1))

        return [
            Action("window", "取某段的顺序帧,用于确认背景是否按估计的方式移动",
                   {"t0": "int: 起始帧", "t1": "int: 结束帧",
                    "n": "int: 取几帧"}, run=window),
            Action("trail", "把多个时刻叠加成轨迹图,用于看出整体运动方向",
                   {"t0": "int: 起始帧", "t1": "int: 结束帧"}, run=trail),
        ]
