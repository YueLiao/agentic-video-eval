"""physical_integrity — does the world in this video behave like a world?

Distinct from motion_quality, which asks whether movement looks natural. This
asks whether it is *possible*: whether things fall the way mass falls, whether
solids stay solid, whether objects that leave the frame stay gone, whether
things resting on surfaces are actually supported.

The evidence is geometric rather than perceptual, and that is the point: a
falling object's vertical velocity is a straight line, a rigid body preserves
its pairwise point distances, and both statements are checkable without any
learned model and without knowing what the object is. What the checks cannot do
is tell a deforming object from a drifting tracker, or a genuinely floating
object from one whose support is out of frame. So every numeric result is
delivered next to a view of the same region, and the model is told plainly that
its job is to confirm what is actually being tracked before it believes any of
the numbers.
"""

from __future__ import annotations

from pathlib import Path

from agenteval.engine.actions import Action
from agenteval.engine.evidence import Evidence
from agenteval.skills.base import Presentation, Skill, SkillContext
from agenteval.tools import physics as P
from agenteval.tools import renders as R

SYSTEM = """\
你在评估一段 **AI 生成视频**是否遵守**基本物理与常识**。

检查这几类违反:

1. **重力与弹道**:下落物体是否匀加速下落?有没有在空中停住、凭空上升、
   或者以完全错误的速度下落。
2. **刚体保持**:本该是刚体的东西(桌椅、车辆、餐具、建筑)有没有在过程中
   融化、拉伸、弯折、流动。柔性物体(布料、头发、水、烟)本来就该形变,不算。
3. **穿模与接触**:物体有没有互相穿透?站/放在表面上的东西是否真的被支撑,
   还是悬浮在空中?接触点是否成立。
4. **物体恒存**:东西有没有凭空出现或消失?离开画面又回来时是否还是同一个。
5. **因果**:有没有无接触就发生的运动?有没有该发生的后果没有发生
   (碰了却没动、倒了却没洒)。

你**不要**判断:动作自不自然(motion_quality)、纹理和闪烁(temporal_integrity)、
人体结构(human_integrity)、是否符合提示词(semantic_conformance)。

**关于工具数值的重要提醒**:
弹道检验、刚体检验都建立在特征点跟踪之上。跟踪可能漂到背景上,
那会产生和"物体真的在形变"一模一样的数字。
而且这些检验各有前提——弹道检验只在物体确实处于自由落体时成立
(被托着、被吊着、本身会飞的都不适用);刚体检验只对刚体成立。
所以**先看画面确认跟踪的是什么、它当时处于什么状态**,再决定数值有没有意义。
"""


class PhysicalIntegrity(Skill):
    name = "physical_integrity"
    dimension = "physical_integrity"
    max_rounds = 5
    presentation = Presentation.ORDERED
    max_images = 14

    def __init__(self, out_dir: str | Path, *, max_frames: int | None = None) -> None:
        self.out_dir = Path(out_dir)
        self.max_frames = max_frames

    @property
    def system_prompt(self) -> str:
        return SYSTEM

    def seed(self, ctx: SkillContext) -> list[Evidence]:
        """Motion curves plus a whole-clip sequence: enough to see what moves
        and pick a region worth tracking, without committing budget to a track
        that may be of the wrong thing."""
        curves = ctx.bus.get_or_run(
            "motion_curves", "1.0",
            lambda: R.motion_curves(ctx.video, self.out_dir,
                                    max_frames=self.max_frames))
        seq = ctx.bus.get_or_run(
            "ordered_frames", "1.0",
            lambda: R.ordered_frames(ctx.video, self.out_dir, t0=0,
                                     t1=ctx.video.total, n=8, tag="phys_all"),
            t0=0, t1=ctx.video.total, n=8)
        return [curves, seq]

    def actions(self, ctx: SkillContext) -> list[Action]:
        fps = ctx.video.fps or 24.0

        def track_region(bbox: list, t0: int, t1: int) -> Evidence:
            """Track, then immediately hand back both the numbers and a view of
            the same region, so the tracking can be checked by eye."""
            bb = tuple(float(v) for v in bbox[:4])
            a, b = int(t0), int(t1)
            tr = ctx.bus.get_or_run(
                "track_points", "1.0",
                lambda: P.track_points(ctx.video, bbox=bb, t_span=(a, b)),
                bbox=[round(v, 3) for v in bb], t0=a, t1=b)
            rig = ctx.bus.get_or_run(
                "rigidity_check", "1.0",
                lambda: P.rigidity_check(tr.result),
                bbox=[round(v, 3) for v in bb], t0=a, t1=b)
            ctx.bus.get_or_run(
                "freefall_check", "1.0",
                lambda: P.freefall_check(tr.result, fps=fps),
                bbox=[round(v, 3) for v in bb], t0=a, t1=b)
            ctx.bus.get_or_run(
                "roi_sequence", "1.0",
                lambda: R.filmstrip(ctx.video, self.out_dir, t0=a, t1=b,
                                    bbox=bb, n=6, cols=6, tag=f"phys_{a}"),
                bbox=[round(v, 3) for v in bb], t0=a, t1=b)
            return rig

        def trajectory(bbox: list, t0: int, t1: int) -> Evidence:
            bb = tuple(float(v) for v in bbox[:4])
            a, b = int(t0), int(t1)
            tr = ctx.bus.get_or_run(
                "track_points", "1.0",
                lambda: P.track_points(ctx.video, bbox=bb, t_span=(a, b)),
                bbox=[round(v, 3) for v in bb], t0=a, t1=b)
            tracks = tr.result.value.get("tracks") or []
            if not tracks:
                raise ValueError("该区域无法跟踪,换一个纹理更明显的区域")
            import numpy as np
            centroid = np.asarray(tracks, float).mean(axis=0).tolist()
            return ctx.bus.get_or_run(
                "trajectory_plot", "1.0",
                lambda: R.trajectory_plot(centroid, self.out_dir, fps=fps,
                                          tag=f"traj_{a}"),
                bbox=[round(v, 3) for v in bb], t0=a, t1=b)

        def inspect(t0: int, t1: int, n: int = 8) -> Evidence:
            return ctx.bus.get_or_run(
                "ordered_frames", "1.0",
                lambda: R.ordered_frames(ctx.video, self.out_dir, t0=int(t0),
                                         t1=int(t1), n=int(n), tag=f"phys_seq_{t0}"),
                t0=int(t0), t1=int(t1), n=int(n))

        return [
            Action("track_region",
                   "在指定区域跟踪特征点,返回刚体保持性与自由落体检验,"
                   "同时给出该区域的裁剪序列供你核查跟踪对象",
                   {"bbox": "list[float]: 归一化 x,y,w,h",
                    "t0": "int: 起始帧", "t1": "int: 结束帧"}, run=track_region),
            Action("trajectory",
                   "画出被跟踪物体的位置与速度曲线,用于判断弹道是否符合重力",
                   {"bbox": "list[float]: 归一化 x,y,w,h",
                    "t0": "int: 起始帧", "t1": "int: 结束帧"}, run=trajectory),
            Action("inspect",
                   "取某个时间窗的顺序帧,用于判断接触、支撑、穿模与因果",
                   {"t0": "int: 起始帧", "t1": "int: 结束帧",
                    "n": "int: 取几帧,默认 8"}, run=inspect),
        ]
