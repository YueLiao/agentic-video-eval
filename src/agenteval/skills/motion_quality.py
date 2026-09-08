"""motion_quality — is the movement itself right?

A separate skill from temporal_integrity on purpose. That one asks whether the
pixels hold together frame to frame; this one asks whether the *movement* is
plausible: does it flow at a believable rate, does it accelerate the way mass
does, does a body move the way a body moves. A clip can be perfectly stable and
still move like nothing in the world does, and a clip full of texture crawl can
have entirely convincing motion. Scoring them together buries both.

The split shows up in the evidence policy. temporal_integrity composites moments
into one image, because its question is comparative. Motion cannot use that: a
grid discards timing, and a freeze looks exactly like a slow passage laid out
spatially. So this skill runs ORDERED -- frames in sequence, captioned with
frame number and timestamp -- backed by explicit curves of speed and
acceleration, which make rate legible in a way frames alone never are.
"""

from __future__ import annotations

from pathlib import Path

from agenteval.engine.actions import Action
from agenteval.engine.evidence import Evidence
from agenteval.skills.base import Presentation, Skill, SkillContext
from agenteval.tools import renders as R

SYSTEM = """\
你在评估一段 **AI 生成视频**的**运动质量**。

你要判断的是"动得对不对",具体分四层:

1. **运动幅度**:该动的东西动了吗?条件描述了动作却几乎静止,是严重问题;
   反过来,条件描述的是静态场景,那么画面变化少是正确的,不是缺陷。
2. **运动平滑性**:速度变化是否连续?有没有卡顿、抽帧、忽快忽慢的顿挫。
3. **运动合理性**:速度和加速度是否符合物体的质量与所处的物理环境?
   自由落体应当匀加速;重物不应瞬间启停;水面波纹应当向外传播而不是原地闪烁。
4. **动作自然度**:人和动物的动作是否符合生物力学?
   步态相位对不对、重心是否随支撑脚转移、肢体摆动是否有惯性、
   还是像被逐帧摆出来的木偶(生成视频最典型的"假动作")。

你**不要**判断:纹理是否沸腾、有没有闪烁、画质清不清晰、是否符合提示词的语义。
那些属于别的维度。你只管**运动本身**。

给你的证据有两类,读法不同:
- **顺序帧**:按时间先后给出,每帧标了帧号和时间戳。这是判断动作自然度的主要依据,
  请按顺序看,想象它们连起来播放是什么样。
- **运动曲线**:横轴帧号的速度/加速度图。它让"速率"变得可读——
  深谷=接近静止,尖峰=速度突变,高频锯齿=抖动。
  但曲线**只描述时间结构,不能判断动作是否自然**:
  一条平滑的曲线完全可能对应一个不合生物力学的假动作。曲线指路,判断靠帧。
"""


class MotionQuality(Skill):
    name = "motion_quality"
    dimension = "motion_quality"
    max_rounds = 4
    presentation = Presentation.ORDERED
    max_images = 14

    def __init__(self, out_dir: str | Path, *, max_frames: int | None = None) -> None:
        self.out_dir = Path(out_dir)
        self.max_frames = max_frames

    @property
    def system_prompt(self) -> str:
        return SYSTEM

    def seed(self, ctx: SkillContext) -> list[Evidence]:
        """Start from the motion curves plus a sequence spanning the whole clip:
        the curves say where the rate is odd, the sequence gives the baseline
        for what the movement generally looks like."""
        curves = ctx.bus.get_or_run(
            "motion_curves", "1.0",
            lambda: R.motion_curves(ctx.video, self.out_dir,
                                    max_frames=self.max_frames))
        seq = ctx.bus.get_or_run(
            "ordered_frames", "1.0",
            lambda: R.ordered_frames(ctx.video, self.out_dir, t0=0,
                                     t1=ctx.video.total, n=8, tag="seq_all"),
            t0=0, t1=ctx.video.total, n=8)
        return [curves, seq]

    def actions(self, ctx: SkillContext) -> list[Action]:
        def inspect(t0: int, t1: int, n: int = 8) -> Evidence:
            return ctx.bus.get_or_run(
                "ordered_frames", "1.0",
                lambda: R.ordered_frames(ctx.video, self.out_dir, t0=int(t0),
                                         t1=int(t1), n=int(n), tag=f"seq_{t0}_{t1}"),
                t0=int(t0), t1=int(t1), n=int(n))

        def trail(t0: int, t1: int) -> Evidence:
            return ctx.bus.get_or_run(
                "motion_trail", "1.0",
                lambda: R.motion_trail(ctx.video, self.out_dir,
                                       t0=int(t0), t1=int(t1)),
                t0=int(t0), t1=int(t1))

        def zoom_motion(bbox: list, t0: int, t1: int, n: int = 8) -> Evidence:
            bb = tuple(float(v) for v in bbox[:4])
            return ctx.bus.get_or_run(
                "ordered_frames_roi", "1.0",
                lambda: R.filmstrip(ctx.video, self.out_dir, t0=int(t0), t1=int(t1),
                                    bbox=bb, n=int(n), cols=int(n),
                                    tag=f"roi_{t0}_{t1}"),
                bbox=[round(v, 3) for v in bb], t0=int(t0), t1=int(t1), n=int(n))

        return [
            Action("inspect",
                   "取某个时间窗内的**顺序帧**细看,用于判断动作自然度与连贯性",
                   {"t0": "int: 起始帧", "t1": "int: 结束帧",
                    "n": "int: 取几帧,默认 8"}, cost=1.0, run=inspect),
            Action("trail",
                   "把一个时间窗的多个时刻叠加成轨迹图,一眼看出运动路径的形状"
                   "(均匀=平滑,断档=瞬移,堆叠=停滞)",
                   {"t0": "int: 起始帧", "t1": "int: 结束帧"}, cost=1.0, run=trail),
            Action("zoom_motion",
                   "把某个区域裁剪出来按时间横向排开,用于看局部肢体/物体的运动细节",
                   {"bbox": "list[float]: 归一化 x,y,w,h",
                    "t0": "int: 起始帧", "t1": "int: 结束帧",
                    "n": "int: 取几帧,默认 8"}, cost=1.0, run=zoom_motion),
        ]
