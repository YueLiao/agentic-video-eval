"""temporal_integrity — does the video hold together over time?

Judges stutter, flicker, popping, texture crawl, identity/appearance switches
and other change that the scene's own motion does not explain. Deliberately
*not* about whether the video matches its prompt; that is conformance's job,
and mixing the two is what produces one uninterpretable number.

This skill is the clearest case for search: its defects are transient and
localized, so which frames and which region you look at decides the answer
before the judge is even asked. Its seed evidence is therefore the suspicion
map, and its action menu is entirely about where to look next.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agenteval.engine.actions import Action
from agenteval.engine.evidence import Evidence
from agenteval.signals import suspicion
from agenteval.skills.base import Skill, SkillContext
from agenteval.tools import views
from agenteval.tools.base import ToolResult

SYSTEM = """\
你在评估一段**AI 生成视频**的时间自洽性(temporal integrity)。

你要找的是**这段视频自己解释不了的变化**,例如:
- 卡顿/抽帧/画面冻结(该动的时候不动)
- 闪烁、亮度或色彩的突跳
- 纹理沸腾/爬行(静止表面上细节在不停重排)
- 物体或人物凭空出现、消失、瞬移
- 同一主体的外观/身份在中途变了

你**不要**判断:视频是否符合提示词、构图好不好、美不美。那是别的维度的事。

重要背景:AI 生成视频的瑕疵通常是**瞬态且局部**的——只持续几帧、只发生在画面的
一小块区域。整段视频的缩略图看不出来。系统已经用信号处理给你标出了若干**可疑点**
(suspicion loci),每个带时间区间、区域和触发它的信号类型。你的工作是决定去看哪些、
怎么看,然后判断那里到底有没有问题。

信号含义:
- mc_residual: 用光流补偿后仍无法解释的残差(已回归掉运动幅度与图像梯度的影响)
- freeze: 变化能量显著低于局部中位数——卡顿/冻结的特征
- crawl: 在近静止表面上持续存在的残差——纹理沸腾
- flow_anomaly: 光流场散度/旋度异常——撕裂、突现突消
- softness: 用运动解释不了的模糊
- luma_jump: 整帧亮度/直方图突跳

注意:**可疑点只是线索,不是缺陷**。信号会被正常的快速运动、遮挡、真实的光照变化
触发。你的判断必须来自你在放大画面里**实际看到的东西**。
"""


class TemporalIntegrity(Skill):
    name = "temporal_integrity"
    dimension = "temporal_integrity"
    max_rounds = 4

    def __init__(self, out_dir: str | Path, *, max_loci: int = 8,
                 max_frames: int | None = None) -> None:
        self.out_dir = Path(out_dir)
        self.max_loci = max_loci
        self.max_frames = max_frames
        self._loci: list[suspicion.SuspicionLocus] = []

    @property
    def system_prompt(self) -> str:
        return SYSTEM

    # ---- seed ------------------------------------------------------------
    def seed(self, ctx: SkillContext) -> list[Evidence]:
        out: list[Evidence] = []

        def _suspicion() -> ToolResult:
            loci, _ = suspicion.suspicion_map(
                str(ctx.video.path), max_frames=self.max_frames,
                max_loci=self.max_loci)
            self._loci = loci
            return ToolResult(
                value={"n_loci": len(loci),
                       "loci": [l.to_json() for l in loci]},
                reliability=0.7, backend="suspicion_map",
                hint=("按可疑度排序的待查清单。score 是 -log10(该强度在本视频中出现的"
                      "比例),越大越异常。这些是线索,不是结论。"),
            )

        ev = ctx.bus.get_or_run("suspicion_map", "1.0", _suspicion,
                                max_loci=self.max_loci)
        out.append(ev)

        out.append(ctx.bus.get_or_run(
            "overview", "1.0",
            lambda: views.overview(ctx.video, self.out_dir, n=8), n=8))
        return out

    def _locus(self, locus_id: str) -> suspicion.SuspicionLocus:
        for l in self._loci:
            if l.locus_id == locus_id:
                return l
        raise KeyError(f"unknown locus {locus_id!r}; "
                       f"available: {[l.locus_id for l in self._loci]}")

    # ---- action menu -----------------------------------------------------
    def actions(self, ctx: SkillContext) -> list[Action]:
        def zoom_locus(locus_id: str, n: int = 6) -> Evidence:
            l = self._locus(locus_id)
            return ctx.bus.get_or_run(
                "zoom", "1.0",
                lambda: views.zoom(ctx.video, self.out_dir, bbox=l.bbox,
                                   t_span=l.t_span, n=int(n),
                                   tag=f"zoom_{locus_id}"),
                locus_id=locus_id, n=int(n))

        def contrast(locus_id: str) -> Evidence:
            l = self._locus(locus_id)
            return ctx.bus.get_or_run(
                "contrast_pair", "1.0",
                lambda: views.contrast_pair(ctx.video, self.out_dir,
                                            bbox=l.bbox, t_span=l.t_span),
                locus_id=locus_id)

        def window(t0: int, t1: int, n: int = 8) -> Evidence:
            return ctx.bus.get_or_run(
                "dense_window", "1.0",
                lambda: views.dense_window(ctx.video, self.out_dir,
                                           t0=int(t0), t1=int(t1), n=int(n)),
                t0=int(t0), t1=int(t1), n=int(n))

        ids = ", ".join(l.locus_id for l in self._loci) or "(none)"
        return [
            Action("zoom_locus",
                   f"把某个可疑点的区域按原生分辨率放大到 448px 逐帧看。可用: {ids}",
                   {"locus_id": "str: 可疑点 id", "n": "int: 看几帧,默认 6"},
                   cost=1.0, run=zoom_locus),
            Action("contrast",
                   "把可疑点区域与它在远处时刻的同一区域并排对照,"
                   "用来区分'真的出问题了'和'这段视频本来就这样'",
                   {"locus_id": "str: 可疑点 id"}, cost=1.0, run=contrast),
            Action("window",
                   "看某个时间窗内的连续整帧,用来定位事件发生的确切时刻",
                   {"t0": "int: 起始帧", "t1": "int: 结束帧",
                    "n": "int: 采几帧,默认 8"}, cost=1.0, run=window),
        ]
