"""static_integrity — inspecting every frame for defects that need no context.

Half the defect taxonomy is visible in a single still: malformed hands, melted
objects, garbled text, impossible shadows, interpenetration. No temporal signal
finds these, because they do not change -- a motion-compensated residual is
blind to a defect that sits perfectly stably in the frame. And no CV detector
recognises "this chair has impossible geometry"; the VLM is the only detector
that generalises across the whole class.

So this skill inverts the usual arrangement: instead of a CV index pointing the
VLM at suspicious moments, the VLM *is* the index. It sweeps the clip, flags
frames and rough regions, and only then is resolution spent on confirming.

That is affordable only because the target is short video. A 5-15 s clip is
80-360 frames, and sampling by *visual change* rather than by frame count
reaches full perceptual coverage in single-digit sheets -- a static passage
holds dozens of near-identical frames that cost budget and teach nothing. The
selective-search assumption comes from long-video understanding and does not
transfer.

Cascade: sweep cheap and wide, then confirm narrow and expensive. A flag from
the sweep is a hypothesis, never a finding; the sweep resolution is deliberately
too low to settle anything, which is what keeps it cheap.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agenteval.engine.actions import Action
from agenteval.engine.evidence import Evidence
from agenteval.rubrics.taxonomy import rubric_for
from agenteval.skills.base import Presentation, Skill, SkillContext
from agenteval.tools import renders as R
from agenteval.tools import scan
from agenteval.tools.base import ToolResult

SYSTEM = """\
你在对一段 **AI 生成视频**做**逐帧结构检查**。

你要找的是**单帧内就能判定**的问题——它们不需要看前后帧:
- 手部畸形(多指少指、指节融合、朝向违反关节)
- 面部畸形(五官数量/比例错误、牙齿眼睛结构崩坏)
- 肢体错误(多肢缺肢、关节反折、与躯干连接错误)
- 物体结构崩坏(几何不成立、融化、对称性破坏,如车轮/窗格/餐具)
- 穿模(两个实体在同一位置同时存在)
- 无支撑悬浮(该接触表面却悬空)
- 文字错乱(字形不成立)
- 光影矛盾(阴影方向与光源不符、缺少应有的影子)
- 塑料感/过度平滑、规则网格伪影

你**不要**判断:闪烁、卡顿、纹理沸腾、身份漂移、运动是否自然——
那些都需要跨帧比较,属于别的维度。**你只看每一帧自己站不站得住。**

## 工作方式(两段式,请严格遵守)

**第一段:粗筛。** 给你的接触印相表分辨率是**故意压低**的,只够定位不够定案。
所以在这个阶段你**只报告可疑的帧号和大致区域**,不要下结论、不要给严重度。
宁可多报几个可疑点(后面会逐个确认),也不要漏。

**第二段:确认。** 对可疑处调用放大动作,在原生分辨率下确认。
**只有放大确认过的才能写进最终 findings。** 粗筛阶段的怀疑不算数。

看不清就是看不清——报"证据不足",不要猜。
"""


class StaticIntegrity(Skill):
    name = "static_integrity"
    dimension = "visual_quality"
    max_rounds = 6
    presentation = Presentation.COMPOSITE
    max_images = 10

    def __init__(self, out_dir: str | Path, *, sweep_calls: int = 6,
                 batch: int = 9, max_gap: int = 12) -> None:
        self.out_dir = Path(out_dir)
        self.sweep_calls = sweep_calls
        self.batch = batch
        self.max_gap = max_gap
        self._plan: dict[str, Any] = {}

    @property
    def system_prompt(self) -> str:
        return SYSTEM + "\n" + rubric_for("appearance_integrity") + "\n\n" + \
            rubric_for("human_integrity", observability="frame")

    def seed(self, ctx: SkillContext) -> list[Evidence]:
        """Plan the sweep, then hand over the first sheets.

        The plan is reported as evidence in its own right so the judge knows how
        much of the clip it is actually seeing -- a verdict of "no defects" means
        something different at 100% perceptual coverage than at 40%.
        """
        plan = scan.adaptive_sweep_plan(
            ctx.video, budget_calls=self.sweep_calls, batch=self.batch,
            max_gap=self.max_gap)
        self._plan = plan

        cov = ctx.bus.put("sweep_plan", ToolResult(
            value={k: v for k, v in plan.items() if k != "batches"},
            reliability=1.0, backend="adaptive_sweep_plan",
            hint=(f"本次逐帧检查覆盖了 {plan.get('frame_coverage', 0):.0%} 的帧,"
                  f"但覆盖了 {plan.get('perceptual_coverage', 0):.0%} 的画面变化量"
                  "(采样按视觉变化分配,静止段少采、变化段密采)。"
                  "若感知覆盖率明显低于 100%,说明有未看到的内容,"
                  "结论中应当说明这一点。")))

        first = plan.get("batches") or [[0]]
        sheets = [ctx.bus.get_or_run(
            "contact_sheet", "1.0",
            lambda b=b: scan.contact_sheet(ctx.video, self.out_dir, b,
                                           tag=f"sweep{n}"),
            batch_index=n) for n, b in enumerate(first[:2])]
        return [cov, *sheets]

    def actions(self, ctx: SkillContext) -> list[Action]:
        batches = self._plan.get("batches") or []

        def next_sheet(batch_index: int) -> Evidence:
            i = int(batch_index)
            if not (0 <= i < len(batches)):
                raise ValueError(f"batch_index 必须在 0..{len(batches)-1}")
            return ctx.bus.get_or_run(
                "contact_sheet", "1.0",
                lambda: scan.contact_sheet(ctx.video, self.out_dir, batches[i],
                                           tag=f"sweep{i}"),
                batch_index=i)

        def tile(t: int, grid: int = 2) -> Evidence:
            g = max(2, min(3, int(grid)))
            return ctx.bus.get_or_run(
                "tile_frame", "1.0",
                lambda: scan.tile_frame(ctx.video, self.out_dir, t=int(t),
                                        grid=(g, g)),
                t=int(t), grid=g)

        def zoom(bbox: list, t: int) -> Evidence:
            bb = tuple(float(v) for v in bbox[:4])
            a = max(0, int(t) - 2)
            return ctx.bus.get_or_run(
                "zoom_confirm", "1.0",
                lambda: R.zoom_grid(ctx.video, self.out_dir, bbox=bb,
                                    t_span=(a, a + 5), n=4, cols=2,
                                    tag=f"conf_{t}"),
                bbox=[round(v, 3) for v in bb], t=int(t))

        def texture(t: int, bbox: list | None = None) -> Evidence:
            bb = tuple(float(v) for v in bbox[:4]) if bbox else None
            return ctx.bus.get_or_run(
                "highfreq_amplify", "1.0",
                lambda: R.highfreq_amplify(ctx.video, self.out_dir, t=int(t), bbox=bb),
                t=int(t), bbox=[round(v, 3) for v in bb] if bb else None)

        n = len(batches)
        return [
            Action("next_sheet",
                   f"取下一批帧的接触印相表继续粗筛。共 {n} 批,索引 0..{max(0,n-1)}",
                   {"batch_index": "int: 批次索引"}, run=next_sheet),
            Action("tile",
                   "把某一帧切成 2x2 或 3x3 分块放大,用于在不知道具体位置时"
                   "查找画面任意处的细小结构问题",
                   {"t": "int: 帧号", "grid": "int: 2 或 3,默认 2"}, run=tile),
            Action("zoom",
                   "把指定帧的指定区域放大到原生分辨率确认。"
                   "**写 findings 之前必须先用它确认过**",
                   {"bbox": "list[float]: 归一化 x,y,w,h", "t": "int: 帧号"},
                   run=zoom),
            Action("texture",
                   "高频放大视图,用于判断过度平滑(塑料感)或规则网格伪影",
                   {"t": "int: 帧号", "bbox": "list[float]: 可选区域"}, run=texture),
        ]
