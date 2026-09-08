"""human_integrity — are the people in this video put together correctly?

The dimension where VLMs are weakest unaided and where the fix is most purely
one of presentation. Faces and hands are small: a face at 6% of frame width,
inside a VLM's image budget, is a few dozen pixels, and six fingers are simply
not resolvable there. The model is not failing to notice, it is being asked to
judge evidence that is absent. Crop at native resolution and the same model
reads the same defect immediately.

So this skill is built around getting the model close enough to look, and
around giving it a reason to look at one place rather than another. The
geometric invariants supply that reason: bone lengths that shift, limbs that
lose bilateral symmetry, keypoint confidence collapsing on particular frames.
None of those convict on their own -- a limb rotating toward the camera
shortens in projection exactly like a limb that deforms -- but they turn a
whole clip into three or four places worth zooming into.
"""

from __future__ import annotations

from pathlib import Path

from agenteval.engine.actions import Action
from agenteval.engine.evidence import Evidence
from agenteval.media.clip import uniform_indices
from agenteval.skills.base import Presentation, Skill, SkillContext
from agenteval.tools import detectors as D
from agenteval.tools import renders as R

SYSTEM = """\
你在评估一段 **AI 生成视频**中**人物的结构完整性**。

按以下优先级检查:

1. **手**(生成模型最常崩的部位):手指数量对不对、有没有粘连融合、
   指节朝向是否可能、手与被握持物体的接触是否成立。
2. **脸**:五官比例在时间上是否稳定、牙齿和眼睛的细节是否成立、
   有没有过度平滑造成的塑料感;真人会眨眼,整段完全不眨眼是常见破绽。
3. **身体结构**:四肢长度是否恒定、左右是否大致对称、关节角度是否在人类范围内、
   躯干与四肢的连接是否成立。
4. **身份一致性**:同一个人的长相/发型/衣着在整段里是否保持一致,
   有没有中途换脸、属性突变。
5. **人数稳定性**:画面里的人数是否会莫名增减。

你**不要**判断:动作是否自然流畅(那是 motion_quality)、纹理沸腾与闪烁
(那是 temporal_integrity)、是否符合提示词语义。你只管**人本身长得对不对**。

**关于工具数值的重要提醒**:
骨长变异系数、左右不对称度都是从 2D 投影关键点算出来的。
肢体朝向或背离镜头旋转时会发生透视缩短,同样会让这些数字变大,而那是完全正常的。
所以这些数字的作用是**告诉你该放大看哪根肢体**,不是判决。
同理,关键点置信度骤降只说明"检测器在这一帧认不出人体",可能是遮挡,也可能是崩坏。
**必须裁剪放大后用眼睛确认。看不清就报证据不足,不要猜。**
"""


class HumanIntegrity(Skill):
    name = "human_integrity"
    dimension = "human_integrity"
    covers = ("hand_structure", "face_structure", "limb_structure",
                              "identity_consistency", "skin_texture", "eye_behavior")
    max_rounds = 5
    presentation = Presentation.COMPOSITE
    max_images = 12

    def __init__(self, out_dir: str | Path, *, n_probe: int = 12) -> None:
        self.out_dir = Path(out_dir)
        self.n_probe = n_probe
        self._idx: list[int] = []
        self._pose = None
        self._hands = None
        self._face = None

    @property
    def system_prompt(self) -> str:
        return SYSTEM

    def applies(self, ctx: SkillContext) -> bool:
        """Skip entirely when there is no person. A skill that cannot apply must
        not return a full score -- that would silently reward clips for lacking
        the hardest content."""
        if ctx.hints.get("has_human") is not None:
            return bool(ctx.hints["has_human"])
        idx = uniform_indices(ctx.video.total, 4)
        f = D.face_detect(ctx.video, idx)
        p = D.pose_detect(ctx.video, idx)
        return bool(f.value.get("n_frames_with_face") or p.value.get("n_persons_max"))

    def seed(self, ctx: SkillContext) -> list[Evidence]:
        self._idx = uniform_indices(ctx.video.total, self.n_probe)
        face = ctx.bus.get_or_run("face_detect", "1.0",
                                  lambda: D.face_detect(ctx.video, self._idx),
                                  n=self.n_probe)
        pose = ctx.bus.get_or_run("pose_detect", "1.0",
                                  lambda: D.pose_detect(ctx.video, self._idx),
                                  n=self.n_probe)
        hands = ctx.bus.get_or_run("hands_detect", "1.0",
                                   lambda: D.hands_detect(ctx.video, self._idx),
                                   n=self.n_probe)
        self._face, self._pose, self._hands = face, pose, hands
        anat = ctx.bus.get_or_run(
            "anatomy_invariants", "1.0",
            lambda: D.anatomy_invariants(pose.result, hands.result),
            n=self.n_probe)
        return [face, pose, hands, anat]

    def _face_bbox(self, frame_idx: int | None = None):
        per = (self._face.result.value.get("per_frame") or []) if self._face else []
        best, best_i = None, None
        for p in per:
            if frame_idx is not None and p["idx"] != frame_idx:
                continue
            for b in p["boxes"]:
                if best is None or b[2] > best[2]:
                    best, best_i = b, p["idx"]
        return (tuple(best), best_i) if best else (None, None)

    def _hand_bbox(self):
        per = (self._hands.result.value.get("per_frame") or []) if self._hands else []
        for p in per:
            for hd in p.get("hands", []):
                xs = [k[0] for k in hd["kpts"]]; ys = [k[1] for k in hd["kpts"]]
                pad = 0.04
                return ((max(0.0, min(xs) - pad), max(0.0, min(ys) - pad),
                         min(1.0, max(xs) - min(xs) + 2 * pad),
                         min(1.0, max(ys) - min(ys) + 2 * pad)), p["idx"])
        return (None, None)

    def actions(self, ctx: SkillContext) -> list[Action]:
        total = ctx.video.total

        def zoom_face(t0: int | None = None, t1: int | None = None) -> Evidence:
            bb, at = self._face_bbox()
            if bb is None:
                raise ValueError("未检出人脸,该动作不可用")
            a = int(t0) if t0 is not None else max(0, (at or 0) - 8)
            b = int(t1) if t1 is not None else min(total, (at or 0) + 8)
            return ctx.bus.get_or_run(
                "zoom_face", "1.0",
                lambda: R.zoom_grid(ctx.video, self.out_dir, bbox=bb,
                                    t_span=(a, b), n=6, cols=3, tag="face"),
                bbox=[round(v, 3) for v in bb], t0=a, t1=b)

        def zoom_hands(t0: int | None = None, t1: int | None = None) -> Evidence:
            bb, at = self._hand_bbox()
            if bb is None:
                raise ValueError("未检出手部,该动作不可用")
            a = int(t0) if t0 is not None else max(0, (at or 0) - 6)
            b = int(t1) if t1 is not None else min(total, (at or 0) + 6)
            return ctx.bus.get_or_run(
                "zoom_hands", "1.0",
                lambda: R.zoom_grid(ctx.video, self.out_dir, bbox=bb,
                                    t_span=(a, b), n=6, cols=3, tag="hand"),
                bbox=[round(v, 3) for v in bb], t0=a, t1=b)

        def skeleton() -> Evidence:
            return ctx.bus.get_or_run(
                "skeleton_overlay", "1.0",
                lambda: R.skeleton_overlay(
                    ctx.video, self.out_dir, pose_result=self._pose.result,
                    hands_result=self._hands.result, indices=self._idx[:8]),
                n=8)

        def zoom_region(bbox: list, t0: int, t1: int) -> Evidence:
            bb = tuple(float(v) for v in bbox[:4])
            return ctx.bus.get_or_run(
                "zoom_region", "1.0",
                lambda: R.zoom_grid(ctx.video, self.out_dir, bbox=bb,
                                    t_span=(int(t0), int(t1)), n=6, cols=3,
                                    tag=f"reg_{t0}"),
                bbox=[round(v, 3) for v in bb], t0=int(t0), t1=int(t1))

        def skin_texture(t: int, bbox: list | None = None) -> Evidence:
            bb = tuple(float(v) for v in bbox[:4]) if bbox else self._face_bbox()[0]
            return ctx.bus.get_or_run(
                "highfreq_amplify", "1.0",
                lambda: R.highfreq_amplify(ctx.video, self.out_dir, t=int(t), bbox=bb),
                t=int(t), bbox=[round(v, 3) for v in bb] if bb else None)

        return [
            Action("zoom_face", "把最大的人脸按原生分辨率裁剪放大,按时间排开细看",
                   {"t0": "int: 可选,起始帧", "t1": "int: 可选,结束帧"},
                   run=zoom_face),
            Action("zoom_hands", "把检出的手部裁剪放大,按时间排开细看手指结构",
                   {"t0": "int: 可选,起始帧", "t1": "int: 可选,结束帧"},
                   run=zoom_hands),
            Action("skeleton", "把姿态骨架画在画面上,用来核查关键点估计是否可信"
                               "(骨长统计异常时先用它判断是真变形还是点落错了)",
                   {}, run=skeleton),
            Action("zoom_region", "放大任意指定区域,用于看骨长异常的那根肢体",
                   {"bbox": "list[float]: 归一化 x,y,w,h",
                    "t0": "int: 起始帧", "t1": "int: 结束帧"}, run=zoom_region),
            Action("skin_texture", "高频放大视图,用于判断皮肤/毛发是否被过度平滑成塑料感",
                   {"t": "int: 帧号", "bbox": "list[float]: 可选区域,默认人脸"},
                   run=skin_texture),
        ]
