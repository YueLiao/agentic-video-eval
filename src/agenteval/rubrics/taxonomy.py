"""Defect taxonomy: what counts as what, and how bad it is.

The system prompts described dimensions but never defined the defects, which
left two holes. A judge with no shared vocabulary invents its own labels, so
findings cannot be counted or compared across clips. And with no severity
anchors, "major" means whatever the model felt at the time, which is exactly the
uncalibrated absolute judgement the scoring design is trying to avoid.

Each entry therefore fixes four things:

``definition``     what the defect is, in one sentence
``confusable``     what it is NOT -- the legitimate phenomenon it gets mistaken
                   for. This is where false positives come from, so it is stated
                   for every defect rather than left implicit.
``anchors``        concrete descriptions of each grade for *this* defect, not a
                   generic scale
``observability``  ``frame`` if one still is enough, ``sequence`` if it can only
                   be seen across frames

Grades are a five-step ladder mapping directly onto scores, rather than a free
1-10. Asked for a continuous score a judge picks the middle and stays there --
in the first live run every one of fourteen findings came back at exactly 0.5
confidence, and across 420 scored aspects only three distinct values ever
appeared. A small set of named steps, each tied to what would have to be visible
for that step to apply, gives the judgement something to attach to.

    grade    score   what it means
    clean     10     nothing found
    trace      8     visible only on deliberate frame-by-frame inspection
    minor      6     noticeable on attentive viewing, does not disrupt
    major      4     obvious on normal viewing, hurts the result
    severe     2     ruins the shot, or makes it unusable for its purpose

The judge never picks a score. It picks a grade, from per-defect criteria, and
the score follows -- so the mapping stays fixed while the criteria stay specific
to the defect.

``observability`` is the load-bearing field. A malformed hand is wrong in a
single still and needs resolution, not context; flicker is invisible in any
single still however large. They demand different inspection strategies, and
treating every defect as temporal -- which is the easy mistake -- means the
static half is hunted with the wrong tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Observability = Literal["frame", "sequence"]

#: Five grades, and the score each one anchors to. Even numbers on purpose: the
#: gaps are where the other axes move the final number, so the anchors stay
#: legible while the score stays continuous.
#:
#: Resolution comes from **more axes, not more steps on one axis**. Subdividing
#: severity would not help: measured over 55 findings the judge already put 47
#: of them (85%) in `major` and only 3 in `minor`, so it is not using the four
#: steps it has, and a free 1-10 scale earlier produced 0.5 on every single
#: finding. Both are the same failure -- a single fine-grained dimension gives a
#: judge nowhere to stand, so it retreats to the middle. Several coarse
#: judgements it can actually make, combined, give far better resolution than
#: one fine one it cannot.
GRADES: tuple[str, ...] = ("trace", "minor", "major", "severe")
GRADE_SCORE: dict[str, float] = {
    "clean": 10.0, "trace": 8.0, "minor": 6.0, "major": 4.0, "severe": 2.0,
}
GRADE_DESC: dict[str, str] = {
    "trace": "只有逐帧仔细看才能发现;正常观看完全不影响",
    "minor": "认真观看时能注意到,但不影响整体观感",
    "major": "正常观看一眼可见,明显损害成片质量",
    "severe": "毁掉这个镜头,或使视频无法用于其预期用途",
}

#: Second axis: how much of the clip's duration the defect occupies. Asked as a
#: category rather than derived from `t_span`, because the judge's frame numbers
#: are approximate -- it reports the same defect as @0-5, @1-9, @1-10 across
#: sampling phases -- while "a flash" versus "most of the clip" is a judgement it
#: makes reliably.
EXTENT: dict[str, float] = {
    "flash": 0.35,      # 一两帧,一闪而过
    "brief": 0.60,      # 一小段,不到片长的四分之一
    "recurring": 0.85,  # 反复出现,或断续贯穿
    "throughout": 1.0,  # 几乎全程持续
}
EXTENT_DESC: dict[str, str] = {
    "flash": "只在一两帧出现,一闪而过",
    "brief": "持续一小段时间(不到片长四分之一)",
    "recurring": "反复出现,或断断续续贯穿全片",
    "throughout": "几乎全程持续存在",
}

#: Third axis: whether it lands where the viewer is looking. A malformed hand on
#: the subject and the same malformation on a background extra are not the same
#: failure, and no severity grade can express that difference.
SALIENCE: dict[str, float] = {
    "peripheral": 0.45,  # 画面边缘/背景次要处
    "secondary": 0.75,   # 次要主体,或主体的非焦点部位
    "primary": 1.0,      # 画面主体上,观众正在看的地方
}
SALIENCE_DESC: dict[str, str] = {
    "peripheral": "位于画面边缘或背景,观众通常不会注意",
    "secondary": "位于次要主体上,或主体的非焦点部位",
    "primary": "就在画面主体上,观众视线所在之处",
}


def normalize_extent(v: str | None) -> str:
    k = (v or "").strip().lower()
    return k if k in EXTENT else "brief"


def normalize_salience(v: str | None) -> str:
    k = (v or "").strip().lower()
    return k if k in SALIENCE else "secondary"

#: Legacy three-grade names still emitted by judges and stored in old runs.
_ALIAS: dict[str, str] = {"critical": "severe", "moderate": "minor",
                          "low": "trace", "high": "major"}


def normalize_grade(g: str | None) -> str:
    """Map whatever the judge said onto the five-step ladder."""
    if not g:
        return "minor"
    k = str(g).strip().lower()
    k = _ALIAS.get(k, k)
    return k if k in GRADE_SCORE else "minor"


@dataclass(frozen=True)
class DefectSpec:
    key: str
    dimension: str
    label_zh: str
    definition: str
    confusable: str
    anchors: dict[str, str]
    observability: Observability
    cues: tuple[str, ...] = ()          # which tools/signals tend to surface it

    def render(self) -> str:
        a = self.anchors
        lines = [f"### {self.key} ({self.label_zh})",
                 f"- 定义:{self.definition}",
                 f"- **不要混淆**:{self.confusable}"]
        for g in GRADES:
            if a.get(g):
                lines.append(f"- `{g}`({GRADE_SCORE[g]:.0f}分):{a[g]}")
        return "\n".join(lines)


_D: list[DefectSpec] = [
    # ---- human_integrity ------------------------------------------------
    DefectSpec(
        "hand_malformation", "human_integrity", "手部畸形",
        "手指数量错误、指节融合粘连、手指朝向违反关节结构,或手掌整体结构崩坏。",
        "手指被物体或身体遮挡而看不全,不算畸形;运动模糊导致手指边界模糊,也不算。"
        "只有在能看清的情况下结构本身就错了,才算。",
        {"trace": "放大后细看才发现某根手指边界略有粘连,整体仍是正常的手",
         "minor": "能数出手指数量或形态有误,但只出现在个别帧或手部较小时",
         "major": "手指数量明显错误或指节反向融合,正常观看即可看出这不是一只正常的手",
         "severe": "手掌整体崩坏成不可辨认的团块,或畸形贯穿大部分镜头"},
        "frame", ("hands_detect", "zoom_hands")),
    DefectSpec(
        "face_malformation", "human_integrity", "面部畸形",
        "五官位置/数量/比例错误,眼睛不对称到非自然程度,牙齿或耳朵结构崩坏。",
        "侧脸、俯仰角、广角镜头畸变都会改变五官比例,那是正常透视,不是缺陷。"
        "人脸太小看不清时不得下结论。",
        {"trace": "放大后发现轻微不对称或牙齿细节略糊",
         "minor": "五官比例或牙齿/眼睛结构有误,但需留意才能察觉",
         "major": "五官比例明显失常,正常观看即觉得脸不对",
         "severe": "面部结构崩坏,不再像一张人脸"},
        "frame", ("face_detect", "zoom_face")),
    DefectSpec(
        "limb_deformation", "human_integrity", "肢体形变",
        "四肢长度在时间上变化、关节角度超出人类范围、肢体数量错误或与躯干连接错误。",
        "**透视缩短**:肢体朝向或背离镜头旋转时,2D 投影长度本来就会变,这不是形变。"
        "遮挡导致的肢体消失也不是。",
        {"trace": "单帧内肢体长度或角度略有异常,可用透视解释一部分",
         "minor": "确认存在长度变化或角度越界,但幅度小或持续时间短",
         "major": "肢体明显伸缩或关节反折,持续可见",
         "severe": "多肢、缺肢,或肢体与躯干连接完全错误"},
        "frame", ("pose_detect", "anatomy_invariants", "skeleton_overlay")),
    DefectSpec(
        "identity_drift", "human_integrity", "身份漂移",
        "同一人物的面部特征、发型、衣着在片段中途改变。",
        "光照变化、转身、换角度导致的外观差异不算;有意的换装/换人如果条件里要求了也不算。",
        {"trace": "发型或衣着的细微细节有变化",
         "minor": "面部特征有可察觉的变化,但仍认得出是同一人",
         "major": "外观明显改变,正常观看会觉得换了个人的状态",
         "severe": "中途完全变成另一个人"},
        "sequence", ("face_track", "identity_drift", "ab_contrast")),
    DefectSpec(
        "plastic_texture", "human_integrity", "塑料感/过度平滑",
        "皮肤、毛发失去应有的纹理层次,呈现均匀无孔的塑料或蜡质外观。",
        "柔光、美颜风格、动画/CG 风格本身就平滑,若条件要求了该风格则不算缺陷。"
        "远景人物细节少是正常的。",
        {"trace": "皮肤纹理略少,但仍有细节层次",
         "minor": "局部明显平滑无毛孔,近看有蜡感",
         "major": "人物皮肤整体呈塑料/蜡质感,正常观看即可察觉",
         "severe": "人物如同塑料模型,完全失去材质真实感"},
        "frame", ("highfreq_amplify", "zoom_face")),
    DefectSpec(
        "no_blink", "human_integrity", "无眨眼",
        "有清晰人脸特写且时长足够,但整段中人物完全不眨眼。",
        "片段过短(<3s)、人脸太小、或人物本就闭眼/背对时不适用。",
        {"trace": "眨眼频率偏低",
         "minor": "4 秒以上无眨眼",
         "major": "全程无眨眼且眼神明显呆滞",
         "severe": "全程无眨眼且面部完全静止,呈死板假人感"},
        "sequence", ("face_detect", "zoom_face")),

    # ---- temporal_integrity ---------------------------------------------
    DefectSpec(
        "flicker", "temporal_integrity", "闪烁",
        "亮度、颜色或局部内容在相邻帧间反复跳变。",
        "真实光源闪烁(霓虹、火焰、屏幕)是内容本身,不是缺陷;有意的频闪同理。",
        {"trace": "逐帧看才发现的局部微弱亮度波动",
         "minor": "留意时能看到画面轻微明暗跳动",
         "major": "正常观看即可见的明显闪烁",
         "severe": "全片持续强烈闪烁,严重影响观看"},
        "sequence", ("luma_jump", "filmstrip")),
    DefectSpec(
        "texture_crawl", "temporal_integrity", "纹理沸腾",
        "静止或缓慢移动表面上的细节在帧间不断重排,像在沸腾。",
        "真实的水面、火焰、树叶抖动是内容;摄像机噪点也不是。"
        "关键判据是:**该表面本身应当静止**。",
        {"trace": "小面积纹理有极轻微蠕动",
         "minor": "局部表面细节持续重排,留意可见",
         "major": "明显区域持续沸腾,正常观看即可察觉",
         "severe": "大面积纹理崩坏重排,画面失去稳定感"},
        "sequence", ("crawl", "mc_residual", "filmstrip")),
    DefectSpec(
        "stutter", "temporal_integrity", "卡顿/抽帧",
        "画面出现重复帧、丢帧或时间上的不连续跳跃。",
        "条件要求的定格、慢动作、静止镜头都不是卡顿。"
        "**该动的东西不动**才是,本来就静止的场景不算。",
        {"trace": "单次极短的顿挫,几乎不可察",
         "minor": "有可察觉的卡顿或重复帧",
         "major": "多次明显卡顿或时间跳跃,破坏动作连贯",
         "severe": "频繁卡顿使动作无法连贯辨识"},
        "sequence", ("freeze", "duplicate_frames", "motion_curves")),
    DefectSpec(
        "object_pop", "temporal_integrity", "物体突现/突消",
        "物体在没有遮挡或出画的情况下凭空出现或消失。",
        "被遮挡、移出画面、进入阴影都是正常的。渐进的淡入淡出也不算。",
        {"trace": "背景中极次要的小物件短暂消失",
         "minor": "次要物体凭空出现或消失,需留意才发现",
         "major": "可辨识物体凭空出现或消失,正常观看可见",
         "severe": "主体凭空出现或消失"},
        "sequence", ("flow_anomaly", "permanence_check")),

    # ---- motion_quality --------------------------------------------------
    DefectSpec(
        "motion_stall", "motion_quality", "运动停滞",
        "条件描述了运动,但主体在片段中几乎不动或运动幅度远小于应有。",
        "条件本身描述静态场景时不适用;镜头运动与主体运动要分开判断。",
        {"trace": "运动幅度略小于条件描述",
         "minor": "运动明显偏弱,但仍在发生",
         "major": "主体基本静止,与条件描述明显不符",
         "severe": "整段近乎静止,完全未执行要求的动作"},
        "sequence", ("motion_curves", "ordered_frames")),
    DefectSpec(
        "unnatural_gait", "motion_quality", "动作不自然",
        "人或动物的动作违反生物力学:步态相位错误、重心不随支撑脚转移、"
        "肢体摆动无惯性、像被逐帧摆出来的木偶。",
        "舞蹈、武术、卡通风格的夸张动作若条件允许则不算;"
        "慢动作下动作显得'飘'也是正常的。",
        {"trace": "个别动作略显生硬",
         "minor": "部分动作不符合真实运动规律,留意可见",
         "major": "整体动作明显违反生物力学,正常观看即觉得假",
         "severe": "动作完全脱离生物力学,呈提线木偶状"},
        "sequence", ("ordered_frames", "motion_trail")),
    DefectSpec(
        "speed_anomaly", "motion_quality", "速度异常",
        "运动速度突变、忽快忽慢,或与物体质量/场景明显不符。",
        "条件要求的变速、真实的加减速不算。",
        {"trace": "速度有极轻微波动",
         "minor": "存在可察觉的速度不均",
         "major": "明显的速度突变或不合理的匀速",
         "severe": "速度完全失控,动作无法辨识"},
        "sequence", ("motion_curves",)),

    # ---- physical_integrity ---------------------------------------------
    DefectSpec(
        "gravity_violation", "physical_integrity", "重力违反",
        "下落物体不做匀加速运动、在空中停滞、或无支撑地上升。",
        "有支撑、被牵引、本身会飞、或处于水中/失重场景时不适用。",
        {"trace": "下落轨迹略不自然",
         "minor": "下落速度或轨迹有可察觉的异常",
         "major": "物体在空中明显停顿,或轨迹明显违反重力",
         "severe": "物体无支撑悬浮或凭空上升"},
        "sequence", ("freefall_check", "trajectory_plot")),
    DefectSpec(
        "rigidity_violation", "physical_integrity", "刚体形变",
        "本应刚性的物体(家具、车辆、餐具、建筑)在运动中拉伸、弯折、融化或流动。",
        "**柔性物体本来就该形变**:布料、头发、水、烟、植物均不适用。"
        "透视变化造成的形状改变也不是形变。",
        {"trace": "刚体边缘略有不稳",
         "minor": "刚体有可察觉的轻微弯折或伸缩",
         "major": "刚体明显弯折、伸缩或流动",
         "severe": "物体融化流动,完全失去原有结构"},
        "sequence", ("rigidity_check", "zoom_grid")),
    DefectSpec(
        "interpenetration", "physical_integrity", "穿模",
        "两个实体互相穿透,或肢体穿过自身/物体。",
        "遮挡不是穿模——判据是能看到两者在同一空间位置**同时存在**。"
        "半透明物体、影子、倒影不适用。",
        {"trace": "极短暂且极轻微的边缘穿插",
         "minor": "有可察觉的物体互穿,但范围小",
         "major": "明显的物体互穿,持续可见",
         "severe": "主体大幅穿过实体"},
        "frame", ("depth_order", "zoom_grid")),
    DefectSpec(
        "unsupported_float", "physical_integrity", "无支撑悬浮",
        "物体或人物应当接触地面/表面却悬空,或接触点不成立。",
        "跳跃、飞行、游泳、条件允许的超现实场景不适用。支撑物在画外时不得判定。",
        {"trace": "接触点略有偏差",
         "minor": "脚部/底部与支撑面有可察觉的错位",
         "major": "明显悬空或穿入地面",
         "severe": "主体完全漂浮于空中"},
        "frame", ("pose_detect", "ordered_frames")),

    # ---- appearance_integrity -------------------------------------------
    DefectSpec(
        "structure_collapse", "appearance_integrity", "结构崩坏",
        "物体失去可辨识的结构,呈现为不连贯的色块或畸形几何。",
        "景深虚化、远景细节缺失、有意的抽象风格不算。",
        {"trace": "背景次要物体轮廓略糊",
         "minor": "次要物体结构有可察觉的错误",
         "major": "可辨识物体结构明显错误,正常观看可见",
         "severe": "主体结构崩坏,不可辨认"},
        "frame", ("zoom_grid", "highfreq_amplify")),
    DefectSpec(
        "repeated_pattern", "appearance_integrity", "重复花纹",
        "画面中出现规则的网格、棋盘或不自然的重复纹理。",
        "真实的重复图案(瓷砖、格纹衣物、建筑立面)是内容,不是缺陷。",
        {"trace": "放大后才见的局部规则纹理",
         "minor": "局部可察觉的网格或重复花纹",
         "major": "明显的棋盘/网格伪影,正常观看可见",
         "severe": "大面积规则伪影覆盖画面"},
        "frame", ("highfreq_amplify", "spectral_signature")),
    DefectSpec(
        "defect_blur", "appearance_integrity", "非运动性模糊",
        "在低运动区域出现的、无法用运动或景深解释的模糊/软化。",
        "**运动模糊在快速运动处是物理正确的**;景深虚化是摄影意图;"
        "低光噪点不是模糊。",
        {"trace": "极小面积的轻微软化",
         "minor": "局部模糊可察觉,且无运动可解释",
         "major": "可辨识区域明显模糊且无运动解释",
         "severe": "主体持续模糊不可辨"},
        "frame", ("softness", "blur_diagnosis")),
    DefectSpec(
        "lighting_inconsistency", "appearance_integrity", "光影矛盾",
        "阴影方向与光源不一致、物体缺少应有的影子或反射、局部光照违反场景。",
        "多光源场景、间接光、有意的风格化打光不算。",
        {"trace": "阴影方向有极细微偏差",
         "minor": "阴影或反射有可察觉的不一致",
         "major": "明显的阴影缺失或方向矛盾",
         "severe": "光照逻辑完全崩坏"},
        "frame", ("zoom_grid",)),
    DefectSpec(
        "text_garbled", "appearance_integrity", "文字错乱",
        "画面中的文字字形不成立、拼写错乱,或在帧间变化。",
        "远景、小字、非拉丁/中文的装饰性字符不适用;有意的模糊招牌不算。",
        {"trace": "小字略有变形,不影响识别",
         "minor": "文字有可察觉的字形问题",
         "major": "可读位置的文字明显错乱",
         "severe": "条件要求的文字完全无法辨认或内容错误"},
        "frame", ("ocr", "zoom_grid")),
]

BY_KEY: dict[str, DefectSpec] = {d.key: d for d in _D}
BY_DIMENSION: dict[str, list[DefectSpec]] = {}
for _d in _D:
    BY_DIMENSION.setdefault(_d.dimension, []).append(_d)

FRAME_DEFECTS = tuple(d.key for d in _D if d.observability == "frame")
SEQUENCE_DEFECTS = tuple(d.key for d in _D if d.observability == "sequence")


def rubric_for(dimension: str, *, observability: Observability | None = None) -> str:
    """Render this dimension's rubric for injection into a system prompt."""
    specs = BY_DIMENSION.get(dimension, [])
    if observability:
        specs = [s for s in specs if s.observability == observability]
    if not specs:
        return ""
    keys = "、".join(s.key for s in specs)
    return ("以下是本维度的缺陷清单。**`kind` 字段必须使用这里的 key**"
            f"(可选值:{keys}),不要自创名称。\n"
            "每类都写明了它容易被误认成什么——报告前先排除那些正常情况。\n\n"
            + "\n\n".join(s.render() for s in specs))


def all_keys() -> tuple[str, ...]:
    return tuple(BY_KEY)
