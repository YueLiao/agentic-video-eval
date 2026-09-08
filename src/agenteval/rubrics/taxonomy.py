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
``anchors``        concrete minor / major / critical descriptions for *this*
                   defect, not a generic scale
``observability``  ``frame`` if one still is enough, ``sequence`` if it can only
                   be seen across frames

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
        return (f"### {self.key} ({self.label_zh})\n"
                f"- 定义:{self.definition}\n"
                f"- **不要混淆**:{self.confusable}\n"
                f"- minor:{a.get('minor','')}\n"
                f"- major:{a.get('major','')}\n"
                f"- critical:{a.get('critical','')}")


_D: list[DefectSpec] = [
    # ---- human_integrity ------------------------------------------------
    DefectSpec(
        "hand_malformation", "human_integrity", "手部畸形",
        "手指数量错误、指节融合粘连、手指朝向违反关节结构,或手掌整体结构崩坏。",
        "手指被物体或身体遮挡而看不全,不算畸形;运动模糊导致手指边界模糊,也不算。"
        "只有在能看清的情况下结构本身就错了,才算。",
        {"minor": "单帧内某根手指边界不清或略有粘连,不影响整体识别",
         "major": "可数出手指数量错误,或明显的指节融合/反向,正常观看即可察觉",
         "critical": "手掌整体崩坏成不可辨认的团块,或长时间持续存在"},
        "frame", ("hands_detect", "zoom_hands")),
    DefectSpec(
        "face_malformation", "human_integrity", "面部畸形",
        "五官位置/数量/比例错误,眼睛不对称到非自然程度,牙齿或耳朵结构崩坏。",
        "侧脸、俯仰角、广角镜头畸变都会改变五官比例,那是正常透视,不是缺陷。"
        "人脸太小看不清时不得下结论。",
        {"minor": "细看才发现的轻微不对称或牙齿细节异常",
         "major": "五官比例明显失常或眼睛/牙齿结构错误,正常观看可察觉",
         "critical": "面部结构崩坏,不再像一张人脸"},
        "frame", ("face_detect", "zoom_face")),
    DefectSpec(
        "limb_deformation", "human_integrity", "肢体形变",
        "四肢长度在时间上变化、关节角度超出人类范围、肢体数量错误或与躯干连接错误。",
        "**透视缩短**:肢体朝向或背离镜头旋转时,2D 投影长度本来就会变,这不是形变。"
        "遮挡导致的肢体消失也不是。",
        {"minor": "短暂且轻微的长度或角度异常",
         "major": "肢体明显伸缩或关节反折,持续可见",
         "critical": "多肢、缺肢,或肢体与躯干连接完全错误"},
        "frame", ("pose_detect", "anatomy_invariants", "skeleton_overlay")),
    DefectSpec(
        "identity_drift", "human_integrity", "身份漂移",
        "同一人物的面部特征、发型、衣着在片段中途改变。",
        "光照变化、转身、换角度导致的外观差异不算;有意的换装/换人如果条件里要求了也不算。",
        {"minor": "细微的发型或衣着细节变化",
         "major": "面部特征或主要衣着明显改变,能看出不是同一状态",
         "critical": "中途完全换成另一个人"},
        "sequence", ("face_track", "identity_drift", "ab_contrast")),
    DefectSpec(
        "plastic_texture", "human_integrity", "塑料感/过度平滑",
        "皮肤、毛发失去应有的纹理层次,呈现均匀无孔的塑料或蜡质外观。",
        "柔光、美颜风格、动画/CG 风格本身就平滑,若条件要求了该风格则不算缺陷。"
        "远景人物细节少是正常的。",
        {"minor": "局部略显平滑但仍有纹理",
         "major": "皮肤明显无毛孔无细节,呈蜡质感",
         "critical": "人物整体呈塑料模型质感"},
        "frame", ("highfreq_amplify", "zoom_face")),
    DefectSpec(
        "no_blink", "human_integrity", "无眨眼",
        "有清晰人脸特写且时长足够,但整段中人物完全不眨眼。",
        "片段过短(<3s)、人脸太小、或人物本就闭眼/背对时不适用。",
        {"minor": "眨眼频率偏低", "major": "全程无眨眼且时长超过 4 秒",
         "critical": "全程无眨眼且伴随眼神完全静止的死板感"},
        "sequence", ("face_detect", "zoom_face")),

    # ---- temporal_integrity ---------------------------------------------
    DefectSpec(
        "flicker", "temporal_integrity", "闪烁",
        "亮度、颜色或局部内容在相邻帧间反复跳变。",
        "真实光源闪烁(霓虹、火焰、屏幕)是内容本身,不是缺陷;有意的频闪同理。",
        {"minor": "局部轻微亮度波动", "major": "画面可察觉的明暗或色彩跳动",
         "critical": "全片持续强烈闪烁,影响观看"},
        "sequence", ("luma_jump", "filmstrip")),
    DefectSpec(
        "texture_crawl", "temporal_integrity", "纹理沸腾",
        "静止或缓慢移动表面上的细节在帧间不断重排,像在沸腾。",
        "真实的水面、火焰、树叶抖动是内容;摄像机噪点也不是。"
        "关键判据是:**该表面本身应当静止**。",
        {"minor": "小面积轻微蠕动", "major": "明显区域持续沸腾,正常观看可见",
         "critical": "大面积纹理持续崩坏重排"},
        "sequence", ("crawl", "mc_residual", "filmstrip")),
    DefectSpec(
        "stutter", "temporal_integrity", "卡顿/抽帧",
        "画面出现重复帧、丢帧或时间上的不连续跳跃。",
        "条件要求的定格、慢动作、静止镜头都不是卡顿。"
        "**该动的东西不动**才是,本来就静止的场景不算。",
        {"minor": "单次短暂顿挫", "major": "多次可察觉的卡顿或明显的时间跳跃",
         "critical": "频繁卡顿使动作无法连贯辨识"},
        "sequence", ("freeze", "duplicate_frames", "motion_curves")),
    DefectSpec(
        "object_pop", "temporal_integrity", "物体突现/突消",
        "物体在没有遮挡或出画的情况下凭空出现或消失。",
        "被遮挡、移出画面、进入阴影都是正常的。渐进的淡入淡出也不算。",
        {"minor": "背景次要物体的短暂消失", "major": "可辨识物体凭空出现或消失",
         "critical": "主体凭空出现或消失"},
        "sequence", ("flow_anomaly", "permanence_check")),

    # ---- motion_quality --------------------------------------------------
    DefectSpec(
        "motion_stall", "motion_quality", "运动停滞",
        "条件描述了运动,但主体在片段中几乎不动或运动幅度远小于应有。",
        "条件本身描述静态场景时不适用;镜头运动与主体运动要分开判断。",
        {"minor": "运动幅度偏小但存在", "major": "主体基本静止,与条件描述明显不符",
         "critical": "整段近乎静止画面"},
        "sequence", ("motion_curves", "ordered_frames")),
    DefectSpec(
        "unnatural_gait", "motion_quality", "动作不自然",
        "人或动物的动作违反生物力学:步态相位错误、重心不随支撑脚转移、"
        "肢体摆动无惯性、像被逐帧摆出来的木偶。",
        "舞蹈、武术、卡通风格的夸张动作若条件允许则不算;"
        "慢动作下动作显得'飘'也是正常的。",
        {"minor": "个别动作略显僵硬", "major": "整体动作明显不符合真实运动规律",
         "critical": "动作完全脱离生物力学,呈提线木偶状"},
        "sequence", ("ordered_frames", "motion_trail")),
    DefectSpec(
        "speed_anomaly", "motion_quality", "速度异常",
        "运动速度突变、忽快忽慢,或与物体质量/场景明显不符。",
        "条件要求的变速、真实的加减速不算。",
        {"minor": "轻微速度波动", "major": "明显的速度突变或不合理的匀速",
         "critical": "速度完全失控"},
        "sequence", ("motion_curves",)),

    # ---- physical_integrity ---------------------------------------------
    DefectSpec(
        "gravity_violation", "physical_integrity", "重力违反",
        "下落物体不做匀加速运动、在空中停滞、或无支撑地上升。",
        "有支撑、被牵引、本身会飞、或处于水中/失重场景时不适用。",
        {"minor": "下落轨迹略不自然", "major": "物体在空中明显停顿或轨迹违反重力",
         "critical": "物体无支撑悬浮或凭空上升"},
        "sequence", ("freefall_check", "trajectory_plot")),
    DefectSpec(
        "rigidity_violation", "physical_integrity", "刚体形变",
        "本应刚性的物体(家具、车辆、餐具、建筑)在运动中拉伸、弯折、融化或流动。",
        "**柔性物体本来就该形变**:布料、头发、水、烟、植物均不适用。"
        "透视变化造成的形状改变也不是形变。",
        {"minor": "轻微的边缘不稳", "major": "刚体明显弯折或伸缩",
         "critical": "物体融化流动,失去原有结构"},
        "sequence", ("rigidity_check", "zoom_grid")),
    DefectSpec(
        "interpenetration", "physical_integrity", "穿模",
        "两个实体互相穿透,或肢体穿过自身/物体。",
        "遮挡不是穿模——判据是能看到两者在同一空间位置**同时存在**。"
        "半透明物体、影子、倒影不适用。",
        {"minor": "短暂的轻微穿插", "major": "明显的物体互穿,持续可见",
         "critical": "主体大幅穿过实体"},
        "frame", ("depth_order", "zoom_grid")),
    DefectSpec(
        "unsupported_float", "physical_integrity", "无支撑悬浮",
        "物体或人物应当接触地面/表面却悬空,或接触点不成立。",
        "跳跃、飞行、游泳、条件允许的超现实场景不适用。支撑物在画外时不得判定。",
        {"minor": "接触点略有偏差", "major": "明显悬空或脚部穿入地面",
         "critical": "主体完全漂浮于空中"},
        "frame", ("pose_detect", "ordered_frames")),

    # ---- appearance_integrity -------------------------------------------
    DefectSpec(
        "structure_collapse", "appearance_integrity", "结构崩坏",
        "物体失去可辨识的结构,呈现为不连贯的色块或畸形几何。",
        "景深虚化、远景细节缺失、有意的抽象风格不算。",
        {"minor": "背景次要物体结构模糊", "major": "可辨识物体结构明显错误",
         "critical": "主体结构崩坏"},
        "frame", ("zoom_grid", "highfreq_amplify")),
    DefectSpec(
        "repeated_pattern", "appearance_integrity", "重复花纹",
        "画面中出现规则的网格、棋盘或不自然的重复纹理。",
        "真实的重复图案(瓷砖、格纹衣物、建筑立面)是内容,不是缺陷。",
        {"minor": "局部轻微规则纹理", "major": "明显的棋盘或网格伪影",
         "critical": "大面积规则伪影"},
        "frame", ("highfreq_amplify", "spectral_signature")),
    DefectSpec(
        "defect_blur", "appearance_integrity", "非运动性模糊",
        "在低运动区域出现的、无法用运动或景深解释的模糊/软化。",
        "**运动模糊在快速运动处是物理正确的**;景深虚化是摄影意图;"
        "低光噪点不是模糊。",
        {"minor": "小面积轻微软化", "major": "可辨识区域明显模糊且无运动解释",
         "critical": "主体持续模糊不可辨"},
        "frame", ("softness", "blur_diagnosis")),
    DefectSpec(
        "lighting_inconsistency", "appearance_integrity", "光影矛盾",
        "阴影方向与光源不一致、物体缺少应有的影子或反射、局部光照违反场景。",
        "多光源场景、间接光、有意的风格化打光不算。",
        {"minor": "细微的阴影方向偏差", "major": "明显的阴影缺失或方向矛盾",
         "critical": "光照逻辑完全崩坏"},
        "frame", ("zoom_grid",)),
    DefectSpec(
        "text_garbled", "appearance_integrity", "文字错乱",
        "画面中的文字字形不成立、拼写错乱,或在帧间变化。",
        "远景、小字、非拉丁/中文的装饰性字符不适用;有意的模糊招牌不算。",
        {"minor": "小字略有变形", "major": "可读位置的文字明显错乱",
         "critical": "条件要求的文字完全无法辨认或错误"},
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
