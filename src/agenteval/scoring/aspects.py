"""What to report, at the granularity the measurement actually supports.

The previous version fixed six dimensions up front and folded everything into
them. That was backwards. A reported number earns its place by two tests, and
neither is about fitting a taxonomy:

1. **Can it be measured reliably on this clip?** Not in principle -- on *this*
   clip. Hand structure is highly judgeable when hands are visible and large
   enough to resolve, and not judgeable at all when they are twenty pixels wide.
   The same aspect flips between the two within one dataset.
2. **Does knowing it change what anyone does?** "Hands malformed in 3 of 22
   sampled frames" tells a model developer to go get hand data. "Aesthetics 7.2"
   tells them nothing.

Two consequences.

*Report finely where the machine is reliable.* Face, hands, limbs and identity
are separately measurable and separately fixable, so collapsing them into one
"subject fidelity" number destroys the only information a developer could act
on -- a 6.5 there could mean anything.

*Report nothing where it is not.* Aesthetic appeal, narrative quality and
creativity have no reliable machine anchor. Emitting a number for them does not
add information, it adds noise that is indistinguishable from signal once it
reaches a leaderboard. They are listed in :data:`NOT_SCORED` with the reason, so
their absence is a stated decision rather than an oversight.

Every aspect therefore carries a **gate**: the evidence conditions under which
it may be scored at all. Failing the gate produces "not judgeable, because X",
which is a genuinely useful output. A guess would not be.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

Judgeability = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class Aspect:
    key: str
    label_zh: str
    group: str                       # optional rollup, for leaderboards only
    defects: tuple[str, ...]         # taxonomy keys that score this aspect
    judgeability: Judgeability
    actionable: str                  # what a developer does when this is bad
    gate: str = ""                   # human-readable evidence requirement
    gate_fn: Callable[[dict[str, Any]], tuple[bool, str]] | None = None

    def check_gate(self, ev: dict[str, Any]) -> tuple[bool, str]:
        if self.gate_fn is None:
            return True, ""
        return self.gate_fn(ev)


# ---- gates ---------------------------------------------------------------
# `ev` is a flat dict of measurements collected during the run: detector
# outputs, coverage, resolution. Gates are intentionally strict: a false "not
# judgeable" costs one missing number, a false score corrupts the report.

def _gate_hands(ev: dict[str, Any]) -> tuple[bool, str]:
    n = int(ev.get("n_hands_max", 0))
    if n == 0:
        return False, "画面中未检出手部"
    if float(ev.get("max_hand_frac", 1.0)) < 0.04:
        return False, "手部占画面过小(<4% 宽),放大后仍不足以判断指节结构"
    return True, ""


def _gate_face(ev: dict[str, Any]) -> tuple[bool, str]:
    if int(ev.get("n_frames_with_face", 0)) == 0:
        return False, "画面中未检出人脸"
    if float(ev.get("max_face_frac", 0.0)) < 0.05:
        return False, "人脸占画面过小(<5% 宽),不足以判断五官结构"
    return True, ""


def _gate_person(ev: dict[str, Any]) -> tuple[bool, str]:
    if int(ev.get("n_persons_max", 0)) == 0:
        return False, "画面中无人物"
    if float(ev.get("pose_mean_conf", 0.0)) < 0.25:
        return False, "姿态估计置信度过低,骨架不可信,无法据此判断肢体结构"
    return True, ""


def _gate_identity(ev: dict[str, Any]) -> tuple[bool, str]:
    ok, why = _gate_face(ev)
    if not ok:
        return False, why
    if float(ev.get("duration_s", 0.0)) < 2.0:
        return False, "片段过短(<2s),不足以判断身份是否漂移"
    return True, ""


def _gate_blink(ev: dict[str, Any]) -> tuple[bool, str]:
    if float(ev.get("max_face_frac", 0.0)) < 0.12:
        return False, "人脸不够大,无法可靠判断眨眼"
    if float(ev.get("duration_s", 0.0)) < 3.0:
        return False, "片段过短(<3s),不眨眼不构成异常"
    return True, ""


def _gate_coverage(ev: dict[str, Any]) -> tuple[bool, str]:
    c = float(ev.get("perceptual_coverage", 1.0))
    if c < 0.6:
        return False, f"逐帧覆盖不足(感知覆盖 {c:.0%}),不足以给出结论"
    return True, ""


def _gate_motion(ev: dict[str, Any]) -> tuple[bool, str]:
    if float(ev.get("motion_level", 1.0)) < 0.03:
        return False, "画面近乎静止,运动质量无从判断"
    return True, ""


def _gate_text(ev: dict[str, Any]) -> tuple[bool, str]:
    if not ev.get("has_text"):
        return False, "画面中无需要判读的文字"
    return True, ""


def _gate_trackable(ev: dict[str, Any]) -> tuple[bool, str]:
    if float(ev.get("track_survival", 0.0)) < 0.4:
        return False, "特征点跟踪存活率过低,物理量测不可信"
    return True, ""


# ---- the registry --------------------------------------------------------

ASPECTS: tuple[Aspect, ...] = (
    # --- 人物,拆到可操作的粒度 ---
    Aspect("hand_structure", "人手结构", "human", ("hand_malformation",),
           "high", "手部数据增强 / 手部区域加权训练",
           "需检出手部且占画面≥4%", _gate_hands),
    Aspect("face_structure", "人脸结构", "human", ("face_malformation",),
           "high", "人脸数据 / 面部区域 loss 加权",
           "需检出人脸且占画面≥5%", _gate_face),
    Aspect("limb_structure", "肢体结构", "human",
           ("limb_deformation",), "medium",
           "姿态条件控制 / 骨架先验约束",
           "需检出人物且姿态置信度≥0.25", _gate_person),
    Aspect("identity_consistency", "身份一致性", "human", ("identity_drift",),
           "medium", "ID 保持机制 / 参考图条件",
           "需人脸可见且片段≥2s", _gate_identity),
    Aspect("skin_texture", "皮肤材质", "human", ("plastic_texture",),
           "medium", "高频细节保持 / 减少过度平滑",
           "需人脸占画面≥5%", _gate_face),
    Aspect("eye_behavior", "眼部行为", "human", ("no_blink",),
           "low", "面部微动态建模",
           "需人脸占画面≥12% 且片段≥3s", _gate_blink),

    # --- 时间稳定性 ---
    Aspect("flicker", "闪烁", "temporal", ("flicker",), "high",
           "时序一致性约束 / 训练时加时序正则", ""),
    Aspect("texture_stability", "纹理稳定性", "temporal", ("texture_crawl",),
           "high", "时序特征对齐 / 提高时序感受野", ""),
    Aspect("frame_continuity", "帧连续性", "temporal", ("stutter",), "high",
           "插帧质量 / 采样步长", ""),
    Aspect("object_permanence", "物体恒存", "temporal", ("object_pop",),
           "medium", "长程时序建模", ""),

    # --- 运动 ---
    Aspect("motion_magnitude", "运动幅度", "motion", ("motion_stall",), "high",
           "运动强度条件控制 / 数据中低运动样本占比", "需画面有运动", _gate_motion),
    Aspect("motion_smoothness", "运动平滑性", "motion", ("speed_anomaly",),
           "high", "时序采样 / 插帧", "需画面有运动", _gate_motion),
    Aspect("motion_naturalness", "动作自然度", "motion", ("unnatural_gait",),
           "medium", "真实动作数据 / 动力学先验", "需画面有运动", _gate_motion),

    # --- 物理 ---
    Aspect("gravity", "重力遵守", "physics", ("gravity_violation",), "medium",
           "物理先验 / 含明确弹道的训练样本",
           "需可跟踪的运动物体", _gate_trackable),
    Aspect("rigidity", "刚体保持", "physics", ("rigidity_violation",), "medium",
           "刚体一致性约束", "需可跟踪的物体", _gate_trackable),
    Aspect("interpenetration", "穿模", "physics", ("interpenetration",),
           "medium", "深度/占据建模", ""),
    Aspect("support_contact", "支撑接触", "physics", ("unsupported_float",),
           "low", "场景几何建模", ""),

    # --- 语义一致性:是否做到了条件要求的事 ---
    # These score against a compiled requirement graph, so their gate is the
    # existence of requirements of that kind -- a condition that asked for no
    # text cannot fail text rendering, and scoring it would be meaningless.
    Aspect("entity_presence", "主体出现", "semantic", (), "high",
           "条件对齐训练 / 主体 grounding",
           "需条件中含实体要求", lambda ev: (
               bool(ev.get("n_req_entity")), "条件中没有可核对的实体要求")),
    Aspect("attribute_binding", "属性绑定", "semantic", (), "medium",
           "属性绑定训练(颜色/材质与主体的对应)",
           "需条件中含属性要求", lambda ev: (
               bool(ev.get("n_req_attribute")), "条件中没有属性要求")),
    Aspect("object_count", "数量正确", "semantic", (), "medium",
           "计数能力 / 多实例生成",
           "需条件中含数量要求", lambda ev: (
               bool(ev.get("n_req_count")), "条件中没有数量要求")),
    Aspect("spatial_relation", "空间关系", "semantic", (), "low",
           "空间关系建模",
           "需条件中含空间关系要求", lambda ev: (
               bool(ev.get("n_req_relation")), "条件中没有空间关系要求")),
    Aspect("action_execution", "动作执行", "semantic", (), "high",
           "动作数据 / 动词条件对齐",
           "需条件中含动作要求", lambda ev: (
               bool(ev.get("n_req_action")), "条件中没有动作要求")),
    Aspect("action_order", "动作顺序", "semantic", (), "medium",
           "时序条件控制",
           "需条件中含多动作时序关系", lambda ev: (
               bool(ev.get("n_req_order")), "条件中没有动作顺序要求")),
    Aspect("camera_control", "运镜控制", "semantic", (), "high",
           "运镜条件控制 / 相机轨迹标注数据",
           "需条件中指定了运镜", lambda ev: (
               bool(ev.get("n_req_camera")), "条件中未指定运镜")),
    Aspect("style_match", "风格匹配", "semantic", (), "medium",
           "风格条件对齐",
           "需条件中指定了风格", lambda ev: (
               bool(ev.get("n_req_style")), "条件中未指定风格")),

    # --- 单帧画质 ---
    Aspect("structure_coherence", "结构完整性", "frame_quality",
           ("structure_collapse",), "high", "分辨率 / 细节保持",
           "需逐帧覆盖≥60%", _gate_coverage),
    Aspect("artifact_free", "伪影", "frame_quality",
           ("repeated_pattern", "defect_blur"), "high",
           "上采样结构 / 去棋盘伪影", "需逐帧覆盖≥60%", _gate_coverage),
    Aspect("lighting_logic", "光影逻辑", "frame_quality",
           ("lighting_inconsistency",), "low",
           "光照一致性建模", "需逐帧覆盖≥60%", _gate_coverage),
    Aspect("text_rendering", "文字渲染", "frame_quality", ("text_garbled",),
           "medium", "文字渲染专项数据", "需画面中有文字", _gate_text),
)

BY_KEY: dict[str, Aspect] = {a.key: a for a in ASPECTS}
BY_GROUP: dict[str, list[Aspect]] = {}
for _a in ASPECTS:
    BY_GROUP.setdefault(_a.group, []).append(_a)

DEFECT_TO_ASPECT: dict[str, str] = {
    d: a.key for a in ASPECTS for d in a.defects
}

#: Aggregate views that cut across groups, for comparing against human
#: evaluation axes that were not defined on this taxonomy.
#:
#: 运动合理性 as the annotation protocol uses it is not the same cut as either
#: the motion group or the physics group here: it asks whether the movement is
#: *believable*, which spans how it flows (smoothness, naturalness) and whether
#: it obeys the world (gravity, rigidity). Reporting only the finer aspects
#: makes correlation against that human axis impossible; reporting only the
#: composite loses the diagnosis. So both are produced, and the composite is
#: explicitly a view over aspects rather than a separate measurement.
COMPOSITE_VIEWS: dict[str, tuple[str, ...]] = {
    "motion_plausibility": ("motion_magnitude", "motion_smoothness",
                            "motion_naturalness", "gravity", "rigidity"),
    "human_quality": ("hand_structure", "face_structure", "limb_structure",
                      "identity_consistency", "skin_texture"),
    "temporal_stability": ("flicker", "texture_stability", "frame_continuity",
                           "object_permanence"),
    "instruction_following": ("entity_presence", "attribute_binding",
                              "object_count", "spatial_relation",
                              "action_execution", "action_order",
                              "camera_control", "style_match"),
}

COMPOSITE_LABEL_ZH: dict[str, str] = {
    "motion_plausibility": "运动合理性",
    "human_quality": "人物质量",
    "temporal_stability": "时间稳定性",
    "instruction_following": "指令遵循",
}

GROUP_LABEL_ZH: dict[str, str] = {
    "human": "人物保真", "temporal": "时间稳定", "motion": "运动",
    "physics": "物理", "frame_quality": "单帧画质",
    "semantic": "语义一致性",
}

#: Deliberately not scored, with the reason. Recorded rather than omitted, so
#: the absence reads as a decision instead of a gap -- and so the list can be
#: argued with.
NOT_SCORED: dict[str, str] = {
    "aesthetic_appeal": "美学偏好没有可靠的机器锚点,绝对分在不同内容间不可比。"
                        "如果需要,只用成对比较 + Bradley-Terry 出相对排名,不出绝对分。",
    "creativity": "创意需要与'该条件下其他可能的输出'比较,单看一条视频无从判断。",
    "narrative": "叙事与编导质量依赖意图和语境,当前 VLM 判定的方差远大于其信号。",
    "emotional_expression": "情绪表达的判定在人评中一致性本身就低,机评更不可靠。",
    "prompt_creativity_tradeoff": "'忠实条件'与'表现更好'的取舍是产品决策,不是可测属性。",
}
