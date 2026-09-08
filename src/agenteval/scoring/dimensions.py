"""The report contract: six dimensions, and how everything rolls up into them.

Skills are built at the granularity that makes them work -- a skill that can
zoom to a face has no business also judging camera moves -- but that is not the
granularity anyone wants to read. Nine numbers are not a report, they are a
spreadsheet, and half of them correlate. So the internal decomposition is kept
fine and the *output* is fixed at six dimensions chosen to be the coarsest set
that still separates failures a viewer would describe differently.

The six are near-orthogonal by construction: a clip can fail any one while
passing the rest. That is the test a dimension set has to meet. "Overall
quality" and "realism" fail it, because nothing fails them independently.

A dimension with nothing to judge reports **not applicable** and is excluded
from the overall, rather than scoring 10. Awarding a landscape full marks for
subject fidelity would let clips gain by containing none of the hard content --
and a model that learned that would produce emptier videos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

REPORT_DIMENSIONS: tuple[str, ...] = (
    "semantic_alignment",
    "motion_quality",
    "subject_fidelity",
    "physical_plausibility",
    "temporal_consistency",
    "visual_quality",
)

LABEL_ZH: dict[str, str] = {
    "semantic_alignment": "语义一致性",
    "motion_quality": "运动质量",
    "subject_fidelity": "主体保真度",
    "physical_plausibility": "物理合理性",
    "temporal_consistency": "时间一致性",
    "visual_quality": "画面质量",
}

DESCRIPTION_ZH: dict[str, str] = {
    "semantic_alignment": "视频是否做到了条件要求的事:主体、属性、数量、动作、顺序、运镜、风格。",
    "motion_quality": "运动本身是否成立:幅度是否足够、是否平滑、速度是否合理、动作是否符合生物力学。",
    "subject_fidelity": "主体(尤其是人)的结构是否正确:手、脸、四肢、身份一致性、材质质感。",
    "physical_plausibility": "世界是否遵守物理:重力、刚体保持、穿模、支撑接触、物体恒存。",
    "temporal_consistency": "画面在时间上是否稳定:闪烁、纹理沸腾、卡顿抽帧、物体突现突消。",
    "visual_quality": "单帧画面本身的质量:结构完整、清晰度、光影逻辑、伪影、美学。",
}

#: internal skill -> report dimension
SKILL_TO_DIMENSION: dict[str, str] = {
    "semantic_conformance": "semantic_alignment",
    "action_conformance": "semantic_alignment",
    "camera_conformance": "semantic_alignment",
    "style_conformance": "semantic_alignment",
    "motion_quality": "motion_quality",
    "human_integrity": "subject_fidelity",
    "subject_integrity": "subject_fidelity",
    "physical_integrity": "physical_plausibility",
    "temporal_integrity": "temporal_consistency",
    "appearance_integrity": "visual_quality",
    "static_integrity": "visual_quality",
}

#: defect key -> report dimension. Kept explicit rather than derived from the
#: taxonomy's own `dimension` field, because a few defects are judged by one
#: skill but belong to a different heading in the report: a garbled sign is
#: found while inspecting appearance, but a reader looks for it under whether
#: the video rendered what was asked.
DEFECT_TO_DIMENSION: dict[str, str] = {
    "hand_malformation": "subject_fidelity",
    "face_malformation": "subject_fidelity",
    "limb_deformation": "subject_fidelity",
    "identity_drift": "subject_fidelity",
    "plastic_texture": "subject_fidelity",
    "no_blink": "subject_fidelity",
    "flicker": "temporal_consistency",
    "texture_crawl": "temporal_consistency",
    "stutter": "temporal_consistency",
    "object_pop": "temporal_consistency",
    "motion_stall": "motion_quality",
    "unnatural_gait": "motion_quality",
    "speed_anomaly": "motion_quality",
    "gravity_violation": "physical_plausibility",
    "rigidity_violation": "physical_plausibility",
    "interpenetration": "physical_plausibility",
    "unsupported_float": "physical_plausibility",
    "structure_collapse": "visual_quality",
    "repeated_pattern": "visual_quality",
    "defect_blur": "visual_quality",
    "lighting_inconsistency": "visual_quality",
    "text_garbled": "visual_quality",
}

#: Report weights. Semantic alignment leads because a video that did not do what
#: was asked has failed at the task regardless of how well it is rendered;
#: visual quality trails because it is the most recoverable and the most
#: subjective.
DIMENSION_WEIGHT: dict[str, float] = {
    "semantic_alignment": 1.3,
    "motion_quality": 1.2,
    "subject_fidelity": 1.2,
    "physical_plausibility": 1.0,
    "temporal_consistency": 1.0,
    "visual_quality": 0.8,
}


def dimension_of(*, skill: str | None = None, defect: str | None = None) -> str | None:
    if defect and defect in DEFECT_TO_DIMENSION:
        return DEFECT_TO_DIMENSION[defect]
    if skill and skill in SKILL_TO_DIMENSION:
        return SKILL_TO_DIMENSION[skill]
    return None


@dataclass
class ReportDimension:
    key: str
    score: float                       # 0..10
    applicable: bool = True
    reason: str = ""
    n_findings: int = 0
    n_retracted: int = 0
    capped_by: str | None = None
    contributing_skills: list[str] = None      # type: ignore[assignment]
    top_findings: list[dict[str, Any]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.contributing_skills is None:
            self.contributing_skills = []
        if self.top_findings is None:
            self.top_findings = []

    def to_json(self) -> dict[str, Any]:
        return {
            "key": self.key, "label": LABEL_ZH.get(self.key, self.key),
            "score": round(self.score, 2) if self.applicable else None,
            "applicable": self.applicable, "reason": self.reason,
            "n_findings": self.n_findings, "n_retracted": self.n_retracted,
            "capped_by": self.capped_by,
            "contributing_skills": self.contributing_skills,
            "top_findings": self.top_findings,
        }
