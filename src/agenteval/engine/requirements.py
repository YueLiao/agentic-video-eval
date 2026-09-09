"""Which evidence can answer which judgement, and what is still missing.

The fixed-chain version of this asked every skill to fetch the same three views
before every verdict. That guaranteed the axes were answerable but ignored the
input: a defect that never moves needs no whole-clip view, one already obvious
in the full frame needs no magnification ladder, and a clip with no people needs
none of it.

Letting the judge choose freely is what caused the original problem, though --
it chose only magnified crops, then could not answer three of the four questions
it was asked and defaulted to the same grade every time.

So neither. The harness declares **what each axis requires**, in terms of
capabilities rather than named tools; the model chooses how to satisfy that; and
the code checks coverage before the verdict and says plainly which axes are
still unbacked. An axis with no evidence behind it does not get to carry a
confident grade -- enforced here rather than requested in a prompt, because a
request is what the model already ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

#: What a view offers, independent of which tool produced it. Tools advertise
#: these so new views become usable without touching this table.
Capability = str

FULL_SCALE = "full_scale"        # the frame at normal viewing size
MAGNIFIED = "magnified"          # native-resolution crop of a region
WHOLE_CLIP = "whole_clip"        # sampling that spans the entire clip
TEMPORAL_LOCAL = "temporal_local"  # consecutive frames within a window
COMPOSITION = "composition"      # enough of the frame to see what the subject is
SCALE_SERIES = "scale_series"    # the same region at several magnifications
CONTRAST = "contrast"            # the same region at a quiet reference time

#: Which capabilities can answer which axis. Alternatives, not conjunctions --
#: several routes to the same answer, so the model has room to choose.
AXIS_REQUIREMENTS: dict[str, tuple[frozenset[Capability], ...]] = {
    "existence": (frozenset({MAGNIFIED}), frozenset({SCALE_SERIES})),
    "severity": (frozenset({FULL_SCALE}), frozenset({SCALE_SERIES})),
    "extent": (frozenset({WHOLE_CLIP}),),
    "salience": (frozenset({COMPOSITION}), frozenset({FULL_SCALE})),
}

AXIS_LABEL_ZH: dict[str, str] = {
    "existence": "存在性", "severity": "严重度",
    "extent": "持续范围", "salience": "显著位置",
}

AXIS_NEED_ZH: dict[str, str] = {
    "existence": "需要放大视图才能确认缺陷确实存在、形态是什么",
    "severity": "需要原始观看尺寸的整帧,或多级放大梯度,才能判断正常观看是否可见",
    "extent": "需要覆盖整段视频的采样,才能判断问题占了多长时间",
    "salience": "需要看到整体构图,才能判断问题落在主体上还是背景",
}

#: Tool name -> what its output can answer. Declared here so a skill's action
#: menu stays about *what to look at*, and this stays about *what that view can
#: settle* -- the two change for different reasons.
TOOL_CAPABILITIES: dict[str, frozenset[Capability]] = {
    "paired_view": frozenset({FULL_SCALE, MAGNIFIED, COMPOSITION}),
    "scale_ladder": frozenset({SCALE_SERIES, FULL_SCALE, MAGNIFIED, COMPOSITION}),
    "temporal_extent": frozenset({WHOLE_CLIP, MAGNIFIED}),
    "overview": frozenset({FULL_SCALE, COMPOSITION, WHOLE_CLIP}),
    "ordered_frames": frozenset({FULL_SCALE, COMPOSITION, TEMPORAL_LOCAL}),
    "contact_sheet": frozenset({FULL_SCALE, COMPOSITION, WHOLE_CLIP}),
    "overview_frames": frozenset({FULL_SCALE, COMPOSITION, WHOLE_CLIP}),
    "zoom": frozenset({MAGNIFIED}),
    "zoom_confirm": frozenset({MAGNIFIED}),
    "zoom_face": frozenset({MAGNIFIED}),
    "zoom_hands": frozenset({MAGNIFIED}),
    "zoom_region": frozenset({MAGNIFIED}),
    "zoom_action": frozenset({MAGNIFIED, TEMPORAL_LOCAL}),
    "roi_sequence": frozenset({MAGNIFIED, TEMPORAL_LOCAL}),
    "ordered_frames_roi": frozenset({MAGNIFIED, TEMPORAL_LOCAL}),
    "tile_frame": frozenset({MAGNIFIED, FULL_SCALE}),
    "contrast_pair": frozenset({CONTRAST, MAGNIFIED}),
    "filmstrip": frozenset({TEMPORAL_LOCAL, FULL_SCALE}),
    "motion_trail": frozenset({TEMPORAL_LOCAL, COMPOSITION}),
    "dense_window": frozenset({TEMPORAL_LOCAL, FULL_SCALE}),
    "skeleton_overlay": frozenset({FULL_SCALE, COMPOSITION}),
    "highfreq_amplify": frozenset({MAGNIFIED, SCALE_SERIES}),
    "motion_curves": frozenset({WHOLE_CLIP}),
}


@dataclass
class Coverage:
    have: set[Capability]
    answered: list[str]
    missing: dict[str, str]        # axis -> what would answer it

    @property
    def complete(self) -> bool:
        return not self.missing


def capabilities_of(tool: str) -> frozenset[Capability]:
    return TOOL_CAPABILITIES.get(tool, frozenset())


def assess(tools_used: Iterable[str],
           axes: Sequence[str] = ("existence", "severity", "extent", "salience"),
           ) -> Coverage:
    """Which axes the gathered evidence can currently support."""
    have: set[Capability] = set()
    for t in tools_used:
        have |= capabilities_of(t)
    answered, missing = [], {}
    for ax in axes:
        opts = AXIS_REQUIREMENTS.get(ax, ())
        if any(o <= have for o in opts):
            answered.append(ax)
        else:
            missing[ax] = AXIS_NEED_ZH.get(ax, "")
    return Coverage(have, answered, missing)


def render_gaps(cov: Coverage) -> str:
    """What to tell the judge about its own evidence before it grades.

    Stated rather than silently patched: the model is better placed than a fixed
    rule to decide whether a missing view is worth another probe or whether the
    honest answer is a low-confidence grade.
    """
    if cov.complete:
        return ("你现在的证据可以支撑全部四个判断轴(存在性/严重度/持续范围/显著位置)。")
    lines = ["**注意:你当前的证据不足以支撑下面这些判断轴**"]
    for ax, need in cov.missing.items():
        lines.append(f"- {AXIS_LABEL_ZH.get(ax, ax)}:{need}")
    lines.append("对这些轴,你有两个选择:再取一次相应的证据,"
                 "或者按现有证据给出结论但**把 confidence 降到 0.4 或更低**"
                 "并在 rationale 里写明缺什么证据。**不要在没有证据的情况下给高置信度。**")
    return "\n".join(lines)


def cap_confidence(finding_axes_missing: Sequence[str], confidence: float,
                   severity: str) -> tuple[float, str]:
    """Enforce the limit rather than request it.

    Grading severity without a normal-scale view is the specific failure that
    put 85% of findings in one grade, so it is capped in code. A prompt asking
    for the same restraint is what the model already ignored.
    """
    if not finding_axes_missing:
        return confidence, severity
    note = ""
    if "severity" in finding_axes_missing:
        if severity in ("major", "severe"):
            severity = "minor"
            note = "缺少原始尺寸视图,无法确认正常观看是否可见,严重度下调至 minor;"
        confidence = min(confidence, 0.4)
    if "existence" in finding_axes_missing:
        confidence = min(confidence, 0.3)
        note += "缺少放大视图,存在性未经确认;"
    if "extent" in finding_axes_missing:
        confidence = min(confidence, 0.6)
    return confidence, severity
