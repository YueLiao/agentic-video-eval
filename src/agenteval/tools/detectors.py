"""Human-centric detectors, and the invariants derived from them.

These exist because of the *prior* limit: a VLM's pretraining is natural,
artifact-free video, so it has no strong notion of what a six-fingered hand or a
thigh that changes length looks like. Detectors supply the prior it lacks.

The most valuable thing here is not any single detector, it is
:func:`anatomy_invariants`, which uses no model at all. It exploits facts that
hold for every real body and hold for no generated one by accident:

* bone lengths are constant over time, so their coefficient of variation is
  near zero for real footage and rises sharply when a limb stretches;
* left and right limbs are near-symmetric in length;
* joint angles stay inside human range;
* the number of people in a shot does not fluctuate.

That makes it cheap, uncheatable, and — unlike a learned artifact classifier —
not tied to the generators it was trained on.

A second design point: detector *failure* is itself evidence. When a pose model's
keypoint confidence collapses on a frame it is reporting "this does not look like
a body", which is exactly the signal wanted. So confidence traces are returned,
not thresholded away.
"""

from __future__ import annotations

import os

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

import numpy as np

from agenteval.media.clip import VideoHandle
from agenteval.tools.base import ToolResult

# COCO-17 skeleton, used for the bone-length invariants.
COCO_BONES: tuple[tuple[int, int, str], ...] = (
    (5, 7, "l_upper_arm"), (7, 9, "l_forearm"),
    (6, 8, "r_upper_arm"), (8, 10, "r_forearm"),
    (11, 13, "l_thigh"), (13, 15, "l_shin"),
    (12, 14, "r_thigh"), (14, 16, "r_shin"),
    (5, 6, "shoulders"), (11, 12, "hips"),
)
BILATERAL: tuple[tuple[str, str], ...] = (
    ("l_upper_arm", "r_upper_arm"), ("l_forearm", "r_forearm"),
    ("l_thigh", "r_thigh"), ("l_shin", "r_shin"),
)


# ---- face ---------------------------------------------------------------

# mediapipe Pose emits 33 landmarks; the bone invariants are defined on COCO-17,
# so map across explicitly. (mp: 0 nose, 2/5 eyes, 7/8 ears, 11/12 shoulders,
# 13/14 elbows, 15/16 wrists, 23/24 hips, 25/26 knees, 27/28 ankles.)
_MP_TO_COCO = {0: 0, 2: 1, 5: 2, 7: 3, 8: 4,
               11: 5, 12: 6, 13: 7, 14: 8, 15: 9, 16: 10,
               23: 11, 24: 12, 25: 13, 26: 14, 27: 15, 28: 16}

_MP_TASKS_NOTE = """mediapipe >=0.10 on py3.11+ ships only the Tasks API; the
legacy `mp.solutions` namespace is gone and the model bundles are NOT in the
wheel. They are fetched once via tools.weights and cached on disk."""


@lru_cache(maxsize=1)
def _face_detector():
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    from agenteval.tools import weights
    opts = vision.FaceDetectorOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(weights.fetch("mp_face"))),
        min_detection_confidence=0.4)
    return vision.FaceDetector.create_from_options(opts)


@lru_cache(maxsize=1)
def _pose_landmarker():
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    from agenteval.tools import weights
    opts = vision.PoseLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(weights.fetch("mp_pose"))),
        num_poses=4, min_pose_detection_confidence=0.4)
    return vision.PoseLandmarker.create_from_options(opts)


@lru_cache(maxsize=1)
def _hand_landmarker():
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    from agenteval.tools import weights
    opts = vision.HandLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(weights.fetch("mp_hands"))),
        num_hands=4, min_hand_detection_confidence=0.4)
    return vision.HandLandmarker.create_from_options(opts)


def _mp_image(bgr: np.ndarray):
    import mediapipe as mp
    return mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(bgr[:, :, ::-1]))


def face_detect(video: VideoHandle, indices: Sequence[int]) -> ToolResult:
    """Per-frame face boxes."""
    try:
        det = _face_detector()
    except Exception as e:  # noqa: BLE001
        return ToolResult(value={"error": str(e)}, reliability=0.0,
                          backend="unavailable")
    per, sizes = [], []
    for i, fr in zip(indices, video.read(indices)):
        h, w = fr.shape[:2]
        res = det.detect(_mp_image(fr))
        boxes, confs = [], []
        for d in (res.detections or []):
            bb = d.bounding_box
            boxes.append([round(bb.origin_x / w, 4), round(bb.origin_y / h, 4),
                          round(bb.width / w, 4), round(bb.height / h, 4)])
            confs.append(round(float(d.categories[0].score), 3))
            sizes.append(bb.width / w)
        per.append({"idx": int(i), "boxes": boxes, "conf": confs})
    n_hit = sum(1 for p in per if p["boxes"])
    max_frac = round(float(max(sizes)), 4) if sizes else 0.0
    return ToolResult(
        value={"per_frame": per, "n_frames_with_face": n_hit,
               "n_frames": len(per), "max_face_frac": max_frac},
        reliability=0.75 if n_hit else 0.4, backend="mediapipe/blaze_face",
        hint=(f"{n_hit}/{len(per)} 帧检出人脸,最大人脸占画面宽度 {max_frac:.1%}。"
              "占比 <8% 时人脸细节在整帧图里不可判读,必须裁剪放大后再看。"
              "检出率骤降本身也是信号——检测器在说'这不像人脸'。"),
    )


def pose_detect(video: VideoHandle, indices: Sequence[int]) -> ToolResult:
    """COCO-17 keypoints per frame (remapped from mediapipe's 33)."""
    try:
        pose = _pose_landmarker()
    except Exception as e:  # noqa: BLE001
        return ToolResult(value={"error": str(e)}, reliability=0.0,
                          backend="unavailable")
    per = []
    for i, fr in zip(indices, video.read(indices)):
        res = pose.detect(_mp_image(fr))
        people = []
        for lms in (res.pose_landmarks or []):
            kp = [[0.0, 0.0, 0.0] for _ in range(17)]
            for mp_i, coco_i in _MP_TO_COCO.items():
                l = lms[mp_i]
                kp[coco_i] = [round(l.x, 4), round(l.y, 4),
                              round(float(l.visibility), 3)]
            people.append(kp)
        best = people[0] if people else [[0.0, 0.0, 0.0]] * 17
        per.append({"idx": int(i), "n_persons": len(people), "kpts": best,
                    "mean_conf": round(float(np.mean([k[2] for k in best])), 3)})
    confs = [p["mean_conf"] for p in per]
    counts = [p["n_persons"] for p in per]
    return ToolResult(
        value={"per_frame": per, "n_frames": len(per),
               "mean_conf": round(float(np.mean(confs)), 3) if confs else 0.0,
               "min_conf": round(float(np.min(confs)), 3) if confs else 0.0,
               "n_persons_max": int(max(counts)) if counts else 0,
               "n_persons_varies": bool(len(set(counts)) > 1)},
        reliability=0.7 if any(counts) else 0.3, backend="mediapipe/pose_landmarker",
        hint=("COCO-17 归一化关键点。置信度骤降的帧值得单独看——"
              "检测器失效通常意味着那一帧的人体结构本身有问题。"
              "人数在帧间变化也是硬信号。"),
    )


def hands_detect(video: VideoHandle, indices: Sequence[int]) -> ToolResult:
    try:
        hands = _hand_landmarker()
    except Exception as e:  # noqa: BLE001
        return ToolResult(value={"error": str(e)}, reliability=0.0,
                          backend="unavailable")
    per, counts = [], []
    for i, fr in zip(indices, video.read(indices)):
        res = hands.detect(_mp_image(fr))
        hs = [{"kpts": [[round(p.x, 4), round(p.y, 4)] for p in lms]}
              for lms in (res.hand_landmarks or [])]
        counts.append(len(hs))
        per.append({"idx": int(i), "n_hands": len(hs), "hands": hs})
    return ToolResult(
        value={"per_frame": per, "n_hands_max": int(max(counts)) if counts else 0,
               "n_hands_varies": bool(len(set(counts)) > 1),
               "hand_count_std": round(float(np.std(counts)), 3) if counts else 0.0},
        reliability=0.55 if any(counts) else 0.3, backend="mediapipe/hand_landmarker",
        hint=("21 点手部关键点。手是生成模型最常崩的部位,但检测器在崩坏的手上"
              "本身也不可靠——检出数在帧间跳变往往正是手出问题的证据,"
              "而不是检测器的噪声。最终判断必须看放大后的手部裁剪。"),
    )


# ---- the model-free invariants ------------------------------------------

def anatomy_invariants(pose: ToolResult, hands: ToolResult | None = None,
                       *, min_conf: float = 0.3) -> ToolResult:
    """Derive time-consistency invariants from a pose track. No model involved.

    A real body has constant bone lengths; a generated one does not. This is the
    cheapest high-signal anatomy check available and it cannot be gamed by
    matching a training distribution, because it is a geometric identity rather
    than a learned appearance.
    """
    per = pose.value.get("per_frame") or []
    if not per:
        return ToolResult(value={"error": "no pose"}, reliability=0.0,
                          backend="anatomy_invariants")

    lengths: dict[str, list[float]] = {n: [] for *_, n in COCO_BONES}
    low_conf_frames: list[int] = []
    for p in per:
        kp = p["kpts"]
        if p.get("mean_conf", 0) < min_conf:
            low_conf_frames.append(p["idx"])
            continue
        for a, b, name in COCO_BONES:
            ka, kb = kp[a], kp[b]
            if ka[2] < min_conf or kb[2] < min_conf:
                continue
            lengths[name].append(float(np.hypot(ka[0] - kb[0], ka[1] - kb[1])))

    cv: dict[str, float] = {}
    for name, vals in lengths.items():
        if len(vals) >= 4:
            m = float(np.mean(vals))
            if m > 1e-4:
                cv[name] = round(float(np.std(vals)) / m, 4)
    worst = max(cv, key=cv.get) if cv else ""
    max_cv = cv.get(worst, 0.0)

    asym = {}
    for l, r in BILATERAL:
        if lengths.get(l) and lengths.get(r):
            ml, mr = float(np.mean(lengths[l])), float(np.mean(lengths[r]))
            if max(ml, mr) > 1e-4:
                asym[f"{l}/{r}"] = round(abs(ml - mr) / max(ml, mr), 4)
    max_asym = max(asym.values()) if asym else 0.0

    n_hands = [p.get("n_hands", 0) for p in (hands.value.get("per_frame") or [])] \
        if hands and hands.value.get("per_frame") else []
    hand_instability = (round(float(np.std(n_hands)), 3) if n_hands else 0.0)

    return ToolResult(
        value={
            "bone_length_cv": cv, "worst_bone": worst, "max_bone_cv": round(max_cv, 4),
            "bilateral_asymmetry": asym, "max_asymmetry": round(max_asym, 4),
            "low_confidence_frames": low_conf_frames[:20],
            "n_low_confidence": len(low_conf_frames),
            "hand_count_instability": hand_instability,
        },
        reliability=0.65 if cv else 0.2, backend="anatomy_invariants",
        hint=("骨长变异系数(bone_length_cv):真人的骨长恒定,该值通常 <0.05;"
              ">0.15 表示投影长度在时间上不稳定,>0.30 更可疑。"
              "**重要局限**:这是 2D 投影长度。肢体朝向/背离镜头旋转时会发生"
              "透视缩短(foreshortening),同样会抬高 CV,而那是完全正常的。"
              "所以高 CV 只说明'这根肢体值得放大去看',不能单独当作畸变的证据——"
              "必须裁剪到该肢体、逐帧确认它是真的在变长变短,还是只是在转向。"
              "同理,低置信度帧是检测器在说'这里不像人体',也只是线索。"),
    )


# ---- availability -------------------------------------------------------

def availability() -> dict[str, tuple[bool, str]]:
    import importlib.util as u
    from agenteval.tools import weights
    out: dict[str, tuple[bool, str]] = {}
    mp_ok = u.find_spec("mediapipe") is not None
    for name, wkey in (("face_detect", "mp_face"), ("pose_detect", "mp_pose"),
                       ("hands_detect", "mp_hands")):
        if not mp_ok:
            out[name] = (False, "pip install mediapipe")
        elif not weights.have(wkey):
            out[name] = (True, f"weight {wkey} will be fetched on first use")
        else:
            out[name] = (True, "")
    out["anatomy_invariants"] = out["pose_detect"]
    return out
