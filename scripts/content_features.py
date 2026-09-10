#!/usr/bin/env python3
"""Content-level features: what is in the picture, not what its texture is like.

The twelve features the point-wise model was fitted on are all low-level motion
and texture statistics, and every one of them carries the generator's
fingerprint -- which is why accuracy on this benchmark is monotone in how far
apart the two generators are, and why the model sits at chance on `seed` pairs
where the fingerprint is identical by construction.

A seed pair is the same model, the same prompt, two samples. Nothing about
style can separate them. What can is whether the body held together, whether
the hands stayed hands, whether the thing that was there stayed there. These
are measurements over detections rather than over pixels, which is what makes
them fingerprint-free in a way a flow statistic is not.

Every value here is a *cue*, not a verdict: bone-length CV rises under ordinary
foreshortening as well as under deformation, and a detector's confidence drops
on a hard pose as well as on a broken one. They earn their place only if they
move the seed-family number.
"""
from __future__ import annotations

import os as _os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CONTENT_FEATS = [
    "max_bone_cv",        # 骨长变异系数最大值:肢体在时间上伸缩
    "mean_bone_cv",
    "max_asymmetry",      # 左右肢长不对称
    "pose_conf_mean",     # 检测器对"这是人体"的平均确信
    "pose_conf_min",
    "pose_lowconf_frac",  # 检测器说"这里不像人体"的帧占比
    "person_count_std",   # 人数不稳定
    "hand_count_std",     # 手数不稳定
    "face_frac_std",      # 脸占画面比例的抖动
    "face_miss_frac",     # 有脸的片段里检不到脸的帧占比
    "kpt_jitter",         # 关键点的帧间抖动(除以人体尺度)
]
# +1 means a larger value should score higher. Every one of these is a defect
# cue except the two confidence readings.
CONTENT_BETTER = {"max_bone_cv": -1, "mean_bone_cv": -1, "max_asymmetry": -1,
                  "pose_conf_mean": +1, "pose_conf_min": +1,
                  "pose_lowconf_frac": -1, "person_count_std": -1,
                  "hand_count_std": -1, "face_frac_std": -1,
                  "face_miss_frac": -1, "kpt_jitter": -1}


def content_features(path: str, *, n: int = 16) -> dict:
    import cv2
    cv2.setNumThreads(1)
    from agenteval.media.clip import VideoHandle, uniform_indices
    from agenteval.tools import detectors as D

    out = {k: 0.0 for k in CONTENT_FEATS}
    try:
        v = VideoHandle(path)
        idx = uniform_indices(v.total, min(n, v.total))
        pose = D.pose_detect(v, idx)
        hands = D.hands_detect(v, idx)
        face = D.face_detect(v, idx)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:90]}

    per = pose.value.get("per_frame") or []
    if per:
        inv = D.anatomy_invariants(pose, hands)
        cvs = list((inv.value.get("bone_length_cv") or {}).values())
        out["max_bone_cv"] = float(inv.value.get("max_bone_cv") or 0.0)
        out["mean_bone_cv"] = float(np.mean(cvs)) if cvs else 0.0
        out["max_asymmetry"] = float(inv.value.get("max_asymmetry") or 0.0)
        confs = [float(p.get("mean_conf", 0.0)) for p in per]
        out["pose_conf_mean"] = float(np.mean(confs))
        out["pose_conf_min"] = float(np.min(confs))
        out["pose_lowconf_frac"] = float(np.mean([c < 0.3 for c in confs]))
        counts = [len(p.get("persons") or []) if "persons" in p else
                  (1 if p.get("kpts") else 0) for p in per]
        out["person_count_std"] = float(np.std(counts))
        # Keypoint jitter, divided by the body's own scale so a close-up and a
        # long shot are on the same footing.
        kp = [np.asarray(p["kpts"], np.float32) for p in per if p.get("kpts")]
        if len(kp) >= 3 and kp[0].shape[0] > 4:
            k = np.stack([x[:, :2] for x in kp if x.shape == kp[0].shape])
            if len(k) >= 3:
                scale = float(np.median(np.ptp(k.reshape(-1, 2), axis=0))) or 1.0
                out["kpt_jitter"] = float(
                    np.abs(np.diff(k, 2, axis=0)).mean() / scale)

    hp = hands.value.get("per_frame") or []
    if hp:
        out["hand_count_std"] = float(np.std([p.get("n_hands", 0) for p in hp]))

    fp = face.value.get("per_frame") or []
    if fp:
        fr = [float(p.get("max_face_frac", 0.0)) if "max_face_frac" in p
              else (float(p["boxes"][0][2] * p["boxes"][0][3])
                    if p.get("boxes") else 0.0) for p in fp]
        seen = [x > 0 for x in fr]
        if any(seen):
            out["face_frac_std"] = float(np.std([x for x in fr if x > 0]))
            out["face_miss_frac"] = float(1.0 - np.mean(seen))
    return out


if __name__ == "__main__":
    import glob
    for p in sorted(glob.glob("/pub/evaluation_group/cy/rm_videos/**/*.mp4",
                              recursive=True))[:3]:
        f = content_features(p)
        print(Path(p).stem[:34],
              {k: round(v, 3) for k, v in f.items()} if "error" not in f else f)
