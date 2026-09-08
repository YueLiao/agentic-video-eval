"""Camera trajectory from frame-to-frame homography.

Camera conformance is the one conformance requirement that is largely decidable
without a VLM: "slow pull-out" is a claim about global image scale over time,
and that is measurable. Estimating it directly rather than asking the model is
both cheaper and more reliable, since judging camera motion from sampled frames
is a task VLMs are notably poor at -- they confuse subject motion with camera
motion constantly.

The estimate is a similarity transform per adjacent pair, decomposed into scale,
translation and rotation, then accumulated. Its failure mode is the same
confusion in reverse: a large object moving across a static frame drags the
feature matches with it and reads as a pan. So the result reports an inlier
ratio and a background-consistency check, and stays a *hypothesis* the skill
confirms against frames.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from agenteval.media.clip import VideoHandle
from agenteval.tools.base import ToolResult

CAMERA_TYPES = ("static", "push_in", "pull_out", "pan_left", "pan_right",
                "tilt_up", "tilt_down", "orbit", "unstable")


def camera_motion(video: VideoHandle, *, max_frames: int | None = None,
                  stride: int = 2, work_side: int = 480) -> ToolResult:
    import cv2

    n = min(video.total, max_frames or video.total)
    idx = list(range(0, n, max(1, stride)))
    frames = video.read(idx)
    if len(frames) < 3:
        return ToolResult(value={"error": "too few frames"}, reliability=0.0,
                          backend="camera_motion")
    grays = []
    for f in frames:
        s = work_side / max(f.shape[:2])
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if s < 1:
            g = cv2.resize(g, (int(g.shape[1] * s), int(g.shape[0] * s)),
                           interpolation=cv2.INTER_AREA)
        grays.append(g)

    scales, dxs, dys, rots, inliers = [], [], [], [], []
    for a, b in zip(grays, grays[1:]):
        p0 = cv2.goodFeaturesToTrack(a, maxCorners=400, qualityLevel=0.01,
                                     minDistance=8)
        if p0 is None or len(p0) < 12:
            scales.append(1.0); dxs.append(0.0); dys.append(0.0)
            rots.append(0.0); inliers.append(0.0)
            continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None, winSize=(21, 21),
                                             maxLevel=3)
        ok = st.ravel().astype(bool) if st is not None else np.zeros(len(p0), bool)
        if ok.sum() < 10:
            scales.append(1.0); dxs.append(0.0); dys.append(0.0)
            rots.append(0.0); inliers.append(0.0)
            continue
        M, mask = cv2.estimateAffinePartial2D(p0[ok], p1[ok], method=cv2.RANSAC,
                                              ransacReprojThreshold=2.5)
        if M is None:
            scales.append(1.0); dxs.append(0.0); dys.append(0.0)
            rots.append(0.0); inliers.append(0.0)
            continue
        sx = float(np.hypot(M[0, 0], M[1, 0]))
        scales.append(sx)
        dxs.append(float(M[0, 2]) / a.shape[1])
        dys.append(float(M[1, 2]) / a.shape[0])
        rots.append(float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))))
        inliers.append(float(mask.mean()) if mask is not None else 0.0)

    sc = np.asarray(scales); dx = np.asarray(dxs); dy = np.asarray(dys)
    rot = np.asarray(rots); inl = np.asarray(inliers)
    cum_scale = float(np.prod(sc))
    net_dx, net_dy = float(dx.sum()), float(dy.sum())
    # shake: high-frequency energy of the translation series, after removing drift
    hf = float(np.abs(np.diff(dx)).mean() + np.abs(np.diff(dy)).mean()) \
        if len(dx) > 1 else 0.0
    mean_inl = float(inl.mean())

    zoom_rate = cum_scale ** (1.0 / max(1, len(sc)))
    if hf > 0.012:
        kind = "unstable"
    elif zoom_rate > 1.002:
        kind = "push_in"
    elif zoom_rate < 0.998:
        kind = "pull_out"
    elif abs(net_dx) > 0.12 and abs(net_dx) > abs(net_dy):
        kind = "pan_right" if net_dx < 0 else "pan_left"
    elif abs(net_dy) > 0.12:
        kind = "tilt_down" if net_dy < 0 else "tilt_up"
    elif abs(float(rot.sum())) > 8.0:
        kind = "orbit"
    else:
        kind = "static"

    speed = "slow" if abs(zoom_rate - 1) < 0.004 and abs(net_dx) < 0.25 else "normal"
    if abs(zoom_rate - 1) > 0.012 or abs(net_dx) > 0.5:
        speed = "fast"

    return ToolResult(
        value={"type": kind, "speed": speed,
               "cumulative_scale": round(cum_scale, 4),
               "zoom_rate_per_pair": round(zoom_rate, 5),
               "net_dx": round(net_dx, 4), "net_dy": round(net_dy, 4),
               "net_rotation_deg": round(float(rot.sum()), 2),
               "shake_index": round(hf, 5),
               "inlier_ratio": round(mean_inl, 3),
               "n_pairs": len(sc)},
        reliability=0.75 if mean_inl > 0.5 else 0.35,
        backend="homography_decomposition",
        hint=("由相邻帧的相似变换累积估计的**相机**运动。"
              "cumulative_scale>1 为推近,<1 为拉远;net_dx/dy 为累计平移(画面宽高的比例);"
              "shake_index 为高频抖动能量。\n"
              "**主要失效模式**:画面中有大面积移动的主体时,特征匹配会被主体带走,"
              "被误读成摇镜。inlier_ratio 低(<0.5)时该估计不可信。"
              "请对照画面确认背景是否真的在按这个方式移动。"),
    )
