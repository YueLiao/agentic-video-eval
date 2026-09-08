"""Physical plausibility — geometric invariants, no learned models.

Three checks, all of the same shape: state something that is true of every real
scene, measure how far the video departs from it, and hand the departure to the
judge as a place to look. Being geometric rather than learned, they do not go
stale when generators improve, and they cannot be satisfied by matching a
training distribution.

``rigidity``      a rigid body's points keep their pairwise distances. The
                  object-level analogue of constant bone length.
``free fall``     vertical velocity under gravity is a straight line in time. A
                  mid-air stall is a flat segment no real trajectory produces.
``permanence``    objects do not blink out and return.

The shared caveat, which must reach the judge every time: all three run on
tracked points, and a tracker that drifts onto the background produces exactly
the same numbers as a genuinely deforming object. So these locate suspicion;
they never convict on their own. Every result here is paired with a view that
lets the model check the tracking with its own eyes.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from agenteval.media.clip import VideoHandle
from agenteval.tools.base import ToolResult


def track_points(video: VideoHandle, *, bbox, t_span, max_points: int = 60,
                 stride: int = 1) -> ToolResult:
    """Track good features inside a region with Lucas-Kanade.

    Returns per-point trajectories plus the fraction of points that survived.
    Survival rate is itself informative: points fail to track where content
    changes in ways motion cannot explain, so a low survival rate on a slow
    shot is a defect signal rather than merely a failed measurement.
    """
    import cv2

    t0, t1 = int(t_span[0]), min(int(t_span[1]), video.total)
    if t1 - t0 < 3:
        return ToolResult(value={"error": "span too short"}, reliability=0.0,
                          backend="track_points")
    idx = list(range(t0, t1, max(1, stride)))
    frames = video.read(idx)
    if len(frames) < 3:
        return ToolResult(value={"error": "too few frames"}, reliability=0.0,
                          backend="track_points")
    h, w = frames[0].shape[:2]
    x, y, bw, bh = bbox
    x0, y0 = int(x * w), int(y * h)
    x1, y1 = int(min(w, (x + bw) * w)), int(min(h, (y + bh) * h))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return ToolResult(value={"error": "bbox too small"}, reliability=0.0,
                          backend="track_points")

    g0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(g0); mask[y0:y1, x0:x1] = 255
    p0 = cv2.goodFeaturesToTrack(g0, maxCorners=max_points, qualityLevel=0.01,
                                 minDistance=6, mask=mask)
    if p0 is None or len(p0) < 4:
        return ToolResult(value={"error": "not enough trackable features",
                                 "n_points": 0 if p0 is None else len(p0)},
                          reliability=0.0, backend="track_points",
                          hint="该区域纹理太弱,无法跟踪。这本身可能意味着区域过度平滑。")

    tracks = [[[float(p0[i, 0, 0] / w), float(p0[i, 0, 1] / h)]]
              for i in range(len(p0))]
    alive = np.ones(len(p0), bool)
    prev_g, prev_p = g0, p0
    for fr in frames[1:]:
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_g, g, prev_p, None,
                                             winSize=(21, 21), maxLevel=3)
        st = st.ravel().astype(bool) if st is not None else np.zeros(len(prev_p), bool)
        for i in range(len(prev_p)):
            if alive[i] and st[i]:
                tracks[i].append([float(p1[i, 0, 0] / w), float(p1[i, 0, 1] / h)])
            elif alive[i]:
                alive[i] = False
        prev_g, prev_p = g, p1
    kept = [t for t, a in zip(tracks, alive) if a and len(t) == len(frames)]
    surv = len(kept) / max(1, len(tracks))
    return ToolResult(
        value={"n_points": len(kept), "n_seeded": len(tracks),
               "survival_rate": round(surv, 3), "indices": idx,
               "tracks": [[[round(v, 5) for v in pt] for pt in t] for t in kept[:40]]},
        reliability=0.6 if surv > 0.4 else 0.3, backend="lucas_kanade",
        hint=(f"在指定区域播种 {len(tracks)} 个特征点,{len(kept)} 个跟满全程"
              f"(存活率 {surv:.0%})。存活率低本身也是信号:"
              "内容以运动解释不了的方式改变时,点就跟丢了。"
              "**但跟踪可能漂到背景上**,请用画面确认跟的是同一个物体。"),
    )


def rigidity_check(track_result: ToolResult) -> ToolResult:
    """How far a set of tracked points departs from moving as one rigid body.

    For a rigid object the distance between any two of its surface points is
    constant, whatever the camera does. The coefficient of variation of those
    pairwise distances is therefore near zero for real rigid content and rises
    when the object stretches, melts or reflows -- a very common generation
    failure that no intensity statistic detects.

    Scale changes (an object approaching the camera) inflate every distance
    together, so distances are normalized by the frame-wise mean distance before
    the statistic is taken; that removes uniform scaling and leaves genuine
    non-rigid deformation.
    """
    tracks = track_result.value.get("tracks") or []
    if len(tracks) < 4:
        return ToolResult(value={"error": "need >=4 tracks"}, reliability=0.0,
                          backend="rigidity_check")
    arr = np.asarray(tracks, np.float64)            # (P, T, 2)
    P, T = arr.shape[0], arr.shape[1]
    iu = np.triu_indices(P, k=1)
    d = np.linalg.norm(arr[iu[0]] - arr[iu[1]], axis=2)   # (pairs, T)
    scale = d.mean(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1e-8
    dn = d / scale                                   # remove uniform scaling
    cv = dn.std(axis=1) / np.maximum(dn.mean(axis=1), 1e-8)
    worst_t = int(np.argmax(dn.std(axis=0)))
    return ToolResult(
        value={"n_points": P, "n_frames": T,
               "pairwise_cv_median": round(float(np.median(cv)), 4),
               "pairwise_cv_p90": round(float(np.percentile(cv, 90)), 4),
               "most_deforming_frame_offset": worst_t,
               "scale_change": round(float(scale.max() / max(scale.min(), 1e-8)), 3)},
        reliability=float(track_result.reliability) * 0.9,
        backend="rigidity_check",
        hint=("刚体上任意两点的距离恒定(已归一化掉整体缩放)。"
              "pairwise_cv 中位数 <0.05 属正常刚体;>0.15 表示物体在非刚性地变形;"
              ">0.30 通常是明显的融化/流动/结构崩坏。"
              "**注意:柔性物体(布料、头发、水、烟)本来就该非刚性形变**,"
              "对它们这个指标没有意义。先确认被跟踪的是不是刚体。"),
    )


def freefall_check(track_result: ToolResult, *, fps: float = 24.0,
                   point_index: int | None = None) -> ToolResult:
    """Test whether vertical motion is consistent with constant acceleration.

    Under gravity v_y is linear in time. Fitting a line and looking at the
    residual separates real ballistic motion from the two failures generators
    make most: hanging in mid-air (a flat run) and drifting upward unsupported
    (a sign flip).
    """
    tracks = track_result.value.get("tracks") or []
    if not tracks:
        return ToolResult(value={"error": "no tracks"}, reliability=0.0,
                          backend="freefall_check")
    arr = np.asarray(tracks, np.float64)
    centroid = arr.mean(axis=0) if point_index is None else arr[point_index]
    y = centroid[:, 1]
    if len(y) < 5:
        return ToolResult(value={"error": "track too short"}, reliability=0.0,
                          backend="freefall_check")
    vy = np.diff(y) * fps
    t = np.arange(len(vy), dtype=np.float64)
    coef = np.polyfit(t, vy, 1)
    resid = float(np.sqrt(((vy - np.polyval(coef, t)) ** 2).mean()))
    ratio = resid / max(float(np.std(vy)), 1e-6)
    stall = [int(i) for i, v in enumerate(vy) if abs(v) < 0.02 * max(abs(vy).max(), 1e-6)]
    return ToolResult(
        value={"vy_accel_per_s2": round(float(coef[0]) * fps, 4),
               "vy_linear_residual": round(resid, 4),
               "vy_residual_ratio": round(ratio, 4),
               "n_near_stall_steps": len(stall),
               "downward_positive": True},
        reliability=float(track_result.reliability) * 0.8,
        backend="freefall_check",
        hint=("图像坐标 y 向下为正。自由落体时 vy 随时间线性增大,"
              "residual_ratio 应接近 0,accel 为正且大致恒定。"
              "中途的水平段=物体在空中停住;accel 显著为负=无支撑地上升。"
              "**只有当该物体确实处于自由落体状态时这个检验才成立**——"
              "有人托着、有绳吊着、或本来就是飞行物,都不适用。"),
    )


def permanence_check(per_frame_counts: Sequence[int],
                     indices: Sequence[int] | None = None) -> ToolResult:
    """Flag objects blinking out and returning.

    Counts alone are noisy -- a detector loses things behind occlusion all the
    time -- so what is reported is the *return* pattern: a count that drops and
    comes back is far more suspicious than one that simply drops, because real
    occlusion usually resolves gradually while generative dropout snaps back.
    """
    c = np.asarray(list(per_frame_counts), int)
    if len(c) < 3:
        return ToolResult(value={"error": "too few frames"}, reliability=0.0,
                          backend="permanence_check")
    idx = list(indices) if indices is not None else list(range(len(c)))
    mode = int(np.bincount(c[c >= 0]).argmax()) if (c >= 0).any() else 0
    events = []
    for i in range(1, len(c) - 1):
        if c[i] < mode and c[i + 1] >= mode and c[i - 1] >= mode:
            events.append({"frame": idx[i], "count": int(c[i]), "expected": mode})
    return ToolResult(
        value={"modal_count": mode, "counts": c.tolist()[:60],
               "blink_events": events[:20], "n_blink_events": len(events),
               "count_varies": bool(len(set(c.tolist())) > 1)},
        reliability=0.5, backend="permanence_check",
        hint=("统计目标数量随时间的变化。报告的是**消失后又出现**的帧"
              "(前后都正常、中间少了),这比单纯的数量下降更可疑——"
              "真实遮挡通常是渐进的,生成性丢失往往是瞬间弹回。"
              "**检测器漏检也会产生同样的模式**,必须回到画面确认。"),
    )
