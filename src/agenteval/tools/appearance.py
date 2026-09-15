"""Did it enter from outside the frame, or appear where it stood?

Half of an 'appeared from nothing' claim is geometry and the model gets it
wrong. Audited on real clips, two of two confirmed claims were objects entering
the frame normally -- a glass falling in from the top edge, a dolphin surfacing
through water -- and in both the model's own written reason asserted the
opposite ("not entering from the frame edge"), with the rule that would have
excluded it stated in its prompt.

That half is measurable. Where new content first shows up, and how far that is
from the frame boundary, is arithmetic. What remains semantic is whether
something in the scene was covering the spot -- water, fog, another object -- and
that stays with the model. This is the same split that made the temporal branch
work: numbers say where, the model says whether it counts.
"""

from __future__ import annotations

import numpy as np

from agenteval.media.clip import VideoHandle
from agenteval.tools.base import ToolResult


def entry_check(video: VideoHandle, *, t_norm: float, bbox=None,
                span_frac: float = 0.25, edge_margin: float = 0.06,
                max_side: int = 320) -> ToolResult:
    """Find where new content first appears near `t_norm`, and how close to the edge.

    `edge_margin` is a fraction of the frame. An object whose first appearance
    touches within that band of the border entered the frame, which is ordinary
    and not a defect however abrupt it looks.
    """
    import cv2

    T = video.total
    if T < 8:
        return ToolResult(value={"error": "too few frames"}, reliability=0.0)
    half = max(3, int(T * span_frac / 2))
    centre = int(min(max(t_norm, 0.0), 1.0) * (T - 1))
    t0, t1 = max(0, centre - half), min(T, centre + half + 1)
    idx = list(range(t0, t1))
    g = video.read_gray(idx, max_side=max_side)
    if len(g) < 4:
        return ToolResult(value={"error": "decode failed"}, reliability=0.0)
    H, W = g[0].shape[:2]

    # The onset is searched over the whole frame, never inside the claim's box.
    # Cropping to the claim first makes `from_edge` unreachable by construction:
    # a component found inside a crop that does not touch the border cannot
    # touch the border, and the check then calls every entry an interior
    # appearance. Measured against a glass falling in from the top edge, the
    # cropped version reported an edge distance of 0.266.
    diffs = [cv2.absdiff(a, b) for a, b in zip(g, g[1:])]
    if not diffs:
        return ToolResult(value={"error": "no diffs"}, reliability=0.0)
    # Two different regions do two different jobs. The onset is timed inside
    # the claimed region, because that is the object being asked about; the
    # component is then found over the whole frame, because a crop that does
    # not touch the border can never report an entry. Doing both globally
    # instead timed the onset to whatever moved most -- a wheel, a kicking leg
    # -- and answered a question about that object rather than the claimed one.
    if bbox:
        x, y, bw, bh = bbox
        pad = 0.35
        cx0 = int(max(0, (x - bw * pad) * W)); cx1 = int(min(W, (x + bw * (1 + pad)) * W))
        cy0 = int(max(0, (y - bh * pad) * H)); cy1 = int(min(H, (y + bh * (1 + pad)) * H))
        if cx1 - cx0 < 8 or cy1 - cy0 < 8:
            cx0, cy0, cx1, cy1 = 0, 0, W, H
    else:
        cx0, cy0, cx1, cy1 = 0, 0, W, H
    energy = np.array([float(d[cy0:cy1, cx0:cx1].mean()) for d in diffs])
    onset = int(np.argmax(energy))
    d = diffs[onset]
    thr = max(float(np.percentile(d, 97)), 8.0)
    mask = (d > thr).astype(np.uint8)
    if mask.sum() < 12:
        return ToolResult(
            value={"onset_frame": idx[onset], "verdict": "no_onset",
                   "note": "窗口内没有明显的新内容出现"},
            reliability=0.5, backend="entry_check")
    n, _lab, stats, _c = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return ToolResult(value={"error": "no component"}, reliability=0.0)
    # Prefer the component that overlaps the claim; the largest one is usually
    # the scene's dominant motion and answers about the wrong object.
    def _ov(j):
        sx, sy = stats[j, cv2.CC_STAT_LEFT] / W, stats[j, cv2.CC_STAT_TOP] / H
        sw, sh = stats[j, cv2.CC_STAT_WIDTH] / W, stats[j, cv2.CC_STAT_HEIGHT] / H
        if not bbox:
            return 1.0
        ix = max(0.0, min(sx + sw, bbox[0] + bbox[2]) - max(sx, bbox[0]))
        iy = max(0.0, min(sy + sh, bbox[1] + bbox[3]) - max(sy, bbox[1]))
        return ix * iy
    cand = [j for j in range(1, n) if _ov(j) > 0
            and stats[j, cv2.CC_STAT_AREA] >= 12]
    if cand:
        k = max(cand, key=lambda j: stats[j, cv2.CC_STAT_AREA])
    else:
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cx, cy, cw, ch = (stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP],
                      stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT])
    gx0, gy0 = cx / W, cy / H
    gx1, gy1 = (cx + cw) / W, (cy + ch) / H
    edge_dist = min(gx0, gy0, 1.0 - gx1, 1.0 - gy1)
    from_edge = edge_dist <= edge_margin
    return ToolResult(
        value={"onset_frame": idx[onset],
               "onset_norm": round(idx[onset] / max(1, T - 1), 3),
               "bbox": [round(gx0, 3), round(gy0, 3),
                        round(gx1 - gx0, 3), round(gy1 - gy0, 3)],
               "edge_distance": round(float(edge_dist), 3),
               "from_edge": bool(from_edge),
               "verdict": "entered_from_edge" if from_edge else "interior",
               "overlaps_claim": (None if not bbox else bool(
                   max(0.0, min(gx1, bbox[0] + bbox[2]) - max(gx0, bbox[0])) *
                   max(0.0, min(gy1, bbox[1] + bbox[3]) - max(gy0, bbox[1])) > 0))},
        reliability=0.8, backend="entry_check",
        hint=("测量结果:新内容最早出现在第 %d 帧,其外接框距画面边界 %.3f(占画面比例)。"
              % (idx[onset], edge_dist)
              + ("**它贴着画面边缘出现,说明是从画面外进入的——这是正常的。**"
                 if from_edge else
                 "**它出现在画面内部,不是从画面外进来的。**"
                 "但这还不足以判定为缺陷:仍需确认此处**有没有遮挡物**"
                 "(水面、雾、阴影、另一个物体的背后都算遮挡,"
                 "从遮挡后面出来是正常的)。")),
    )
