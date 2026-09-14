"""Depth, because penetration is a depth-ordering violation and nothing else.

Penetration is the third-largest defect category in the in-house taxonomy (57
items) and the branch that claimed it had no way to judge it: a side view of an
otter and a basket cannot distinguish passing through from passing behind, and
the audit found exactly that -- every penetration claim came back unconfirmable
from the evidence offered for it.

Ordering is what settles it. If the near object's depth stays in front while its
silhouette overlaps the far one, that is occlusion; if the ordering inverts
inside the overlap while the objects stay adjacent, that is penetration.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from agenteval.media.clip import VideoHandle
from agenteval.tools.base import ToolResult
from agenteval.tools.renders import _label, _tile, _write


@lru_cache(maxsize=1)
def _pipe(device: int = 0):
    from transformers import pipeline
    return pipeline("depth-estimation",
                    model="depth-anything/Depth-Anything-V2-Small-hf",
                    device=device)


def depth_pair(video: VideoHandle, out_dir: Path, *, t_span, bbox=None,
               n: int = 4, side: int = 420, tag: str = "depth") -> ToolResult:
    """Frames and their depth maps side by side over a span.

    The depth map is shown next to the frame rather than instead of it: the
    model needs to see what the objects are before an ordering means anything,
    and a depth map alone is unreadable as content.
    """
    import cv2
    from PIL import Image

    t0, t1 = int(t_span[0]), int(t_span[1])
    t0 = max(0, min(t0, video.total - 2))
    t1 = max(t0 + 2, min(t1, video.total))
    idx = sorted({min(video.total - 1, int(t))
                  for t in np.linspace(t0, t1 - 1, min(n, t1 - t0))})
    frames = video.read(idx)
    if not frames:
        return ToolResult(value={"error": "decode failed"}, reliability=0.0)

    pipe = _pipe()
    rows, stats = [], []
    for i, f in zip(idx, frames):
        if bbox:
            h, w = f.shape[:2]
            x, y, bw, bh = bbox
            pad = 0.25
            x0 = int(max(0, (x - bw * pad) * w)); x1 = int(min(w, (x + bw * (1 + pad)) * w))
            y0 = int(max(0, (y - bh * pad) * h)); y1 = int(min(h, (y + bh * (1 + pad)) * h))
            crop = f[y0:y1, x0:x1]
        else:
            crop = f
        if crop.size == 0:
            continue
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        d = np.asarray(pipe(Image.fromarray(rgb))["depth"], dtype=np.float32)
        dn = (d - d.min()) / max(d.max() - d.min(), 1e-6)
        vis = cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        s = side / max(crop.shape[:2])
        rz = lambda im: cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)),
                                   interpolation=cv2.INTER_CUBIC)
        rows.append(_tile([_label(rz(crop), f"f{i} 画面"),
                           _label(rz(vis), f"f{i} 深度")], 2))
        stats.append({"frame": int(i), "near_frac": float((dn > 0.6).mean())})
    if not rows:
        return ToolResult(value={"error": "no frames"}, reliability=0.0)
    p = _write(out_dir / tag, f"{tag}_{video.path.stem}_{t0}_{t1}"[:110],
               _tile(rows, 1))
    return ToolResult(
        value={"indices": idx, "per_frame": stats},
        images=[p], reliability=0.7, backend="depth_pair",
        hint=("每一行左边是原画面,右边是同一画面的**深度图**:"
              "**越偏红/黄 = 离镜头越近,越偏蓝/紫 = 越远**。\n"
              "判据:两个物体的轮廓重叠时——\n"
              "· **遮挡(正常)**:近的那个在深度图上始终整片偏红,远的整片偏蓝,"
              "边界清晰,重叠处颜色取近者;\n"
              "· **穿模(缺陷)**:重叠处的深度出现**穿插或反转**——本该在后面的部分"
              "在深度图上跑到了前面,或同一物体被另一物体从中间切开成前后两段。\n"
              "深度图本身有噪声,**只有当前后关系明确反转时才算穿模**,边界模糊不算。"),
    )
