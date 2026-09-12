"""Two clips as one video, so their order is spatial instead of sequential.

Presented one after the other, the judge's errors are dominated by which clip
came second: 22-23% of pairs flip when the order is swapped, and on the flipped
ones accuracy is 49.6% against 69.9% on the rest. Sampling does not fix it --
six samples at temperature 0.7 returned the same side six times -- because the
preference is stable, not noisy.

Stacking removes the sequence. Both clips occupy the same 32-frame budget and
the model sees them at the same instant, which is also the comparison it is
being asked to make. The cost is real and has to be measured rather than
assumed: each clip now gets half the frame, so per-clip resolution halves.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from agenteval.media.clip import VideoHandle, uniform_indices
from agenteval.tools.base import ToolResult


def split_screen(a: VideoHandle, b: VideoHandle, out_dir: Path, *,
                 layout: str = "vertical", n: int = 48, height: int = 480,
                 label: bool = True, tag: str = "split") -> ToolResult:
    """One clip above the other (or side by side), sampled at matched times.

    Matched by fraction of duration rather than frame index: the two clips can
    differ in length and fps, and frame 40 of an 81-frame clip is not the same
    moment as frame 40 of a 121-frame one.
    """
    import cv2

    ia, ib = uniform_indices(a.total, n), uniform_indices(b.total, n)
    fa, fb = a.read(ia), b.read(ib)
    if not fa or not fb:
        return ToolResult(value={"error": "decode failed"}, reliability=0.0)
    m = min(len(fa), len(fb))
    fa, fb = fa[:m], fb[:m]

    if layout == "vertical":
        w = max(f.shape[1] * height // 2 // max(1, f.shape[0]) for f in (fa[0], fb[0]))
        w = max(64, w - w % 2)
        cell = (w, height // 2)
    else:
        h = height
        w = max(64, (fa[0].shape[1] * h // fa[0].shape[0]))
        w -= w % 2
        cell = (w, h)

    def fit(f):
        return cv2.resize(f, cell, interpolation=cv2.INTER_AREA)

    frames = []
    for x, y in zip(fa, fb):
        top, bot = fit(x), fit(y)
        if label:
            # Neutral marks, not "A"/"B": a letter that also names the answer
            # options invites the model to read the label instead of the clip.
            for img, txt in ((top, "1"), (bot, "2")):
                cv2.rectangle(img, (0, 0), (34, 30), (0, 0, 0), -1)
                cv2.putText(img, txt, (9, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                            (255, 255, 255), 2, cv2.LINE_AA)
        sep = np.full(((4, top.shape[1], 3) if layout == "vertical"
                       else (top.shape[0], 4, 3)), 255, np.uint8)
        frames.append(np.concatenate([top, sep, bot],
                                     axis=0 if layout == "vertical" else 1))

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{tag}_{a.path.stem}__{b.path.stem}"[:110]
    p = out_dir / f"{stem}.mp4"
    if not p.exists():
        h, w = frames[0].shape[:2]
        fps = min(a.fps or 24.0, b.fps or 24.0)
        vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in frames:
            vw.write(f)
        vw.release()
    return ToolResult(
        value={"n": len(frames), "layout": layout,
               "cell": list(cell), "top_is": "a", "path": str(p)},
        images=[], reliability=1.0, backend="split_screen",
        hint=("这**一段视频里同时有两个片段**:"
              + ("上半部分标着「1」,下半部分标着「2」" if layout == "vertical"
                 else "左半部分标着「1」,右半部分标着「2」")
              + ",两者按**相同的时间比例**采样,所以同一时刻看到的是两段的同一进度。\n"
              "请同时观察两半,比较它们的运动质量。"),
    )
