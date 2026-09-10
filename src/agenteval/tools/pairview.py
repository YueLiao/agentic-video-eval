"""Side-by-side views for pairwise comparison.

The benchmark is pairwise, and the framework was answering it by scoring each
clip alone and subtracting. That detour introduces a quantity the model gives
badly -- magnitude -- when the question only ever needed a direction. Measured:
direction accuracy 61.8% against 50% random, while overall accuracy including
ties topped out at 38.3% against a 34.5% trivial baseline. The sign carries
signal; the size does not.

So the comparison is put to the model directly, which needs the two clips in one
bundle, sampled at matched times so a difference in the pictures is a difference
in the videos rather than a difference in when they were sampled.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from agenteval.media.clip import VideoHandle, uniform_indices
from agenteval.tools.base import ToolResult
from agenteval.tools.renders import _label, _resize_side, _tile, _write


def aligned_pair(a: VideoHandle, b: VideoHandle, out_dir: Path, *, n: int = 6,
                 side: int = 300, tag: str = "pair") -> ToolResult:
    """Both clips at matched normalized timestamps, stacked A over B.

    Matched by fraction of duration rather than frame index, since the two clips
    can differ in length and fps -- comparing frame 40 of an 81-frame clip with
    frame 40 of a 121-frame clip compares different moments.
    """
    ia = uniform_indices(a.total, n)
    ib = uniform_indices(b.total, n)
    fa, fb = a.read(ia), b.read(ib)
    if not fa or not fb:
        return ToolResult(value={"error": "decode failed"}, reliability=0.0)
    fps_a, fps_b = a.fps or 24.0, b.fps or 24.0
    row_a = [_label(_resize_side(f, side), f"A t={i/fps_a:.1f}s")
             for i, f in zip(ia, fa)]
    row_b = [_label(_resize_side(f, side), f"B t={i/fps_b:.1f}s")
             for i, f in zip(ib, fb)]
    img = _tile(row_a + row_b, len(row_a))
    p = _write(out_dir / tag, f"{tag}_{a.path.stem}__{b.path.stem}"[:120], img)
    return ToolResult(
        value={"n": n, "a_indices": ia, "b_indices": ib,
               "a_duration": round(a.duration_s, 2),
               "b_duration": round(b.duration_s, 2)},
        images=[p], reliability=1.0, backend="aligned_pair",
        hint=("上排是视频 A,下排是视频 B,两排按**相同的时间比例**采样并标注了时间戳,"
              "所以同一列是两段视频的同一时刻。\n"
              "请逐列对比,判断哪一段的运动更合理。"),
    )


def stacked_strips(a: VideoHandle, b: VideoHandle, out_dir: Path, *,
                   n: int = 8, side: int = 260,
                   tag: str = "strips") -> ToolResult:
    """Two separate images, one per clip, for endpoints that handle several
    images better than one dense grid. Kept as an alternative because which
    works better is a per-model fact the capability profile decides."""
    out = []
    for name, v in (("A", a), ("B", b)):
        idx = uniform_indices(v.total, n)
        fps = v.fps or 24.0
        tiles = [_label(_resize_side(f, side), f"{name} t={i/fps:.1f}s")
                 for i, f in zip(idx, v.read(idx))]
        out.append(_write(out_dir / tag, f"{tag}_{name}_{v.path.stem}"[:120],
                          _tile(tiles, 4)))
    return ToolResult(
        value={"n": n}, images=out, reliability=1.0, backend="stacked_strips",
        hint="第一张是视频 A 的时间采样,第二张是视频 B 的。每格标注了时间戳。",
    )
