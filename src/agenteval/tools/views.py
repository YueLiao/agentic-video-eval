"""Views — moving evidence into the model's input.

These exist because of the resolution limit. A 0.2x0.2 region of a 720p frame,
after the frame is downsampled into a VLM's image budget, is a few dozen pixels:
the evidence needed to judge it is not merely overlooked, it is absent. Cropping
that region and upscaling it puts the pixels back.

Every view writes JPEGs under the run directory and returns them as images on a
ToolResult, so the judge sees them and the trajectory can be replayed and
inspected afterwards.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from agenteval.media.clip import VideoHandle, bgr_to_jpeg_bytes, uniform_indices
from agenteval.tools.base import ToolResult


def _write(out_dir: Path, stem: str, frame: np.ndarray, quality: int = 92) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{stem}.jpg"
    p.write_bytes(bgr_to_jpeg_bytes(frame, quality))
    return p


def _crop(frame: np.ndarray, bbox, pad: float, upscale: int) -> np.ndarray:
    import cv2
    h, w = frame.shape[:2]
    x, y, bw, bh = bbox
    cx, cy = x + bw / 2, y + bh / 2
    bw, bh = bw * (1 + pad), bh * (1 + pad)
    # keep it square so upscaling does not distort aspect
    side = max(bw * w, bh * h)
    x0 = int(max(0, min(w - 1, cx * w - side / 2)))
    y0 = int(max(0, min(h - 1, cy * h - side / 2)))
    x1 = int(min(w, x0 + side)); y1 = int(min(h, y0 + side))
    reg = frame[y0:y1, x0:x1]
    if reg.size == 0:
        reg = frame
    s = upscale / max(reg.shape[:2])
    if s > 1:
        reg = cv2.resize(reg, (int(reg.shape[1] * s), int(reg.shape[0] * s)),
                         interpolation=cv2.INTER_CUBIC)
    return reg


def overview(video: VideoHandle, out_dir: Path, n: int = 8) -> ToolResult:
    """Uniform frames spanning the clip. Always included alongside any targeted
    view, so a judge shown a cherry-picked worst crop still has global context
    and cannot mistake one bad region for a bad video."""
    idx = uniform_indices(video.total, n)
    frames = video.read(idx)
    paths = [_write(out_dir / "overview", f"t{i:04d}", f) for i, f in zip(idx, frames)]
    return ToolResult(
        value={"indices": idx, "n": len(paths)},
        images=paths, reliability=1.0, backend="overview",
        hint=f"整段视频的 {len(paths)} 帧均匀采样,帧号见文件名。用于建立全局印象。",
    )


def zoom(video: VideoHandle, out_dir: Path, *, bbox, t_span, n: int = 6,
         upscale: int = 448, pad: float = 0.35, tag: str = "zoom") -> ToolResult:
    """Native-resolution crop of a region across a time span, upscaled."""
    t0, t1 = int(t_span[0]), int(t_span[1])
    t1 = max(t1, t0 + 1)
    idx = uniform_indices(t1 - t0, min(n, t1 - t0))
    idx = [min(video.total - 1, t0 + i) for i in idx]
    frames = video.read(idx)
    paths = [_write(out_dir / tag, f"{tag}_t{i:04d}", _crop(f, bbox, pad, upscale))
             for i, f in zip(idx, frames)]
    return ToolResult(
        value={"indices": idx, "bbox": [round(v, 4) for v in bbox],
               "t_span": [t0, t1], "upscale": upscale},
        images=paths, reliability=1.0, backend="zoom",
        hint=(f"区域 {[round(v,2) for v in bbox]} 在第 {t0}-{t1} 帧的原生分辨率裁剪,"
              f"已放大到 {upscale}px。这些细节在整帧缩略图里是看不到的。"),
    )


def dense_window(video: VideoHandle, out_dir: Path, *, t0: int, t1: int,
                 n: int = 8) -> ToolResult:
    """Consecutive full frames in a narrow window — for judging *when* something
    happens, which uniform sampling across the whole clip cannot resolve."""
    t0 = max(0, int(t0)); t1 = min(video.total, max(int(t1), t0 + 2))
    step = max(1, (t1 - t0) // max(1, n))
    idx = list(range(t0, t1, step))[:n]
    frames = video.read(idx)
    paths = [_write(out_dir / "window", f"w_t{i:04d}", f) for i, f in zip(idx, frames)]
    return ToolResult(
        value={"indices": idx, "t_span": [t0, t1]},
        images=paths, reliability=1.0, backend="dense_window",
        hint=f"第 {t0}-{t1} 帧的连续采样,帧号见文件名,用于判断事件发生的确切时刻。",
    )


def contrast_pair(video: VideoHandle, out_dir: Path, *, bbox, t_span,
                  upscale: int = 448, pad: float = 0.35) -> ToolResult:
    """The same region at the suspicious time and at a distant, quiet time.

    This is what makes a defect claim falsifiable: if the 'defect' looks the
    same in the reference crop, it is how this video renders that region, not
    an event. Without a contrast the judge has no way to tell the two apart.
    """
    t0, t1 = int(t_span[0]), int(t_span[1])
    mid = min(video.total - 1, (t0 + t1) // 2)
    span = max(1, t1 - t0)
    ref = mid + 3 * span
    if ref >= video.total:
        ref = max(0, mid - 3 * span)
    frames = video.read([mid, ref])
    if len(frames) < 2:
        return ToolResult(value={"error": "cannot build contrast"}, reliability=0.0)
    sus = _write(out_dir / "contrast", f"suspect_t{mid:04d}",
                 _crop(frames[0], bbox, pad, upscale))
    nor = _write(out_dir / "contrast", f"reference_t{ref:04d}",
                 _crop(frames[1], bbox, pad, upscale))
    return ToolResult(
        value={"suspect_frame": mid, "reference_frame": ref,
               "bbox": [round(v, 4) for v in bbox]},
        images=[sus, nor], reliability=1.0, backend="contrast_pair",
        hint=(f"同一区域的对照:第 {mid} 帧(可疑)与第 {ref} 帧(参照)。"
              "若两者表现一致,说明这是该视频渲染该区域的常态,不是缺陷。"),
    )
