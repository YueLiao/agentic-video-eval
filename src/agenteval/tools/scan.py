"""Systematic coverage — looking at everything, affordably.

The suspicion map is a *temporal* index: it finds moments where change cannot be
explained. Half the defect taxonomy is not temporal at all. A six-fingered hand,
a melted chair, a garbled sign are wrong in a single still; no amount of
cross-frame analysis surfaces them, and a motion-compensated residual is blind
to a defect that sits there perfectly stably.

For that half the best available detector is the VLM itself. No CV signal
recognises "this chair has impossible geometry", and building a classifier per
defect type would be both worse and endless. So static defects get a different
index: sweep the frames and let the model flag them.

This is affordable precisely because the target is short video. A 5-15 s clip is
80-360 frames; at 6-9 frames per contact sheet that is 10-40 sheets, and a
coarse pass over all of them costs less than a handful of careless zooms. The
selective-search assumption is inherited from long-video understanding, where
exhaustive coverage is genuinely impossible, and it does not transfer here.

The pattern is coarse-to-fine: sweep everything cheaply, then spend resolution
only where the sweep pointed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from agenteval.media.clip import VideoHandle, bgr_to_jpeg_bytes
from agenteval.tools.base import ToolResult
from agenteval.tools.renders import _label, _resize_side, _tile, _write


def frame_batches(total: int, *, batch: int = 8, stride: int = 1,
                  overlap: int = 0) -> list[list[int]]:
    """Partition every frame index into batches, optionally overlapping.

    Overlap exists so a defect straddling a batch boundary is seen whole by at
    least one batch; without it, boundary frames are the systematic blind spot
    of an otherwise exhaustive sweep.
    """
    idx = list(range(0, total, max(1, stride)))
    if not idx:
        return []
    step = max(1, batch - max(0, overlap))
    return [idx[i:i + batch] for i in range(0, len(idx), step) if idx[i:i + batch]]


def contact_sheet(video: VideoHandle, out_dir: Path, indices: Sequence[int],
                  *, side: int = 360, cols: int = 3,
                  tag: str = "sheet") -> ToolResult:
    """One labelled grid of the given frames, for a coarse pass.

    Resolution here is deliberately moderate: the sweep's job is to say *which
    frame and roughly where*, not to settle anything. Spending native resolution
    on every frame would cost the same as looking carefully at all of them,
    which is what the cascade exists to avoid.
    """
    idx = list(indices)
    frames = video.read(idx)
    if not frames:
        return ToolResult(value={"error": "no frames"}, reliability=0.0)
    tiles = [_label(_resize_side(f, side), f"f{i}") for i, f in zip(idx, frames)]
    p = _write(out_dir / tag, f"{tag}_{idx[0]:04d}_{idx[-1]:04d}", _tile(tiles, cols))
    return ToolResult(
        value={"indices": idx, "n": len(idx), "cols": cols},
        images=[p], reliability=1.0, backend="contact_sheet",
        hint=(f"第 {idx[0]}-{idx[-1]} 帧的接触印相表,每格左上角标了帧号。"
              "**逐格独立检查**每一帧内部的结构问题(手、脸、肢体、物体几何、文字),"
              "这类问题在单帧内就能判定,不需要看前后帧。"
              "这是粗筛:只报出帧号和大致区域,细节留给后续放大确认。"),
    )


def tile_frame(video: VideoHandle, out_dir: Path, *, t: int, grid: tuple[int, int] = (2, 2),
               side: int = 448, tag: str = "tiles") -> ToolResult:
    """One frame cut into a grid, each tile upscaled to native-ish resolution.

    For finding small defects whose location is not known in advance. A 720p
    frame downsampled whole into an image budget loses everything below a few
    dozen pixels; quartered, each piece keeps four times the detail, and small
    structural failures anywhere in the frame become visible without having to
    guess where to crop.
    """
    import cv2
    t = max(0, min(video.total - 1, int(t)))
    fr = video.read([t])
    if not fr:
        return ToolResult(value={"error": "no frame"}, reliability=0.0)
    img = fr[0]
    h, w = img.shape[:2]
    gy, gx = grid
    paths, boxes = [], []
    for i in range(gy):
        for j in range(gx):
            y0, y1 = int(i * h / gy), int((i + 1) * h / gy)
            x0, x1 = int(j * w / gx), int((j + 1) * w / gx)
            reg = img[y0:y1, x0:x1]
            s = side / max(reg.shape[:2])
            if s > 1:
                reg = cv2.resize(reg, (int(reg.shape[1] * s), int(reg.shape[0] * s)),
                                 interpolation=cv2.INTER_CUBIC)
            bb = [round(x0 / w, 4), round(y0 / h, 4),
                  round((x1 - x0) / w, 4), round((y1 - y0) / h, 4)]
            boxes.append(bb)
            paths.append(_write(out_dir / tag, f"{tag}_t{t:04d}_r{i}c{j}",
                                _label(reg, f"f{t} [{i},{j}]")))
    return ToolResult(
        value={"t": t, "grid": [gy, gx], "boxes": boxes, "n_tiles": len(paths)},
        images=paths, reliability=1.0, backend="tile_frame",
        hint=(f"第 {t} 帧被切成 {gy}x{gx} 块,每块单独放大。"
              "格标 [行,列] 从左上角 [0,0] 开始。"
              "用于在不预先知道位置的情况下查找画面任意处的细小结构问题。"),
    )


def sweep_plan(total: int, *, budget_calls: int = 12, batch: int = 9,
               ensure_full_coverage: bool = True) -> dict[str, object]:
    """Work out how to cover a clip within a call budget.

    Prefers full coverage at a coarser stride over partial coverage at full
    density: a defect in an unvisited frame cannot be found later, whereas a
    defect visited at stride 2 usually persists long enough to be caught.
    """
    if total <= 0:
        return {"batches": [], "stride": 1, "covered": 0.0}
    stride = 1
    if ensure_full_coverage:
        while (total / stride) / batch > budget_calls and stride < 8:
            stride += 1
    batches = frame_batches(total, batch=batch, stride=stride, overlap=1)
    batches = batches[:budget_calls]
    seen = {i for b in batches for i in b}
    return {"batches": batches, "stride": stride,
            "n_calls": len(batches),
            "covered": round(len(seen) / total, 3)}


def change_profile(video: VideoHandle, *, max_side: int = 128,
                   stride: int = 1) -> np.ndarray:
    """Cheap per-adjacent-pair visual change, used to budget the sweep.

    Downscaled greyscale absolute difference. Crude on purpose: this only has to
    rank which stretches of the clip carry new information, and it must cost far
    less than the sweep it is budgeting for.
    """
    import cv2
    idx = list(range(0, video.total, max(1, stride)))
    g = video.read_gray(idx, max_side=max_side)
    if len(g) < 2:
        return np.zeros(0, np.float32)
    d = np.abs(g[1:].astype(np.float32) - g[:-1].astype(np.float32)).mean(axis=(1, 2))
    return d.astype(np.float32)


def adaptive_sweep_plan(video: VideoHandle, *, budget_calls: int = 10,
                        batch: int = 9, max_gap: int = 12,
                        dedup_eps: float = 0.6) -> dict[str, object]:
    """Cover the clip by equal *visual change* rather than equal frame count.

    Uniform stride spends the budget where nothing is happening. A locked-off
    shot with a still subject can hold dozens of near-identical frames, and
    sampling them at stride 1 buys nothing, while a fast passage at the same
    stride still under-samples the part that actually changes.

    So samples are placed at equal increments of cumulative change: dense where
    content moves, sparse where it does not. Two guards keep it honest --
    ``max_gap`` forces a sample at least every N frames however static the clip
    looks, so a defect appearing in a still passage cannot be skipped entirely,
    and near-duplicate frames are dropped since re-showing them adds cost and no
    information.

    Reports ``perceptual_coverage``: the fraction of total visual change that
    falls between adjacent chosen samples of bounded size. That is the honest
    measure -- covering 25% of *frames* in a static clip may be covering ~100%
    of what there is to see.
    """
    total = video.total
    if total <= 0:
        return {"batches": [], "indices": [], "perceptual_coverage": 0.0}

    d = change_profile(video)
    if d.size == 0:
        return {"batches": [[0]], "indices": [0], "perceptual_coverage": 1.0,
                "frame_coverage": 1.0 / max(1, total), "n_calls": 1}

    capacity = budget_calls * batch
    cum = np.concatenate([[0.0], np.cumsum(d)])
    total_change = float(cum[-1]) or 1e-6

    # equal-change placement
    n = max(2, min(capacity, total))
    targets = np.linspace(0, total_change, n)
    picks = sorted({int(np.searchsorted(cum, t)) for t in targets})
    picks = [min(total - 1, max(0, p)) for p in picks]

    # guard: never leave a gap longer than max_gap
    filled: list[int] = []
    prev = -max_gap
    for p in sorted(set(picks)):
        while p - prev > max_gap:
            prev = min(prev + max_gap, total - 1)
            filled.append(prev)
        filled.append(p)
        prev = p
    while total - 1 - prev > max_gap:
        prev = min(prev + max_gap, total - 1)
        filled.append(prev)
    idx = sorted(set(i for i in filled if 0 <= i < total))

    # drop near-duplicates: consecutive picks with negligible change between them
    kept: list[int] = []
    for i in idx:
        if not kept:
            kept.append(i); continue
        seg = float(cum[min(i, len(cum) - 1)] - cum[min(kept[-1], len(cum) - 1)])
        if seg >= dedup_eps or i - kept[-1] >= max_gap:
            kept.append(i)
    if kept[-1] != idx[-1]:
        kept.append(idx[-1])
    kept = kept[:capacity]

    batches = [kept[i:i + batch] for i in range(0, len(kept), batch)][:budget_calls]
    flat = [i for b in batches for i in b]
    # change actually bracketed by adjacent samples
    covered = 0.0
    for a, b in zip(flat, flat[1:]):
        seg = float(cum[min(b, len(cum) - 1)] - cum[min(a, len(cum) - 1)])
        if b - a <= max_gap:
            covered += seg
    return {
        "batches": batches, "indices": flat, "n_calls": len(batches),
        "stride_equivalent": round(total / max(1, len(flat)), 2),
        "frame_coverage": round(len(flat) / total, 3),
        "perceptual_coverage": round(min(1.0, covered / total_change), 3),
        "max_gap": max_gap,
    }
