"""Magnified views of the places a signal says are suspicious.

Every attempt so far handed the model more evidence at the same resolution --
8 frames, then 16, then 32, then difference strips and motion curves -- and none
of them moved the seed-family number. The capability probe says why: this model's
smallest legible detail is 448px, and sixteen 400px frames tiled into one image
arrive at roughly 100px each after the encoder's fixed resize. Asked about a
seed pair at that resolution the model answers, honestly, that both clips look
fine: it returns identical answers on 55-57% of them.

So this module spends the pixel budget differently. The suspicion signals run at
full frame rate and nominate a few loci -- a time span and a region; each locus
is then cropped and rendered *large*, one image per locus, at or above the
measured legibility floor. Fewer moments, seen properly.

The signal cannot say what it found; it only says where change is not explained
by motion. Naming it is the model's job, and that division is the point.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from agenteval.media.clip import VideoHandle
from agenteval.signals.suspicion import SuspicionLocus
from agenteval.tools.base import ToolResult
from agenteval.tools.renders import _label, _tile, _write

# What the encoder actually does, from the model's processor_config.json:
# max_soft_tokens 280, patch_size 16, pooling_kernel_size 3. So one pooled cell
# covers 48px and the grid is chosen to fit the aspect ratio within 280 cells --
# the resize is adaptive, not a fixed square. Effective resolution per side is
# therefore about 48 x grid_dim, which makes the layout, not the source size,
# what decides whether a detail survives:
#
#   8-column strip 3242x456 -> 44x6  grid -> 2112x288 -> 264px per frame
#   2x2 tile       1030x1030-> 16x16 grid ->  768x768 -> 384px per crop
#   1x2 pair       1536x768 -> 22x11 grid -> 1056x528 -> 528px per crop
#
# The capability probe put this model's legibility floor at 448px, so the 1x2
# pair is the densest layout that still clears it while keeping two consecutive
# moments in one image -- which the question at a locus needs.
LEGIBLE = 768
PAIR_COLS = 2


def _plain(o):
    """numpy scalars reach here from the frame indices; json refuses them."""
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o).__name__)


def _crop_box(fr: np.ndarray, bbox, pad: float, side: int) -> np.ndarray:
    """Crop a normalized box with context, then scale up to `side`.

    Upscaling a small region is not new information, but it is the difference
    between a detail the encoder resizes away and one it keeps -- the model is
    shown the pixels that exist, at a size where it can use them.
    """
    import cv2
    h, w = fr.shape[:2]
    x, y, bw, bh = bbox
    cx, cy = (x + bw / 2) * w, (y + bh / 2) * h
    half = max(bw * w, bh * h) * (0.5 + pad)
    half = max(half, 48.0)
    x0, x1 = int(max(0, cx - half)), int(min(w, cx + half))
    y0, y1 = int(max(0, cy - half)), int(min(h, cy + half))
    if x1 - x0 < 8 or y1 - y0 < 8:
        x0, y0, x1, y1 = 0, 0, w, h
    patch = fr[y0:y1, x0:x1]
    s = side / max(1, max(patch.shape[:2]))
    interp = cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA
    return cv2.resize(patch, (max(1, int(patch.shape[1] * s)),
                              max(1, int(patch.shape[0] * s))),
                      interpolation=interp)


def locus_strip(video: VideoHandle, locus: SuspicionLocus, out_dir: Path, *,
                n: int = 4, side: int = LEGIBLE, pad: float = 0.6,
                tag: str = "locus") -> ToolResult:
    """One locus over consecutive moments, cropped and magnified.

    Consecutive rather than sampled: the locus is a few frames wide, and the
    question at it -- did this deform, did it jump, did it stall -- is about
    what changes from one frame to the next.
    """
    t0, t1 = locus.t_span
    t0 = max(0, min(int(t0), video.total - 2))
    t1 = max(t0 + 2, min(int(t1), video.total))
    idx = sorted({min(video.total - 1, t) for t in
                  np.linspace(t0, t1 - 1, min(n, t1 - t0)).astype(int)})
    frames = video.read(idx)
    if not frames:
        return ToolResult(value={"error": "decode failed"}, reliability=0.0)
    tiles = [_label(_crop_box(f, locus.bbox, pad, side), f"f{i}", org=(10, 40),
                    scale=1.2) for i, f in zip(idx, frames)]
    # Consecutive moments two to an image: any denser and each crop falls under
    # the legibility floor, any sparser and the model has to carry the
    # comparison across images.
    paths = []
    for k in range(0, len(tiles), PAIR_COLS):
        img = _tile(tiles[k:k + PAIR_COLS], PAIR_COLS)
        paths.append(_write(out_dir / tag,
                            f"{tag}_{video.path.stem}_{locus.locus_id}_{k}"[:110],
                            img))
    return ToolResult(
        value={"locus_id": locus.locus_id, "t_span": [t0, t1],
               "bbox": [round(v, 4) for v in locus.bbox],
               "indices": idx, "score": round(locus.score, 3),
               "dominant": locus.dominant},
        images=paths, reliability=0.7, backend="locus_strip",
        hint=(f"这是信号标记为可疑的一处,已裁剪并放大(每格约 {side}px),按时间顺序排列,"
              "格上是帧号。\n"
              "**信号只知道这里的变化无法用运动解释,不知道这是什么。**"
              "它可能是真的崩坏,也可能是正常的遮挡、转向、光影变化。\n"
              "请看清楚这几格之间**同一个物体**的形状、边缘、纹理是怎么变的,再下结论。"),
    )


def worst_loci(video: VideoHandle, out_dir: Path, *, k: int = 3,
               max_frames: int | None = 96, side: int = LEGIBLE,
               tag: str = "worst") -> ToolResult:
    """The k highest-scoring loci of a clip, each as its own magnified strip.

    Loci are taken from distinct times where possible: three views of one
    moment answer one question three times, and the clip has a whole duration
    to account for.
    """
    import json

    from agenteval.signals.suspicion import compute_maps, extract_loci

    # The loci a clip yields depend on the clip and k, not on who reads them, so
    # a second judge over the same corpus should not pay for the optical flow
    # again. Content addressing already deduplicates the images; this
    # deduplicates the computation behind them, which is the expensive half.
    cache = out_dir / tag / f"_loci_{video.path.stem}_{k}.json"[:120]
    if cache.exists():
        try:
            c = json.loads(cache.read_text())
            paths = [Path(p) for p in c["images"]]
            if all(p.exists() for p in paths):
                return ToolResult(value=c["value"], images=paths,
                                  reliability=0.7, backend="worst_loci",
                                  hint=c.get("hint", ""))
        except Exception:  # noqa: BLE001
            pass

    try:
        maps = compute_maps(str(video.path), max_frames=max_frames)
        loci = extract_loci(maps, z=2.5, max_loci=24)
    except Exception as e:  # noqa: BLE001
        return ToolResult(value={"error": f"{type(e).__name__}: {e}"[:90]},
                          reliability=0.0)
    if not loci:
        return ToolResult(value={"loci": [], "n": 0}, reliability=0.3,
                          backend="worst_loci",
                          hint="信号没有标记出可疑处——这本身是一条证据,但不是清白的证明。")
    picked: list[SuspicionLocus] = []
    for l in loci:
        if len(picked) >= k:
            break
        # Keep loci apart in time so k views cover the clip rather than one moment
        if all(abs(l.t_span[0] - p.t_span[0]) >= 6 for p in picked):
            picked.append(l)
    for l in loci:
        if len(picked) >= k:
            break
        if l not in picked:
            picked.append(l)

    paths, vals = [], []
    for l in picked:
        r = locus_strip(video, l, out_dir, side=side, tag=tag)
        if r.images:
            paths += r.images
            vals.append(r.value)
    if not paths:
        return ToolResult(value={"error": "no strips"}, reliability=0.0)
    hint = ("下面是这段视频里信号认为**最可疑的几处**,每处已裁剪放大,按时间排列。\n"
            "可疑不等于有问题:信号测的是'这里的变化无法用运动解释',"
            "遮挡、转向、光影变化都会触发它。请逐处确认。")
    value = {"loci": vals, "n": len(vals), "top_score": round(picked[0].score, 3)}
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(
            {"value": value, "images": [str(p) for p in paths], "hint": hint},
            ensure_ascii=False, default=_plain))
    except (OSError, TypeError):
        pass
    return ToolResult(value=value, images=paths, reliability=0.7,
                      backend="worst_loci", hint=hint)


def paired_loci(a: VideoHandle, b: VideoHandle, out_dir: Path, *, k: int = 3,
                max_frames: int | None = 96, side: int = LEGIBLE,
                z: float = 2.5) -> tuple[ToolResult, ToolResult]:
    """Loci for two clips on one shared scale.

    `compute_maps` exceedance-normalises within a video, which is right for
    ranking moments inside one clip and wrong for every use across clips. Taken
    per clip, `worst_loci` therefore hands back each clip's own top three
    whether or not anything is wrong -- measured: three unrelated clips all
    returned a top score of 3.784, which is the exceedance ceiling
    -log10(1/n), identical to the digit. Magnifying those and asking which set
    looks worse compares a broken clip's collapse against a clean clip's
    ordinary motion blur, and the comparison has no common denominator.

    Pooling both clips' raw cells before the transform gives one denominator:
    a clean clip scores low and can legitimately return nothing, which is the
    evidence that it is clean.
    """
    import numpy as np
    from agenteval.signals.suspicion import (SIGNALS, _exceedance, compute_maps,
                                             extract_loci, fuse)

    try:
        ra = compute_maps(str(a.path), max_frames=max_frames, raw=True)
        rb = compute_maps(str(b.path), max_frames=max_frames, raw=True)
    except Exception as e:  # noqa: BLE001
        err = ToolResult(value={"error": f"{type(e).__name__}: {e}"[:90]},
                         reliability=0.0)
        return err, err

    # One exceedance fit over the concatenation, applied to each half. Same
    # transform, same denominator, so the two clips' scores mean the same thing.
    na, out = {}, {}
    for key in SIGNALS:
        ma, mb = ra.get(key), rb.get(key)
        if ma is None or mb is None:
            continue
        flat = np.concatenate([ma.ravel(), mb.ravel()])
        joint = _exceedance(flat)
        na[key] = joint[:ma.size].reshape(ma.shape)
        out[key] = joint[ma.size:].reshape(mb.shape)

    res = []
    for maps, v, tag in ((na, a, "A"), (out, b, "B")):
        loci = extract_loci(maps, z=z, max_loci=24)
        picked: list[SuspicionLocus] = []
        for l in loci:
            if len(picked) >= k:
                break
            if all(abs(l.t_span[0] - p.t_span[0]) >= 6 for p in picked):
                picked.append(l)
        paths, vals = [], []
        for l in picked:
            r = locus_strip(v, l, out_dir, side=side, tag=tag)
            if r.images:
                paths += r.images
                vals.append(r.value)
        peak = float(fuse(maps).max()) if maps else 0.0
        res.append(ToolResult(
            value={"loci": vals, "n": len(vals), "peak": round(peak, 3),
                   "shared_scale": True},
            images=paths, reliability=0.7, backend="paired_loci",
            hint=("这些可疑处的分数是**两段视频放在一起算的**,所以 A 和 B 的分数可以直接比。\n"
                  "一段视频可疑处少或没有,本身就是它更干净的证据。")))
    return res[0], res[1]
