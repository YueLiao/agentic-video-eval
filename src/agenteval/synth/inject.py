"""Synthetic defect injection — ground truth for free.

Take a *real* video (no generation artifacts) and inject defects whose type,
spatio-temporal extent and strength are known exactly. That yields, at zero
labelling cost:

  * detection recall / false-positive rate, per defect type
  * localization accuracy (spatial IoU, temporal IoU)
  * severity monotonicity: does the score fall as strength rises?
  * hallucination rate: inject *nothing* and count alleged defects

It decouples "can the system find defects" from "do humans agree with the
score". The first question becomes answerable today, without annotators.

Caveat, deliberately stated: injected artifacts are not generation artifacts.
This measures a capability *lower bound* and guards regressions; it never
substitutes for human-localized defects on real generated video.

Coordinates: bboxes are normalized (x, y, w, h) in [0,1]; time spans are
(start_frame, end_frame) inclusive-exclusive.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

DefectType = str

DEFECT_TYPES: tuple[DefectType, ...] = (
    "local_blur",        # generation softness
    "frame_drop",        # inter-frame jump / stutter
    "frame_repeat",      # frozen motion
    "region_shuffle",    # texture crawl / boiling
    "patch_jump",        # object teleport
    "luma_pulse",        # flicker
    "affine_warp",       # structural deformation
    "patch_swap",        # identity / appearance switch
)


@dataclass
class Defect:
    """One injected defect, with exact ground truth."""

    defect_id: str
    type: DefectType
    t_span: tuple[int, int]                 # [start, end)
    bbox: tuple[float, float, float, float] | None  # normalized; None = whole frame
    strength: float                         # 0..1, monotone in perceptual severity
    params: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["t_span"] = list(self.t_span)
        d["bbox"] = list(self.bbox) if self.bbox else None
        return d


def _px(bbox, h: int, w: int) -> tuple[int, int, int, int]:
    """Normalized bbox -> integer pixel slice bounds (y0, y1, x0, x1)."""
    if bbox is None:
        return 0, h, 0, w
    x, y, bw, bh = bbox
    x0 = int(round(x * w)); y0 = int(round(y * h))
    x1 = int(round((x + bw) * w)); y1 = int(round((y + bh) * h))
    x0, x1 = max(0, min(x0, w - 1)), max(1, min(x1, w))
    y0, y1 = max(0, min(y0, h - 1)), max(1, min(y1, h))
    if x1 <= x0: x1 = min(w, x0 + 1)
    if y1 <= y0: y1 = min(h, y0 + 1)
    return y0, y1, x0, x1


def _feather(shape: tuple[int, int], border: int = 6) -> np.ndarray:
    """Soft-edged mask so injected regions do not add a hard rectangle edge —
    a hard edge would be trivially detectable and would flatter the detector."""
    h, w = shape
    m = np.ones((h, w), np.float32)
    b = max(1, min(border, h // 2, w // 2))
    ramp = np.linspace(0.0, 1.0, b, dtype=np.float32)
    m[:b, :] *= ramp[:, None]
    m[-b:, :] *= ramp[::-1, None]
    m[:, :b] *= ramp[None, :]
    m[:, -b:] *= ramp[None, ::-1]
    return m


# ---- per-type injectors --------------------------------------------------
# Each takes the full (T,H,W,3) uint8 BGR clip and mutates a copy in place.

def _inject_local_blur(frames: np.ndarray, d: Defect) -> np.ndarray:
    import cv2
    t0, t1 = d.t_span
    y0, y1, x0, x1 = _px(d.bbox, *frames.shape[1:3])
    k = int(3 + round(d.strength * 20)) | 1          # odd kernel, 3..23
    m = _feather((y1 - y0, x1 - x0))[..., None]
    for t in range(t0, min(t1, len(frames))):
        reg = frames[t, y0:y1, x0:x1].astype(np.float32)
        blur = cv2.GaussianBlur(reg, (k, k), 0)
        frames[t, y0:y1, x0:x1] = (reg * (1 - m) + blur * m).astype(np.uint8)
    d.params["kernel"] = k
    return frames


def _inject_frame_drop(frames: np.ndarray, d: Defect) -> np.ndarray:
    """Remove n frames and hold the previous one — a stutter, not a shortening,
    so the clip length (and every downstream index) stays comparable."""
    t0, t1 = d.t_span
    for t in range(t0, min(t1, len(frames))):
        frames[t] = frames[max(0, t0 - 1)]
    d.params["held_from"] = max(0, t0 - 1)
    return frames


def _inject_frame_repeat(frames: np.ndarray, d: Defect) -> np.ndarray:
    t0, t1 = d.t_span
    src = frames[t0].copy()
    for t in range(t0, min(t1, len(frames))):
        frames[t] = src
    return frames


def _inject_region_shuffle(frames: np.ndarray, d: Defect) -> np.ndarray:
    """Randomly permute a region's content across time inside the window.
    Motion elsewhere stays coherent, so this is exactly 'un-explainable change'."""
    t0, t1 = d.t_span
    t1 = min(t1, len(frames))
    y0, y1, x0, x1 = _px(d.bbox, *frames.shape[1:3])
    idx = list(range(t0, t1))
    rng = random.Random(d.params.get("seed", 0))
    span = max(2, int(2 + d.strength * 6))
    out = frames.copy()
    for i in range(0, len(idx), span):
        chunk = idx[i:i + span]
        perm = chunk[:]
        rng.shuffle(perm)
        for dst, src in zip(chunk, perm):
            out[dst, y0:y1, x0:x1] = frames[src, y0:y1, x0:x1]
    d.params["window"] = span
    return out


def _inject_patch_jump(frames: np.ndarray, d: Defect) -> np.ndarray:
    """Displace a patch's content by delta for the span — an object teleport."""
    t0, t1 = d.t_span
    h, w = frames.shape[1:3]
    y0, y1, x0, x1 = _px(d.bbox, h, w)
    dx = int(round(d.strength * 0.15 * w))
    dy = int(round(d.strength * 0.05 * h))
    m = _feather((y1 - y0, x1 - x0))[..., None]
    for t in range(t0, min(t1, len(frames))):
        sy0, sy1 = np.clip([y0 + dy, y1 + dy], 0, h)
        sx0, sx1 = np.clip([x0 + dx, x1 + dx], 0, w)
        src = frames[t, sy0:sy1, sx0:sx1]
        if src.shape[0] != y1 - y0 or src.shape[1] != x1 - x0:
            continue
        dstr = frames[t, y0:y1, x0:x1].astype(np.float32)
        frames[t, y0:y1, x0:x1] = (dstr * (1 - m) + src.astype(np.float32) * m).astype(np.uint8)
    d.params["delta_px"] = [dx, dy]
    return frames


def _inject_luma_pulse(frames: np.ndarray, d: Defect) -> np.ndarray:
    t0, t1 = d.t_span
    y0, y1, x0, x1 = _px(d.bbox, *frames.shape[1:3])
    amp = 0.15 + 0.55 * d.strength
    m = _feather((y1 - y0, x1 - x0))[..., None]
    for i, t in enumerate(range(t0, min(t1, len(frames)))):
        # alternate sign so it reads as flicker rather than a grade change
        g = 1.0 + amp * (1 if i % 2 == 0 else -1)
        reg = frames[t, y0:y1, x0:x1].astype(np.float32)
        frames[t, y0:y1, x0:x1] = np.clip(reg * (1 - m) + reg * g * m, 0, 255).astype(np.uint8)
    d.params["amp"] = round(amp, 3)
    return frames


def _inject_affine_warp(frames: np.ndarray, d: Defect) -> np.ndarray:
    """Time-varying local affine — limbs stretching / bending impossibly."""
    import cv2
    t0, t1 = d.t_span
    t1 = min(t1, len(frames))
    y0, y1, x0, x1 = _px(d.bbox, *frames.shape[1:3])
    hh, ww = y1 - y0, x1 - x0
    m = _feather((hh, ww))[..., None]
    for i, t in enumerate(range(t0, t1)):
        phase = np.sin(2 * np.pi * i / max(2, (t1 - t0)))
        sx = 1.0 + 0.35 * d.strength * phase
        sh = 0.30 * d.strength * phase
        M = np.float32([[sx, sh, (1 - sx) * ww / 2], [0, 1.0, 0]])
        reg = frames[t, y0:y1, x0:x1]
        wrp = cv2.warpAffine(reg, M, (ww, hh), borderMode=cv2.BORDER_REFLECT)
        frames[t, y0:y1, x0:x1] = (reg * (1 - m) + wrp * m).astype(np.uint8)
    return frames


def _inject_patch_swap(frames: np.ndarray, d: Defect) -> np.ndarray:
    """Replace a region with its content from a distant time — identity drift."""
    t0, t1 = d.t_span
    t1 = min(t1, len(frames))
    y0, y1, x0, x1 = _px(d.bbox, *frames.shape[1:3])
    donor = int(d.params.get("donor", (t0 + len(frames) // 2) % len(frames)))
    m = (_feather((y1 - y0, x1 - x0)) * (0.4 + 0.6 * d.strength))[..., None]
    src = frames[donor, y0:y1, x0:x1].astype(np.float32)
    for t in range(t0, t1):
        reg = frames[t, y0:y1, x0:x1].astype(np.float32)
        frames[t, y0:y1, x0:x1] = (reg * (1 - m) + src * m).astype(np.uint8)
    d.params["donor"] = donor
    return frames


_INJECTORS: dict[DefectType, Callable[[np.ndarray, Defect], np.ndarray]] = {
    "local_blur": _inject_local_blur,
    "frame_drop": _inject_frame_drop,
    "frame_repeat": _inject_frame_repeat,
    "region_shuffle": _inject_region_shuffle,
    "patch_jump": _inject_patch_jump,
    "luma_pulse": _inject_luma_pulse,
    "affine_warp": _inject_affine_warp,
    "patch_swap": _inject_patch_swap,
}


def apply(frames: np.ndarray, defects: Sequence[Defect]) -> np.ndarray:
    out = frames.copy()
    for d in defects:
        out = _INJECTORS[d.type](out, d)
    return out


# ---- sampling a defect plan ---------------------------------------------

def sample_defects(
    n_frames: int,
    *,
    types: Sequence[DefectType] = DEFECT_TYPES,
    count: int = 1,
    strength: float | tuple[float, float] = (0.4, 0.9),
    seed: int = 0,
    min_span: int = 3,
    max_span_frac: float = 0.35,
) -> list[Defect]:
    """Draw a random, non-overlapping-in-time defect plan."""
    rng = random.Random(seed)
    out: list[Defect] = []
    used: list[tuple[int, int]] = []
    for i in range(count):
        typ = rng.choice(list(types))
        span = rng.randint(min_span, max(min_span, int(n_frames * max_span_frac)))
        for _ in range(20):
            t0 = rng.randint(0, max(0, n_frames - span - 1))
            t1 = t0 + span
            if all(t1 <= a or t0 >= b for a, b in used):
                break
        else:
            continue
        used.append((t0, t1))
        # whole-frame types get no bbox
        if typ in ("frame_drop", "frame_repeat"):
            bbox = None
        else:
            bw = rng.uniform(0.15, 0.45)
            bh = rng.uniform(0.15, 0.45)
            bbox = (rng.uniform(0, 1 - bw), rng.uniform(0, 1 - bh), bw, bh)
        s = strength if isinstance(strength, float) else rng.uniform(*strength)
        out.append(Defect(
            defect_id=f"d{i:02d}", type=typ, t_span=(t0, t1),
            bbox=bbox, strength=round(s, 3), params={"seed": rng.randint(0, 10**6)},
        ))
    return out


# ---- video IO ------------------------------------------------------------

def read_all(path: str | Path, max_frames: int | None = None) -> tuple[np.ndarray, float]:
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 24.0
    out = []
    while True:
        ok, fr = cap.read()
        if not ok or (max_frames and len(out) >= max_frames):
            break
        out.append(fr)
    cap.release()
    if not out:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.stack(out), fps


def write(path: str | Path, frames: np.ndarray, fps: float) -> Path:
    import cv2
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames.shape[1:3]
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"cannot open writer for {p}")
    for f in frames:
        vw.write(f)
    vw.release()
    return p


def make_case(
    src: str | Path,
    out_dir: str | Path,
    *,
    case_id: str,
    defects: Sequence[Defect],
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Write ``<out_dir>/<case_id>.mp4`` plus ``<case_id>.json`` ground truth."""
    frames, fps = read_all(src, max_frames)
    out = apply(frames, defects) if defects else frames.copy()
    vp = write(Path(out_dir) / f"{case_id}.mp4", out, fps)
    gt = {
        "case_id": case_id,
        "source": str(src),
        "video": str(vp),
        "n_frames": int(len(out)),
        "fps": fps,
        "height": int(out.shape[1]),
        "width": int(out.shape[2]),
        "defects": [d.to_json() for d in defects],
        "clean": not defects,
    }
    Path(out_dir, f"{case_id}.json").write_text(
        json.dumps(gt, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return gt
