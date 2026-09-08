"""Suspicion mapping — the search index.

Cheap, deterministic, VLM-free signal processing over *every* frame, producing
a ranked list of places worth looking at. This is what turns evaluation from a
fixed blind sample into an adaptive search.

The central signal is the **motion-compensated residual**: warp frame *t* into
frame *t+1* using the estimated optical flow, then measure what is left over.
Content that merely moved is explained away by the warp; what remains is change
the motion field cannot account for — which is precisely what generative
instability looks like. Ordinary fast motion, which naive frame-differencing
flags constantly, is largely cancelled.

The other signals are chosen to be *complementary*, i.e. to fail on different
things than the residual does:

  flow_anomaly    divergence/curl spikes  -> tearing, popping, non-rigid collapse
  softness        tile sharpness after regressing out local flow
                  -> softness that motion does NOT explain (motion blur is often
                     correct and must not be punished as a defect)
  crawl           high-frequency temporal energy after motion compensation
                  -> texture boiling on otherwise static surfaces
  luma_jump       frame-level luminance/histogram steps -> flicker, hard cuts

All maps live on a coarse (T-1, GY, GX) tile grid, are robustly normalized
per video (median/MAD, so a uniformly bad video does not read as uniformly
clean), then fused. Loci are connected components in that 3-D grid.

Recall matters more than precision here: a locus that is merely suspicious
costs one cheap probe, but a defect missing from the index can never be found
downstream, no matter how good the judge is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

GRID = (8, 8)          # (GY, GX) tiles
WORK_SIDE = 256        # longest side for signal computation

SIGNALS = ("mc_residual", "flow_anomaly", "softness", "crawl", "luma_jump", "freeze")

# luma_jump is frame-global by construction. Letting it into the spatial fusion
# lights every tile at once and merges unrelated events into one useless blob,
# so it is excluded there and handled as a separate whole-frame temporal signal.
SPATIAL_SIGNALS = ("mc_residual", "flow_anomaly", "softness", "crawl", "freeze")

# Fusion weights. mc_residual leads because it is the only signal that is
# explicitly decorrelated from legitimate motion.
DEFAULT_WEIGHTS: dict[str, float] = {
    "mc_residual": 1.0,
    "flow_anomaly": 0.7,
    "softness": 0.6,
    "crawl": 0.7,
    "luma_jump": 0.8,
    "freeze": 0.9,
}


@dataclass
class SuspicionLocus:
    """One place worth probing: a time span, a region, and why."""

    locus_id: str
    t_span: tuple[int, int]                        # [start, end)
    bbox: tuple[float, float, float, float]        # normalized (x, y, w, h)
    score: float                                   # fused, robust-z units
    signals: dict[str, float] = field(default_factory=dict)  # per-signal peak
    n_cells: int = 0

    @property
    def dominant(self) -> str:
        return max(self.signals, key=self.signals.get) if self.signals else ""

    def to_json(self) -> dict[str, Any]:
        return {
            "locus_id": self.locus_id,
            "t_span": list(self.t_span),
            "bbox": [round(v, 4) for v in self.bbox],
            "score": round(self.score, 3),
            "dominant": self.dominant,
            "signals": {k: round(v, 3) for k, v in self.signals.items()},
            "n_cells": self.n_cells,
        }


# ---- helpers -------------------------------------------------------------

def _tile_reduce(x: np.ndarray, grid: tuple[int, int] = GRID) -> np.ndarray:
    """(H, W) -> (GY, GX) mean, tolerating non-divisible sizes."""
    gy, gx = grid
    h, w = x.shape
    ys = np.linspace(0, h, gy + 1).astype(int)
    xs = np.linspace(0, w, gx + 1).astype(int)
    out = np.empty((gy, gx), np.float32)
    for i in range(gy):
        for j in range(gx):
            blk = x[ys[i]:max(ys[i] + 1, ys[i + 1]), xs[j]:max(xs[j] + 1, xs[j + 1])]
            out[i, j] = float(blk.mean()) if blk.size else 0.0
    return out


def _exceedance(m: np.ndarray) -> np.ndarray:
    """Map each cell to ``-log10(P(value >= x))`` within this video.

    Rank-based rather than median/MAD, because the raw signals have wildly
    different tail shapes: motion-compensated residual is heavy-tailed and
    reaches z~50, while unexplained softness rarely passes z~5. Under max-fusion
    that lets one signal win everywhere on tail shape alone, not on evidence.
    Exceedance puts every signal on the same "this is a 1-in-10^z cell" scale,
    so fusion compares like with like.
    """
    flat = m.ravel()
    n = flat.size
    order = np.argsort(flat, kind="stable")
    ranks = np.empty(n, np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    # cells strictly above this one, +1 so the maximum is finite
    p = (n - ranks) / n
    return (-np.log10(np.maximum(p, 1.0 / n))).reshape(m.shape).astype(np.float32)


def _decode_work(video: str, max_frames: int | None = None) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Decode to working resolution. Returns (gray TxhxW, small_bgr, H0, W0)."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    grays, smalls = [], []
    h0 = w0 = 0
    while True:
        ok, fr = cap.read()
        if not ok or (max_frames and len(grays) >= max_frames):
            break
        if not h0:
            h0, w0 = fr.shape[:2]
        s = WORK_SIDE / max(fr.shape[:2])
        small = cv2.resize(fr, (max(8, int(fr.shape[1] * s)), max(8, int(fr.shape[0] * s))),
                           interpolation=cv2.INTER_AREA) if s < 1 else fr
        smalls.append(small)
        grays.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    cap.release()
    if len(grays) < 2:
        raise RuntimeError(f"need >=2 frames, got {len(grays)} from {video}")
    return np.stack(grays), np.stack(smalls), h0, w0


# ---- the signal maps -----------------------------------------------------

def compute_maps(video: str, *, max_frames: int | None = None,
                 grid: tuple[int, int] = GRID) -> dict[str, np.ndarray]:
    """Return {signal_name: (T-1, GY, GX) float32 robust-z map}."""
    import cv2

    gray, small, _h0, _w0 = _decode_work(video, max_frames)
    T = len(gray)
    gy, gx = grid
    maps = {k: np.zeros((T - 1, gy, gx), np.float32) for k in SIGNALS}

    yy, xx = np.mgrid[0:gray.shape[1], 0:gray.shape[2]].astype(np.float32)
    prev_warp_err: np.ndarray | None = None
    raw_resid = np.zeros((T - 1, gy, gx), np.float32)
    change_energy = np.zeros((T - 1, gy, gx), np.float32)
    cov_flow = np.zeros((T - 1, gy, gx), np.float32)
    cov_grad = np.zeros((T - 1, gy, gx), np.float32)

    for t in range(T - 1):
        a, b = gray[t], gray[t + 1]
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        fx, fy = flow[..., 0], flow[..., 1]
        mag = np.sqrt(fx * fx + fy * fy)

        # 1. motion-compensated residual: warp a -> b, keep what motion cannot explain.
        #    Raw warp error is NOT that: it grows with flow magnitude and with image
        #    gradient (flow is least accurate at fast-moving, high-detail, occluding
        #    edges), so raw error just re-finds wherever the video moves fastest.
        #    Those two covariates are regressed out below, across the whole clip.
        warped = cv2.remap(a, (xx + fx), (yy + fy), cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        err = np.abs(warped.astype(np.float32) - b.astype(np.float32))
        gxg = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3)
        gyg = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gxg * gxg + gyg * gyg)
        raw_resid[t] = _tile_reduce(err, grid)
        cov_flow[t] = _tile_reduce(mag, grid)
        cov_grad[t] = _tile_reduce(grad, grid)
        # total change energy, for the two-sided (freeze) side of the anomaly
        change_energy[t] = _tile_reduce(np.abs(a.astype(np.float32) - b.astype(np.float32)),
                                        grid) + cov_flow[t]

        # 2. flow field anomaly: |divergence| + |curl|
        dfx_dx = np.gradient(fx, axis=1); dfy_dy = np.gradient(fy, axis=0)
        dfy_dx = np.gradient(fy, axis=1); dfx_dy = np.gradient(fx, axis=0)
        anom = np.abs(dfx_dx + dfy_dy) + np.abs(dfy_dx - dfx_dy)
        maps["flow_anomaly"][t] = _tile_reduce(anom, grid)

        # 3. softness NOT explained by motion: regress tile sharpness on tile flow
        lap = np.abs(cv2.Laplacian(b, cv2.CV_32F, ksize=3))
        sharp_t = _tile_reduce(lap, grid)
        flow_t = _tile_reduce(mag, grid)
        flow_t_for_crawl = flow_t
        f = flow_t.ravel(); s = sharp_t.ravel()
        if f.std() > 1e-6:
            slope, icpt = np.polyfit(f, s, 1)
            resid = (slope * f + icpt) - s          # positive = softer than motion predicts
        else:
            resid = float(s.mean()) - s
        maps["softness"][t] = np.maximum(resid, 0).reshape(gy, gx)

        # 4. crawl: residual that PERSISTS across consecutive pairs *and* sits on a
        #    near-static surface. Unexplained change where nothing is moving is
        #    texture boiling; the same residual under fast motion is just flow error,
        #    which mc_residual already reports. Gating makes the two complementary
        #    instead of redundant.
        if prev_warp_err is not None:
            persistent = _tile_reduce(np.minimum(err, prev_warp_err), grid)
            calm = 1.0 / (1.0 + flow_t_for_crawl)      # ~1 when still, ->0 when moving
            maps["crawl"][t] = persistent * calm
        prev_warp_err = err

        # 5. luma jump: frame-global, broadcast across tiles (it has no locality)
        d = abs(float(b.mean()) - float(a.mean()))
        hist_a = cv2.calcHist([a], [0], None, [32], [0, 256]).ravel()
        hist_b = cv2.calcHist([b], [0], None, [32], [0, 256]).ravel()
        hist_a /= max(hist_a.sum(), 1); hist_b /= max(hist_b.sum(), 1)
        maps["luma_jump"][t] = d + float(np.abs(hist_a - hist_b).sum()) * 10.0

    # Regress raw warp error on (flow magnitude, gradient energy, their product)
    # over every tile of the whole clip; keep only the positive residual, i.e.
    # change that motion and detail together do NOT account for.
    f = cov_flow.ravel(); g = cov_grad.ravel(); y = raw_resid.ravel()
    A = np.stack([f, g, f * g, np.ones_like(f)], axis=1)
    try:
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
    except np.linalg.LinAlgError:
        pred = np.full_like(y, float(y.mean()))
    maps["mc_residual"] = np.maximum(y - pred, 0).reshape(raw_resid.shape).astype(np.float32)

    # Anomaly is two-sided. A dropped or repeated frame produces *less* change
    # than its neighbours, not more -- warping an identical frame leaves almost
    # no residual -- so a purely "high residual is suspicious" index is blind to
    # freezes by construction. `freeze` scores how far change energy falls below
    # the local temporal median, which is exactly a stutter.
    med = np.median(change_energy, axis=0, keepdims=True)
    win = 5
    local = np.stack([
        np.median(change_energy[max(0, t - win):t + win + 1], axis=0)
        for t in range(len(change_energy))
    ]) if len(change_energy) else change_energy
    ref = np.maximum(np.maximum(local, med), 1e-3)
    maps["freeze"] = np.clip(1.0 - change_energy / ref, 0.0, 1.0).astype(np.float32)

    return {k: _exceedance(v) for k, v in maps.items()}


def fuse(maps: dict[str, np.ndarray],
         weights: dict[str, float] | None = None,
         signals: Sequence[str] = SPATIAL_SIGNALS) -> np.ndarray:
    """Weighted max-fusion. Max, not mean: one strongly anomalous signal is
    enough reason to look, and averaging would let three quiet signals bury it."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    use = [k for k in signals if k in maps]
    stack = np.stack([maps[k] * w.get(k, 1.0) for k in use])
    return stack.max(axis=0)


# ---- loci ----------------------------------------------------------------

def _grow(fused: np.ndarray, claimed: np.ndarray, seed: tuple[int, int, int],
          *, floor: float, max_t: int, max_cells: int) -> list[tuple[int, int, int]]:
    """Grow a locus outward from a peak, bounded in time and area.

    Bounded on purpose: a locus is a *probe target*. A component allowed to
    sprawl across half the clip and the whole frame cannot be cropped or
    zoomed into, so it carries no more information than "something is wrong
    somewhere" — which is what we are trying to get away from.
    """
    T, gy, gx = fused.shape
    st, si, sj = seed
    comp: list[tuple[int, int, int]] = []
    frontier = [seed]
    seen = {seed}
    while frontier and len(comp) < max_cells:
        # always expand the strongest frontier cell first, so a bounded locus
        # keeps the most anomalous cells rather than whatever BFS reached first
        frontier.sort(key=lambda c: -fused[c])
        cur = frontier.pop(0)
        ct, ci, cj = cur
        if claimed[cur] or fused[cur] < floor or abs(ct - st) > max_t:
            continue
        comp.append(cur)
        for dt, di, dj in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
            nb = (ct + dt, ci + di, cj + dj)
            if (0 <= nb[0] < T and 0 <= nb[1] < gy and 0 <= nb[2] < gx
                    and nb not in seen and not claimed[nb] and fused[nb] >= floor):
                seen.add(nb)
                frontier.append(nb)
    return comp


def extract_loci(maps: dict[str, np.ndarray], *, z: float = 2.5,
                 max_loci: int = 24, min_cells: int = 2, rel_floor: float = 0.45,
                 max_t_span: int = 12, max_area_frac: float = 0.45,
                 pad_t: int = 1, grid: tuple[int, int] = GRID) -> list[SuspicionLocus]:
    """Peak-growing extraction: repeatedly take the strongest unclaimed cell and
    grow a bounded region around it.

    Peak-growing rather than plain connected components, because thresholding a
    heavy-tailed map and taking components produces one giant blob that bridges
    unrelated events through time. Peaks give ranked, probe-sized, non-overlapping
    loci by construction.
    """
    fused = fuse(maps)
    T, gy, gx = fused.shape
    max_cells = max(min_cells, int(max_area_frac * gy * gx * min(max_t_span, T)))
    claimed = np.zeros_like(fused, bool)
    loci: list[SuspicionLocus] = []

    while len(loci) < max_loci:
        masked = np.where(claimed, -np.inf, fused)
        peak = np.unravel_index(int(np.argmax(masked)), fused.shape)
        pv = float(fused[peak])
        if not np.isfinite(masked[peak]) or pv < z:
            break
        comp = _grow(fused, claimed, tuple(int(v) for v in peak),
                     floor=max(z * 0.6, pv * rel_floor),
                     max_t=max_t_span // 2, max_cells=max_cells)
        for c in comp:
            claimed[c] = True
        if len(comp) < min_cells:
            continue
        ts = [c[0] for c in comp]; iy = [c[1] for c in comp]; ix = [c[2] for c in comp]
        t0 = max(0, min(ts) - pad_t)
        t1 = min(T + 1, max(ts) + 2 + pad_t)   # map index t covers frames t..t+1
        x0, x1 = min(ix) / gx, (max(ix) + 1) / gx
        y0, y1 = min(iy) / gy, (max(iy) + 1) / gy
        loci.append(SuspicionLocus(
            locus_id="", t_span=(t0, t1), bbox=(x0, y0, x1 - x0, y1 - y0),
            score=pv, n_cells=len(comp),
            signals={k: float(max(m[c] for c in comp)) for k, m in maps.items()},
        ))

    # luma_jump has no spatial locality, so it contributes whole-frame temporal loci
    if "luma_jump" in maps:
        lj = maps["luma_jump"][:, 0, 0]
        for t in np.argsort(-lj)[:4]:
            if lj[t] < z:
                break
            loci.append(SuspicionLocus(
                locus_id="", t_span=(max(0, int(t) - pad_t), min(T + 1, int(t) + 2 + pad_t)),
                bbox=(0.0, 0.0, 1.0, 1.0), score=float(lj[t]), n_cells=1,
                signals={"luma_jump": float(lj[t])},
            ))

    loci.sort(key=lambda l: -l.score)
    loci = loci[:max_loci]
    for n, l in enumerate(loci):
        l.locus_id = f"L{n:02d}"
    return loci


def suspicion_map(video: str, *, max_frames: int | None = None, z: float = 2.0,
                  max_loci: int = 24) -> tuple[list[SuspicionLocus], dict[str, np.ndarray]]:
    maps = compute_maps(video, max_frames=max_frames)
    return extract_loci(maps, z=z, max_loci=max_loci), maps
