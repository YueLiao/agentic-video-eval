#!/usr/bin/env python3
"""Which signal detects which defect, at what false-positive rate.

The existing injection bench measured paired sensitivity and hit@k, both of
which answer "does the injected clip score higher than its own clean twin". A
detector that has to run on clips with no twin needs the other number: how often
it fires on an untouched clip. The freeze work made the cost of skipping that
concrete -- 85% detection looked like a result until the clean control came back
at 65%.

So every cell here is measured against both populations: the same clip with and
without the defect, with the signal read in raw units so the two are comparable
(exceedance normalisation is a within-video rank and is identical across clips
by construction).

Output is a signal x defect matrix of detection at a fixed 5% false-positive
rate, plus localisation, so the gaps are visible: a defect no signal covers is a
defect the harness cannot find, and that is a design input rather than a bug.

    python scripts/run_detector_matrix.py --n 24 --out runs/detmat
"""
from __future__ import annotations

import os as _os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agenteval.signals.suspicion import SIGNALS, compute_maps   # noqa: E402
from agenteval.synth.inject import (DEFECT_TYPES, Defect,       # noqa: E402
                                    apply, read_all, write)

ROOT = "/pub/evaluation_group/cy/rm_videos"


def peak_and_where(m: np.ndarray, local: bool = False) -> tuple[float, float, float]:
    """Strongest cell of a (T-1, gy, gx) map, and the time it sits at.

    `local` normalises each cell against its own history. It is off by default
    because it measurably hurts: the MAD of a signal that is near zero most of
    the time is near zero too, so dividing by it amplifies noise, and freeze --
    which detected both frame-level defects at 100% under the global peak --
    fell to 0%. A defect confined to a fifth of the frame still does not win a
    global maximum, so the bbox-confined types need a different statistic
    rather than this one.
    """
    if m.size == 0:
        return 0.0, 0.0, 0.0
    if local:
        flat = m.reshape(len(m), -1)
        base = np.median(flat, axis=0, keepdims=True)
        scale = np.median(np.abs(flat - base), axis=0, keepdims=True) * 1.4826
        z = (flat - base) / np.maximum(scale, 1e-3)
        t_prof = z.max(axis=1)
    else:
        t_prof = m.reshape(len(m), -1).max(axis=1)
    i = int(np.argmax(t_prof))
    n = max(1, len(t_prof))
    return float(t_prof[i]), i / n, (i + 1) / n


def _one(job):
    path, seed, span_frac, types = job
    import cv2
    cv2.setNumThreads(1)
    try:
        frames, fps = read_all(path, max_frames=96)
    except Exception:  # noqa: BLE001
        return None
    T = len(frames)
    if T < 60:
        return None
    rng = random.Random(seed)
    tmp = Path("/tmp/_detmat") / Path(path).stem
    tmp.mkdir(parents=True, exist_ok=True)
    out = {"video": Path(path).stem, "T": T, "cells": {}}

    clean_p = tmp / "clean.mp4"
    if not clean_p.exists():
        write(clean_p, frames, fps)
    try:
        clean_maps = compute_maps(str(clean_p), max_frames=96, raw=True)
    except Exception:  # noqa: BLE001
        return None
    out["clean"] = {k: peak_and_where(v)[0] for k, v in clean_maps.items()}

    span = max(4, int(T * span_frac))
    for typ in types:
        t0 = rng.randint(int(T * 0.2), max(int(T * 0.2) + 1, int(T * 0.7)))
        bbox = None
        if typ not in ("frame_drop", "frame_repeat"):
            bw, bh = rng.uniform(0.2, 0.4), rng.uniform(0.2, 0.4)
            bbox = (rng.uniform(0, 1 - bw), rng.uniform(0, 1 - bh), bw, bh)
        d = Defect(defect_id="d0", type=typ, t_span=(t0, min(T, t0 + span)),
                   bbox=bbox, strength=0.8, params={"seed": seed})
        p = tmp / f"{typ}.mp4"
        if not p.exists():
            write(p, apply(frames, [d]), fps)
        try:
            maps = compute_maps(str(p), max_frames=96, raw=True)
        except Exception:  # noqa: BLE001
            continue
        cell = {}
        for k, v in maps.items():
            pk, a, b = peak_and_where(v)
            ta, tb = t0 / T, min(T, t0 + span) / T
            cell[k] = {"peak": pk, "loc_ok": bool(min(b, tb) - max(a, ta) > -0.06)}
        out["cells"][typ] = {"truth": [round(t0 / T, 3), round(min(T, t0 + span) / T, 3)],
                             "sig": cell}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--span-frac", type=float, default=0.12)
    ap.add_argument("--fp", type=float, default=0.05,
                    help="false-positive rate the threshold is set at")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sys.stdout.reconfigure(line_buffering=True)

    vids = sorted(glob.glob(f"{ROOT}/**/*.mp4", recursive=True))
    random.Random(5).shuffle(vids)
    vids = vids[:args.n]
    print(f"{len(vids)} 条视频 × {len(DEFECT_TYPES)} 种缺陷 × {len(SIGNALS)} 路信号")

    cache = out / "raw.json"
    recs = json.loads(cache.read_text()) if cache.exists() else []
    if not recs:
        jobs = [(v, i, args.span_frac, DEFECT_TYPES) for i, v in enumerate(vids)]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, r in enumerate(ex.map(_one, jobs), 1):
                if r:
                    recs.append(r)
                if i % 6 == 0:
                    print(f"  {i}/{len(vids)}")
        cache.write_text(json.dumps(recs))
    print(f"可用 {len(recs)} 条\n")

    # One threshold per signal, set on the clean population at the target FP.
    thr = {}
    for s in SIGNALS:
        vals = np.array([r["clean"].get(s, 0.0) for r in recs])
        thr[s] = float(np.quantile(vals, 1 - args.fp)) if len(vals) else 0.0

    print(f"  阈值按干净片段的 {1-args.fp:.0%} 分位设定(即误报率 {args.fp:.0%})")
    print(f"\n  {'缺陷类型':<16}" + "".join(f"{s[:9]:>11}" for s in SIGNALS) + f"{'最佳':>12}")
    summary = {}
    for typ in DEFECT_TYPES:
        row, best = [], (0.0, "", 0.0)
        for s in SIGNALS:
            hits = [r["cells"][typ]["sig"][s] for r in recs
                    if typ in r["cells"] and s in r["cells"][typ]["sig"]]
            if not hits:
                row.append("--")
                continue
            det = np.mean([h["peak"] > thr[s] for h in hits])
            loc = np.mean([h["loc_ok"] for h in hits if h["peak"] > thr[s]]) \
                if any(h["peak"] > thr[s] for h in hits) else 0.0
            row.append(f"{det:.0%}/{loc:.0%}")
            if det > best[0]:
                best = (det, s, loc)
        summary[typ] = {"best_signal": best[1], "detection": round(best[0], 3),
                        "localisation": round(best[2], 3)}
        flag = "  ← 无检测器" if best[0] < 0.5 else ""
        print(f"  {typ:<16}" + "".join(f"{c:>11}" for c in row)
              + f"{best[1][:10]:>12}{flag}")
    print("\n  单元格 = 检出率/定位正确率 · 「最佳」列是该缺陷检出最高的信号")
    (out / "matrix.json").write_text(json.dumps(
        {"threshold_fp": args.fp, "thresholds": thr, "summary": summary},
        ensure_ascii=False, indent=1))
    gaps = [t for t, v in summary.items() if v["detection"] < 0.5]
    print(f"\n  覆盖缺口({len(gaps)}/{len(DEFECT_TYPES)}): {gaps or '无'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
