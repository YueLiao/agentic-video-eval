#!/usr/bin/env python3
"""Per-signal corpus statistics, so 'nothing unusual' becomes representable.

Within-video exceedance makes the six signals commensurable and destroys any
absolute meaning: every clip, clean or broken, produces the same ranked loci.
Raw magnitudes keep absolute meaning and are not commensurable: one threshold
across signals saturates the locus cap on every clip. Calibrating each signal on
its own spread over untouched clips gives both.

    python scripts/calibrate_signals.py --n 60 --out profiles/signals.json
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

ROOT = "/pub/evaluation_group/cy/rm_videos"


def _one(path: str):
    import cv2
    cv2.setNumThreads(1)
    try:
        m = compute_maps(path, max_frames=96, raw=True)
    except Exception:  # noqa: BLE001
        return None
    # Quantiles per clip, pooled after: a few clips with enormous motion would
    # otherwise set the scale for everything.
    return {k: [float(np.percentile(v, q)) for q in (50, 90, 99, 99.9)]
            for k, v in m.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    vids = sorted(glob.glob(f"{ROOT}/**/*.mp4", recursive=True))
    random.Random(101).shuffle(vids)          # a different draw from the benches
    vids = vids[:args.n]
    print(f"在 {len(vids)} 条未改动片段上标定 {len(SIGNALS)} 路信号")

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(_one, vids), 1):
            if r:
                rows.append(r)
            if i % 10 == 0:
                print(f"  {i}/{len(vids)}")

    stats = {}
    print(f"\n  {'信号':<14}{'中位':>12}{'90%':>12}{'99%':>12}{'MAD':>12}")
    for s in SIGNALS:
        med = np.array([r[s][0] for r in rows if s in r])
        p90 = np.array([r[s][1] for r in rows if s in r])
        p99 = np.array([r[s][2] for r in rows if s in r])
        if not len(med):
            continue
        m = float(np.median(med))
        mad = float(np.median(np.abs(p99 - m))) or 1.0
        stats[s] = {"median": m, "mad": mad,
                    "p90": float(np.median(p90)), "p99": float(np.median(p99)),
                    "n_clips": len(med)}
        print(f"  {s:<14}{m:>12.3f}{np.median(p90):>12.3f}"
              f"{np.median(p99):>12.3f}{mad:>12.3f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    print(f"\n  wrote {args.out}")
    print("  之后 compute_maps(..., calibration=stats) 即用语料尺度,"
          "干净片段可以合法地产出 0 处可疑点")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
