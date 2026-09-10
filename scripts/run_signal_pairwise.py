#!/usr/bin/env python3
"""Can the measured signals alone order a pair the way humans did?

No VLM. Each clip is reduced to a handful of numbers -- the suspicion signals
plus plain motion statistics -- and each number is asked, on its own, to pick the
better clip. This separates two questions that the agentic pipeline answers
together and therefore confuses:

  * is the *evidence* informative about human preference at all?
  * is the *judge* reading the evidence?

A signal that beats chance here is a foundation to build a point-wise score on.
A signal that does not is one the judge cannot be blamed for ignoring.

    python scripts/run_signal_pairwise.py --split dev --n 100 --out runs/sig
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = "/pub/evaluation_group/cy/rm_videos"
R012 = ("/pub/evaluation_group/cy/mq_promptgen/pairing/review_results/"
        "rounds/R012_20260908")

TRUTH = {"AA": "a", "A": "a", "BB": "b", "B": "b", "same": "same"}
# Sign of "more of this is better". Suspicion signals are defect evidence, so
# less is better; motion magnitude is the opposite -- a clip that fails by not
# moving scores clean on every defect signal there is.
BETTER_IS = {"mc_residual": -1, "flow_anomaly": -1, "softness": -1, "crawl": -1,
             "luma_jump": -1, "freeze": -1, "fused_max": -1, "fused_mean": -1,
             "flow_mean": +1, "flow_p95": +1, "jerk": -1, "still_frac": -1}


def features(path: str) -> dict:
    import numpy as np
    from agenteval.signals.suspicion import compute_maps, fuse
    try:
        # raw, not exceedance-normalized: the normalized map is a within-video
        # rank transform and is literally constant across clips.
        maps = compute_maps(path, max_frames=64, raw=True)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:80]}
    out = {}
    for k, m in maps.items():
        # p99 rather than mean: a defect is local in space and time, and a mean
        # over the whole grid averages it away against the clean majority.
        out[k] = float(np.percentile(m, 99))
    f = fuse(maps)
    out["fused_max"] = float(f.max())
    out["fused_mean"] = float(f.mean())

    import cv2
    cap = cv2.VideoCapture(path)
    prev, mags = None, []
    while len(mags) < 63:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(fr, (160, 96)), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            fl = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 2, 13, 2, 5, 1.1, 0)
            mags.append(float(np.hypot(fl[..., 0], fl[..., 1]).mean()))
        prev = g
    cap.release()
    a = np.asarray(mags, np.float64) if mags else np.zeros(1)
    out["flow_mean"] = float(a.mean())
    out["flow_p95"] = float(np.percentile(a, 95))
    out["jerk"] = float(np.abs(np.diff(a, 2)).mean()) if len(a) > 2 else 0.0
    out["still_frac"] = float((a < 0.15 * max(a.mean(), 1e-6)).mean())
    return out


def _work(job):
    return job[0], features(job[1])


def sample(path: str, n: int, seed: int) -> list[dict]:
    rows = list(csv.DictReader(open(path)))
    rng = random.Random(seed)
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["MQ"], []).append(r)
    out: list[dict] = []
    for lab, g in by.items():
        out += rng.sample(g, min(max(1, round(n * len(g) / len(rows))), len(g)))
    rng.shuffle(out)
    return out[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "val_ac"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--margin", type=float, default=0.05,
                    help="relative gap below which the signal is called a tie")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sys.stdout.reconfigure(line_buffering=True)

    pairs = sample(f"{R012}/{args.split}.csv", args.n, seed=11)
    vids = sorted({os.path.join(ROOT, r[k]) for r in pairs
                   for k in ("path_A", "path_B")})
    print(f"{args.split}: {len(pairs)} 对, {len(vids)} 条视频")

    cache_p = out / "features.json"
    feats: dict = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    todo = [(v, v) for v in vids if v not in feats]
    if todo:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, (v, f) in enumerate(ex.map(_work, todo), 1):
                feats[v] = f
                if i % 25 == 0:
                    print(f"  特征 {i}/{len(todo)}")
        cache_p.write_text(json.dumps(feats))

    keys = list(BETTER_IS)
    strata = {"强偏好": ("AA", "BB"), "弱偏好": ("A", "B")}
    rows = []
    for k in keys:
        rec = {"signal": k}
        for sname, labs in strata.items():
            hit = tot = 0
            for r in pairs:
                if r["MQ"] not in labs:
                    continue
                fa = feats.get(os.path.join(ROOT, r["path_A"]), {})
                fb = feats.get(os.path.join(ROOT, r["path_B"]), {})
                if k not in fa or k not in fb:
                    continue
                va, vb = fa[k], fb[k]
                scale = max(abs(va), abs(vb), 1e-9)
                if abs(va - vb) / scale < args.margin:
                    continue            # signal declines to separate them
                pick = "a" if (va - vb) * BETTER_IS[k] > 0 else "b"
                tot += 1
                hit += pick == TRUTH[r["MQ"]]
            rec[sname] = (hit, tot)
        rows.append(rec)

    print(f"\n  单信号方向准确率(相对差 <{args.margin:.0%} 视为不表态)")
    print(f"  {'信号':<14}{'强偏好':>16}{'弱偏好':>16}")
    for rec in sorted(rows, key=lambda r: -(r["强偏好"][0] / max(1, r["强偏好"][1]))):
        cells = []
        for sname in strata:
            h, t = rec[sname]
            cells.append(f"{h/t:6.1%} ({h}/{t})" if t else "     -- (0/0)")
        print(f"  {rec['signal']:<14}{cells[0]:>16}{cells[1]:>16}")
    print("\n  50% = 随机。信号本身若不过 50%,判官读不出偏好就不是判官的问题。")
    (out / "signal_accuracy.json").write_text(json.dumps(rows, ensure_ascii=False,
                                                         indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
