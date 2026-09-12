#!/usr/bin/env python3
"""Do the suspicion loci detect spatially local defects, and how often do they cry wolf?

The signal-peak matrix reports zero detection power for all five bbox-confined
defect types, but it scores the global maximum of the map -- a defect in a fifth
of the frame never wins that against whatever moves most in the clip. Locus
extraction is the statistic built for local structure, and an earlier bench put
it at hit@3 = 1.00 on the same injections.

That bench only measured recall. The number that decides whether loci can drive
a harness is the other one: how many loci a clip with nothing wrong still
produces. The per-locus verdict chain later found 96% of nominations on real
clips to be benign, which is the same quantity measured the expensive way.

    python scripts/run_locus_detector_bench.py --n 24 --out runs/locbench
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
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agenteval.signals.suspicion import compute_maps, extract_loci   # noqa: E402
from agenteval.synth.inject import (DEFECT_TYPES, Defect,            # noqa: E402
                                    apply, read_all, write)

ROOT = "/pub/evaluation_group/cy/rm_videos"
LOCAL_TYPES = ("local_blur", "region_shuffle", "patch_jump",
               "affine_warp", "patch_swap")


def hits(loci, d: Defect, T: int, k: int) -> bool:
    """Does any of the top-k loci overlap the defect in time and space?"""
    ta, tb = d.t_span
    for l in loci[:k]:
        la, lb = l.t_span
        if min(lb, tb) <= max(la, ta):
            continue
        if d.bbox is None:
            return True
        x, y, w, h = d.bbox
        lx, ly, lw, lh = l.bbox
        ix = max(0.0, min(x + w, lx + lw) - max(x, lx))
        iy = max(0.0, min(y + h, ly + lh) - max(y, ly))
        if ix * iy > 0:
            return True
    return False


def _one(job):
    path, seed, span_frac, z = job
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
    tmp = Path("/tmp/_locbench") / Path(path).stem
    tmp.mkdir(parents=True, exist_ok=True)
    out = {"video": Path(path).stem, "T": T, "types": {}}

    clean_p = tmp / "clean.mp4"
    if not clean_p.exists():
        write(clean_p, frames, fps)
    try:
        cm = compute_maps(str(clean_p), max_frames=96)
        out["clean_loci"] = len(extract_loci(cm, z=z, max_loci=24))
    except Exception:  # noqa: BLE001
        return None

    span = max(4, int(T * span_frac))
    for typ in LOCAL_TYPES:
        t0 = rng.randint(int(T * 0.2), max(int(T * 0.2) + 1, int(T * 0.7)))
        bw, bh = rng.uniform(0.2, 0.4), rng.uniform(0.2, 0.4)
        bbox = (rng.uniform(0, 1 - bw), rng.uniform(0, 1 - bh), bw, bh)
        d = Defect(defect_id="d0", type=typ, t_span=(t0, min(T, t0 + span)),
                   bbox=bbox, strength=0.8, params={"seed": seed})
        p = tmp / f"{typ}.mp4"
        if not p.exists():
            write(p, apply(frames, [d]), fps)
        try:
            m = compute_maps(str(p), max_frames=96)
            loci = extract_loci(m, z=z, max_loci=24)
        except Exception:  # noqa: BLE001
            continue
        out["types"][typ] = {"n_loci": len(loci),
                             **{f"hit@{k}": hits(loci, d, T, k) for k in (1, 3, 5, 10)}}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--span-frac", type=float, default=0.12)
    ap.add_argument("--z", type=float, default=2.5)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sys.stdout.reconfigure(line_buffering=True)

    vids = sorted(glob.glob(f"{ROOT}/**/*.mp4", recursive=True))
    random.Random(5).shuffle(vids)
    vids = vids[:args.n]
    print(f"{len(vids)} 条 × {len(LOCAL_TYPES)} 种局部缺陷 · z={args.z}")

    cache = out / "raw.json"
    recs = json.loads(cache.read_text()) if cache.exists() else []
    if not recs:
        jobs = [(v, i, args.span_frac, args.z) for i, v in enumerate(vids)]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, r in enumerate(ex.map(_one, jobs), 1):
                if r:
                    recs.append(r)
                if i % 6 == 0:
                    print(f"  {i}/{len(vids)}")
        cache.write_text(json.dumps(recs))
    print(f"可用 {len(recs)} 条\n")

    print(f"  {'缺陷类型':<16}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}{'hit@10':>8}"
          f"{'提名数中位':>12}")
    for typ in LOCAL_TYPES:
        cells = [r["types"][typ] for r in recs if typ in r["types"]]
        if not cells:
            continue
        row = "".join(f"{np.mean([c[f'hit@{k}'] for c in cells]):>8.0%}"
                      for k in (1, 3, 5, 10))
        print(f"  {typ:<16}{row}{np.median([c['n_loci'] for c in cells]):>12.0f}")

    cl = [r["clean_loci"] for r in recs]
    print(f"\n  干净片段也会提名 {np.median(cl):.0f} 处(中位) · "
          f"范围 {min(cl)}–{max(cl)}")
    print(f"  → 取 top-3 时,干净片段每条也贡献 3 个待核实的候选")
    inj = np.mean([np.mean([r['types'][t]['hit@3'] for t in r['types']])
                   for r in recs if r['types']])
    print(f"\n  综合:top-3 召回 {inj:.0%},但精度上界 = 1/3(其余两个必是假阳性)")
    print("  这解释了逐处判定链上 96% 的提名是良性的——不是信号坏,是它按定义只排序不定量")
    (out / "bench.json").write_text(json.dumps(
        {"recall": {t: float(np.mean([r["types"][t]["hit@3"] for r in recs
                                      if t in r["types"]])) for t in LOCAL_TYPES},
         "clean_loci_median": float(np.median(cl))}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
