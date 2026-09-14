#!/usr/bin/env python3
"""Run the routed per-clip chain, and validate it where ground truth is free.

Two validations, because human preference and defect detection are different
claims and this session has repeatedly confused them:

  injection   exact ground truth, no labels needed -- does the chain find a
              defect that is definitely there, and does it stay quiet on the
              untouched original
  benchmark   the point-wise scores differenced across a pair, against human
              direction, reported by agreement strength

    python scripts/run_routed.py --mode inject --n 24 --out runs/routed
    python scripts/run_routed.py --mode bench  --n 200 --out runs/routed_b
"""
from __future__ import annotations

import os as _os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse
import os
import csv
import glob
import json
import math
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agenteval.engine.routed import evaluate_clip                 # noqa: E402
from agenteval.llm.client import VLMClient                        # noqa: E402
from agenteval.synth.inject import Defect, apply, read_all, write  # noqa: E402

ROOT = "/pub/evaluation_group/cy/rm_videos"
R012 = ("/pub/evaluation_group/cy/mq_promptgen/pairing/review_results/"
        "rounds/R012_20260908")
TRUTH = {"AA": "a", "A": "a", "BB": "b", "B": "b", "same": "same"}


def z(h, n):
    return (h / n - 0.5) / math.sqrt(0.25 / n) if n else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["inject", "bench"], default="inject")
    ap.add_argument("--defect", default="frame_repeat",
                    help="which defect to inject. The temporal branch was "
                         "validated on frame_repeat; the spatial branch has "
                         "never been validated at all, and its 25% firing rate "
                         "on untouched clips could be detection or invention.")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--endpoints", nargs="*",
                    default=["http://127.0.0.1:8005/v1", "http://127.0.0.1:8006/v1"])
    ap.add_argument("--model", default="gemma-4-31b-it")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sys.stdout.reconfigure(line_buffering=True)

    vlms = [VLMClient(model=args.model, base_url=e, max_tokens=700,
                      timeout_s=420, cache_dir=out / "llm_cache")
            for e in args.endpoints]
    cache_p = out / "reports.json"
    done = json.loads(cache_p.read_text()) if cache_p.exists() else {}

    def run(i, key, path):
        if key in done:
            return key, done[key]
        try:
            r = evaluate_clip(path, vlms[i % len(vlms)], out / "views")
            return key, r.to_json()
        except Exception as e:  # noqa: BLE001
            return key, {"error": f"{type(e).__name__}: {e}"[:110]}

    if args.mode == "inject":
        vids = sorted(glob.glob(f"{ROOT}/**/*.mp4", recursive=True))
        random.Random(17).shuffle(vids)
        jobs, truth = [], {}
        rng = random.Random(4)
        for i, p in enumerate(vids):
            if len(jobs) >= args.n * 2:
                break
            try:
                frames, fps = read_all(p, max_frames=96)
            except Exception:  # noqa: BLE001
                continue
            T = len(frames)
            if T < 60:
                continue
            t0 = rng.randint(int(T * 0.3), int(T * 0.6))
            spatial = args.defect not in ("frame_repeat", "frame_drop")
            bbox = None
            if spatial:
                bw, bh = rng.uniform(0.22, 0.42), rng.uniform(0.22, 0.42)
                bbox = (rng.uniform(0, 1 - bw), rng.uniform(0, 1 - bh), bw, bh)
            span = 8 if not spatial else max(8, int(T * 0.18))
            d = Defect(defect_id="d0", type=args.defect,
                       t_span=(t0, min(T, t0 + span)), bbox=bbox,
                       strength=1.0, params={"seed": i})
            tmp = out / "clips"; tmp.mkdir(parents=True, exist_ok=True)
            for cond, arr in (("inj", apply(frames, [d])), ("clean", frames)):
                q = tmp / f"{Path(p).stem}_{cond}.mp4"
                if not q.exists():
                    write(q, arr, fps)
                key = f"{Path(p).stem}|{cond}"
                jobs.append((len(jobs), key, str(q)))
                truth[key] = ({"t": [t0 / T, min(T, t0 + span) / T],
                           "bbox": bbox, "spatial": spatial}
                          if cond == "inj" else None)
        print(f"注入验证: {len(jobs)//2} 条 × (注入版 + 干净版)")
        todo = [j for j in jobs if j[1] not in done]
        if todo:
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                for k, (key, r) in enumerate(ex.map(lambda j: run(*j), todo), 1):
                    done[key] = r
                    if k % 8 == 0:
                        print(f"  {k}/{len(todo)}  {time.time()-t0:.0f}s")
                        cache_p.write_text(json.dumps(done, ensure_ascii=False))
            cache_p.write_text(json.dumps(done, ensure_ascii=False))

        ok = {k: v for k, v in done.items() if "error" not in v}
        inj = [(k, v) for k, v in ok.items() if k.endswith("|inj")]
        cln = [(k, v) for k, v in ok.items() if k.endswith("|clean")]
        det = loc = 0
        for k, v in inj:
            tf = [f for f in v["findings"] if f["kind"] == "temporal"]
            if tf:
                det += 1
                ta, tb = (truth.get(k) or {}).get("t", (0.0, 0.0))
                loc += any(min(f["t_span"][1], tb) - max(f["t_span"][0], ta) > -0.08
                           for f in tf)
        fp = sum(1 for _k, v in cln
                 if any(f["kind"] == "temporal" for f in v["findings"]))
        sp_i = sum(1 for _k, v in inj
                   if any(f["kind"] == "spatial" for f in v["findings"]))
        sp_c = sum(1 for _k, v in cln
                   if any(f["kind"] == "spatial" for f in v["findings"]))
        spatial_truth = any((truth.get(k) or {}).get("spatial") for k, _v in inj)
        if spatial_truth:
            sl = 0
            for k, v in inj:
                t = truth.get(k) or {}
                ta, tb = t.get("t", (0, 0))
                bx = t.get("bbox")
                for f in v["findings"]:
                    if f["kind"] != "spatial" or not f["bbox"]:
                        continue
                    if not (ta - 0.15 <= f["t_span"][0] <= tb + 0.15):
                        continue
                    x, y, w_, h_ = f["bbox"]
                    gx, gy, gw, gh = bx
                    ix = max(0.0, min(x + w_, gx + gw) - max(x, gx))
                    iy = max(0.0, min(y + h_, gy + gh) - max(y, gy))
                    if ix * iy > 0:
                        sl += 1
                        break
            print(f"    其中定位到注入区域的 {sl}/{max(1,sp_i)} · "
                  f"净检出(注入-干净) {(sp_i-sp_c)/max(1,len(inj)):+.0%}")
        print(f"\n  时间类(注入的就是卡顿)")
        print(f"    检出 {det}/{len(inj)} = {det/max(1,len(inj)):.0%} · "
              f"定位正确 {loc}/{max(1,det)} · "
              f"干净版误报 {fp}/{len(cln)} = {fp/max(1,len(cln)):.0%}")
        print(f"  空间类(两边都没有注入空间缺陷,差异即噪声)")
        print(f"    注入版报出 {sp_i}/{len(inj)} · 干净版报出 {sp_c}/{len(cln)}")
        sc_i = [v["score"] for _k, v in inj]
        sc_c = [v["score"] for _k, v in cln]
        print(f"  分数: 注入版中位 {sorted(sc_i)[len(sc_i)//2]:.1f} · "
              f"干净版中位 {sorted(sc_c)[len(sc_c)//2]:.1f}")
        return 0

    rows = [r for r in csv.DictReader(open(f"{R012}/val_ac.csv"))][:args.n]
    vids = sorted({r[k] for r in rows for k in ("path_A", "path_B")})
    print(f"benchmark: {len(rows)} 对 · {len(vids)} 条视频")
    todo = [(i, v, os.path.join(ROOT, v)) for i, v in enumerate(vids)
            if v not in done]
    if todo:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for k, (key, r) in enumerate(ex.map(lambda j: run(*j), todo), 1):
                done[key] = r
                if k % 20 == 0:
                    print(f"  {k}/{len(todo)}  {time.time()-t0:.0f}s")
                    cache_p.write_text(json.dumps(done, ensure_ascii=False))
        cache_p.write_text(json.dumps(done, ensure_ascii=False))
    ok = {k: v for k, v in done.items() if "error" not in v}
    use = [r for r in rows if r["path_A"] in ok and r["path_B"] in ok
           and r["MQ"] != "same"]
    for name, labs in (("强偏好", ("AA", "BB")), ("弱偏好", ("A", "B")),
                       ("全部", ("AA", "A", "B", "BB"))):
        sub = [r for r in use if r["MQ"] in labs]
        dec = [r for r in sub if ok[r["path_A"]]["score"] != ok[r["path_B"]]["score"]]
        hit = sum(1 for r in dec
                  if ("a" if ok[r["path_A"]]["score"] > ok[r["path_B"]]["score"]
                      else "b") == TRUTH[r["MQ"]])
        print(f"  {name:<6} 可分 {len(dec):>3}/{len(sub):<3} "
              f"方向 {hit/max(1,len(dec)):.1%} (z={z(hit,len(dec)):+.2f})")
    return 0


if __name__ == "__main__":
    import os
    raise SystemExit(main())
