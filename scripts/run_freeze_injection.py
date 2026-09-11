#!/usr/bin/env python3
"""Can the model read an injected freeze off a space-time slice but not off frames?

The per-clip checklist asked about freezing, slipping, speed and amplitude on a
16-frame grid and scored 0% on every one of those items across 1220 clips --
not reluctance, an absence of evidence: a stall between two samples leaves both
samples looking fine. A space-time slice stacks every frame's scanline, so the
full frame rate is on one axis and a stall is a vertical band.

Injection gives exact ground truth for free: the span is known, so detection,
false-positive rate on the untouched clip, and localisation error are all
measurable without a single human label. Both conditions see the same clip and
the same defect, so the comparison isolates the view.

    python scripts/run_freeze_injection.py --n 40 --out runs/freeze
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
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agenteval.llm.client import ImageRef, VLMClient          # noqa: E402
from agenteval.media.clip import VideoHandle                  # noqa: E402
from agenteval.synth.inject import Defect, apply, read_all    # noqa: E402
from agenteval.tools.renders import filmstrip                 # noqa: E402
from agenteval.tools.temporal_views import space_time_slice   # noqa: E402

ROOT = "/pub/evaluation_group/cy/rm_videos"

ASK = """\
这段视频里有没有**画面停住**的区间(也就是连续若干帧几乎完全相同、本该有的运动停了)?

注意区分:
- **静止镜头拍静止场景**不算——那是本来就没东西动;
- 只有**原本在动的东西突然不动了一小段、之后又继续**才算。

输出 JSON:{"has_freeze": true|false, "t_start": 0~1 的时间比例或 null,
 "t_end": 0~1 或 null, "evidence": "你看到了什么"}
如果判断不了就 has_freeze=false。"""

# The reading instructions come from the tool, never from a copy here: a
# hardcoded note went on describing the raw-scanline rule ("a vertical band of
# unchanging texture is a freeze") after the view had switched to frame
# differences, where a freeze is a *black* band. The model followed the note
# faithfully, looked for the wrong thing, and scored 0/6 -- the picture was
# right and the manual was stale.


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--endpoints", nargs="*",
                    default=["http://127.0.0.1:8005/v1", "http://127.0.0.1:8006/v1"])
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sys.stdout.reconfigure(line_buffering=True)

    vids = sorted(glob.glob(f"{ROOT}/**/*.mp4", recursive=True))
    rng = random.Random(11)
    rng.shuffle(vids)
    vids = vids[:args.n]
    print(f"{len(vids)} 条干净视频 · 每条产出 注入版 + 原版,两种视图各问一次")

    vlms = [VLMClient(model="gemma-4-31b-it", base_url=e, max_tokens=500,
                      timeout_s=300, cache_dir=out / "llm_cache")
            for e in args.endpoints]

    def one(job):
        i, path = job
        vlm = vlms[i % len(vlms)]
        try:
            frames, fps = read_all(path, max_frames=96)
            T = len(frames)
            if T < 40:
                return None
            # A freeze long enough to be real but short enough to be missed by
            # a 16-frame grid: 8 frames is a third of a second.
            span = 8
            t0 = rng.randint(int(T * 0.25), int(T * 0.65))
            d = Defect(defect_id="f0", type="frame_repeat", t_span=(t0, t0 + span),
                       bbox=None, strength=args.strength, params={})
            hurt = apply(frames, [d])
            rec = {"video": Path(path).stem, "t0": t0, "t1": t0 + span, "T": T,
                   "truth_norm": [round(t0 / T, 3), round((t0 + span) / T, 3)]}

            for cond, arr, injected in (("inj", hurt, True), ("clean", frames, False)):
                tmp = out / "clips" / f"{Path(path).stem}_{cond}.mp4"
                tmp.parent.mkdir(parents=True, exist_ok=True)
                if not tmp.exists():
                    import cv2
                    h, w = arr.shape[1:3]
                    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps or 24.0, (w, h))
                    for f in arr:
                        vw.write(f)
                    vw.release()
                v = VideoHandle(tmp)
                for view in ("xt", "strip"):
                    if view == "xt":
                        r = space_time_slice(v, out / "views", n_lines=3, tag=f"xt{cond}")
                        note = r.hint
                    else:
                        r = filmstrip(v, out / "views", t0=0, t1=v.total - 1,
                                      n=16, cols=8, side=360, tag=f"st{cond}")
                        note = r.hint
                    if not r.images:
                        continue
                    resp = vlm.ask(system="只输出 JSON。", user=note + "\n\n" + ASK,
                                   images=[ImageRef(path=p) for p in r.images],
                                   schema={"type": "object"},
                                   tag=f"fz/{view}/{cond}/{Path(path).stem}")
                    p_ = resp.parsed or {}
                    rec[f"{view}_{cond}"] = {
                        "has": bool(p_.get("has_freeze")),
                        "t": [p_.get("t_start"), p_.get("t_end")],
                        "why": str(p_.get("evidence") or "")[:110]}
            return rec
        except Exception as e:  # noqa: BLE001
            return {"error": f"{type(e).__name__}: {e}"[:90]}

    cache_p = out / "results.json"
    res = json.loads(cache_p.read_text()) if cache_p.exists() else []
    if not res:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for k, r in enumerate(ex.map(one, enumerate(vids)), 1):
                if r:
                    res.append(r)
                if k % 10 == 0:
                    print(f"  {k}/{len(vids)}  {time.time()-t0:.0f}s")
        cache_p.write_text(json.dumps(res, ensure_ascii=False, indent=1))

    ok = [r for r in res if "error" not in r and "xt_inj" in r]
    print(f"\n完成 {len(ok)}/{len(res)}")
    print(f"\n  {'视图':<10}{'注入版检出率':>14}{'干净版误报率':>14}{'定位命中':>12}")
    for view in ("xt", "strip"):
        det = [r for r in ok if f"{view}_inj" in r]
        hit = sum(1 for r in det if r[f"{view}_inj"]["has"])
        cl = [r for r in ok if f"{view}_clean" in r]
        fp = sum(1 for r in cl if r[f"{view}_clean"]["has"])
        loc = 0
        for r in det:
            if not r[f"{view}_inj"]["has"]:
                continue
            t = r[f"{view}_inj"]["t"]
            try:
                a, b = float(t[0]), float(t[1])
            except (TypeError, ValueError):
                continue
            ta, tb = r["truth_norm"]
            # overlap with the true span, tolerant of a loose report
            if min(b, tb) - max(a, ta) > -0.10:
                loc += 1
        name = "时空切片" if view == "xt" else "16帧拼图"
        print(f"  {name:<10}{f'{hit}/{len(det)} = {hit/max(1,len(det)):.0%}':>14}"
              f"{f'{fp}/{len(cl)} = {fp/max(1,len(cl)):.0%}':>14}"
              f"{f'{loc}/{max(1,hit)} = {loc/max(1,hit):.0%}':>12}")
    print("\n  真实缺陷跨度 = 8 帧(约 0.33 秒),16 帧采样的相邻两帧间隔约 6 帧")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
