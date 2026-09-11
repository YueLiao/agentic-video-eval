#!/usr/bin/env python3
"""A fixed, small, pre-rendered debug set with a measured signal floor.

Iterating on a prompt against a freshly sampled batch every time confounds two
things: whether the wording improved, and which clips happened to be drawn. This
set is the same clips every run, rendered once and cached, so a prompt change is
the only thing that moves.

It also refuses to ship a case whose rendered view does not actually contain the
signal. Four times today a prompt was tuned against an image that could not have
answered it -- a reversed time axis, a gain that left the band black-on-black,
a stale reading rule -- and each looked like the model failing. `contrast` here
is measured from the rendered pixels: how much darker the injected span is than
the rest of the same panel. A case below the floor is marked unusable rather
than counted as a miss.

    python scripts/build_debug_set.py --out runs/debug_freeze
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
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agenteval.media.clip import VideoHandle                  # noqa: E402
from agenteval.synth.inject import Defect, apply, read_all    # noqa: E402
from agenteval.tools.renders import filmstrip                 # noqa: E402
from agenteval.tools.temporal_views import space_time_slice   # noqa: E402

ROOT = "/pub/evaluation_group/cy/rm_videos"


def ambient_motion(frames: np.ndarray) -> float:
    import cv2
    g = [cv2.cvtColor(cv2.resize(f, (160, 96)), cv2.COLOR_BGR2GRAY) for f in frames]
    return float(np.mean([cv2.absdiff(a, b).mean() for a, b in zip(g, g[1:])]))


def band_contrast(panels: list[dict], t0f: float, t1f: float) -> float:
    """How much darker the injected span is, measured on single panels.

    Read from the rendered pixels so encoding, resizing and jpeg are included --
    but per panel, never on the tile. Each panel has its own time axis and two
    of them run horizontally; profiled across the stack an injected freeze came
    out with *negative* contrast, which a freeze cannot have. Returns the best
    contrast over the panels, since one scanline can miss the subject.
    """
    import cv2
    best = 0.0
    for m in panels:
        im = cv2.imread(str(m.get("path", "")))
        if im is None:
            continue
        axis = 1 if m["kind"] == "row" else 0      # row: time down; col: right
        prof = im.mean(axis=(axis, 2))
        n = len(prof)
        a, b = int(t0f * n), max(int(t1f * n), int(t0f * n) + 1)
        if b >= n:
            continue
        inside = float(prof[a:b].mean())
        outside = float(np.concatenate([prof[:a], prof[b:]]).mean()) or 1.0
        best = max(best, (outside - inside) / outside)
    return round(best, 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--span", type=int, default=8)
    ap.add_argument("--min-contrast", type=float, default=0.12,
                    help="reject a case whose band is not visibly darker in the "
                         "rendered view; debugging a prompt against it measures "
                         "nothing")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sys.stdout.reconfigure(line_buffering=True)

    pool = sorted(glob.glob(f"{ROOT}/**/*.mp4", recursive=True))
    random.Random(7).shuffle(pool)

    cases, seen_motion = [], []
    rng = random.Random(3)
    for path in pool:
        if len(cases) >= args.n:
            break
        try:
            frames, fps = read_all(path, max_frames=96)
        except Exception:  # noqa: BLE001
            continue
        T = len(frames)
        if T < 60:
            continue
        amb = ambient_motion(frames)
        # Spread over ambient motion: a freeze is trivial to see in a busy scene
        # and nearly invisible in a still one, and a debug set of only one kind
        # teaches the wrong lesson.
        bucket = 0 if amb < 2 else (1 if amb < 6 else 2)
        if seen_motion.count(bucket) >= args.n // 3 + 1:
            continue

        t0 = rng.randint(int(T * 0.25), int(T * 0.65))
        d = Defect(defect_id="f0", type="frame_repeat",
                   t_span=(t0, t0 + args.span), bbox=None, strength=1.0, params={})
        stem = Path(path).stem
        rec = {"id": f"D{len(cases):02d}", "video": stem, "T": T,
               "ambient_motion": round(amb, 2), "bucket": bucket,
               "t_span": [t0, t0 + args.span],
               "truth_norm": [round(t0 / T, 3), round((t0 + args.span) / T, 3)]}

        import cv2
        for cond, arr in (("inj", apply(frames, [d])), ("clean", frames)):
            mp4 = out / "clips" / f"{stem}_{cond}.mp4"
            mp4.parent.mkdir(parents=True, exist_ok=True)
            if not mp4.exists():
                h, w = arr.shape[1:3]
                vw = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps or 24.0, (w, h))
                for f in arr:
                    vw.write(f)
                vw.release()
            v = VideoHandle(mp4)
            xt = space_time_slice(v, out / "views", n_lines=3, tag=f"xt_{cond}")
            st = filmstrip(v, out / "views", t0=0, t1=v.total - 1, n=16,
                           cols=8, side=360, tag=f"st_{cond}")
            rec[f"xt_{cond}"] = str(xt.images[0]) if xt.images else None
            if cond == "inj":
                rec["panels"] = xt.value.get("lines", [])
            rec[f"st_{cond}"] = str(st.images[0]) if st.images else None
            rec[f"hint_xt"] = xt.hint
            rec[f"hint_st"] = st.hint
        if rec["xt_inj"]:
            rec["contrast"] = band_contrast(rec.get("panels", []), *rec["truth_norm"])
            rec["usable"] = rec["contrast"] >= args.min_contrast
        else:
            rec["contrast"], rec["usable"] = 0.0, False
        seen_motion.append(bucket)
        cases.append(rec)
        print(f"  {rec['id']}  运动量 {amb:5.2f}  真值 {rec['truth_norm']}  "
              f"band 对比度 {rec['contrast']:+.3f}  "
              f"{'可用' if rec['usable'] else '★不可用(图里看不出)'}")

    (out / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=1))
    good = [c for c in cases if c["usable"]]
    print(f"\n{len(cases)} 例,其中 {len(good)} 例的 band 在渲染图上确实可见")
    print(f"运动量分布: 静 {sum(1 for c in cases if c['bucket']==0)} · "
          f"中 {sum(1 for c in cases if c['bucket']==1)} · "
          f"动 {sum(1 for c in cases if c['bucket']==2)}")
    print(f"wrote {out}/cases.json  —— 之后改 prompt 只需重跑提问,不用重渲染")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
