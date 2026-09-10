#!/usr/bin/env python3
"""Score the fake-motion trap set: does the model reward motion that is not real?

The trap set is 80 pairs in which one clip was flagged by annotators as *fake
motion* -- it moves, but the movement is not the movement the prompt asked for.
The humans are near-unanimous: the control wins 66, the trap wins 2, 12 tie.

This is the sharpest available falsification of the fitted point-wise score,
whose largest weight by some margin is `flow_mean` -- more motion is better. A
model that reached its accuracy by rewarding motion per se will invert here,
and on this set inverting is unambiguous rather than a matter of degree. The
pairs live in R012's train split, so a model fitted on dev has never seen them.

    python scripts/run_trap_eval.py --model runs/pw/model.json --out runs/trap
"""
from __future__ import annotations

import os as _os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse
import os
import csv
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit_pointwise import FEATS, design                  # noqa: E402
from run_signal_pairwise import features                 # noqa: E402

RR = ("/pub/evaluation_group/cy/mq_promptgen/pairing/review_results")
ROOT = "/pub/evaluation_group/cy/rm_videos"


def build_pairs() -> list[dict]:
    """Recover each trap/control pair together with which side each was on."""
    ts = json.loads(Path(f"{RR}/static_trap_set.json").read_text())
    want = {t["pair_id"]: (t["trap_video_id"], c["video_id"], t["final_label"])
            for c in ts["controls"] for t in c["vs_traps"]}
    out = []
    seen = set()
    for rnd in ("R012_20260908", "R011_20260820"):
        for sp in ("train", "dev", "val_ac"):
            p = Path(f"{RR}/rounds/{rnd}/{sp}.csv")
            if not p.exists():
                continue
            for r in csv.DictReader(p.open()):
                pid = r["pair_id"]
                if pid not in want or pid in seen:
                    continue
                trap_id, ctrl_id, label = want[pid]
                if label is None:
                    continue
                seen.add(pid)
                a_id = Path(r["path_A"]).stem
                out.append({"pair_id": pid, "split": sp, "round": rnd,
                            "path_A": r["path_A"], "path_B": r["path_B"],
                            "label": label,
                            "trap_side": "a" if a_id == trap_id else "b"})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model.json from fit_pointwise")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--margin", type=float, default=0.25)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sys.stdout.reconfigure(line_buffering=True)

    pairs = build_pairs()
    print(f"陷阱对 {len(pairs)} 个 · 所在 split "
          f"{dict(Counter(p['split'] for p in pairs))}")
    human = Counter("平局" if p["label"] == "same"
                    else ("陷阱胜" if (p["label"] in ("A", "AA")) ==
                          (p["trap_side"] == "a") else "对照胜")
                    for p in pairs)
    print(f"人评: {dict(human)}")

    rels = sorted({p[k] for p in pairs for k in ("path_A", "path_B")})
    cache = out / "features.json"
    feats = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [r for r in rels if r not in feats]
    if todo:
        print(f"  提特征 {len(todo)} 条 ...")
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for rel, f in zip(todo, ex.map(features,
                                           [os.path.join(ROOT, r) for r in todo])):
                feats[rel] = f
        cache.write_text(json.dumps(feats))
    good = {k: v for k, v in feats.items() if "error" not in v}
    pairs = [p for p in pairs
             if p["path_A"] in good and p["path_B"] in good]
    print(f"  特征可用,可评 {len(pairs)} 对")

    m = json.loads(Path(args.model).read_text())
    w = np.array(m["w"]); mu = np.array(m["mu"]); sd = np.array(m["sd"])
    rels = sorted({p[k] for p in pairs for k in ("path_A", "path_B")})
    S = dict(zip(rels, (design(good, rels) - mu) / sd @ w))
    # flow_mean alone, as the straw man the trap set exists to knock down
    fi = FEATS.index("flow_mean")
    F = dict(zip(rels, ((design(good, rels) - mu) / sd)[:, fi]))

    def report(name, sc):
        r = Counter()
        for p in pairs:
            d = sc[p["path_A"]] - sc[p["path_B"]]
            if abs(d) < args.margin:
                pred = "tie"
            else:
                pred = "a" if d > 0 else "b"
            truth = ("same" if p["label"] == "same"
                     else ("a" if p["label"] in ("A", "AA") else "b"))
            if pred == "tie":
                r["判平局"] += 1
                r["判平局·人评也平" if truth == "same" else "判平局·人评有方向"] += 1
            else:
                picked_trap = (pred == p["trap_side"])
                r["选中陷阱片" if picked_trap else "选中对照片"] += 1
                if truth != "same":
                    r["方向对" if pred == truth else "方向错"] += 1
        dec = r["方向对"] + r["方向错"]
        picked = r["选中陷阱片"] + r["选中对照片"]
        print(f"\n  === {name} ===")
        print(f"  表态 {picked}/{len(pairs)}  ·  方向准确 "
              f"{r['方向对']/max(1,dec):.1%} (n={dec})")
        print(f"  在表态的对里选中**陷阱片**的比例 "
              f"{r['选中陷阱片']/max(1,picked):.1%} "
              f"({r['选中陷阱片']}/{picked})  ← 人评只有 2/68 认为陷阱片更好")
        return r

    report("拟合的 point-wise 分数", S)
    report("只用 flow_mean(稻草人)", F)

    rows = []
    for p in pairs:
        d = S[p["path_A"]] - S[p["path_B"]]
        trap_minus_ctrl = d if p["trap_side"] == "a" else -d
        rows.append({**p, "score_gap_trap_minus_ctrl": round(float(trap_minus_ctrl), 3)})
    rows.sort(key=lambda r: -r["score_gap_trap_minus_ctrl"])
    (out / "trap_detail.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    gaps = np.array([r["score_gap_trap_minus_ctrl"] for r in rows])
    print(f"\n  陷阱片减对照片的分差: 中位 {np.median(gaps):+.3f} "
          f"(<0 才是对的) · 为正的 {int((gaps > 0).sum())}/{len(gaps)}")
    print(f"  wrote {out}/trap_detail.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
