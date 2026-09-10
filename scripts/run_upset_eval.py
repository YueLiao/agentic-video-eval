#!/usr/bin/env python3
"""Score the upset set: pairs where the weaker generator won.

Every cross-model pair carries a prior -- one of the two generators is simply
better on average -- and a scorer can ride that prior a long way without ever
looking at the clip. The upset set is the 689 pairs where the prior is wrong:
a W- or M-tier model beat an S-tier one, by human judgement. Riding the prior
scores zero here by construction.

The set lives entirely in R012's train split (0 of its pair_ids appear in dev,
val_ac or dev_all2998), so it is clean held-out data for a model fitted on dev.
Its labels are thinner than val_ac's, though -- 616 of 689 have a single
annotator against val_ac's 3-5 consensus -- so its absolute numbers are not
comparable with val_ac's and are reported separately.

    python scripts/run_upset_eval.py --model runs/pw/model.json --out runs/upset
"""
from __future__ import annotations

import os as _os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse
import csv
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit_pointwise import design                          # noqa: E402
from run_signal_pairwise import features                  # noqa: E402

ROOT = "/pub/evaluation_group/cy/rm_videos"
R012 = ("/pub/evaluation_group/cy/mq_promptgen/pairing/review_results/"
        "rounds/R012_20260908")
UPSET = ("/pub/evaluation_group/cy/mq_promptgen/pairing/relabel_r2/"
         "P1_train_upset_relabel.csv")

TIER = {"wan5b": "W", "cogvideox": "W",
        "pangu": "M", "cosmos_nano": "M",
        "wan14b": "S", "cosmos_super": "S", "hunyuan": "S", "ltx": "S"}
RANK = {"W": 0, "M": 1, "S": 2}
TRUTH = {"AA": "a", "A": "a", "BB": "b", "B": "b", "same": "same"}


def model_of(rel: str) -> str:
    """The generator name, from whichever layout the split happens to use.

    val_ac stores 953 of its 998 rows as bench_results/<bench>/<M?>/<model>/...
    and the rest as .../videos/<model>/..., so splitting on a fixed marker
    raises on most of the corpus. The filename carries it too --
    <prompt_id>__<model>__<seed>.mp4 -- and that layout is stable across both.
    """
    parts = Path(rel).stem.split("__")
    if len(parts) >= 3 and parts[1] in TIER:
        return parts[1]
    for seg in Path(rel).parts[::-1]:
        if seg in TIER:
            return seg
    return ""


def tier_prior(r: dict) -> str | None:
    """Which side the generator ranking alone would pick. None when tied."""
    ta, tb = TIER.get(model_of(r["path_A"])), TIER.get(model_of(r["path_B"]))
    if ta is None or tb is None or RANK[ta] == RANK[tb]:
        return None
    return "a" if RANK[ta] > RANK[tb] else "b"


def load_features(rels, cache: Path, workers: int) -> dict:
    feats = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [r for r in rels if r not in feats]
    if todo:
        print(f"  提特征 {len(todo)} 条 ...")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, (rel, f) in enumerate(
                    zip(todo, ex.map(features,
                                     [os.path.join(ROOT, r) for r in todo])), 1):
                feats[rel] = f
                if i % 100 == 0:
                    print(f"    {i}/{len(todo)}")
        cache.write_text(json.dumps(feats))
    return feats


def score_rows(rows, feats, m, margin):
    good = {k: v for k, v in feats.items() if "error" not in v}
    rows = [r for r in rows if r["path_A"] in good and r["path_B"] in good]
    rels = sorted({r[k] for r in rows for k in ("path_A", "path_B")})
    Z = (design(good, rels) - np.array(m["mu"])) / np.array(m["sd"])
    s = dict(zip(rels, Z @ np.array(m["w"])))

    def pred(r):
        d = s[r["path_A"]] - s[r["path_B"]]
        return "same" if abs(d) < margin else ("a" if d > 0 else "b")
    return rows, pred


def report(name, rows, pred, extra_prior=True):
    nt = [r for r in rows if TRUTH.get(r["MQ"]) != "same"]
    dec = [r for r in nt if pred(r) != "same"]
    hit = sum(1 for r in dec if pred(r) == TRUTH[r["MQ"]])
    print(f"\n  === {name} ===  {len(rows)} 对 (非平局 {len(nt)})")
    print(f"  方向准确 {hit/max(1,len(dec)):.1%}  (表态 {len(dec)}/{len(nt)})")
    if extra_prior:
        # What the generator ranking alone scores on the same pairs -- the
        # thing the upset set exists to make worthless.
        pr = [r for r in nt if tier_prior(r)]
        ph = sum(1 for r in pr if tier_prior(r) == TRUTH[r["MQ"]])
        print(f"  对照·只看模型档位: {ph/max(1,len(pr)):.1%} (n={len(pr)})")
        # And how often our own prediction happens to agree with that prior.
        both = [r for r in dec if tier_prior(r)]
        ag = sum(1 for r in both if pred(r) == tier_prior(r))
        print(f"  我们的判断与档位先验一致 {ag/max(1,len(both)):.1%} "
              f"({ag}/{len(both)})")
    return hit / max(1, len(dec)), len(dec)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--margin", type=float, default=0.25)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sys.stdout.reconfigure(line_buffering=True)
    m = json.loads(Path(args.model).read_text())

    ups = list(csv.DictReader(open(UPSET)))
    val = list(csv.DictReader(open(f"{R012}/val_ac.csv")))
    print(f"upset {len(ups)} 对 · val_ac {len(val)} 对")
    print(f"  upset 的 family: {dict(Counter(r['family'] for r in ups))}")
    print(f"  upset 的标注人数: {dict(Counter(r['n_annotations'] for r in ups))}")

    rels = sorted({r[k] for rows in (ups, val) for r in rows
                   for k in ("path_A", "path_B")})
    feats = load_features(rels, out / "features.json", args.workers)

    rows_u, pred_u = score_rows(ups, feats, m, args.margin)
    rows_v, pred_v = score_rows(val, feats, m, args.margin)
    report("upset (弱档赢了强档,train 内,单标为主)", rows_u, pred_u)
    report("val_ac 全集 (对照)", rows_v, pred_v)

    # Cut val_ac the same way: pairs whose outcome contradicts the tier prior.
    vu = [r for r in rows_v if tier_prior(r) and TRUTH.get(r["MQ"]) != "same"
          and tier_prior(r) != TRUTH[r["MQ"]]]
    vc = [r for r in rows_v if tier_prior(r) and TRUTH.get(r["MQ"]) != "same"
          and tier_prior(r) == TRUTH[r["MQ"]]]
    print(f"\n  val_ac 内部按档位先验切开:  爆冷 {len(vu)} 对 · 顺风 {len(vc)} 对")
    report("val_ac·爆冷子集", vu, pred_v, extra_prior=False)
    report("val_ac·顺风子集", vc, pred_v, extra_prior=False)

    # family breakdown on the upset set
    print("\n  upset 分 family:")
    for fam in sorted({r["family"] for r in rows_u}):
        sub = [r for r in rows_u if r["family"] == fam]
        dec = [r for r in sub if pred_u(r) != "same"]
        hit = sum(1 for r in dec if pred_u(r) == TRUTH[r["MQ"]])
        print(f"    {fam:<6} {hit/max(1,len(dec)):6.1%}  (表态 {len(dec)}/{len(sub)})")
    print("\n  upset 分标注人数:")
    for n in sorted({r["n_annotations"] for r in rows_u}, key=int):
        sub = [r for r in rows_u if r["n_annotations"] == n]
        dec = [r for r in sub if pred_u(r) != "same"]
        hit = sum(1 for r in dec if pred_u(r) == TRUTH[r["MQ"]])
        print(f"    {n} 标  {hit/max(1,len(dec)):6.1%}  (表态 {len(dec)}/{len(sub)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
