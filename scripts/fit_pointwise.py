#!/usr/bin/env python3
"""Fit a point-wise motion score from measured features on pairwise labels.

The benchmark only ever says "A beat B", and a point-wise model is the thing
actually wanted. Bradley-Terry recovers a scale from comparisons but needs a
connected comparison graph, and this one is 97% isolated pairs -- 1894
components over 3894 videos -- so it recovers nothing.

Parameterising the strength instead fixes that: `s(v) = w . f(v)` over features
measured on the clip itself. The comparisons only fit `w`; the score then
applies to any clip, including one that was never compared to anything, and the
disconnected graph stops mattering because the videos share parameters instead
of sharing edges.

    python scripts/fit_pointwise.py --train dev --test val_ac --out runs/pw
"""
from __future__ import annotations

import os as _os

# Every worker is one video, so the parallelism is already at the process level.
# Left alone, each of them starts 64 BLAS threads and OpenCV starts as many
# more; at 32 workers that exhausts the process limit and the failure surfaces
# as silent decode errors in *other* jobs on the machine, not as an error here.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = "/pub/evaluation_group/cy/rm_videos"
R012 = ("/pub/evaluation_group/cy/mq_promptgen/pairing/review_results/"
        "rounds/R012_20260908")
TRUTH = {"AA": "a", "A": "a", "BB": "b", "B": "b", "same": "same"}
STRONG = ("AA", "BB")

FEATS = ["mc_residual", "flow_anomaly", "softness", "crawl", "luma_jump",
         "freeze", "fused_max", "fused_mean", "flow_mean", "flow_p95",
         "jerk", "still_frac"]


def load_features(rels: list[str], cache: Path, workers: int) -> dict:
    from run_signal_pairwise import features   # noqa: PLC0415
    feats: dict = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [r for r in rels if r not in feats]
    if todo:
        print(f"  提特征 {len(todo)} 条 ...")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, (rel, f) in enumerate(
                    zip(todo, ex.map(features,
                                     [os.path.join(ROOT, r) for r in todo])), 1):
                feats[rel] = f
                if i % 50 == 0:
                    print(f"    {i}/{len(todo)}")
        cache.write_text(json.dumps(feats))
    return feats


def design(feats: dict, rels: list[str]) -> np.ndarray:
    """log1p the heavy-tailed magnitudes, then leave standardisation to the fit.

    Every feature here is a non-negative magnitude with a long right tail, and
    on a raw scale one clip with a huge residual dominates the fit.
    """
    x = np.array([[float(feats[r].get(k, 0.0)) for k in FEATS] for r in rels])
    return np.log1p(np.maximum(x, 0.0))


def set_feats(names):
    """Replace the feature list in place, so callers holding FEATS see it."""
    FEATS.clear()
    FEATS.extend(names)


def pairs_of(split: str) -> list[dict]:
    return list(csv.DictReader(open(f"{R012}/{split}.csv")))


def evaluate(name: str, w: np.ndarray, mu, sd, feats, rows, margin: float):
    """Report both the margin view and VideoAlign's, which is the yardstick."""
    got = [r for r in rows
           if r["path_A"] in feats and r["path_B"] in feats
           and "error" not in feats[r["path_A"]]
           and "error" not in feats[r["path_B"]]]
    rels = sorted({r[k] for r in got for k in ("path_A", "path_B")})
    X = (design(feats, rels) - mu) / sd
    s = dict(zip(rels, X @ w))

    print(f"\n  === {name} ({len(got)} 对) ===")
    print(f"  {'分层':<8}{'n':>5}{'方向准确率':>13}{'可判定':>11}")
    out = {}
    for sname, labs in (("强偏好", STRONG), ("弱偏好", ("A", "B")),
                        ("平局", ("same",))):
        sel = [r for r in got if r["MQ"] in labs]
        dec = [r for r in sel if abs(s[r["path_A"]] - s[r["path_B"]]) >= margin]
        hit = sum(1 for r in dec
                  if ("a" if s[r["path_A"]] > s[r["path_B"]] else "b")
                  == TRUTH[r["MQ"]])
        acc = f"{hit/len(dec):.1%}" if dec else "--"
        print(f"  {sname:<8}{len(sel):>5}{acc:>13}{len(dec)}/{len(sel):>8}")
        out[sname] = {"n": len(sel), "decidable": len(dec),
                      "acc": hit / len(dec) if dec else None}
    nt = [r for r in got if r["MQ"] != "same"]
    dec = [r for r in nt if abs(s[r["path_A"]] - s[r["path_B"]]) >= margin]
    hit = sum(1 for r in dec
              if ("a" if s[r["path_A"]] > s[r["path_B"]] else "b")
              == TRUTH[r["MQ"]])
    # Overall counts a predicted tie as a prediction of "same", so it is
    # comparable with the trivial baseline of always answering "same".
    exact = sum(1 for r in got
                if (TRUTH[r["MQ"]] == "same") ==
                (abs(s[r["path_A"]] - s[r["path_B"]]) < margin)
                and (TRUTH[r["MQ"]] == "same"
                     or ("a" if s[r["path_A"]] > s[r["path_B"]] else "b")
                     == TRUTH[r["MQ"]]))
    base = sum(1 for r in got if r["MQ"] == "same") / len(got)
    print(f"  非平局方向 {hit/max(1,len(dec)):.1%} (n={len(dec)}/{len(nt)})  ·  "
          f"全体 {exact/len(got):.1%} (平凡基线 {base:.1%})")
    from agenteval.meta.videoalign import acc_with_ties, acc_without_ties
    H = {"AA": 1, "A": 1, "BB": -1, "B": -1, "same": 0}
    h = [H[r["MQ"]] for r in got]
    md = [s[r["path_A"]] - s[r["path_B"]] for r in got]
    star, eps = acc_with_ties(h, md)
    print(f"  [VideoAlign 口径] acc(无平局) {acc_without_ties(h, md):.1%}"
          f"  ·  acc* {star:.1%} (eps*={eps:.3f})")
    out["videoalign"] = {"acc_without_ties": round(acc_without_ties(h, md) * 100, 1),
                         "acc_star": round(star * 100, 1),
                         "epsilon_star": round(eps, 4)}
    out["nontie"] = {"acc": hit / max(1, len(dec)), "n": len(dec)}
    out["overall"] = {"acc": exact / len(got), "baseline": base}
    return out, s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="dev")
    ap.add_argument("--test", default="val_ac")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--strong-only", action="store_true",
                    help="fit on strongly-agreed pairs only")
    ap.add_argument("--family", default=None,
                    help="fit and test on one pair family, e.g. seed")
    ap.add_argument("--ladder-only", action="store_true",
                    help="restrict to the same-recipe capacity ladders "
                         "(cosmos_nano/super, wan5b/14b) -- the only cross-model "
                         "pairs where the quality gap and the fingerprint gap "
                         "are not collinear")
    ap.add_argument("--min-annotators", type=int, default=0,
                    help="drop pairs with fewer annotators than this. dev is "
                         "entirely single-annotator and calls 39%% of seed pairs "
                         "a tie; val_ac has 3-5 and calls 64%% a tie. Fitting "
                         "across that gap fits one question and scores another.")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="additional {rel: {name: value}} json feature files")
    ap.add_argument("--drop-base", action="store_true",
                    help="use only --extra features, to isolate their share")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.stdout.reconfigure(line_buffering=True)

    tr_rows, te_rows = pairs_of(args.train), pairs_of(args.test)
    if args.family:
        tr_rows = [r for r in tr_rows if r["family"] == args.family]
        te_rows = [r for r in te_rows if r["family"] == args.family]
        print(f"限定 family={args.family}")
    if args.ladder_only:
        lad = lambda rs: [r for r in rs
                          if str(r.get("is_ladder")).lower() in ("1", "true")]
        tr_rows, te_rows = lad(tr_rows), lad(te_rows)
        print("限定 is_ladder(同配方梯队)")
    if args.min_annotators:
        n = args.min_annotators
        tr_rows = [r for r in tr_rows if int(r.get("n_annotations") or 1) >= n]
        print(f"训练集只保留 ≥{n} 名标注员的对: {len(tr_rows)}")
    rels = sorted({r[k] for rows in (tr_rows, te_rows) for r in rows
                   for k in ("path_A", "path_B")})
    print(f"{args.train}: {len(tr_rows)} 对 · {args.test}: {len(te_rows)} 对 · "
          f"{len(rels)} 条视频")
    feats = load_features(rels, out / "features.json", args.workers)
    if args.drop_base:
        FEATS.clear()
        feats = {k: {} for k in feats}
    # Merge each extra source in under its own prefix, and require every clip to
    # carry every extra feature -- a clip silently defaulting to 0 on a source
    # that failed for it would be scored as if it had no defects.
    for path in args.extra:
        src = json.loads(Path(path).read_text())
        name = Path(path).stem
        keys = sorted({k for v in src.values() if isinstance(v, dict)
                       and "error" not in v for k in v})
        print(f"  并入 {name}: {len(keys)} 个特征, 覆盖 {len(src)} 条")
        for k in keys:
            FEATS.append(f"{name}.{k}")
        for rel, v in list(feats.items()):
            e = src.get(rel)
            if not isinstance(e, dict) or "error" in e or any(k not in e for k in keys):
                feats[rel] = {"error": f"missing {name}"}
                continue
            feats[rel] = {**v, **{f"{name}.{k}": e[k] for k in keys}}
    good = {k: v for k, v in feats.items() if "error" not in v}
    print(f"  特征可用 {len(good)}/{len(rels)}")

    fit_rows = [r for r in tr_rows if r["MQ"] != "same"
                and r["path_A"] in good and r["path_B"] in good]
    if args.strong_only:
        fit_rows = [r for r in fit_rows if r["MQ"] in STRONG]
    tr_rels = sorted({r[k] for r in fit_rows for k in ("path_A", "path_B")})
    Xtr = design(good, tr_rels)
    mu, sd = Xtr.mean(0), np.maximum(Xtr.std(0), 1e-6)
    idx = {r: i for i, r in enumerate(tr_rels)}
    Z = (Xtr - mu) / sd
    # One row per comparison: the difference of the two clips' features. A
    # logistic on that difference is exactly Bradley-Terry with the strength
    # parameterised by the features instead of estimated per video.
    D = np.array([Z[idx[r["path_A"]]] - Z[idx[r["path_B"]]] for r in fit_rows])
    y = np.array([1 if TRUTH[r["MQ"]] == "a" else 0 for r in fit_rows])
    # Mirror every comparison so the fit cannot learn an intercept from which
    # side a clip happened to be listed on.
    D = np.vstack([D, -D]); y = np.concatenate([y, 1 - y])

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(fit_intercept=False, C=0.3, max_iter=2000).fit(D, y)
    w = clf.coef_[0]
    print(f"\n  拟合于 {len(fit_rows)} 对"
          f"{'(仅强偏好)' if args.strong_only else ''}")
    print("  权重(正=该量越大分越高):")
    for k, v in sorted(zip(FEATS, w), key=lambda kv: -abs(kv[1])):
        print(f"    {k:<14}{v:+.3f}")

    res_tr, _ = evaluate(f"{args.train}(拟合集,不算证据)", w, mu, sd, good,
                         tr_rows, args.margin)
    res_te, s_te = evaluate(f"{args.test}(留出,这个才算)", w, mu, sd, good,
                            te_rows, args.margin)

    # 0-10 point-wise scale by corpus percentile: the fitted strength is in
    # arbitrary units, and a rank map is the only monotone transform that does
    # not invent a spacing the comparisons never carried.
    vals = np.array(sorted(s_te.values()))
    scale = {k: round(float(np.searchsorted(vals, v) / len(vals) * 10), 2)
             for k, v in s_te.items()}
    sv = sorted(scale.values())
    print(f"\n  point-wise 0-10 分布: 最低 {sv[0]:.1f} 中位 {sv[len(sv)//2]:.1f} "
          f"最高 {sv[-1]:.1f} · 不同取值 {len(set(sv))} 个")
    (out / "model.json").write_text(json.dumps(
        {"features": FEATS, "w": w.tolist(), "mu": mu.tolist(),
         "sd": sd.tolist(), "margin": args.margin,
         "train": args.train, "test": args.test,
         "result_train": res_tr, "result_test": res_te},
        ensure_ascii=False, indent=1))
    (out / "scores.json").write_text(json.dumps(scale, ensure_ascii=False))
    print(f"\n  wrote {out}/model.json  {out}/scores.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
