#!/usr/bin/env python3
"""Report a point-wise scorer on our benchmark using VideoAlign's metrics.

The metric algorithm is theirs (see agenteval.meta.videoalign) so numbers are
comparable with the trained reward model's; the data is our own R012 splits.

The slices matter more than the headline. Accuracy on this benchmark is monotone
in how far apart the two generators are, because in an ordinary cross-model pair
the quality gap and the generator-fingerprint gap are collinear -- so a scorer
that has learned only to recognise the generator posts a strong overall number.
The slices that break that collinearity are `seed` (same model, same prompt,
different seed), `is_ladder` (same recipe, capacity differs) and `upset` (the
weaker generator won). Those are the honest numbers, and for a reward model over
a single policy the seed family is also the deployment distribution.

    python scripts/report_pointwise.py --model runs/pw/model.json --split val_ac
"""
from __future__ import annotations

import os as _os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
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

from agenteval.meta.videoalign import acc_with_ties, acc_without_ties  # noqa: E402
from fit_pointwise import design                                       # noqa: E402
from run_signal_pairwise import features                               # noqa: E402

ROOT = "/pub/evaluation_group/cy/rm_videos"
R012 = ("/pub/evaluation_group/cy/mq_promptgen/pairing/review_results/"
        "rounds/R012_20260908")
H = {"AA": 1, "A": 1, "BB": -1, "B": -1, "same": 0}
TIER = {"wan5b": "W", "cogvideox": "W", "pangu": "M", "cosmos_nano": "M",
        "wan14b": "S", "cosmos_super": "S", "hunyuan": "S", "ltx": "S"}
RANK = {"W": 0, "M": 1, "S": 2}


def model_of(rel: str) -> str:
    parts = Path(rel).stem.split("__")
    if len(parts) >= 3 and parts[1] in TIER:
        return parts[1]
    return next((s for s in Path(rel).parts[::-1] if s in TIER), "")


def prior(r) -> int:
    """+1 if the generator ranking favours A, -1 for B, 0 when it cannot say."""
    a, b = TIER.get(model_of(r["path_A"])), TIER.get(model_of(r["path_B"]))
    if not a or not b or RANK[a] == RANK[b]:
        return 0
    return 1 if RANK[a] > RANK[b] else -1


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


def line(name, rows, hm):
    """One slice. acc* is reported only where the slice contains human ties."""
    if not rows:
        return None
    h = [hm[r["pair_id"]][0] for r in rows]
    m = [hm[r["pair_id"]][1] for r in rows]
    nt = sum(1 for x in h if x != 0)
    aw = acc_without_ties(h, m)
    star, eps = acc_with_ties(h, m)
    has_tie = nt != len(h)
    print(f"  {name:<24}{len(rows):>6}{nt:>7}{aw:>10.1%}"
          + (f"{star:>10.1%}{eps:>9.3f}" if has_tie else f"{'--':>10}{'--':>9}"))
    return {"n": len(rows), "n_nontie": nt, "acc_without_ties": round(aw * 100, 1),
            "acc_star": round(star * 100, 1) if has_tie else None,
            "epsilon_star": round(eps, 4) if has_tie else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="val_ac")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--features", default=None,
                    help="reuse an existing features.json")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    rows = list(csv.DictReader(open(f"{R012}/{args.split}.csv")))
    cache = Path(args.features or (args.out or "/tmp/rp") + "/features.json")
    cache.parent.mkdir(parents=True, exist_ok=True)
    feats = load_features(sorted({r[k] for r in rows
                                  for k in ("path_A", "path_B")}),
                          cache, args.workers)
    good = {k: v for k, v in feats.items() if "error" not in v}
    rows = [r for r in rows if r["path_A"] in good and r["path_B"] in good
            and r["MQ"] in H]

    m = json.loads(Path(args.model).read_text())
    rels = sorted({r[k] for r in rows for k in ("path_A", "path_B")})
    Z = (design(good, rels) - np.array(m["mu"])) / np.array(m["sd"])
    s = dict(zip(rels, Z @ np.array(m["w"])))
    hm = {r["pair_id"]: (H[r["MQ"]], s[r["path_A"]] - s[r["path_B"]])
          for r in rows}

    print(f"\n{'='*72}\n{args.split}: {len(rows)} 对  ·  指标算法 = VideoAlign "
          f"(arXiv:2305.14324 tie calibration)\n{'='*72}")
    print("  acc(无平局) = 不许弃权,人评非平局的对都必须给方向")
    print("  acc*        = 扫平局阈值取最优,分母含人评平局的对\n")
    print(f"  {'切片':<24}{'对数':>6}{'非平局':>7}{'acc(无平局)':>10}"
          f"{'acc*':>10}{'eps*':>9}")
    res = {"overall": line("全部", rows, hm)}

    print()
    res["by_family"] = {}
    for fam, _ in Counter(r["family"] for r in rows).most_common():
        res["by_family"][fam] = line(f"family {fam}", 
                                     [r for r in rows if r["family"] == fam], hm)
    print()
    lad = [r for r in rows if str(r.get("is_ladder")).lower() in ("1", "true")]
    res["ladder"] = line("is_ladder 同配方梯队", lad, hm)
    nt = [r for r in rows if H[r["MQ"]] != 0]
    up = [r for r in nt if prior(r) and prior(r) != H[r["MQ"]]]
    od = [r for r in nt if prior(r) and prior(r) == H[r["MQ"]]]
    res["upset"] = line("upset 爆冷(先验反向)", up, hm)
    res["ordercheck"] = line("ordercheck 顺风", od, hm)
    res["no_prior"] = line("档位相同(先验失效)",
                           [r for r in nt if not prior(r)], hm)

    print()
    res["by_n_annotations"] = {}
    for n, _ in sorted(Counter(r["n_annotations"] for r in rows).items()):
        res["by_n_annotations"][n] = line(
            f"{n} 名标注员", [r for r in rows if r["n_annotations"] == n], hm)
    if any(r.get("label_source") for r in rows):
        print()
        res["by_label_source"] = {}
        for src, _ in Counter(r.get("label_source", "") for r in rows).most_common():
            res["by_label_source"][src] = line(
                f"来源 {src}", [r for r in rows if r.get("label_source") == src], hm)

    print("\n  参考·档位先验本身(只看是哪个模型生成的,不看画面):")
    h = [H[r["MQ"]] for r in rows]
    pm = [float(prior(r)) for r in rows]
    print(f"    acc(无平局) {acc_without_ties(h, pm):.1%}   "
          f"acc* {acc_with_ties(h, pm)[0]:.1%}")

    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / f"report_{args.split}.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1))
        print(f"\n  wrote {args.out}/report_{args.split}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
